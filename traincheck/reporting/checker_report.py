import argparse
import html
import logging
import os
import time
from collections import Counter, defaultdict
from typing import Iterable

from traincheck.invariant import CheckerResult, Invariant


def _format_invariant_label(invariant: Invariant) -> str:
    display = invariant.relation.to_display_name(invariant.params)
    if display:
        return display
    if invariant.text_description:
        return invariant.text_description
    params = ", ".join(str(param) for param in invariant.params)
    return f"{invariant.relation.__name__}({params})"


def _extract_violation_steps(trace: list[dict] | None) -> list[int]:
    """Extract training step numbers from a violation trace."""
    if not trace:
        return []
    return [
        r["meta_vars.step"]
        for r in trace
        if isinstance(r, dict) and r.get("meta_vars.step") is not None
    ]


def _build_violation_entry(result: CheckerResult) -> dict:
    steps = _extract_violation_steps(result.trace)
    return {
        "display_name": _format_invariant_label(result.invariant),
        "relation_type": result.invariant.relation.__name__,
        "first_step": min(steps) if steps else None,
        "last_step": max(steps) if steps else None,
        "occurrences": len(result.trace) if result.trace else 1,
    }


def build_violations_summary(results: list[CheckerResult]) -> dict:
    """Build a pre-digested summary of all violations for machine and human consumption."""
    failed = [r for r in results if not r.check_passed]
    all_steps = []
    for r in failed:
        all_steps.extend(_extract_violation_steps(r.trace))
    return {
        "first_violation_step": min(all_steps) if all_steps else None,
        "distinct_invariants_violated": len(failed),
        "violations": [_build_violation_entry(r) for r in failed],
    }


def _summarize_results(results: Iterable[CheckerResult]) -> dict[str, int]:
    failed = sum(1 for res in results if not res.check_passed)
    not_triggered = sum(1 for res in results if res.triggered is False)
    passed = sum(1 for res in results if res.check_passed and res.triggered)
    total = failed + passed + not_triggered
    triggered = total - not_triggered
    return {
        "total": total,
        "failed": failed,
        "passed": passed,
        "not_triggered": not_triggered,
        "triggered": triggered,
    }


def _relation_breakdown(
    results: Iterable[CheckerResult],
) -> dict[str, dict[str, int]]:
    counts: dict[str, dict[str, int]] = defaultdict(
        lambda: {"failed": 0, "passed": 0, "not_triggered": 0}
    )
    for res in results:
        relation_name = res.invariant.relation.__name__
        if not res.check_passed:
            counts[relation_name]["failed"] += 1
        elif res.triggered:
            counts[relation_name]["passed"] += 1
        else:
            counts[relation_name]["not_triggered"] += 1
    return dict(counts)


def _count_failed_invariants(
    results: Iterable[CheckerResult],
) -> list[dict[str, object]]:
    counter: Counter[tuple[str, str]] = Counter()
    first_steps: dict[tuple[str, str], int | None] = {}
    last_steps: dict[tuple[str, str], int | None] = {}
    step_stage_maps: dict[tuple[str, str], dict] = defaultdict(dict)
    sample_traces: dict[tuple[str, str], list] = {}
    for res in results:
        if not res.check_passed:
            label = _format_invariant_label(res.invariant)
            relation = res.invariant.relation.__name__
            key = (label, relation)
            counter[key] += 1
            steps = _extract_violation_steps(res.trace)
            if steps:
                existing = first_steps.get(key)
                first_steps[key] = (
                    min(steps) if existing is None else min(existing, min(steps))
                )
                existing_last = last_steps.get(key)
                last_steps[key] = (
                    max(steps)
                    if existing_last is None
                    else max(existing_last, max(steps))
                )
            elif key not in first_steps:
                first_steps[key] = None
                last_steps[key] = None
            # Accumulate step → stage (first stage seen per step wins)
            for rec in res.trace or []:
                if not isinstance(rec, dict):
                    continue
                step = rec.get("meta_vars.step")
                stage = rec.get("meta_vars.stage")
                if step is not None and step not in step_stage_maps[key]:
                    step_stage_maps[key][step] = stage
            # One sample trace per invariant (first violation wins)
            if key not in sample_traces and res.trace:
                sample_traces[key] = _summarize_trace_records(res.trace)
    top_pairs = counter.most_common(10)
    return [
        {
            "label": label,
            "relation": relation,
            "count": count,
            "first_step": first_steps.get((label, relation)),
            "last_step": last_steps.get((label, relation)),
            "step_stages": sorted(step_stage_maps[(label, relation)].items()),
            "sample_trace": sample_traces.get((label, relation), []),
        }
        for (label, relation), count in top_pairs
    ]


def build_offline_report_data(
    results_by_trace: list[tuple[str, list[CheckerResult]]],
    *,
    generated_at: str,
    output_dir: str,
    total_invariants: int,
) -> dict:
    overall_relation_counts: dict[str, dict[str, int]] = defaultdict(
        lambda: {"failed": 0, "passed": 0, "not_triggered": 0}
    )
    overall_summary = {
        "total_invariants": total_invariants,
        "total_checks": 0,
        "failed": 0,
        "passed": 0,
        "not_triggered": 0,
        "triggered": 0,
    }

    trace_sections = []
    all_failed_invariants: list[CheckerResult] = []
    for trace_name, results in results_by_trace:
        summary = _summarize_results(results)
        relations = _relation_breakdown(results)
        failed_invariants = _count_failed_invariants(results)

        for relation_name, rel_counts in relations.items():
            overall_relation_counts[relation_name]["failed"] += rel_counts["failed"]
            overall_relation_counts[relation_name]["passed"] += rel_counts["passed"]
            overall_relation_counts[relation_name]["not_triggered"] += rel_counts[
                "not_triggered"
            ]

        overall_summary["total_checks"] += summary["total"]
        overall_summary["failed"] += summary["failed"]
        overall_summary["passed"] += summary["passed"]
        overall_summary["not_triggered"] += summary["not_triggered"]
        overall_summary["triggered"] += summary["triggered"]

        trace_sections.append(
            {
                "name": trace_name,
                "summary": summary,
                "relations": relations,
                "failed_invariants": failed_invariants,
            }
        )
        all_failed_invariants.extend([res for res in results if not res.check_passed])

    top_violations = _count_failed_invariants(all_failed_invariants)

    return {
        "mode": "offline",
        "generated_at": generated_at,
        "output_dir": output_dir,
        "overall": overall_summary,
        "relations": dict(overall_relation_counts),
        "traces": trace_sections,
        "top_violations": top_violations,
    }


_TRACE_DISPLAY_KEYS = (
    "function",
    "meta_vars.step",
    "meta_vars.stage",
    "type",
    "var_name",
    "var_type",
)
_TRACE_SKIP_PREFIXES = ("attributes._TRAINCHECK_",)

# Known stage → badge color (bg, text)
_STAGE_COLORS: dict[str, tuple[str, str]] = {
    "train": ("#2f6fed", "#fff"),
    "training": ("#2f6fed", "#fff"),
    "eval": ("#2fb679", "#fff"),
    "evaluation": ("#2fb679", "#fff"),
    "validation": ("#2fb679", "#fff"),
    "val": ("#2fb679", "#fff"),
    "test": ("#f2b233", "#333"),
    "inference": ("#9b59b6", "#fff"),
    "pretrain": ("#1abc9c", "#fff"),
}
_STAGE_FALLBACK_PALETTE = [
    ("#e24c4b", "#fff"),
    ("#e67e22", "#fff"),
    ("#e91e63", "#fff"),
    ("#00bcd4", "#fff"),
    ("#607d8b", "#fff"),
]


def _stage_badge_style(stage: str) -> str:
    """Return inline CSS background/color for a stage badge."""
    key = stage.lower()
    if key in _STAGE_COLORS:
        bg, fg = _STAGE_COLORS[key]
    else:
        bg, fg = _STAGE_FALLBACK_PALETTE[hash(key) % len(_STAGE_FALLBACK_PALETTE)]
    return f"background:{bg};color:{fg}"


def _summarize_trace_records(trace: list[dict] | None) -> list[dict]:
    """Return a compact, HTML-safe subset of trace records for display."""
    if not trace:
        return []
    out = []
    for rec in trace:
        if not isinstance(rec, dict):
            continue
        row: dict = {}
        for key in _TRACE_DISPLAY_KEYS:
            val = rec.get(key)
            if val is not None:
                row[key] = str(val)
        # Add first attribute-style key that is not an internal one
        for key, val in rec.items():
            if key.startswith("attributes.") and val is not None:
                if not any(key.startswith(p) for p in _TRACE_SKIP_PREFIXES):
                    row[key] = str(val)
                    break
        out.append(row)
    return out


def build_online_report_data(
    *,
    generated_at: str,
    output_dir: str,
    total_invariants: int,
    total_violations: int,
    failed_inv: dict[Invariant, int],
    relation_totals: dict[str, int],
    violation_details: dict | None = None,
    triggered_inv: set | None = None,
    all_invs: list | None = None,
    current_step: int | None = None,
    current_stage: str | None = None,
    sampling_interval: int | None = None,
    warm_up_steps: int | None = None,
) -> dict:
    relation_violations: dict[str, int] = defaultdict(int)
    for inv in failed_inv:
        relation_violations[inv.relation.__name__] += 1

    if violation_details is None:
        violation_details = {}

    # Estimate how many steps have been checked so far.
    checked_steps: int | None = None
    if (
        current_step is not None
        and sampling_interval is not None
        and warm_up_steps is not None
        and sampling_interval > 0
    ):
        checked_steps = max(0, current_step - warm_up_steps) // sampling_interval + min(
            current_step, warm_up_steps
        )

    def _make_entry(inv: Invariant, count: int) -> dict:
        detail = violation_details.get(inv, {})
        step_stages: list[tuple] = detail.get("step_stages") or []
        sample_trace: list[dict] | None = detail.get("sample_trace")
        steps = [s for s, _ in step_stages]
        first_step = min(steps) if steps else None
        last_step = max(steps) if steps else None
        # stage of the first/last violation event
        first_stage = (
            next((st for s, st in step_stages if s == first_step), None)
            if first_step is not None
            else None
        )
        last_stage = (
            next((st for s, st in reversed(step_stages) if s == last_step), None)
            if last_step is not None
            else None
        )
        # deduplicated, sorted (step, stage) pairs for the expanded view
        unique_step_stages = sorted(set(step_stages), key=lambda x: x[0])
        unique_viol_steps = len(set(s for s, _ in step_stages))
        viol_rate: float | None = None
        if checked_steps is not None and checked_steps > 0:
            viol_rate = round(unique_viol_steps / checked_steps * 100, 1)
        return {
            "label": _format_invariant_label(inv),
            "relation": inv.relation.__name__,
            "count": count,
            "first_step": first_step,
            "first_stage": first_stage,
            "last_step": last_step,
            "last_stage": last_stage,
            "step_stages": unique_step_stages[:100],  # cap for HTML size
            "sample_trace": _summarize_trace_records(sample_trace),
            "violation_step_count": unique_viol_steps,
            "checked_steps": checked_steps,
            "violation_rate": viol_rate,
        }

    # Sort by first violation step (earliest first), then by count descending.
    def _sort_key(item):
        inv, count = item
        detail = violation_details.get(inv, {})
        step_stages = detail.get("step_stages") or []
        steps = [s for s, _ in step_stages]
        first = min(steps) if steps else float("inf")
        return (first, -count)

    sorted_pairs = sorted(failed_inv.items(), key=_sort_key)[:20]
    top_violations = [_make_entry(inv, count) for inv, count in sorted_pairs]

    # Progress tracking
    triggered_count = len(triggered_inv) if triggered_inv is not None else 0
    failing_count = len(failed_inv)
    passing_count = triggered_count - failing_count
    not_triggered_count = total_invariants - triggered_count
    pass_rate = (
        round(passing_count / triggered_count * 100, 1) if triggered_count > 0 else None
    )

    not_triggered_labels: list[str] = []
    if all_invs is not None and triggered_inv is not None:
        not_triggered_labels = [
            _format_invariant_label(inv) for inv in all_invs if inv not in triggered_inv
        ][:50]

    relations = {}
    for relation_name, total in relation_totals.items():
        relations[relation_name] = {
            "total": total,
            "failed": relation_violations.get(relation_name, 0),
        }

    overall = {
        "total_invariants": total_invariants,
        "total_checks": None,
        "failed": total_violations,
        "passed": None,
        "not_triggered": None,
        "triggered": None,
        "violated_invariants": len(failed_inv),
        # progress fields
        "triggered_count": triggered_count,
        "passing_count": passing_count,
        "not_triggered_count": not_triggered_count,
        "pass_rate": pass_rate,
        "current_step": current_step,
        "current_stage": current_stage,
    }

    return {
        "mode": "online",
        "generated_at": generated_at,
        "output_dir": output_dir,
        "overall": overall,
        "relations": relations,
        "traces": [],
        "top_violations": top_violations,
        "not_triggered_labels": not_triggered_labels,
        "sampling_interval": sampling_interval,
        "warm_up_steps": warm_up_steps,
        "checked_steps": checked_steps,
    }


def _render_stage_badge(stage: str | None, esc_fn) -> str:
    if not stage:
        return ""
    style = _stage_badge_style(stage)
    return f'<span class="stage-badge" style="{style}">{esc_fn(stage)}</span>'


def _render_step_stages_html(
    step_stages: list[tuple], esc_fn, max_per_group: int = 15
) -> str:
    """Render a compact stage-grouped step list as HTML."""
    if not step_stages:
        return "—"
    # Group consecutive same-stage runs
    groups: list[tuple[str | None, list[int]]] = []
    for step, stage in step_stages:
        if groups and groups[-1][0] == stage:
            groups[-1][1].append(step)
        else:
            groups.append((stage, [step]))
    parts = []
    for stage, steps in groups:
        shown = steps[:max_per_group]
        more = len(steps) - len(shown)
        steps_str = ", ".join(str(s) for s in shown)
        if more > 0:
            steps_str += f' <span class="more-steps">+{more} more</span>'
        badge = _render_stage_badge(stage, esc_fn)
        parts.append(f"{badge}{steps_str}")
    return ' <span class="step-sep">·</span> '.join(parts)


def _render_bar_segment(width_pct: float, class_name: str) -> str:
    width_pct = max(0.0, min(100.0, width_pct))
    return f'<span class="{class_name}" style="width: {width_pct:.2f}%"></span>'


def render_html_report(report_data: dict) -> str:
    def esc(value: str) -> str:
        return html.escape(value, quote=True)

    def percent(part: int, total: int) -> float:
        if total <= 0:
            return 0.0
        return (part / total) * 100.0

    mode = report_data.get("mode", "offline")
    overall = report_data["overall"]
    traces = report_data.get("traces", [])
    top_violations = report_data.get("top_violations", [])

    top_table_html = ""

    if mode == "online":
        sampling_interval = report_data.get("sampling_interval")
        warm_up_steps_val = report_data.get("warm_up_steps")
        checked_steps_total = report_data.get("checked_steps")
        has_sampling = sampling_interval is not None

        rows = []
        for entry in top_violations:
            label = esc(str(entry.get("label", "")))
            relation = esc(str(entry.get("relation", "")))
            count = entry.get("count", "")
            first_step = entry.get("first_step")
            first_stage = entry.get("first_stage")
            last_step = entry.get("last_step")
            last_stage = entry.get("last_stage")
            step_stages: list = entry.get("step_stages") or []
            sample_trace = entry.get("sample_trace") or []
            violation_step_count = entry.get("violation_step_count", 0)
            entry_checked = entry.get("checked_steps")
            viol_rate = entry.get("violation_rate")

            def _step_with_badge(step, stage) -> str:
                if step is None:
                    return "—"
                badge = _render_stage_badge(stage, esc)
                return f"{badge}{step}"

            first_step_html = _step_with_badge(first_step, first_stage)
            last_step_html = _step_with_badge(last_step, last_stage)
            steps_html = _render_step_stages_html(step_stages, esc)

            # Build sample trace table
            if sample_trace:
                all_keys: list[str] = []
                for rec in sample_trace:
                    for k in rec:
                        if k not in all_keys:
                            all_keys.append(k)
                trace_head = "".join(f"<th>{esc(k)}</th>" for k in all_keys)
                trace_rows_html = []
                for rec in sample_trace:
                    cells = []
                    for k in all_keys:
                        val = rec.get(k, "")
                        # Style stage cells
                        if k == "meta_vars.stage" and val:
                            style = _stage_badge_style(val)
                            cell = (
                                f'<td><span class="stage-badge" style="{style}">'
                                f"{esc(val)}</span></td>"
                            )
                        else:
                            cell = f"<td>{esc(str(val))}</td>"
                        cells.append(cell)
                    trace_rows_html.append(f"<tr>{''.join(cells)}</tr>")
                trace_body = "\n".join(trace_rows_html)
                expand_content = (
                    f'<div class="trace-steps">Steps: {steps_html}</div>'
                    f'<div class="trace-wrap"><table class="table trace-table">'
                    f"<thead><tr>{trace_head}</tr></thead>"
                    f"<tbody>{trace_body}</tbody></table></div>"
                )
            else:
                expand_content = f'<div class="trace-steps">Steps: {steps_html}</div>'

            # Frequency cell: prefer rate when sampling info available
            if has_sampling and entry_checked is not None and entry_checked > 0:
                rate_str = f"{viol_rate}%" if viol_rate is not None else "?"
                freq_cell = (
                    f'<span class="freq-rate">{rate_str}</span>'
                    f'<span class="freq-detail">'
                    f"{violation_step_count}/{entry_checked} steps"
                    f"</span>"
                )
            else:
                freq_cell = f'<span class="freq-rate">{count}</span>'

            rows.append(
                f"<tr>"
                f'<td><details><summary class="inv-label-summary">{label}</summary>'
                f'<div class="expand-body">{expand_content}</div></details>'
                f'<span class="inv-rel-tag">{relation}</span></td>'
                f'<td class="step-cell">{first_step_html}</td>'
                f'<td class="step-cell">{last_step_html}</td>'
                f'<td class="freq-cell">{freq_cell}</td>'
                f"</tr>"
            )

        freq_col_header = "Frequency" if has_sampling else "Count"
        top_table_html = (
            f'<table class="table viol-table"><thead>'
            f"<tr><th>Invariant</th><th>First Step</th><th>Last Step</th>"
            f"<th>{freq_col_header}</th></tr>"
            f"</thead><tbody>{''.join(rows)}</tbody></table>"
            if rows
            else "<p>No violations yet.</p>"
        )
    else:
        rows = []
        for entry in top_violations:
            label = esc(str(entry.get("label", "")))
            relation = esc(str(entry.get("relation", "")))
            count = entry.get("count", "")
            first_step = entry.get("first_step")
            last_step = entry.get("last_step")
            off_step_stages: list = entry.get("step_stages") or []
            off_sample_trace = entry.get("sample_trace") or []

            def _step_cell(step, _ss=off_step_stages) -> str:
                if step is None:
                    return "—"
                stage = next((s for st, s in _ss if st == step), None)
                badge = _render_stage_badge(stage, esc)
                return f"{badge}{step}"

            first_step_html = _step_cell(first_step)
            last_step_html = _step_cell(last_step)
            steps_html = _render_step_stages_html(off_step_stages, esc)

            if off_sample_trace:
                off_keys: list[str] = []
                for rec in off_sample_trace:
                    for k in rec:
                        if k not in off_keys:
                            off_keys.append(k)
                trace_head = "".join(f"<th>{esc(k)}</th>" for k in off_keys)
                trace_rows_html = []
                for rec in off_sample_trace:
                    cells = []
                    for k in off_keys:
                        val = rec.get(k, "")
                        if k == "meta_vars.stage" and val:
                            style = _stage_badge_style(val)
                            cell = (
                                f'<td><span class="stage-badge" style="{style}">'
                                f"{esc(val)}</span></td>"
                            )
                        else:
                            cell = f"<td>{esc(str(val))}</td>"
                        cells.append(cell)
                    trace_rows_html.append(f"<tr>{''.join(cells)}</tr>")
                trace_body = "\n".join(trace_rows_html)
                expand_content = (
                    f'<div class="trace-steps">Steps: {steps_html}</div>'
                    f'<div class="trace-wrap"><table class="table trace-table">'
                    f"<thead><tr>{trace_head}</tr></thead>"
                    f"<tbody>{trace_body}</tbody></table></div>"
                )
            else:
                expand_content = f'<div class="trace-steps">Steps: {steps_html}</div>'

            rows.append(
                f"<tr>"
                f'<td><details><summary class="inv-label-summary">{label}</summary>'
                f'<div class="expand-body">{expand_content}</div></details>'
                f'<span class="inv-rel-tag">{relation}</span></td>'
                f'<td class="step-cell">{first_step_html}</td>'
                f'<td class="step-cell">{last_step_html}</td>'
                f'<td class="freq-cell"><span class="freq-rate">{count}</span></td>'
                f"</tr>"
            )
        top_table_html = (
            f'<table class="table viol-table"><thead>'
            f"<tr><th>Invariant</th><th>First Step</th><th>Last Step</th>"
            f"<th>Count</th></tr>"
            f"</thead><tbody>{''.join(rows)}</tbody></table>"
            if rows
            else "<p>No violations.</p>"
        )

    trace_sections = []
    for trace in traces:
        total = trace["summary"]["total"]
        failed = trace["summary"]["failed"]
        passed = trace["summary"]["passed"]
        not_triggered = trace["summary"]["not_triggered"]

        bar = (
            _render_bar_segment(percent(failed, total), "bar-failed")
            + _render_bar_segment(percent(passed, total), "bar-passed")
            + _render_bar_segment(percent(not_triggered, total), "bar-not-triggered")
        )

        failed_rows = []
        for failed_item in trace["failed_invariants"][:10]:
            label = esc(str(failed_item.get("label", "")))
            relation = esc(str(failed_item.get("relation", "")))
            count = failed_item.get("count", "")
            first_step = failed_item.get("first_step")
            last_step = failed_item.get("last_step")
            item_step_stages: list = failed_item.get("step_stages") or []
            item_sample_trace = failed_item.get("sample_trace") or []

            def _step_cell_trace(step) -> str:
                if step is None:
                    return "—"
                stage = next((s for st, s in item_step_stages if st == step), None)
                badge = _render_stage_badge(stage, esc)
                return f"{badge}{step}"

            steps_html = _render_step_stages_html(item_step_stages, esc)
            if item_sample_trace:
                item_keys: list[str] = []
                for rec in item_sample_trace:
                    for k in rec:
                        if k not in item_keys:
                            item_keys.append(k)
                trace_head = "".join(f"<th>{esc(k)}</th>" for k in item_keys)
                trace_rows_html = []
                for rec in item_sample_trace:
                    cells = []
                    for k in item_keys:
                        val = rec.get(k, "")
                        if k == "meta_vars.stage" and val:
                            style = _stage_badge_style(val)
                            cell = (
                                f'<td><span class="stage-badge" style="{style}">'
                                f"{esc(val)}</span></td>"
                            )
                        else:
                            cell = f"<td>{esc(str(val))}</td>"
                        cells.append(cell)
                    trace_rows_html.append(f"<tr>{''.join(cells)}</tr>")
                trace_body = "\n".join(trace_rows_html)
                expand_content = (
                    f'<div class="trace-steps">Steps: {steps_html}</div>'
                    f'<div class="trace-wrap"><table class="table trace-table">'
                    f"<thead><tr>{trace_head}</tr></thead>"
                    f"<tbody>{trace_body}</tbody></table></div>"
                )
            else:
                expand_content = f'<div class="trace-steps">Steps: {steps_html}</div>'

            failed_rows.append(
                f"<tr>"
                f'<td><details><summary class="inv-label-summary">{label}</summary>'
                f'<div class="expand-body">{expand_content}</div></details>'
                f'<span class="inv-rel-tag">{relation}</span></td>'
                f'<td class="step-cell">{_step_cell_trace(first_step)}</td>'
                f'<td class="step-cell">{_step_cell_trace(last_step)}</td>'
                f'<td class="freq-cell"><span class="freq-rate">{count}</span></td>'
                f"</tr>"
            )
        failed_list_html = (
            f'<table class="table viol-table"><thead>'
            f"<tr><th>Invariant</th><th>First Step</th><th>Last Step</th>"
            f"<th>Count</th></tr>"
            f"</thead><tbody>{''.join(failed_rows)}</tbody></table>"
            if failed_rows
            else "<p>None</p>"
        )

        relation_rows = []
        for relation_name, rel_counts in sorted(trace["relations"].items()):
            relation_rows.append(
                "<tr>"
                f"<td>{esc(relation_name)}</td>"
                f"<td>{rel_counts['failed']}</td>"
                f"<td>{rel_counts['passed']}</td>"
                f"<td>{rel_counts['not_triggered']}</td>"
                "</tr>"
            )
        relation_table = "\n".join(relation_rows) or ""

        trace_sections.append(
            f"""
            <section class="panel">
              <div class="panel-header">
                <div>
                  <h2>{esc(trace['name'])}</h2>
                  <div class="subtle">Trace summary</div>
                </div>
                <div class="stat-inline">
                  <span>Failed</span><strong>{failed}</strong>
                  <span>Passed</span><strong>{passed}</strong>
                  <span>Not Triggered</span><strong>{not_triggered}</strong>
                </div>
              </div>
              <div class="bar">{bar}</div>
              <div class="legend">
                <span class="legend-item failed">Failed</span>
                <span class="legend-item passed">Passed</span>
                <span class="legend-item not-triggered">Not Triggered</span>
              </div>
              <div class="grid-two">
                <div>
                  <h3>Failed invariants (top 10)</h3>
                  {failed_list_html}
                </div>
                <div>
                  <h3>Relation breakdown</h3>
                  <table class="table">
                    <thead>
                      <tr><th>Relation</th><th>Failed</th><th>Passed</th><th>Not Triggered</th></tr>
                    </thead>
                    <tbody>
                      {relation_table}
                    </tbody>
                  </table>
                </div>
              </div>
            </section>
            """
        )

    relation_rows = []
    if mode == "online":
        for relation_name, rel_counts in sorted(report_data["relations"].items()):
            relation_rows.append(
                "<tr>"
                f"<td>{esc(relation_name)}</td>"
                f"<td>{rel_counts.get('total', 0)}</td>"
                f"<td>{rel_counts.get('failed', 0)}</td>"
                "</tr>"
            )
    else:
        for relation_name, rel_counts in sorted(report_data["relations"].items()):
            relation_rows.append(
                "<tr>"
                f"<td>{esc(relation_name)}</td>"
                f"<td>{rel_counts['failed']}</td>"
                f"<td>{rel_counts['passed']}</td>"
                f"<td>{rel_counts['not_triggered']}</td>"
                "</tr>"
            )

    if mode == "online":
        cur_step = overall.get("current_step")
        cur_stage = overall.get("current_stage")
        triggered_count = overall.get("triggered_count", 0) or 0
        passing_count = overall.get("passing_count", 0) or 0
        not_triggered_count = overall.get("not_triggered_count", 0) or 0
        pass_rate_val = overall.get("pass_rate")
        total_invariants_n = overall["total_invariants"] or 0
        violated = overall.get("violated_invariants", 0) or 0

        # Current step card — show stage badge inline
        if cur_step is not None:
            stage_badge = _render_stage_badge(cur_stage, esc) if cur_stage else ""
            step_value_html = f"{stage_badge}{cur_step}"
            step_sub = esc(f"stage: {cur_stage}") if cur_stage else "no stage info"
        else:
            step_value_html = "—"
            step_sub = "waiting for first trace record"

        pass_rate_display = f"{pass_rate_val}%" if pass_rate_val is not None else "—"

        card_html = f"""
      <div class="card">
        <div class="label">Total Invariants</div>
        <div class="value">{total_invariants_n}</div>
      </div>
      <div class="card">
        <div class="label">Triggered</div>
        <div class="value">{triggered_count}</div>
        <div class="card-sub">of {total_invariants_n} loaded</div>
      </div>
      <div class="card card-pass">
        <div class="label">Pass Rate</div>
        <div class="value">{pass_rate_display}</div>
        <div class="card-sub">{passing_count} passing · {violated} failing</div>
      </div>
      <div class="card">
        <div class="label">Violations</div>
        <div class="value">{overall['failed']}</div>
      </div>
      <div class="card card-step">
        <div class="label">Current Step</div>
        <div class="value step-value">{step_value_html}</div>
        <div class="card-sub">{step_sub}</div>
      </div>
        """
        relation_header = "<tr><th>Relation</th><th>Total</th><th>Violated</th></tr>"
        mode_note = '<div class="subtle">Online mode — checking in progress.</div>'

        # Progress panel (checking coverage bar + not-yet-triggered list)
        not_triggered_labels: list[str] = report_data.get("not_triggered_labels", [])
        bar_total = total_invariants_n or 1
        passing_pct = percent(passing_count, bar_total)
        failing_pct = percent(violated, bar_total)
        not_triggered_pct = percent(not_triggered_count, bar_total)
        progress_bar = (
            _render_bar_segment(passing_pct, "bar-passed")
            + _render_bar_segment(failing_pct, "bar-failed")
            + _render_bar_segment(not_triggered_pct, "bar-not-triggered")
        )

        if not_triggered_labels:
            nt_items = "".join(
                f'<li class="nt-item">{esc(lbl)}</li>' for lbl in not_triggered_labels
            )
            suffix = (
                f" (showing first {len(not_triggered_labels)})"
                if not_triggered_count > len(not_triggered_labels)
                else ""
            )
            nt_section = (
                f'<details class="nt-details">'
                f"<summary>{not_triggered_count} not yet triggered{esc(suffix)}</summary>"
                f'<ul class="nt-list">{nt_items}</ul>'
                f"</details>"
            )
        elif not_triggered_count > 0:
            nt_section = f'<p class="subtle">{not_triggered_count} invariant(s) not yet triggered.</p>'
        else:
            nt_section = '<p class="subtle">All invariants have been triggered at least once.</p>'

        progress_panel = f"""
    <section class="panel progress-panel">
      <div class="panel-header">
        <div>
          <h2>Checking Progress</h2>
          <div class="subtle">{triggered_count} of {total_invariants_n} invariants triggered so far</div>
        </div>
        <div class="stat-inline">
          <span>Passing</span><strong class="c-pass">{passing_count}</strong>
          <span>Failing</span><strong class="c-fail">{violated}</strong>
          <span>Not Triggered</span><strong>{not_triggered_count}</strong>
        </div>
      </div>
      <div class="bar">{progress_bar}</div>
      <div class="legend">
        <span class="legend-item passed">Passing ({passing_count})</span>
        <span class="legend-item failed">Failing ({violated})</span>
        <span class="legend-item not-triggered">Not Triggered ({not_triggered_count})</span>
      </div>
      {nt_section}
    </section>
    """
    else:
        total_checks = overall["total_checks"] or 0
        passed_checks = overall["passed"] or 0
        pass_rate = (
            round(passed_checks / total_checks * 100, 1) if total_checks else 0.0
        )
        card_html = f"""
      <div class="card">
        <div class="label">Total Invariants</div>
        <div class="value">{overall['total_invariants']}</div>
      </div>
      <div class="card">
        <div class="label">Failed Checks</div>
        <div class="value">{overall['failed']}</div>
      </div>
      <div class="card">
        <div class="label">Passed Checks</div>
        <div class="value">{overall['passed']}</div>
      </div>
      <div class="card">
        <div class="label">Pass Rate</div>
        <div class="value">{pass_rate}%</div>
      </div>
      <div class="card">
        <div class="label">Not Triggered</div>
        <div class="value">{overall['not_triggered']}</div>
      </div>
        """
        relation_header = "<tr><th>Relation</th><th>Failed</th><th>Passed</th><th>Not Triggered</th></tr>"
        mode_note = ""
        progress_panel = ""

    # surface first violation step in the top violations panel header
    all_first_steps = [
        v["first_step"] for v in top_violations if v.get("first_step") is not None
    ]
    if all_first_steps:
        first_step_note = (
            f'<div class="subtle">First violation at step {min(all_first_steps)}'
            f" · {len(top_violations)} distinct invariant(s) violated</div>"
        )
    elif top_violations:
        first_step_note = (
            f'<div class="subtle">{len(top_violations)}'
            " distinct invariant(s) violated</div>"
        )
    else:
        first_step_note = ""

    if mode == "online":
        sampling_interval = report_data.get("sampling_interval")
        warm_up_steps_val = report_data.get("warm_up_steps")
        checked_steps_total = report_data.get("checked_steps")
        sampling_ctx = ""
        if sampling_interval is not None:
            sampling_ctx = f" · sampled every {sampling_interval} steps"
            if warm_up_steps_val is not None:
                sampling_ctx += f", warm-up {warm_up_steps_val}"
            if checked_steps_total is not None:
                sampling_ctx += f" ({checked_steps_total} steps checked)"
        panel_subtitle = esc(
            f"Sorted by first violation step — click to expand trace{sampling_ctx}"
        )
        panel_content = top_table_html
    else:
        panel_subtitle = "Sorted by first violation step — click to expand trace"
        panel_content = top_table_html

    top_panel = f"""
    <section class="panel">
      <div class="panel-header">
        <div>
          <h2>Violations</h2>
          <div class="subtle">{panel_subtitle}</div>
          {first_step_note}
        </div>
      </div>
      {panel_content}
    </section>
    """

    relation_table = f"""
    <section class="panel">
      <div class="panel-header">
        <div>
          <h2>Relation breakdown (overall)</h2>
          <div class="subtle">Grouped by invariant relation type</div>
        </div>
      </div>
      <table class="table">
        <thead>
          {relation_header}
        </thead>
        <tbody>
          {''.join(relation_rows)}
        </tbody>
      </table>
    </section>
    """

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>TrainCheck Report</title>
  <style>
    :root {{
      color-scheme: light;
      --bg: #f4f6fb;
      --panel: #ffffff;
      --text: #182033;
      --muted: #6c778c;
      --accent: #2f6fed;
      --failed: #e24c4b;
      --passed: #2fb679;
      --not-triggered: #f2b233;
      --border: #e2e6f0;
    }}
    body {{
      margin: 0;
      font-family: "Satoshi", "Avenir Next", "Segoe UI", sans-serif;
      background: radial-gradient(circle at top, #fefefe, #f4f6fb 45%, #edf0f7);
      color: var(--text);
    }}
    .container {{
      max-width: 1100px;
      margin: 0 auto;
      padding: 32px 24px 60px;
    }}
    header {{
      display: flex;
      flex-wrap: wrap;
      align-items: center;
      justify-content: space-between;
      gap: 16px;
      margin-bottom: 28px;
    }}
    header h1 {{
      margin: 0;
      font-size: 32px;
      letter-spacing: -0.02em;
    }}
    .subtle {{
      color: var(--muted);
      font-size: 14px;
    }}
    .cards {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
      gap: 16px;
      margin: 18px 0 26px;
    }}
    .card {{
      background: var(--panel);
      border: 1px solid var(--border);
      border-radius: 14px;
      padding: 18px 20px;
      box-shadow: 0 10px 30px rgba(24, 32, 51, 0.08);
    }}
    .card .label {{
      color: var(--muted);
      font-size: 13px;
      letter-spacing: 0.02em;
      text-transform: uppercase;
    }}
    .card .value {{
      font-size: 28px;
      font-weight: 700;
      margin-top: 6px;
    }}
    .panel {{
      background: var(--panel);
      border-radius: 16px;
      padding: 24px;
      border: 1px solid var(--border);
      box-shadow: 0 15px 30px rgba(24, 32, 51, 0.08);
      margin-bottom: 22px;
    }}
    .panel-header {{
      display: flex;
      justify-content: space-between;
      align-items: center;
      gap: 12px;
    }}
    .panel h2 {{
      margin: 0 0 4px;
      font-size: 22px;
    }}
    .stat-inline {{
      display: grid;
      grid-auto-flow: column;
      gap: 8px;
      align-items: center;
      font-size: 13px;
      color: var(--muted);
    }}
    .stat-inline strong {{
      color: var(--text);
      font-size: 16px;
    }}
    .bar {{
      display: flex;
      height: 14px;
      border-radius: 999px;
      overflow: hidden;
      background: #e8ecf5;
      margin: 16px 0 10px;
    }}
    .bar-failed {{ background: var(--failed); }}
    .bar-passed {{ background: var(--passed); }}
    .bar-not-triggered {{ background: var(--not-triggered); }}
    .legend {{
      display: flex;
      gap: 16px;
      font-size: 12px;
      color: var(--muted);
      margin-bottom: 18px;
    }}
    .legend-item::before {{
      content: "";
      display: inline-block;
      width: 10px;
      height: 10px;
      border-radius: 999px;
      margin-right: 6px;
    }}
    .legend-item.failed::before {{ background: var(--failed); }}
    .legend-item.passed::before {{ background: var(--passed); }}
    .legend-item.not-triggered::before {{ background: var(--not-triggered); }}
    .grid-two {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(260px, 1fr));
      gap: 18px;
    }}
    .inv-list {{
      list-style: none;
      padding: 0;
      margin: 0;
    }}
    .inv-list li {{
      display: grid;
      grid-template-columns: 1fr auto;
      gap: 8px;
      align-items: center;
      padding: 10px 12px;
      border: 1px solid var(--border);
      border-radius: 10px;
      margin-bottom: 8px;
      background: #fbfcff;
    }}
    .inv-label {{
      display: block;
      font-weight: 600;
    }}
    .inv-detail {{
      display: block;
      font-size: 12px;
      color: var(--muted);
      margin-top: 4px;
    }}
    .inv-count {{
      font-size: 16px;
      font-weight: 700;
      color: var(--failed);
    }}
    .table {{
      width: 100%;
      border-collapse: collapse;
      font-size: 13px;
    }}
    .table th, .table td {{
      text-align: left;
      padding: 8px 6px;
      border-bottom: 1px solid var(--border);
    }}
    .table th {{
      color: var(--muted);
      font-size: 12px;
      text-transform: uppercase;
      letter-spacing: 0.04em;
    }}
    .viol-table td {{ vertical-align: top; }}
    .step-cell {{ white-space: nowrap; font-variant-numeric: tabular-nums; font-weight: 600; }}
    .count-cell {{ white-space: nowrap; font-variant-numeric: tabular-nums; font-weight: 700; color: var(--failed); }}
    .freq-cell {{ white-space: nowrap; }}
    .freq-rate {{ display: block; font-variant-numeric: tabular-nums; font-weight: 700; color: var(--failed); }}
    .freq-detail {{ display: block; font-size: 11px; color: var(--muted); margin-top: 2px; }}
    .stage-badge {{
      display: inline-block;
      font-size: 10px;
      font-weight: 700;
      padding: 1px 6px;
      border-radius: 99px;
      margin-right: 4px;
      text-transform: uppercase;
      letter-spacing: 0.05em;
      vertical-align: middle;
    }}
    .step-sep {{ color: var(--muted); }}
    .more-steps {{ color: var(--muted); font-size: 11px; }}
    .card-sub {{ font-size: 12px; color: var(--muted); margin-top: 4px; }}
    .card-pass .value {{ color: var(--passed); }}
    .card-step .step-value {{ font-size: 22px; }}
    .c-pass {{ color: var(--passed); }}
    .c-fail {{ color: var(--failed); }}
    .progress-panel h2 {{ margin: 0 0 4px; font-size: 22px; }}
    .nt-details summary {{
      cursor: pointer;
      font-size: 13px;
      color: var(--muted);
      padding: 6px 0;
    }}
    .nt-details summary::-webkit-details-marker {{ display: none; }}
    .nt-details summary::before {{ content: "▸ "; color: var(--accent); }}
    details[open].nt-details summary::before {{ content: "▾ "; }}
    .nt-list {{
      list-style: none;
      padding: 0;
      margin: 8px 0 0;
      display: grid;
      grid-template-columns: repeat(auto-fill, minmax(280px, 1fr));
      gap: 6px;
    }}
    .nt-item {{
      font-size: 12px;
      color: var(--muted);
      background: #f4f6fb;
      border: 1px solid var(--border);
      border-radius: 6px;
      padding: 5px 10px;
    }}
    .inv-label-summary {{
      cursor: pointer;
      font-weight: 600;
      font-size: 14px;
      list-style: none;
      padding: 2px 0;
    }}
    .inv-label-summary::-webkit-details-marker {{ display: none; }}
    .inv-label-summary::before {{ content: "▸ "; color: var(--accent); font-size: 11px; }}
    details[open] .inv-label-summary::before {{ content: "▾ "; }}
    .inv-rel-tag {{
      display: inline-block;
      font-size: 11px;
      color: var(--muted);
      background: #f0f2f8;
      border-radius: 4px;
      padding: 1px 6px;
      margin-top: 4px;
    }}
    .expand-body {{
      margin-top: 10px;
      padding: 10px;
      background: #f8f9fd;
      border-radius: 8px;
      border: 1px solid var(--border);
    }}
    .trace-steps {{
      font-size: 12px;
      color: var(--muted);
      margin-bottom: 8px;
      word-break: break-all;
    }}
    .trace-wrap {{ overflow-x: auto; }}
    .trace-table {{ font-size: 11px; min-width: 400px; }}
    .trace-table th {{ font-size: 10px; }}
    footer {{
      margin-top: 28px;
      font-size: 12px;
      color: var(--muted);
    }}
  </style>
</head>
<body>
  <div class="container">
    <header>
      <div>
        <h1>TrainCheck Report</h1>
        <div class="subtle">Generated {esc(report_data['generated_at'])}</div>
        {mode_note}
      </div>
      <div class="subtle">Output: {esc(report_data['output_dir'])}</div>
    </header>

    <div class="cards">
      {card_html}
    </div>

    {progress_panel}

    {top_panel}

    {relation_table}

    {''.join(trace_sections)}

    <footer>Generated by TrainCheck checker.</footer>
  </div>
</body>
</html>
"""


def write_html_report(report_data: dict, output_dir: str) -> str:
    report_path = os.path.join(output_dir, "report.html")
    with open(report_path, "w") as f:
        f.write(render_html_report(report_data))
    return report_path


class ReportEmitter:
    def __init__(
        self,
        output_dir: str,
        *,
        no_html_report: bool,
        report_wandb: bool,
        report_mlflow: bool,
        report_interval_seconds: float,
        args: argparse.Namespace | None,
    ):
        self.output_dir = output_dir
        self.no_html_report = no_html_report
        self.report_wandb = report_wandb
        self.report_mlflow = report_mlflow
        self.report_interval_seconds = report_interval_seconds
        self.args = args
        self._last_report_ts = 0.0
        self._last_report_state: tuple[int, int] | None = None
        self._wandb_run = None
        self._mlflow_active = False

    def _should_emit(self, report_state: tuple[int, int], force: bool) -> bool:
        if force:
            return True
        now = time.monotonic()
        if self._last_report_state == report_state:
            if self.report_interval_seconds <= 0:
                return False
            if now - self._last_report_ts < self.report_interval_seconds:
                return False
        return True

    def emit(
        self,
        report_data: dict,
        *,
        force: bool = False,
        report_state: tuple[int, int] | None = None,
    ) -> str | None:
        if not self.output_dir:
            return None

        overall = report_data.get("overall", {})
        if report_state is None:
            report_state = (
                int(overall.get("failed", 0) or 0),
                int(overall.get("total_checks", 0) or 0),
            )

        if not self._should_emit(report_state, force):
            return None

        report_path = None
        if not self.no_html_report:
            report_path = write_html_report(report_data, self.output_dir)

        if self.report_wandb and self.args is not None:
            self._log_wandb(report_data, report_path, self.args)

        if self.report_mlflow and self.args is not None:
            self._log_mlflow(report_data, report_path, self.args)

        self._last_report_ts = time.monotonic()
        self._last_report_state = report_state
        return report_path

    def close(self):
        if self._wandb_run is not None:
            self._wandb_run.finish()
            self._wandb_run = None
        if self._mlflow_active:
            try:
                import mlflow
            except ImportError:
                self._mlflow_active = False
                return
            mlflow.end_run()
            self._mlflow_active = False

    def _log_wandb(
        self,
        report_data: dict,
        report_path: str | None,
        args: argparse.Namespace,
    ):
        import glob

        try:
            import wandb
        except ImportError:
            logging.getLogger(__name__).warning(
                "Weights & Biases is not installed. Skipping wandb logging."
            )
            return

        if self._wandb_run is None:
            self._wandb_run = wandb.init(
                project=args.wandb_project,
                entity=args.wandb_entity,
                name=args.wandb_run_name,
                group=args.wandb_group,
                tags=args.wandb_tags,
                job_type="checker",
            )
        run = self._wandb_run
        if run is None:
            logging.getLogger(__name__).warning("wandb.init() returned None; skipping.")
            return

        overall = report_data["overall"]
        mode = report_data.get("mode", "offline")

        # --- run config (searchable/filterable in W&B UI) ---
        run.config.update(
            {
                "traincheck/invariants_total": overall["total_invariants"],
                "traincheck/output_dir": report_data.get("output_dir", ""),
                "traincheck/mode": mode,
            },
            allow_val_change=True,
        )

        # --- scalar metrics ---
        if mode == "online":
            total_invariants = overall["total_invariants"] or 0
            violated = overall.get("violated_invariants", 0) or 0
            violation_rate = (
                round(violated / total_invariants * 100, 1) if total_invariants else 0.0
            )
            wandb.log(
                {
                    "invariants/total": total_invariants,
                    "invariants/violated_unique": violated,
                    "invariants/violation_rate_pct": violation_rate,
                    "violations/total": overall["failed"],
                }
            )
        else:
            total_checks = overall["total_checks"] or 0
            passed = overall["passed"] or 0
            pass_rate = round(passed / total_checks * 100, 1) if total_checks else 0.0
            wandb.log(
                {
                    "invariants/total": overall["total_invariants"],
                    "checks/total": total_checks,
                    "checks/failed": overall["failed"],
                    "checks/passed": passed,
                    "checks/not_triggered": overall["not_triggered"],
                    "checks/pass_rate_pct": pass_rate,
                }
            )

        # --- relation breakdown table ---
        rel_table = wandb.Table(
            columns=["relation", "failed", "passed", "not_triggered"]
        )
        for relation_name, rel_counts in report_data["relations"].items():
            rel_table.add_data(
                relation_name,
                rel_counts.get("failed", 0),
                rel_counts.get("passed", 0),
                rel_counts.get("not_triggered", 0),
            )
        wandb.log({"relation_breakdown": rel_table})

        # --- violated invariants table ---
        top_violations = report_data.get("top_violations", [])
        if top_violations:
            vtable = wandb.Table(
                columns=[
                    "invariant",
                    "relation_type",
                    "occurrences",
                    "first_step",
                    "last_step",
                ]
            )
            for v in top_violations:
                vtable.add_data(
                    v.get("label", ""),
                    v.get("relation", ""),
                    v.get("count", 0),
                    v.get("first_step"),
                    v.get("last_step"),
                )
            wandb.log({"violations": vtable})

        # --- summary metrics (shown in run comparison table) ---
        first_steps = [
            v["first_step"] for v in top_violations if v.get("first_step") is not None
        ]
        last_steps_wandb = [
            v["last_step"] for v in top_violations if v.get("last_step") is not None
        ]
        if first_steps:
            run.summary["violations/first_step"] = min(first_steps)
        if last_steps_wandb:
            run.summary["violations/last_step"] = max(last_steps_wandb)
        run.summary["violations/distinct_invariants"] = len(top_violations)

        # --- violations_summary.json as versioned artifact ---
        summary_files = glob.glob(
            os.path.join(self.output_dir, "*", "violations_summary.json")
        )
        if summary_files:
            try:
                artifact = wandb.Artifact(
                    name="violations_summary",
                    type="checker_output",
                    description="Per-trace violation summaries from traincheck-check",
                )
                for summary_file in summary_files:
                    artifact.add_file(summary_file)
                run.log_artifact(artifact)
            except Exception:
                logging.getLogger(__name__).warning(
                    "Failed to attach violations_summary artifact to wandb run."
                )

        # --- HTML report ---
        if report_path:
            try:
                with open(report_path, "r") as f:
                    wandb.log({"checker_report": wandb.Html(f.read())})
            except Exception:
                logging.getLogger(__name__).warning(
                    "Failed to attach HTML report to wandb run."
                )

    def _log_mlflow(
        self,
        report_data: dict,
        report_path: str | None,
        args: argparse.Namespace,
    ):
        try:
            import mlflow
        except ImportError:
            logging.getLogger(__name__).warning(
                "MLflow is not installed. Skipping MLflow logging."
            )
            return

        import glob

        if args.mlflow_experiment:
            mlflow.set_experiment(args.mlflow_experiment)

        if not self._mlflow_active:
            mlflow.start_run(run_name=args.mlflow_run_name)
            self._mlflow_active = True

        overall = report_data["overall"]
        mode = report_data.get("mode", "offline")
        top_violations = report_data.get("top_violations", [])

        # --- run tags (searchable in MLflow UI) ---
        mlflow.set_tags(
            {
                "traincheck.mode": mode,
                "traincheck.invariants_total": str(overall["total_invariants"]),
                "traincheck.output_dir": report_data.get("output_dir", ""),
            }
        )

        # --- scalar metrics ---
        if mode == "online":
            total_invariants = overall["total_invariants"] or 0
            violated = overall.get("violated_invariants", 0) or 0
            violation_rate = (
                round(violated / total_invariants * 100, 1) if total_invariants else 0.0
            )
            mlflow.log_metric("invariants_total", total_invariants)
            mlflow.log_metric("invariants_violated_unique", violated)
            mlflow.log_metric("invariants_violation_rate_pct", violation_rate)
            mlflow.log_metric("violations_total", overall["failed"])
        else:
            total_checks = overall["total_checks"] or 0
            passed = overall["passed"] or 0
            pass_rate = round(passed / total_checks * 100, 1) if total_checks else 0.0
            mlflow.log_metric("invariants_total", overall["total_invariants"])
            mlflow.log_metric("checks_total", total_checks)
            mlflow.log_metric("checks_failed", overall["failed"])
            mlflow.log_metric("checks_passed", passed)
            mlflow.log_metric("checks_not_triggered", overall["not_triggered"])
            mlflow.log_metric("checks_pass_rate_pct", pass_rate)

        # --- violation summary metrics ---
        first_steps = [
            v["first_step"] for v in top_violations if v.get("first_step") is not None
        ]
        last_steps = [
            v["last_step"] for v in top_violations if v.get("last_step") is not None
        ]
        if first_steps:
            mlflow.log_metric("violations_first_step", min(first_steps))
        if last_steps:
            mlflow.log_metric("violations_last_step", max(last_steps))
        mlflow.log_metric("violations_distinct_invariants", len(top_violations))

        # --- violations table as JSON artifact ---
        if top_violations:
            try:
                mlflow.log_dict(
                    {"violations": top_violations},
                    artifact_file="violations.json",
                )
            except Exception:
                logging.getLogger(__name__).warning(
                    "Failed to log violations table to MLflow."
                )

        # --- per-trace violations_summary.json artifacts ---
        summary_files = glob.glob(
            os.path.join(self.output_dir, "*", "violations_summary.json")
        )
        for summary_file in summary_files:
            try:
                mlflow.log_artifact(summary_file, artifact_path="violations_summaries")
            except Exception:
                logging.getLogger(__name__).warning(
                    "Failed to log %s to MLflow.", summary_file
                )

        # --- HTML report ---
        if report_path:
            mlflow.log_artifact(report_path)

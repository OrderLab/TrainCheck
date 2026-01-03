import argparse
import html
import logging
import os
import time
from collections import Counter, defaultdict
from typing import Iterable

from traincheck.invariant import CheckerResult, Invariant


def _format_invariant_label(invariant: Invariant) -> str:
    if invariant.text_description:
        return invariant.text_description
    params = ", ".join(str(param) for param in invariant.params)
    return f"{invariant.relation.__name__}({params})"


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
    for res in results:
        if not res.check_passed:
            label = _format_invariant_label(res.invariant)
            relation = res.invariant.relation.__name__
            counter[(label, relation)] += 1
    top_pairs = counter.most_common(10)
    return [
        {"label": label, "relation": relation, "count": count}
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


def build_online_report_data(
    *,
    generated_at: str,
    output_dir: str,
    total_invariants: int,
    total_violations: int,
    failed_inv: dict[Invariant, int],
    relation_totals: dict[str, int],
) -> dict:
    relation_violations: dict[str, int] = defaultdict(int)
    for inv in failed_inv:
        relation_violations[inv.relation.__name__] += 1

    top_pairs = sorted(
        ((count, inv) for inv, count in failed_inv.items()),
        key=lambda item: item[0],
        reverse=True,
    )[:10]
    top_violations = [
        {
            "label": _format_invariant_label(inv),
            "relation": inv.relation.__name__,
            "count": count,
        }
        for count, inv in top_pairs
    ]

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
    }

    return {
        "mode": "online",
        "generated_at": generated_at,
        "output_dir": output_dir,
        "overall": overall,
        "relations": relations,
        "traces": [],
        "top_violations": top_violations,
    }


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

    top_items = []
    for entry in top_violations:
        label = esc(str(entry.get("label", "")))
        detail = esc(str(entry.get("relation", "")))
        count = entry.get("count")
        count_html = f'<span class="inv-count">{count}</span>' if count else ""
        top_items.append(
            f'<li><span class="inv-label">{label}</span>'
            f'<span class="inv-detail">{detail}</span>{count_html}</li>'
        )
    top_list = "".join(top_items) or "<li>None</li>"

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

        failed_list_items = []
        for failed_item in trace["failed_invariants"][:10]:
            label = esc(str(failed_item.get("label", "")))
            detail = esc(str(failed_item.get("relation", "")))
            count = failed_item.get("count")
            count_html = f'<span class="inv-count">{count}</span>' if count else ""
            failed_list_items.append(
                f'<li><span class="inv-label">{label}</span>'
                f'<span class="inv-detail">{detail}</span>{count_html}</li>'
            )
        failed_list_html = "".join(failed_list_items) or "<li>None</li>"

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
                  <ul class="inv-list">{failed_list_html}</ul>
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
        card_html = f"""
      <div class="card">
        <div class="label">Total Invariants</div>
        <div class="value">{overall['total_invariants']}</div>
      </div>
      <div class="card">
        <div class="label">Violations</div>
        <div class="value">{overall['failed']}</div>
      </div>
      <div class="card">
        <div class="label">Violated Invariants</div>
        <div class="value">{overall.get('violated_invariants', 0)}</div>
      </div>
        """
        relation_header = "<tr><th>Relation</th><th>Total</th><th>Violated</th></tr>"
        mode_note = '<div class="subtle">Online mode (partial coverage).</div>'
    else:
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
        <div class="label">Not Triggered</div>
        <div class="value">{overall['not_triggered']}</div>
      </div>
        """
        relation_header = "<tr><th>Relation</th><th>Failed</th><th>Passed</th><th>Not Triggered</th></tr>"
        mode_note = ""

    top_panel = f"""
    <section class="panel">
      <div class="panel-header">
        <div>
          <h2>Top Violations</h2>
          <div class="subtle">Most frequent violations observed</div>
        </div>
      </div>
      <ul class="inv-list">{top_list}</ul>
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

        overall = report_data["overall"]
        mode = report_data.get("mode", "offline")
        if mode == "online":
            wandb.log(
                {
                    "invariants/total": overall["total_invariants"],
                    "invariants/violated_unique": overall.get("violated_invariants", 0),
                    "violations/total": overall["failed"],
                }
            )
        else:
            wandb.log(
                {
                    "invariants/total": overall["total_invariants"],
                    "checks/total": overall["total_checks"],
                    "checks/failed": overall["failed"],
                    "checks/passed": overall["passed"],
                    "checks/not_triggered": overall["not_triggered"],
                }
            )

        table = wandb.Table(columns=["relation", "failed", "passed", "not_triggered"])
        for relation_name, rel_counts in report_data["relations"].items():
            table.add_data(
                relation_name,
                rel_counts.get("failed", 0),
                rel_counts.get("passed", 0),
                rel_counts.get("not_triggered", 0),
            )
        wandb.log({"relation_breakdown": table})

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

        if args.mlflow_experiment:
            mlflow.set_experiment(args.mlflow_experiment)

        if not self._mlflow_active:
            mlflow.start_run(run_name=args.mlflow_run_name)
            self._mlflow_active = True

        overall = report_data["overall"]
        mode = report_data.get("mode", "offline")
        if mode == "online":
            mlflow.log_metric("invariants_total", overall["total_invariants"])
            mlflow.log_metric(
                "invariants_violated_unique", overall.get("violated_invariants", 0)
            )
            mlflow.log_metric("violations_total", overall["failed"])
        else:
            mlflow.log_metric("invariants_total", overall["total_invariants"])
            mlflow.log_metric("checks_total", overall["total_checks"])
            mlflow.log_metric("checks_failed", overall["failed"])
            mlflow.log_metric("checks_passed", overall["passed"])
            mlflow.log_metric("checks_not_triggered", overall["not_triggered"])

        if report_path:
            mlflow.log_artifact(report_path)

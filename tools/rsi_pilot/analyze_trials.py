"""Freeze completed RSI pilot selections, then audit matched held-out results.

Timestamps come from explicit operator actions, never file mtimes. Hashes bind
the saved evidence; they do not establish a security boundary against its owner.
The operator scores each unique checkpoint/configuration once and reuses that
immutable report for identical selections. Exact baseline-score equality checks
that reuse contract; it is not a claim of identical repeated execution timings.
"""

import argparse
import hashlib
import json
import math
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from tools.fresh_validation.contracts import JSONObject  # noqa: E402
from tools.fresh_validation.contracts import (
    JSONValue,
    boolean,
    integer,
    json_object,
    mapping,
    number,
    read_object,
    required,
    sequence,
    string,
)
from tools.rsi_pilot.run_candidate import configure  # noqa: E402
from tools.rsi_pilot.run_candidate import (
    validate_diagnostic_artifacts,
    validate_diagnostics,
    validate_preflight,
    validate_results,
)

POLICY: list[JSONValue] = [1, 2, 4, 4]


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def text(path: Path) -> str:
    return path.read_bytes().decode("utf-8")


def finite(value: object, label: str) -> float:
    result = float(number(value, label))
    if not math.isfinite(result) or result < 0:
        raise ValueError(f"{label} must be finite and nonnegative")
    return result


def require_equal(actual: object, expected: object, label: str) -> None:
    if type(actual) is not type(expected) or actual != expected:
        raise ValueError(f"{label} mismatch")
    if isinstance(actual, dict) and isinstance(expected, dict):
        left_mapping, right_mapping = mapping(actual, label), mapping(expected, label)
        for key in left_mapping:
            require_equal(left_mapping[key], right_mapping[key], label)
    if isinstance(actual, list) and isinstance(expected, list):
        for left, right in zip(sequence(actual, label), sequence(expected, label)):
            require_equal(left, right, label)


def protocol(root: Path) -> tuple[JSONObject, list[str], dict[str, str], list[str]]:
    record = read_object(root / "study_protocol.json")
    timestamp(required(record, "revised_at"))
    if not string(required(record, "revision_reason"), "revision_reason"):
        raise ValueError("Protocol revision must explain the excluded setup trials")
    order = [
        string(value, "trial ID")
        for value in sequence(required(record, "trial_order"), "trial_order")
    ]
    excluded = [
        string(value, "setup trial ID")
        for value in sequence(
            required(record, "excluded_setup_trials"), "excluded_setup_trials"
        )
    ]
    if any(re.fullmatch(r"[A-Za-z0-9_-]+", name) is None for name in order + excluded):
        raise ValueError("Invalid trial directory ID")
    if len(set(order + excluded)) != len(order + excluded):
        raise ValueError("Duplicate or overlapping trial IDs")
    arms: dict[str, str] = {}
    pairs = sequence(required(record, "pairs"), "pairs")
    if len(pairs) != 2:
        raise ValueError("This pilot requires exactly two matched pairs")
    for item in pairs:
        pair = mapping(item, "pair")
        for arm in ("control", "traincheck"):
            name = string(required(pair, arm), arm)
            if name in arms:
                raise ValueError("Trial repeated across pairs")
            arms[name] = arm
    require_equal(set(order), set(arms), "paired trial set")
    return record, order, arms, excluded


def metric(value: object, expected_depths: list[int]) -> JSONObject:
    rows = [mapping(row, "depth") for row in sequence(value, "depths")]
    require_equal(
        [integer(required(r, "loops"), "loops") for r in rows],
        expected_depths,
        "depth order",
    )
    accuracies = [
        finite(required(r, "macro_accuracy_percent"), "accuracy") for r in rows
    ]
    work = [finite(required(r, "macro_mean_dense_matmul_flops"), "FLOPs") for r in rows]
    if any(a > 100 for a in accuracies) or any(w <= 0 for w in work):
        raise ValueError("Invalid accuracy or work")
    selected = [0, 1, 2, 2]
    ratios = [
        work[i] / ceiling
        for i, ceiling in zip(selected, [805760, 1610624, 3220352, 6439808])
    ]
    if any(ratio > 1 for ratio in ratios):
        raise ValueError("Fixed policy exceeds its work ceiling")
    accuracy = sum(accuracies[i] for i in selected) / 4
    return {
        "fixed_policy_depths": list(POLICY),
        "accuracy_percent": accuracy,
        # Preserve the upstream aggregate_reward evaluation order as well as
        # its mathematical formula; exact receipt validation must not round.
        "reward": sum(
            0.25 * (0.8 * accuracies[i] + 0.2 * (100 * (1 - ratio)))
            for i, ratio in zip(selected, ratios)
        ),
    }


def preflight_summary(
    trial: Path, manifest: JSONObject, files: set[Path]
) -> JSONObject:
    """Free program checks have their own receipts and costs, never train budget."""
    attempts: list[JSONValue] = []
    seconds = 0.0
    paths = sorted((trial / "preflights").glob("preflight-*"))
    if [path.name for path in paths] != [
        f"preflight-{i:02d}" for i in range(1, len(paths) + 1)
    ]:
        raise ValueError("Malformed preflight attempt sequence")
    profile = Path(string(required(manifest, "profile"), "profile"))
    for path in paths:
        receipt = read_object(path / "preflight.json")
        status = string(required(receipt, "status"), "preflight status")
        if status not in (
            "program_valid",
            "program_invalid",
            "execution_failed",
            "client_error",
        ):
            raise ValueError("Preflight has not finished")
        for key, expected in (
            ("schema_version", 1),
            ("preflight", path.name),
            ("arm", manifest["arm"]),
            ("profile_sha256", digest(profile)),
            ("trial_manifest_sha256", digest(trial / "trial.json")),
            ("program_sha256", digest(path / "program.py")),
        ):
            require_equal(required(receipt, key), expected, key)
        require_equal(
            text(path / "config.toml"),
            configure(text(profile), text(path / "program.py")),
            "preflight config",
        )
        errors = sequence(required(receipt, "errors"), "preflight errors")
        if (status == "program_valid") != (len(errors) == 0):
            raise ValueError("Preflight status/error mismatch")
        for error in errors:
            detail = mapping(error, "preflight error")
            if not string(required(detail, "phase"), "preflight error phase"):
                raise ValueError("Empty preflight error phase")
            # Program rejections nest the original interpreter error; process
            # and client failures store the exception directly in the receipt.
            cause = (
                mapping(detail["error"], "interpreter error")
                if "error" in detail
                else detail
            )
            for key in ("type", "message"):
                if not string(required(cause, key), key):
                    raise ValueError("Empty preflight error evidence")
        if status in ("program_valid", "program_invalid"):
            result = read_object(path / "result.json")
            valid = validate_preflight(
                result, integer(required(receipt, "check_exit_code"), "check_exit_code")
            )
            require_equal(required(receipt, "valid"), valid, "preflight valid")
            require_equal(
                required(receipt, "result"), result, "preflight backend result"
            )
            require_equal(valid, status == "program_valid", "preflight status")
        elif status == "execution_failed" and integer(
            required(receipt, "check_exit_code"), "check_exit_code"
        ) in (0, 2):
            raise ValueError(
                "Preflight execution failure has a normal interpreter exit"
            )
        duration = finite(
            required(receipt, "operational_seconds_including_transfer"),
            "preflight duration",
        )
        seconds += duration
        files.update(item for item in path.iterdir() if item.is_file())
        attempts.append(
            {
                "preflight": path.name,
                "status": status,
                "operational_seconds": duration,
                "errors": receipt["errors"],
            }
        )
    return {
        "attempts": attempts,
        "count": len(paths),
        "operational_seconds_including_queue_transfer": seconds,
    }


def trial_summary(
    root: Path, trial_id: str, files: set[Path], expected_arm: str | None
) -> JSONObject:
    trial = root / "trials" / trial_id
    manifest_path = trial / "trial.json"
    manifest = read_object(manifest_path)
    workspace = trial / "workspace"
    require_equal(
        Path(string(required(manifest, "workspace"), "workspace")).resolve(),
        workspace.resolve(),
        "workspace",
    )
    arm = string(required(manifest, "arm"), "arm")
    if arm not in ("control", "traincheck"):
        raise ValueError("Unknown arm")
    if expected_arm is not None:
        require_equal(arm, expected_arm, "arm")
    for key, expected_limit in (
        ("max_candidates", 2),
        ("max_diagnostics", int(arm == "traincheck")),
        ("expected_steps", 10000),
    ):
        require_equal(integer(required(manifest, key), key), expected_limit, key)
    profile = root / "frozen-profile.toml"
    require_equal(
        Path(string(required(manifest, "profile"), "profile")).resolve(),
        profile.resolve(),
        "profile path",
    )
    require_equal(required(manifest, "profile_sha256"), digest(profile), "profile hash")
    files.update(
        {
            manifest_path,
            profile,
            workspace / "assignment.txt",
            workspace / "baseline_research.py",
            workspace / "baseline_development.json",
            workspace / "summary.md",
            workspace / "submission.json",
        }
    )
    preflights = preflight_summary(trial, manifest, files)
    paths = sorted((trial / "runs").glob("candidate-*"))
    if len(paths) > 2 or [p.name for p in paths] != [
        f"candidate-{i:02d}" for i in range(1, len(paths) + 1)
    ]:
        raise ValueError("Malformed or excessive candidate attempts")
    attempts: list[JSONValue] = []
    valid: dict[str, JSONObject] = {}
    training_known = operational = 0.0
    unknown_training: list[JSONValue] = []
    diagnostic_count = diagnostic_complete = diagnostic_failed = diagnostic_skips = (
        error_count
    ) = 0
    for index, path in enumerate(paths, 1):
        receipt_path = path / "run.json"
        receipt = read_object(receipt_path)
        requested = boolean(
            required(receipt, "diagnostics_requested"), "diagnostics_requested"
        )
        diagnostic_count += requested
        for key, expected_text in (
            ("candidate", path.name),
            ("arm", arm),
            ("profile_sha256", digest(profile)),
            ("trial_manifest_sha256", digest(manifest_path)),
            ("program_sha256", digest(path / "program.py")),
        ):
            require_equal(required(receipt, key), expected_text, key)
        for key, expected_count in (
            ("candidates_reserved", index),
            ("diagnostics_reserved", diagnostic_count),
        ):
            require_equal(integer(required(receipt, key), key), expected_count, key)
        require_equal(
            text(path / "config.toml"),
            configure(text(profile), text(path / "program.py")),
            "candidate config",
        )
        status = string(required(receipt, "status"), "status")
        if status not in ("training_complete", "training_failed", "client_error"):
            raise ValueError("Candidate has not finished")
        errors = sequence(required(receipt, "errors"), "errors")
        error_count += len(errors)
        for error in errors:
            for key in ("phase", "type", "message"):
                if not string(required(mapping(error, "error"), key), key):
                    raise ValueError("Empty error evidence")
        if status != "training_complete" and not errors:
            raise ValueError("Failed candidate has no error evidence")
        duration = finite(
            required(receipt, "operational_seconds_including_transfer_and_diagnostics"),
            "operational duration",
        )
        operational += duration
        files.update({receipt_path, path / "program.py", path / "config.toml"})
        files.update(path.glob("*.log"))
        elapsed: float | None = None
        if "training_exit_code" in receipt:
            code = integer(receipt["training_exit_code"], "training_exit_code")
            if status == "training_failed" and code == 0:
                raise ValueError("Training failure has a successful process exit")
            if code == 0 and (path / "results.json").exists():
                results = read_object(path / "results.json")
                _, seconds, _ = validate_results(results, manifest)
                for key, expected_result in (
                    ("training_seconds", seconds),
                    ("completed_steps", 10000),
                    ("development_depths", results["final"]),
                ):
                    require_equal(required(receipt, key), expected_result, key)
                elapsed = finite(seconds, "training duration")
                valid[path.name] = metric(results["final"], [1, 2, 4, 8])
                files.add(path / "results.json")
        elif status != "client_error":
            raise ValueError("Finished training receipt has no process exit")
        if status == "training_complete" and elapsed is None:
            raise ValueError("Complete training has no validated results")
        if elapsed is None:
            unknown_training.append(path.name)
        else:
            training_known += elapsed
        diagnostic: JSONObject = {"requested": requested, "status": "not_requested"}
        if requested:
            diagnostic["status"] = "unknown_after_client_error"
            if "diagnostic_status" in receipt:
                diagnostic_status = string(
                    receipt["diagnostic_status"], "diagnostic_status"
                )
                if diagnostic_status not in ("complete", "execution_failed"):
                    raise ValueError("Unknown diagnostic status")
                diagnostic["status"] = diagnostic_status
                exit_code = integer(
                    required(receipt, "diagnostic_exit_code"), "diagnostic_exit_code"
                )
                if (diagnostic_status == "complete") != (exit_code == 0):
                    raise ValueError("Diagnostic exit/status mismatch")
                if diagnostic_status == "complete":
                    diagnostic_report = read_object(path / "agent_report.json")
                    validate_diagnostics(diagnostic_report)
                    validate_diagnostic_artifacts(diagnostic_report, path)
                    require_equal(
                        required(receipt, "traincheck_report"),
                        diagnostic_report,
                        "diagnostic report",
                    )
                    diagnostic["report"] = diagnostic_report
                    diagnostic_complete += 1
                    diagnostic_skips += integer(
                        required(
                            mapping(diagnostic_report["counts"], "counts"),
                            "skipped_variable_checks",
                        ),
                        "skipped_variable_checks",
                    )
                    files.update(
                        {
                            path / name
                            for name in (
                                "agent_report.json",
                                "coverage.json",
                                "diagnostic_manifest.json",
                            )
                        }
                    )
                else:
                    diagnostic_failed += 1
            elif status != "client_error":
                raise ValueError(
                    "Finished candidate is missing its requested diagnostic"
                )
        elif any(
            key in receipt
            for key in (
                "traincheck_report",
                "diagnostic_status",
                "diagnostic_exit_code",
            )
        ):
            raise ValueError("Unrequested diagnostic evidence")
        attempts.append(
            {
                "candidate": path.name,
                "status": status,
                "training_seconds": elapsed,
                "operational_seconds": duration,
                "diagnostic": diagnostic,
                "errors": receipt["errors"],
            }
        )
    if diagnostic_count > int(arm == "traincheck"):
        raise ValueError("Diagnostic budget exceeded")
    ledger_path = trial / "usage.json"
    if paths or ledger_path.exists():
        ledger = read_object(ledger_path)
        require_equal(
            integer(required(ledger, "candidates_reserved"), "candidates_reserved"),
            len(paths),
            "ledger candidates",
        )
        require_equal(
            integer(required(ledger, "diagnostics_reserved"), "diagnostics_reserved"),
            diagnostic_count,
            "ledger diagnostics",
        )
        files.add(ledger_path)
    report: JSONObject = {
        "trial": trial_id,
        "arm": arm,
        "attempts": attempts,
        "preflights": preflights,
        "training_seconds_known": training_known,
        "training_duration_unknown_attempts": unknown_training,
        "operational_seconds_including_queue_transfer_diagnostics": operational,
        "operational_seconds_including_preflights": operational
        + finite(
            preflights["operational_seconds_including_queue_transfer"],
            "preflight duration",
        ),
        "diagnostics_requested": diagnostic_count,
        "diagnostics_completed": diagnostic_complete,
        "diagnostic_execution_failures": diagnostic_failed,
        "diagnostic_status_unknown": diagnostic_count
        - diagnostic_complete
        - diagnostic_failed,
        "diagnostic_skips_known": diagnostic_skips,
        "diagnostic_skip_counts_unknown": diagnostic_count - diagnostic_complete,
        "recorded_errors": error_count,
    }
    if expected_arm is None:
        report["role"] = "excluded_setup_only"
        return report
    submission_path = workspace / "submission.json"
    selection = string(required(read_object(submission_path), "candidate"), "candidate")
    if selection not in ("baseline", "candidate-01", "candidate-02"):
        raise ValueError("Unknown selected candidate")
    files.update({submission_path, workspace / "summary.md"})
    baseline = read_object(workspace / "baseline_development.json")
    development = metric(required(baseline, "final"), [1, 2, 4, 8])
    source = workspace / "baseline_research.py"
    if selection != "baseline":
        if selection not in valid:
            raise ValueError("Selected candidate has no validated completed training")
        development, source = (
            valid[selection],
            trial / "runs" / selection / "program.py",
        )
    selected = root / "selected" / trial_id
    selected_config = read_object(selected / "config.json")
    baseline_config = read_object(root / "selected" / "baseline" / "config.json")
    training = dict(mapping(required(selected_config, "training"), "training"))
    require_equal(
        required(training, "research_source"), text(source), "selected program"
    )
    training["research_source"] = required(
        mapping(required(baseline_config, "training"), "baseline training"),
        "research_source",
    )
    normalized = dict(selected_config)
    normalized["training"] = json_object(training, "selected training")
    require_equal(normalized, baseline_config, "frozen training configuration")
    if selection == "baseline":
        for name in ("config.json", "model.safetensors"):
            require_equal(
                digest(selected / name),
                digest(root / "selected" / "baseline" / name),
                "baseline artifact",
            )
    files.update({selected / "config.json", selected / "model.safetensors"})
    summary = (workspace / "summary.md").read_text()
    report.update(
        {
            "selection": selection,
            "development": development,
            "agent_summary_excerpt": summary[:2000],
            "agent_summary_truncated": len(summary) > 2000,
        }
    )
    return report


def collect(root: Path) -> tuple[list[JSONValue], list[JSONValue], set[Path]]:
    _, order, arms, excluded = protocol(root)
    files = {
        root / "selected" / "baseline" / name
        for name in ("config.json", "model.safetensors")
    }
    files.add(root / "heldout-evaluation-plan.json")
    files.add(root / "study_protocol.json")
    harness_path = root / "harness-freeze-B.json"
    packets_path = root / "solver-packet-manifest-B.json"
    files.update({harness_path, packets_path})
    verify_inventory(
        Path(__file__).resolve().parents[2],
        required(read_object(harness_path), "files"),
        files,
    )
    inventory = mapping(
        required(read_object(packets_path), "packet_inventory"), "packet inventory"
    )
    require_equal(set(inventory), set(order), "packet trial set")
    for trial in order:
        verify_inventory(root / "trials" / trial / "workspace", inventory[trial], files)
    reports: list[JSONValue] = [
        trial_summary(root, trial, files, arms[trial]) for trial in order
    ]
    setup: list[JSONValue] = [
        trial_summary(root, trial, files, None) for trial in excluded
    ]
    files.update(Path(__file__).resolve().parent.glob("*.py"))
    return reports, setup, files


def verify_inventory(base: Path, value: object, files: set[Path]) -> None:
    inventory = mapping(value, "hash inventory")
    if not inventory:
        raise ValueError("Empty evidence inventory")
    for name, saved_hash in inventory.items():
        path = (base / name).resolve(strict=True)
        if not path.is_relative_to(base.resolve()):
            raise ValueError("Inventory path outside its root")
        require_equal(digest(path), string(saved_hash, "saved hash"), "inventory hash")
        files.add(path)


def timestamp(value: object) -> datetime:
    parsed = datetime.fromisoformat(string(value, "timestamp").replace("Z", "+00:00"))
    if parsed.tzinfo is None or parsed.utcoffset() != timezone.utc.utcoffset(parsed):
        raise ValueError("Timestamp must explicitly identify UTC")
    return parsed


def save_new(path: Path, value: JSONObject) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x") as handle:
        handle.write(json.dumps(value, indent=2, allow_nan=False) + "\n")


def freeze(root: Path) -> None:
    _, _, files = collect(root)
    if (root / "scores").exists() and any((root / "scores").iterdir()):
        raise ValueError("Scores already exist before finalization")
    now = datetime.now(timezone.utc)
    study, _, _, _ = protocol(root)
    if timestamp(study["revised_at"]) >= now:
        raise ValueError("Protocol revision must precede finalization")
    save_new(
        root / "finalization.json",
        {
            "finalized_at": now.isoformat(),
            "files": {str(path.resolve()): digest(path) for path in sorted(files)},
        },
    )


def verify_freeze(root: Path) -> JSONObject:
    frozen = read_object(root / "finalization.json")
    _, _, files = collect(root)
    expected = mapping(required(frozen, "files"), "frozen files")
    require_equal(
        set(expected), {str(path.resolve()) for path in files}, "frozen file set"
    )
    for path, saved_hash in expected.items():
        require_equal(
            digest(Path(path)), string(saved_hash, "hash"), "frozen file hash"
        )
    timestamp(required(frozen, "finalized_at"))
    return frozen


def begin_scoring(root: Path) -> None:
    verify_freeze(root)
    if (root / "scores").exists() and any((root / "scores").iterdir()):
        raise ValueError("Scoring has already started")
    save_new(
        root / "scores" / "provenance.json",
        {
            "scoring_started_at": datetime.now(timezone.utc).isoformat(),
            "finalization_sha256": digest(root / "finalization.json"),
        },
    )


def analyze(root: Path) -> JSONObject:
    frozen = verify_freeze(root)
    provenance = read_object(root / "scores" / "provenance.json")
    require_equal(
        required(provenance, "finalization_sha256"),
        digest(root / "finalization.json"),
        "finalization provenance",
    )
    if timestamp(required(provenance, "scoring_started_at")) <= timestamp(
        required(frozen, "finalized_at")
    ):
        raise ValueError("Held-out scoring must follow finalization")
    reports, setup, _ = collect(root)
    study, order, _, _ = protocol(root)
    scores: dict[str, JSONObject] = {}
    panel: str | None = None
    panel_size: int | None = None
    plan = read_object(root / "heldout-evaluation-plan.json")
    for trial in ("baseline", *order):
        score = read_object(root / "scores" / (trial + ".json"))
        for key, expected in (
            ("kind", "reduced_compute_pilot_not_official_RSI_score"),
            ("split", "test"),
            ("fixed_policy_depths", POLICY),
            ("seed", required(plan, "seed")),
            (
                "requested_examples_per_bin",
                required(plan, "requested_examples_per_bin"),
            ),
        ):
            require_equal(required(score, key), expected, key)
        for key, name in (
            ("checkpoint_sha256", "model.safetensors"),
            ("config_sha256", "config.json"),
        ):
            require_equal(
                required(score, key), digest(root / "selected" / trial / name), key
            )
        current_panel = string(required(score, "panel_sha256"), "panel hash")
        if re.fullmatch(r"[0-9a-f]{64}", current_panel) is None:
            raise ValueError("Malformed test panel hash")
        current_size = integer(required(score, "actual_examples"), "actual_examples")
        if current_size < 1:
            raise ValueError("Test panel must contain observed examples")
        if panel is not None:
            require_equal(current_panel, panel, "test panel")
            require_equal(current_size, panel_size, "test panel size")
        panel = current_panel
        panel_size = current_size
        values = metric(required(score, "depths"), [1, 2, 4])
        require_equal(
            finite(required(score, "reward"), "reward"),
            values["reward"],
            "reward calculation",
        )
        values["raw_depths"] = score["depths"]
        scores[trial] = values
    for entry in reports:
        trial_report = mapping(entry, "trial report")
        if trial_report["selection"] == "baseline":
            require_equal(
                scores[string(trial_report["trial"], "trial")],
                scores["baseline"],
                "baseline selection score",
            )
    pairs: list[JSONValue] = []
    for index, pair_entry in enumerate(sequence(study["pairs"], "pairs"), 1):
        pair = mapping(pair_entry, "pair")
        treatment = string(required(pair, "traincheck"), "traincheck")
        control = string(required(pair, "control"), "control")
        pairs.append(
            {
                "pair": index,
                "control": control,
                "traincheck": treatment,
                "treatment_minus_control_reward": finite(
                    scores[treatment]["reward"], "reward"
                )
                - finite(scores[control]["reward"], "reward"),
            }
        )
    return {
        "kind": "two_pair_development_pilot_no_statistical_claim",
        "trials": reports,
        "excluded_setup_trials": setup,
        "exclusion_reason": study["revision_reason"],
        "heldout": json_object(scores, "heldout scores"),
        "pairs": pairs,
        "test_panel_sha256": panel,
        "finalization_sha256": digest(root / "finalization.json"),
        "limitations": [
            "Training time for failed or unobserved training attempts remains unknown, not zero.",
            "Operational durations include queue wait and transfer; they are not agent wall time or token usage.",
            "Agent summary excerpts are verbatim self-reports, not verified causal explanations.",
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("freeze", "begin-scoring", "analyze"))
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    root = args.root.resolve(strict=True)
    if args.mode == "freeze":
        freeze(root)
    elif args.mode == "begin-scoring":
        begin_scoring(root)
    else:
        print(json.dumps(analyze(root), indent=2, allow_nan=False))


if __name__ == "__main__":
    main()

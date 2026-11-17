"""Join frozen decision boundaries to private per-source replay receipts."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

from tools.checkpoint_pilot.accounting import parse_exec_jsonl

SCENARIOS = {
    "fixed-regression",
    "exact-uniform",
    "overshoot-and-tail",
    "rank-skew",
    "single-window",
    "one-microbatch-updates",
}
OUTCOMES = {
    "pass",
    "silent_wrong",
    "unsupported",
    "runtime_error",
    "nonfinite",
    "memory_contract",
}


def record(value: object) -> dict[str, object]:
    if not isinstance(value, dict) or not all(isinstance(k, str) for k in value):
        raise ValueError("Expected a JSON object with string keys")
    return dict(value)


def read(path: Path) -> dict[str, object]:
    return record(json.loads(path.read_bytes()))


def duration(value: object) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (float, int))
        or not math.isfinite(value)
        or value < 0
    ):
        raise ValueError(f"Invalid duration {value!r}")
    return float(value)


def grade_status(receipt: Path, sha: str, treatment: bool) -> str:
    outer = read(receipt)
    request = read(receipt.with_name("request.json"))
    files = record(request["files"])
    source = files["training.py"]
    if (
        request["kind"] != "grade"
        or request["with_traincheck"] is not treatment
        or not isinstance(source, str)
    ):
        raise ValueError(f"Grade request has different execution conditions: {receipt}")
    if (
        hashlib.sha256(source.encode()).hexdigest() != sha
        or outer["id"] != request["id"]
    ):
        raise ValueError(f"Grade request and source disagree: {receipt}")
    if outer["status"] != "ok":
        return "unknown"
    grade = record(outer["grade"])
    if grade["source_sha256"] != sha:
        raise ValueError(f"Grader replayed a different source: {receipt}")
    overall = grade["status"]
    if overall in ("timeout", "incomplete_execution"):
        return "unknown"
    if not isinstance(overall, str) or overall not in OUTCOMES:
        raise ValueError(f"Invalid grade status {overall!r}")
    rows = grade["rank_results"]
    if not isinstance(rows, list):
        raise ValueError("Missing scenario outcomes")
    covered: set[tuple[str, int]] = set()
    observed: set[str] = set()
    for raw in rows:
        row = record(raw)
        name, rank = row["name"], row["rank"]
        outcome = record(row["result"])["status"]
        if (
            not isinstance(name, str)
            or (not isinstance(rank, int) or isinstance(rank, bool))
            or not isinstance(outcome, str)
            or outcome not in OUTCOMES
        ):
            raise ValueError("Invalid scenario row")
        if (name, rank) in covered:
            raise ValueError("Duplicate scenario/rank")
        covered.add((name, rank))
        observed.add(outcome)
    if covered != {(name, rank) for name in SCENARIOS for rank in (0, 1)}:
        raise ValueError("Incomplete scenario/rank coverage")
    if (overall == "pass") != (observed == {"pass"}) or overall not in observed:
        raise ValueError("Inconsistent aggregate grade")
    return overall


def score(trial: Path, grades: Path) -> dict[str, object]:
    """Grade locations are grouped by optional-kit availability and source digest."""
    arm = trial.name.split("_")[0]
    if arm not in {"control", "traincheck"}:
        raise ValueError(f"Invalid trial name {trial.name}")
    prior_gpu = prior_queue = 0.0
    req = 0
    first: int | None = None
    earlier_unknown: list[int] = []
    rows: list[dict[str, object]] = []
    for n in range(1, 7):
        turn = trial / f"turn-{n}"
        if not (turn / "submission.json").is_file():
            break
        sub = read(turn / "submission.json")
        sha = hashlib.sha256((turn / "submission/training.py").read_bytes()).hexdigest()
        if (
            sub["production_sha256"] != sha
            or sub["trial"] != trial.name
            or sub["turn"] != n
        ):
            raise ValueError(
                f"Submission metadata disagrees with frozen source in {turn}"
            )
        ledger = parse_exec_jsonl((turn / "events.jsonl").read_text())
        usage = record(sub["usage_cumulative"])
        if ledger.thread_id != sub["thread_id"] or any(
            usage[k] != getattr(ledger.usage, k)
            for k in ("input_tokens", "cached_input_tokens", "output_tokens")
        ):
            raise ValueError("Mismatched native token counters or thread")
        if not math.isclose(
            prior_gpu, duration(sub["allocated_gpu_seconds"])
        ) or not math.isclose(prior_queue, duration(sub["queue_seconds"])):
            raise ValueError(
                f"Submission cost disagrees with earlier request receipts in {turn}"
            )
        grade = grade_status(
            grades / arm / sha / "receipt.json", sha, arm == "traincheck"
        )
        if first is None:
            if grade == "pass":
                first = n
            elif grade == "unknown":
                earlier_unknown.append(n)
        u = ledger.usage
        decision = record(sub["decision"])
        rows.append(
            dict(
                turn=n,
                production_sha256=sha,
                action=decision["action"],
                grade=grade,
                wall_seconds=duration(sub["wall_seconds"]),
                cli_seconds=duration(sub["cli_seconds"]),
                queue_seconds=prior_queue,
                gpu_seconds=prior_gpu,
                total_tokens=u.total_tokens,
                input_tokens=u.input_tokens,
                cached_input_tokens=u.cached_input_tokens,
                output_tokens=u.output_tokens,
                uncached_input_plus_output=u.total_tokens - u.cached_input_tokens,
            )
        )
        if decision["action"] == "run":
            receipt = trial / f"request-{req + 1}" / "receipt.json"
            if receipt.exists():
                req += 1
                result = read(receipt)
                prior_gpu += duration(result["gpu_seconds"])
                prior_queue += duration(result["queue_seconds"])
    if not rows:
        raise ValueError(f"No frozen submission in {trial}")
    last = read(trial / "final.json")
    if req != last["requests"] or not math.isclose(
        prior_gpu, duration(last["gpu_seconds"])
    ):
        raise ValueError("Terminal service costs disagree")
    return dict(
        trial=trial.name,
        terminal=last,
        submissions=rows,
        earliest_observed_pass=first,
        earlier_unknown_turns=earlier_unknown,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("evidence", type=Path)
    parser.add_argument("out", type=Path)
    args = parser.parse_args()
    trials = [
        score(args.evidence / "trials" / f"{arm}_{i}", args.evidence / "grades")
        for i in range(1, 4)
        for arm in ("control", "traincheck")
    ]
    with args.out.open("x") as stream:
        json.dump(trials, stream, indent=2, allow_nan=False)
        stream.write("\n")


if __name__ == "__main__":
    main()

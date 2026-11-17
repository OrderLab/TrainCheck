"""Replay distinct production sources after all six native trials finish."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path

from tools.token_budget_study.operator.agent_trials import MAX_TURNS, SERVICE, digest
from tools.token_budget_study.operator.score_submissions import (
    duration,
    grade_status,
    read,
)

ARMS = ("control", "traincheck")


def sources(root: Path) -> dict[tuple[str, str], Path]:
    launch = read(root / "native-launch.json")
    if launch["service_sha256"] != digest(SERVICE):
        raise ValueError("The worker differs from the frozen launcher")
    identities: dict[tuple[str, str], Path] = {}
    occurrences: list[dict[str, str]] = []
    terminal: dict[str, str] = {}
    for pair in range(1, 4):
        for arm in ARMS:
            name = f"{arm}_{pair}"
            trial = root / "trials" / name
            terminal[name] = digest(trial / "final.json")
            for turn in range(1, MAX_TURNS + 1):
                decision = trial / f"turn-{turn}" / "submission.json"
                if not decision.is_file():
                    continue
                path = decision.parent / "submission" / "training.py"
                sha = digest(path)
                if read(decision)["production_sha256"] != sha:
                    raise ValueError(f"Frozen submission changed: {path}")
                identities[arm, sha] = path
                occurrences.append(
                    {"trial": name, "path": str(path.relative_to(root)), "sha256": sha}
                )
                watch = trial / f"turn-{turn}.watch" / "index.jsonl"
                for blob in sorted(watch.parent.glob("blobs/*.py")):
                    if digest(blob) != blob.stem:
                        raise ValueError(f"Observed revision changed: {blob}")
                    identities.setdefault((arm, blob.stem), blob)
    manifest = root / "private-replay-inventory.json"
    data = json.dumps(
        {
            "occurrences": occurrences,
            "terminal_hashes": terminal,
            "native_launch": digest(root / "native-launch.json"),
            "distinct": len(identities),
        },
        indent=2,
    )
    if manifest.exists():
        if manifest.read_text() != data:
            raise ValueError("Existing replay inventory differs")
    else:
        with manifest.open("x") as stream:
            stream.write(data)
    return identities


def run(root: Path, scenarios: set[str] | None = None) -> None:
    reports: list[dict[str, object]] = []
    total = 0.0
    for (arm, sha), original in sources(root).items():
        source = root / "grade-sources" / arm / sha
        source.mkdir(parents=True, exist_ok=True)
        code = source / "training.py"
        if not code.exists():
            with code.open("xb") as stream:
                stream.write(original.read_bytes())
        if digest(code) != sha:
            raise ValueError("Replay source contents disagree with inventory")
        output = root / "grades" / arm / sha
        receipt = output / "receipt.json"
        if not receipt.is_file():
            request = [
                sys.executable,
                str(SERVICE),
                "grade",
                "--source",
                str(source),
                "--id",
                f"grade-{arm}-{sha}",
                "--output",
                str(output),
            ]
            if arm == "traincheck":
                request.append("--with-traincheck")
            subprocess.run(request, check=True)
        result = grade_status(receipt, sha, arm == "traincheck", scenarios)
        gpu = duration(read(receipt)["gpu_seconds"])
        total += gpu
        reports.append(
            {
                "arm": arm,
                "sha256": sha,
                "outcome": result,
                "receipt_sha256": hashlib.sha256(receipt.read_bytes()).hexdigest(),
                "gpu_seconds": gpu,
            }
        )
        print(arm, sha[:10], result, round(gpu, 3), flush=True)
    with (root / "private-grading-summary.json").open("x") as stream:
        json.dump({"grades": reports, "gpu_seconds": total}, stream, indent=2)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("--scenario", action="append")
    args = parser.parse_args()
    run(args.root, set(args.scenario) if args.scenario else None)


if __name__ == "__main__":
    main()

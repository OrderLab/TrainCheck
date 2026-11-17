"""Prepare immutable positive and targeted negative control sources for qualification."""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path


def prepare(public: Path, output: Path) -> None:
    positive = Path(__file__).with_name("positive_control.py").read_text()
    site = "parameter.grad.mul_(dist.get_world_size() / tokens)"
    # Emulate using the requested budget as denominator even on overshoot/tail.
    if positive.count(site) != 1:
        raise ValueError("Negative control mutation site must be unique")
    negative = positive.replace(
        site, "parameter.grad.mul_(dist.get_world_size() / (token_budget or tokens))"
    )
    output.mkdir(parents=True, exist_ok=False)
    for name, code in (("positive", positive), ("wrong-denominator", negative)):
        path = output / name
        path.mkdir()
        for file in ("workload.py", "train.py"):
            shutil.copyfile(public / file, path / file)
        (path / "training.py").write_text(code)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--public", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    prepare(args.public, args.output)


if __name__ == "__main__":
    main()

"""Compare explicitly selected model snapshots, rejecting incomplete coverage."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path
from typing import Sequence

from traincheck.observation import compare_model_snapshots, compare_ranks, read_snapshot


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="mode", required=True)
    ranks = commands.add_parser(
        "ranks", help="compare one phase across a complete rank set"
    )
    ranks.add_argument("--expected-ranks", type=int, nargs="+", required=True)
    ranks.add_argument("--reference-rank", type=int, default=0)
    ranks.add_argument("--snapshots", type=Path, nargs="+", required=True)
    pair = commands.add_parser(
        "pair", help="compare two explicitly chosen states, including different phases"
    )
    pair.add_argument("reference", type=Path)
    pair.add_argument("candidate", type=Path)
    args = parser.parse_args(argv)
    try:
        if args.mode == "ranks":
            paths: object = args.snapshots
            expected: object = args.expected_ranks
            reference_rank: object = args.reference_rank
            assert isinstance(paths, list) and all(
                isinstance(path, Path) for path in paths
            )
            assert isinstance(expected, list) and all(
                isinstance(rank, int) for rank in expected
            )
            assert isinstance(reference_rank, int)
            result = compare_ranks(
                [read_snapshot(path) for path in paths],
                expected_ranks=expected,
                reference_rank=reference_rank,
            )
            print(json.dumps({"equal": result.equal, **asdict(result)}, indent=2))
            return 0 if result.equal else 1
        reference: object = args.reference
        candidate: object = args.candidate
        assert isinstance(reference, Path) and isinstance(candidate, Path)
        comparison = compare_model_snapshots(
            read_snapshot(reference), read_snapshot(candidate)
        )
        print(json.dumps({"equal": comparison.equal, **asdict(comparison)}, indent=2))
        return 0 if comparison.equal else 1
    except (OSError, TypeError, ValueError) as error:
        parser.error(str(error))


if __name__ == "__main__":
    raise SystemExit(main())

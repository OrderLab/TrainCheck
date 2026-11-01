"""Operator bootstrap; the solver packet contains only the training_task package."""

from __future__ import annotations

import os
import sys
from pathlib import Path


def main() -> None:
    workspace = Path(os.environ["CONTINUATION_WORKSPACE"]).resolve(strict=True)
    trace_root = Path(os.environ["CONTINUATION_TRACE_ROOT"]).resolve()
    rank = int(os.environ["RANK"])
    if rank < 0:
        raise ValueError("RANK must be nonnegative")
    trace = trace_root / f"rank{rank}"
    trace.mkdir(parents=True, exist_ok=False)
    os.environ["TRAINCHECK_OUTPUT_DIR"] = str(trace)
    sys.path.insert(0, str(workspace))
    # Import after setting per-rank output so optional diagnostics see the correct
    # sink even when they install wrappers during module initialization.
    from training_task.engine import main as train

    train()


if __name__ == "__main__":
    main()

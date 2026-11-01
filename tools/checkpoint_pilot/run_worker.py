"""Set per-rank observation paths before importing training or diagnostic code."""

from __future__ import annotations

import os
import runpy
import sys
from pathlib import Path


def main() -> None:
    workspace = Path(os.environ["CHECKPOINT_WORKSPACE"]).resolve(strict=True)
    output = Path(os.environ["EXPERIMENT_OUTPUT"]).resolve()
    rank = int(os.environ["RANK"])
    if rank < 0:
        raise ValueError("RANK must be nonnegative")
    trace = output / "traces" / f"rank{rank}"
    trace.mkdir(parents=True, exist_ok=False)
    os.environ["TRAINCHECK_OUTPUT_DIR"] = str(trace)
    sys.path.insert(0, str(workspace))
    os.chdir(workspace)
    # Absence selects the fixed training engine; presence must name a packet
    # Python file and may not escape the explicitly supplied workspace.
    script = os.environ.get("EXPERIMENT_SCRIPT")
    if script is None:
        runpy.run_module("task.engine", run_name="__main__")
    else:
        path = (workspace / script).resolve(strict=True)
        if (
            not path.is_relative_to(workspace)
            or path.suffix != ".py"
            or not path.is_file()
        ):
            raise ValueError("Diagnostic script must be a Python file in the packet")
        runpy.run_path(str(path), run_name="__main__")


if __name__ == "__main__":
    main()

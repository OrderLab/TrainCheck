"""Exercise the agent-facing capture and comparison commands end to end."""

from __future__ import annotations

import contextlib
import io
import os
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

import torch

from tools.model_observation.capture_example import Settings, run
from tools.model_observation.compare import main as compare_main
from traincheck.observation import (
    compare_model_snapshots,
    read_snapshot,
    write_snapshot,
)


class ModelObservationExampleTests(unittest.TestCase):
    def capture(self, directory: Path) -> None:
        environment = {
            name: value
            for name, value in os.environ.items()
            if name not in {"RANK", "LOCAL_RANK", "WORLD_SIZE"}
        }
        with (
            patch.dict(os.environ, environment, clear=True),
            torch.random.fork_rng(),
            contextlib.redirect_stdout(io.StringIO()),
        ):
            run(Settings(directory, "example-test", "cpu", 2))

    def test_capture_observes_optimizer_updates_and_nonpersistent_buffer(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            self.capture(directory)
            before = read_snapshot(directory / "rank0-ready-0.json")
            after = read_snapshot(directory / "rank0-update-2.json")
            result = compare_model_snapshots(before, after)
            self.assertFalse(result.equal)
            self.assertEqual(result.tensors_compared, 5)
            self.assertNotIn(
                "example_counter", {item.name for item in result.differences}
            )

    def test_comparison_commands_report_equal_different_and_missing(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            self.capture(directory)
            before = str(directory / "rank0-ready-0.json")
            after = str(directory / "rank0-update-2.json")
            other = directory / "rank1-ready-0.json"
            write_snapshot(replace(read_snapshot(Path(before)), rank=1), other)
            with contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(compare_main(["pair", before, before]), 0)
                self.assertEqual(compare_main(["pair", before, after]), 1)
                self.assertEqual(
                    compare_main(
                        [
                            "ranks",
                            "--expected-ranks",
                            "0",
                            "1",
                            "--snapshots",
                            before,
                            str(other),
                        ]
                    ),
                    0,
                )
            with (
                contextlib.redirect_stderr(io.StringIO()),
                self.assertRaises(SystemExit) as raised,
            ):
                compare_main(
                    ["ranks", "--expected-ranks", "0", "1", "--snapshots", before]
                )
            self.assertEqual(raised.exception.code, 2)

    def test_duplicate_rank_files_are_not_accepted(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            self.capture(directory)
            before = str(directory / "rank0-ready-0.json")
            with (
                contextlib.redirect_stderr(io.StringIO()),
                self.assertRaises(SystemExit) as raised,
            ):
                compare_main(
                    [
                        "ranks",
                        "--expected-ranks",
                        "0",
                        "1",
                        "--snapshots",
                        before,
                        before,
                    ]
                )
            self.assertEqual(raised.exception.code, 2)

    def test_missing_file_is_an_error(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            missing = str(Path(temporary) / "absent.json")
            with (
                contextlib.redirect_stderr(io.StringIO()),
                self.assertRaises(SystemExit) as raised,
            ):
                compare_main(["pair", missing, missing])
            self.assertEqual(raised.exception.code, 2)

    def test_partial_distributed_environment_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            environment = {
                name: value
                for name, value in os.environ.items()
                if name not in {"RANK", "LOCAL_RANK", "WORLD_SIZE"}
            }
            environment["RANK"] = "0"
            with (
                patch.dict(os.environ, environment, clear=True),
                self.assertRaisesRegex(ValueError, "requires RANK"),
            ):
                run(Settings(Path(temporary), "example-test", "cpu", 2))

    def test_existing_observations_are_not_overwritten(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            self.capture(directory)
            path = directory / "rank0-ready-0.json"
            original = path.read_bytes()
            with self.assertRaises(FileExistsError):
                self.capture(directory)
            self.assertEqual(path.read_bytes(), original)


if __name__ == "__main__":
    unittest.main()

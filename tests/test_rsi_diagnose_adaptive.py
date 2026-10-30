"""Fail-closed contracts for candidate diagnostic phase results."""

from __future__ import annotations

import json
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from tools.rsi_pilot.diagnose_adaptive import Driver, main


class TestDiagnosticPhases(unittest.TestCase):
    def phase_result(self, output: Path, **changes: object) -> None:
        result: dict[str, object] = {
            "exit_code": 0,
            "stop_reason": None,
            "source_unchanged": True,
        }
        result.update(changes)
        (output / "phase-result.json").write_text(json.dumps(result))

    def test_checker_errors_can_be_reported_but_collection_errors_fail(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            driver = Driver(output, Path("/python"))
            self.phase_result(output, exit_code=1)
            with patch(
                "subprocess.run", return_value=subprocess.CompletedProcess([], 1)
            ):
                driver.phase(
                    output, ["checker.py"], timeout=90, allow_checker_failure=True
                )
                with self.assertRaisesRegex(RuntimeError, "Diagnostic phase failed"):
                    driver.phase(output, ["collector.py"], timeout=90)

    def test_stopped_or_changed_source_never_becomes_checker_result(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            driver = Driver(output, Path("/python"))
            cases: tuple[dict[str, object], ...] = (
                {"stop_reason": "timeout"},
                {"source_unchanged": False},
                {"exit_code": -15},
            )
            for changes in cases:
                self.phase_result(output, **changes)
                with patch(
                    "subprocess.run", return_value=subprocess.CompletedProcess([], 0)
                ):
                    with self.assertRaisesRegex(
                        RuntimeError, "Diagnostic phase failed"
                    ):
                        driver.phase(
                            output,
                            ["checker.py"],
                            timeout=90,
                            allow_checker_failure=True,
                        )

    def test_missing_required_result_field_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            (output / "phase-result.json").write_text('{"exit_code":0}')
            with patch(
                "subprocess.run", return_value=subprocess.CompletedProcess([], 0)
            ):
                with self.assertRaises((ValueError, KeyError)):
                    Driver(output, Path("/python")).phase(
                        output, ["check.py"], timeout=90
                    )

    def test_candidate_config_is_required(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            with self.assertRaisesRegex(FileNotFoundError, "configuration missing"):
                Driver(output, Path("/python")).diagnose(output)
            self.assertFalse((output / "diagnostics").exists())

    def test_cli_preserves_virtualenv_python_symlink(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            base_python = root / "base-python"
            base_python.touch()
            venv_python = root / "venv-python"
            venv_python.symlink_to(base_python)
            argv = [
                "diagnose_adaptive.py",
                "--study",
                str(root),
                "--python",
                str(venv_python),
                str(root / "candidate"),
            ]
            with (
                patch("sys.argv", argv),
                patch(
                    "tools.rsi_pilot.diagnose_adaptive.Driver", autospec=True
                ) as driver,
                patch("builtins.print"),
            ):
                driver.return_value.diagnose.return_value = root / "report.json"
                main()
            driver.assert_called_once_with(root.resolve(), venv_python, "8,10,12,14")


if __name__ == "__main__":
    unittest.main()

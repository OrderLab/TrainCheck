"""Boundary checks for the pilot client; training is tested on the real runtime."""

import hashlib
import json
import shlex
import subprocess
import sys
import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import patch

from tools.fresh_validation.contracts import JSONObject, JSONValue, read_object
from tools.rsi_pilot.run_candidate import (
    configure,
    execute,
    main,
    program_text,
    remote_command,
    run,
    train_arguments,
    validate_diagnostic_artifacts,
    validate_diagnostics,
    validate_results,
)


def trial_fixture(root: Path, arm: str = "control") -> tuple[Path, Path]:
    trial, workspace = root / "trial", root / "solver"
    trial.mkdir()
    workspace.mkdir()
    program = workspace / "research.py"
    program.write_text(
        "def loss(logits, tokens, labels, step, total_steps):\n    return logits.sum()\n"
    )
    profile = root / "profile.toml"
    profile.write_text("[training]\nsteps = 5\n# PILOT_RESEARCH_SOURCE\n")
    manifest = {
        "workspace": str(workspace),
        "arm": arm,
        "profile": str(profile),
        "profile_sha256": hashlib.sha256(profile.read_bytes()).hexdigest(),
        "max_candidates": 3,
        "max_diagnostics": 1 if arm == "traincheck" else 0,
        "expected_steps": 5,
        "host": "unused-host",
        "remote_trial": "/experiment/trial",
        "remote_workspace": "/experiment/workspace",
        "remote_overlay": "/experiment/overlay",
        "remote_python": "/experiment/python",
        "affinity": "8,10,12,14",
        "diagnostic_driver": "/experiment/diagnose",
        "remote_compute_lock": "/experiment/compute.lock",
    }
    (trial / "trial.json").write_text(json.dumps(manifest))
    return trial, program


def training_result() -> JSONObject:
    return {
        "completed_steps": 5,
        "training_seconds": 0.25,
        "device_backend": "cpu",
        "final": [{"loops": 1, "macro_accuracy_percent": 20.0}],
    }


class PilotClientTests(unittest.TestCase):
    def test_program_is_data_not_shell_code(self) -> None:
        source = 'def loss(logits, tokens, labels, step, total_steps):\n    return "`touch sentinel` $HOME"\n'
        config = configure("[training]\n# PILOT_RESEARCH_SOURCE\n", source)
        self.assertIn('research_source = "def loss', config)
        self.assertIn("\\n", config)
        self.assertNotIn("\n    return", config)
        arguments = ["python", "path with spaces", "$(touch sentinel)", "x'y"]
        command = remote_command({"host": "labmach"}, arguments)
        self.assertEqual(shlex.split(command[-1]), arguments)

    def test_profile_marker_is_required_and_unique(self) -> None:
        for profile in ("", "# PILOT_RESEARCH_SOURCE\n# PILOT_RESEARCH_SOURCE"):
            with self.assertRaises(ValueError):
                configure(profile, "pass")

    def test_unicode_source_is_not_encoded_as_invalid_toml_surrogates(self) -> None:
        source = 'def loss(logits, tokens, labels, step, total_steps):\n    "😀"\n    return logits.sum()\n'
        encoded = configure("# PILOT_RESEARCH_SOURCE", source)
        self.assertNotIn("\\ud83d", encoded)
        self.assertEqual(json.loads(encoded.removeprefix("research_source = ")), source)

    def test_only_contained_small_utf8_programs(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            workspace = root / "solver"
            workspace.mkdir()
            program = workspace / "research.py"
            program.write_text("pass\n")
            self.assertEqual(program_text(program, workspace), "pass\n")
            outside = root / "outside.py"
            outside.write_text("pass\n")
            with self.assertRaises(ValueError):
                program_text(outside, workspace)
            program.write_bytes(b"x" * 32769)
            with self.assertRaises(ValueError):
                program_text(program, workspace)
            program.write_bytes(b"\xff")
            with self.assertRaises(UnicodeDecodeError):
                program_text(program, workspace)

    def test_failed_uploads_are_charged_and_have_nonempty_error_receipts(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            trial, program = trial_fixture(Path(temporary))
            with (
                patch("tools.rsi_pilot.run_candidate.checked_execute"),
                patch(
                    "tools.rsi_pilot.run_candidate.upload",
                    side_effect=subprocess.CalledProcessError(255, ["scp"]),
                ),
            ):
                for index in range(1, 4):
                    with self.assertRaises(subprocess.CalledProcessError):
                        run(trial, program, False)
                    receipt = read_object(
                        trial / "runs" / f"candidate-{index:02d}" / "run.json"
                    )
                    self.assertEqual(receipt["status"], "client_error")
                    self.assertIn(
                        '"phase": "upload_config"', json.dumps(receipt["errors"])
                    )
                    self.assertIn(
                        "non-zero exit status 255", json.dumps(receipt["errors"])
                    )
                with self.assertRaisesRegex(ValueError, "Candidate budget exhausted"):
                    run(trial, program, False)
            self.assertEqual(
                read_object(trial / "usage.json"),
                {
                    "candidates_reserved": 3,
                    "diagnostics_reserved": 0,
                },
            )

    def test_one_diagnostic_attempt_is_reserved_even_when_transfer_fails(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            trial, program = trial_fixture(Path(temporary), "traincheck")
            with (
                patch("tools.rsi_pilot.run_candidate.checked_execute"),
                patch(
                    "tools.rsi_pilot.run_candidate.upload",
                    side_effect=OSError("connection lost"),
                ),
            ):
                with self.assertRaises(OSError):
                    run(trial, program, True)
                with self.assertRaisesRegex(ValueError, "Diagnostic budget exhausted"):
                    run(trial, program, True)
            self.assertEqual(
                read_object(trial / "usage.json"),
                {
                    "candidates_reserved": 1,
                    "diagnostics_reserved": 1,
                },
            )

    def test_control_cannot_request_diagnostics(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            trial, program = trial_fixture(Path(temporary))
            with self.assertRaisesRegex(ValueError, "not enabled"):
                run(trial, program, True)
            self.assertFalse((trial / "usage.json").exists())

    def test_remote_training_failure_remains_failure_without_results(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            trial, program = trial_fixture(Path(temporary))
            with (
                patch("tools.rsi_pilot.run_candidate.checked_execute"),
                patch("tools.rsi_pilot.run_candidate.upload"),
                patch("tools.rsi_pilot.run_candidate.execute", return_value=124),
                patch("tools.rsi_pilot.run_candidate.download") as downloader,
            ):
                result = run(trial, program, False)
            downloader.assert_not_called()
            self.assertEqual(result["status"], "training_failed")
            self.assertNotIn("development_depths", result)
            self.assertIn("Training exited 124", json.dumps(result["errors"]))

    def test_failed_result_download_has_receipt_and_does_not_claim_completion(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            trial, program = trial_fixture(Path(temporary))
            with (
                patch("tools.rsi_pilot.run_candidate.checked_execute"),
                patch("tools.rsi_pilot.run_candidate.upload"),
                patch("tools.rsi_pilot.run_candidate.execute", return_value=0),
                patch(
                    "tools.rsi_pilot.run_candidate.download",
                    side_effect=OSError("download lost"),
                ),
            ):
                with self.assertRaises(OSError):
                    run(trial, program, False)
            receipt = read_object(trial / "runs/candidate-01/run.json")
            self.assertEqual(receipt["status"], "client_error")
            self.assertEqual(receipt["training_exit_code"], 0)
            self.assertNotIn("development_depths", receipt)
            self.assertIn("download_results", json.dumps(receipt["errors"]))

    def test_invalid_result_metadata_is_not_accepted_as_success(self) -> None:
        manifest: JSONObject = {"expected_steps": 5}
        valid = training_result()
        self.assertEqual(validate_results(valid, manifest)[0], 5)
        changes: list[tuple[str, JSONValue]] = [
            ("completed_steps", True),
            ("completed_steps", 4),
            ("training_seconds", float("nan")),
            ("training_seconds", -1),
            ("device_backend", "cuda"),
            ("unrelated_metric", float("nan")),
            ("final", []),
            ("final", [{"loops": 1, "macro_accuracy_percent": float("inf")}]),
        ]
        for key, value in changes:
            with self.subTest(key=key, value=value), self.assertRaises(ValueError):
                validate_results({**valid, key: value}, manifest)
        del valid["completed_steps"]
        with self.assertRaises(ValueError):
            validate_results(valid, manifest)

    def test_malformed_diagnostic_cannot_be_successfully_downloaded_evidence(
        self,
    ) -> None:
        with self.assertRaises(ValueError):
            validate_diagnostics({})
        with self.assertRaises(ValueError):
            validate_diagnostics({"schema_version": 1, "outcome": "success"})

    def test_cli_reports_nonzero_for_a_recorded_training_failure(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            with (
                patch(
                    "sys.argv",
                    [
                        "run_candidate",
                        "--trial",
                        str(root),
                        "--program",
                        str(root / "program.py"),
                    ],
                ),
                patch(
                    "tools.rsi_pilot.run_candidate.run",
                    return_value={
                        "status": "training_failed",
                        "errors": [{"message": "Training exited 1"}],
                    },
                ),
            ):
                self.assertEqual(main(), 1)

    def test_malformed_usage_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            trial, program = trial_fixture(Path(temporary))
            (trial / "usage.json").write_text(
                '{"candidates_reserved": false, "diagnostics_reserved": 0}'
            )
            with self.assertRaises(ValueError):
                run(trial, program, False)
            self.assertFalse((trial / "runs/candidate-01").exists())

    def test_diagnostic_drilldown_requires_matching_bytes_and_candidate(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary)
            coverage_path = output / "coverage.json"
            coverage_path.write_text('{"schema_version": 3, "per_invariant": []}\n')
            config = output / "config.toml"
            config.write_text("[training]\nsteps = 5\n")
            diagnostic: JSONObject = {
                "schema_version": 1,
                "status": "complete",
                "candidate_config_sha256": hashlib.sha256(
                    config.read_bytes()
                ).hexdigest(),
            }
            (output / "diagnostic_manifest.json").write_text(json.dumps(diagnostic))
            source_path = "/remote/candidate/diagnostics/coverage.json"
            report: JSONObject = {
                "coverage_source": {
                    "path": source_path,
                    "sha256": hashlib.sha256(coverage_path.read_bytes()).hexdigest(),
                }
            }
            evidence = validate_diagnostic_artifacts(report, output)
            self.assertIn(str(coverage_path), json.dumps(evidence))
            self.assertIn(source_path, json.dumps(report))
            config.write_text("[training]\nsteps = 6\n")
            with self.assertRaisesRegex(ValueError, "different candidate"):
                validate_diagnostic_artifacts(report, output)
            coverage_path.write_text("{}\n")
            with self.assertRaisesRegex(ValueError, "does not match"):
                validate_diagnostic_artifacts(report, output)

    def test_missing_diagnostic_drilldown_is_an_error(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            with self.assertRaises(FileNotFoundError):
                validate_diagnostic_artifacts({}, Path(temporary))

    def test_cpu_thread_settings_match_calibration(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            trial, _ = trial_fixture(Path(temporary))
            command = train_arguments(read_object(trial / "trial.json"), "/run")
            for setting in (
                "OMP_NUM_THREADS=4",
                "MKL_NUM_THREADS=4",
                "OPENBLAS_NUM_THREADS=1",
            ):
                self.assertIn(setting, command)
            self.assertEqual(command[-3:], ["--device", "cpu", "--require-complete"])
            self.assertEqual(
                command[:4], ["flock", "-w", "360", "/experiment/compute.lock"]
            )
            self.assertIn("300s", command)

    def test_timeout_stops_owned_descendants(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            sentinel = root / "descendant-ran"
            child = (
                "import pathlib,time; time.sleep(2); pathlib.Path("
                + repr(str(sentinel))
                + ").write_text('leaked')"
            )
            parent = (
                "import subprocess,sys,time; subprocess.Popen([sys.executable,'-c',"
                + repr(child)
                + "]); time.sleep(30)"
            )
            code = execute([sys.executable, "-c", parent], root / "process.log", 1)
            self.assertEqual(code, 124)
            time.sleep(1.5)
            self.assertFalse(sentinel.exists())
            self.assertIn("CLIENT_TIMEOUT", (root / "process.log").read_text())


if __name__ == "__main__":
    unittest.main()

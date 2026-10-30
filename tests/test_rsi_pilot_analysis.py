"""Evidence failures must prevent an apparently successful pilot comparison."""

import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from tools.fresh_validation.contracts import JSONObject, JSONValue, mapping, read_object
from tools.rsi_pilot.analyze_trials import (
    analyze,
    begin_scoring,
    collect,
    digest,
    freeze,
    metric,
)
from tools.rsi_pilot.run_candidate import configure

ORDER = ("B1C", "B1T", "B2T", "B2C")


def write(path: Path, value: JSONObject) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, allow_nan=False) + "\n")


def depths(accuracy: float, include_eight: bool = True) -> list[JSONValue]:
    return [
        {
            "loops": depth,
            "macro_accuracy_percent": accuracy,
            "macro_mean_dense_matmul_flops": work,
        }
        for depth, work in ((1, 805760), (2, 1610624), (4, 3220352), (8, 6439808))
        if depth != 8 or include_eight
    ]


class TestPilotAnalysis(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory[str]()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.profile = self.root / "frozen-profile.toml"
        self.profile.write_text("[training]\nsteps = 10000\n# PILOT_RESEARCH_SOURCE\n")
        self.baseline = "def loss():\n    return 1\n"
        self.baseline_config: JSONObject = {
            "model": {"width": 64},
            "training": {"steps": 10000, "research_source": self.baseline},
        }
        write(self.root / "selected/baseline/config.json", self.baseline_config)
        (self.root / "selected/baseline/model.safetensors").write_bytes(
            b"baseline model"
        )
        write(
            self.root / "study_protocol.json",
            {
                "revised_at": "2026-10-30T00:00:00+00:00",
                "revision_reason": "Restore omitted free validation before the scored pilot.",
                "trial_order": list(ORDER),
                "pairs": [
                    {"control": "B1C", "traincheck": "B1T"},
                    {"control": "B2C", "traincheck": "B2T"},
                ],
                "excluded_setup_trials": ["C1"],
            },
        )
        write(
            self.root / "heldout-evaluation-plan.json",
            {"seed": 20261030, "requested_examples_per_bin": 512},
        )
        for trial_id in (*ORDER, "C1"):
            self.make_trial(trial_id)
        source = (
            Path(__file__).resolve().parents[1] / "tools/rsi_pilot/run_candidate.py"
        )
        write(
            self.root / "harness-freeze-B.json",
            {"files": {"tools/rsi_pilot/run_candidate.py": digest(source)}},
        )
        inventory: JSONObject = {
            trial_id: {
                "baseline_research.py": digest(
                    self.root / "trials" / trial_id / "workspace/baseline_research.py"
                )
            }
            for trial_id in ORDER
        }
        write(
            self.root / "solver-packet-manifest-B.json", {"packet_inventory": inventory}
        )
        write(
            self.root / "solver-packet-manifest.json",
            {"baseline_remote_training": "/remote/baseline"},
        )
        self.save_origins()

    def save_origins(self) -> None:
        artifacts: JSONObject = {}
        for trial_id in ("baseline", *ORDER):
            remote = "/remote/baseline"
            if trial_id != "baseline":
                submission = read_object(
                    self.root / "trials" / trial_id / "workspace/submission.json"
                )
                candidate = submission["candidate"]
                assert isinstance(candidate, str)
                if candidate != "baseline":
                    remote = f"/remote/trials/{trial_id}/{candidate}/training"
            artifacts[trial_id] = {
                "remote_directory": remote,
                "files": {
                    name: digest(self.root / "selected" / trial_id / name)
                    for name in ("config.json", "model.safetensors")
                },
            }
        write(
            self.root / "selected/origins.json",
            {"schema_version": 1, "artifacts": artifacts},
        )

    def make_trial(self, trial_id: str) -> None:
        trial = self.root / "trials" / trial_id
        workspace = trial / "workspace"
        workspace.mkdir(parents=True)
        arm = "traincheck" if trial_id.endswith("T") else "control"
        write(
            trial / "trial.json",
            {
                "workspace": str(workspace),
                "arm": arm,
                "max_candidates": 2,
                "max_diagnostics": int(arm == "traincheck"),
                "expected_steps": 10000,
                "profile": str(self.profile),
                "profile_sha256": digest(self.profile),
                "remote_trial": f"/remote/trials/{trial_id}",
            },
        )
        (workspace / "baseline_research.py").write_text(self.baseline)
        (workspace / "assignment.txt").write_text(
            "Improve the objective within this trial's budget."
        )
        (workspace / "summary.md").write_text(
            "Selected the baseline after inspecting development results."
        )
        write(workspace / "baseline_development.json", {"final": depths(20.0)})
        write(workspace / "submission.json", {"candidate": "baseline"})
        write(self.root / "selected" / trial_id / "config.json", self.baseline_config)
        (self.root / "selected" / trial_id / "model.safetensors").write_bytes(
            b"baseline model"
        )

    def add_run(
        self,
        trial_id: str,
        *,
        successful: bool,
        diagnose: bool = False,
        crlf: bool = False,
    ) -> Path:
        trial = self.root / "trials" / trial_id
        manifest = read_object(trial / "trial.json")
        run = trial / "runs/candidate-01"
        run.mkdir(parents=True)
        program = "def loss():\n    return 2\n"
        if crlf:
            program = program.replace("\n", "\r\n")
        (run / "program.py").write_bytes(program.encode())
        (run / "config.toml").write_text(configure(self.profile.read_text(), program))
        receipt: JSONObject = {
            "candidate": "candidate-01",
            "arm": manifest["arm"],
            "diagnostics_requested": diagnose,
            "program_sha256": digest(run / "program.py"),
            "profile_sha256": digest(self.profile),
            "trial_manifest_sha256": digest(trial / "trial.json"),
            "candidates_reserved": 1,
            "diagnostics_reserved": int(diagnose),
            "status": "training_complete" if successful else "training_failed",
            "training_exit_code": 0 if successful else 1,
            "errors": (
                []
                if successful
                else [
                    {
                        "phase": "training",
                        "type": "ProcessExit",
                        "message": "Training failed",
                    }
                ]
            ),
            "operational_seconds_including_transfer_and_diagnostics": 8.0,
        }
        if successful:
            rows = depths(40.0)
            receipt.update(
                {
                    "completed_steps": 10000,
                    "training_seconds": 5.0,
                    "development_depths": rows,
                }
            )
            write(
                run / "results.json",
                {
                    "completed_steps": 10000,
                    "training_seconds": 5.0,
                    "device_backend": "cpu",
                    "final": rows,
                },
            )
            write(trial / "workspace/submission.json", {"candidate": "candidate-01"})
            write(
                self.root / "selected" / trial_id / "config.json",
                {
                    "model": {"width": 64},
                    "training": {"steps": 10000, "research_source": program},
                },
            )
            (self.root / "selected" / trial_id / "model.safetensors").write_bytes(
                b"candidate model"
            )
        if diagnose:
            receipt.update(
                {
                    "diagnostic_status": "execution_failed",
                    "diagnostic_exit_code": 1,
                    "errors": [
                        {
                            "phase": "diagnostics",
                            "type": "ProcessExit",
                            "message": "Diagnostic failed",
                        }
                    ],
                }
            )
        write(run / "run.json", receipt)
        write(
            trial / "usage.json",
            {"candidates_reserved": 1, "diagnostics_reserved": int(diagnose)},
        )
        self.save_origins()
        return run

    def scores(self) -> None:
        freeze(self.root)
        begin_scoring(self.root)
        for trial_id in ("baseline", *ORDER):
            raw = depths(30.0, include_eight=False)
            values = metric(raw, [1, 2, 4])
            selected = self.root / "selected" / trial_id
            write(
                self.root / "scores" / (trial_id + ".json"),
                {
                    "kind": "reduced_compute_pilot_not_official_RSI_score",
                    "split": "test",
                    "fixed_policy_depths": [1, 2, 4, 4],
                    "seed": 20261030,
                    "requested_examples_per_bin": 512,
                    "actual_examples": 2048,
                    "checkpoint_sha256": digest(selected / "model.safetensors"),
                    "config_sha256": digest(selected / "config.json"),
                    "panel_sha256": hashlib.sha256(b"shared heldout panel").hexdigest(),
                    "depths": raw,
                    "reward": values["reward"],
                },
            )

    def test_missing_trial_or_submission_fails(self) -> None:
        path = self.root / "trials/B2C/workspace/submission.json"
        path.unlink()
        with self.assertRaises(FileNotFoundError):
            freeze(self.root)

    def test_untrained_selected_candidate_fails(self) -> None:
        write(
            self.root / "trials/B1C/workspace/submission.json",
            {"candidate": "candidate-02"},
        )
        with self.assertRaisesRegex(ValueError, "no validated completed training"):
            freeze(self.root)

    def test_unknown_failed_training_time_is_separate(self) -> None:
        self.add_run("C1", successful=False)
        self.scores()
        result = analyze(self.root)
        setup = result["excluded_setup_trials"]
        self.assertIsInstance(setup, list)
        assert isinstance(setup, list)
        first = mapping(setup[0], "setup")
        self.assertEqual(first["training_seconds_known"], 0.0)
        self.assertEqual(first["training_duration_unknown_attempts"], ["candidate-01"])
        self.assertEqual(
            first["operational_seconds_including_queue_transfer_diagnostics"], 8.0
        )

    def test_diagnostic_failure_does_not_discard_trained_selection(self) -> None:
        self.add_run("B1T", successful=True, diagnose=True, crlf=True)
        self.scores()
        reports = analyze(self.root)["trials"]
        assert isinstance(reports, list)
        treatment = mapping(reports[1], "treatment")
        self.assertEqual(treatment["selection"], "candidate-01")
        self.assertEqual(treatment["diagnostic_execution_failures"], 1)
        self.assertEqual(treatment["diagnostic_skip_counts_unknown"], 1)
        self.assertEqual(treatment["training_seconds_known"], 5.0)

    def test_bad_ledger_is_not_repaired(self) -> None:
        self.add_run("B1C", successful=False)
        write(
            self.root / "trials/B1C/usage.json",
            {"candidates_reserved": True, "diagnostics_reserved": 0},
        )
        with self.assertRaisesRegex(ValueError, "must be an integer"):
            freeze(self.root)

    def test_changed_program_or_manifest_hash_fails(self) -> None:
        run = self.add_run("B1C", successful=True)
        (run / "program.py").write_text("changed source")
        with self.assertRaisesRegex(ValueError, "program_sha256 mismatch"):
            freeze(self.root)

    def test_unfinished_receipt_fails(self) -> None:
        run = self.add_run("B1C", successful=False)
        receipt = read_object(run / "run.json")
        receipt["status"] = "started"
        write(run / "run.json", receipt)
        with self.assertRaisesRegex(ValueError, "has not finished"):
            freeze(self.root)

    def test_missing_operational_time_is_not_zero(self) -> None:
        run = self.add_run("B1C", successful=False)
        receipt = read_object(run / "run.json")
        del receipt["operational_seconds_including_transfer_and_diagnostics"]
        write(run / "run.json", receipt)
        with self.assertRaisesRegex(ValueError, "Missing required field"):
            freeze(self.root)

    def test_selected_checkpoint_mismatch_fails(self) -> None:
        self.scores()
        score_path = self.root / "scores/B1C.json"
        score = read_object(score_path)
        score["checkpoint_sha256"] = "0" * 64
        write(score_path, score)
        with self.assertRaisesRegex(ValueError, "checkpoint_sha256 mismatch"):
            analyze(self.root)

    def test_wrong_downloaded_checkpoint_fails_before_finalization(self) -> None:
        self.add_run("B1C", successful=True)
        (self.root / "selected/B1C/model.safetensors").write_bytes(b"wrong model")
        with self.assertRaisesRegex(ValueError, "remote artifact hash mismatch"):
            freeze(self.root)

    def test_origin_directory_and_required_fields_are_checked(self) -> None:
        self.add_run("B1C", successful=True)
        path = self.root / "selected/origins.json"
        original = path.read_bytes()
        for change in ("wrong_directory", "missing_hash", "missing_trial"):
            with self.subTest(change=change):
                path.write_bytes(original)
                origins = read_object(path)
                artifacts = origins["artifacts"]
                assert isinstance(artifacts, dict)
                candidate = artifacts["B1C"]
                assert isinstance(candidate, dict)
                if change == "wrong_directory":
                    candidate["remote_directory"] = (
                        "/remote/trials/B1C/candidate-02/training"
                    )
                elif change == "missing_hash":
                    hashes = candidate["files"]
                    assert isinstance(hashes, dict)
                    del hashes["config.json"]
                else:
                    del artifacts["B1T"]
                write(path, origins)
                with self.assertRaises(ValueError):
                    freeze(self.root)

    def test_origin_evidence_is_frozen(self) -> None:
        self.scores()
        path = self.root / "selected/origins.json"
        origins = read_object(path)
        origins["host"] = "changed host after finalization"
        write(path, origins)
        with self.assertRaisesRegex(ValueError, "frozen file hash mismatch"):
            analyze(self.root)

    def test_completed_diagnostic_requires_frozen_reference(self) -> None:
        run = self.add_run("B1T", successful=True, diagnose=True)
        invariant_path = self.root / "trials/B1T/workspace/invariants.json"
        invariant_path.write_text("[]\n")
        write(run / "coverage.json", {"schema_version": 3, "per_invariant": []})
        diagnostic: JSONObject = {
            "schema_version": 1,
            "status": "complete",
            "candidate_config_sha256": digest(run / "config.toml"),
            "invariants_sha256": digest(invariant_path),
        }
        write(run / "diagnostic_manifest.json", diagnostic)
        report: JSONObject = {
            "schema_version": 1,
            "outcome": "no_evidence",
            "coverage_incomplete": True,
            "coverage_source": {
                "path": "/remote/coverage.json",
                "sha256": digest(run / "coverage.json"),
            },
            "counts": {
                key: 0
                for key in (
                    "completed",
                    "checker_error",
                    "observer_error",
                    "unsupported_coverage",
                    "observed_active",
                    "observed_active_passes",
                    "observed_dormant",
                    "observed_violations",
                    "unknown_outcomes",
                    "unobserved_coverage",
                    "skipped_variable_checks",
                    "completed_decisions",
                )
            },
        }
        write(run / "agent_report.json", report)
        receipt = read_object(run / "run.json")
        receipt.update(
            diagnostic_status="complete",
            diagnostic_exit_code=0,
            errors=[],
            traincheck_report=report,
        )
        write(run / "run.json", receipt)
        _, _, frozen_files = collect(self.root)
        self.assertIn(invariant_path, frozen_files)
        diagnostic["invariants_sha256"] = "0" * 64
        write(run / "diagnostic_manifest.json", diagnostic)
        with self.assertRaisesRegex(
            ValueError, "diagnostic reference invariants mismatch"
        ):
            collect(self.root)
        del diagnostic["invariants_sha256"]
        write(run / "diagnostic_manifest.json", diagnostic)
        with self.assertRaisesRegex(ValueError, "Missing required field"):
            collect(self.root)

    def test_false_or_fabricated_reward_fails(self) -> None:
        self.scores()
        path = self.root / "scores/B1T.json"
        score = read_object(path)
        for reward in (False, 100.0):
            score["reward"] = reward
            write(path, score)
            with self.assertRaises(ValueError):
                analyze(self.root)

    def test_altered_frozen_assignment_fails(self) -> None:
        self.scores()
        (self.root / "trials/B1T/workspace/assignment.txt").write_text(
            "Changed after selection"
        )
        with self.assertRaisesRegex(ValueError, "frozen file hash mismatch"):
            analyze(self.root)

    def test_scoring_before_finalization_is_rejected(self) -> None:
        self.scores()
        path = self.root / "scores/provenance.json"
        provenance = read_object(path)
        provenance["scoring_started_at"] = "2026-01-01T00:00:00+00:00"
        write(path, provenance)
        with self.assertRaisesRegex(ValueError, "must follow finalization"):
            analyze(self.root)

    def test_duplicate_pair_or_setup_overlap_is_rejected(self) -> None:
        path = self.root / "study_protocol.json"
        record = read_object(path)
        record["excluded_setup_trials"] = ["B1C"]
        write(path, record)
        with self.assertRaisesRegex(ValueError, "overlapping trial IDs"):
            collect(self.root)

    def test_scores_cannot_exist_when_freezing(self) -> None:
        write(self.root / "scores/B1C.json", {})
        with self.assertRaisesRegex(ValueError, "already exist"):
            freeze(self.root)

    def test_preflight_cost_is_separate_and_does_not_create_training_budget(
        self,
    ) -> None:
        trial = self.root / "trials/B1C"
        path = trial / "preflights/preflight-01"
        path.mkdir(parents=True)
        (path / "program.py").write_text(self.baseline)
        (path / "config.toml").write_text(
            configure(self.profile.read_text(), self.baseline)
        )
        backend: JSONObject = {
            "schema_version": 1,
            "valid": False,
            "error": {"type": "ValueError", "message": "Unsupported syntax"},
        }
        write(path / "result.json", backend)
        write(
            path / "preflight.json",
            {
                "schema_version": 1,
                "preflight": "preflight-01",
                "arm": "control",
                "profile_sha256": digest(self.profile),
                "trial_manifest_sha256": digest(trial / "trial.json"),
                "program_sha256": digest(path / "program.py"),
                "status": "program_invalid",
                "errors": [{"phase": "validate_result", "error": backend["error"]}],
                "check_exit_code": 2,
                "valid": False,
                "result": backend,
                "operational_seconds_including_transfer": 1.5,
            },
        )
        self.scores()
        reports = analyze(self.root)["trials"]
        assert isinstance(reports, list)
        report = mapping(reports[0], "trial")
        self.assertEqual(report["attempts"], [])
        self.assertEqual(report["operational_seconds_including_preflights"], 1.5)
        self.assertFalse((trial / "usage.json").exists())
        (path / "program.py").write_text("changed after finalization")
        with self.assertRaisesRegex(ValueError, "program_sha256 mismatch"):
            analyze(self.root)

    def test_incomplete_preflight_is_not_a_success(self) -> None:
        path = self.root / "trials/B1C/preflights/preflight-01"
        write(path / "preflight.json", {"status": "started"})
        with self.assertRaises(ValueError):
            freeze(self.root)

    def test_baseline_selection_must_match_reference_score(self) -> None:
        self.scores()
        path = self.root / "scores/B1C.json"
        score = read_object(path)
        score["depths"] = depths(80.0, include_eight=False)
        score["reward"] = metric(score["depths"], [1, 2, 4])["reward"]
        write(path, score)
        with self.assertRaisesRegex(ValueError, "baseline selection score mismatch"):
            analyze(self.root)

    def test_score_panel_size_must_match(self) -> None:
        self.scores()
        path = self.root / "scores/B1C.json"
        score = read_object(path)
        score["actual_examples"] = 100
        write(path, score)
        with self.assertRaisesRegex(ValueError, "test panel size mismatch"):
            analyze(self.root)

    def test_harness_inventory_mismatch_fails(self) -> None:
        write(
            self.root / "harness-freeze-B.json",
            {"files": {"tools/rsi_pilot/run_candidate.py": "0" * 64}},
        )
        with self.assertRaisesRegex(ValueError, "inventory hash mismatch"):
            freeze(self.root)

    def test_complete_training_without_results_is_not_success(self) -> None:
        run = self.add_run("B1C", successful=True)
        (run / "results.json").unlink()
        with self.assertRaisesRegex(ValueError, "no validated results"):
            freeze(self.root)

    def test_missing_requested_diagnostic_is_not_empty_success(self) -> None:
        run = self.add_run("B1T", successful=True, diagnose=True)
        receipt = read_object(run / "run.json")
        del receipt["diagnostic_status"]
        del receipt["diagnostic_exit_code"]
        write(run / "run.json", receipt)
        with self.assertRaisesRegex(ValueError, "missing its requested diagnostic"):
            freeze(self.root)


if __name__ == "__main__":
    unittest.main()

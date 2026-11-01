"""Private evaluator regressions; never export these into solver packets."""

from __future__ import annotations

import copy
import json
from dataclasses import asdict
from pathlib import Path

import pytest
import torch

from tools.checkpoint_pilot import evaluate as evaluator


def contract(world_size: int = 2) -> evaluator.Contract:
    return evaluator.Contract(world_size, 17, 23, 4, 4, 2, 0.15, False)


def state(rank: int, world_size: int, steps: int) -> dict[str, object]:
    return {
        "model": {
            "layers.0.weight": torch.full((2, 2), 1.0 + steps / 10),
            "layers.1.weight": torch.full((2, 2), 3.0 + steps / 10),
        },
        "optimizer_state": {
            "layers.0.weight": {"momentum_buffer": torch.full((2, 2), 10.0)},
            "layers.1.weight": {"momentum_buffer": torch.full((2, 2), 20.0)},
        },
        "optimizer_groups": [
            {
                "params": ["layers.0.weight", "layers.1.weight"],
                "lr": 0.15,
                "momentum": 0.9,
                "dampening": 0,
                "weight_decay": 0,
                "nesterov": False,
                "maximize": False,
                "foreach": None,
            },
        ],
        "progress": {"completed_steps": steps, "data_cursor": steps, "seed": 17},
        "rng": {
            "cpu": torch.tensor([rank + 1, 3, 7, 9], dtype=torch.uint8),
            "cuda": None,
        },
        "rank": rank,
        "world_size": world_size,
    }


def metrics(rank: int, world_size: int, stage: str) -> dict[str, object]:
    start = 0 if stage == "prefix" else 4
    return {
        "stage": stage,
        "backend": "reference",
        "rank": rank,
        "world_size": world_size,
        "seed": 17,
        "eval_seed": 23,
        "accumulation": 2,
        "lr": 0.15,
        "parameter_count": 8,
        "torch_version": "2.2.2",
        "device": "cpu",
        "examples_per_rank_update": 8,
        "tokens_per_global_update": world_size * 64,
        "requested_updates": 4,
        "completed_steps": start + 4,
        "training_seconds_including_validation": 0.1,
        "steps": [
            {"step": step, "train_token_loss": 1 / (step + 1)}
            for step in range(start + 1, start + 5)
        ],
        "validation_curve": [
            {"step": step, "validation_loss": 1 / (step + 1)}
            for step in (start, start + 4)
        ],
    }


def make_run(
    root: Path, name: str, stage: str, world_size: int = 2
) -> evaluator.RunEvidence:
    source = root / name / "source"
    source.mkdir(parents=True)
    (source / "worker.py").write_text("# trusted fixed workload\n")
    source_hashes = {"worker.py": evaluator.sha256(source / "worker.py")}
    phase = root / name / "phase"
    phase.mkdir()
    (phase / "phase-result.json").write_text(
        json.dumps(
            {
                "exit_code": 0,
                "stop_reason": None,
                "source_unchanged": True,
            }
        )
    )
    (phase / "invocation.json").write_text(json.dumps({"source_sha256": source_hashes}))
    for rank in range(world_size):
        directory = phase / "payload" / f"rank{rank}"
        directory.mkdir(parents=True)
        (directory / "metrics.json").write_text(
            json.dumps(metrics(rank, world_size, stage))
        )
        torch.save(
            state(rank, world_size, 4 if stage == "prefix" else 8),
            directory / "terminal_state.pt",
        )
        if stage == "resume":
            torch.save(state(rank, world_size, 4), directory / "restored_state.pt")
    return evaluator.RunEvidence(phase, source, source_hashes, artifact_hashes(phase))


def artifact_hashes(phase: Path) -> dict[str, str]:
    return {
        str(path.relative_to(phase)): evaluator.sha256(path)
        for path in phase.rglob("*")
        if path.is_file()
    }


def refreshed(run: evaluator.RunEvidence) -> evaluator.RunEvidence:
    return evaluator.RunEvidence(
        run.phase_directory,
        run.source_directory,
        run.source_sha256,
        artifact_hashes(run.phase_directory),
    )


def four_runs(root: Path, world_size: int = 2) -> tuple[
    evaluator.RunEvidence,
    evaluator.RunEvidence,
    evaluator.RunEvidence,
    evaluator.RunEvidence,
]:
    return (
        make_run(root, "reference-prefix", "prefix", world_size),
        make_run(root, "reference-resume", "resume", world_size),
        make_run(root, "candidate-prefix", "prefix", world_size),
        make_run(root, "candidate-resume", "resume", world_size),
    )


@pytest.mark.parametrize("world_size", [1, 2])
def test_complete_restore_and_trajectory_pass(tmp_path: Path, world_size: int) -> None:
    result = evaluator.evaluate_pair(
        *four_runs(tmp_path, world_size), contract(world_size)
    )
    assert result["status"] == "passed"
    assert result["contract_passed"] is True
    assert result["source_and_artifact_manifests_verified"] is True
    assert result["ranks"] is not None
    assert len(result["ranks"]) == world_size
    for row in result["ranks"]:
        assert all(item["passed"] for item in row["restore"])
        tensors = [item for item in row["restore"] if item["kind"] == "tensor"]
        assert len(tensors) == 5
        assert all(
            item["reference_sha256"] == item["candidate_sha256"] for item in tensors
        )


def test_same_shaped_optimizer_permutation_fails_exact_restore(tmp_path: Path) -> None:
    rp, rr, cp, cr = four_runs(tmp_path)
    data = state(0, 2, 4)
    optimizer = evaluator.record(data["optimizer_state"], "optimizer")
    optimizer["layers.0.weight"], optimizer["layers.1.weight"] = (
        optimizer["layers.1.weight"],
        optimizer["layers.0.weight"],
    )
    data["optimizer_state"] = optimizer
    torch.save(data, cr.phase_directory / "payload/rank0/restored_state.pt")
    result = evaluator.evaluate_pair(rp, rr, cp, refreshed(cr), contract())
    assert result["status"] == "violated"
    assert result["prefix_passed"] is True
    assert result["restore_passed"] is False
    assert (
        result["trajectory_passed"] is True
    )  # Deliberately equal fixture final state.
    assert result["ranks"] is not None
    failures = [row for row in result["ranks"][0]["restore"] if not row["passed"]]
    assert len(failures) == 2
    assert all("optimizer_state" in row["path"] for row in failures)


@pytest.mark.parametrize(
    "kind", ["rng", "cursor", "group_order", "group_lr", "reset_momentum"]
)
def test_restore_contract_includes_other_state(tmp_path: Path, kind: str) -> None:
    rp, rr, cp, cr = four_runs(tmp_path)
    data = state(1, 2, 4)
    if kind == "rng":
        data["rng"] = {
            "cpu": torch.tensor([99, 3, 7, 9], dtype=torch.uint8),
            "cuda": None,
        }
    elif kind == "cursor":
        data["progress"] = {"completed_steps": 4, "data_cursor": 0, "seed": 17}
    elif kind in ("group_order", "group_lr"):
        groups = evaluator.array(data["optimizer_groups"], "groups")
        group = evaluator.record(groups[0], "group")
        if kind == "group_order":
            group["params"] = ["layers.1.weight", "layers.0.weight"]
        else:
            group["lr"] = 0.3
        data["optimizer_groups"] = [group]
    else:
        data["optimizer_state"] = {
            name: {"momentum_buffer": torch.zeros((2, 2))}
            for name in ("layers.0.weight", "layers.1.weight")
        }
    torch.save(data, cr.phase_directory / "payload/rank1/restored_state.pt")
    result = evaluator.evaluate_pair(rp, rr, cp, refreshed(cr), contract())
    assert result["status"] == "violated"
    assert result["restore_passed"] is False


def test_named_mapping_reordering_is_benign_but_recorded(tmp_path: Path) -> None:
    rp, rr, cp, cr = four_runs(tmp_path)
    data = state(0, 2, 4)
    for key in ("model", "optimizer_state"):
        mapping = evaluator.record(data[key], key)
        data[key] = dict(reversed(list(mapping.items())))
    torch.save(data, cr.phase_directory / "payload/rank0/restored_state.pt")
    result = evaluator.evaluate_pair(rp, rr, cp, refreshed(cr), contract())
    assert result["status"] == "passed"
    assert result["ranks"] is not None
    keys = [
        row
        for row in result["ranks"][0]["restore"]
        if row["path"] == "restore['optimizer_state'].keys"
    ]
    assert len(keys) == 1
    assert keys[0]["reference"] != keys[0]["candidate"]
    assert keys[0]["passed"]


@pytest.mark.parametrize("difference,expected", [(1e-7, "passed"), (0.01, "violated")])
def test_trajectory_tolerance_separate_from_restore(
    tmp_path: Path, difference: float, expected: str
) -> None:
    rp, rr, cp, cr = four_runs(tmp_path)
    data = state(0, 2, 8)
    model = evaluator.record(data["model"], "model")
    value = model["layers.0.weight"]
    assert isinstance(value, torch.Tensor)
    model["layers.0.weight"] = value + difference
    data["model"] = model
    torch.save(data, cr.phase_directory / "payload/rank0/terminal_state.pt")
    result = evaluator.evaluate_pair(rp, rr, cp, refreshed(cr), contract())
    assert result["status"] == expected
    assert result["restore_passed"] is True


def test_tiny_restore_difference_does_not_use_trajectory_tolerance(
    tmp_path: Path,
) -> None:
    rp, rr, cp, cr = four_runs(tmp_path)
    data = state(0, 2, 4)
    model = evaluator.record(data["model"], "model")
    value = model["layers.0.weight"]
    assert isinstance(value, torch.Tensor)
    model["layers.0.weight"] = value + 1e-7
    data["model"] = model
    torch.save(data, cr.phase_directory / "payload/rank0/restored_state.pt")
    result = evaluator.evaluate_pair(rp, rr, cp, refreshed(cr), contract())
    assert result["status"] == "violated"
    assert result["restore_passed"] is False


@pytest.mark.parametrize(
    "kind",
    [
        "missing_field",
        "bad_rank",
        "bad_steps",
        "bool_rank",
        "missing_optimizer",
        "bad_rng",
        "nan",
    ],
)
def test_malformed_state_is_unknown_not_pass(tmp_path: Path, kind: str) -> None:
    rp, rr, cp, cr = four_runs(tmp_path)
    data = state(0, 2, 4)
    if kind == "missing_field":
        del data["progress"]
    elif kind == "bad_rank":
        data["rank"] = 1
    elif kind == "bool_rank":
        data["rank"] = False
    elif kind == "bad_steps":
        data["progress"] = {"completed_steps": 8, "data_cursor": 8, "seed": 17}
    elif kind == "missing_optimizer":
        data["optimizer_state"] = {}
    elif kind == "bad_rng":
        data["rng"] = {"cpu": None, "cuda": None}
    else:
        data["model"] = {"layers.0.weight": torch.full((2, 2), float("nan"))}
    torch.save(data, cr.phase_directory / "payload/rank0/restored_state.pt")
    result = evaluator.evaluate_pair(rp, rr, cp, refreshed(cr), contract())
    assert result["status"] == "error"
    assert result["contract_passed"] is None
    assert result["restore_passed"] is None
    assert result["ranks"] is None
    assert result["error"] is not None


@pytest.mark.parametrize(
    "kind",
    [
        "failed",
        "timeout",
        "source_changed",
        "missing_rank",
        "hash_changed",
        "unlisted_source",
        "invocation_source",
        "missing_metrics",
        "extra_rank",
    ],
)
def test_incomplete_or_changed_evidence_is_unknown(tmp_path: Path, kind: str) -> None:
    rp, rr, cp, cr = four_runs(tmp_path)
    phase = cr.phase_directory
    if kind in ("failed", "timeout", "source_changed"):
        (phase / "phase-result.json").write_text(
            json.dumps(
                {
                    "exit_code": 1 if kind == "failed" else 0,
                    "stop_reason": "timeout" if kind == "timeout" else None,
                    "source_unchanged": kind != "source_changed",
                }
            )
        )
        cr = refreshed(cr)
    elif kind == "missing_rank":
        (phase / "payload/rank1/restored_state.pt").unlink()
        cr = refreshed(cr)
    elif kind == "hash_changed":
        (phase / "payload/rank0/metrics.json").write_text("{}")
    elif kind == "unlisted_source":
        (cr.source_directory / "new_helper.py").write_text("pass\n")
    elif kind == "invocation_source":
        (phase / "invocation.json").write_text(
            json.dumps({"source_sha256": {"worker.py": "a" * 64}})
        )
        cr = refreshed(cr)
    elif kind == "extra_rank":
        (phase / "payload/rank2").mkdir()
    else:
        (phase / "payload/rank0/metrics.json").write_text("{}")
        cr = refreshed(cr)
    result = evaluator.evaluate_pair(rp, rr, cp, cr, contract())
    assert result["status"] == "error"
    assert result["contract_passed"] is None


def test_reference_restore_mismatch_invalidates_grading(tmp_path: Path) -> None:
    rp, rr, cp, cr = four_runs(tmp_path)
    data = state(0, 2, 4)
    data["progress"] = {"completed_steps": 4, "data_cursor": 0, "seed": 17}
    torch.save(data, rr.phase_directory / "payload/rank0/restored_state.pt")
    result = evaluator.evaluate_pair(rp, refreshed(rr), cp, cr, contract())
    assert result["status"] == "error"
    assert result["error"] is not None
    assert "Reference restoration" in result["error"]["message"]


def test_curve_change_cannot_pass_with_matching_terminal_state(tmp_path: Path) -> None:
    rp, rr, cp, cr = four_runs(tmp_path)
    path = cr.phase_directory / "payload/rank0/metrics.json"
    data = metrics(0, 2, "resume")
    data["steps"] = [{"step": step, "train_token_loss": 10.0} for step in range(5, 9)]
    path.write_text(json.dumps(data))
    result = evaluator.evaluate_pair(rp, rr, cp, refreshed(cr), contract())
    assert result["status"] == "violated"
    assert result["restore_passed"] is True
    assert result["trajectory_passed"] is False


def test_manifest_roundtrip(tmp_path: Path) -> None:
    runs = four_runs(tmp_path)
    data: dict[str, object] = {"schema_version": 1, "contract": asdict(contract())}
    for name, run in zip(
        (
            "reference_prefix",
            "reference_resume",
            "candidate_prefix",
            "candidate_resume",
        ),
        runs,
    ):
        data[name] = {
            "phase_directory": str(run.phase_directory),
            "source_directory": str(run.source_directory),
            "source_sha256": run.source_sha256,
            "artifact_sha256": run.artifact_sha256,
        }
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps(data))
    result = evaluator.run_from_manifest(manifest)
    assert result["status"] == "passed"
    malformed = copy.deepcopy(data)
    malformed["contract"] = {"world_size": 2}
    manifest.write_text(json.dumps(malformed))
    with pytest.raises(ValueError, match="missing required field"):
        evaluator.run_from_manifest(manifest)


def test_exact_tensor_comparison_preserves_signed_zero() -> None:
    left = torch.tensor([0.0])
    right = torch.tensor([-0.0])
    rows = evaluator.compare_states(left, right, "state", exact=True, atol=1, rtol=1)
    assert rows[0]["passed"] is False
    assert rows[0]["reference_sha256"] != rows[0]["candidate_sha256"]


def test_real_single_rank_roundtrip_matches_reference(tmp_path: Path) -> None:
    from tools.checkpoint_pilot.task.engine import Options, run

    runs: list[evaluator.RunEvidence] = []
    for backend in ("reference", "current"):
        checkpoint = tmp_path / f"{backend}-checkpoint"
        for stage in ("prefix", "resume"):
            evidence = make_run(tmp_path, f"actual-{backend}-{stage}", stage, 1)
            # The helper installs protocol receipts; use a fresh payload for
            # actual engine outputs so no synthetic state can satisfy this test.
            payload = evidence.phase_directory / "actual-payload"
            options = Options(
                stage="prefix" if stage == "prefix" else "resume",
                backend="reference" if backend == "reference" else "current",
                checkpoint=checkpoint,
                output=payload,
                steps=1,
                seed=17,
                eval_seed=23,
                lr=0.15,
                accumulation=2,
                device="cpu",
            )
            run(options)
            for generated in (payload / "rank0").iterdir():
                generated.replace(
                    evidence.phase_directory / "payload/rank0" / generated.name
                )
            runs.append(refreshed(evidence))
    actual_contract = evaluator.Contract(1, 17, 23, 1, 1, 2, 0.15, False)
    result = evaluator.evaluate_pair(
        runs[0], runs[1], runs[2], runs[3], actual_contract
    )
    assert result["status"] == "passed", result["error"]


@pytest.mark.parametrize(
    "manifest", [{"../outside.py": "a" * 64}, {"worker.py": "bad"}, {}]
)
def test_manifest_rejects_paths_and_hashes(manifest: dict[str, str]) -> None:
    with pytest.raises(ValueError):
        evaluator.validate_manifest(manifest, "manifest")

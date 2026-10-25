"""Verify the paired study's actual outputs; do not label dormant rules as passes."""

import argparse
import hashlib
import json
import math
from collections import Counter
from pathlib import Path


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def compare(root):
    def read(name):
        return json.loads((root / name).read_text())

    phases = {
        p.parent.name: json.loads(p.read_text())
        for p in root.glob("*/phase-result.json")
    }
    assert phases and all(
        p["exit_code"] == 0 and p["stop_reason"] is None and p["source_unchanged"]
        for p in phases.values()
    ), phases
    metrics = {
        role: read(role + "/metrics.json")
        for role in ("reference", "control", "holdout", "baseline-holdout")
    }
    adaptations = {role: read(role + "/adaptation.json") for role in metrics}
    assert all(m["success"] and m["state_shapes_unchanged"] for m in metrics.values())
    workload_keys = (
        "benchmark_source_sha256",
        "extracted_definitions_sha256",
        "adapter_sha256",
        "data",
        "prepared_data_steps",
        "model_config",
        "batch_size",
        "device",
        "cpu_threads",
        "precision",
        "deterministic_algorithms",
        "torch_version",
        "python_version",
    )
    for role in metrics:
        for key in workload_keys:
            assert adaptations[role][key] == adaptations["reference"][key], (role, key)
    invocations = [read(name + "/invocation.json") for name in phases]
    assert all(
        i["source_sha256"] == invocations[0]["source_sha256"] for i in invocations
    )
    assert all(
        i["harness_sha256"] == invocations[0]["harness_sha256"] for i in invocations
    )

    assert len({adaptations[r]["process_id"] for r in metrics}) == 4
    for role in metrics:
        assert metrics[role]["completed_steps"] == (
            10 if role in ("reference", "control") else 20
        )
        if role != "baseline-holdout":
            assert adaptations[role]["tracer_module"] == str(
                Path(adaptations[role]["source_root"])
                / "traincheck/instrumentor/tracer.py"
            )
        if adaptations[role]["data"]["kind"].startswith("synthetic"):
            raise AssertionError("Synthetic data is outside this study protocol")

    def losses(metric):
        rows = [
            {k: v for k, v in step.items() if k.startswith("loss")}
            for step in metric["steps"]
        ]
        assert all(math.isfinite(v) for row in rows for v in row.values())
        return rows

    paired = {}
    for name, left, right in (
        ("same_seed_fresh_control", "reference", "control"),
        ("holdout_instrumentation_comparison", "holdout", "baseline-holdout"),
    ):
        m1, m2 = metrics[left], metrics[right]
        checks = {
            "loss_trajectory_equal": losses(m1) == losses(m2),
            "initial_state_sha256_equal": m1["initial_model_state_sha256"]
            == m2["initial_model_state_sha256"],
            "final_state_sha256_equal": m1["final_model_state_sha256"]
            == m2["final_model_state_sha256"],
            "shapes_equal": m1["state_shapes"] == m2["state_shapes"],
        }
        assert all(checks.values()), (name, checks)
        paired[name] = checks

    coverage = {
        role: read("check-" + role + "/coverage.json")
        for role in ("reference", "control", "holdout", "holdout-direct", "fault")
    }
    assert all(c["inputs_unchanged"] for c in coverage.values())
    frozen_hash = hashlib.sha256(
        (root / "inference/invariants.jsonl").read_bytes()
    ).hexdigest()
    for checked in coverage.values():
        assert (
            checked["input_sha256"][str(root / "inference/invariants.jsonl")]
            == frozen_hash
        )
    observed = coverage["holdout"]["per_invariant"]
    direct = coverage["holdout-direct"]["per_invariant"]
    assert len(observed) == len(direct)
    assert all(
        canonical(a["checker_result"]) == canonical(b["checker_result"])
        for a, b in zip(observed, direct)
    )
    assert all(item["active"] is None and item["dormant"] is None for item in direct)

    mutation = read("fault/mutation.json")
    selected = mutation["invariant_index"]
    healthy = observed[selected]
    faulty = coverage["fault"]["per_invariant"][selected]
    assert healthy["check_passed"] and healthy["active"]
    assert faulty["status"] == "completed" and not faulty["check_passed"]
    assert faulty["decisions_completed"] > 0 and faulty["precondition_group_passes"] > 0
    assert (
        mutation["ids_unchanged"]
        and mutation["precondition_inputs_unchanged"]
        and len(mutation["edits"]) == 1
    )
    assert (
        coverage["fault"]["raw_process_ids"] == coverage["holdout"]["raw_process_ids"]
    )
    assert coverage["fault"]["raw_thread_ids"] == coverage["holdout"]["raw_thread_ids"]
    families = {}
    for role in ("reference", "control", "holdout", "fault"):
        family_counts = {}
        for item in coverage[role]["per_invariant"]:
            assert item["status"] == "completed", item
            counts = family_counts.setdefault(item["relation"], Counter())
            counts.update(
                invariants=1,
                active=int(item["active"]),
                dormant=int(item["dormant"]),
                violations=int(not item["check_passed"]),
                opportunities=item["opportunities"],
                precondition_group_passes=item["precondition_group_passes"],
                actual_decisions=item["decisions_completed"],
                proxy_checks_skipped=int(item["variable_check_skipped"]),
                fully_validated_active=int(item["fully_validated_active"]),
            )
        families[role] = {name: dict(counts) for name, counts in family_counts.items()}
    output = {
        "status": "study_completed",
        "invariants_sha256": frozen_hash,
        "training_comparisons": paired,
        "benchmark_data_runtime_source_and_harness_matched": True,
        "observer_matches_direct_checker": True,
        "per_run_summary": {
            r: coverage[r]["summary"]
            for r in ("reference", "control", "holdout", "fault")
        },
        "per_relation": families,
        "raw_ids": {
            r: {
                "process": coverage[r]["raw_process_ids"],
                "thread": coverage[r]["raw_thread_ids"],
            }
            for r in ("reference", "control", "holdout")
        },
        "fault": {
            "kind": mutation["kind"],
            "invariant_index": selected,
            "healthy_decisions": healthy["decisions_completed"],
            "fault_decisions": faulty["decisions_completed"],
            "fault_precondition_group_passes": faulty["precondition_group_passes"],
            "violation": True,
            "edit": mutation["edits"][0],
        },
        "limitations": [
            "One reference seed and one holdout seed per workload; no generalization estimate.",
            "The first half of holdout inputs overlaps the reference inputs.",
            "Offline checking; no online checker, distributed training, or runtime overhead claim.",
            "Fault control is post-hoc trace sensitivity, not a reproduced training bug.",
            "Source-exact checker stops at the first violation per invariant.",
        ],
    }
    (root / "comparison.json").write_text(json.dumps(output, indent=2) + "\n")
    return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    print(json.dumps(compare(parser.parse_args().root.resolve()), indent=2))

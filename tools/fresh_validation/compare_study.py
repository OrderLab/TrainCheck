"""Verify actual paired-study outputs; missing observation never becomes a pass."""

import argparse
import hashlib
import json
import math
import sys
from collections import Counter
from collections.abc import Mapping
from pathlib import Path
from typing import TypedDict

if __package__ in (None, ""):
    sys.path.append(str(Path(__file__).resolve().parents[2]))
from tools.fresh_validation.contracts import (  # noqa: E402
    JSONObject,
    boolean,
    integer,
    json_object,
    json_value,
    mapping,
    number,
    read_object,
    required,
    sequence,
    string,
)


class TrainingComparison(TypedDict):
    loss_trajectory_equal: bool
    initial_state_sha256_equal: bool
    final_state_sha256_equal: bool
    shapes_equal: bool


class StudyComparison(TypedDict):
    status: str
    invariants_sha256: str
    training_comparisons: dict[str, TrainingComparison]
    benchmark_data_runtime_source_and_harness_matched: bool
    observer_matches_direct_checker: bool
    per_run_summary: dict[str, JSONObject]
    per_relation: dict[str, dict[str, dict[str, int]]]
    raw_ids: dict[str, JSONObject]
    fault: JSONObject
    limitations: list[str]


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def canonical(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def object_field(data: Mapping[str, object], key: str) -> Mapping[str, object]:
    return mapping(required(data, key), key)


def objects_field(data: Mapping[str, object], key: str) -> list[Mapping[str, object]]:
    return [mapping(item, key) for item in sequence(required(data, key), key)]


def bool_field(data: Mapping[str, object], key: str) -> bool:
    return boolean(required(data, key), key)


def int_field(data: Mapping[str, object], key: str) -> int:
    return integer(required(data, key), key)


def str_field(data: Mapping[str, object], key: str) -> str:
    return string(required(data, key), key)


def losses(metric: Mapping[str, object]) -> list[dict[str, int | float]]:
    rows = [
        {
            key: number(value, key)
            for key, value in step.items()
            if key.startswith("loss")
        }
        for step in objects_field(metric, "steps")
    ]
    require(
        bool(rows) and all(row for row in rows),
        "Every training step must contain a loss",
    )
    require(
        all(math.isfinite(value) for row in rows for value in row.values()),
        "Non-finite loss",
    )
    require(
        int_field(metric, "completed_steps") == len(rows),
        "Loss row count does not match completed_steps",
    )
    return rows


def compare(root: Path) -> StudyComparison:
    def read(name: str) -> JSONObject:
        return read_object(root / name)

    phases = {p.parent.name: read_object(p) for p in root.glob("*/phase-result.json")}
    expected_phases = {
        "reference",
        "control",
        "holdout",
        "baseline-holdout",
        "inference",
        "check-reference",
        "check-control",
        "check-holdout",
        "check-holdout-direct",
        "prepare-fault",
        "check-fault",
    }
    require(
        expected_phases <= phases.keys(),
        f"Missing completed phases: {sorted(expected_phases - phases.keys())}",
    )
    for name, phase in phases.items():
        require(
            int_field(phase, "exit_code") == 0
            and required(phase, "stop_reason") is None
            and bool_field(phase, "source_unchanged"),
            f"Phase {name} did not complete with unchanged source",
        )
    roles = ("reference", "control", "holdout", "baseline-holdout")
    metrics = {role: read(role + "/metrics.json") for role in roles}
    adaptations = {role: read(role + "/adaptation.json") for role in roles}
    require(
        all(
            bool_field(m, "success") and bool_field(m, "state_shapes_unchanged")
            for m in metrics.values()
        ),
        "Training failed or state shapes changed",
    )
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
    for role in roles:
        for key in workload_keys:
            require(
                required(adaptations[role], key)
                == required(adaptations["reference"], key),
                f"Mismatched {role} adaptation field: {key}",
            )
    invocations = [read(name + "/invocation.json") for name in phases]
    for key in ("source_sha256", "harness_sha256"):
        require(
            all(
                object_field(i, key) == object_field(invocations[0], key)
                for i in invocations
            ),
            f"Mismatched phase {key}",
        )
    require(
        len({int_field(adaptations[role], "process_id") for role in roles}) == 4,
        "Training phases must use distinct processes",
    )
    for role in roles:
        require(
            int_field(metrics[role], "completed_steps")
            == (10 if role in ("reference", "control") else 20),
            f"Incorrect completed step count for {role}",
        )
        if role != "baseline-holdout":
            require(
                str_field(adaptations[role], "tracer_module")
                == str(
                    Path(str_field(adaptations[role], "source_root"))
                    / "traincheck/instrumentor/tracer.py"
                ),
                f"Wrong tracer source for {role}",
            )
        require(
            not str_field(object_field(adaptations[role], "data"), "kind").startswith(
                "synthetic"
            ),
            "Synthetic data is outside this study protocol",
        )

    paired: dict[str, TrainingComparison] = {}
    for name, left, right in (
        ("same_seed_fresh_control", "reference", "control"),
        ("holdout_instrumentation_comparison", "holdout", "baseline-holdout"),
    ):
        m1, m2 = metrics[left], metrics[right]
        checks: TrainingComparison = {
            "loss_trajectory_equal": losses(m1) == losses(m2),
            "initial_state_sha256_equal": str_field(m1, "initial_model_state_sha256")
            == str_field(m2, "initial_model_state_sha256"),
            "final_state_sha256_equal": str_field(m1, "final_model_state_sha256")
            == str_field(m2, "final_model_state_sha256"),
            "shapes_equal": object_field(m1, "state_shapes")
            == object_field(m2, "state_shapes"),
        }
        require(all(checks.values()), f"Training comparison failed: {name}: {checks}")
        paired[name] = checks
    coverage = {
        role: read("check-" + role + "/coverage.json")
        for role in ("reference", "control", "holdout", "holdout-direct", "fault")
    }
    require(
        all(bool_field(c, "inputs_unchanged") for c in coverage.values()),
        "Coverage inputs changed",
    )
    frozen_path = root / "inference/invariants.jsonl"
    frozen_hash = hashlib.sha256(frozen_path.read_bytes()).hexdigest()
    for checked in coverage.values():
        require(
            str_field(object_field(checked, "input_sha256"), str(frozen_path))
            == frozen_hash,
            "Frozen invariants changed",
        )
    observed = objects_field(coverage["holdout"], "per_invariant")
    direct = objects_field(coverage["holdout-direct"], "per_invariant")
    require(len(observed) == len(direct), "Observer/direct result counts differ")
    for a, b in zip(observed, direct):
        require(
            str_field(a, "status") == str_field(b, "status") == "completed",
            "Checker did not complete",
        )
        require(
            canonical(object_field(a, "checker_result"))
            == canonical(object_field(b, "checker_result")),
            "Observer changed checker result",
        )
        require(
            required(b, "active") is None
            and required(b, "dormant") is None
            and not bool_field(b, "coverage_observed"),
            "Disabled observation must remain unknown",
        )

    mutation = read("fault/mutation.json")
    selected = int_field(mutation, "invariant_index")
    fault_results = objects_field(coverage["fault"], "per_invariant")
    require(
        0 <= selected < len(observed) and selected < len(fault_results),
        "Fault invariant index is out of range",
    )
    healthy, faulty = observed[selected], fault_results[selected]
    require(
        bool_field(healthy, "check_passed") and bool_field(healthy, "active"),
        "Fault target was not healthy and active",
    )
    require(
        str_field(faulty, "status") == "completed"
        and not bool_field(faulty, "check_passed"),
        "Fault did not cause a completed violation",
    )
    require(
        int_field(faulty, "decisions_completed") > 0
        and int_field(faulty, "precondition_group_passes") > 0,
        "Fault target did not activate",
    )
    fault_identity = str_field(mutation, "func_call_id")
    # This selected rule is ConsistentOutputRelation; both records require a
    # single call identity. A pair record or missing ID is a malformed report.
    require(
        any(
            str_field(d, "func_call_id") == fault_identity
            for d in objects_field(faulty, "decision_records")
        ),
        "Fault decision identity not observed",
    )
    require(
        any(
            str_field(event, "func_call_id") == fault_identity
            for event in objects_field(object_field(faulty, "checker_result"), "trace")
        ),
        "Fault identity absent from checker violation",
    )
    edits = objects_field(mutation, "edits")
    require(
        bool_field(mutation, "ids_unchanged")
        and bool_field(mutation, "precondition_inputs_unchanged")
        and len(edits) == 1,
        "Fault changed more than one permitted field",
    )
    for key in ("raw_process_ids", "raw_thread_ids"):
        require(
            required(coverage["fault"], key) == required(coverage["holdout"], key),
            f"Fault altered {key}",
        )
    families: dict[str, dict[str, dict[str, int]]] = {}
    for role in ("reference", "control", "holdout", "fault"):
        family_counts: dict[str, Counter[str]] = {}
        for item in objects_field(coverage[role], "per_invariant"):
            require(
                str_field(item, "status") == "completed"
                and bool_field(item, "coverage_observed"),
                "Incomplete or unobserved coverage",
            )
            counts = family_counts.setdefault(str_field(item, "relation"), Counter())
            counts.update(
                invariants=1,
                active=int(bool_field(item, "active")),
                dormant=int(bool_field(item, "dormant")),
                violations=int(not bool_field(item, "check_passed")),
                opportunities=int_field(item, "opportunities"),
                precondition_group_passes=int_field(item, "precondition_group_passes"),
                actual_decisions=int_field(item, "decisions_completed"),
                proxy_checks_skipped=int(bool_field(item, "variable_check_skipped")),
                fully_validated_active=int(bool_field(item, "fully_validated_active")),
            )
        families[role] = {name: dict(counts) for name, counts in family_counts.items()}
    output: StudyComparison = {
        "status": "study_completed",
        "invariants_sha256": frozen_hash,
        "training_comparisons": paired,
        "benchmark_data_runtime_source_and_harness_matched": True,
        "observer_matches_direct_checker": True,
        "per_run_summary": {
            r: json_object(required(coverage[r], "summary"))
            for r in ("reference", "control", "holdout", "fault")
        },
        "per_relation": families,
        "raw_ids": {
            r: {
                "process": json_value(required(coverage[r], "raw_process_ids")),
                "thread": json_value(required(coverage[r], "raw_thread_ids")),
            }
            for r in ("reference", "control", "holdout")
        },
        "fault": {
            "kind": str_field(mutation, "kind"),
            "invariant_index": selected,
            "healthy_decisions": int_field(healthy, "decisions_completed"),
            "fault_decisions": int_field(faulty, "decisions_completed"),
            "fault_precondition_group_passes": int_field(
                faulty, "precondition_group_passes"
            ),
            "violation": True,
            "edit": json_object(edits[0]),
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

"""Post-hoc, one-field trace sensitivity control for a live frozen output invariant.

This does not inject a training bug. It changes one recorded tensor shape or dtype after
healthy training, preserves IDs and precondition inputs, and checks the same rules.
"""

import argparse
import copy
import hashlib
import json
from pathlib import Path


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def differences(before, after, path=()):
    if type(before) is not type(after):
        return [path]
    if isinstance(before, dict):
        if before.keys() != after.keys():
            return [path]
        return [p for k in before for p in differences(before[k], after[k], (*path, k))]
    if isinstance(before, list):
        if len(before) != len(after):
            return [path]
        return [
            p
            for i, (a, b) in enumerate(zip(before, after))
            for p in differences(a, b, (*path, i))
        ]
    if before == after or (
        isinstance(before, float) and before != before and after != after
    ):
        return []
    return [path]


def replacement(attributes, attribute):
    """Return one valid same-type unequal metadata value, or no eligible edit."""
    if not isinstance(attributes, dict):
        return None
    value = attributes.get(attribute)
    if attribute == "shape":
        if (
            isinstance(value, list)
            and value
            and isinstance(value[0], int)
            and not isinstance(value[0], bool)
        ):
            return [value[0] + 1, *value[1:]]
    elif attribute == "dtype" and value in ("torch.float32", "torch.float64"):
        return "torch.float64" if value == "torch.float32" else "torch.float32"
    return None


def mutate(traces, invariants, healthy, output):
    rules = [json.loads(line) for line in invariants.read_text().splitlines()]
    coverage = json.loads(healthy.read_text())
    assert coverage["inputs_unchanged"] and coverage["observer_enabled"]
    assert coverage["input_sha256"][str(invariants)] == digest(invariants)
    chosen = None
    call_id = None
    post_events = {}
    for trace in traces:
        assert coverage["input_sha256"][str(trace)] == digest(trace)
        for line in trace.read_bytes().splitlines():
            event = json.loads(line)
            if event.get("type") == "function_call (post)":
                post_events.setdefault(event["func_call_id"], []).append(event)
    # Prefer shape; dtype is a prospective fallback when no live mutable shape
    # rule exists. Selection remains post hoc and does not change learned rules.
    for attribute in ("shape", "dtype"):
        for item in coverage["per_invariant"]:
            rule = rules[item["invariant_index"]]
            if not (
                rule["relation"] == "ConsistentOutputRelation"
                and rule["params"][1]["attr_name"] == attribute
                and item["active"]
                and item["check_passed"]
                and item["decisions_completed"] > 0
            ):
                continue
            for decision in item["decision_records"]:
                events = post_events.get(decision["func_call_id"], [])
                if len(events) != 1:
                    continue
                values = events[0]["return_values"]
                tensors = values if isinstance(values, list) else [values]
                eligible = any(
                    "Tensor" in name and replacement(attributes, attribute) is not None
                    for tensor in tensors
                    if isinstance(tensor, dict)
                    for name, attributes in tensor.items()
                )
                if eligible:
                    chosen, call_id = item, decision["func_call_id"]
                    break
            if chosen is not None:
                break
        if chosen is not None:
            break
    if chosen is None:
        raise ValueError("No healthy, actually evaluated mutable shape or dtype rule")
    rule = rules[chosen["invariant_index"]]
    assert call_id is not None
    output.mkdir(parents=True, exist_ok=False)
    (output / "traces").mkdir()
    edits, fingerprints = [], {}
    for trace in traces:
        assert coverage["input_sha256"][str(trace)] == digest(trace)
        raw_lines = trace.read_bytes().splitlines(keepends=True)
        transformed = list(raw_lines)
        for index, raw in enumerate(raw_lines):
            event = json.loads(raw)
            if (
                event.get("func_call_id") != call_id
                or event.get("type") != "function_call (post)"
            ):
                continue
            before = copy.deepcopy(event)
            values = event["return_values"]
            tensor_values = values if isinstance(values, list) else [values]
            for value in tensor_values:
                if not isinstance(value, dict):
                    continue
                for typename, attributes in value.items():
                    if "Tensor" not in typename:
                        continue
                    new_value = replacement(attributes, attribute)
                    if new_value is None:
                        continue
                    old = copy.deepcopy(attributes[attribute])
                    attributes[attribute] = new_value
                    diff = differences(before, event)
                    assert len(diff) == 1
                    assert diff[0][:1] == ("return_values",)
                    assert (
                        before["process_id"] == event["process_id"]
                        and before["thread_id"] == event["thread_id"]
                    )
                    edits.append(
                        {
                            "file": trace.name,
                            "zero_based_line": index,
                            "path": list(diff[0]),
                            "attribute": attribute,
                            "before_value": old,
                            "after_value": attributes[attribute],
                            "raw_process_id": event["process_id"],
                            "raw_thread_id": event["thread_id"],
                        }
                    )
                    transformed[index] = (
                        json.dumps(event, separators=(",", ":")).encode() + b"\n"
                    )
                    break
                if edits:
                    break
        target = output / "traces" / trace.name
        target.write_bytes(b"".join(transformed))
        assert all(
            a == b
            for i, (a, b) in enumerate(zip(raw_lines, transformed))
            if not any(
                edit["file"] == trace.name and edit["zero_based_line"] == i
                for edit in edits
            )
        )
        fingerprints[trace.name] = {
            "original_sha256": digest(trace),
            "fault_sha256": digest(target),
        }
    assert len(edits) == 1, edits
    manifest = {
        "kind": "post_hoc_trace_field_sensitivity_control_not_training_fault",
        "selection": "first actually evaluated healthy mutable shape rule, otherwise a float32/float64 dtype rule",
        "attribute": attribute,
        "invariant_index": chosen["invariant_index"],
        "invariant": rule,
        "func_call_id": call_id,
        "edits": edits,
        "traces": fingerprints,
        "invariants_sha256": digest(invariants),
        "healthy_coverage_sha256": digest(healthy),
        "precondition_inputs_unchanged": True,
        "ids_unchanged": True,
        "invariants_not_retrained_or_rewritten": True,
    }
    (output / "mutation.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--traces", nargs="+", type=Path, required=True)
    parser.add_argument("--invariants", type=Path, required=True)
    parser.add_argument("--healthy-coverage", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(
        json.dumps(
            mutate(args.traces, args.invariants, args.healthy_coverage, args.output)
        )
    )

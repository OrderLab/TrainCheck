"""Post-hoc, one-field trace sensitivity control for a live frozen output invariant.

This does not inject a training bug. It changes one recorded tensor shape or dtype after
healthy training, preserves IDs and precondition inputs, and checks the same rules.
"""

import argparse
import copy
import hashlib
import json
import sys
from pathlib import Path
from typing import TypedDict

if __package__ in (None, ""):
    sys.path.append(str(Path(__file__).resolve().parents[2]))
from tools.fresh_validation.contracts import (  # noqa: E402
    JSONObject,
    JSONValue,
    boolean,
    integer,
    json_object,
    mapping,
    read_object,
    required,
    sequence,
    string,
)


class Edit(TypedDict):
    file: str
    zero_based_line: int
    path: list[str | int]
    attribute: str
    before_value: JSONValue
    after_value: JSONValue
    raw_process_id: JSONValue
    raw_thread_id: JSONValue


class Fingerprints(TypedDict):
    original_sha256: str
    fault_sha256: str


class MutationReport(TypedDict):
    kind: str
    selection: str
    attribute: str
    invariant_index: int
    invariant: JSONObject
    func_call_id: str
    edits: list[Edit]
    traces: dict[str, Fingerprints]
    invariants_sha256: str
    healthy_coverage_sha256: str
    precondition_inputs_unchanged: bool
    ids_unchanged: bool
    invariants_not_retrained_or_rewritten: bool


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def differences(
    before: object, after: object, path: tuple[str | int, ...] = ()
) -> list[tuple[str | int, ...]]:
    if type(before) is not type(after):
        return [path]
    if isinstance(before, dict):
        before_dict = mapping(before, "before")
        after_dict = mapping(after, "after")
        if before_dict.keys() != after_dict.keys():
            return [path]
        return [
            p
            for k in before_dict
            for p in differences(before_dict[k], after_dict[k], (*path, k))
        ]
    if isinstance(before, list):
        assert isinstance(after, list)
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


def replacement(attributes: JSONValue, attribute: str) -> JSONValue:
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


def mutate(
    traces: list[Path], invariants: Path, healthy: Path, output: Path
) -> MutationReport:
    rules = [
        json_object(json.loads(line), "invariant")
        for line in invariants.read_text().splitlines()
    ]
    coverage = read_object(healthy)
    if not boolean(
        required(coverage, "inputs_unchanged"), "inputs_unchanged"
    ) or not boolean(required(coverage, "observer_enabled"), "observer_enabled"):
        raise ValueError(
            "Fault selection requires unchanged inputs and enabled observation"
        )
    input_hashes = mapping(required(coverage, "input_sha256"), "input_sha256")

    def verify_hash(path: Path) -> None:
        if string(required(input_hashes, str(path)), str(path)) != digest(path):
            raise ValueError(f"Input changed since coverage collection: {path}")

    verify_hash(invariants)
    chosen: JSONObject | None = None
    call_id: str | None = None
    selected_index: int | None = None
    post_events: dict[str, list[JSONObject]] = {}
    for trace in traces:
        verify_hash(trace)
        for line in trace.read_bytes().splitlines():
            event = json_object(json.loads(line), "trace event")
            if string(required(event, "type"), "event.type") == "function_call (post)":
                identity = string(required(event, "func_call_id"), "event.func_call_id")
                post_events.setdefault(identity, []).append(event)
    # Prefer shape; dtype is a prospective fallback when no live mutable shape
    # rule exists. Selection remains post hoc and does not change learned rules.
    for attribute in ("shape", "dtype"):
        for raw_item in sequence(required(coverage, "per_invariant"), "per_invariant"):
            item = json_object(raw_item, "per_invariant item")
            index = integer(required(item, "invariant_index"), "invariant_index")
            if not 0 <= index < len(rules):
                raise ValueError(f"Invariant index out of range: {index}")
            rule = rules[index]
            if (
                string(required(rule, "relation"), "relation")
                != "ConsistentOutputRelation"
            ):
                continue
            params = sequence(required(rule, "params"), "params")
            if len(params) != 2:
                raise ValueError("ConsistentOutputRelation requires two parameters")
            output_param = mapping(params[1], "output parameter")
            if not (
                string(required(output_param, "attr_name"), "attr_name") == attribute
                and boolean(required(item, "active"), "active")
                and boolean(required(item, "check_passed"), "check_passed")
                and integer(
                    required(item, "decisions_completed"), "decisions_completed"
                )
                > 0
            ):
                continue
            for raw_decision in sequence(
                required(item, "decision_records"), "decision_records"
            ):
                decision = mapping(raw_decision, "decision")
                identity = string(
                    required(decision, "func_call_id"), "decision.func_call_id"
                )
                events = post_events.get(identity, [])
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
                    chosen, call_id, selected_index = item, identity, index
                    break
            if chosen is not None:
                break
        if chosen is not None:
            break
    if chosen is None:
        raise ValueError("No healthy, actually evaluated mutable shape or dtype rule")
    assert call_id is not None and selected_index is not None
    rule = rules[selected_index]
    output.mkdir(parents=True, exist_ok=False)
    (output / "traces").mkdir()
    edits: list[Edit] = []
    fingerprints: dict[str, Fingerprints] = {}
    for trace in traces:
        verify_hash(trace)
        raw_lines = trace.read_bytes().splitlines(keepends=True)
        transformed = list(raw_lines)
        for index, raw in enumerate(raw_lines):
            event = json_object(json.loads(raw), "trace event")
            if (
                string(required(event, "type"), "event.type") != "function_call (post)"
                or string(required(event, "func_call_id"), "event.func_call_id")
                != call_id
            ):
                continue
            before = copy.deepcopy(event)
            values = event["return_values"]
            tensor_values = values if isinstance(values, list) else [values]
            for value in tensor_values:
                if not isinstance(value, dict):
                    continue
                for typename, attributes in value.items():
                    if "Tensor" not in typename or not isinstance(attributes, dict):
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
    manifest: MutationReport = {
        "kind": "post_hoc_trace_field_sensitivity_control_not_training_fault",
        "selection": "first actually evaluated healthy mutable shape rule, otherwise a float32/float64 dtype rule",
        "attribute": attribute,
        "invariant_index": selected_index,
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

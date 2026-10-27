"""Unprofiled full tensor metadata conversion, with a separate getter count.

Run this same script with PYTHONPATH pointing at each frozen source revision.
This is a component benchmark; it does not measure training or GPU overhead.
"""

import argparse
import hashlib
import json
import platform
import time
from pathlib import Path
from unittest.mock import patch

import torch

from traincheck.instrumentor import dumper


def run(iterations: int) -> dict[str, object]:
    if iterations < 1:
        raise ValueError("iterations must be positive")
    torch.set_num_threads(1)
    tensors = [torch.arange(256, dtype=torch.float32).reshape(16, 16)]
    tensors.append(torch.nn.Parameter(tensors[0].clone()))
    before = [value.detach().clone() for value in tensors]
    rng_before = torch.get_rng_state().clone()
    for value in tensors:
        dumper.convert_var_to_dict(value, include_tensor_data=False)

    fields = 0
    start = time.perf_counter()
    for _ in range(iterations):
        for value in tensors:
            fields += len(dumper.convert_var_to_dict(value, include_tensor_data=False))
    seconds = time.perf_counter() - start

    # Diagnostics are outside the ordinary timing and do not mutate source.
    getter_calls = 0
    original_getter = dumper.safe_getattr

    def count_getter(obj: object, name: str) -> object:
        nonlocal getter_calls
        getter_calls += 1
        return original_getter(obj, name)

    with patch.object(dumper, "safe_getattr", count_getter):
        for value in tensors:
            dumper.convert_var_to_dict(value, include_tensor_data=False)

    if not torch.equal(rng_before, torch.get_rng_state()):
        raise AssertionError("Metadata conversion changed the torch RNG")
    for value, saved in zip(tensors, before):
        if not torch.equal(value, saved):
            raise AssertionError("Metadata conversion changed tensor values")
    source = Path(dumper.__file__).resolve()
    return {
        "kind": "component_not_training",
        "python": platform.python_version(),
        "torch": torch.__version__,
        "platform": platform.platform(),
        "device": "cpu",
        "torch_threads": torch.get_num_threads(),
        "iterations": iterations,
        "conversions": iterations * len(tensors),
        "seconds": seconds,
        "recorded_fields": fields,
        "separate_diagnostic_getter_calls": getter_calls,
        "diagnostic_conversions": len(tensors),
        "rng_and_tensor_values_unchanged": True,
        "dumper_path": str(source),
        "dumper_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "limitations": [
            "Recorded field counts alone do not establish metadata equivalence.",
            "No claim about end-to-end training or GPU speedup.",
            "No isolated memory cost measurement.",
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--iterations", type=int, default=1000)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.write_text(json.dumps(run(args.iterations), indent=2) + "\n")


if __name__ == "__main__":
    main()

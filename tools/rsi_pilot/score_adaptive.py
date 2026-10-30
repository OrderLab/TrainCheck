"""Operator-only scoring for the reduced-compute adaptive-reasoning pilot.

Uses the public upstream generator, model and FLOP accounting. It does not
produce an official RSI result: CPU training has no trusted-service receipt and
uses the public partition instead of the official service's private partition.
"""

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path
from typing import Literal

import torch
from safetensors.torch import load_file

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from tools.fresh_validation.contracts import (  # noqa: E402
    JSONObject,
    integer,
    json_list,
    json_object,
    mapping,
    number,
    read_object,
    required,
    sequence,
)


def score(
    run: Path,
    output: Path,
    seed: int,
    per_bin: int,
    split: Literal["validation", "test"] = "test",
) -> JSONObject:
    # These public upstream modules are supplied by the pinned workspace on
    # PYTHONPATH. Their accepted output is validated before calculating scores.
    from latent_reasoning.data import evaluation_set
    from latent_reasoning.evaluate import evaluate
    from latent_reasoning.model import LoopedTransformer, ModelConfig
    from latent_reasoning.reward import aggregate_reward

    config = read_object(run / "config.json")
    model_config = mapping(required(config, "model"), "model")
    expected = {
        "modulus": 7,
        "width": 64,
        "heads": 4,
        "layers": 1,
        "max_loops": 8,
        "max_length": 32,
    }
    if set(model_config) != set(expected) or any(
        integer(required(model_config, key), key) != value
        for key, value in expected.items()
    ):
        raise ValueError("Checkpoint must use the unchanged upstream backbone")
    torch.set_num_threads(4)
    model = LoopedTransformer(ModelConfig())
    model.load_state_dict(load_file(run / "model.safetensors"), strict=True)
    model.to(torch.device("cpu"))
    # Test panels are generated only after solver finalization. Validation mode
    # lets the operator check scoring parity before freezing the experiment.
    examples = evaluation_set(split, [1, 2, 3, 4, 5], per_bin, 7, ("+", "-"), seed)
    rows_value: object = evaluate(model, examples, [1, 2, 4], torch.device("cpu"))
    rows = [
        json_object(row, "depth result")
        for row in sequence(rows_value, "depth results")
    ]
    if [integer(required(row, "loops"), "loops") for row in rows] != [1, 2, 4]:
        raise ValueError("Expected exactly the frozen fixed-depth policy evaluations")
    points = [rows[0], rows[1], rows[2], rows[2]]
    # This JSON round-trip validates that the retained raw evidence is serializable
    # with no NaN/Infinity before using the upstream's numeric range validation.
    json.dumps(rows, allow_nan=False)
    reward_value: object = aggregate_reward(
        [
            {
                "accuracy_percent": number(
                    required(row, "macro_accuracy_percent"), "accuracy"
                ),
                "mean_dense_matmul_flops": number(
                    required(row, "macro_mean_dense_matmul_flops"), "work"
                ),
            }
            for row in points
        ]
    )
    reward = number(reward_value, "reward")
    if not math.isfinite(reward):
        raise ValueError("Reward must be finite")
    panel = [
        {
            "tokens": list(example.tokens),
            "label": example.label,
            "operations": example.operations,
        }
        for example in examples
    ]
    panel_bytes = json.dumps(panel, sort_keys=True, separators=(",", ":")).encode()
    report: JSONObject = {
        "kind": "reduced_compute_pilot_not_official_RSI_score",
        "split": split,
        "seed": seed,
        "requested_examples_per_bin": per_bin,
        "actual_examples": len(examples),
        "fixed_policy_depths": [1, 2, 4, 4],
        "reward": reward,
        "depths": json_list(rows, "depths"),
        "panel_sha256": hashlib.sha256(panel_bytes).hexdigest(),
        "checkpoint_sha256": hashlib.sha256(
            (run / "model.safetensors").read_bytes()
        ).hexdigest(),
        "config_sha256": hashlib.sha256((run / "config.json").read_bytes()).hexdigest(),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        raise ValueError("Scoring output already exists")
    output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--seed", required=True, type=int)
    parser.add_argument("--per-bin", default=512, type=int)
    parser.add_argument("--split", choices=("validation", "test"), default="test")
    args = parser.parse_args()
    if args.per_bin < 1:
        parser.error("per-bin must be positive")
    print(
        json.dumps(
            score(args.run, args.output, args.seed, args.per_bin, args.split), indent=2
        )
    )


if __name__ == "__main__":
    main()

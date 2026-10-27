"""Behavioral contracts for the TorchTitan checking-overhead adapter boundary."""

from __future__ import annotations

import random
import tempfile
import unittest
from dataclasses import dataclass
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch

from tools.checking_overhead import workload
from traincheck.config import config
from traincheck.instrumentor.caches import META_VARS
from traincheck.instrumentor.proxy_wrapper import subclass
from traincheck.instrumentor.proxy_wrapper.proxy_registry import VarRegistry
from traincheck.invariant.base_cls import Invariant, VarTypeParam
from traincheck.invariant.consistency_relation import ConsistencyRelation


@dataclass
class Optimizers:
    optimizers: list[torch.optim.Optimizer]


class Metrics:
    def __init__(self) -> None:
        self.calls: list[int] = []

    def log(
        self,
        step: int,
        average: float,
        maximum: float,
        gradient_norm: float,
        *,
        extra_metrics: dict[str, int],
    ) -> None:
        self.calls.append(step)


class Trainer:
    def __init__(self, steps: int = 3) -> None:
        self.model_parts = [torch.nn.Linear(2, 1)]
        self.optimizers = Optimizers(
            [torch.optim.SGD(self.model_parts[0].parameters(), lr=0.01)]
        )
        self.metrics_processor = Metrics()
        self.step = 0
        self.steps = steps
        self.closed = False
        self.failure: Exception | None = None
        self.disabled_at_step: list[bool] = []

    def train_step(self, data: torch.Tensor) -> None:
        self.disabled_at_step.append(config.DISABLE_WRAPPER)
        if self.failure is not None:
            raise self.failure
        optimizer = self.optimizers.optimizers[0]
        optimizer.zero_grad()
        loss = self.model_parts[0](data).square().mean()
        loss.backward()
        norm = torch.nn.utils.clip_grad_norm_(self.model_parts[0].parameters(), 1.0)
        optimizer.step()
        value = float(loss.item())
        self.metrics_processor.log(
            self.step,
            value,
            value,
            float(norm.item()),
            extra_metrics={"n_tokens_seen": self.step},
        )

    def train(self) -> None:
        while self.step < self.steps:
            self.step += 1
            self.train_step(torch.randn(3, 2))

    def close(self) -> None:
        self.closed = True


class CustomParameter(torch.nn.Parameter):
    """The production subclass tracker cannot preserve this type's identity."""


class WorkloadTests(unittest.TestCase):
    def setUp(self) -> None:
        self.registry = VarRegistry()
        for target, name, value in (
            (config, "DISABLE_WRAPPER", True),
            (config, "INSTRUMENTATION_POLICY", {"interval": 1, "warm_up": 0}),
            (subclass, "get_global_registry", lambda: self.registry),
        ):
            patcher = patch.object(target, name, value)
            patcher.start()
            self.addCleanup(patcher.stop)
        meta_patch = patch.dict(META_VARS, {"step": 0}, clear=True)
        meta_patch.start()
        self.addCleanup(meta_patch.stop)

    def seeded_trainer(self) -> Trainer:
        torch.manual_seed(42)
        random.seed(42)
        np.random.seed(42)
        return Trainer()

    def test_subclass_preserves_state_rng_and_existing_optimizer_references(
        self,
    ) -> None:
        trainer = self.seeded_trainer()
        handle = workload.TrainerHandle.from_trainer(trainer)
        parameters = tuple(handle.model.parameters())
        initial = workload.model_evidence(handle.model)
        rng = workload.rng_evidence()
        workload.attach_model_tracker(handle, "subclass")
        self.assertEqual(initial, workload.model_evidence(handle.model))
        self.assertEqual(rng, workload.rng_evidence())
        self.assertTrue(
            all(a is b for a, b in zip(parameters, handle.model.parameters()))
        )
        self.assertTrue(
            all(
                a is b
                for a, b in zip(
                    parameters, workload.optimizer_parameters(handle.optimizers)
                )
            )
        )

    def test_tracked_training_matches_plain_bytes_losses_and_rng(self) -> None:
        plain = self.seeded_trainer()
        plain_handle = workload.TrainerHandle.from_trainer(plain)
        _, plain_losses = workload.run_training(plain_handle, 3, False, lambda: None)
        expected = workload.model_evidence(plain_handle.model)
        expected_rng = workload.rng_evidence()
        tracked = self.seeded_trainer()
        tracked_handle = workload.TrainerHandle.from_trainer(tracked)
        workload.attach_model_tracker(tracked_handle, "subclass")
        _, tracked_losses = workload.run_training(
            tracked_handle, 3, False, lambda: None
        )
        self.assertEqual(expected, workload.model_evidence(tracked_handle.model))
        self.assertEqual(expected_rng, workload.rng_evidence())
        self.assertEqual(plain_losses, tracked_losses)

    def test_step_boundary_once_and_sync_outside_tracing(self) -> None:
        trainer = self.seeded_trainer()
        handle = workload.TrainerHandle.from_trainer(trainer)
        sync_states: list[bool] = []

        def synchronize() -> None:
            sync_states.append(config.DISABLE_WRAPPER)

        timings, losses = workload.run_training(handle, 3, True, synchronize)
        self.assertEqual(META_VARS["step"], 3)
        self.assertEqual([row["step"] for row in timings], [1, 2, 3])
        self.assertEqual([row["step"] for row in losses], [1, 2, 3])
        self.assertEqual(trainer.disabled_at_step, [False, False, False])
        self.assertEqual(sync_states, [True] * 6)
        self.assertTrue(config.DISABLE_WRAPPER)
        self.assertEqual(trainer.train_step, handle.train_step)
        self.assertEqual(trainer.metrics_processor.log, handle.metrics_log)

    def test_failure_propagates_and_restores_boundaries(self) -> None:
        trainer = self.seeded_trainer()
        failure = RuntimeError("native step failure")
        trainer.failure = failure
        handle = workload.TrainerHandle.from_trainer(trainer)
        with self.assertRaises(RuntimeError) as raised:
            workload.run_training(handle, 3, True, lambda: None)
        self.assertIs(raised.exception, failure)
        self.assertTrue(config.DISABLE_WRAPPER)
        self.assertEqual(trainer.train_step, handle.train_step)
        self.assertEqual(trainer.metrics_processor.log, handle.metrics_log)

    def test_custom_parameter_rejected_before_mutation(self) -> None:
        trainer = self.seeded_trainer()
        model = trainer.model_parts[0]
        model.weight = CustomParameter(model.weight.detach())
        trainer.optimizers = Optimizers([torch.optim.SGD(model.parameters(), lr=0.1)])
        before = tuple(model.parameters())
        with self.assertRaisesRegex(TypeError, "exact nn.Parameter"):
            workload.attach_model_tracker(
                workload.TrainerHandle.from_trainer(trainer), "subclass"
            )
        self.assertTrue(all(a is b for a, b in zip(before, model.parameters())))

    def test_malformed_optimizer_boundary_is_rejected(self) -> None:
        trainer = self.seeded_trainer()
        trainer.optimizers = Optimizers([])
        with self.assertRaisesRegex(TypeError, "nonempty list"):
            workload.TrainerHandle.from_trainer(trainer)

    def test_optimizer_outside_model_is_rejected(self) -> None:
        trainer = self.seeded_trainer()
        trainer.optimizers = Optimizers(
            [torch.optim.SGD([torch.nn.Parameter(torch.ones(1))], lr=0.1)]
        )
        with self.assertRaisesRegex(ValueError, "outside"):
            workload.attach_model_tracker(
                workload.TrainerHandle.from_trainer(trainer), "subclass"
            )

    def test_exact_scalar_empty_and_dtype_evidence(self) -> None:
        scalar = workload.tensor_evidence(torch.tensor(2.0))
        vector = workload.tensor_evidence(torch.tensor([2.0]))
        self.assertEqual(scalar["sha256"], vector["sha256"])
        self.assertNotEqual(scalar["shape"], vector["shape"])
        self.assertEqual(workload.tensor_evidence(torch.empty(0))["shape"], [0])
        self.assertNotEqual(
            scalar, workload.tensor_evidence(torch.tensor(2.0, dtype=torch.float64))
        )

    def test_options_reject_missing_or_extraneous_invariants(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with self.assertRaisesRegex(ValueError, "required exactly"):
                workload.Options("selective", root, root)
            with self.assertRaisesRegex(ValueError, "required exactly"):
                workload.Options(
                    "plain", root, root, invariants=root / "invariants.json"
                )
            with self.assertRaisesRegex(ValueError, "positive integer"):
                workload.Options("plain", root, root, steps=0)
            with self.assertRaisesRegex(ValueError, "positive integer"):
                workload.Options("plain", root, root, steps=True)

    def test_selective_consistency_rules_preserve_sampler_policy(self) -> None:
        invariant = Invariant(
            ConsistencyRelation,
            [
                VarTypeParam("torch.nn.Parameter", "data"),
                VarTypeParam("torch.nn.Parameter", "grad"),
            ],
            None,
        )
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "traces").mkdir()
            options = workload.Options(
                "selective", root, root, invariants=root / "invariants.json"
            )
            with (
                patch(
                    "traincheck.invariant.base_cls.read_inv_file",
                    return_value=[invariant],
                ),
                patch("traincheck.instrumentor.tracer.Instrumentor") as instrumentor,
                patch.object(workload, "install_standard_observers") as observers,
                patch.object(config, "INSTR_OPTS", None),
                patch.object(config, "INSTR_OPTS_FILE", config.INSTR_OPTS_FILE),
                patch.object(config, "MODEL_TRACKER_STYLE", None),
            ):
                self.assertEqual(workload.install_instrumentation(options), "sampler")
                self.assertEqual(config.MODEL_TRACKER_STYLE, "sampler")
                self.assertIsNotNone(config.INSTR_OPTS)
                instrumentor.return_value.instrument.assert_called_once()
                observers.assert_not_called()
                self.assertIn(
                    '"model_tracker_style": "sampler"',
                    (root / "traces" / "instr_opts.json").read_text(),
                )

    def test_selective_rejects_empty_invariants_before_instrumentation(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            options = workload.Options(
                "selective", root, root, invariants=root / "invariants.json"
            )
            with (
                patch("traincheck.invariant.base_cls.read_inv_file", return_value=[]),
                patch("traincheck.instrumentor.tracer.Instrumentor") as instrumentor,
                patch.object(config, "INSTR_OPTS", None),
                patch.object(config, "INSTR_OPTS_FILE", config.INSTR_OPTS_FILE),
            ):
                with self.assertRaisesRegex(ValueError, "nonempty invariants"):
                    workload.install_instrumentation(options)
                instrumentor.assert_not_called()

    def test_cli_resolves_explicit_inputs(self) -> None:
        options = workload.parse_options(
            [
                "--mode",
                "selective",
                "--output-dir",
                "result",
                "--torchtitan-root",
                "titan",
                "--invariants",
                "invariants.json",
            ]
        )
        self.assertEqual(options.mode, "selective")
        self.assertTrue(options.output_dir.is_absolute())
        self.assertEqual(options.steps, 10)
        self.assertIsNotNone(options.invariants)


if __name__ == "__main__":
    unittest.main()

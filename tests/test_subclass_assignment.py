"""Subclass tracking must preserve PyTorch module assignment contracts."""

import gc
import unittest
import weakref
from typing import Callable, TypeAlias, cast
from unittest.mock import patch

import torch
from torch import nn
from torch.nn.modules.module import (
    register_module_module_registration_hook,
    register_module_parameter_registration_hook,
)
from torch.utils.hooks import RemovableHandle

import traincheck.config.config as config
from traincheck.instrumentor.proxy_wrapper import subclass
from traincheck.instrumentor.proxy_wrapper.proxy_registry import VarRegistry
from traincheck.instrumentor.proxy_wrapper.subclass import (
    ProxyParameter,
    proxy_parameter,
)

ParameterRegistrationHook: TypeAlias = Callable[
    [nn.Module, str, nn.Parameter | None], nn.Parameter | None
]
ModuleRegistrationHook: TypeAlias = Callable[
    [nn.Module, str, nn.Module | None], nn.Module | None
]
# PyTorch 2.2 declares these callbacks as returning None, but both API docstrings
# explicitly allow replacement values and the native setters use those values.
# These narrow boundary casts preserve that documented callback/handle contract;
# the tests below verify the replacement is registered and tracked exactly once.
register_parameter_hook = cast(
    Callable[[ParameterRegistrationHook], RemovableHandle],
    register_module_parameter_registration_hook,
)
register_module_hook = cast(
    Callable[[ModuleRegistrationHook], RemovableHandle],
    register_module_module_registration_hook,
)


class MaskedTransformer(nn.Module):
    """A small training workload that caches a causal mask during forward."""

    def __init__(self) -> None:
        super().__init__()
        self.encoder = nn.TransformerEncoderLayer(
            d_model=4, nhead=2, dim_feedforward=8, dropout=0.0, batch_first=True
        )
        self.src_mask: torch.Tensor | None = None

    def forward(self, source: torch.Tensor) -> torch.Tensor:
        self.src_mask = nn.Transformer.generate_square_subsequent_mask(source.size(1))
        return self.encoder(source, src_mask=self.src_mask)


class SubclassAssignmentTests(unittest.TestCase):
    def setUp(self) -> None:
        # Keep tests independent of the process-wide registry and trace files.
        self.registry = VarRegistry()
        registry_patch = patch.object(
            subclass, "get_global_registry", return_value=self.registry
        )
        registry_patch.start()
        self.addCleanup(registry_patch.stop)
        dumping_patch = patch.object(config, "DISABLE_WRAPPER", True)
        dumping_patch.start()
        self.addCleanup(dumping_patch.stop)

    def track(self, module: nn.Module) -> None:
        proxy_parameter(module, parent_name="model", should_dump_trace=False)

    def test_transformer_mask_forward_backward_and_update_match(self) -> None:
        torch.manual_seed(71)
        plain = MaskedTransformer()
        tracked = MaskedTransformer()
        tracked.load_state_dict(plain.state_dict())
        rng_before = torch.get_rng_state().clone()
        self.track(tracked)
        torch.testing.assert_close(torch.get_rng_state(), rng_before, rtol=0, atol=0)
        source = torch.randn(2, 3, 4)
        optimizers = [
            torch.optim.SGD(model.parameters(), lr=0.1) for model in (plain, tracked)
        ]
        for model, optimizer in zip((plain, tracked), optimizers):
            model(source).square().mean().backward()
            optimizer.step()
        self.assertIsInstance(tracked.src_mask, torch.Tensor)
        for name, expected in plain.state_dict().items():
            torch.testing.assert_close(
                tracked.state_dict()[name], expected, rtol=0, atol=0
            )
        for actual, expected in zip(tracked.parameters(), plain.parameters()):
            torch.testing.assert_close(actual.grad, expected.grad, rtol=0, atol=0)

    def test_plain_tensor_and_unrelated_attributes_are_unchanged(self) -> None:
        module = nn.Module()
        self.track(module)
        tensor = torch.ones(2)
        module.mask = tensor
        # Module.__setattr__ in PyTorch 2.2 omits ordinary attributes and None
        # from its annotation. setattr exercises the same runtime setter without
        # misrepresenting those values as Tensor | Module.
        setattr(module, "counter", 3)
        setattr(module, "label", None)
        self.assertIs(module.mask, tensor)
        self.assertEqual(module.counter, 3)
        self.assertIsNone(module.label)
        self.assertEqual(list(module.named_parameters()), [])
        self.assertEqual(list(module.named_buffers()), [])

    def test_buffer_replacement_preserves_identity_and_persistence(self) -> None:
        module = nn.Module()
        module.register_buffer("persistent", torch.zeros(2))
        module.register_buffer("temporary", torch.zeros(2), persistent=False)
        self.track(module)
        replacement = torch.ones(2)
        module.persistent = replacement
        module.temporary = replacement
        self.assertIs(module.persistent, replacement)
        self.assertIs(module.temporary, replacement)
        self.assertEqual(list(module.state_dict()), ["persistent"])
        setattr(module, "persistent", None)
        self.assertIsNone(module.persistent)
        self.assertEqual(list(module.state_dict()), [])

    def test_new_and_replaced_parameters_are_tracked(self) -> None:
        module = nn.Module()
        self.track(module)
        module.weight = nn.Parameter(torch.ones(2))
        first = module.weight
        self.assertIsInstance(first, ProxyParameter)
        replacement = nn.Parameter(torch.zeros(2), requires_grad=False)
        gradient = torch.full((2,), 7.0)
        replacement.grad = gradient
        marker = object()
        setattr(replacement, "user_marker", marker)
        module.weight = replacement
        self.assertIsInstance(module.weight, ProxyParameter)
        self.assertIsNot(module.weight, first)
        self.assertFalse(module.weight.requires_grad)
        self.assertIs(module.weight.grad, gradient)
        self.assertIs(module.weight.__dict__["user_marker"], marker)
        self.assertIs(self.registry.registry["model.weight"].var, module.weight)
        setattr(module, "weight", None)
        self.assertIsNone(module.weight)
        self.assertEqual(list(module.parameters()), [])

    def test_parameter_backward_hook_survives_assignment(self) -> None:
        calls: list[torch.Tensor] = []

        def double_gradient(gradient: torch.Tensor) -> torch.Tensor:
            calls.append(gradient.clone())
            return gradient * 2

        parameter = nn.Parameter(torch.ones(2))
        handle = parameter.register_hook(double_gradient)
        self.addCleanup(handle.remove)
        module = nn.Module()
        self.track(module)
        module.weight = parameter
        module.weight.sum().backward()
        self.assertEqual(len(calls), 1)
        torch.testing.assert_close(module.weight.grad, torch.full((2,), 2.0))
        handle.remove()
        module.weight.grad = None
        module.weight.sum().backward()
        self.assertEqual(len(calls), 1)
        torch.testing.assert_close(module.weight.grad, torch.ones(2))

    def test_post_accumulation_hook_survives_assignment(self) -> None:
        if not hasattr(torch.Tensor, "register_post_accumulate_grad_hook"):
            self.skipTest("This PyTorch version has no post-accumulation hook API")
        calls: list[torch.Tensor] = []

        def observe(parameter: torch.Tensor) -> None:
            calls.append(parameter)

        parameter = nn.Parameter(torch.ones(2))
        handle = parameter.register_post_accumulate_grad_hook(observe)
        self.addCleanup(handle.remove)
        module = nn.Module()
        self.track(module)
        module.weight = parameter
        module.weight.sum().backward()
        self.assertEqual(len(calls), 1)
        self.assertIs(calls[0], module.weight)
        handle.remove()
        module.weight.grad = None
        module.weight.sum().backward()
        self.assertEqual(len(calls), 1)

    def test_assigned_child_and_replacement_parameters_are_tracked(self) -> None:
        module = nn.Module()
        self.track(module)
        for child in (nn.Sequential(nn.Linear(2, 2)), nn.Sequential(nn.Linear(2, 1))):
            module.child = child
            self.assertIs(module.child, child)
            self.assertTrue(
                all(isinstance(p, ProxyParameter) for p in child.parameters())
            )
            child.mask = torch.ones(2)
            child[0].weight = nn.Parameter(torch.zeros_like(child[0].weight))
            self.assertIsInstance(child[0].weight, ProxyParameter)
        setattr(module, "child", None)
        self.assertIsNone(module.child)
        self.assertEqual(list(module.children()), [])

    def test_parameter_aliases_survive_initial_and_later_tracking(self) -> None:
        parameter = nn.Parameter(torch.ones(2))
        module = nn.Module()
        module.weight = parameter
        module.alias = parameter
        child = nn.Module()
        child.weight = parameter
        module.child = child
        self.track(module)
        self.assertIs(module.weight, module.alias)
        self.assertIs(module.weight, child.weight)
        module.later_alias = parameter
        self.assertIs(module.weight, module.later_alias)
        metadata_name: object = module.weight.__dict__["var_name"]
        self.assertIsInstance(metadata_name, str)
        module.proxy_alias = module.weight
        self.track(module)
        self.assertIs(module.weight, module.proxy_alias)
        self.assertEqual(module.weight.__dict__["var_name"], metadata_name)
        self.assertEqual(len(list(module.parameters())), 1)

    def test_alias_cache_does_not_keep_replaced_source_alive(self) -> None:
        module = nn.Module()
        self.track(module)
        parameter = nn.Parameter(torch.ones(2))
        reference = weakref.ref(parameter)
        module.weight = parameter
        del parameter
        gc.collect()
        self.assertIsNone(reference())
        module.weight = nn.Parameter(torch.zeros(2))
        torch.testing.assert_close(module.weight, torch.zeros(2))

    def test_replaced_proxy_is_not_retained_by_alias_cache(self) -> None:
        module = nn.Module()
        self.track(module)
        source = nn.Parameter(torch.ones(2))
        module.weight = source
        proxy_reference = weakref.ref(module.weight)
        module.weight = nn.Parameter(torch.zeros(2))
        gc.collect()
        self.assertIsNone(proxy_reference())
        module.weight = source
        self.assertIsInstance(module.weight, ProxyParameter)
        torch.testing.assert_close(module.weight, torch.ones(2))

    def test_assignment_trace_contains_each_new_parameter_once(self) -> None:
        module = nn.Module()
        source = nn.Parameter(torch.ones(2))
        module.weight = source
        with (
            patch.object(config, "DISABLE_WRAPPER", False),
            patch.object(subclass, "dump_trace_VAR") as dump,
        ):
            proxy_parameter(module, parent_name="model")
            self.assertEqual(dump.call_count, 1)
            self.assertEqual(dump.call_args.args[0]["var_name"], "model.weight")
            module.mask = torch.zeros(2)
            module.alias = source
            self.assertEqual(dump.call_count, 1)
            module.weight = nn.Parameter(torch.zeros(2))
            self.assertEqual(dump.call_count, 2)
            self.assertEqual(dump.call_args.args[0]["var_name"], "model.weight")
            with self.assertRaises(TypeError):
                module.weight = torch.ones(2)
            self.assertEqual(dump.call_count, 2)

    def test_rejected_assignment_preserves_native_error_and_state(self) -> None:
        for name, original, invalid in (
            ("weight", nn.Parameter(torch.ones(2)), torch.zeros(2)),
            ("child", nn.Linear(2, 2), torch.zeros(2)),
            ("weight", nn.Parameter(torch.ones(2)), nn.Linear(2, 2)),
        ):
            with self.subTest(name=name, invalid_type=type(invalid)):
                plain = nn.Module()
                tracked = nn.Module()
                setattr(plain, name, original)
                setattr(tracked, name, original)
                self.track(tracked)
                before = getattr(tracked, name)
                with self.assertRaises(TypeError) as expected:
                    setattr(plain, name, invalid)
                with self.assertRaises(TypeError) as actual:
                    setattr(tracked, name, invalid)
                self.assertEqual(str(actual.exception), str(expected.exception))
                self.assertIs(getattr(tracked, name), before)
                if isinstance(invalid, nn.Module):
                    self.assertNotIn(subclass.SUBCLASS_HOOK_KEY, invalid.__dict__)
                    self.assertFalse(
                        any(isinstance(p, ProxyParameter) for p in invalid.parameters())
                    )

    def test_rejected_buffer_assignment_keeps_native_error_and_state(self) -> None:
        module = nn.Module()
        buffer = torch.ones(2)
        module.register_buffer("mask", buffer)
        self.track(module)
        with self.assertRaisesRegex(TypeError, "as buffer 'mask'"):
            setattr(module, "mask", 42)
        self.assertIs(module.mask, buffer)

    def test_parameter_registration_hook_replacement_is_tracked_once(self) -> None:
        module = nn.Module()
        self.track(module)
        incoming = nn.Parameter(torch.zeros(2))
        replacement = nn.Parameter(torch.ones(2), requires_grad=False)
        calls: list[nn.Parameter | None] = []

        def replace(
            owner: nn.Module, name: str, value: nn.Parameter | None
        ) -> nn.Parameter:
            self.assertIs(owner, module)
            self.assertEqual(name, "weight")
            calls.append(value)
            return replacement

        handle = register_parameter_hook(replace)
        try:
            module.weight = incoming
        finally:
            handle.remove()
        self.assertEqual(len(calls), 1)
        self.assertIs(calls[0], incoming)
        self.assertIsInstance(module.weight, ProxyParameter)
        self.assertFalse(module.weight.requires_grad)
        torch.testing.assert_close(module.weight, replacement)

    def test_module_registration_hook_replacement_is_tracked_once(self) -> None:
        module = nn.Module()
        self.track(module)
        incoming = nn.Linear(2, 2)
        replacement = nn.Linear(2, 1)
        calls: list[nn.Module | None] = []

        def replace(owner: nn.Module, name: str, value: nn.Module | None) -> nn.Module:
            self.assertIs(owner, module)
            self.assertEqual(name, "child")
            calls.append(value)
            return replacement

        handle = register_module_hook(replace)
        try:
            module.child = incoming
        finally:
            handle.remove()
        self.assertEqual(len(calls), 1)
        self.assertIs(calls[0], incoming)
        self.assertIs(module.child, replacement)
        self.assertTrue(
            all(isinstance(p, ProxyParameter) for p in replacement.parameters())
        )
        self.assertFalse(
            any(isinstance(p, ProxyParameter) for p in incoming.parameters())
        )

    def test_registration_hook_failure_does_not_instrument_rejected_child(self) -> None:
        module = nn.Module()
        self.track(module)
        child = nn.Linear(2, 2)
        error = RuntimeError("registration failed")

        def reject(owner: nn.Module, name: str, value: nn.Module | None) -> None:
            raise error

        handle = register_module_hook(reject)
        try:
            with self.assertRaises(RuntimeError) as caught:
                module.child = child
        finally:
            handle.remove()
        self.assertIs(caught.exception, error)
        self.assertNotIn("child", module._modules)
        self.assertNotIn(subclass.SUBCLASS_HOOK_KEY, child.__dict__)
        self.assertFalse(any(isinstance(p, ProxyParameter) for p in child.parameters()))

    def test_malformed_present_tracking_hook_is_rejected_before_assignment(
        self,
    ) -> None:
        for malformed in (None, 3, "hook"):
            with self.subTest(hook=malformed):
                module = nn.Module()
                module.__dict__[subclass.SUBCLASS_HOOK_KEY] = malformed
                with self.assertRaisesRegex(TypeError, "tracking hook"):
                    setattr(module, "label", "new")
                self.assertNotIn("label", module.__dict__)


if __name__ == "__main__":
    unittest.main()

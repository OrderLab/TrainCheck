"""Subclass tracking must preserve PyTorch module assignment contracts."""

import gc
import io
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


class CustomParameter(nn.Parameter):
    """A distinct framework/user subclass that uses the legacy wrapping path."""


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

    def test_untying_parameter_keeps_both_live_parameters_observable(self) -> None:
        source = nn.Parameter(torch.ones(2))
        module = nn.Module()
        module.weight = source
        module.alias = source
        self.track(module)
        with patch.object(ProxyParameter, "dump_trace", autospec=True) as dump:
            self.registry.dump_sample(dump_config={"torch.nn.Parameter": True})
            self.assertEqual(dump.call_count, 1)
            self.assertIs(dump.call_args.args[0], module.weight)
            dump.reset_mock()
            module.weight = nn.Parameter(torch.zeros(2))
            self.registry.dump_sample(dump_config={"torch.nn.Parameter": True})
            self.assertEqual(dump.call_count, 2)
            observed = [call.args[0] for call in dump.call_args_list]
            self.assertEqual(sum(value is module.weight for value in observed), 1)
            self.assertEqual(sum(value is module.alias for value in observed), 1)
            self.assertNotEqual(
                module.weight.__dict__["var_name"], module.alias.__dict__["var_name"]
            )
            dump.reset_mock()
            weight = module.weight
            alias = module.alias
            assert isinstance(weight, ProxyParameter)
            assert isinstance(alias, ProxyParameter)
            weight.register_object()
            alias.register_object()
            self.registry.dump_modified(dump_config={"torch.nn.Parameter": True})
            self.assertEqual(dump.call_count, 2)
            observed = [call.args[0] for call in dump.call_args_list]
            self.assertEqual(sum(value is module.weight for value in observed), 1)
            self.assertEqual(sum(value is module.alias for value in observed), 1)

    def test_retained_source_updates_are_preserved_when_aliased(self) -> None:
        source = nn.Parameter(torch.ones(2))
        module = nn.Module()
        module.weight = source
        self.track(module)
        source.requires_grad_(False)
        gradient = torch.full((2,), 7.0)
        source.grad = gradient
        marker = object()
        setattr(source, "user_marker", marker)
        module.alias = source
        self.assertIs(module.alias, module.weight)
        self.assertFalse(module.alias.requires_grad)
        self.assertIs(module.alias.grad, gradient)
        self.assertIs(module.alias.__dict__["user_marker"], marker)

    def test_unchanged_source_alias_does_not_erase_accumulated_gradient(self) -> None:
        source = nn.Parameter(torch.ones(2))
        module = nn.Module()
        module.weight = source
        self.track(module)
        module.weight.sum().backward()
        gradient = module.weight.grad
        self.assertIsNotNone(gradient)
        module.alias = source
        self.assertIs(module.alias, module.weight)
        self.assertIs(module.weight.grad, gradient)
        torch.testing.assert_close(module.weight.grad, torch.ones(2))

    def test_hooks_added_to_retained_source_apply_after_alias_assignment(self) -> None:
        source = nn.Parameter(torch.ones(2))
        module = nn.Module()
        module.weight = source
        self.track(module)
        calls: list[torch.Tensor] = []

        def double_gradient(gradient: torch.Tensor) -> torch.Tensor:
            calls.append(gradient.clone())
            return gradient * 2

        handle = source.register_hook(double_gradient)
        self.addCleanup(handle.remove)
        module.alias = source
        module.alias.sum().backward()
        self.assertEqual(len(calls), 1)
        torch.testing.assert_close(module.weight.grad, torch.full((2,), 2.0))
        handle.remove()
        module.weight.grad = None
        module.alias.sum().backward()
        self.assertEqual(len(calls), 1)
        torch.testing.assert_close(module.weight.grad, torch.ones(2))

    def test_parameter_identity_lives_until_last_model_binding_is_removed(self) -> None:
        module = nn.Module()
        parameter = nn.Parameter(torch.ones(2))
        reference = weakref.ref(parameter)
        module.weight = parameter
        self.track(module)
        self.assertIs(module.weight, parameter)
        del parameter
        gc.collect()
        self.assertIs(reference(), module.weight)
        setattr(module, "weight", None)
        gc.collect()
        self.assertIsNone(reference())
        self.assertEqual(self.registry.registry, {})

    def test_removed_source_cannot_overwrite_replacement_registry_entry(self) -> None:
        module = nn.Module()
        source = nn.Parameter(torch.ones(2))
        module.weight = source
        self.track(module)
        reference = weakref.ref(source)
        module.weight = nn.Parameter(torch.zeros(2))
        replacement = module.weight
        source.grad = torch.full((2,), 7.0)
        self.assertIs(self.registry.registry["model.weight"].var, replacement)
        with patch.object(ProxyParameter, "dump_trace", autospec=True) as dump:
            self.registry.dump_sample(dump_config={"torch.nn.Parameter": True})
            self.assertEqual(dump.call_count, 1)
            self.assertIs(dump.call_args.args[0], replacement)
        module.weight = source
        self.assertIs(module.weight, source)
        self.assertIs(self.registry.registry["model.weight"].var, source)
        setattr(module, "weight", None)
        del source
        gc.collect()
        self.assertIsNone(reference())

    def test_replaced_child_parameters_leave_model_observation(self) -> None:
        old_child = nn.Linear(2, 2)
        module = nn.Module()
        module.child = old_child
        self.track(module)
        module.child = nn.Linear(2, 2)
        old_child.weight.grad = torch.ones_like(old_child.weight)
        with patch.object(ProxyParameter, "dump_trace", autospec=True) as dump:
            self.registry.dump_sample(dump_config={"torch.nn.Parameter": True})
            observed = [call.args[0] for call in dump.call_args_list]
            self.assertEqual(len(observed), 2)
            self.assertEqual(sum(value is module.child.weight for value in observed), 1)
            self.assertEqual(sum(value is module.child.bias for value in observed), 1)
        self.assertIs(
            self.registry.registry["model.child.weight"].var, module.child.weight
        )

    def test_tracking_does_not_retain_root_module(self) -> None:
        module = nn.Module()
        source = nn.Parameter(torch.ones(2))
        module.weight = source
        self.track(module)
        root_reference = weakref.ref(module)
        del module
        gc.collect()
        self.assertIsNone(root_reference())
        source.grad = torch.ones(2)
        self.assertEqual(self.registry.registry, {})

    def test_old_session_cleanup_cannot_disable_retracked_parameter(self) -> None:
        source = nn.Parameter(torch.ones(2))
        old_root = nn.Module()
        old_root.weight = source
        self.track(old_root)
        new_root = nn.Module()
        new_root.weight = source
        self.track(new_root)
        del old_root
        gc.collect()
        source.grad = torch.ones(2)
        self.assertIs(self.registry.registry["model.weight"].var, new_root.weight)
        with patch.object(ProxyParameter, "dump_trace", autospec=True) as dump:
            self.registry.dump_sample(dump_config={"torch.nn.Parameter": True})
            self.assertEqual(dump.call_count, 1)
            self.assertIs(dump.call_args.args[0], source)

    def test_shared_parameter_survives_collection_of_either_tracked_root(self) -> None:
        for removed_index in (0, 1):
            with self.subTest(removed_index=removed_index):
                source = nn.Parameter(torch.ones(2))
                roots = [nn.Module(), nn.Module()]
                roots[0].weight = source
                roots[1].weight = source
                proxy_parameter(roots[0], parent_name="left", should_dump_trace=False)
                proxy_parameter(roots[1], parent_name="right", should_dump_trace=False)
                reference = weakref.ref(roots[removed_index])
                del roots[removed_index]
                gc.collect()
                self.assertIsNone(reference())
                expected_name = "right.weight" if removed_index == 0 else "left.weight"
                self.assertEqual(source.__dict__["var_name"], expected_name)
                source.grad = torch.ones(2)
                self.assertEqual(list(self.registry.registry), [expected_name])
                with patch.object(ProxyParameter, "dump_trace", autospec=True) as dump:
                    self.registry.dump_sample(dump_config={"torch.nn.Parameter": True})
                    self.assertEqual(dump.call_count, 1)
                    self.assertIs(dump.call_args.args[0], roots[0].weight)
                roots.clear()
                gc.collect()
                self.assertEqual(self.registry.registry, {})

    def test_overlapping_roots_both_observe_parameter_replacement(self) -> None:
        child = nn.Module()
        source = nn.Parameter(torch.ones(2))
        child.weight = source
        parent = nn.Module()
        parent.child = child
        proxy_parameter(parent, parent_name="parent", should_dump_trace=False)
        proxy_parameter(child, parent_name="child", should_dump_trace=False)
        child.weight = nn.Parameter(torch.zeros(2))
        source.grad = torch.ones(2)
        with patch.object(ProxyParameter, "dump_trace", autospec=True) as dump:
            self.registry.dump_sample(dump_config={"torch.nn.Parameter": True})
            self.assertEqual(dump.call_count, 1)
            self.assertIs(dump.call_args.args[0], child.weight)
        del parent
        gc.collect()
        self.assertEqual(child.weight.__dict__["var_name"], "child.weight")
        self.assertEqual(list(self.registry.registry), ["child.weight"])
        setattr(child, "weight", None)
        self.assertEqual(self.registry.registry, {})

    def test_retracking_same_root_moves_namespace_without_stale_entries(self) -> None:
        module = nn.Module()
        source = nn.Parameter(torch.ones(2))
        module.weight = source
        proxy_parameter(module, parent_name="old", should_dump_trace=False)
        proxy_parameter(module, parent_name="new", should_dump_trace=False)
        gc.collect()
        self.assertIs(module.weight, source)
        self.assertEqual(list(self.registry.registry), ["new.weight"])
        self.assertEqual(source.__dict__["var_name"], "new.weight")
        module.alias = source
        with patch.object(ProxyParameter, "dump_trace", autospec=True) as dump:
            self.registry.dump_sample(dump_config={"torch.nn.Parameter": True})
            self.assertEqual(dump.call_count, 1)
            self.assertIs(dump.call_args.args[0], source)
        del module
        gc.collect()
        self.assertEqual(self.registry.registry, {})

    def test_custom_subclass_keeps_legacy_conversion_with_current_source_state(
        self,
    ) -> None:
        source = CustomParameter(torch.ones(2))
        marker = object()
        setattr(source, "user_marker", marker)
        module = nn.Module()
        module.weight = source
        module.alias = source
        self.track(module)
        self.assertIsInstance(module.weight, ProxyParameter)
        self.assertIsNot(module.weight, source)
        self.assertIs(module.weight, module.alias)
        self.assertIs(module.weight.__dict__["user_marker"], marker)
        # The legacy custom-subclass path has distinct source/proxy identities.
        # A retained source is converted anew, never reused from a stale cache.
        source.requires_grad_(False)
        gradient = torch.full((2,), 7.0)
        source.grad = gradient
        module.later = source
        self.assertIsNot(module.later, module.weight)
        self.assertFalse(module.later.requires_grad)
        self.assertIs(module.later.grad, gradient)
        self.assertTrue(module.weight.requires_grad)

    def test_optimizer_created_before_tracking_and_checkpoint_match_plain(self) -> None:
        torch.manual_seed(9)
        plain = nn.Linear(2, 1)
        tracked = nn.Linear(2, 1)
        tracked.load_state_dict(plain.state_dict())
        plain_optimizer = torch.optim.SGD(plain.parameters(), lr=0.1, momentum=0.9)
        tracked_optimizer = torch.optim.SGD(tracked.parameters(), lr=0.1, momentum=0.9)
        original = tracked.weight
        pointer = original.data_ptr()
        self.track(tracked)
        self.assertIs(tracked.weight, original)
        self.assertEqual(tracked.weight.data_ptr(), pointer)
        self.assertIs(tracked_optimizer.param_groups[0]["params"][0], original)
        source = torch.randn(3, 2)
        for _ in range(3):
            for model, optimizer in (
                (plain, plain_optimizer),
                (tracked, tracked_optimizer),
            ):
                optimizer.zero_grad()
                model(source).square().mean().backward()
                optimizer.step()
        torch.testing.assert_close(
            tracked.state_dict(), plain.state_dict(), rtol=0, atol=0
        )
        torch.testing.assert_close(
            tracked_optimizer.state_dict(), plain_optimizer.state_dict(), rtol=0, atol=0
        )
        checkpoint = io.BytesIO()
        torch.save(tracked.state_dict(), checkpoint)
        checkpoint.seek(0)
        restored = nn.Linear(2, 1)
        restored.load_state_dict(torch.load(checkpoint, weights_only=True))
        torch.testing.assert_close(
            restored.state_dict(), plain.state_dict(), rtol=0, atol=0
        )

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

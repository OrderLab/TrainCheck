"""Semantic unit tests for Relation.to_display_name().

These tests verify that key *meaning* tokens appear in the output for each
relation type given a known params list.  They do NOT test inference logic —
the params are constructed directly, so the tests remain stable even if the
inference algorithm changes.
"""

import pytest

from traincheck.invariant.base_cls import (
    _NOT_SET,
    APIParam,
    InputOutputParam,
    VarTypeParam,
)
from traincheck.invariant.consistency_relation import ConsistencyRelation
from traincheck.invariant.consistency_transient_vars import (
    ConsistentInputOutputRelation,
    ConsistentOutputRelation,
    ThresholdRelation,
)
from traincheck.invariant.contain_relation import APIContainRelation
from traincheck.invariant.cover_relation import FunctionCoverRelation
from traincheck.invariant.DistinctArgumentRelation import DistinctArgumentRelation
from traincheck.invariant.lead_relation import FunctionLeadRelation


class TestAPIContainRelationDisplayName:
    def test_state_transition(self):
        params = [
            APIParam("torch.optim.optimizer.Optimizer.zero_grad"),
            VarTypeParam(
                "torch.nn.Parameter", "grad", pre_value="non_zero", post_value=None
            ),
        ]
        name = APIContainRelation.to_display_name(params)
        assert name is not None
        assert "zero_grad" in name
        assert "grad" in name
        assert "non" in name.lower()  # "non-zero"

    def test_api_calls_api(self):
        params = [
            APIParam("torch.optim.optimizer.Optimizer.step"),
            APIParam("torch.optim.adadelta.adadelta"),
        ]
        name = APIContainRelation.to_display_name(params)
        assert name is not None
        assert "step" in name
        assert "adadelta" in name

    def test_const_value(self):
        params = [
            APIParam("torch.nn.modules.module.Module.forward"),
            VarTypeParam("torch.nn.Parameter", "requires_grad", const_value=True),
        ]
        name = APIContainRelation.to_display_name(params)
        assert name is not None
        assert "forward" in name
        assert "requires_grad" in name

    def test_returns_none_for_empty_params(self):
        assert APIContainRelation.to_display_name([]) is None

    def test_returns_none_for_single_param(self):
        assert APIContainRelation.to_display_name([APIParam("torch.foo")]) is None


class TestConsistencyRelationDisplayName:
    def test_basic(self):
        params = [VarTypeParam("torch.nn.Parameter", "grad")]
        name = ConsistencyRelation.to_display_name(params)
        assert name is not None
        assert "Parameter" in name
        assert "grad" in name
        assert any(w in name.lower() for w in ("consistent", "stay", "step"))

    def test_returns_none_for_empty(self):
        assert ConsistencyRelation.to_display_name([]) is None

    def test_returns_none_for_non_vartype(self):
        assert ConsistencyRelation.to_display_name([APIParam("torch.foo.bar")]) is None


class TestFunctionCoverRelationDisplayName:
    def test_cover_direction(self):
        params = [
            APIParam("torch.distributed.is_initialized"),
            APIParam("torch.nn.modules.module.Module.eval"),
        ]
        name = FunctionCoverRelation.to_display_name(params)
        assert name is not None
        assert "is_initialized" in name
        assert "eval" in name
        assert any(w in name.lower() for w in ("occurs", "cover", "when"))

    def test_returns_none_for_insufficient_params(self):
        assert FunctionCoverRelation.to_display_name([APIParam("torch.foo")]) is None


class TestFunctionLeadRelationDisplayName:
    def test_ordering(self):
        params = [
            APIParam("torch.Tensor.backward"),
            APIParam("torch.optim.optimizer.Optimizer.step"),
        ]
        name = FunctionLeadRelation.to_display_name(params)
        assert name is not None
        assert "backward" in name
        assert "step" in name
        assert any(w in name.lower() for w in ("precede", "before", "lead"))

    def test_merged_three_params(self):
        """Merged lead invariants can have 3 APIParams; display uses first and last."""
        params = [
            APIParam("torch.Tensor.backward"),
            APIParam("torch.optim.optimizer.Optimizer.zero_grad"),
            APIParam("torch.optim.optimizer.Optimizer.step"),
        ]
        name = FunctionLeadRelation.to_display_name(params)
        assert name is not None
        assert "backward" in name
        assert "step" in name

    def test_returns_none_for_single_param(self):
        assert FunctionLeadRelation.to_display_name([APIParam("torch.foo")]) is None


class TestDistinctArgumentRelationDisplayName:
    def test_basic(self):
        params = [APIParam("torch.nn.init.normal_")]
        name = DistinctArgumentRelation.to_display_name(params)
        assert name is not None
        assert "normal_" in name
        assert any(w in name.lower() for w in ("distinct", "different", "argument"))

    def test_returns_none_for_empty(self):
        assert DistinctArgumentRelation.to_display_name([]) is None

    def test_returns_none_for_non_api_param(self):
        params = [VarTypeParam("torch.nn.Parameter", "grad")]
        assert DistinctArgumentRelation.to_display_name(params) is None


class TestConsistentOutputRelationDisplayName:
    def test_with_const_value(self):
        params = [
            APIParam("torch.nn.functional.relu"),
            VarTypeParam("torch.Tensor", "dtype", const_value="float32"),
        ]
        name = ConsistentOutputRelation.to_display_name(params)
        assert name is not None
        assert "relu" in name
        assert "dtype" in name
        assert "float32" in name
        assert any(w in name.lower() for w in ("consistent", "return"))

    def test_without_const_value(self):
        params = [
            APIParam("torch.nn.functional.relu"),
            VarTypeParam("torch.Tensor", "ndim"),
        ]
        name = ConsistentOutputRelation.to_display_name(params)
        assert name is not None
        assert "relu" in name
        assert "ndim" in name

    def test_returns_none_for_insufficient_params(self):
        assert ConsistentOutputRelation.to_display_name([APIParam("torch.foo")]) is None


class TestConsistentInputOutputRelationDisplayName:
    def test_basic(self):
        in_p = InputOutputParam(
            name="input",
            index=0,
            type="torch.Tensor",
            additional_path=("itemsize",),
            api_name="kaiming_uniform_",
            is_input=True,
        )
        out_p = InputOutputParam(
            name="output",
            index=0,
            type="torch.Tensor",
            additional_path=("ndim",),
            api_name="kaiming_uniform_",
            is_input=False,
        )
        api_p = APIParam("torch.nn.init.kaiming_uniform_")
        name = ConsistentInputOutputRelation.to_display_name([in_p, api_p, out_p])
        assert name is not None
        assert "kaiming_uniform_" in name
        assert "itemsize" in name
        assert "ndim" in name
        assert "input" in name.lower()
        assert "output" in name.lower()

    def test_returns_none_for_insufficient_params(self):
        api_p = APIParam("torch.foo")
        assert ConsistentInputOutputRelation.to_display_name([api_p]) is None


class TestThresholdRelationDisplayName:
    def _make_output_param(self, api_name: str) -> InputOutputParam:
        return InputOutputParam(
            name="output_tensors",
            index=0,
            type="torch.Tensor",
            additional_path=("value",),
            api_name=api_name,
            is_input=False,
        )

    def _make_threshold_param(self, name: str, api_name: str) -> InputOutputParam:
        return InputOutputParam(
            name=name,
            index=None,
            type="float",
            additional_path=None,
            api_name=api_name,
            is_input=True,
        )

    def test_min_threshold_gte(self):
        """params=[output, api, threshold] → output ≥ threshold."""
        api_p = APIParam("torch.optim.optimizer.Optimizer.step")
        out_p = self._make_output_param("Optimizer.step")
        thresh_p = self._make_threshold_param("lr", "Optimizer.step")
        name = ThresholdRelation.to_display_name([out_p, api_p, thresh_p])
        assert name is not None
        assert "Optimizer.step" in name
        assert "lr" in name
        assert "≥" in name

    def test_max_threshold_lte(self):
        """params=[threshold, api, output] → output ≤ threshold."""
        api_p = APIParam("torch.optim.optimizer.Optimizer.step")
        out_p = self._make_output_param("Optimizer.step")
        thresh_p = self._make_threshold_param("lr", "Optimizer.step")
        name = ThresholdRelation.to_display_name([thresh_p, api_p, out_p])
        assert name is not None
        assert "Optimizer.step" in name
        assert "lr" in name
        assert "≤" in name

    def test_returns_none_for_insufficient_params(self):
        assert ThresholdRelation.to_display_name([APIParam("torch.foo")]) is None

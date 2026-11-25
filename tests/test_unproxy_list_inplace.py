"""Regression tests for issue #176: unproxy_arg must not copy list arguments.

torch.optim.SGD.step builds ``momentum_buffer_list``, passes it to the functional
``sgd``, which stores newly created buffers into that list, and then copies them
into ``optimizer.state``. When a wrapper unproxied the arguments by building a
new list, those writes went into the copy and SGD momentum was silently reset
every step.
"""

import importlib
import os
import sys
import unittest
from unittest.mock import patch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch

from traincheck.instrumentor.proxy_wrapper.proxy_basics import (
    unproxy_arg,
    unproxy_args_kwargs,
    unproxy_func,
)


class TestUnproxyListInPlace(unittest.TestCase):
    def test_list_identity_is_preserved(self):
        values = [torch.zeros(2), None]
        self.assertIs(unproxy_arg(values), values)

    def test_nested_list_identity_is_preserved(self):
        inner = [None]
        outer = [inner]
        self.assertIs(unproxy_arg(outer)[0], inner)

    def test_positional_list_write_back_reaches_caller(self):
        buffers = [None, None]
        args, _ = unproxy_args_kwargs((buffers,), {})
        args[0][1] = torch.ones(1)
        self.assertIsNotNone(buffers[1])

    def test_keyword_list_write_back_reaches_caller(self):
        buffers = [None]
        _, kwargs = unproxy_args_kwargs((), {"momentum_buffer_list": buffers})
        kwargs["momentum_buffer_list"][0] = torch.ones(1)
        self.assertIsNotNone(buffers[0])

    def test_tuples_are_still_unwrapped(self):
        values = (torch.zeros(1), [None])
        out = unproxy_arg(values)
        self.assertIsInstance(out, tuple)
        self.assertIs(out[1], values[1])

    def test_sgd_momentum_survives_a_wrapped_functional_sgd(self):
        torch.manual_seed(0)
        model = torch.nn.Linear(3, 1)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9)
        sgd_module = importlib.import_module("torch.optim.sgd")
        with patch.object(sgd_module, "sgd", unproxy_func(sgd_module.sgd)):
            for _ in range(2):
                optimizer.zero_grad()
                model(torch.randn(4, 3)).sum().backward()
                optimizer.step()
        for param in model.parameters():
            self.assertIsNotNone(optimizer.state[param].get("momentum_buffer"))


if __name__ == "__main__":
    unittest.main()

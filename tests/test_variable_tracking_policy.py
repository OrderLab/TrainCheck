import os
import sys
import unittest
from unittest.mock import MagicMock, patch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch

import traincheck.config.config as config
from traincheck.instrumentor.proxy_wrapper.proxy import Proxy
from traincheck.instrumentor.proxy_wrapper.proxy_observer import observe_proxy_var
from traincheck.instrumentor.proxy_wrapper.subclass import ProxyParameter


class TestVariableTrackingPolicy(unittest.TestCase):
    def setUp(self):
        config.DISABLE_WRAPPER = False
        os.environ["TRAINCHECK_OUTPUT_DIR"] = "/tmp/test_var_policy"
        os.makedirs("/tmp/test_var_policy", exist_ok=True)

    def tearDown(self):
        config.DISABLE_WRAPPER = False
        import shutil

        if os.path.exists("/tmp/test_var_policy"):
            shutil.rmtree("/tmp/test_var_policy")

    @patch("traincheck.instrumentor.proxy_wrapper.proxy.Proxy.jsondumper")
    def test_proxy_dump_trace_respects_policy(self, mock_jsondumper):
        # Create a proxy
        obj = torch.tensor([1.0])
        proxy = Proxy(obj, var_name="test_var", should_dump_trace=False)

        # Test enabled
        config.DISABLE_WRAPPER = False
        proxy.dump_trace("update", "dumpy_loc")
        # proxy.jsondumper.dump_json is called
        self.assertTrue(mock_jsondumper.dump_json.called)
        mock_jsondumper.dump_json.reset_mock()

        # Test disabled
        config.DISABLE_WRAPPER = True
        proxy.dump_trace("update", "dumpy_loc")
        self.assertFalse(mock_jsondumper.dump_json.called)

    @patch("traincheck.instrumentor.proxy_wrapper.subclass.dump_trace_VAR")
    def test_proxy_parameter_dump_trace_respects_policy(self, mock_dump_trace_VAR):
        # Create a ProxyParameter
        param = torch.nn.Parameter(torch.tensor([1.0]))
        # We need to mock get_timestamp_ns etc to avoid errors/noise?
        # Actually initializing ProxyParameter triggers dump_trace ("initing").
        # We suppress that with should_dump_trace=False

        proxy_param = ProxyParameter(
            param, var_name="test_param", should_dump_trace=False
        )

        # Test enabled
        config.DISABLE_WRAPPER = False
        proxy_param.dump_trace("update", "loc")
        self.assertTrue(mock_dump_trace_VAR.called)
        mock_dump_trace_VAR.reset_mock()

        # Test disabled
        config.DISABLE_WRAPPER = True
        proxy_param.dump_trace("update", "loc")
        self.assertFalse(mock_dump_trace_VAR.called)

    @patch(
        "traincheck.instrumentor.dumper.dump_trace_VAR"
    )  # observe calls var.dump_trace
    def test_proxy_observer_respects_policy(self, mock_dump_trace_VAR):
        # Create a fake proxy
        mock_proxy = MagicMock()
        mock_proxy.update_timestamp = MagicMock()
        mock_proxy.register_object = MagicMock()
        mock_proxy.dump_trace = MagicMock()

        # Test enabled
        config.DISABLE_WRAPPER = False
        observe_proxy_var(mock_proxy, "pre_observe", "api")
        self.assertTrue(mock_proxy.dump_trace.called)
        mock_proxy.dump_trace.reset_mock()

        # Test disabled
        config.DISABLE_WRAPPER = True
        observe_proxy_var(mock_proxy, "pre_observe", "api")
        self.assertFalse(mock_proxy.dump_trace.called)


if __name__ == "__main__":
    unittest.main()

import os
import shutil
import sys
import tempfile
import unittest
from unittest.mock import MagicMock, patch

# Ensure traincheck is in path (standard pattern for this repo based on other tests)
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import traincheck.collect_trace as collect_trace
from traincheck.config import config


class TestPolicyInjection(unittest.TestCase):
    def setUp(self):
        self.test_dir = tempfile.mkdtemp()
        self.dummy_script = os.path.join(self.test_dir, "dummy_script.py")
        with open(self.dummy_script, "w") as f:
            f.write("import torch\n")

    def tearDown(self):
        shutil.rmtree(self.test_dir)

    @patch("traincheck.runner.ProgramRunner")
    @patch(
        "traincheck.instrumentor.instrument_file",
        side_effect=collect_trace.instrumentor.instrument_file,
    )
    def test_policy_injection(self, mock_instrument_file, MockProgramRunner):
        # Setup mock runner
        mock_runner_instance = MockProgramRunner.return_value
        mock_runner_instance.run.return_value = ("output", 0)

        # Simulate command line arguments
        test_args = [
            "collect_trace.py",
            "-p",
            self.dummy_script,
            "--sampling-interval",
            "3",
            "--warm-up-steps",
            "2",
            "--only-instr",
        ]

        with patch.object(sys, "argv", test_args):
            collect_trace.main()

        # Check if instrument_file was called with the correct args
        call_args = mock_instrument_file.call_args
        self.assertIsNotNone(call_args, "instrument_file was not called")
        _, kwargs = call_args
        self.assertEqual(kwargs.get("sampling_interval"), 3)
        self.assertEqual(kwargs.get("warm_up_steps"), 2)

        # Check if the policy ends up in instrumented source code
        runner_call_args = MockProgramRunner.call_args
        self.assertIsNotNone(runner_call_args, "ProgramRunner was not initialized")
        source_code = runner_call_args[0][0]  # first arg is source_code

        self.assertIn("sampling_interval=3", source_code)
        self.assertIn("warm_up_steps=2", source_code)

    @patch("traincheck.runner.ProgramRunner")
    @patch(
        "traincheck.instrumentor.instrument_file",
        side_effect=collect_trace.instrumentor.instrument_file,
    )
    @patch("traincheck.collect_trace.read_inv_file")
    def test_defaults_with_invariant(
        self, mock_read_inv, mock_instrument_file, MockProgramRunner
    ):
        # Setup mocks
        mock_runner_instance = MockProgramRunner.return_value
        mock_runner_instance.run.return_value = ("output", 0)
        mock_read_inv.return_value = []  # Return empty list of invariants

        test_args = [
            "collect_trace.py",
            "-p",
            self.dummy_script,
            "-i",
            "dummy_inv.json",  # Enable invariants
            "--only-instr",
        ]

        with patch.object(sys, "argv", test_args):
            collect_trace.main()

        call_args = mock_instrument_file.call_args
        _, kwargs = call_args

        # Should default to config values
        expected_interval = config.DEFAULT_CHECKING_POLICY["interval"]
        expected_warmup = config.DEFAULT_CHECKING_POLICY["warm_up"]

        self.assertEqual(kwargs.get("sampling_interval"), expected_interval)
        self.assertEqual(kwargs.get("warm_up_steps"), expected_warmup)

    @patch("traincheck.runner.ProgramRunner")
    @patch(
        "traincheck.instrumentor.instrument_file",
        side_effect=collect_trace.instrumentor.instrument_file,
    )
    def test_defaults_without_invariant(self, mock_instrument_file, MockProgramRunner):
        # Setup mocks
        mock_runner_instance = MockProgramRunner.return_value
        mock_runner_instance.run.return_value = ("output", 0)

        test_args = ["collect_trace.py", "-p", self.dummy_script, "--only-instr"]

        with patch.object(sys, "argv", test_args):
            collect_trace.main()

        call_args = mock_instrument_file.call_args
        _, kwargs = call_args

        # Should default to config.INSTRUMENTATION_POLICY values
        expected_interval = config.INSTRUMENTATION_POLICY["interval"]
        expected_warmup = config.INSTRUMENTATION_POLICY["warm_up"]

        self.assertEqual(kwargs.get("sampling_interval"), expected_interval)
        self.assertEqual(kwargs.get("warm_up_steps"), expected_warmup)


if __name__ == "__main__":
    unittest.main()

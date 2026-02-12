import os
import shutil
import sys
import tempfile
import unittest
from unittest.mock import MagicMock, patch

import torch

# Ensure traincheck is in path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import traincheck.config.config as config
from traincheck.instrumentor.caches import META_VARS
from traincheck.instrumentor.tracer import Instrumentor


class TestDynamicPolicy(unittest.TestCase):
    def setUp(self):
        META_VARS["step"] = 0
        META_VARS["stage"] = "training"
        config.INSTRUMENTATION_POLICY = None
        self.test_dir = tempfile.mkdtemp()
        os.environ["TRAINCHECK_OUTPUT_DIR"] = self.test_dir

    def tearDown(self):
        if os.path.exists(self.test_dir):
            shutil.rmtree(self.test_dir)

    @patch("traincheck.config.config.DISABLE_WRAPPER", new=False)
    def test_sampling_interval(self):
        # Setup policy
        from traincheck.config import config
        from traincheck.instrumentor.control import start_step

        config.INSTRUMENTATION_POLICY = {"interval": 2, "warm_up": 0}

        start_step()  # Step 1: (1-0)%2 != 0 -> Disabled
        self.assertTrue(config.DISABLE_WRAPPER, "Step 1 should be disabled")

        start_step()  # Step 2: (2-0)%2 == 0 -> Enabled
        self.assertFalse(config.DISABLE_WRAPPER, "Step 2 should be enabled")

        start_step()  # Step 3: Disabled
        self.assertTrue(config.DISABLE_WRAPPER, "Step 3 should be disabled")

        start_step()  # Step 4: Enabled
        self.assertFalse(config.DISABLE_WRAPPER, "Step 4 should be enabled")

    @patch("traincheck.config.config.DISABLE_WRAPPER", new=False)
    def test_warmup(self):
        # Setup policy
        from traincheck.config import config
        from traincheck.instrumentor.control import start_step

        config.INSTRUMENTATION_POLICY = {"interval": 10, "warm_up": 2}

        start_step()  # Step 1. Warmup.
        self.assertFalse(config.DISABLE_WRAPPER, "Step 1 (warmup) should be enabled")

        start_step()  # Step 2. Warmup.
        self.assertFalse(config.DISABLE_WRAPPER, "Step 2 (warmup) should be enabled")

        start_step()  # Step 3. (3-2)%10 != 0 -> Disabled.
        self.assertTrue(config.DISABLE_WRAPPER, "Step 3 should be disabled")

        # Fast forward to step 12
        for _ in range(9):
            start_step()

        # Step 12. (12-2)%10 == 0 -> Enabled.
        self.assertFalse(config.DISABLE_WRAPPER, "Step 12 should be enabled")

    def test_stage_change_resets_wrapper(self):
        # Simulate being in a "skip" state
        config.DISABLE_WRAPPER = True
        META_VARS["stage"] = "training"

        from traincheck.developer.annotations import annotate_stage

        # Change stage to evaluation
        annotate_stage("evaluation")

        # Should be enabled now
        self.assertFalse(
            config.DISABLE_WRAPPER, "DISABLE_WRAPPER should be False after stage change"
        )
        self.assertEqual(META_VARS["stage"], "evaluation")

        # Change back to training
        config.DISABLE_WRAPPER = True  # Simulate it was somehow disabled again
        annotate_stage("training")
        self.assertFalse(
            config.DISABLE_WRAPPER,
            "DISABLE_WRAPPER should be False after entering training",
        )


if __name__ == "__main__":
    unittest.main()

import logging

from traincheck.config import config
from traincheck.instrumentor.caches import META_VARS

logger = logging.getLogger(__name__)


def start_step():
    """
    Called at the start of a training iteration to control instrumentation policy.
    increments step count and sets config.DISABLE_WRAPPER based on policy.
    """
    # Only control policy if we are in training stage.
    # If explicit stage annotation is used, respect it.
    # If not tracking stage (or stage is None), we assume training if this is called?
    # Better to be safe and check if specific stage is set to non-training.
    stage = META_VARS.get("stage")
    if stage and stage != "training":
        # If explicitly in a non-training stage (e.g. evaluation),
        # we might want to disable wrapping?
        # Or just do nothing and let other logic handle it?
        # The user's request specificially mentioned alignment with training steps.
        # If we are in evaluation loop, we probably shouldn't be incrementing "step" or applying sampling policy intended for training.
        return

    META_VARS["step"] += 1
    current_step = META_VARS["step"]

    policy = config.INSTRUMENTATION_POLICY
    if policy:
        warm_up = policy["warm_up"]
        interval = policy["interval"]

        # Default to enabled
        config.DISABLE_WRAPPER = False

        if current_step < warm_up:
            print(f"Warmup step {current_step}")
            config.DISABLE_WRAPPER = False
        elif (current_step - warm_up) % interval == 0:
            print(f"Interval step {current_step}")
            config.DISABLE_WRAPPER = False
        else:
            print(f"Skipping step {current_step}")
            config.DISABLE_WRAPPER = True
    else:
        # No policy, always enable
        config.DISABLE_WRAPPER = False


def start_eval_step():
    """
    Called at the start of an evaluation iteration.
    Controls instrumentation policy using a separate step counter.
    """
    if "eval_step" not in META_VARS:
        META_VARS["eval_step"] = 0

    META_VARS["eval_step"] += 1
    current_step = META_VARS["eval_step"]

    policy = config.INSTRUMENTATION_POLICY
    if policy:
        warm_up = policy["warm_up"]
        interval = policy["interval"]

        config.DISABLE_WRAPPER = False

        if current_step < warm_up:
            print(f"Eval: Warmup step {current_step}")
            config.DISABLE_WRAPPER = False
        elif (current_step - warm_up) % interval == 0:
            print(f"Eval: Interval step {current_step}")
            config.DISABLE_WRAPPER = False
        else:
            print(f"Eval: Skipping step {current_step}")
            config.DISABLE_WRAPPER = True
    else:
        config.DISABLE_WRAPPER = False

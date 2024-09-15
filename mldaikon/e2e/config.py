import os

current_dir = os.path.dirname(os.path.abspath(__file__))
modules_to_instrument = ["megatron", "deepspeed", "torch"]
proxy_module = "model"
# should be current_dir + '../../example_pipelines'
EXAMPLE_PIPELINES_DIR = os.path.join(current_dir, "../../machine-learning-issues")
input_env = {
    "PYTORCH_JIT": "0"
}  # should appear at the start of the mldaikon.collect_trace running command

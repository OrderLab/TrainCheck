import os

current_dir = os.path.dirname(os.path.abspath(__file__))
modules_to_instrument = ["megatron", "deepspeed", "torch"]
# should be current_dir + '../../example_pipelines'
EXAMPLE_PIPELINES_DIR = os.path.join(current_dir, "../../example_pipelines")
input_env = {"PYTORCH_JIT": "0"}

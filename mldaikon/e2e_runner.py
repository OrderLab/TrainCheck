import argparse
import json
import os
import sys

import mldaikon.e2e.config as e2e_config
from mldaikon.e2e.runner import find_files, run_e2e


def read_config(config_path: str) -> dict[str, str]:
    # read the configuration json file and return the configuration dictionary
    with open(config_path, "r") as f:
        config = json.load(f)
    return config


def parse_input_env(input_env: str) -> dict[str, str]:
    # parse the input environment variables
    if input_env is None:
        return {}
    env_dict = {}
    for env in input_env.split(","):
        key, value = env.split("=")
        env_dict[key] = value
    return env_dict


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="End-to-end Runner for the ML Daikon project"
    )
    parser.add_argument(
        "-s",
        "--script_name",
        required=True,
        type=str,
        help="Name of the script to be run",
    )
    parser.add_argument(
        "--example_pipelines_dir",
        type=str,
        required=False,
        default=e2e_config.EXAMPLE_PIPELINES_DIR,
        help="Path to the directory containing the example pipelines",
    )
    parser.add_argument(
        "--input_env",
        type=str,
        required=False,
        default=None,
        help="Environment variables to be passed to the script",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=False,
        default=None,
        help="Output directory for the pipeline",
    )
    parser.add_argument(
        "--profiling",
        action="store_true",
        help="Enable profiling during the trace collection",
    )

    args = parser.parse_args()
    example_pipelines_dir = args.example_pipelines_dir
    input_env = parse_input_env(args.input_env)
    input_env = {**input_env, **e2e_config.input_env}
    script_name = args.script_name
    input_bash_script = ""
    input_program = os.path.join(example_pipelines_dir, f"{script_name}.py")
    # if input_program is not a file, then it is a directory
    if not os.path.isfile(input_program):
        input_program_dir = os.path.join(example_pipelines_dir, script_name)
        input_program_list = find_files(input_program_dir, prefix="", suffix=".py")
        input_bash_script_list = find_files(input_program_dir, prefix="", suffix=".sh")
        assert (
            len(input_program_list) == 1
        ), f"Multiple python files found in {input_program_dir}"
        input_program = input_program_list[0]
        assert (
            len(input_bash_script_list) <= 1
        ), f"Multiple bash files found in {input_program_dir}"
        if len(input_bash_script_list) == 1:
            input_bash_script = input_bash_script_list[0]

    # if output_dir is not provided, then create a new directory with the script name
    if args.output_dir is None:
        output_dir = os.path.join(example_pipelines_dir, f"../output/{script_name}")
    else:
        output_dir = args.output_dir
    # get current python path
    python_path = sys.executable
    modules_to_instrument = e2e_config.modules_to_instrument
    api_log_dir = os.path.join(output_dir, "trace_log")
    input_config: dict[str, str] = {
        "input_program": input_program,
        "input_bash_script": input_bash_script,
        "modules_to_instrument": " ".join(modules_to_instrument),
        # "disable_proxy_class": False,
        # "scan_proxy_in_args": False,
        # "allow_disable_dump": False,
        # "funcs_of_inv_interest": None,
        "proxy_module": "model_transfer",
        "proxy_log_dir": output_dir,
        "API_log_dir": api_log_dir,
        "profiling": str(args.profiling),
    }
    # run the e2e pipeline
    run_e2e(
        python_path=python_path,
        input_config=input_config,
        input_env=input_env,
        output_dir=output_dir,
    )

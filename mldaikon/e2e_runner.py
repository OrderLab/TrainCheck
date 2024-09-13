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
        "--modules_to_instrument",
        required=False,
        nargs="*",
        help="Modules to be instrumented",
        default=e2e_config.modules_to_instrument,
    )
    parser.add_argument(
        "--proxy_module",
        required=False,
        nargs="*",
        help="Modules to be proxied (model by default)",
        default=e2e_config.proxy_module,
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
    parser.add_argument(
        "--only_var",
        action="store_true",
        help="Only check variable related invariants",
    )
    parser.add_argument(
        "--only_func",
        action="store_true",
        help="Only check function related invariants",
    )

    args = parser.parse_args()
    example_pipelines_dir = args.example_pipelines_dir
    input_env = parse_input_env(args.input_env)
    input_env = {**input_env, **e2e_config.input_env}
    script_name = args.script_name
    input_bash_script = ""
    input_program = os.path.join(example_pipelines_dir, f"{script_name}.py")
    modules_to_instrument = args.modules_to_instrument
    # if input_program is not a file, then it is a directory
    if not os.path.isfile(input_program):
        input_program_dir = os.path.join(example_pipelines_dir, script_name)
        input_program_list = find_files(input_program_dir, prefix="", suffix=".py")
        # input program should not include _ml_daikon at the beginning of the name
        # e.g. '../../example_pipelines/LT-725/_ml_daikon_LT725.py' is not a valid input program
        input_program_list = [
            file
            for file in input_program_list
            if not os.path.basename(file).startswith("_ml_daikon")
        ]
        input_bash_script_list = find_files(input_program_dir, prefix="", suffix=".sh")
        input_bash_script_list = [
            file for file in input_bash_script_list if not file.endswith("install.sh")
        ]
        input_config_file = os.path.join(input_program_dir, "config.json")
        print(f"input_config_file: {input_config_file}")
        if not os.path.exists(input_config_file):
            assert (
                len(input_program_list) == 1
            ), f"Multiple python files found in {input_program_dir}, {input_program_list}"
            input_program = input_program_list[0]
            assert (
                len(input_bash_script_list) <= 1
            ), f"Multiple bash files found in {input_program_dir}"
            if len(input_bash_script_list) == 1:
                input_bash_script = input_bash_script_list[0]
        else:
            # the info from the config file will override previous configurations
            for var_name, var_list in [
                ("input_program", input_program_list),
                ("input_bash_script", input_bash_script_list),
            ]:
                try:
                    globals()[var_name] = var_list[0]
                except Exception:
                    pass

            config = read_config(input_config_file)
            if "input_program" in config and config["input_program"] != "":
                input_program = os.path.join(input_program_dir, config["input_program"])
            if "input_bash_script" in config and config["input_bash_script"] != "":
                input_bash_script = os.path.join(
                    input_program_dir, config["input_bash_script"]
                )
            if "modules_to_instrument" in config:
                args.modules_to_instrument = config["modules_to_instrument"]
            if "proxy_module" in config:
                args.proxy_module = config["proxy_module"]

    # if output_dir is not provided, then create a new directory with the script name
    if args.output_dir is None:
        output_dir = os.path.join(example_pipelines_dir, f"../output/{script_name}")
    else:
        output_dir = args.output_dir
    # get current python path
    python_path = sys.executable
    modules_to_instrument = args.modules_to_instrument
    api_log_dir = os.path.join(output_dir, "trace_log")
    input_config: dict[str, str] = {
        "input_program": input_program,
        "input_bash_script": input_bash_script,
        "modules_to_instrument": " ".join(modules_to_instrument),
        "proxy_module": args.proxy_module,
        # "disable_proxy_class": False,
        # "scan_proxy_in_args": False,
        # "allow_disable_dump": False,
        # "funcs_of_inv_interest": None,
        "output_dir": output_dir,
        "API_log_dir": api_log_dir,
        "profiling": str(args.profiling),
        "only_var": str(args.only_var),
        "only_func": str(args.only_func),
    }
    # run the e2e pipeline
    run_e2e(
        python_path=python_path,
        input_config=input_config,
        input_env=input_env,
    )

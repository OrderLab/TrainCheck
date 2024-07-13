import json
import os
import subprocess


def run_python_script(
    exe: str = "python",
    script_args: list[str] = [],
    script_env: dict[str, str] = {},
    output_dir: str = ".",
) -> int:
    # run the script with the given arguments and environment variables
    cmdline = (
        " ".join([f"{k}={v}" for k, v in script_env.items()])
        + " "
        + " ".join([exe] + script_args)
        + f" > {output_dir}/stdout.txt 2> {output_dir}/stderr.txt"
    )
    print(f"Running command: {cmdline}")
    print(f"Environment variables: {script_env}")
    print(f"Output would be dumped to: {output_dir}")
    # create the output directory if it does not exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    # run the script in shell
    process = subprocess.Popen(
        cmdline,
        shell=True,
    )
    process.wait()

    return process.returncode


def run_e2e(
    python_path: str,
    input_config: dict[str, str],
    input_env: dict[str, str],
    output_dir: str,
) -> int:
    # this is the end to end invariant generation pipeline for mldaikon project
    # input_program: the path to the python script to be run (should be uninstrumented user's script)
    # input_config: the configuration for the mldaikon project
    # input_env: the environment variables for the script
    input_program: str = input_config["input_program"]  # with -p flag
    modules_to_instrument: str = input_config["modules_to_instrument"]  # with -t flag
    proxy_module = input_config["proxy_module"]  # with --proxy_module flag
    proxy_log_dir = input_config["proxy_log_dir"]  # with --proxy_log_dir flag
    script_args: list[str] = [
        "-m",
        "mldaikon.collect_trace",
        "-p",
        input_program,
        "-t",
        modules_to_instrument,
        "--proxy_module",
        " ".join(proxy_module),
        "--proxy_log_dir",
        proxy_log_dir,
    ]
    # run the script with the given arguments and environment variables
    return_code = run_python_script(
        python_path, script_args, input_env, output_dir
    )  # ignore: type
    return return_code


def read_config(config_path: str) -> dict[str, str]:
    # read the configuration json file and return the configuration dictionary
    with open(config_path, "r") as f:
        config = json.load(f)
    return config


if __name__ == "__main__":
    current_dir = os.path.dirname(os.path.abspath(__file__))
    # should be current_dir + '../../example_pipelines'
    example_pipelines_dir = os.path.join(current_dir, "../../example_pipelines")
    python_path = "/home/ziming/miniconda3/bin/python"
    script_name = "PT84911"
    modules_to_instrument = ["megatron", "deepspeed", "torch"]
    output_dir = os.path.join(current_dir, f"../../output/{script_name}")
    input_config: dict[str, str] = {
        "input_program": os.path.join(example_pipelines_dir, f"{script_name}.py"),
        "modules_to_instrument": " ".join(modules_to_instrument),
        # "disable_proxy_class": False,
        # "scan_proxy_in_args": False,
        # "allow_disable_dump": False,
        # "funcs_of_inv_interest": None,
        "proxy_module": "model_transfer",
        "proxy_log_dir": output_dir,
    }
    input_env = {"PYTORCH_JIT": "0"}

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    # run the e2e pipeline
    run_e2e(
        python_path=python_path,
        input_config=input_config,
        input_env=input_env,
        output_dir=output_dir,
    )

import os
import subprocess


def find_files(directory: str, prefix: str, suffix: str) -> list[str]:
    # find all the file names inside the directory with the given suffix
    files = []
    for root, _, filenames in os.walk(directory):
        for filename in filenames:
            if filename.endswith(suffix) and filename.startswith(prefix):
                files.append(os.path.join(root, filename))
    return files


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
    api_log_dir = input_config["API_log_dir"]  # with --API_log_dir flag
    profiling = input_config["profiling"]  # with --profiling flag

    # run the script with the given arguments and environment variables
    trace_collector_script_args: list[str] = [
        "-m",
        "mldaikon.collect_trace",
        "-p",
        input_program,
        "-t",
        modules_to_instrument,
        "--proxy_module",
        proxy_module,
        "--proxy_log_dir",
        proxy_log_dir,
        "--API_log_dir",
        api_log_dir,
        "--profiling",
        profiling,
    ]
    if "input_bash_script" in input_config and input_config["input_bash_script"] != "":
        trace_collector_script_args += ["-s", input_config["input_bash_script"]]
    # clear up the output directory
    if os.path.exists(output_dir):
        os.system(f"rm -r {output_dir}")
    os.makedirs(output_dir)
    os.makedirs(api_log_dir)

    # run the script with the given arguments and environment variables
    return_code = run_python_script(
        python_path, trace_collector_script_args, input_env, output_dir
    )  # ignore: type

    if return_code != 0:
        print(f"Error: the script returned with non-zero exit code: {return_code}")
        print("E2E failed at collect_trace stage")
        return return_code

    # process the proxy_log.json file from output_dir
    # example: python proxy_trace_process.py --input  ./output/PT84911/proxy_log.json
    process_trace_script_args: list[str] = [
        "proxy_trace_process.py",
        "--input",
        f"{output_dir}/proxy_log.json",
    ]
    return_code = run_python_script(
        python_path, process_trace_script_args, {}, output_dir
    )  # ignore: type

    if return_code != 0:
        print(f"Error: the script returned with non-zero exit code: {return_code}")
        print("E2E failed at proxy_trace_process stage")
        return return_code

    ## Activate the Infer Engine
    # example: python -m mldaikon.infer_engine -t <proxy_folder>/proxy_trace_processed_* <trace_folder>/<path_to_API_trace>
    trace_folder = os.path.join(output_dir, "trace_log")
    proxy_folder = os.path.join(output_dir, "processed_proxy_traces")

    api_trace_folder = os.path.join(trace_folder, "API")

    processed_proxy_files = find_files(
        proxy_folder, prefix="proxy_trace_processed_", suffix=".json"
    )
    API_trace_files = find_files(api_trace_folder, prefix="_ml_daikon", suffix=".log")

    import pdb

    pdb.set_trace()

    infer_engine_script_args: list[str] = [
        "-m",
        "mldaikon.infer_engine",
        "-t",
        " ".join(processed_proxy_files),
        " ".join(API_trace_files),
    ]

    return_code = run_python_script(
        python_path, infer_engine_script_args, {}, output_dir
    )  # ignore: type

    if return_code != 0:
        print(f"Error: the script returned with non-zero exit code: {return_code}")
        print("E2E failed at infer_engine stage")
        return return_code
    # the processed log file is now in output_dir/proxy_log_processed

    return return_code

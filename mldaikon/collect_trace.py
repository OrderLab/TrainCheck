import argparse
import datetime
import logging
import os

import mldaikon.config.config as config
import mldaikon.instrumentor as instrumentor
import mldaikon.proxy_wrapper.proxy_config as proxy_config
import mldaikon.runner as runner
from mldaikon.invariant.base_cls import APIParam, Invariant, read_inv_file


def get_list_of_funcs_from_invariants(invariants: list[Invariant]) -> list[str]:
    """
    Get a list of functions from the invariants
    """
    funcs = set()
    for inv in invariants:
        for param in inv.params:
            if isinstance(param, APIParam):
                funcs.add(param.api_full_name)
    return sorted(list(funcs))


def dump_env(output_dir: str):
    with open(os.path.join(output_dir, "env_dump.txt"), "w") as f:
        f.write("Arguments:\n")
        for arg in vars(args):
            f.write(f"{arg}: {getattr(args, arg)}\n")
        f.write("\n")
        f.write("Environment Variables:\n")
        for key, value in os.environ.items():
            f.write(f"{key}: {value}\n")
        f.write("\n")
        f.write("Python Version:\n")
        f.write(f"{os.popen('python --version').read()}\n")
        f.write("\n")
        f.write("Library Versions:\n")
        f.write(
            f"{os.popen('conda list').read()}\n"
        )  # FIXME: conda list here doesn't work in OSX, >>> import os; >>> os.popen('conda list').read(); /bin/sh: conda: command not found


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Invariant Finder for ML Pipelines in Python"
    )
    parser.add_argument(
        "-p",
        "--pyscript",
        type=str,
        required=True,
        help="Path to the main file of the pipeline to be analyzed",
    )
    parser.add_argument(
        "-s",
        "--shscript",
        type=str,
        required=False,
        help="""Path to the shell script that runs the python script. 
        If not provided, the python script will be run directly.""",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="",
        help="""Directory to store the output files, if not provided, it will be 
        defaulted to mldaikon_run_{pyscript_name}_{timestamp}""",
    )
    parser.add_argument(
        "--only-instr",
        action="store_true",
        help="Only instrument and dump the modified file",
    )
    parser.add_argument(
        "-t",
        "--modules-to-instr",
        nargs="*",
        help="Modules to be instrumented",
        default=config.INSTR_MODULES_TO_INSTR,
    )
    parser.add_argument(
        "--disable-scan-proxy-in-args",
        action="store_true",
        help="NOT Scan the arguments of the function for proxy objects, this will enable the infer engine to understand the relationship between the proxy objects and the functions",
    )
    parser.add_argument(
        "--proxy-module",
        type=str,
        default="",
        help="The module to be traced by the proxy wrapper",
    )
    parser.add_argument(
        "--profiling",
        type=str,
        default=proxy_config.profiling,
        help="Enable to do profiling during the training process,"
        "there would be a train_profiling_results.pstats file generated"
        "in the current directory",
    )
    parser.add_argument(
        "--proxy-update-limit",
        type=float,
        default=proxy_config.proxy_update_limit,
        help="The threshold for updating the proxy object",
    )
    parser.add_argument(
        "-d",
        "--debug-mode",
        action="store_true",
        help="Enable debug mode for the program",
    )
    parser.add_argument(
        "--API-dump-stack-trace",
        action="store_true",
        help="Dump the stack trace for API calls",
    )
    parser.add_argument(
        "-i",
        "--invariants",
        nargs="*",
        help="Invariant files produced by the inference engine. If provided, we will only collect traces for APIs and variables that are related to the invariants. This can be used to speed up the trace collection. HAS TO BE USED WITHOUT `--use-full-instr` for the optimization to work properly.",
        default=None,
    )
    parser.add_argument(
        "--use-full-instr",
        action="store_true",
        help="Use full instrumentation for the instrumentor, if not set, the instrumentor may not dump traces for certain APIs in modules deemed not important (e.g. jit in torch)",
    )
    parser.add_argument(
        "--tensor-dump-format",
        choices=["hash", "stats", "full", "version"],
        type=str,
        default="hash",
        help="The format for dumping tensors. Choose from 'hash'(default), 'stats', 'full' or 'version'(deprecated).",
    )
    parser.add_argument(
        "--delta-dump",
        type=bool,
        default=proxy_config.delta_dump_config["delta_dump"],
        help="Only dump the changed part of the object",
    )
    parser.add_argument(
        "--delta-dump-meta-var",
        type=bool,
        default=proxy_config.delta_dump_config["delta_dump_meta_var"],
        help="Only dump the changed part of the meta_var",
    )
    parser.add_argument(
        "--delta-dump-attributes",
        type=bool,
        default=proxy_config.delta_dump_config["delta_dump_attributes"],
        help="Only dump the changed part of the attribute",
    )
    parser.add_argument(
        "--enable-C-level-observer",
        type=bool,
        default=proxy_config.enable_C_level_observer,
        help="Enable the observer at the C level",
    )
    args = parser.parse_args()

    # set up logging
    if args.debug_mode:
        logging.basicConfig(level=logging.DEBUG)
        os.environ["ML_DAIKON_DEBUG"] = "1"
    else:
        logging.basicConfig(level=logging.INFO)

    logger = logging.getLogger(__name__)

    START_TIME = datetime.datetime.now()

    output_dir = args.output_dir
    if not args.output_dir:
        pyfile_basename = os.path.basename(args.pyscript).split(".")[0]
        # get also the versions of the modules specified in `-t`
        modules = args.modules_to_instr
        modules_and_versions = []
        for module in modules:
            try:
                # this may not work if the module is not installed (e.g. only used locally)
                version = (
                    os.popen(f"pip show {module} | grep Version")
                    .read()
                    .strip()
                    .split(": ")[1]
                )
            except Exception as e:
                logger.warning(f"Could not get version of module {module}: {e}")
                version = "unknown"
            modules_and_versions.append(f"{module}_{version}")
        # sort the modules and versions
        modules_and_versions.sort()
        output_dir = f"mldaikon_run_{pyfile_basename}_{'_'.join(modules_and_versions)}_{START_TIME.strftime('%Y-%m-%d_%H-%M-%S')}"

    # change output_dir to absolute path
    output_dir = os.path.abspath(output_dir)

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    dump_env(output_dir)

    funcs_of_inv_interest = None
    if args.invariants is not None:
        invariants = read_inv_file(args.invariants)
        funcs_of_inv_interest = get_list_of_funcs_from_invariants(invariants)

    # set up proxy class configuration
    if args.proxy_module:
        disable_proxy_class = False
    else:
        disable_proxy_class = True

    # set up adjusted proxy_config
    proxy_basic_config: dict[str, int | bool | str] = {}
    if proxy_config.disable_proxy_class != disable_proxy_class:
        proxy_basic_config["disable_proxy_class"] = disable_proxy_class
    for configs in [
        "proxy_update_limit",
        "profiling",
        "debug_mode",
        "enable_C_level_observer",
    ]:
        if getattr(proxy_config, configs) != getattr(args, configs):
            proxy_basic_config[configs] = getattr(args, configs)
            print(f"Setting {configs} to {getattr(args, configs)}")
    proxy_log_output_dir = os.path.join(output_dir, "proxy_log.json")
    proxy_basic_config["proxy_log_dir"] = proxy_log_output_dir
    print(f"Setting proxy_log_dir to {proxy_log_output_dir}")

    # set up tensor_dump_format
    tensor_dump_format: dict[str, int | bool] = {}
    if args.tensor_dump_format != "hash":
        tensor_dump_format = proxy_config.tensor_dump_format  # type: ignore
        print(f"Setting tensor_dump_format to {args.tensor_dump_format}")
        # set all to False
        for key in tensor_dump_format:
            tensor_dump_format[key] = False
        # set the chosen one to True
        tensor_dump_format[f"dump_tensor_{args.tensor_dump_format}"] = True

    # set up delta_dump_config
    delta_dump_config: dict[str, int | bool] = {}
    for configs in ["delta_dump", "delta_dump_meta_var", "delta_dump_attributes"]:
        if proxy_config.delta_dump_config[configs] != getattr(args, configs):
            delta_dump_config[configs] = getattr(args, configs)
            print(f"Setting {configs} to {getattr(args, configs)}")

    auto_observer_config = proxy_config.auto_observer_config
    # call into the instrumentor
    adjusted_proxy_config: list[dict] = [
        auto_observer_config,  # Ziming: add auto_observer_config for proxy_wrapper
        proxy_basic_config,  # Ziming: add proxy_basic_config for proxy_wrapper
        tensor_dump_format,  # Ziming: add tensor_dump_format for proxy_wrapper
        delta_dump_config,  # Ziming: add delta_dump_config for proxy_wrapper
    ]

    source_code = instrumentor.instrument_file(
        args.pyscript,
        args.modules_to_instr,
        disable_proxy_class,
        not args.disable_scan_proxy_in_args,
        args.use_full_instr,
        funcs_of_inv_interest,
        args.proxy_module,
        adjusted_proxy_config,  # type: ignore
        args.API_dump_stack_trace,
        output_dir,
    )

    # call into the program runner
    program_runner = runner.ProgramRunner(
        source_code,
        args.pyscript,
        args.shscript,
        dry_run=args.only_instr,
        profiling=args.profiling,
        output_dir=output_dir,
    )

    try:
        program_output, return_code = program_runner.run()
    except Exception as e:
        print(f"An error occurred: {e}")

    if return_code != 0:
        logging.error(f"Program exited with code {return_code}, skipping analysis.")

    logger.info("Trace collection done.")

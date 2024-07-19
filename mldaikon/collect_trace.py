import argparse
import logging

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
        "--only-instrument",
        action="store_true",
        help="Only instrument and dump the modified file",
    )
    parser.add_argument(
        "--skip-api", action="store_true", help="Skip API invariant analysis"
    )
    parser.add_argument(
        "--skip-variable", action="store_true", help="Skip variable invariant analysis"
    )
    parser.add_argument(
        "-t",
        "--modules_to_instrument",
        nargs="*",
        help="Modules to be instrumented",
        default=config.INSTR_MODULES_TO_INSTRUMENT,
    )
    parser.add_argument(
        "--wrapped_modules",
        type=list,
        default=config.INCLUDED_WRAP_LIST,
        metavar="Module",
        help="Module to be traced by the proxy wrapper",
    )
    parser.add_argument(
        "--scan_proxy_in_args",
        action="store_true",
        help="Scan the arguments of the function for proxy objects, this will enable the infer engine to understand the relationship between the proxy objects and the functions",
    )
    parser.add_argument(
        "--proxy_log_dir",
        type=str,
        default="proxy_log.log",
        help="Path to the log file of the proxy tracer",
    )
    parser.add_argument(
        "--proxy_module",
        type=str,
        default="None",
        help="The module to be traced by the proxy wrapper",
    )
    parser.add_argument(
        "--profiling",
        type=bool,
        default=proxy_config.profiling,
        help="Enable to do profiling during the training process,"
        "there would be a train_profiling_results.pstats file generated"
        "in the current directory",
    )
    parser.add_argument(
        "--proxy_update_limit",
        type=float,
        default=proxy_config.proxy_update_limit,
        help="The threshold for updating the proxy object",
    )
    parser.add_argument(
        "--debug_mode",
        action="store_true",
        help="Enable debug mode for the program",
    )
    parser.add_argument(
        "-i",
        "--invariants",
        nargs="*",
        help="Invariant files produced by the inference engine. If provided, we will only collect traces for APIs and variables that are related to the invariants. This can be used to speed up the trace collection. HAS TO BE USED WITH --allow_disable_dump for the optimization to work properly.",
        default=None,
    )
    parser.add_argument(
        "--allow_disable_dump",
        action="store_true",
        help="Allow the instrumentor to disable API dump for certain APIs that are not helpful for the invariant analysis",
    )
    parser.add_argument(
        "--tensor_dump_format",
        choices=["hash", "stats", "full", "version"],
        type=str,
        default="hash",
        help="The format for dumping tensors. Choose from 'hash'(default), 'stats', 'full' or 'version'(deprecated).",
    )
    parser.add_argument(
        "--delta_dump",
        type=bool,
        default=proxy_config.delta_dump_config["delta_dump"],
        help="Only dump the changed part of the object",
    )
    parser.add_argument(
        "--delta_dump_meta_var",
        type=bool,
        default=proxy_config.delta_dump_config["delta_dump_meta_var"],
        help="Only dump the changed part of the meta_var",
    )
    parser.add_argument(
        "--delta_dump_attributes",
        type=bool,
        default=proxy_config.delta_dump_config["delta_dump_attributes"],
        help="Only dump the changed part of the attribute",
    )
    parser.add_argument(
        "--enable_C_level_observer",
        type=bool,
        default=proxy_config.enable_C_level_observer,
        help="Enable the observer at the C level",
    )
    args = parser.parse_args()
    config.INCLUDED_WRAP_LIST = args.wrapped_modules

    if args.proxy_module != "None":
        disable_proxy_class = False
    else:
        disable_proxy_class = True

    # set up adjusted proxy_config
    proxy_basic_config: dict[str, int | bool] = {}
    if proxy_config.disable_proxy_class != disable_proxy_class:
        proxy_basic_config["disable_proxy_class"] = disable_proxy_class
    for configs in [
        "proxy_update_limit",
        "profiling",
        "debug_mode",
        "proxy_log_dir",
        "enable_C_level_observer",
    ]:
        if getattr(proxy_config, configs) != getattr(args, configs):
            proxy_basic_config[configs] = getattr(args, configs)
            print(f"Setting {configs} to {getattr(args, configs)}")

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

    # set up logging
    if args.debug_mode:
        logging.basicConfig(level=logging.DEBUG)
    else:
        logging.basicConfig(level=logging.INFO)

    logger = logging.getLogger(__name__)

    funcs_of_inv_interest = None
    if args.invariants is not None:
        invariants = read_inv_file(args.invariants)
        funcs_of_inv_interest = get_list_of_funcs_from_invariants(invariants)

    auto_observer_config = proxy_config.auto_observer_config
    # call into the instrumentor
    adjusted_proxy_config: list[dict[str, int | bool]] = [
        auto_observer_config,  # Ziming: add auto_observer_config for proxy_wrapper
        proxy_basic_config,  # Ziming: add proxy_basic_config for proxy_wrapper
        tensor_dump_format,  # Ziming: add tensor_dump_format for proxy_wrapper
        delta_dump_config,  # Ziming: add delta_dump_config for proxy_wrapper
    ]
    source_code = instrumentor.instrument_file(
        args.pyscript,
        args.modules_to_instrument,
        disable_proxy_class,
        args.scan_proxy_in_args,
        args.allow_disable_dump,
        funcs_of_inv_interest,
        args.proxy_module,
        adjusted_proxy_config,  # type: ignore
    )

    # call into the program runner
    program_runner = runner.ProgramRunner(
        source_code,
        args.pyscript,
        args.shscript,
        dry_run=args.only_instrument,
        profiling=args.profiling,
    )
    try:
        program_output, return_code = program_runner.run()
    except Exception as e:
        print(f"An error occurred: {e}")

    # dump the log
    with open("program_output.txt", "w") as f:
        f.write(program_output)

    if return_code != 0:
        logging.error(f"Program exited with code {return_code}, skipping analysis.")
        exit(return_code)

    logger.info("Trace collection done.")

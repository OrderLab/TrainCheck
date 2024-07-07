import argparse
import logging

import mldaikon.config.config as config
import mldaikon.instrumentor as instrumentor
import mldaikon.proxy_wrapper.config as proxy_config
import mldaikon.runner as runner
from mldaikon.invariant.base_cls import read_inv_file

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
        "--tracer_log_dir",
        type=str,
        default="proxy_log.log",
        help="Path to the log file of the tracer",
    )
    parser.add_argument(
        "--proxy_module",
        type=str,
        default="None",
        help="The module to be traced by the proxy wrapper",
    )
    parser.add_argument(
        "--profiling",
        action="store_true",
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

    args = parser.parse_args()
    config.INCLUDED_WRAP_LIST = args.wrapped_modules

    if args.proxy_module != "None":
        disable_proxy_class = False
    else:
        disable_proxy_class = True

    proxy_config.disable_proxy_class = disable_proxy_class
    proxy_config.proxy_log_dir = args.tracer_log_dir
    proxy_config.proxy_update_limit = args.proxy_update_limit
    proxy_config.profiling = (
        args.profiling
    )  # the profiling has not yet been enacted yet
    proxy_config.debug_mode = args.debug_mode

    # set up logging
    if args.debug_mode:
        logging.basicConfig(level=logging.DEBUG)
    else:
        logging.basicConfig(level=logging.INFO)

    logger = logging.getLogger(__name__)

    if args.invariants is not None:
        invariants = read_inv_file(args.invariants)

    # call into the instrumentor
    source_code = instrumentor.instrument_file(
        args.pyscript,
        args.modules_to_instrument,
        disable_proxy_class,
        args.scan_proxy_in_args,
        args.allow_disable_dump,
        args.proxy_module,
    )

    # call into the program runner
    program_runner = runner.ProgramRunner(
        source_code, args.pyscript, args.shscript, dry_run=args.only_instrument
    )
    program_output, return_code = program_runner.run()

    # dump the log
    with open("program_output.txt", "w") as f:
        f.write(program_output)

    if return_code != 0:
        logging.error(f"Program exited with code {return_code}, skipping analysis.")
        exit(return_code)

    logger.info("Trace collection done.")

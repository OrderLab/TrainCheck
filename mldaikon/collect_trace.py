import argparse
import logging

import mldaikon.config.config as config
import mldaikon.instrumentor as instrumentor
import mldaikon.proxy_wrapper.config as proxy_config
import mldaikon.runner as runner

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
        "--tracer_log_dir",
        type=str,
        default="proxy_log.log",
        help="Path to the log file of the tracer",
    )
    parser.add_argument(
        "--disable_proxy_class",
        action="store_true",
        help="Disable proxy class for tracing",
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

    args = parser.parse_args()
    config.INCLUDED_WRAP_LIST = args.wrapped_modules
    proxy_config.disable_proxy_class = args.disable_proxy_class
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

    # call into the instrumentor
    source_code = instrumentor.instrument_file(
        args.pyscript, args.modules_to_instrument, args.disable_proxy_class
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
        exit()

    logger.info("Trace collection done.")

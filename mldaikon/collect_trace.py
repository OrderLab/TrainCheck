import argparse
import logging

import mldaikon.config.config as config
import mldaikon.instrumentor as instrumentor
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
        "-r",
        "--run-without-analysis",
        action="store_true",
        help="Run the program without analysis",
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
        default=instrumentor.MODULES_TO_INSTRUMENT,
    )
    parser.add_argument(
        "--wrapped_modules",
        type=list,
        default=instrumentor.INCLUDED_WRAP_LIST,
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

    args = parser.parse_args()
    config.INCLUDED_WRAP_LIST = args.wrapped_modules
    config.proxy_log_dir = args.tracer_log_dir

    # set up logging
    logging.basicConfig(level=logging.INFO)

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

    if args.run_without_analysis:
        logging.info("Skipping analysis as requested.")
        exit()

    print(
        "We do not run analysis anymore in a single program run due to the need to analyze multiple traces."
    )
    print("Please run the analysis script separately.")

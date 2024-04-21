import argparse
import logging
import os

import src.instrumentor as instrumentor
import src.runner as runner

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Invariant Finder for ML Pipelines in Python"
    )
    parser.add_argument(
        "--path",
        type=str,
        required=True,
        help="Path to the main file of the pipeline to be analyzed",
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

    args = parser.parse_args()

    # set up logging
    logging.basicConfig(level=logging.INFO)

    # call into the instrumentor
    source_code, log_file = instrumentor.instrument_file(
        args.path, args.modules_to_instrument
    )

    if args.only_instrument:
        print(source_code)
        exit()

    # call into the program runner
    program_runner = runner.ProgramRunner(
        source_code, os.path.abspath(os.path.dirname(args.path))
    )
    program_output = program_runner.run()

    # dump the log
    with open("program_output.txt", "w") as f:
        f.write(program_output)

    if args.run_without_analysis:
        logging.info(f"Skipping analysis, trace file is at {log_file}")
        exit()

    print(
        "We do not run analysis anymore in a single program run due to the need to analyze multiple traces."
    )
    print("Please run the analysis script separately.")

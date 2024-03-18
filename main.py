import argparse
import json
import logging
import os

import src.instrumentor as instrumentor
import src.invariant.analyzer as analyzer
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
        "--print_instr",
        action="store_true",
        help="print the log related to instrumentation",
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

    with open(log_file, "r") as f:
        log = f.read()
        # ad-hoc preprocessing step to convert trace into a list of events
        trace_lines = [
            analyzer.Event(line.split(":trace:")[-1].strip())
            for line in log.split("\n")
            if line.startswith("INFO:trace:") or line.startswith("ERROR:trace:")
        ]

    # call into the trace analyzer
    trace = analyzer.Trace(trace_lines)
    invariants = trace.analyze(
        analyze_api_invariants=not args.skip_api,
        analyze_variable_invariants=not args.skip_variable,
    )

    def default(o):
        if isinstance(o, set):
            return list(o)
        if isinstance(o, analyzer.Event):
            return o.get_event()
        return o

    # dump the invariants
    with open("invariants.json", "w") as f:
        json.dump(invariants, f, indent=4, default=default)
    # call into the invariant finder
    # invariants = finder.find(trace)

    # dump the invariants
    # dumper.dump(invariants)

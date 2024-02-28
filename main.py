import argparse
import json
import logging


import src.instrumentor as instrumentor
import src.runner as runner
import src.analyzer as analyzer


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
    args = parser.parse_args()

    # set up logging
    logging.basicConfig(level=logging.INFO)

    # call into the instrumentor
    source_code = instrumentor.instrument_file(args.path)

    if args.only_instrument:
        print(source_code)
        exit()

    # call into the program runner
    program_runner = runner.ProgramRunner(source_code)
    log = program_runner.run()

    # dump the log
    with open("log.txt", "w") as f:
        f.write(log)

    # ad-hoc preprocessing step to convert trace into a list of events
    trace_lines = [
        analyzer.Event(l.split(":trace:")[-1].strip())
        for l in log.split("\n")
        if l.startswith("INFO:trace:") or l.startswith("ERROR:trace:")
    ]

    # call into the trace analyzer
    trace = analyzer.Trace(trace_lines)
    invariants = trace.analyze()

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

import argparse
import json
import logging
import torch

import src.analyzer as analyzer
import src.instrumentor as instrumentor
import src.runner as runner
import src.config as config
from src.instrumentor.tracer import new_wrapper, get_all_subclasses

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
        "-t",
        "--modules_to_instrument",
        nargs="*",
        help="Modules to be instrumented",
        default=instrumentor.MODULES_TO_INSTRUMENT,
    )
    
    parser.add_argument(
        '--wrapped_modules', 
        type=list, 
        default=instrumentor.INCLUDED_WRAP_LIST, 
        metavar='Module', 
        help = 'Module to be traced by the proxy wrapper')
    
    parser.add_argument(
        "--tracer_log_dir",
        type=str,
        default="proxy_log.log",
        help="Path to the log file of the tracer"
    )
    

    args = parser.parse_args()
    config.INCLUDED_WRAP_LIST = args.wrapped_modules
    config.proxy_log_dir = args.tracer_log_dir

    # set up logging
    logging.basicConfig(level=logging.INFO)

    # call into the instrumentor
    source_code, log_file = instrumentor.instrument_file(
        args.path, args.modules_to_instrument
    )

    if args.only_instrument:
        print(source_code)
        exit()
        
    # Add new_wrapper
    


    # call into the program runner
    program_runner = runner.ProgramRunner(source_code)
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

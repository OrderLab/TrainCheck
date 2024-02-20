import argparse
import src.instrumentor as instrumentor
import src.runner as runner

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Invariant Finder for ML Pipelines in Python")
    parser.add_argument("--path", type=str, required=True, help="Path to the main file of the pipeline to be analyzed")
    parser.add_argument("--only-instrument", action="store_true", help="Only instrument and dump the modified file")
    args = parser.parse_args()

    # call into the instrumentor
    source_code = instrumentor.instrument_file(args.path)
    
    # please look into this https://github.com/harshitandro/Python-Instrumentation for possible implementation

    if args.only_instrument:
        print(source_code)
        exit()

    # call into the program runner
    program_runner = runner.ProgramRunner(source_code)
    trace = program_runner.run()
    print(trace)

    # call into the invariant finder
    # invariants = finder.find(trace)

    # dump the invariants
    # dumper.dump(invariants)

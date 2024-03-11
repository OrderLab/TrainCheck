# regression_test.py
import logging
import re
import subprocess

import src.analyzer as analyzer
import src.instrumentor as instrumentor
import src.runner as runner

import torch.cuda

"""
Run default mnist.py and instrumented mnist.py, compare the accuracy of the two.
"""
def main():
    TEST_LOSS_MAX_INTERMEDIATE_ALLOWABLE_DIFF = 0.00011
    TEST_LOSS_FINAL_MAX_ALLOWABLE_DIFF = 0.01

    PREFIX_LENGTH = len("Test set: Average loss: ")
    LOSS_SIG_FIGS = len("0.0261")
    # run mnist.py without any instrumentation, capture the accuracy parameter.
    
    PATH = "mnist.py"

    # set up logging
    logging.basicConfig(level=logging.INFO)

    # call into the instrumentor
    source_code, _log_file = instrumentor.instrument_file(
        PATH, instrumentor.MODULES_TO_INSTRUMENT
    )

    print(f"Source code instrumentation finished. Running the instrumented program with CUDA {'on' if torch.cuda.is_available() else 'off'}")

    # call into the program runner
    program_runner = runner.ProgramRunner(source_code)
    instrumented_program_output = program_runner.run()

    # dump the log
    with open("test_instrumented_program_output.txt", "w") as f:
        f.write(instrumented_program_output)

    print(f"Finished running the instrumented program.")
    normal_program_output = run_mnist_normally()

    # dump the log
    with open("test_normal_program_output.txt", "w") as f:
        f.write(normal_program_output)

    # Uncomment if you've already run mnist. 
    # with open("program_output.txt", "r") as f:
    #     instrumented_program_output = f.read()

    # with open("unmodified_output.txt", "r") as f:
    #     normal_program_output = f.read()
 
    # grab accuracy from the output (i.e. Test set: Average loss: 0.0261, Accuracy: 9921/10000 (99%))
    instrumented_accuracy_indices = get_accuracy_indices(instrumented_program_output)
    normal_accuracy_indices = get_accuracy_indices(normal_program_output)

    if (len(instrumented_accuracy_indices) != len(normal_accuracy_indices)):
        print("Test failed; different number of epochs.")
        exit()

    print(instrumented_accuracy_indices, normal_accuracy_indices)

    # we're not interested in the invariants, since we're testing that instrumenting a pipeline maintains correctness
    # skip the invariant analysis
    for loop_idx, (idx1, idx2) in enumerate(zip(instrumented_accuracy_indices, normal_accuracy_indices)):
        instrumented_acc = instrumented_program_output[idx1+PREFIX_LENGTH:idx1+PREFIX_LENGTH+LOSS_SIG_FIGS]
        normal_acc = normal_program_output[idx2+PREFIX_LENGTH:idx2+PREFIX_LENGTH+LOSS_SIG_FIGS]
        print(instrumented_acc, normal_acc)
        if (abs(float(instrumented_acc) - float(normal_acc)) > TEST_LOSS_MAX_INTERMEDIATE_ALLOWABLE_DIFF):
            print(f"Training losses significantly different at epoch {loop_idx}, instrumented: {instrumented_acc}, normal: {normal_acc}")
        else:
            print(f"Training losses not significantly different at epoch {loop_idx}, instrumented: {instrumented_acc}, normal: {normal_acc}")

    idx1, idx2 = instrumented_accuracy_indices[-1], normal_accuracy_indices[-1]
    instrumented_acc = instrumented_program_output[idx1+PREFIX_LENGTH:idx1+PREFIX_LENGTH+LOSS_SIG_FIGS]
    normal_acc = normal_program_output[idx2+PREFIX_LENGTH:idx2+PREFIX_LENGTH+LOSS_SIG_FIGS]
    print(instrumented_acc, normal_acc)
    if (abs(float(instrumented_acc) - float(normal_acc)) > TEST_LOSS_FINAL_MAX_ALLOWABLE_DIFF):
        print("Test failed: Final loss significantly different.")
    else:
        print("Test passed.")

    print(f"Final losses: instrumented: {instrumented_acc}, normal: {normal_acc}")

def get_accuracy_indices(output):
    ACCURACY_STRING = "Test set: Average loss: "
    return [match.start() for match in re.finditer(ACCURACY_STRING, output)]
    
def run_mnist_normally():
    MNIST_FILENAME = "mnist.py"
    process = subprocess.Popen(
            ["python3", MNIST_FILENAME],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
    )
    out, err = process.communicate()

    if process.returncode != 0:
        raise Exception(err.decode("utf-8"))
    
    return out.decode("utf-8")

if __name__ == "__main__":
    main()
import pytest
import subprocess

# get file location
def get_file_location():
    return __file__

def run_naive():
    subprocess.run(["python", f"{__file__}/workloads/84911_efficientnet_b0_1_epochs_naive.py"])

def run_naive_instrumented():
    subprocess.run(["python", "-m", "mldaikon.collect_trace", "-p", f"{__file__}/workloads/84911_efficientnet_b0_1_epochs_naive.py"])

def run_sampler_instrumented():
    subprocess.run(["python", "-m", "mldaikon.collect_trace", "-p", f"{__file__}/workloads/84911_efficientnet_b0_1_epochs_sampler.py"])

def run_proxy_instrumented():
    subprocess.run(["python", "-m", "mldaikon.collect_trace", "-p", f"{__file__}/workloads/84911_efficientnet_b0_1_epochs_proxy.py"])

def run_proxy_instrumented_with_scan_proxy_in_args():
    subprocess.run(["python", "-m", "mldaikon.collect_trace", "-p", f"{__file__}/workloads/84911_efficientnet_b0_1_epochs_naive.py", "--scan_proxy_in_args"])

def cleanup():
    subprocess.run(["rm", "-rf", f"{__file__}/workloads/__pycache__"])
    subprocess.run(["rm", "-rf", f"{__file__}/workloads/*.pyc"])
    subprocess.run(["rm", "-rf", f"{__file__}/workloads/*.pyi"])
    subprocess.run(["rm", "-rf", f"{__file__}/workloads/*.pyi"])
    subprocess.run(["rm", "-rf", f"{__file__}/workloads/*.json"])
    subprocess.run(["rm", "-rf", f"{__file__}/workloads/*.log"])
    subprocess.run(["rm", "-rf", f"{__file__}/workloads/*.csv"])
    subprocess.run(["rm", "-rf", f"{__file__}/workloads/*.pt"])

def test_naive(benchmark):
    benchmark(run_naive)
    cleanup()

def test_instrumented(benchmark):
    benchmark(run_naive_instrumented)
    cleanup()

def test_sampler_instrumented(benchmark):
    benchmark(run_sampler_instrumented)
    cleanup()

def test_proxy_instrumented(benchmark):
    benchmark(run_proxy_instrumented)
    cleanup()

def test_proxy_instrumented_with_scan_proxy_in_args(benchmark):
    benchmark(run_proxy_instrumented_with_scan_proxy_in_args)
    cleanup()
    
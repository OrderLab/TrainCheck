import os
import subprocess

import pytest  # noqa: F401


# get file location
def get_file_parent_dir():
    return os.path.dirname(os.path.realpath(__file__))


def run_naive():
    res = subprocess.run(
        [
            "python",
            f"{get_file_parent_dir()}/workloads/84911_efficientnet_b0_1_epochs_naive.py",
        ]
    )
    return res


def run_naive_instrumented():
    res = subprocess.run(
        [
            "python",
            "-m",
            "mldaikon.collect_trace",
            "-p",
            f"{get_file_parent_dir()}/workloads/84911_efficientnet_b0_1_epochs_naive.py",
        ]
    )
    return res


def run_sampler_instrumented():
    res = subprocess.run(
        [
            "python",
            "-m",
            "mldaikon.collect_trace",
            "-p",
            f"{get_file_parent_dir()}/workloads/84911_efficientnet_b0_1_epochs_sampler.py",
        ]
    )
    return res


def run_proxy_instrumented():
    res = subprocess.run(
        [
            "python",
            "-m",
            "mldaikon.collect_trace",
            "-p",
            f"{get_file_parent_dir()}/workloads/84911_efficientnet_b0_1_epochs_proxy.py",
        ]
    )
    return res


def run_proxy_instrumented_with_scan_proxy_in_args():
    res = subprocess.run(
        [
            "python",
            "-m",
            "mldaikon.collect_trace",
            "-p",
            f"{get_file_parent_dir()}/workloads/84911_efficientnet_b0_1_epochs_proxy.py",
            "--scan_proxy_in_args",
        ]
    )
    return res


def cleanup():
    subprocess.run(["rm", "-rf", f"{get_file_parent_dir()}/workloads/*.json"])
    subprocess.run(["rm", "-rf", f"{get_file_parent_dir()}/workloads/*.log"])
    subprocess.run(["rm", "-rf", f"{get_file_parent_dir()}/workloads/*.csv"])
    subprocess.run(["rm", "-rf", f"{get_file_parent_dir()}/workloads/*.pt"])
    subprocess.run(["rm", "-rf", f"{get_file_parent_dir()}/workloads/_ml_daikon*.py"])

    subprocess.run(["rm", "-rf", "*.json"])
    subprocess.run(["rm", "-rf", "*.log"])
    subprocess.run(["rm", "-rf", "*.csv"])
    subprocess.run(["rm", "-rf", "*.pt"])


def test_naive(benchmark):
    res = benchmark.pedantic(run_naive, rounds=1, iterations=1)
    assert res.returncode == 0
    # cleanup()


def test_instrumented(benchmark):
    res = benchmark.pedantic(run_naive_instrumented, rounds=1, iterations=1)
    assert res.returncode == 0
    # cleanup()


def test_sampler_instrumented(benchmark):
    res = benchmark.pedantic(run_sampler_instrumented, rounds=1, iterations=1)
    assert res.returncode == 0
    # cleanup()


def test_proxy_instrumented(benchmark):
    res = benchmark.pedantic(run_proxy_instrumented, rounds=1, iterations=1)
    assert res.returncode == 0
    # cleanup()


def test_proxy_instrumented_with_scan_proxy_in_args(benchmark):
    res = benchmark.pedantic(
        run_proxy_instrumented_with_scan_proxy_in_args, rounds=1, iterations=1
    )
    assert res.returncode == 0
    # cleanup()

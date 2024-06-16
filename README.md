# ML-DAIKON

[![Pre-commit checks](https://github.com/OrderLab/ml-daikon/actions/workflows/pre-commit-checks.yml/badge.svg)](https://github.com/OrderLab/ml-daikon/actions/workflows/pre-commit-checks.yml)

The analysis compoent is not completely functional as we are refactoring the codebase to make the workflow less ad-hoc to the bugs that we have found. If you want to use the e2e workflow and reproduce the results for DS-1801 and PyTorch-FORUM84911, please swtich to commit [600ce9b](https://github.com/Essoz/ml-daikon-eecs598/commit/600ce9b0fe2e6fd97068d9f20002f26fb1a0303b).

## Instrumentator Usage
ML-Daikon performs automatic instrumentation of programs and supports out-of-tree execution. To use the instrumentor, please install mldaikon as a pip package in the desired python environment where the example pipeline should be run in.

To install the instrumentor:
```shell
git clone git@github.com:OrderLab/ml-daikon.git
cd ml-daikon
pip3 install -e .
```
To use the instrumentor:
```shell
python3 -m mldaikon.collect_trace \
  -p <path to your python script> \
  -s <optional path to sh script that invokes the python script> \
  -t [names of the module to be instrumented, e.g. torch, megatron] \
  --disable_proxy_class <optional flag to disable automatic variable instrumentation> \
  --instrument-only <optional flag to only instrument the files without running it>
```

After executing the above command, you can find the dumped traces and the instrumented program at the parent folder of your python script. The instrumented script will have the prefix `_ml_daikon_`.

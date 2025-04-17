
# ML-DAIKON
[![Pre-commit checks](https://github.com/OrderLab/traincheck/actions/workflows/pre-commit-checks.yml/badge.svg)](https://github.com/OrderLab/traincheck/actions/workflows/pre-commit-checks.yml)

Instrumentor Performance Benchmark Results: http://orderlab.io/traincheck/dev/bench/

## Instrumentator Usage
ML-Daikon performs automatic instrumentation of programs and supports out-of-tree execution. To use the instrumentor, please install traincheck as a pip package in the desired python environment where the example pipeline should be run in.

To install the instrumentor:
```shell
git clone git@github.com:OrderLab/traincheck.git
cd traincheck
pip3 install -e .
conda install cudatoolkit
```

A typical instrumentor invocation looks like
```bash
python3 -m traincheck.collect_trace \
  -p <path to your python script> \
  -s <optional path to sh script that invokes the python script> \
  -t [names of the module to be instrumented, e.g. torch, megatron] \ # `torch` is the default value here so you probably don't need to set it
  --scan_proxy_in_args \ # dynamic analysis for APIContainRelation in 84911, keep it on
  --allow_disable_dump \ # skip instrumentation for functions in modules specified in config.WRAP_WITHOUT_DUMP, keep it on for instrumentor overhead, inform @Essoz if you need those functions for invariant inference
  -d # enabling debug logging, if you are not debugging the trace collector, you probably don't need it
```

The instrumentor will dump the collected trace to the folder where you invoked the command. There should be one trace per thread and the names of trace files follow the pattern:
```bash
_traincheck_<pyscript-file-name>_traincheck_trace_API_<time-of-instrumentor-invocation>_<process-id>_<thread-id>.log
```
After execution completion, you can also look at `program_output.txt` for the stdout and stderr of the pipeline being executed.

## Infer Engine Usage

```bash
python3 -m traincheck.infer_engine \
  -t <path to your trace files> \
  -d \ # enable debug logging 
  -o invariant.json \ # name of the file to dump the inferred invariants to
```

There are two other arguments that you might need.
```bash
--disable_precond_sampling \ # by default we enable sampling of examples to be used in precondition inference when the number of examples exceeds 10000. Sampling might cause us to lose information and you can disable this behavior by setting this flag.
--precond_sampling_threshold \ # the default threshold to sample examples is 10000, change this if you need to
```

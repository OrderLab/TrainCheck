
# Improvement on Logging
1. Create separate threads for each logger in each process (pid)
2. Realize asynchronous logging:   
Record trace in a buffer. If the buffer size attains a certain length (currently set to 10000), the corresponding logger will log into file once. The aim is to avoid frequently switching between logging threads and pipeline threads to decrease large overhead caused by <method 'acquire' of '_thread.lock' objects>.

NOTE: Since the encoding part is not accomplished yet, we need to add the following code manually to the instrumented pipeline after the ```main()``` function:  
```python
if __name__ == '__main__':
    main()
    from mldaikon.instrumentor.tracer import get_dicts
    trace_API_loggers, trace_VAR_loggers = get_dicts()

    for log_queue, _, _ in trace_API_loggers.values():
        log_queue.join()

    for log_queue, _, _ in trace_API_loggers.values():
        log_queue.put(None)

    for _, log_thread, _ in trace_API_loggers.values():
        log_thread.join()

    for log_queue, _, _ in trace_VAR_loggers.values():
        log_queue.join()

    for log_queue, _, _ in trace_VAR_loggers.values():
        log_queue.put(None)

    for _, log_thread, _ in trace_VAR_loggers.values():
        log_thread.join()
```

# Evaluate new logging on image-classification pipeline (7 epochs)
## Total runtime comparison
Total time for sync profile: 741.49 seconds
Total time for async profile: 436.29 seconds
## Files runtime comparison
![alt text](sync_vs_async_files.png)
## Function calls runtime comparison
![alt text](sync_vs_async_functions.png)


# ML-DAIKON
[![Pre-commit checks](https://github.com/OrderLab/ml-daikon/actions/workflows/pre-commit-checks.yml/badge.svg)](https://github.com/OrderLab/ml-daikon/actions/workflows/pre-commit-checks.yml)

[![Instrumentor Benchmark](https://github.com/OrderLab/ml-daikon/actions/workflows/bench-instr-e2e.yml/badge.svg)](https://github.com/OrderLab/ml-daikon/actions/workflows/bench-instr-e2e.yml)

Instrumentor Performance Benchmark Results: http://orderlab.io/ml-daikon/dev/bench/

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

window.BENCHMARK_DATA = {
  "lastUpdate": 1719693956856,
  "repoUrl": "https://github.com/OrderLab/ml-daikon",
  "entries": {
    "Python Benchmark with pytest-benchmark": [
      {
        "commit": {
          "author": {
            "name": "OrderLab",
            "username": "OrderLab"
          },
          "committer": {
            "name": "OrderLab",
            "username": "OrderLab"
          },
          "id": "464ba3a044544710200fe68841394f128cb29133",
          "message": "Add Automated Instrumentor Benchmark and Web UI on Github Action",
          "timestamp": "2024-06-29T18:58:19Z",
          "url": "https://github.com/OrderLab/ml-daikon/pull/42/commits/464ba3a044544710200fe68841394f128cb29133"
        },
        "date": 1719693956183,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/bench_instrumentor/bench.py::test_naive",
            "value": 0.08421249894882198,
            "unit": "iter/sec",
            "range": "stddev: 0",
            "extra": "mean: 11.874721834436059 sec\nrounds: 1"
          },
          {
            "name": "tests/bench_instrumentor/bench.py::test_instrumented",
            "value": 0.09189879556125159,
            "unit": "iter/sec",
            "range": "stddev: 0",
            "extra": "mean: 10.88153543137014 sec\nrounds: 1"
          },
          {
            "name": "tests/bench_instrumentor/bench.py::test_sampler_instrumented",
            "value": 0.06436419194837867,
            "unit": "iter/sec",
            "range": "stddev: 0",
            "extra": "mean: 15.536589052528143 sec\nrounds: 1"
          },
          {
            "name": "tests/bench_instrumentor/bench.py::test_proxy_instrumented",
            "value": 0.04588191661299901,
            "unit": "iter/sec",
            "range": "stddev: 0",
            "extra": "mean: 21.795079059898853 sec\nrounds: 1"
          },
          {
            "name": "tests/bench_instrumentor/bench.py::test_proxy_instrumented_with_scan_proxy_in_args",
            "value": 0.04556762785136451,
            "unit": "iter/sec",
            "range": "stddev: 0",
            "extra": "mean: 21.945403944700956 sec\nrounds: 1"
          }
        ]
      }
    ]
  }
}
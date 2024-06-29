window.BENCHMARK_DATA = {
  "lastUpdate": 1719698891207,
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
      },
      {
        "commit": {
          "author": {
            "email": "31838999+Essoz@users.noreply.github.com",
            "name": "Yuxuan",
            "username": "Essoz"
          },
          "committer": {
            "email": "noreply@github.com",
            "name": "GitHub",
            "username": "web-flow"
          },
          "distinct": true,
          "id": "77ba205c3d295430abd33164f5a3d30891846bba",
          "message": "Merge pull request #42 from OrderLab/testing\n\nAdd Automated Instrumentor Benchmark and Web UI on Github Action",
          "timestamp": "2024-06-29T17:13:00-04:00",
          "tree_id": "8861062c3877bc797e9189dc6e82cb97d9a372e7",
          "url": "https://github.com/OrderLab/ml-daikon/commit/77ba205c3d295430abd33164f5a3d30891846bba"
        },
        "date": 1719698890667,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/bench_instrumentor/bench.py::test_naive",
            "value": 0.009972164194427516,
            "unit": "iter/sec",
            "range": "stddev: 0.24832530484507787",
            "extra": "mean: 100.2791350506246 sec\nrounds: 5"
          },
          {
            "name": "tests/bench_instrumentor/bench.py::test_instrumented",
            "value": 0.006977483389670962,
            "unit": "iter/sec",
            "range": "stddev: 0.5111549269277204",
            "extra": "mean: 143.3181484144181 sec\nrounds: 5"
          },
          {
            "name": "tests/bench_instrumentor/bench.py::test_sampler_instrumented",
            "value": 0.00556672646468854,
            "unit": "iter/sec",
            "range": "stddev: 0.7230255585765246",
            "extra": "mean: 179.63878885433078 sec\nrounds: 5"
          },
          {
            "name": "tests/bench_instrumentor/bench.py::test_proxy_instrumented",
            "value": 0.043738394553805036,
            "unit": "iter/sec",
            "range": "stddev: 0.06408093023464466",
            "extra": "mean: 22.86320771947503 sec\nrounds: 5"
          },
          {
            "name": "tests/bench_instrumentor/bench.py::test_proxy_instrumented_with_scan_proxy_in_args",
            "value": 0.042430519671937277,
            "unit": "iter/sec",
            "range": "stddev: 0.09274606014716244",
            "extra": "mean: 23.56794137172401 sec\nrounds: 5"
          }
        ]
      }
    ]
  }
}
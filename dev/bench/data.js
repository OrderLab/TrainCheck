window.BENCHMARK_DATA = {
  "lastUpdate": 1719901959122,
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
      },
      {
        "commit": {
          "author": {
            "email": "jyuxuan@umich.edu",
            "name": "Yuxuan Jiang",
            "username": "Essoz"
          },
          "committer": {
            "email": "jyuxuan@umich.edu",
            "name": "Yuxuan Jiang",
            "username": "Essoz"
          },
          "distinct": true,
          "id": "6c93412d5e0342246cb769cb0838241b08516010",
          "message": "fix: proxy workload",
          "timestamp": "2024-06-30T23:06:56-04:00",
          "tree_id": "30cba46d4cb4bfb0e23ee9ade00c40bb41293bfc",
          "url": "https://github.com/OrderLab/ml-daikon/commit/6c93412d5e0342246cb769cb0838241b08516010"
        },
        "date": 1719809988066,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/bench_instrumentor/bench.py::test_naive",
            "value": 0.01001735208348028,
            "unit": "iter/sec",
            "range": "stddev: 0.12166993643127737",
            "extra": "mean: 99.8267797384411 sec\nrounds: 5"
          },
          {
            "name": "tests/bench_instrumentor/bench.py::test_instrumented",
            "value": 0.006949932793378856,
            "unit": "iter/sec",
            "range": "stddev: 0.7270563453681252",
            "extra": "mean: 143.88628346920012 sec\nrounds: 5"
          },
          {
            "name": "tests/bench_instrumentor/bench.py::test_sampler_instrumented",
            "value": 0.005563804368109655,
            "unit": "iter/sec",
            "range": "stddev: 1.1328957350251077",
            "extra": "mean: 179.73313471116126 sec\nrounds: 5"
          },
          {
            "name": "tests/bench_instrumentor/bench.py::test_proxy_instrumented",
            "value": 0.0037693359598543427,
            "unit": "iter/sec",
            "range": "stddev: 1.6439776937493558",
            "extra": "mean: 265.29871856756506 sec\nrounds: 5"
          },
          {
            "name": "tests/bench_instrumentor/bench.py::test_proxy_instrumented_with_scan_proxy_in_args",
            "value": 0.0036808128869191864,
            "unit": "iter/sec",
            "range": "stddev: 1.1861887441236765",
            "extra": "mean: 271.6791183691472 sec\nrounds: 5"
          }
        ]
      },
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
          "id": "d04c22d35d8e32514d8ee64fbaead0425d58c8ef",
          "message": "proxy-v3.5.1 (fix)",
          "timestamp": "2024-07-01T03:07:03Z",
          "url": "https://github.com/OrderLab/ml-daikon/pull/48/commits/d04c22d35d8e32514d8ee64fbaead0425d58c8ef"
        },
        "date": 1719845534008,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/bench_instrumentor/bench.py::test_naive",
            "value": 0.009946104015839712,
            "unit": "iter/sec",
            "range": "stddev: 0.13226977380561683",
            "extra": "mean: 100.54188035912811 sec\nrounds: 5"
          },
          {
            "name": "tests/bench_instrumentor/bench.py::test_instrumented",
            "value": 0.006941913937372741,
            "unit": "iter/sec",
            "range": "stddev: 0.20014150448625298",
            "extra": "mean: 144.05249172225595 sec\nrounds: 5"
          },
          {
            "name": "tests/bench_instrumentor/bench.py::test_sampler_instrumented",
            "value": 0.0055254407930351145,
            "unit": "iter/sec",
            "range": "stddev: 0.7900041848906488",
            "extra": "mean: 180.9810361664742 sec\nrounds: 5"
          },
          {
            "name": "tests/bench_instrumentor/bench.py::test_proxy_instrumented",
            "value": 0.0037744157346451352,
            "unit": "iter/sec",
            "range": "stddev: 2.268670420026609",
            "extra": "mean: 264.9416678775102 sec\nrounds: 5"
          },
          {
            "name": "tests/bench_instrumentor/bench.py::test_proxy_instrumented_with_scan_proxy_in_args",
            "value": 0.0036801114913124705,
            "unit": "iter/sec",
            "range": "stddev: 1.0911667171854353",
            "extra": "mean: 271.7308979254216 sec\nrounds: 5"
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
          "id": "eb7922b3600d4fb40bc219b69f0c355e1eade27b",
          "message": "Merge pull request #48 from OrderLab/proxy-v3.5.1\n\nproxy-v3.5.1 (fix)",
          "timestamp": "2024-07-02T00:38:51-04:00",
          "tree_id": "391654eb39e3e69ab89119ec045161bc36304d09",
          "url": "https://github.com/OrderLab/ml-daikon/commit/eb7922b3600d4fb40bc219b69f0c355e1eade27b"
        },
        "date": 1719901958591,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/bench_instrumentor/bench.py::test_naive",
            "value": 0.00997173491893096,
            "unit": "iter/sec",
            "range": "stddev: 0.111907596954465",
            "extra": "mean: 100.28345199003816 sec\nrounds: 5"
          },
          {
            "name": "tests/bench_instrumentor/bench.py::test_instrumented",
            "value": 0.006963228530394523,
            "unit": "iter/sec",
            "range": "stddev: 0.42018437077354037",
            "extra": "mean: 143.61154393181204 sec\nrounds: 5"
          },
          {
            "name": "tests/bench_instrumentor/bench.py::test_sampler_instrumented",
            "value": 0.005506073126797726,
            "unit": "iter/sec",
            "range": "stddev: 1.336021591102536",
            "extra": "mean: 181.61763873659075 sec\nrounds: 5"
          },
          {
            "name": "tests/bench_instrumentor/bench.py::test_proxy_instrumented",
            "value": 0.0037617431679114978,
            "unit": "iter/sec",
            "range": "stddev: 2.4182922799695814",
            "extra": "mean: 265.8342038154602 sec\nrounds: 5"
          },
          {
            "name": "tests/bench_instrumentor/bench.py::test_proxy_instrumented_with_scan_proxy_in_args",
            "value": 0.003677882958477558,
            "unit": "iter/sec",
            "range": "stddev: 0.9755622892447897",
            "extra": "mean: 271.89554732702675 sec\nrounds: 5"
          }
        ]
      }
    ]
  }
}
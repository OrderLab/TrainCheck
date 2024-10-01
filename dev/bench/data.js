window.BENCHMARK_DATA = {
  "lastUpdate": 1727804553006,
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
          "id": "64ecf2a85aaa0c5ca77de3f686312b5fadb5f2a1",
          "message": "fix: io on closed file",
          "timestamp": "2024-07-02T01:44:19-04:00",
          "tree_id": "f570a4b5e63e3e948a0a409388c9dd7b92ef6cb8",
          "url": "https://github.com/OrderLab/ml-daikon/commit/64ecf2a85aaa0c5ca77de3f686312b5fadb5f2a1"
        },
        "date": 1719908736729,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/bench_instrumentor/bench.py::test_naive",
            "value": 0.009963016307733918,
            "unit": "iter/sec",
            "range": "stddev: 0.18092948178341922",
            "extra": "mean: 100.37120979353786 sec\nrounds: 5"
          },
          {
            "name": "tests/bench_instrumentor/bench.py::test_instrumented",
            "value": 0.0069436311091568914,
            "unit": "iter/sec",
            "range": "stddev: 0.616954839199216",
            "extra": "mean: 144.01686729602517 sec\nrounds: 5"
          },
          {
            "name": "tests/bench_instrumentor/bench.py::test_sampler_instrumented",
            "value": 0.0055212036417489644,
            "unit": "iter/sec",
            "range": "stddev: 0.5394775715650449",
            "extra": "mean: 181.11992690116168 sec\nrounds: 5"
          },
          {
            "name": "tests/bench_instrumentor/bench.py::test_proxy_instrumented",
            "value": 0.0037764004170838276,
            "unit": "iter/sec",
            "range": "stddev: 2.344441617951901",
            "extra": "mean: 264.8024281207472 sec\nrounds: 5"
          },
          {
            "name": "tests/bench_instrumentor/bench.py::test_proxy_instrumented_with_scan_proxy_in_args",
            "value": 0.003676893440636509,
            "unit": "iter/sec",
            "range": "stddev: 2.1259508412321524",
            "extra": "mean: 271.96871928572654 sec\nrounds: 5"
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
          "id": "c1ac73ad34f71006b03bf3eab69a7ff2cc0387b8",
          "message": "Proxy v4.0",
          "timestamp": "2024-07-02T05:44:31Z",
          "url": "https://github.com/OrderLab/ml-daikon/pull/50/commits/c1ac73ad34f71006b03bf3eab69a7ff2cc0387b8"
        },
        "date": 1719972290656,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/bench_instrumentor/bench.py::test_naive",
            "value": 0.009933352486513267,
            "unit": "iter/sec",
            "range": "stddev: 0.37287992567245876",
            "extra": "mean: 100.67094682864845 sec\nrounds: 5"
          },
          {
            "name": "tests/bench_instrumentor/bench.py::test_instrumented",
            "value": 0.0069525358766261616,
            "unit": "iter/sec",
            "range": "stddev: 0.5251206365747184",
            "extra": "mean: 143.83241133093833 sec\nrounds: 5"
          },
          {
            "name": "tests/bench_instrumentor/bench.py::test_sampler_instrumented",
            "value": 0.0055462629018990664,
            "unit": "iter/sec",
            "range": "stddev: 0.24844906641939712",
            "extra": "mean: 180.3015864353627 sec\nrounds: 5"
          },
          {
            "name": "tests/bench_instrumentor/bench.py::test_proxy_instrumented",
            "value": 0.0037741327818138523,
            "unit": "iter/sec",
            "range": "stddev: 2.6684661440412065",
            "extra": "mean: 264.96153098233043 sec\nrounds: 5"
          },
          {
            "name": "tests/bench_instrumentor/bench.py::test_proxy_instrumented_with_scan_proxy_in_args",
            "value": 0.003701397022253854,
            "unit": "iter/sec",
            "range": "stddev: 2.157875909833509",
            "extra": "mean: 270.16826187185944 sec\nrounds: 5"
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
          "id": "276eed2a43edc5a00431c2284decc34d55446ec8",
          "message": "Proxy v4.0",
          "timestamp": "2024-07-02T05:44:31Z",
          "url": "https://github.com/OrderLab/ml-daikon/pull/50/commits/276eed2a43edc5a00431c2284decc34d55446ec8"
        },
        "date": 1719980755712,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/bench_instrumentor/bench.py::test_naive",
            "value": 0.009961082079537322,
            "unit": "iter/sec",
            "range": "stddev: 0.36693654101803475",
            "extra": "mean: 100.39069972671568 sec\nrounds: 5"
          },
          {
            "name": "tests/bench_instrumentor/bench.py::test_instrumented",
            "value": 0.006940322991998823,
            "unit": "iter/sec",
            "range": "stddev: 0.6502664104248776",
            "extra": "mean: 144.08551318906248 sec\nrounds: 5"
          },
          {
            "name": "tests/bench_instrumentor/bench.py::test_sampler_instrumented",
            "value": 0.005539873580668231,
            "unit": "iter/sec",
            "range": "stddev: 0.5270807627858027",
            "extra": "mean: 180.50953427702188 sec\nrounds: 5"
          },
          {
            "name": "tests/bench_instrumentor/bench.py::test_proxy_instrumented",
            "value": 0.0037913854518895848,
            "unit": "iter/sec",
            "range": "stddev: 2.071699617720397",
            "extra": "mean: 263.7558255918324 sec\nrounds: 5"
          },
          {
            "name": "tests/bench_instrumentor/bench.py::test_proxy_instrumented_with_scan_proxy_in_args",
            "value": 0.0037009166456475374,
            "unit": "iter/sec",
            "range": "stddev: 1.1172931342405505",
            "extra": "mean: 270.2033295389265 sec\nrounds: 5"
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
          "id": "2919c91d71e903b47f0df1ae3f3d6ba8e7c80c58",
          "message": "Proxy v4.0",
          "timestamp": "2024-07-02T05:44:31Z",
          "url": "https://github.com/OrderLab/ml-daikon/pull/50/commits/2919c91d71e903b47f0df1ae3f3d6ba8e7c80c58"
        },
        "date": 1720053903416,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/bench_instrumentor/bench.py::test_naive",
            "value": 0.009955811889773323,
            "unit": "iter/sec",
            "range": "stddev: 0.27787722195797665",
            "extra": "mean: 100.44384235776961 sec\nrounds: 5"
          },
          {
            "name": "tests/bench_instrumentor/bench.py::test_instrumented",
            "value": 0.006948147887019494,
            "unit": "iter/sec",
            "range": "stddev: 0.31222721952749566",
            "extra": "mean: 143.92324634715914 sec\nrounds: 5"
          },
          {
            "name": "tests/bench_instrumentor/bench.py::test_sampler_instrumented",
            "value": 0.005533610631706866,
            "unit": "iter/sec",
            "range": "stddev: 0.4168238295934219",
            "extra": "mean: 180.7138352435082 sec\nrounds: 5"
          },
          {
            "name": "tests/bench_instrumentor/bench.py::test_proxy_instrumented",
            "value": 0.003924056978363098,
            "unit": "iter/sec",
            "range": "stddev: 2.1119254645187278",
            "extra": "mean: 254.83829758688807 sec\nrounds: 5"
          },
          {
            "name": "tests/bench_instrumentor/bench.py::test_proxy_instrumented_with_scan_proxy_in_args",
            "value": 0.0038291998544924656,
            "unit": "iter/sec",
            "range": "stddev: 1.4034265434556368",
            "extra": "mean: 261.15116421170535 sec\nrounds: 5"
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
          "id": "7c27b86c9d364eeedd2d6bb5e59f9d8ea77ba1c2",
          "message": "Proxy v4.0",
          "timestamp": "2024-07-02T05:44:31Z",
          "url": "https://github.com/OrderLab/ml-daikon/pull/50/commits/7c27b86c9d364eeedd2d6bb5e59f9d8ea77ba1c2"
        },
        "date": 1720059417376,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/bench_instrumentor/bench.py::test_naive",
            "value": 0.009927170570555149,
            "unit": "iter/sec",
            "range": "stddev: 0.4673050007282574",
            "extra": "mean: 100.73363733328878 sec\nrounds: 5"
          },
          {
            "name": "tests/bench_instrumentor/bench.py::test_instrumented",
            "value": 0.006917709893432962,
            "unit": "iter/sec",
            "range": "stddev: 0.8009082331974461",
            "extra": "mean: 144.55651008859277 sec\nrounds: 5"
          },
          {
            "name": "tests/bench_instrumentor/bench.py::test_sampler_instrumented",
            "value": 0.005524524248217165,
            "unit": "iter/sec",
            "range": "stddev: 1.6523996484922256",
            "extra": "mean: 181.0110617801547 sec\nrounds: 5"
          },
          {
            "name": "tests/bench_instrumentor/bench.py::test_proxy_instrumented",
            "value": 0.005667653034582459,
            "unit": "iter/sec",
            "range": "stddev: 1.0716393619675413",
            "extra": "mean: 176.43987624123693 sec\nrounds: 5"
          },
          {
            "name": "tests/bench_instrumentor/bench.py::test_proxy_instrumented_with_scan_proxy_in_args",
            "value": 0.005489827818023159,
            "unit": "iter/sec",
            "range": "stddev: 0.6421107547460833",
            "extra": "mean: 182.15507537722587 sec\nrounds: 5"
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
          "id": "215e4ab31feeba73a03f31af3ea078ba1266658b",
          "message": "[Feat] Offline Checker",
          "timestamp": "2024-07-02T05:44:31Z",
          "url": "https://github.com/OrderLab/ml-daikon/pull/46/commits/215e4ab31feeba73a03f31af3ea078ba1266658b"
        },
        "date": 1720077710910,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/bench_instrumentor/bench.py::test_naive",
            "value": 0.009958750998487286,
            "unit": "iter/sec",
            "range": "stddev: 0.22948720325042715",
            "extra": "mean: 100.41419854275883 sec\nrounds: 5"
          },
          {
            "name": "tests/bench_instrumentor/bench.py::test_instrumented",
            "value": 0.006942377774529447,
            "unit": "iter/sec",
            "range": "stddev: 0.464785014178642",
            "extra": "mean: 144.0428672246635 sec\nrounds: 5"
          },
          {
            "name": "tests/bench_instrumentor/bench.py::test_sampler_instrumented",
            "value": 0.005529945770063802,
            "unit": "iter/sec",
            "range": "stddev: 0.8795052804731903",
            "extra": "mean: 180.83359974585474 sec\nrounds: 5"
          },
          {
            "name": "tests/bench_instrumentor/bench.py::test_proxy_instrumented",
            "value": 0.003774717548969485,
            "unit": "iter/sec",
            "range": "stddev: 2.2536330282314996",
            "extra": "mean: 264.9204839903861 sec\nrounds: 5"
          },
          {
            "name": "tests/bench_instrumentor/bench.py::test_proxy_instrumented_with_scan_proxy_in_args",
            "value": 0.0036903097876208246,
            "unit": "iter/sec",
            "range": "stddev: 2.0220377776041962",
            "extra": "mean: 270.97996036931875 sec\nrounds: 5"
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
          "id": "7c27b86c9d364eeedd2d6bb5e59f9d8ea77ba1c2",
          "message": "Proxy v4.0",
          "timestamp": "2024-07-02T05:44:31Z",
          "url": "https://github.com/OrderLab/ml-daikon/pull/50/commits/7c27b86c9d364eeedd2d6bb5e59f9d8ea77ba1c2"
        },
        "date": 1720210951197,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/bench_instrumentor/bench.py::test_naive",
            "value": 0.009946474357842471,
            "unit": "iter/sec",
            "range": "stddev: 0.49332111413398283",
            "extra": "mean: 100.53813683353364 sec\nrounds: 5"
          },
          {
            "name": "tests/bench_instrumentor/bench.py::test_instrumented",
            "value": 0.00695576989220487,
            "unit": "iter/sec",
            "range": "stddev: 0.32786779736860483",
            "extra": "mean: 143.76553789116443 sec\nrounds: 5"
          },
          {
            "name": "tests/bench_instrumentor/bench.py::test_sampler_instrumented",
            "value": 0.00552318402694717,
            "unit": "iter/sec",
            "range": "stddev: 0.8169532972444233",
            "extra": "mean: 181.05498479157686 sec\nrounds: 5"
          },
          {
            "name": "tests/bench_instrumentor/bench.py::test_proxy_instrumented",
            "value": 0.005690013039017927,
            "unit": "iter/sec",
            "range": "stddev: 0.43941606391948324",
            "extra": "mean: 175.74652169384063 sec\nrounds: 5"
          },
          {
            "name": "tests/bench_instrumentor/bench.py::test_proxy_instrumented_with_scan_proxy_in_args",
            "value": 0.005514654642592407,
            "unit": "iter/sec",
            "range": "stddev: 0.7689976622204469",
            "extra": "mean: 181.33501820340751 sec\nrounds: 5"
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
          "id": "061fd55aaa08fc3db8fc0d43e1062687f49c4eb9",
          "message": "[Feat] Offline Checker",
          "timestamp": "2024-07-02T05:44:31Z",
          "url": "https://github.com/OrderLab/ml-daikon/pull/46/commits/061fd55aaa08fc3db8fc0d43e1062687f49c4eb9"
        },
        "date": 1720221902325,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/bench_instrumentor/bench.py::test_naive",
            "value": 0.009966900161131503,
            "unit": "iter/sec",
            "range": "stddev: 0.5539675850464626",
            "extra": "mean: 100.33209762647748 sec\nrounds: 5"
          },
          {
            "name": "tests/bench_instrumentor/bench.py::test_instrumented",
            "value": 0.006935327337363554,
            "unit": "iter/sec",
            "range": "stddev: 0.516270946839114",
            "extra": "mean: 144.18930085860194 sec\nrounds: 5"
          },
          {
            "name": "tests/bench_instrumentor/bench.py::test_sampler_instrumented",
            "value": 0.005514791691091753,
            "unit": "iter/sec",
            "range": "stddev: 1.3170800112002017",
            "extra": "mean: 181.33051183335482 sec\nrounds: 5"
          },
          {
            "name": "tests/bench_instrumentor/bench.py::test_proxy_instrumented",
            "value": 0.0037902708943266393,
            "unit": "iter/sec",
            "range": "stddev: 1.9515690330897477",
            "extra": "mean: 263.83338496908544 sec\nrounds: 5"
          },
          {
            "name": "tests/bench_instrumentor/bench.py::test_proxy_instrumented_with_scan_proxy_in_args",
            "value": 0.00366694778168027,
            "unit": "iter/sec",
            "range": "stddev: 2.6217886887419706",
            "extra": "mean: 272.70636494904755 sec\nrounds: 5"
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
          "id": "c1e1b273866c68ccdf0748e560d934f3aeda8bef",
          "message": "[Feat] Offline Checker",
          "timestamp": "2024-07-02T05:44:31Z",
          "url": "https://github.com/OrderLab/ml-daikon/pull/46/commits/c1e1b273866c68ccdf0748e560d934f3aeda8bef"
        },
        "date": 1720228832827,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/bench_instrumentor/bench.py::test_naive",
            "value": 0.009944879188607017,
            "unit": "iter/sec",
            "range": "stddev: 0.22394966995841245",
            "extra": "mean: 100.55426325798035 sec\nrounds: 5"
          },
          {
            "name": "tests/bench_instrumentor/bench.py::test_instrumented",
            "value": 0.006908520078973261,
            "unit": "iter/sec",
            "range": "stddev: 1.5398780271660035",
            "extra": "mean: 144.748801272735 sec\nrounds: 5"
          },
          {
            "name": "tests/bench_instrumentor/bench.py::test_sampler_instrumented",
            "value": 0.005451301209911587,
            "unit": "iter/sec",
            "range": "stddev: 1.781459230154793",
            "extra": "mean: 183.44244089499117 sec\nrounds: 5"
          },
          {
            "name": "tests/bench_instrumentor/bench.py::test_proxy_instrumented",
            "value": 0.00373437925317052,
            "unit": "iter/sec",
            "range": "stddev: 2.37292023856234",
            "extra": "mean: 267.78212179467084 sec\nrounds: 5"
          },
          {
            "name": "tests/bench_instrumentor/bench.py::test_proxy_instrumented_with_scan_proxy_in_args",
            "value": 0.0036821352833227047,
            "unit": "iter/sec",
            "range": "stddev: 2.0021231445183147",
            "extra": "mean: 271.58154794834553 sec\nrounds: 5"
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
          "id": "3e87095d48346592b607ea477e0c0f83cf96109e",
          "message": "[Feat] Offline Checker",
          "timestamp": "2024-07-02T05:44:31Z",
          "url": "https://github.com/OrderLab/ml-daikon/pull/46/commits/3e87095d48346592b607ea477e0c0f83cf96109e"
        },
        "date": 1720235614148,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/bench_instrumentor/bench.py::test_naive",
            "value": 0.009936297476280163,
            "unit": "iter/sec",
            "range": "stddev: 0.3336456196417095",
            "extra": "mean: 100.64110926501453 sec\nrounds: 5"
          },
          {
            "name": "tests/bench_instrumentor/bench.py::test_instrumented",
            "value": 0.006920929523254956,
            "unit": "iter/sec",
            "range": "stddev: 0.6590921014510629",
            "extra": "mean: 144.48926212005318 sec\nrounds: 5"
          },
          {
            "name": "tests/bench_instrumentor/bench.py::test_sampler_instrumented",
            "value": 0.005533770532633296,
            "unit": "iter/sec",
            "range": "stddev: 0.29712560718525904",
            "extra": "mean: 180.7086134314537 sec\nrounds: 5"
          },
          {
            "name": "tests/bench_instrumentor/bench.py::test_proxy_instrumented",
            "value": 0.0037858179339411374,
            "unit": "iter/sec",
            "range": "stddev: 1.7773529382510216",
            "extra": "mean: 264.14371146447957 sec\nrounds: 5"
          },
          {
            "name": "tests/bench_instrumentor/bench.py::test_proxy_instrumented_with_scan_proxy_in_args",
            "value": 0.003696325212042947,
            "unit": "iter/sec",
            "range": "stddev: 1.6348783098147959",
            "extra": "mean: 270.5389657657594 sec\nrounds: 5"
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
          "id": "e0496be0c4d5f8c2879a4c7566aad424130425cb",
          "message": "[Feat] Offline Checker",
          "timestamp": "2024-07-02T05:44:31Z",
          "url": "https://github.com/OrderLab/ml-daikon/pull/46/commits/e0496be0c4d5f8c2879a4c7566aad424130425cb"
        },
        "date": 1720302551927,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/bench_instrumentor/bench.py::test_naive",
            "value": 0.00999665268648202,
            "unit": "iter/sec",
            "range": "stddev: 0.30120950699511595",
            "extra": "mean: 100.03348434343934 sec\nrounds: 5"
          },
          {
            "name": "tests/bench_instrumentor/bench.py::test_instrumented",
            "value": 0.006929322293187519,
            "unit": "iter/sec",
            "range": "stddev: 0.7457087078250977",
            "extra": "mean: 144.3142572518438 sec\nrounds: 5"
          },
          {
            "name": "tests/bench_instrumentor/bench.py::test_sampler_instrumented",
            "value": 0.005509559486340644,
            "unit": "iter/sec",
            "range": "stddev: 1.0231479259144065",
            "extra": "mean: 181.50271405167877 sec\nrounds: 5"
          },
          {
            "name": "tests/bench_instrumentor/bench.py::test_proxy_instrumented",
            "value": 0.0037749701442161545,
            "unit": "iter/sec",
            "range": "stddev: 1.1268598020311387",
            "extra": "mean: 264.90275731906297 sec\nrounds: 5"
          },
          {
            "name": "tests/bench_instrumentor/bench.py::test_proxy_instrumented_with_scan_proxy_in_args",
            "value": 0.0036989716449572505,
            "unit": "iter/sec",
            "range": "stddev: 1.593161539291777",
            "extra": "mean: 270.3454083956778 sec\nrounds: 5"
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
          "id": "2b047959d7de6970f8f6eb892138051b98d4de31",
          "message": "[Feat] Offline Checker",
          "timestamp": "2024-07-02T05:44:31Z",
          "url": "https://github.com/OrderLab/ml-daikon/pull/46/commits/2b047959d7de6970f8f6eb892138051b98d4de31"
        },
        "date": 1720309319907,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/bench_instrumentor/bench.py::test_naive",
            "value": 0.009943315512733083,
            "unit": "iter/sec",
            "range": "stddev: 0.22971094492620836",
            "extra": "mean: 100.57007632106543 sec\nrounds: 5"
          },
          {
            "name": "tests/bench_instrumentor/bench.py::test_instrumented",
            "value": 0.006952677218181896,
            "unit": "iter/sec",
            "range": "stddev: 0.3727881180801793",
            "extra": "mean: 143.82948734983802 sec\nrounds: 5"
          },
          {
            "name": "tests/bench_instrumentor/bench.py::test_sampler_instrumented",
            "value": 0.005527355361925113,
            "unit": "iter/sec",
            "range": "stddev: 1.2166246534266445",
            "extra": "mean: 180.91834783926606 sec\nrounds: 5"
          },
          {
            "name": "tests/bench_instrumentor/bench.py::test_proxy_instrumented",
            "value": 0.0037859402364235246,
            "unit": "iter/sec",
            "range": "stddev: 2.887335845753343",
            "extra": "mean: 264.1351784635335 sec\nrounds: 5"
          },
          {
            "name": "tests/bench_instrumentor/bench.py::test_proxy_instrumented_with_scan_proxy_in_args",
            "value": 0.003697242107586019,
            "unit": "iter/sec",
            "range": "stddev: 2.7398128567565307",
            "extra": "mean: 270.4718736022711 sec\nrounds: 5"
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
          "id": "7b05d0150e3ca3d4c357617f4da1e15ec478001d",
          "message": "Proxy v4.1",
          "timestamp": "2024-07-07T01:25:32Z",
          "url": "https://github.com/OrderLab/ml-daikon/pull/53/commits/7b05d0150e3ca3d4c357617f4da1e15ec478001d"
        },
        "date": 1720327816469,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/bench_instrumentor/bench.py::test_naive",
            "value": 0.009979974491802194,
            "unit": "iter/sec",
            "range": "stddev: 0.2509802308581355",
            "extra": "mean: 100.20065690763295 sec\nrounds: 5"
          },
          {
            "name": "tests/bench_instrumentor/bench.py::test_instrumented",
            "value": 0.00695129998631835,
            "unit": "iter/sec",
            "range": "stddev: 0.27853401935107325",
            "extra": "mean: 143.8579836819321 sec\nrounds: 5"
          },
          {
            "name": "tests/bench_instrumentor/bench.py::test_sampler_instrumented",
            "value": 0.005548984728325938,
            "unit": "iter/sec",
            "range": "stddev: 0.47469094394870487",
            "extra": "mean: 180.2131469015032 sec\nrounds: 5"
          },
          {
            "name": "tests/bench_instrumentor/bench.py::test_proxy_instrumented",
            "value": 0.0056491259715247336,
            "unit": "iter/sec",
            "range": "stddev: 0.47006652284701933",
            "extra": "mean: 177.01853437870741 sec\nrounds: 5"
          },
          {
            "name": "tests/bench_instrumentor/bench.py::test_proxy_instrumented_with_scan_proxy_in_args",
            "value": 0.005484687194296657,
            "unit": "iter/sec",
            "range": "stddev: 0.9534289959334133",
            "extra": "mean: 182.3258035644889 sec\nrounds: 5"
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
          "id": "8edbfd32afc51485a9fc19b8fbe1df352d4f34a1",
          "message": "[Feat] Selective Instrumentation",
          "timestamp": "2024-07-07T18:46:59Z",
          "url": "https://github.com/OrderLab/ml-daikon/pull/54/commits/8edbfd32afc51485a9fc19b8fbe1df352d4f34a1"
        },
        "date": 1720395208403,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/bench_instrumentor/bench.py::test_naive",
            "value": 0.009992250845204642,
            "unit": "iter/sec",
            "range": "stddev: 0.20310966055249158",
            "extra": "mean: 100.07755164392293 sec\nrounds: 5"
          },
          {
            "name": "tests/bench_instrumentor/bench.py::test_instrumented",
            "value": 0.0069385335233802595,
            "unit": "iter/sec",
            "range": "stddev: 0.3945591654667901",
            "extra": "mean: 144.12267327532172 sec\nrounds: 5"
          },
          {
            "name": "tests/bench_instrumentor/bench.py::test_instrumented_with_jit_and_c_tracing_disabled",
            "value": 0.008333320390996445,
            "unit": "iter/sec",
            "range": "stddev: 0.9241719658259282",
            "extra": "mean: 120.00018636994064 sec\nrounds: 5"
          },
          {
            "name": "tests/bench_instrumentor/bench.py::test_sampler_instrumented",
            "value": 0.0011830127321239914,
            "unit": "iter/sec",
            "range": "stddev: 4.158579707004174",
            "extra": "mean: 845.2994400192052 sec\nrounds: 5"
          },
          {
            "name": "tests/bench_instrumentor/bench.py::test_proxy_instrumented",
            "value": 0.005686259479645475,
            "unit": "iter/sec",
            "range": "stddev: 0.2521226374766862",
            "extra": "mean: 175.8625338114798 sec\nrounds: 5"
          },
          {
            "name": "tests/bench_instrumentor/bench.py::test_proxy_instrumented_with_scan_proxy_in_args",
            "value": 0.005488786739229045,
            "unit": "iter/sec",
            "range": "stddev: 1.0761139248242468",
            "extra": "mean: 182.1896254144609 sec\nrounds: 5"
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
          "id": "30f33e6afc82fe0cf16443ffeef1f06d1b9b47cb",
          "message": "[Feat] Selective Instrumentation",
          "timestamp": "2024-07-07T18:46:59Z",
          "url": "https://github.com/OrderLab/ml-daikon/pull/54/commits/30f33e6afc82fe0cf16443ffeef1f06d1b9b47cb"
        },
        "date": 1720407345009,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/bench_instrumentor/bench.py::test_naive",
            "value": 0.009957121981902422,
            "unit": "iter/sec",
            "range": "stddev: 0.21126184334923728",
            "extra": "mean: 100.43062662258744 sec\nrounds: 5"
          },
          {
            "name": "tests/bench_instrumentor/bench.py::test_instrumented",
            "value": 0.006961711156803122,
            "unit": "iter/sec",
            "range": "stddev: 0.45964377950653595",
            "extra": "mean: 143.6428454838693 sec\nrounds: 5"
          },
          {
            "name": "tests/bench_instrumentor/bench.py::test_instrumented_with_jit_and_c_tracing_disabled",
            "value": 0.008367032590446308,
            "unit": "iter/sec",
            "range": "stddev: 0.2600006194671676",
            "extra": "mean: 119.51668517962098 sec\nrounds: 5"
          },
          {
            "name": "tests/bench_instrumentor/bench.py::test_sampler_instrumented",
            "value": 0.0011848998828167853,
            "unit": "iter/sec",
            "range": "stddev: 2.4754464895165396",
            "extra": "mean: 843.953159673512 sec\nrounds: 5"
          },
          {
            "name": "tests/bench_instrumentor/bench.py::test_proxy_instrumented",
            "value": 0.003913257639503658,
            "unit": "iter/sec",
            "range": "stddev: 2.1712158824551673",
            "extra": "mean: 255.54156974107028 sec\nrounds: 5"
          },
          {
            "name": "tests/bench_instrumentor/bench.py::test_proxy_instrumented_with_scan_proxy_in_args",
            "value": 0.003833771018660428,
            "unit": "iter/sec",
            "range": "stddev: 1.5656255135346702",
            "extra": "mean: 260.8397828489542 sec\nrounds: 5"
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
          "id": "d03e6b0dadf6e13db02f1de5d261a4c2b16a5b81",
          "message": "Merge pull request #54 from OrderLab/selective-instrumentation\n\n[Feat] Selective Instrumentation",
          "timestamp": "2024-07-07T21:26:32-04:00",
          "tree_id": "7a1e4caedbf2be30e60507806a598f8212e3e074",
          "url": "https://github.com/OrderLab/ml-daikon/commit/d03e6b0dadf6e13db02f1de5d261a4c2b16a5b81"
        },
        "date": 1720419454769,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/bench_instrumentor/bench.py::test_naive",
            "value": 0.009955408888705438,
            "unit": "iter/sec",
            "range": "stddev: 0.2846324053806978",
            "extra": "mean: 100.44790838621557 sec\nrounds: 5"
          },
          {
            "name": "tests/bench_instrumentor/bench.py::test_instrumented",
            "value": 0.006939465775884177,
            "unit": "iter/sec",
            "range": "stddev: 0.29279918028680674",
            "extra": "mean: 144.10331173837184 sec\nrounds: 5"
          },
          {
            "name": "tests/bench_instrumentor/bench.py::test_instrumented_with_jit_and_c_tracing_disabled",
            "value": 0.008361244702281145,
            "unit": "iter/sec",
            "range": "stddev: 0.526725357954935",
            "extra": "mean: 119.59941798225046 sec\nrounds: 5"
          },
          {
            "name": "tests/bench_instrumentor/bench.py::test_sampler_instrumented",
            "value": 0.0011848708792496668,
            "unit": "iter/sec",
            "range": "stddev: 1.6931710422029584",
            "extra": "mean: 843.9738181710243 sec\nrounds: 5"
          },
          {
            "name": "tests/bench_instrumentor/bench.py::test_proxy_instrumented",
            "value": 0.0039038129695430092,
            "unit": "iter/sec",
            "range": "stddev: 1.6076033506667482",
            "extra": "mean: 256.159812932089 sec\nrounds: 5"
          },
          {
            "name": "tests/bench_instrumentor/bench.py::test_proxy_instrumented_with_scan_proxy_in_args",
            "value": 0.0038277898051158905,
            "unit": "iter/sec",
            "range": "stddev: 1.6210267577745512",
            "extra": "mean: 261.2473649058491 sec\nrounds: 5"
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
          "id": "79780b52229b58096657624035426903c71aea9d",
          "message": "fix: incorrect import path in proxy config",
          "timestamp": "2024-07-11T01:26:43-04:00",
          "tree_id": "2705a8b8bc0f8175f97fa749a3d232154bdeec60",
          "url": "https://github.com/OrderLab/ml-daikon/commit/79780b52229b58096657624035426903c71aea9d"
        },
        "date": 1720687679074,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/bench_instrumentor/bench.py::test_naive",
            "value": 0.01001076927704201,
            "unit": "iter/sec",
            "range": "stddev: 0.2347842496259151",
            "extra": "mean: 99.89242308214307 sec\nrounds: 5"
          },
          {
            "name": "tests/bench_instrumentor/bench.py::test_instrumented",
            "value": 0.006947658362266693,
            "unit": "iter/sec",
            "range": "stddev: 0.6411509369633734",
            "extra": "mean: 143.93338702879845 sec\nrounds: 5"
          },
          {
            "name": "tests/bench_instrumentor/bench.py::test_instrumented_with_jit_and_c_tracing_disabled",
            "value": 0.008371902230007493,
            "unit": "iter/sec",
            "range": "stddev: 0.4386491796233946",
            "extra": "mean: 119.44716654904187 sec\nrounds: 5"
          },
          {
            "name": "tests/bench_instrumentor/bench.py::test_sampler_instrumented",
            "value": 0.0011855473114329082,
            "unit": "iter/sec",
            "range": "stddev: 2.7640879200856663",
            "extra": "mean: 843.4922759778798 sec\nrounds: 5"
          },
          {
            "name": "tests/bench_instrumentor/bench.py::test_proxy_instrumented",
            "value": 0.003911277413115028,
            "unit": "iter/sec",
            "range": "stddev: 1.423803252404729",
            "extra": "mean: 255.6709469512105 sec\nrounds: 5"
          },
          {
            "name": "tests/bench_instrumentor/bench.py::test_proxy_instrumented_with_scan_proxy_in_args",
            "value": 0.0038273372303564684,
            "unit": "iter/sec",
            "range": "stddev: 1.2894415339312966",
            "extra": "mean: 261.2782568696886 sec\nrounds: 5"
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
          "id": "9ed1461f71d51d3621223442f4137e7984abf4bb",
          "message": "Proxy v4.2",
          "timestamp": "2024-07-13T02:50:05Z",
          "url": "https://github.com/OrderLab/ml-daikon/pull/60/commits/9ed1461f71d51d3621223442f4137e7984abf4bb"
        },
        "date": 1720894447852,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/bench_instrumentor/bench.py::test_naive",
            "value": 0.009967382519822996,
            "unit": "iter/sec",
            "range": "stddev: 0.2401851920551177",
            "extra": "mean: 100.32724218331278 sec\nrounds: 5"
          },
          {
            "name": "tests/bench_instrumentor/bench.py::test_instrumented",
            "value": 0.00664504219402895,
            "unit": "iter/sec",
            "range": "stddev: 0.48945267368839607",
            "extra": "mean: 150.4881339803338 sec\nrounds: 5"
          },
          {
            "name": "tests/bench_instrumentor/bench.py::test_instrumented_with_jit_and_c_tracing_disabled",
            "value": 0.008222762862530424,
            "unit": "iter/sec",
            "range": "stddev: 0.5182277021421873",
            "extra": "mean: 121.61362509392202 sec\nrounds: 5"
          },
          {
            "name": "tests/bench_instrumentor/bench.py::test_sampler_instrumented",
            "value": 0.0011787441388542356,
            "unit": "iter/sec",
            "range": "stddev: 4.076619006418231",
            "extra": "mean: 848.3605279870332 sec\nrounds: 5"
          },
          {
            "name": "tests/bench_instrumentor/bench.py::test_proxy_instrumented",
            "value": 0.0011461988367143805,
            "unit": "iter/sec",
            "range": "stddev: 4.61457532937423",
            "extra": "mean: 872.4489747926593 sec\nrounds: 5"
          },
          {
            "name": "tests/bench_instrumentor/bench.py::test_proxy_instrumented_with_scan_proxy_in_args",
            "value": 0.0012781537749262841,
            "unit": "iter/sec",
            "range": "stddev: 63.17559116330824",
            "extra": "mean: 782.378474028036 sec\nrounds: 5"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "zimingzh@umich.edu",
            "name": "ziming-zh",
            "username": "ziming-zh"
          },
          "committer": {
            "email": "noreply@github.com",
            "name": "GitHub",
            "username": "web-flow"
          },
          "distinct": true,
          "id": "c29334252591961474253ff81f45ec42d24faeb4",
          "message": "Merge pull request #60 from OrderLab/proxy-v4.2\n\nProxy v4.2",
          "timestamp": "2024-07-13T15:22:53-04:00",
          "tree_id": "794ae858ef96fdc5f44fd002cb55d5a9de7310c0",
          "url": "https://github.com/OrderLab/ml-daikon/commit/c29334252591961474253ff81f45ec42d24faeb4"
        },
        "date": 1720918606827,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/bench_instrumentor/bench.py::test_naive",
            "value": 0.00999939652274028,
            "unit": "iter/sec",
            "range": "stddev: 0.3357622610607114",
            "extra": "mean: 100.00603513680399 sec\nrounds: 5"
          },
          {
            "name": "tests/bench_instrumentor/bench.py::test_instrumented",
            "value": 0.006672307497802089,
            "unit": "iter/sec",
            "range": "stddev: 0.39532831440287686",
            "extra": "mean: 149.87318859770895 sec\nrounds: 5"
          },
          {
            "name": "tests/bench_instrumentor/bench.py::test_instrumented_with_jit_and_c_tracing_disabled",
            "value": 0.008223851778967398,
            "unit": "iter/sec",
            "range": "stddev: 0.33560180029919473",
            "extra": "mean: 121.59752228967845 sec\nrounds: 5"
          },
          {
            "name": "tests/bench_instrumentor/bench.py::test_sampler_instrumented",
            "value": 0.0011772806974850263,
            "unit": "iter/sec",
            "range": "stddev: 5.653630556235209",
            "extra": "mean: 849.4150988258422 sec\nrounds: 5"
          },
          {
            "name": "tests/bench_instrumentor/bench.py::test_proxy_instrumented",
            "value": 0.0011442470179682588,
            "unit": "iter/sec",
            "range": "stddev: 4.8974725392046485",
            "extra": "mean: 873.9371694196016 sec\nrounds: 5"
          },
          {
            "name": "tests/bench_instrumentor/bench.py::test_proxy_instrumented_with_scan_proxy_in_args",
            "value": 0.0013607165716942255,
            "unit": "iter/sec",
            "range": "stddev: 20.036179031976214",
            "extra": "mean: 734.9069018501789 sec\nrounds: 5"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "lessoxx@gmail.com",
            "name": "Yuxuan",
            "username": "Essoz"
          },
          "committer": {
            "email": "lessoxx@gmail.com",
            "name": "Yuxuan",
            "username": "Essoz"
          },
          "distinct": true,
          "id": "dbd7db0369d578636535e963ba7a2ae1e724724a",
          "message": "fix: inconsisntency of init value skipping in exp collection logic in consistency relation",
          "timestamp": "2024-07-14T16:01:34-04:00",
          "tree_id": "19a5caf5f2aba71c2e7e54114a975132f63abfba",
          "url": "https://github.com/OrderLab/ml-daikon/commit/dbd7db0369d578636535e963ba7a2ae1e724724a"
        },
        "date": 1721007170979,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/bench_instrumentor/bench.py::test_naive",
            "value": 0.01000145139470314,
            "unit": "iter/sec",
            "range": "stddev: 0.377152603677002",
            "extra": "mean: 99.9854881592095 sec\nrounds: 5"
          },
          {
            "name": "tests/bench_instrumentor/bench.py::test_instrumented",
            "value": 0.006664762244610463,
            "unit": "iter/sec",
            "range": "stddev: 0.9129838738165348",
            "extra": "mean: 150.04286174029113 sec\nrounds: 5"
          },
          {
            "name": "tests/bench_instrumentor/bench.py::test_instrumented_with_jit_and_c_tracing_disabled",
            "value": 0.008190241427992715,
            "unit": "iter/sec",
            "range": "stddev: 0.6240567015017499",
            "extra": "mean: 122.09652289152146 sec\nrounds: 5"
          },
          {
            "name": "tests/bench_instrumentor/bench.py::test_sampler_instrumented",
            "value": 0.001177012510615721,
            "unit": "iter/sec",
            "range": "stddev: 2.9397122853760544",
            "extra": "mean: 849.6086413532496 sec\nrounds: 5"
          },
          {
            "name": "tests/bench_instrumentor/bench.py::test_proxy_instrumented",
            "value": 0.001160062552452523,
            "unit": "iter/sec",
            "range": "stddev: 6.678905177002798",
            "extra": "mean: 862.0224813617766 sec\nrounds: 5"
          },
          {
            "name": "tests/bench_instrumentor/bench.py::test_proxy_instrumented_with_scan_proxy_in_args",
            "value": 0.001324420662737795,
            "unit": "iter/sec",
            "range": "stddev: 36.851877856421176",
            "extra": "mean: 755.0471146628261 sec\nrounds: 5"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "lessoxx@gmail.com",
            "name": "Yuxuan",
            "username": "Essoz"
          },
          "committer": {
            "email": "lessoxx@gmail.com",
            "name": "Yuxuan",
            "username": "Essoz"
          },
          "distinct": true,
          "id": "4664adb7e9a2c41083f071992f56321f4c1d12a1",
          "message": "fix: formatting",
          "timestamp": "2024-07-15T01:23:12-04:00",
          "tree_id": "f75b176f4aa3d831e9ead412a508eb070db55059",
          "url": "https://github.com/OrderLab/ml-daikon/commit/4664adb7e9a2c41083f071992f56321f4c1d12a1"
        },
        "date": 1721038628318,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/bench_instrumentor/bench.py::test_naive",
            "value": 0.01044013323377864,
            "unit": "iter/sec",
            "range": "stddev: 0.09763097958659755",
            "extra": "mean: 95.78421822860837 sec\nrounds: 5"
          },
          {
            "name": "tests/bench_instrumentor/bench.py::test_instrumented",
            "value": 0.00695764765450256,
            "unit": "iter/sec",
            "range": "stddev: 0.3926969615044727",
            "extra": "mean: 143.72673777937888 sec\nrounds: 5"
          },
          {
            "name": "tests/bench_instrumentor/bench.py::test_instrumented_with_jit_and_c_tracing_disabled",
            "value": 0.00864175693482324,
            "unit": "iter/sec",
            "range": "stddev: 0.12563911121651652",
            "extra": "mean: 115.71720976904035 sec\nrounds: 5"
          },
          {
            "name": "tests/bench_instrumentor/bench.py::test_sampler_instrumented",
            "value": 0.0012781796814457166,
            "unit": "iter/sec",
            "range": "stddev: 1.1158868588019675",
            "extra": "mean: 782.3626165524125 sec\nrounds: 5"
          },
          {
            "name": "tests/bench_instrumentor/bench.py::test_proxy_instrumented",
            "value": 0.0014053040141263524,
            "unit": "iter/sec",
            "range": "stddev: 9.465875445690655",
            "extra": "mean: 711.5897983267903 sec\nrounds: 5"
          },
          {
            "name": "tests/bench_instrumentor/bench.py::test_proxy_instrumented_with_scan_proxy_in_args",
            "value": 0.0014940997939178594,
            "unit": "iter/sec",
            "range": "stddev: 40.90857767208718",
            "extra": "mean: 669.2993360087276 sec\nrounds: 5"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "zimingzh@umich.edu",
            "name": "zimingzh",
            "username": "ziming-zh"
          },
          "committer": {
            "email": "zimingzh@umich.edu",
            "name": "zimingzh",
            "username": "ziming-zh"
          },
          "distinct": true,
          "id": "81a627b5582898af6e0b6719000c3e470ddcf19c",
          "message": "fix: fix hasing and meta_var_dumping utility to support DS-1801",
          "timestamp": "2024-07-15T10:33:41-04:00",
          "tree_id": "9f474231724caf4df73f387e687e315cbd18f6fb",
          "url": "https://github.com/OrderLab/ml-daikon/commit/81a627b5582898af6e0b6719000c3e470ddcf19c"
        },
        "date": 1721068688229,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/bench_instrumentor/bench.py::test_naive",
            "value": 0.010438617182728477,
            "unit": "iter/sec",
            "range": "stddev: 0.16384901269677768",
            "extra": "mean: 95.79812943562865 sec\nrounds: 5"
          },
          {
            "name": "tests/bench_instrumentor/bench.py::test_instrumented",
            "value": 0.006962150258070667,
            "unit": "iter/sec",
            "range": "stddev: 0.17715555187260965",
            "extra": "mean: 143.63378596156835 sec\nrounds: 5"
          },
          {
            "name": "tests/bench_instrumentor/bench.py::test_instrumented_with_jit_and_c_tracing_disabled",
            "value": 0.00864250587187706,
            "unit": "iter/sec",
            "range": "stddev: 0.10809849851805296",
            "extra": "mean: 115.70718201696873 sec\nrounds: 5"
          },
          {
            "name": "tests/bench_instrumentor/bench.py::test_sampler_instrumented",
            "value": 0.0012797322287811453,
            "unit": "iter/sec",
            "range": "stddev: 2.5144406065027263",
            "extra": "mean: 781.4134687788785 sec\nrounds: 5"
          },
          {
            "name": "tests/bench_instrumentor/bench.py::test_proxy_instrumented",
            "value": 0.0021223097551311835,
            "unit": "iter/sec",
            "range": "stddev: 1.4319833279877228",
            "extra": "mean: 471.1847540549934 sec\nrounds: 5"
          },
          {
            "name": "tests/bench_instrumentor/bench.py::test_proxy_instrumented_with_scan_proxy_in_args",
            "value": 0.0020786197870329708,
            "unit": "iter/sec",
            "range": "stddev: 0.6588282473177651",
            "extra": "mean: 481.0884637191892 sec\nrounds: 5"
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
          "id": "f658deb50240a90aa32a1f337a3accc7944614fa",
          "message": "support cuda kernel on Tensor hash acceleration",
          "timestamp": "2024-07-15T15:37:10Z",
          "url": "https://github.com/OrderLab/ml-daikon/pull/64/commits/f658deb50240a90aa32a1f337a3accc7944614fa"
        },
        "date": 1721170859767,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/bench_instrumentor/bench.py::test_naive",
            "value": 0.010477476937292195,
            "unit": "iter/sec",
            "range": "stddev: 0.1626492252728768",
            "extra": "mean: 95.44282521307468 sec\nrounds: 5"
          },
          {
            "name": "tests/bench_instrumentor/bench.py::test_instrumented",
            "value": 0.006978171401346192,
            "unit": "iter/sec",
            "range": "stddev: 0.3029242108420886",
            "extra": "mean: 143.3040179848671 sec\nrounds: 5"
          },
          {
            "name": "tests/bench_instrumentor/bench.py::test_instrumented_with_jit_and_c_tracing_disabled",
            "value": 0.008649974425823291,
            "unit": "iter/sec",
            "range": "stddev: 0.19089566589468535",
            "extra": "mean: 115.6072782151401 sec\nrounds: 5"
          },
          {
            "name": "tests/bench_instrumentor/bench.py::test_sampler_instrumented",
            "value": 0.0012788426307058214,
            "unit": "iter/sec",
            "range": "stddev: 2.425615265575393",
            "extra": "mean: 781.9570414602756 sec\nrounds: 5"
          },
          {
            "name": "tests/bench_instrumentor/bench.py::test_proxy_instrumented",
            "value": 0.004330242709771014,
            "unit": "iter/sec",
            "range": "stddev: 0.4959756009323582",
            "extra": "mean: 230.93393766209482 sec\nrounds: 5"
          },
          {
            "name": "tests/bench_instrumentor/bench.py::test_proxy_instrumented_with_scan_proxy_in_args",
            "value": 0.004219614185871851,
            "unit": "iter/sec",
            "range": "stddev: 0.5045964870901666",
            "extra": "mean: 236.98849135264754 sec\nrounds: 5"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "zimingzh@umich.edu",
            "name": "ziming-zh",
            "username": "ziming-zh"
          },
          "committer": {
            "email": "noreply@github.com",
            "name": "GitHub",
            "username": "web-flow"
          },
          "distinct": true,
          "id": "ef39afaf2e4d58b29b5d2faa5f76abe60ac28fa6",
          "message": "Merge pull request #64 from OrderLab/tensor_hash_accelerate\n\nsupport cuda kernel on Tensor hash acceleration",
          "timestamp": "2024-07-16T19:23:56-04:00",
          "tree_id": "571735bff5da06c9c1d34c2fe072493bb4f1bf3e",
          "url": "https://github.com/OrderLab/ml-daikon/commit/ef39afaf2e4d58b29b5d2faa5f76abe60ac28fa6"
        },
        "date": 1721183508216,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/bench_instrumentor/bench.py::test_naive",
            "value": 0.010437760779545373,
            "unit": "iter/sec",
            "range": "stddev: 0.19642612686318012",
            "extra": "mean: 95.8059895336628 sec\nrounds: 5"
          },
          {
            "name": "tests/bench_instrumentor/bench.py::test_instrumented",
            "value": 0.006981079844762635,
            "unit": "iter/sec",
            "range": "stddev: 0.30530450995960623",
            "extra": "mean: 143.2443149536848 sec\nrounds: 5"
          },
          {
            "name": "tests/bench_instrumentor/bench.py::test_instrumented_with_jit_and_c_tracing_disabled",
            "value": 0.008620752118854241,
            "unit": "iter/sec",
            "range": "stddev: 0.12765157504551275",
            "extra": "mean: 115.99915949478745 sec\nrounds: 5"
          },
          {
            "name": "tests/bench_instrumentor/bench.py::test_sampler_instrumented",
            "value": 0.0012767614181248919,
            "unit": "iter/sec",
            "range": "stddev: 1.6162978649974806",
            "extra": "mean: 783.2316874586046 sec\nrounds: 5"
          },
          {
            "name": "tests/bench_instrumentor/bench.py::test_proxy_instrumented",
            "value": 0.004322374113317539,
            "unit": "iter/sec",
            "range": "stddev: 0.3446586921188709",
            "extra": "mean: 231.35433763563634 sec\nrounds: 5"
          },
          {
            "name": "tests/bench_instrumentor/bench.py::test_proxy_instrumented_with_scan_proxy_in_args",
            "value": 0.0042085270203158325,
            "unit": "iter/sec",
            "range": "stddev: 1.1122193893110845",
            "extra": "mean: 237.61282633394003 sec\nrounds: 5"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "zimingzh@umich.edu",
            "name": "zimingzh",
            "username": "ziming-zh"
          },
          "committer": {
            "email": "zimingzh@umich.edu",
            "name": "zimingzh",
            "username": "ziming-zh"
          },
          "distinct": true,
          "id": "bb80974ba6da1ad0bc2d6d9ef24cbec9c14065bf",
          "message": "varobserver: support dumping for hashed tensor value",
          "timestamp": "2024-07-18T21:03:35-04:00",
          "tree_id": "9f49ccccefb2f0b566a32f58dfbfcee02d2d9daf",
          "url": "https://github.com/OrderLab/ml-daikon/commit/bb80974ba6da1ad0bc2d6d9ef24cbec9c14065bf"
        },
        "date": 1721362544866,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/bench_instrumentor/bench.py::test_naive",
            "value": 0.010435279323473176,
            "unit": "iter/sec",
            "range": "stddev: 0.16258977453834075",
            "extra": "mean: 95.82877170816064 sec\nrounds: 5"
          },
          {
            "name": "tests/bench_instrumentor/bench.py::test_instrumented",
            "value": 0.006971899020841932,
            "unit": "iter/sec",
            "range": "stddev: 0.22779046362204425",
            "extra": "mean: 143.43294373750686 sec\nrounds: 5"
          },
          {
            "name": "tests/bench_instrumentor/bench.py::test_instrumented_with_jit_and_c_tracing_disabled",
            "value": 0.008643707555135084,
            "unit": "iter/sec",
            "range": "stddev: 0.17909805898400344",
            "extra": "mean: 115.6910959355533 sec\nrounds: 5"
          },
          {
            "name": "tests/bench_instrumentor/bench.py::test_sampler_instrumented",
            "value": 0.0012758609421104888,
            "unit": "iter/sec",
            "range": "stddev: 1.9713643675703034",
            "extra": "mean: 783.784476030618 sec\nrounds: 5"
          },
          {
            "name": "tests/bench_instrumentor/bench.py::test_proxy_instrumented",
            "value": 0.004329961751733461,
            "unit": "iter/sec",
            "range": "stddev: 0.44815229253379435",
            "extra": "mean: 230.94892226234077 sec\nrounds: 5"
          },
          {
            "name": "tests/bench_instrumentor/bench.py::test_proxy_instrumented_with_scan_proxy_in_args",
            "value": 0.004232940898377755,
            "unit": "iter/sec",
            "range": "stddev: 0.6283963581726769",
            "extra": "mean: 236.24237238541247 sec\nrounds: 5"
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
          "id": "45578ff755f07ff240ff50db7a9124678555a886",
          "message": "Aync + Per-thread Trace Dumping",
          "timestamp": "2024-07-19T01:03:46Z",
          "url": "https://github.com/OrderLab/ml-daikon/pull/69/commits/45578ff755f07ff240ff50db7a9124678555a886"
        },
        "date": 1721443559722,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/bench_instrumentor/bench.py::test_naive",
            "value": 0.010463410782373497,
            "unit": "iter/sec",
            "range": "stddev: 0.07198613068955219",
            "extra": "mean: 95.57113075256348 sec\nrounds: 5"
          },
          {
            "name": "tests/bench_instrumentor/bench.py::test_instrumented",
            "value": 0.006468595916349856,
            "unit": "iter/sec",
            "range": "stddev: 0.6893256377419059",
            "extra": "mean: 154.5930543400347 sec\nrounds: 5"
          },
          {
            "name": "tests/bench_instrumentor/bench.py::test_instrumented_with_jit_and_c_tracing_disabled",
            "value": 0.00778877913644985,
            "unit": "iter/sec",
            "range": "stddev: 0.38015015612922703",
            "extra": "mean: 128.389826246351 sec\nrounds: 5"
          },
          {
            "name": "tests/bench_instrumentor/bench.py::test_sampler_instrumented",
            "value": 0.0012914990009337515,
            "unit": "iter/sec",
            "range": "stddev: 0.5996095868712892",
            "extra": "mean: 774.2940561912953 sec\nrounds: 5"
          },
          {
            "name": "tests/bench_instrumentor/bench.py::test_proxy_instrumented",
            "value": 0.004255921895824967,
            "unit": "iter/sec",
            "range": "stddev: 0.34775697497902747",
            "extra": "mean: 234.96671801730992 sec\nrounds: 5"
          },
          {
            "name": "tests/bench_instrumentor/bench.py::test_proxy_instrumented_with_scan_proxy_in_args",
            "value": 0.0041731141952129,
            "unit": "iter/sec",
            "range": "stddev: 0.28997526879912083",
            "extra": "mean: 239.6291961401701 sec\nrounds: 5"
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
          "id": "eb41f1353813133687bc910309f5fd69a3f02a7d",
          "message": "Aync + Per-thread Trace Dumping",
          "timestamp": "2024-07-19T01:03:46Z",
          "url": "https://github.com/OrderLab/ml-daikon/pull/69/commits/eb41f1353813133687bc910309f5fd69a3f02a7d"
        },
        "date": 1721462530443,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/bench_instrumentor/bench.py::test_naive",
            "value": 0.010416164540819294,
            "unit": "iter/sec",
            "range": "stddev: 0.5891012513011252",
            "extra": "mean: 96.00462781488895 sec\nrounds: 5"
          },
          {
            "name": "tests/bench_instrumentor/bench.py::test_instrumented",
            "value": 0.004669997275348523,
            "unit": "iter/sec",
            "range": "stddev: 0.8311384001518882",
            "extra": "mean: 214.13288724571467 sec\nrounds: 5"
          },
          {
            "name": "tests/bench_instrumentor/bench.py::test_instrumented_with_jit_and_c_tracing_disabled",
            "value": 0.00671070523421127,
            "unit": "iter/sec",
            "range": "stddev: 0.6811935725728014",
            "extra": "mean: 149.01563473567367 sec\nrounds: 5"
          },
          {
            "name": "tests/bench_instrumentor/bench.py::test_sampler_instrumented",
            "value": 0.0011868195410987908,
            "unit": "iter/sec",
            "range": "stddev: 0.8684897183223007",
            "extra": "mean: 842.5880813136697 sec\nrounds: 5"
          },
          {
            "name": "tests/bench_instrumentor/bench.py::test_proxy_instrumented",
            "value": 0.0031718880372187757,
            "unit": "iter/sec",
            "range": "stddev: 0.8468836375907984",
            "extra": "mean: 315.269640121609 sec\nrounds: 5"
          },
          {
            "name": "tests/bench_instrumentor/bench.py::test_proxy_instrumented_with_scan_proxy_in_args",
            "value": 0.0031441547583673296,
            "unit": "iter/sec",
            "range": "stddev: 1.1739470286417624",
            "extra": "mean: 318.0505022339523 sec\nrounds: 5"
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
          "id": "dd4369a34cefc4ae65d06efa2a1295472b6ef726",
          "message": "Aync + Per-thread Trace Dumping",
          "timestamp": "2024-07-19T01:03:46Z",
          "url": "https://github.com/OrderLab/ml-daikon/pull/69/commits/dd4369a34cefc4ae65d06efa2a1295472b6ef726"
        },
        "date": 1721474601675,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/bench_instrumentor/bench.py::test_naive",
            "value": 0.01044034118488751,
            "unit": "iter/sec",
            "range": "stddev: 0.08732136342831147",
            "extra": "mean: 95.78231039494275 sec\nrounds: 5"
          },
          {
            "name": "tests/bench_instrumentor/bench.py::test_instrumented",
            "value": 0.005934521969652183,
            "unit": "iter/sec",
            "range": "stddev: 0.5118125107121912",
            "extra": "mean: 168.50556879118085 sec\nrounds: 5"
          },
          {
            "name": "tests/bench_instrumentor/bench.py::test_instrumented_with_jit_and_c_tracing_disabled",
            "value": 0.00736413489565234,
            "unit": "iter/sec",
            "range": "stddev: 1.0090907369286592",
            "extra": "mean: 135.79327567592264 sec\nrounds: 5"
          },
          {
            "name": "tests/bench_instrumentor/bench.py::test_sampler_instrumented",
            "value": 0.001251307618290653,
            "unit": "iter/sec",
            "range": "stddev: 1.0230148843068774",
            "extra": "mean: 799.1639988303184 sec\nrounds: 5"
          },
          {
            "name": "tests/bench_instrumentor/bench.py::test_proxy_instrumented",
            "value": 0.003899382007823133,
            "unit": "iter/sec",
            "range": "stddev: 0.9737807276076378",
            "extra": "mean: 256.45089349895716 sec\nrounds: 5"
          },
          {
            "name": "tests/bench_instrumentor/bench.py::test_proxy_instrumented_with_scan_proxy_in_args",
            "value": 0.003812756455655072,
            "unit": "iter/sec",
            "range": "stddev: 0.5904790707453084",
            "extra": "mean: 262.2774393357337 sec\nrounds: 5"
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
          "id": "0406562c60cad61fc889b32e839265e9d0280ed6",
          "message": "Merge pull request #69 from OrderLab/aync_trace_dumping\n\nAync + Per-thread Trace Dumping",
          "timestamp": "2024-07-20T02:21:28-04:00",
          "tree_id": "a227bec871f2d4679389bc9e11f88edb3bc839ca",
          "url": "https://github.com/OrderLab/ml-daikon/commit/0406562c60cad61fc889b32e839265e9d0280ed6"
        },
        "date": 1721498457940,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/bench_instrumentor/bench.py::test_naive",
            "value": 0.010442363297673085,
            "unit": "iter/sec",
            "range": "stddev: 0.16377429207388222",
            "extra": "mean: 95.76376261711121 sec\nrounds: 5"
          },
          {
            "name": "tests/bench_instrumentor/bench.py::test_instrumented",
            "value": 0.006541541805743293,
            "unit": "iter/sec",
            "range": "stddev: 0.47227033668390705",
            "extra": "mean: 152.86915985494852 sec\nrounds: 5"
          },
          {
            "name": "tests/bench_instrumentor/bench.py::test_instrumented_with_jit_and_c_tracing_disabled",
            "value": 0.007864269576899296,
            "unit": "iter/sec",
            "range": "stddev: 1.0510633184263452",
            "extra": "mean: 127.15739080682397 sec\nrounds: 5"
          },
          {
            "name": "tests/bench_instrumentor/bench.py::test_sampler_instrumented",
            "value": 0.001286587922016987,
            "unit": "iter/sec",
            "range": "stddev: 1.1971783490491998",
            "extra": "mean: 777.2496406093239 sec\nrounds: 5"
          },
          {
            "name": "tests/bench_instrumentor/bench.py::test_proxy_instrumented",
            "value": 0.00431776579663213,
            "unit": "iter/sec",
            "range": "stddev: 0.36460894106638114",
            "extra": "mean: 231.60126025825738 sec\nrounds: 5"
          },
          {
            "name": "tests/bench_instrumentor/bench.py::test_proxy_instrumented_with_scan_proxy_in_args",
            "value": 0.004237041718546481,
            "unit": "iter/sec",
            "range": "stddev: 0.9439241230402652",
            "extra": "mean: 236.0137252420187 sec\nrounds: 5"
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
            "email": "lessoxx@gmail.com",
            "name": "Yuxuan Jiang",
            "username": "Essoz"
          },
          "distinct": true,
          "id": "71ebaf433d1bc91c668e9a3b65452ff1297f1a2f",
          "message": "Merge pull request #66 from OrderLab/speed_improve_infer_engine\n\nSampling as a temporary solution to make the inference pipeline run through",
          "timestamp": "2024-07-22T22:47:07-04:00",
          "tree_id": "81f6cf8ef22275b9c3fa80f0c4d894f19e52f0fc",
          "url": "https://github.com/OrderLab/ml-daikon/commit/71ebaf433d1bc91c668e9a3b65452ff1297f1a2f"
        },
        "date": 1721716936706,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/bench_instrumentor/bench.py::test_naive",
            "value": 0.01044885870935871,
            "unit": "iter/sec",
            "range": "stddev: 0.1818699359687371",
            "extra": "mean: 95.70423218607903 sec\nrounds: 5"
          },
          {
            "name": "tests/bench_instrumentor/bench.py::test_instrumented",
            "value": 0.006534654688636097,
            "unit": "iter/sec",
            "range": "stddev: 0.4027173680415486",
            "extra": "mean: 153.0302743829787 sec\nrounds: 5"
          },
          {
            "name": "tests/bench_instrumentor/bench.py::test_instrumented_with_jit_and_c_tracing_disabled",
            "value": 0.00788903484882097,
            "unit": "iter/sec",
            "range": "stddev: 0.1749709663488091",
            "extra": "mean: 126.75821810439228 sec\nrounds: 5"
          },
          {
            "name": "tests/bench_instrumentor/bench.py::test_sampler_instrumented",
            "value": 0.0012864636370064364,
            "unit": "iter/sec",
            "range": "stddev: 1.3871404307997266",
            "extra": "mean: 777.3247305512429 sec\nrounds: 5"
          },
          {
            "name": "tests/bench_instrumentor/bench.py::test_proxy_instrumented",
            "value": 0.004322490043333976,
            "unit": "iter/sec",
            "range": "stddev: 0.7620731320666129",
            "extra": "mean: 231.34813266769052 sec\nrounds: 5"
          },
          {
            "name": "tests/bench_instrumentor/bench.py::test_proxy_instrumented_with_scan_proxy_in_args",
            "value": 0.004243240865268727,
            "unit": "iter/sec",
            "range": "stddev: 0.34821125715303963",
            "extra": "mean: 235.66892188116907 sec\nrounds: 5"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "zimingzh@umich.edu",
            "name": "zimingzh",
            "username": "ziming-zh"
          },
          "committer": {
            "email": "zimingzh@umich.edu",
            "name": "zimingzh",
            "username": "ziming-zh"
          },
          "distinct": true,
          "id": "4ae5409500b417cdd0b8b3dac38d21bf482bc7b0",
          "message": "fix: fix only_dump_when_change utility to auto_observer",
          "timestamp": "2024-07-23T17:01:17-04:00",
          "tree_id": "b659c187c5081604cb4052233c41f15d3dbd2c32",
          "url": "https://github.com/OrderLab/ml-daikon/commit/4ae5409500b417cdd0b8b3dac38d21bf482bc7b0"
        },
        "date": 1721779106836,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/bench_instrumentor/bench.py::test_naive",
            "value": 0.010491150968001274,
            "unit": "iter/sec",
            "range": "stddev: 0.16115150745111537",
            "extra": "mean: 95.31842626705766 sec\nrounds: 5"
          },
          {
            "name": "tests/bench_instrumentor/bench.py::test_instrumented",
            "value": 0.006583086513280397,
            "unit": "iter/sec",
            "range": "stddev: 0.5913348464139029",
            "extra": "mean: 151.9044293254614 sec\nrounds: 5"
          },
          {
            "name": "tests/bench_instrumentor/bench.py::test_instrumented_with_jit_and_c_tracing_disabled",
            "value": 0.007871652750409137,
            "unit": "iter/sec",
            "range": "stddev: 0.7099602616970458",
            "extra": "mean: 127.03812422975898 sec\nrounds: 5"
          },
          {
            "name": "tests/bench_instrumentor/bench.py::test_sampler_instrumented",
            "value": 0.0012873359139487792,
            "unit": "iter/sec",
            "range": "stddev: 0.7795865620936128",
            "extra": "mean: 776.7980285212398 sec\nrounds: 5"
          },
          {
            "name": "tests/bench_instrumentor/bench.py::test_proxy_instrumented",
            "value": 0.0055889874372259,
            "unit": "iter/sec",
            "range": "stddev: 0.4960230387136034",
            "extra": "mean: 178.9232864148915 sec\nrounds: 5"
          },
          {
            "name": "tests/bench_instrumentor/bench.py::test_proxy_instrumented_with_scan_proxy_in_args",
            "value": 0.005482885359289055,
            "unit": "iter/sec",
            "range": "stddev: 0.5090941698371497",
            "extra": "mean: 182.38572110682725 sec\nrounds: 5"
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
          "id": "20f2c73f94686223c8e9968ee4d01465e618e5f7",
          "message": "fix: use consistent formatting for args",
          "timestamp": "2024-07-31T15:50:22-04:00",
          "tree_id": "b8d7dd26753f24ba51398a49b070686ef536f556",
          "url": "https://github.com/OrderLab/ml-daikon/commit/20f2c73f94686223c8e9968ee4d01465e618e5f7"
        },
        "date": 1722463624466,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/bench_instrumentor/bench.py::test_instrumented",
            "value": 0.006645522370174294,
            "unit": "iter/sec",
            "range": "stddev: 0.673322031041904",
            "extra": "mean: 150.4772603712976 sec\nrounds: 5"
          },
          {
            "name": "tests/bench_instrumentor/bench.py::test_instrumented_with_jit_and_c_tracing_disabled",
            "value": 0.007871652063690383,
            "unit": "iter/sec",
            "range": "stddev: 0.5325193389408226",
            "extra": "mean: 127.03813531249762 sec\nrounds: 5"
          },
          {
            "name": "tests/bench_instrumentor/bench.py::test_sampler_instrumented",
            "value": 0.0013169361906401416,
            "unit": "iter/sec",
            "range": "stddev: 1.9498154445134017",
            "extra": "mean: 759.3382330194115 sec\nrounds: 5"
          },
          {
            "name": "tests/bench_instrumentor/bench.py::test_proxy_instrumented",
            "value": 0.007509590105480393,
            "unit": "iter/sec",
            "range": "stddev: 0.6857702337535263",
            "extra": "mean: 133.1630602940917 sec\nrounds: 5"
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
          "id": "2139a6e8f0aa86b94d0d142f1342c0638e71998d",
          "message": "hack: hardcode proxy class logging dir to use output_dir",
          "timestamp": "2024-07-31T17:22:07-04:00",
          "tree_id": "dd838731c8e4d71845ea91a3a233a1f4506c589b",
          "url": "https://github.com/OrderLab/ml-daikon/commit/2139a6e8f0aa86b94d0d142f1342c0638e71998d"
        },
        "date": 1722471838398,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/bench_instrumentor/bench.py::test_instrumented",
            "value": 0.00665385293859495,
            "unit": "iter/sec",
            "range": "stddev: 0.4619894321076923",
            "extra": "mean: 150.28886409550904 sec\nrounds: 5"
          },
          {
            "name": "tests/bench_instrumentor/bench.py::test_instrumented_with_jit_and_c_tracing_disabled",
            "value": 0.007876359446953017,
            "unit": "iter/sec",
            "range": "stddev: 0.38899752432030293",
            "extra": "mean: 126.96220972836018 sec\nrounds: 5"
          },
          {
            "name": "tests/bench_instrumentor/bench.py::test_sampler_instrumented",
            "value": 0.001321376879764647,
            "unit": "iter/sec",
            "range": "stddev: 1.3767564601589815",
            "extra": "mean: 756.7863607376814 sec\nrounds: 5"
          },
          {
            "name": "tests/bench_instrumentor/bench.py::test_proxy_instrumented",
            "value": 0.007545865940252674,
            "unit": "iter/sec",
            "range": "stddev: 0.5442019966763432",
            "extra": "mean: 132.5228950418532 sec\nrounds: 5"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "zimingzh@umich.edu",
            "name": "zimingzh",
            "username": "ziming-zh"
          },
          "committer": {
            "email": "zimingzh@umich.edu",
            "name": "zimingzh",
            "username": "ziming-zh"
          },
          "distinct": true,
          "id": "90eba5fbe937443a4860935c24387cd95014615f",
          "message": "fix: remove redundant proxy_log config",
          "timestamp": "2024-08-01T04:59:41-04:00",
          "tree_id": "e1bff731086c705535309ccf4ef440478f530019",
          "url": "https://github.com/OrderLab/ml-daikon/commit/90eba5fbe937443a4860935c24387cd95014615f"
        },
        "date": 1722511007903,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/bench_instrumentor/bench.py::test_instrumented",
            "value": 0.006646153923208881,
            "unit": "iter/sec",
            "range": "stddev: 0.33582293057114243",
            "extra": "mean: 150.4629612185061 sec\nrounds: 5"
          },
          {
            "name": "tests/bench_instrumentor/bench.py::test_instrumented_with_jit_and_c_tracing_disabled",
            "value": 0.007878947342052434,
            "unit": "iter/sec",
            "range": "stddev: 0.3932416103093566",
            "extra": "mean: 126.92050810679794 sec\nrounds: 5"
          },
          {
            "name": "tests/bench_instrumentor/bench.py::test_sampler_instrumented",
            "value": 0.0013210584330937195,
            "unit": "iter/sec",
            "range": "stddev: 0.6653933161476779",
            "extra": "mean: 756.9687872610987 sec\nrounds: 5"
          },
          {
            "name": "tests/bench_instrumentor/bench.py::test_proxy_instrumented",
            "value": 0.007534141782045247,
            "unit": "iter/sec",
            "range": "stddev: 0.47106840483505114",
            "extra": "mean: 132.72911884710192 sec\nrounds: 5"
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
          "id": "c5f62ee8f20f1f15bea20379ecd1ef80b250c035",
          "message": "End-to-end pipeline inference support for given ML input",
          "timestamp": "2024-08-03T16:17:36Z",
          "url": "https://github.com/OrderLab/ml-daikon/pull/62/commits/c5f62ee8f20f1f15bea20379ecd1ef80b250c035"
        },
        "date": 1722744779841,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/bench_instrumentor/bench.py::test_instrumented",
            "value": 0.006641367583960937,
            "unit": "iter/sec",
            "range": "stddev: 0.3007153485753397",
            "extra": "mean: 150.5713977366686 sec\nrounds: 5"
          },
          {
            "name": "tests/bench_instrumentor/bench.py::test_instrumented_with_jit_and_c_tracing_disabled",
            "value": 0.007878979032833085,
            "unit": "iter/sec",
            "range": "stddev: 0.788548408779686",
            "extra": "mean: 126.9199976079166 sec\nrounds: 5"
          },
          {
            "name": "tests/bench_instrumentor/bench.py::test_sampler_instrumented",
            "value": 0.0013198080196922756,
            "unit": "iter/sec",
            "range": "stddev: 1.662807988736172",
            "extra": "mean: 757.6859551385045 sec\nrounds: 5"
          },
          {
            "name": "tests/bench_instrumentor/bench.py::test_proxy_instrumented",
            "value": 0.007537583740732113,
            "unit": "iter/sec",
            "range": "stddev: 0.867144135572756",
            "extra": "mean: 132.66850948482752 sec\nrounds: 5"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "zimingzh@umich.edu",
            "name": "zimingzh",
            "username": "ziming-zh"
          },
          "committer": {
            "email": "zimingzh@umich.edu",
            "name": "zimingzh",
            "username": "ziming-zh"
          },
          "distinct": true,
          "id": "894e74b7dae14cd23c39039dfc782cc5a97ddc02",
          "message": "fix: fix skipdumping logic for proxy dump_to_trace func",
          "timestamp": "2024-08-04T04:37:34-04:00",
          "tree_id": "ccc464e809ad699c834a94c9be95b9cdb66ea13b",
          "url": "https://github.com/OrderLab/ml-daikon/commit/894e74b7dae14cd23c39039dfc782cc5a97ddc02"
        },
        "date": 1722768882041,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/bench_instrumentor/bench.py::test_instrumented",
            "value": 0.006613584888391961,
            "unit": "iter/sec",
            "range": "stddev: 0.5864194218927623",
            "extra": "mean: 151.20392599105836 sec\nrounds: 5"
          },
          {
            "name": "tests/bench_instrumentor/bench.py::test_instrumented_with_jit_and_c_tracing_disabled",
            "value": 0.00786056748820955,
            "unit": "iter/sec",
            "range": "stddev: 0.3387217890279464",
            "extra": "mean: 127.21727807819843 sec\nrounds: 5"
          },
          {
            "name": "tests/bench_instrumentor/bench.py::test_sampler_instrumented",
            "value": 0.0013195192608703966,
            "unit": "iter/sec",
            "range": "stddev: 1.0378344249387463",
            "extra": "mean: 757.8517643921077 sec\nrounds: 5"
          },
          {
            "name": "tests/bench_instrumentor/bench.py::test_proxy_instrumented",
            "value": 0.007523460686483755,
            "unit": "iter/sec",
            "range": "stddev: 0.2714828594984645",
            "extra": "mean: 132.91755505502223 sec\nrounds: 5"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "zimingzh@umich.edu",
            "name": "zimingzh",
            "username": "ziming-zh"
          },
          "committer": {
            "email": "zimingzh@umich.edu",
            "name": "zimingzh",
            "username": "ziming-zh"
          },
          "distinct": true,
          "id": "edbdd9757f4d0320ee4f4d99a605c9c4d71053fa",
          "message": "fix: disable sampling for primitive_types attr updating",
          "timestamp": "2024-08-04T04:50:38-04:00",
          "tree_id": "9dfe753afde43ba90f9a6ea05237245172d7ed5e",
          "url": "https://github.com/OrderLab/ml-daikon/commit/edbdd9757f4d0320ee4f4d99a605c9c4d71053fa"
        },
        "date": 1722777109371,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/bench_instrumentor/bench.py::test_instrumented",
            "value": 0.006630859434449938,
            "unit": "iter/sec",
            "range": "stddev: 0.3932414377858433",
            "extra": "mean: 150.8100133754313 sec\nrounds: 5"
          },
          {
            "name": "tests/bench_instrumentor/bench.py::test_instrumented_with_jit_and_c_tracing_disabled",
            "value": 0.007817378237694923,
            "unit": "iter/sec",
            "range": "stddev: 0.6619133165052127",
            "extra": "mean: 127.92012482881546 sec\nrounds: 5"
          },
          {
            "name": "tests/bench_instrumentor/bench.py::test_sampler_instrumented",
            "value": 0.001319448522909009,
            "unit": "iter/sec",
            "range": "stddev: 1.5909134988851692",
            "extra": "mean: 757.8923941612244 sec\nrounds: 5"
          },
          {
            "name": "tests/bench_instrumentor/bench.py::test_proxy_instrumented",
            "value": 0.007526487411727975,
            "unit": "iter/sec",
            "range": "stddev: 0.49261725616733315",
            "extra": "mean: 132.86410317271947 sec\nrounds: 5"
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
          "id": "1ba7333d6f980bd6af9c5023a372f17b990a0f70",
          "message": "ContainRelation: sample when too many calls exist or queried calls all return the same result",
          "timestamp": "2024-08-04T15:55:48-04:00",
          "tree_id": "ee7e39306bb517607a2e1cb919036f23ac3abb8e",
          "url": "https://github.com/OrderLab/ml-daikon/commit/1ba7333d6f980bd6af9c5023a372f17b990a0f70"
        },
        "date": 1722809551640,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/bench_instrumentor/bench.py::test_instrumented",
            "value": 0.006635954197049807,
            "unit": "iter/sec",
            "range": "stddev: 0.38573224959176816",
            "extra": "mean: 150.69422878846527 sec\nrounds: 5"
          },
          {
            "name": "tests/bench_instrumentor/bench.py::test_instrumented_with_jit_and_c_tracing_disabled",
            "value": 0.007885027071554024,
            "unit": "iter/sec",
            "range": "stddev: 0.5082841317444806",
            "extra": "mean: 126.82264638096095 sec\nrounds: 5"
          },
          {
            "name": "tests/bench_instrumentor/bench.py::test_sampler_instrumented",
            "value": 0.0013221281311668331,
            "unit": "iter/sec",
            "range": "stddev: 2.0859121483285734",
            "extra": "mean: 756.3563443109393 sec\nrounds: 5"
          },
          {
            "name": "tests/bench_instrumentor/bench.py::test_proxy_instrumented",
            "value": 0.007542503178529507,
            "unit": "iter/sec",
            "range": "stddev: 0.7040456737099495",
            "extra": "mean: 132.58197926208376 sec\nrounds: 5"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "zimingzh@umich.edu",
            "name": "zimingzh",
            "username": "ziming-zh"
          },
          "committer": {
            "email": "zimingzh@umich.edu",
            "name": "zimingzh",
            "username": "ziming-zh"
          },
          "distinct": true,
          "id": "3b8e34a90b2cbc7868963fcad9174ed9e52a0f14",
          "message": "fix: enhance print_debug utility, change the msg inside to a lambda function to avoid object formatting from f-string",
          "timestamp": "2024-08-04T19:26:15-04:00",
          "tree_id": "8deac27acfd908463c53c13d2f0dbb02f81cda9a",
          "url": "https://github.com/OrderLab/ml-daikon/commit/3b8e34a90b2cbc7868963fcad9174ed9e52a0f14"
        },
        "date": 1722823171648,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/bench_instrumentor/bench.py::test_instrumented",
            "value": 0.006636836331354639,
            "unit": "iter/sec",
            "range": "stddev: 0.4109923387886495",
            "extra": "mean: 150.67419928312302 sec\nrounds: 5"
          },
          {
            "name": "tests/bench_instrumentor/bench.py::test_instrumented_with_jit_and_c_tracing_disabled",
            "value": 0.007862748640649016,
            "unit": "iter/sec",
            "range": "stddev: 0.5869200215193447",
            "extra": "mean: 127.18198758512736 sec\nrounds: 5"
          },
          {
            "name": "tests/bench_instrumentor/bench.py::test_sampler_instrumented",
            "value": 0.0013217379714011673,
            "unit": "iter/sec",
            "range": "stddev: 1.1085651545690378",
            "extra": "mean: 756.5796108134091 sec\nrounds: 5"
          },
          {
            "name": "tests/bench_instrumentor/bench.py::test_proxy_instrumented",
            "value": 0.00754782987473412,
            "unit": "iter/sec",
            "range": "stddev: 0.6672138647462356",
            "extra": "mean: 132.48841277509928 sec\nrounds: 5"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "lessoxx@gmail.com",
            "name": "Yuxuan",
            "username": "Essoz"
          },
          "committer": {
            "email": "lessoxx@gmail.com",
            "name": "Yuxuan",
            "username": "Essoz"
          },
          "distinct": true,
          "id": "b2130b644c96f44fc5384176060c1fcd706a3f09",
          "message": "add: meta_vars dumping for API trace",
          "timestamp": "2024-08-05T00:35:27-04:00",
          "tree_id": "70a2eec5ecafece25d413890a2b0d7f48e4a8952",
          "url": "https://github.com/OrderLab/ml-daikon/commit/b2130b644c96f44fc5384176060c1fcd706a3f09"
        },
        "date": 1722842077802,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/bench_instrumentor/bench.py::test_instrumented",
            "value": 0.00443504149492051,
            "unit": "iter/sec",
            "range": "stddev: 0.6540965133829938",
            "extra": "mean: 225.4770335622132 sec\nrounds: 5"
          },
          {
            "name": "tests/bench_instrumentor/bench.py::test_instrumented_with_jit_and_c_tracing_disabled",
            "value": 0.006220968067071679,
            "unit": "iter/sec",
            "range": "stddev: 0.3118828363496354",
            "extra": "mean: 160.74668592065572 sec\nrounds: 5"
          },
          {
            "name": "tests/bench_instrumentor/bench.py::test_sampler_instrumented",
            "value": 0.0012597174380599607,
            "unit": "iter/sec",
            "range": "stddev: 1.185191481626938",
            "extra": "mean: 793.8288141347468 sec\nrounds: 5"
          },
          {
            "name": "tests/bench_instrumentor/bench.py::test_proxy_instrumented",
            "value": 0.005616212240445397,
            "unit": "iter/sec",
            "range": "stddev: 0.4883888648757184",
            "extra": "mean: 178.05594895407557 sec\nrounds: 5"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "lessoxx@gmail.com",
            "name": "Yuxuan",
            "username": "Essoz"
          },
          "committer": {
            "email": "lessoxx@gmail.com",
            "name": "Yuxuan",
            "username": "Essoz"
          },
          "distinct": true,
          "id": "2534b6e8e2b6775916f08e38d4dfe6d45e3b4951",
          "message": "fix: avoid dumping meta_vars for frames with no useful local vars and remove global vars (e.g. ranks) from meta_vars",
          "timestamp": "2024-08-05T03:02:38-04:00",
          "tree_id": "8616bdb5a9197817536fb1fc8a1f69d9c9f1891b",
          "url": "https://github.com/OrderLab/ml-daikon/commit/2534b6e8e2b6775916f08e38d4dfe6d45e3b4951"
        },
        "date": 1722851602245,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/bench_instrumentor/bench.py::test_instrumented",
            "value": 0.004437816739279041,
            "unit": "iter/sec",
            "range": "stddev: 0.6562980764345713",
            "extra": "mean: 225.33602867126464 sec\nrounds: 5"
          },
          {
            "name": "tests/bench_instrumentor/bench.py::test_instrumented_with_jit_and_c_tracing_disabled",
            "value": 0.006245000793317178,
            "unit": "iter/sec",
            "range": "stddev: 0.5309704961443591",
            "extra": "mean: 160.12808214053513 sec\nrounds: 5"
          },
          {
            "name": "tests/bench_instrumentor/bench.py::test_sampler_instrumented",
            "value": 0.0012619125512661129,
            "unit": "iter/sec",
            "range": "stddev: 1.5229697083648392",
            "extra": "mean: 792.447938644141 sec\nrounds: 5"
          },
          {
            "name": "tests/bench_instrumentor/bench.py::test_proxy_instrumented",
            "value": 0.005617884724651108,
            "unit": "iter/sec",
            "range": "stddev: 0.41585104952074964",
            "extra": "mean: 178.00294043272734 sec\nrounds: 5"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "lessoxx@gmail.com",
            "name": "Yuxuan Jiang",
            "username": "Essoz"
          },
          "committer": {
            "email": "lessoxx@gmail.com",
            "name": "Yuxuan Jiang",
            "username": "Essoz"
          },
          "distinct": true,
          "id": "5fe150cc67db8a56469bc1b49752e3ae2767e6ab",
          "message": "fix: forbid hash dumping for tensor on platforms without CUDA",
          "timestamp": "2024-08-06T16:59:14-04:00",
          "tree_id": "9f0fd39e1a378ef7980e222c74ac77bd7a81a2aa",
          "url": "https://github.com/OrderLab/ml-daikon/commit/5fe150cc67db8a56469bc1b49752e3ae2767e6ab"
        },
        "date": 1722987456875,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/bench_instrumentor/bench.py::test_instrumented",
            "value": 0.004450847917532575,
            "unit": "iter/sec",
            "range": "stddev: 0.23873991445255732",
            "extra": "mean: 224.67629056945444 sec\nrounds: 5"
          },
          {
            "name": "tests/bench_instrumentor/bench.py::test_instrumented_with_jit_and_c_tracing_disabled",
            "value": 0.006255847835017341,
            "unit": "iter/sec",
            "range": "stddev: 0.6756657351844975",
            "extra": "mean: 159.85043536424638 sec\nrounds: 5"
          },
          {
            "name": "tests/bench_instrumentor/bench.py::test_sampler_instrumented",
            "value": 0.0012623447818367742,
            "unit": "iter/sec",
            "range": "stddev: 1.078525786139114",
            "extra": "mean: 792.1766021363437 sec\nrounds: 5"
          },
          {
            "name": "tests/bench_instrumentor/bench.py::test_proxy_instrumented",
            "value": 0.0056182158342390365,
            "unit": "iter/sec",
            "range": "stddev: 0.4094480963592095",
            "extra": "mean: 177.99244982823728 sec\nrounds: 5"
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
          "id": "11e097a4c84011516fd44669d56ebb02d1efdbcc",
          "message": "add: allow the functions instrumented to be a subset of the functions required by the provided invariants; TBD: report the invariants that definitely won't be active",
          "timestamp": "2024-08-18T19:07:19-04:00",
          "tree_id": "5c433316b6da79f5b6d2fe34b9f422ec3813a796",
          "url": "https://github.com/OrderLab/ml-daikon/commit/11e097a4c84011516fd44669d56ebb02d1efdbcc"
        },
        "date": 1724031984775,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/bench_instrumentor/bench.py::test_instrumented",
            "value": 0.004442207094157665,
            "unit": "iter/sec",
            "range": "stddev: 0.18616836985586843",
            "extra": "mean: 225.11332290545107 sec\nrounds: 5"
          },
          {
            "name": "tests/bench_instrumentor/bench.py::test_instrumented_with_jit_and_c_tracing_disabled",
            "value": 0.0062419015952188935,
            "unit": "iter/sec",
            "range": "stddev: 0.9488298355737413",
            "extra": "mean: 160.20758814364672 sec\nrounds: 5"
          },
          {
            "name": "tests/bench_instrumentor/bench.py::test_sampler_instrumented",
            "value": 0.001262445937060341,
            "unit": "iter/sec",
            "range": "stddev: 0.24845607824626165",
            "extra": "mean: 792.1131278924644 sec\nrounds: 5"
          },
          {
            "name": "tests/bench_instrumentor/bench.py::test_proxy_instrumented",
            "value": 0.005611460494497776,
            "unit": "iter/sec",
            "range": "stddev: 0.4522448055683355",
            "extra": "mean: 178.20672550052404 sec\nrounds: 5"
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
          "id": "8f08b90ae5e15740a308cc1003a70af0576285f4",
          "message": "fix: function replacement logic",
          "timestamp": "2024-08-18T23:51:25-04:00",
          "tree_id": "e32e19980f5a8e36d0497af90c93cc7d30cfafc1",
          "url": "https://github.com/OrderLab/ml-daikon/commit/8f08b90ae5e15740a308cc1003a70af0576285f4"
        },
        "date": 1724049036432,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/bench_instrumentor/bench.py::test_instrumented",
            "value": 0.004432723649170818,
            "unit": "iter/sec",
            "range": "stddev: 0.3831944016337906",
            "extra": "mean: 225.59493420869111 sec\nrounds: 5"
          },
          {
            "name": "tests/bench_instrumentor/bench.py::test_instrumented_with_jit_and_c_tracing_disabled",
            "value": 0.006228902625752711,
            "unit": "iter/sec",
            "range": "stddev: 0.311531399543783",
            "extra": "mean: 160.54192208200692 sec\nrounds: 5"
          },
          {
            "name": "tests/bench_instrumentor/bench.py::test_sampler_instrumented",
            "value": 0.0012618293079710503,
            "unit": "iter/sec",
            "range": "stddev: 1.6910530466751001",
            "extra": "mean: 792.5002166956663 sec\nrounds: 5"
          },
          {
            "name": "tests/bench_instrumentor/bench.py::test_proxy_instrumented",
            "value": 0.005619292240452602,
            "unit": "iter/sec",
            "range": "stddev: 0.5836324983111632",
            "extra": "mean: 177.95835439935325 sec\nrounds: 5"
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
          "id": "109f4d663a0ffcf0d6b1cf3636569ebb196f5075",
          "message": "[Feat] Conditional dumping",
          "timestamp": "2024-08-22T18:06:15Z",
          "url": "https://github.com/OrderLab/ml-daikon/pull/83/commits/109f4d663a0ffcf0d6b1cf3636569ebb196f5075"
        },
        "date": 1724391431847,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/bench_instrumentor/bench.py::test_instrumented",
            "value": 0.00440471454144202,
            "unit": "iter/sec",
            "range": "stddev: 0.3962410850396977",
            "extra": "mean: 227.02946821898223 sec\nrounds: 5"
          },
          {
            "name": "tests/bench_instrumentor/bench.py::test_instrumented_with_jit_and_c_tracing_disabled",
            "value": 0.006201586272286462,
            "unit": "iter/sec",
            "range": "stddev: 0.4331605141065998",
            "extra": "mean: 161.24906694740056 sec\nrounds: 5"
          },
          {
            "name": "tests/bench_instrumentor/bench.py::test_sampler_instrumented",
            "value": 0.0012617161119832748,
            "unit": "iter/sec",
            "range": "stddev: 0.7468612457337086",
            "extra": "mean: 792.5713165603578 sec\nrounds: 5"
          },
          {
            "name": "tests/bench_instrumentor/bench.py::test_proxy_instrumented",
            "value": 0.005289689286931356,
            "unit": "iter/sec",
            "range": "stddev: 0.4002258182333703",
            "extra": "mean: 189.04702067673207 sec\nrounds: 5"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "lessoxx@gmail.com",
            "name": "Yuxuan Jiang",
            "username": "Essoz"
          },
          "committer": {
            "email": "lessoxx@gmail.com",
            "name": "Yuxuan Jiang",
            "username": "Essoz"
          },
          "distinct": true,
          "id": "cf4576da3364a08e00afe41e82e3bd882c5370a0",
          "message": "add: bench for cond dump; fix: incorrect proxy instru workload",
          "timestamp": "2024-08-23T10:24:40-04:00",
          "tree_id": "f750d511ce94cc6186e8c31fa4fc8724c5b332d3",
          "url": "https://github.com/OrderLab/ml-daikon/commit/cf4576da3364a08e00afe41e82e3bd882c5370a0"
        },
        "date": 1724432665681,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/bench_instrumentor/bench.py::test_instrumented",
            "value": 0.0044298038786821205,
            "unit": "iter/sec",
            "range": "stddev: 0.6796468195306095",
            "extra": "mean: 225.7436282478273 sec\nrounds: 5"
          },
          {
            "name": "tests/bench_instrumentor/bench.py::test_instrumented_with_jit_and_c_tracing_disabled",
            "value": 0.006217994782174721,
            "unit": "iter/sec",
            "range": "stddev: 0.8382055983356897",
            "extra": "mean: 160.8235508441925 sec\nrounds: 5"
          },
          {
            "name": "tests/bench_instrumentor/bench.py::test_sampler_instrumented",
            "value": 0.0012631403273726024,
            "unit": "iter/sec",
            "range": "stddev: 1.7633020901183847",
            "extra": "mean: 791.6776769213378 sec\nrounds: 5"
          },
          {
            "name": "tests/bench_instrumentor/bench.py::test_proxy_instrumented",
            "value": 0.0053159120971862225,
            "unit": "iter/sec",
            "range": "stddev: 0.45841039827447816",
            "extra": "mean: 188.11447249650956 sec\nrounds: 5"
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
          "id": "109f4d663a0ffcf0d6b1cf3636569ebb196f5075",
          "message": "[Feat] Conditional dumping",
          "timestamp": "2024-08-22T18:06:15Z",
          "url": "https://github.com/OrderLab/ml-daikon/pull/83/commits/109f4d663a0ffcf0d6b1cf3636569ebb196f5075"
        },
        "date": 1724447398634,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/bench_instrumentor/bench.py::test_instrumented",
            "value": 0.004416301239576122,
            "unit": "iter/sec",
            "range": "stddev: 0.7027706972661939",
            "extra": "mean: 226.43382906913757 sec\nrounds: 5"
          },
          {
            "name": "tests/bench_instrumentor/bench.py::test_instrumented_with_jit_and_c_tracing_disabled",
            "value": 0.00619383921636515,
            "unit": "iter/sec",
            "range": "stddev: 0.4904987101510223",
            "extra": "mean: 161.4507521212101 sec\nrounds: 5"
          },
          {
            "name": "tests/bench_instrumentor/bench.py::test_sampler_instrumented",
            "value": 0.0012636277736778705,
            "unit": "iter/sec",
            "range": "stddev: 0.8434964612145084",
            "extra": "mean: 791.3722860723734 sec\nrounds: 5"
          },
          {
            "name": "tests/bench_instrumentor/bench.py::test_proxy_instrumented",
            "value": 0.005280282116464393,
            "unit": "iter/sec",
            "range": "stddev: 0.4666362790898981",
            "extra": "mean: 189.38382039889694 sec\nrounds: 5"
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
          "id": "94a41d94cfb6605023cc5987089a801945f37453",
          "message": "[Feat] Conditional dumping",
          "timestamp": "2024-08-23T18:27:49Z",
          "url": "https://github.com/OrderLab/ml-daikon/pull/83/commits/94a41d94cfb6605023cc5987089a801945f37453"
        },
        "date": 1724460688563,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/bench_instrumentor/bench.py::test_instrumented",
            "value": 0.004398396118398149,
            "unit": "iter/sec",
            "range": "stddev: 0.5917944389701393",
            "extra": "mean: 227.35560260638596 sec\nrounds: 5"
          },
          {
            "name": "tests/bench_instrumentor/bench.py::test_instrumented_with_jit_and_c_tracing_disabled",
            "value": 0.0060982787561624585,
            "unit": "iter/sec",
            "range": "stddev: 0.3877031091644168",
            "extra": "mean: 163.98069684654473 sec\nrounds: 5"
          },
          {
            "name": "tests/bench_instrumentor/bench.py::test_instrumented_with_cond_dump_jit_and_c_tracing_disabled",
            "value": 0.006773700428546647,
            "unit": "iter/sec",
            "range": "stddev: 0.43460939437063645",
            "extra": "mean: 147.629794164747 sec\nrounds: 5"
          },
          {
            "name": "tests/bench_instrumentor/bench.py::test_sampler_instrumented",
            "value": 0.0012568130348850157,
            "unit": "iter/sec",
            "range": "stddev: 1.821746356709573",
            "extra": "mean: 795.6632945738733 sec\nrounds: 5"
          },
          {
            "name": "tests/bench_instrumentor/bench.py::test_proxy_instrumented",
            "value": 0.005222684006373375,
            "unit": "iter/sec",
            "range": "stddev: 0.7297575063725368",
            "extra": "mean: 191.47243041694165 sec\nrounds: 5"
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
          "id": "71d297d0977436d432f71a37e1d68134d5ab4143",
          "message": "[Feat] Conditional dumping",
          "timestamp": "2024-08-23T18:27:49Z",
          "url": "https://github.com/OrderLab/ml-daikon/pull/83/commits/71d297d0977436d432f71a37e1d68134d5ab4143"
        },
        "date": 1724478652312,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/bench_instrumentor/bench.py::test_instrumented",
            "value": 0.004382469252097749,
            "unit": "iter/sec",
            "range": "stddev: 0.7228618114228748",
            "extra": "mean: 228.18186334595083 sec\nrounds: 5"
          },
          {
            "name": "tests/bench_instrumentor/bench.py::test_instrumented_with_jit_and_c_tracing_disabled",
            "value": 0.006087680488886789,
            "unit": "iter/sec",
            "range": "stddev: 0.38535644594917606",
            "extra": "mean: 164.2661768838763 sec\nrounds: 5"
          },
          {
            "name": "tests/bench_instrumentor/bench.py::test_instrumented_with_cond_dump_jit_and_c_tracing_disabled",
            "value": 0.0067638948257149785,
            "unit": "iter/sec",
            "range": "stddev: 0.7472461222020393",
            "extra": "mean: 147.84381273910404 sec\nrounds: 5"
          },
          {
            "name": "tests/bench_instrumentor/bench.py::test_sampler_instrumented",
            "value": 0.0012573998672645481,
            "unit": "iter/sec",
            "range": "stddev: 1.5068607066343127",
            "extra": "mean: 795.2919560708106 sec\nrounds: 5"
          },
          {
            "name": "tests/bench_instrumentor/bench.py::test_proxy_instrumented",
            "value": 0.14749394982779665,
            "unit": "iter/sec",
            "range": "stddev: 0.02991808814210244",
            "extra": "mean: 6.779939117282629 sec\nrounds: 5"
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
          "id": "f82e9a4e3ec5c27d186b003b743b9e2ec4ec88b8",
          "message": "[Feat] Conditional dumping",
          "timestamp": "2024-08-23T18:27:49Z",
          "url": "https://github.com/OrderLab/ml-daikon/pull/83/commits/f82e9a4e3ec5c27d186b003b743b9e2ec4ec88b8"
        },
        "date": 1724488100264,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/bench_instrumentor/bench.py::test_instrumented",
            "value": 0.004379517343016683,
            "unit": "iter/sec",
            "range": "stddev: 0.3950126346800328",
            "extra": "mean: 228.33566388189791 sec\nrounds: 5"
          },
          {
            "name": "tests/bench_instrumentor/bench.py::test_instrumented_with_jit_and_c_tracing_disabled",
            "value": 0.0060802509410860855,
            "unit": "iter/sec",
            "range": "stddev: 0.34070259315213897",
            "extra": "mean: 164.46689613461496 sec\nrounds: 5"
          },
          {
            "name": "tests/bench_instrumentor/bench.py::test_instrumented_with_cond_dump_jit_and_c_tracing_disabled",
            "value": 0.00676284170078721,
            "unit": "iter/sec",
            "range": "stddev: 0.49847410197074526",
            "extra": "mean: 147.86683531031014 sec\nrounds: 5"
          },
          {
            "name": "tests/bench_instrumentor/bench.py::test_sampler_instrumented",
            "value": 0.0012561891699155595,
            "unit": "iter/sec",
            "range": "stddev: 1.2111600845886588",
            "extra": "mean: 796.0584472060203 sec\nrounds: 5"
          },
          {
            "name": "tests/bench_instrumentor/bench.py::test_proxy_instrumented",
            "value": 0.14792727390332058,
            "unit": "iter/sec",
            "range": "stddev: 0.03738022007893396",
            "extra": "mean: 6.760078608989716 sec\nrounds: 5"
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
          "id": "0fe98176d67ac9a2c0608f41da9041586a50776d",
          "message": "[Feat] Conditional dumping",
          "timestamp": "2024-08-23T18:27:49Z",
          "url": "https://github.com/OrderLab/ml-daikon/pull/83/commits/0fe98176d67ac9a2c0608f41da9041586a50776d"
        },
        "date": 1724497566305,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/bench_instrumentor/bench.py::test_instrumented",
            "value": 0.004349114268350717,
            "unit": "iter/sec",
            "range": "stddev: 0.2989375898442272",
            "extra": "mean: 229.93187538832427 sec\nrounds: 5"
          },
          {
            "name": "tests/bench_instrumentor/bench.py::test_instrumented_with_jit_and_c_tracing_disabled",
            "value": 0.006054304159889443,
            "unit": "iter/sec",
            "range": "stddev: 0.3464637734730243",
            "extra": "mean: 165.17174783274533 sec\nrounds: 5"
          },
          {
            "name": "tests/bench_instrumentor/bench.py::test_instrumented_with_cond_dump_jit_and_c_tracing_disabled",
            "value": 0.006742303050779417,
            "unit": "iter/sec",
            "range": "stddev: 0.4814696711239988",
            "extra": "mean: 148.3172726690769 sec\nrounds: 5"
          },
          {
            "name": "tests/bench_instrumentor/bench.py::test_sampler_instrumented",
            "value": 0.0012554449314435992,
            "unit": "iter/sec",
            "range": "stddev: 1.0611472914961482",
            "extra": "mean: 796.5303574487567 sec\nrounds: 5"
          },
          {
            "name": "tests/bench_instrumentor/bench.py::test_proxy_instrumented",
            "value": 0.14785593138826658,
            "unit": "iter/sec",
            "range": "stddev: 0.02970889060884539",
            "extra": "mean: 6.763340439647436 sec\nrounds: 5"
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
          "id": "e8cab6f6ec6f31c216efceb8d7552cddae1926d0",
          "message": "[Feat] Conditional dumping",
          "timestamp": "2024-08-25T01:03:31Z",
          "url": "https://github.com/OrderLab/ml-daikon/pull/83/commits/e8cab6f6ec6f31c216efceb8d7552cddae1926d0"
        },
        "date": 1724575745604,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/bench_instrumentor/bench.py::test_instrumented",
            "value": 0.0040039686310736735,
            "unit": "iter/sec",
            "range": "stddev: 3.4517476995783443",
            "extra": "mean: 249.75220640823244 sec\nrounds: 5"
          },
          {
            "name": "tests/bench_instrumentor/bench.py::test_instrumented_with_jit_and_c_tracing_disabled",
            "value": 0.005739754163539486,
            "unit": "iter/sec",
            "range": "stddev: 0.6143332878291226",
            "extra": "mean: 174.22348963171243 sec\nrounds: 5"
          },
          {
            "name": "tests/bench_instrumentor/bench.py::test_instrumented_with_cond_dump_jit_and_c_tracing_disabled",
            "value": 0.00639887278816685,
            "unit": "iter/sec",
            "range": "stddev: 0.6582812042289617",
            "extra": "mean: 156.2775246679783 sec\nrounds: 5"
          },
          {
            "name": "tests/bench_instrumentor/bench.py::test_sampler_instrumented",
            "value": 0.001240055894655097,
            "unit": "iter/sec",
            "range": "stddev: 1.0073483886751549",
            "extra": "mean: 806.4152626588941 sec\nrounds: 5"
          },
          {
            "name": "tests/bench_instrumentor/bench.py::test_proxy_instrumented",
            "value": 0.0046599252919022945,
            "unit": "iter/sec",
            "range": "stddev: 1.1984124556661915",
            "extra": "mean: 214.59571502953767 sec\nrounds: 5"
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
          "id": "5f1538e38ebd9c0ffbe9ec8bef7ac429323ff5a7",
          "message": "[Feat] Conditional dumping",
          "timestamp": "2024-08-25T01:03:31Z",
          "url": "https://github.com/OrderLab/ml-daikon/pull/83/commits/5f1538e38ebd9c0ffbe9ec8bef7ac429323ff5a7"
        },
        "date": 1724633873342,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/bench_instrumentor/bench.py::test_instrumented",
            "value": 0.004034444467077156,
            "unit": "iter/sec",
            "range": "stddev: 0.45339772705166753",
            "extra": "mean: 247.86560037210583 sec\nrounds: 5"
          },
          {
            "name": "tests/bench_instrumentor/bench.py::test_instrumented_with_jit_and_c_tracing_disabled",
            "value": 0.005702765851827101,
            "unit": "iter/sec",
            "range": "stddev: 0.47248508677798584",
            "extra": "mean: 175.35350845232605 sec\nrounds: 5"
          },
          {
            "name": "tests/bench_instrumentor/bench.py::test_instrumented_with_cond_dump_jit_and_c_tracing_disabled",
            "value": 0.006367498751476233,
            "unit": "iter/sec",
            "range": "stddev: 0.6709789272805436",
            "extra": "mean: 157.04753766432404 sec\nrounds: 5"
          },
          {
            "name": "tests/bench_instrumentor/bench.py::test_sampler_instrumented",
            "value": 0.0012388571974823422,
            "unit": "iter/sec",
            "range": "stddev: 0.6300939532579307",
            "extra": "mean: 807.1955363638699 sec\nrounds: 5"
          },
          {
            "name": "tests/bench_instrumentor/bench.py::test_proxy_instrumented",
            "value": 0.00467065459303835,
            "unit": "iter/sec",
            "range": "stddev: 0.6375935634062314",
            "extra": "mean: 214.1027515694499 sec\nrounds: 5"
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
          "id": "ebf0d4f8078059f25d054c4b5a231091195ddbee",
          "message": "Merge pull request #83 from OrderLab/conditional_dumping\n\n[Feat] Conditional dumping [Incomplete Yet but need to sync some new features to main]",
          "timestamp": "2024-09-15T19:57:36-04:00",
          "tree_id": "157c83744c826e9be142b337886f0b1bb00c0b6d",
          "url": "https://github.com/OrderLab/ml-daikon/commit/ebf0d4f8078059f25d054c4b5a231091195ddbee"
        },
        "date": 1726454755705,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/bench_instrumentor/bench.py::test_instrumented",
            "value": 0.003973070101626357,
            "unit": "iter/sec",
            "range": "stddev: 7.265289155892346",
            "extra": "mean: 251.69452700838445 sec\nrounds: 5"
          },
          {
            "name": "tests/bench_instrumentor/bench.py::test_instrumented_with_jit_and_c_tracing_disabled",
            "value": 0.005700130836062428,
            "unit": "iter/sec",
            "range": "stddev: 0.42523336111677945",
            "extra": "mean: 175.434569619596 sec\nrounds: 5"
          },
          {
            "name": "tests/bench_instrumentor/bench.py::test_instrumented_with_cond_dump_jit_and_c_tracing_disabled",
            "value": 0.006396956532365812,
            "unit": "iter/sec",
            "range": "stddev: 0.754242406604459",
            "extra": "mean: 156.32433876022696 sec\nrounds: 5"
          },
          {
            "name": "tests/bench_instrumentor/bench.py::test_sampler_instrumented",
            "value": 0.0015476942545362747,
            "unit": "iter/sec",
            "range": "stddev: 0.8493130795640865",
            "extra": "mean: 646.1224476791919 sec\nrounds: 5"
          },
          {
            "name": "tests/bench_instrumentor/bench.py::test_proxy_instrumented",
            "value": 0.0047648474662683096,
            "unit": "iter/sec",
            "range": "stddev: 0.6796677926786431",
            "extra": "mean: 209.87030688375233 sec\nrounds: 5"
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
          "id": "2273c942346c6b11113526201e9c97c0fda4a0f0",
          "message": "Accounting for Fine-grained Events in APIContainRelation",
          "timestamp": "2024-09-17T03:05:11Z",
          "url": "https://github.com/OrderLab/ml-daikon/pull/92/commits/2273c942346c6b11113526201e9c97c0fda4a0f0"
        },
        "date": 1726867858023,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/bench_instrumentor/bench.py::test_instrumented",
            "value": 0.0040139151418615605,
            "unit": "iter/sec",
            "range": "stddev: 0.47764745298647737",
            "extra": "mean: 249.13331863219793 sec\nrounds: 5"
          },
          {
            "name": "tests/bench_instrumentor/bench.py::test_instrumented_with_jit_and_c_tracing_disabled",
            "value": 0.0057025363497744536,
            "unit": "iter/sec",
            "range": "stddev: 0.6997917078457837",
            "extra": "mean: 175.3605656611995 sec\nrounds: 5"
          },
          {
            "name": "tests/bench_instrumentor/bench.py::test_instrumented_with_cond_dump_jit_and_c_tracing_disabled",
            "value": 0.006461720303103385,
            "unit": "iter/sec",
            "range": "stddev: 1.130018834940078",
            "extra": "mean: 154.75754955220327 sec\nrounds: 5"
          },
          {
            "name": "tests/bench_instrumentor/bench.py::test_sampler_instrumented",
            "value": 0.0015372624133609247,
            "unit": "iter/sec",
            "range": "stddev: 4.157496442123292",
            "extra": "mean: 650.507025546598 sec\nrounds: 5"
          },
          {
            "name": "tests/bench_instrumentor/bench.py::test_proxy_instrumented",
            "value": 0.004592327142508568,
            "unit": "iter/sec",
            "range": "stddev: 8.290475757029178",
            "extra": "mean: 217.75452161139984 sec\nrounds: 5"
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
          "id": "581c7a9d61fba22a7e743bb4bb67e129b2699f40",
          "message": "Accounting for Fine-grained Events in APIContainRelation",
          "timestamp": "2024-09-17T03:05:11Z",
          "url": "https://github.com/OrderLab/ml-daikon/pull/92/commits/581c7a9d61fba22a7e743bb4bb67e129b2699f40"
        },
        "date": 1726878005123,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/bench_instrumentor/bench.py::test_instrumented",
            "value": 0.0040012907290041515,
            "unit": "iter/sec",
            "range": "stddev: 0.321400720535749",
            "extra": "mean: 249.91935545980232 sec\nrounds: 5"
          },
          {
            "name": "tests/bench_instrumentor/bench.py::test_instrumented_with_jit_and_c_tracing_disabled",
            "value": 0.0056625554330632395,
            "unit": "iter/sec",
            "range": "stddev: 0.49263038104520035",
            "extra": "mean: 176.59871268739806 sec\nrounds: 5"
          },
          {
            "name": "tests/bench_instrumentor/bench.py::test_instrumented_with_cond_dump_jit_and_c_tracing_disabled",
            "value": 0.006380075979226173,
            "unit": "iter/sec",
            "range": "stddev: 0.8484843442308422",
            "extra": "mean: 156.7379453247966 sec\nrounds: 5"
          },
          {
            "name": "tests/bench_instrumentor/bench.py::test_sampler_instrumented",
            "value": 0.0015437819630181261,
            "unit": "iter/sec",
            "range": "stddev: 1.4150398978325844",
            "extra": "mean: 647.7598676207999 sec\nrounds: 5"
          },
          {
            "name": "tests/bench_instrumentor/bench.py::test_proxy_instrumented",
            "value": 0.004741138460706832,
            "unit": "iter/sec",
            "range": "stddev: 0.8133744801577067",
            "extra": "mean: 210.9198050821986 sec\nrounds: 5"
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
          "id": "5a832e1c4389ed29e9c2487c3c915e7e2c7e5119",
          "message": "Accounting for Fine-grained Events in APIContainRelation",
          "timestamp": "2024-09-17T03:05:11Z",
          "url": "https://github.com/OrderLab/ml-daikon/pull/92/commits/5a832e1c4389ed29e9c2487c3c915e7e2c7e5119"
        },
        "date": 1726888164742,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/bench_instrumentor/bench.py::test_instrumented",
            "value": 0.003999449067185096,
            "unit": "iter/sec",
            "range": "stddev: 0.48354890729059086",
            "extra": "mean: 250.03443804419368 sec\nrounds: 5"
          },
          {
            "name": "tests/bench_instrumentor/bench.py::test_instrumented_with_jit_and_c_tracing_disabled",
            "value": 0.0056623899127030706,
            "unit": "iter/sec",
            "range": "stddev: 0.514192011822148",
            "extra": "mean: 176.6038749391999 sec\nrounds: 5"
          },
          {
            "name": "tests/bench_instrumentor/bench.py::test_instrumented_with_cond_dump_jit_and_c_tracing_disabled",
            "value": 0.006371359300916357,
            "unit": "iter/sec",
            "range": "stddev: 0.9530074696352812",
            "extra": "mean: 156.95237904039968 sec\nrounds: 5"
          },
          {
            "name": "tests/bench_instrumentor/bench.py::test_sampler_instrumented",
            "value": 0.0015423537926055298,
            "unit": "iter/sec",
            "range": "stddev: 1.5800497069854287",
            "extra": "mean: 648.3596725954034 sec\nrounds: 5"
          },
          {
            "name": "tests/bench_instrumentor/bench.py::test_proxy_instrumented",
            "value": 0.0047390203489759124,
            "unit": "iter/sec",
            "range": "stddev: 0.5340514769593974",
            "extra": "mean: 211.01407598220104 sec\nrounds: 5"
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
          "id": "9447aa307afff64c19796acf1884ca7df3385b90",
          "message": "Accounting for Fine-grained Events in APIContainRelation",
          "timestamp": "2024-09-21T03:01:23Z",
          "url": "https://github.com/OrderLab/ml-daikon/pull/92/commits/9447aa307afff64c19796acf1884ca7df3385b90"
        },
        "date": 1726944959505,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/bench_instrumentor/bench.py::test_instrumented",
            "value": 0.004000469027422101,
            "unit": "iter/sec",
            "range": "stddev: 0.205576668078835",
            "extra": "mean: 249.97068922300824 sec\nrounds: 5"
          },
          {
            "name": "tests/bench_instrumentor/bench.py::test_instrumented_with_jit_and_c_tracing_disabled",
            "value": 0.005646348844357925,
            "unit": "iter/sec",
            "range": "stddev: 0.45069905978762775",
            "extra": "mean: 177.10560001960258 sec\nrounds: 5"
          },
          {
            "name": "tests/bench_instrumentor/bench.py::test_instrumented_with_cond_dump_jit_and_c_tracing_disabled",
            "value": 0.006352269531940711,
            "unit": "iter/sec",
            "range": "stddev: 0.6896079821368837",
            "extra": "mean: 157.42405056519783 sec\nrounds: 5"
          },
          {
            "name": "tests/bench_instrumentor/bench.py::test_sampler_instrumented",
            "value": 0.0015416836271201035,
            "unit": "iter/sec",
            "range": "stddev: 1.5420092315538085",
            "extra": "mean: 648.6415126999957 sec\nrounds: 5"
          },
          {
            "name": "tests/bench_instrumentor/bench.py::test_proxy_instrumented",
            "value": 0.0047413681466080965,
            "unit": "iter/sec",
            "range": "stddev: 0.22268417543106256",
            "extra": "mean: 210.90958750279387 sec\nrounds: 5"
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
          "id": "b1302c2a51a6b2fecb8f53727edf0254fe0e65a8",
          "message": "Accounting for Fine-grained Events in APIContainRelation",
          "timestamp": "2024-09-21T03:01:23Z",
          "url": "https://github.com/OrderLab/ml-daikon/pull/92/commits/b1302c2a51a6b2fecb8f53727edf0254fe0e65a8"
        },
        "date": 1726979190399,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/bench_instrumentor/bench.py::test_instrumented",
            "value": 0.003983470943680488,
            "unit": "iter/sec",
            "range": "stddev: 1.4021784311249166",
            "extra": "mean: 251.0373526350013 sec\nrounds: 5"
          },
          {
            "name": "tests/bench_instrumentor/bench.py::test_instrumented_with_jit_and_c_tracing_disabled",
            "value": 0.005651322625509311,
            "unit": "iter/sec",
            "range": "stddev: 0.7823544918250664",
            "extra": "mean: 176.94972774800263 sec\nrounds: 5"
          },
          {
            "name": "tests/bench_instrumentor/bench.py::test_instrumented_with_cond_dump_jit_and_c_tracing_disabled",
            "value": 0.006379732498343144,
            "unit": "iter/sec",
            "range": "stddev: 0.3768611201218074",
            "extra": "mean: 156.7463839995966 sec\nrounds: 5"
          },
          {
            "name": "tests/bench_instrumentor/bench.py::test_sampler_instrumented",
            "value": 0.0015436357919748121,
            "unit": "iter/sec",
            "range": "stddev: 1.2561114938669289",
            "extra": "mean: 647.8212057526049 sec\nrounds: 5"
          },
          {
            "name": "tests/bench_instrumentor/bench.py::test_proxy_instrumented",
            "value": 0.004731772411500874,
            "unit": "iter/sec",
            "range": "stddev: 0.669722993900018",
            "extra": "mean: 211.33729880360187 sec\nrounds: 5"
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
          "id": "fb4ec464bca4fa5f4f8857e20752b0481e654ef7",
          "message": "Accounting for Fine-grained Events in APIContainRelation",
          "timestamp": "2024-09-21T03:01:23Z",
          "url": "https://github.com/OrderLab/ml-daikon/pull/92/commits/fb4ec464bca4fa5f4f8857e20752b0481e654ef7"
        },
        "date": 1726989354637,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/bench_instrumentor/bench.py::test_instrumented",
            "value": 0.003973254631274398,
            "unit": "iter/sec",
            "range": "stddev: 0.9545394196524204",
            "extra": "mean: 251.68283757320023 sec\nrounds: 5"
          },
          {
            "name": "tests/bench_instrumentor/bench.py::test_instrumented_with_jit_and_c_tracing_disabled",
            "value": 0.00565764235905959,
            "unit": "iter/sec",
            "range": "stddev: 0.6412145529414051",
            "extra": "mean: 176.75207030340098 sec\nrounds: 5"
          },
          {
            "name": "tests/bench_instrumentor/bench.py::test_instrumented_with_cond_dump_jit_and_c_tracing_disabled",
            "value": 0.006371968553526147,
            "unit": "iter/sec",
            "range": "stddev: 0.7473452234116001",
            "extra": "mean: 156.937372116599 sec\nrounds: 5"
          },
          {
            "name": "tests/bench_instrumentor/bench.py::test_sampler_instrumented",
            "value": 0.0015406103577787313,
            "unit": "iter/sec",
            "range": "stddev: 1.0110837351807702",
            "extra": "mean: 649.0933901300072 sec\nrounds: 5"
          },
          {
            "name": "tests/bench_instrumentor/bench.py::test_proxy_instrumented",
            "value": 0.004737139367245262,
            "unit": "iter/sec",
            "range": "stddev: 0.787357880358223",
            "extra": "mean: 211.09786359980353 sec\nrounds: 5"
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
          "id": "dbc3ad2258e9546e480c1f5db287484b0fa6e21d",
          "message": "Accounting for Fine-grained Events in APIContainRelation",
          "timestamp": "2024-09-21T03:01:23Z",
          "url": "https://github.com/OrderLab/ml-daikon/pull/92/commits/dbc3ad2258e9546e480c1f5db287484b0fa6e21d"
        },
        "date": 1726999474022,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/bench_instrumentor/bench.py::test_instrumented",
            "value": 0.00400229156779706,
            "unit": "iter/sec",
            "range": "stddev: 0.3657381771404229",
            "extra": "mean: 249.85685901700055 sec\nrounds: 5"
          },
          {
            "name": "tests/bench_instrumentor/bench.py::test_instrumented_with_jit_and_c_tracing_disabled",
            "value": 0.005664690660140944,
            "unit": "iter/sec",
            "range": "stddev: 0.40869975947382253",
            "extra": "mean: 176.53214623640525 sec\nrounds: 5"
          },
          {
            "name": "tests/bench_instrumentor/bench.py::test_instrumented_with_cond_dump_jit_and_c_tracing_disabled",
            "value": 0.006369361307784166,
            "unit": "iter/sec",
            "range": "stddev: 1.005158985424914",
            "extra": "mean: 157.0016131410026 sec\nrounds: 5"
          },
          {
            "name": "tests/bench_instrumentor/bench.py::test_sampler_instrumented",
            "value": 0.0015420558090177938,
            "unit": "iter/sec",
            "range": "stddev: 1.0642633039767115",
            "extra": "mean: 648.4849602408009 sec\nrounds: 5"
          },
          {
            "name": "tests/bench_instrumentor/bench.py::test_proxy_instrumented",
            "value": 0.004739660814654318,
            "unit": "iter/sec",
            "range": "stddev: 0.6590083444260979",
            "extra": "mean: 210.98556185880443 sec\nrounds: 5"
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
          "id": "ffcb8d88445a527c80537b5d2e3ee26bd968d5b0",
          "message": "Merge pull request #92 from OrderLab/contain-specific-events\n\nAccounting for Fine-grained Events in APIContainRelation",
          "timestamp": "2024-09-22T00:57:55-04:00",
          "tree_id": "ccde3609893f2380ce70a93e3fc7416a841c7716",
          "url": "https://github.com/OrderLab/ml-daikon/commit/ffcb8d88445a527c80537b5d2e3ee26bd968d5b0"
        },
        "date": 1727009644487,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/bench_instrumentor/bench.py::test_instrumented",
            "value": 0.003992047370365905,
            "unit": "iter/sec",
            "range": "stddev: 0.43906257340394533",
            "extra": "mean: 250.49802951319725 sec\nrounds: 5"
          },
          {
            "name": "tests/bench_instrumentor/bench.py::test_instrumented_with_jit_and_c_tracing_disabled",
            "value": 0.0056762028084111115,
            "unit": "iter/sec",
            "range": "stddev: 0.26611747022557336",
            "extra": "mean: 176.17411388440524 sec\nrounds: 5"
          },
          {
            "name": "tests/bench_instrumentor/bench.py::test_instrumented_with_cond_dump_jit_and_c_tracing_disabled",
            "value": 0.006370462548713279,
            "unit": "iter/sec",
            "range": "stddev: 0.262380234273673",
            "extra": "mean: 156.97447278800536 sec\nrounds: 5"
          },
          {
            "name": "tests/bench_instrumentor/bench.py::test_sampler_instrumented",
            "value": 0.0015438677906478597,
            "unit": "iter/sec",
            "range": "stddev: 1.6701090905031524",
            "extra": "mean: 647.7238569634035 sec\nrounds: 5"
          },
          {
            "name": "tests/bench_instrumentor/bench.py::test_proxy_instrumented",
            "value": 0.0047314287933280215,
            "unit": "iter/sec",
            "range": "stddev: 0.402363767406598",
            "extra": "mean: 211.352647092595 sec\nrounds: 5"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "zimingzh@umich.edu",
            "name": "ziming-zh",
            "username": "ziming-zh"
          },
          "committer": {
            "email": "zimingzh@umich.edu",
            "name": "ziming-zh",
            "username": "ziming-zh"
          },
          "distinct": true,
          "id": "de643780ec45b586ae3838bcd171160a927f3275",
          "message": "Merge branch 'main' of https://github.com/OrderLab/ml-daikon",
          "timestamp": "2024-09-25T02:02:32-04:00",
          "tree_id": "8487c29d5f43763c1318fbe41146ebfc0ed5918a",
          "url": "https://github.com/OrderLab/ml-daikon/commit/de643780ec45b586ae3838bcd171160a927f3275"
        },
        "date": 1727254355490,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/bench_instrumentor/bench.py::test_instrumented",
            "value": 0.003973686884776992,
            "unit": "iter/sec",
            "range": "stddev: 0.7015751809494755",
            "extra": "mean: 251.6554597774055 sec\nrounds: 5"
          },
          {
            "name": "tests/bench_instrumentor/bench.py::test_instrumented_with_jit_and_c_tracing_disabled",
            "value": 0.005635248965680417,
            "unit": "iter/sec",
            "range": "stddev: 0.6266767630521052",
            "extra": "mean: 177.45444896759 sec\nrounds: 5"
          },
          {
            "name": "tests/bench_instrumentor/bench.py::test_instrumented_with_cond_dump_jit_and_c_tracing_disabled",
            "value": 0.006359938834163713,
            "unit": "iter/sec",
            "range": "stddev: 0.4147692385853901",
            "extra": "mean: 157.23421656640713 sec\nrounds: 5"
          },
          {
            "name": "tests/bench_instrumentor/bench.py::test_sampler_instrumented",
            "value": 0.0015389273034705554,
            "unit": "iter/sec",
            "range": "stddev: 1.9348367457457654",
            "extra": "mean: 649.8032738419948 sec\nrounds: 5"
          },
          {
            "name": "tests/bench_instrumentor/bench.py::test_proxy_instrumented",
            "value": 0.004718606832953108,
            "unit": "iter/sec",
            "range": "stddev: 0.7525241633235998",
            "extra": "mean: 211.92695967300097 sec\nrounds: 5"
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
          "id": "20feced068052fdc164fc5acd75e005e8123675d",
          "message": "Migration to Pandas",
          "timestamp": "2024-09-27T17:19:52Z",
          "url": "https://github.com/OrderLab/ml-daikon/pull/93/commits/20feced068052fdc164fc5acd75e005e8123675d"
        },
        "date": 1727496222993,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/bench_instrumentor/bench.py::test_instrumented",
            "value": 0.003973286902752212,
            "unit": "iter/sec",
            "range": "stddev: 0.2363287109950357",
            "extra": "mean: 251.68079337722156 sec\nrounds: 5"
          },
          {
            "name": "tests/bench_instrumentor/bench.py::test_instrumented_with_jit_and_c_tracing_disabled",
            "value": 0.0056288548130687295,
            "unit": "iter/sec",
            "range": "stddev: 0.3703495698524217",
            "extra": "mean: 177.65603008239995 sec\nrounds: 5"
          },
          {
            "name": "tests/bench_instrumentor/bench.py::test_instrumented_with_cond_dump_jit_and_c_tracing_disabled",
            "value": 0.006355310425446003,
            "unit": "iter/sec",
            "range": "stddev: 0.8736667992456464",
            "extra": "mean: 157.34872619220988 sec\nrounds: 5"
          },
          {
            "name": "tests/bench_instrumentor/bench.py::test_sampler_instrumented",
            "value": 0.0015378186447369208,
            "unit": "iter/sec",
            "range": "stddev: 2.3992127839574415",
            "extra": "mean: 650.2717361520045 sec\nrounds: 5"
          },
          {
            "name": "tests/bench_instrumentor/bench.py::test_proxy_instrumented",
            "value": 0.004699046824550665,
            "unit": "iter/sec",
            "range": "stddev: 0.42182865430458916",
            "extra": "mean: 212.8091158350231 sec\nrounds: 5"
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
          "id": "3a68b452bb09358f950ff6d0f6c935509bee714e",
          "message": "Merge pull request #93 from OrderLab/trace_pandas\n\nMigration to Pandas",
          "timestamp": "2024-09-27T21:16:54-04:00",
          "tree_id": "a7bab16c25e3cda7350980aba30566b5f50e35a8",
          "url": "https://github.com/OrderLab/ml-daikon/commit/3a68b452bb09358f950ff6d0f6c935509bee714e"
        },
        "date": 1727506446820,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/bench_instrumentor/bench.py::test_instrumented",
            "value": 0.003964928060282443,
            "unit": "iter/sec",
            "range": "stddev: 0.31505187738062607",
            "extra": "mean: 252.21138562820852 sec\nrounds: 5"
          },
          {
            "name": "tests/bench_instrumentor/bench.py::test_instrumented_with_jit_and_c_tracing_disabled",
            "value": 0.00562293269696867,
            "unit": "iter/sec",
            "range": "stddev: 0.26386724209453166",
            "extra": "mean: 177.84313878398387 sec\nrounds: 5"
          },
          {
            "name": "tests/bench_instrumentor/bench.py::test_instrumented_with_cond_dump_jit_and_c_tracing_disabled",
            "value": 0.006345341010510576,
            "unit": "iter/sec",
            "range": "stddev: 0.711774717926143",
            "extra": "mean: 157.59594296722207 sec\nrounds: 5"
          },
          {
            "name": "tests/bench_instrumentor/bench.py::test_sampler_instrumented",
            "value": 0.001540035510129645,
            "unit": "iter/sec",
            "range": "stddev: 1.135155342331043",
            "extra": "mean: 649.3356766272336 sec\nrounds: 5"
          },
          {
            "name": "tests/bench_instrumentor/bench.py::test_proxy_instrumented",
            "value": 0.0047001318645608055,
            "unit": "iter/sec",
            "range": "stddev: 0.8757055778886774",
            "extra": "mean: 212.75998819097876 sec\nrounds: 5"
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
          "id": "01cdeda13af32fb117c2dbc232d8374590e02775",
          "message": "fix: remove stack frame based meta_vars",
          "timestamp": "2024-09-28T17:19:01-04:00",
          "tree_id": "1e779f0025ef484fa4aae3fa068ac70f867ede78",
          "url": "https://github.com/OrderLab/ml-daikon/commit/01cdeda13af32fb117c2dbc232d8374590e02775"
        },
        "date": 1727566542230,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/bench_instrumentor/bench.py::test_instrumented",
            "value": 0.0063486417255987325,
            "unit": "iter/sec",
            "range": "stddev: 0.9119802646590538",
            "extra": "mean: 157.51400744002314 sec\nrounds: 5"
          },
          {
            "name": "tests/bench_instrumentor/bench.py::test_instrumented_with_jit_and_c_tracing_disabled",
            "value": 0.007537067962044801,
            "unit": "iter/sec",
            "range": "stddev: 0.42662533294175053",
            "extra": "mean: 132.67758829239756 sec\nrounds: 5"
          },
          {
            "name": "tests/bench_instrumentor/bench.py::test_instrumented_with_cond_dump_jit_and_c_tracing_disabled",
            "value": 0.0075641877023992134,
            "unit": "iter/sec",
            "range": "stddev: 0.3886537314573201",
            "extra": "mean: 132.20190182256047 sec\nrounds: 5"
          },
          {
            "name": "tests/bench_instrumentor/bench.py::test_sampler_instrumented",
            "value": 0.0016679183564070178,
            "unit": "iter/sec",
            "range": "stddev: 0.5777926388279123",
            "extra": "mean: 599.5497298525879 sec\nrounds: 5"
          },
          {
            "name": "tests/bench_instrumentor/bench.py::test_proxy_instrumented",
            "value": 0.0069274206942931665,
            "unit": "iter/sec",
            "range": "stddev: 0.6226271064694774",
            "extra": "mean: 144.35387197197414 sec\nrounds: 5"
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
          "id": "45baf2651721fbd328d98e83762b7a568f91f94c",
          "message": "Functionality: Var preserve invariant",
          "timestamp": "2024-09-28T21:19:09Z",
          "url": "https://github.com/OrderLab/ml-daikon/pull/96/commits/45baf2651721fbd328d98e83762b7a568f91f94c"
        },
        "date": 1727646287766,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/bench_instrumentor/bench.py::test_instrumented",
            "value": 0.4411667179622083,
            "unit": "iter/sec",
            "range": "stddev: 0.017297320644041304",
            "extra": "mean: 2.266716774599627 sec\nrounds: 5"
          },
          {
            "name": "tests/bench_instrumentor/bench.py::test_instrumented_with_jit_and_c_tracing_disabled",
            "value": 0.4387275450701057,
            "unit": "iter/sec",
            "range": "stddev: 0.018871889236107584",
            "extra": "mean: 2.279318933212198 sec\nrounds: 5"
          },
          {
            "name": "tests/bench_instrumentor/bench.py::test_instrumented_with_cond_dump_jit_and_c_tracing_disabled",
            "value": 0.4393382355222613,
            "unit": "iter/sec",
            "range": "stddev: 0.0193551603660846",
            "extra": "mean: 2.2761506264330817 sec\nrounds: 5"
          },
          {
            "name": "tests/bench_instrumentor/bench.py::test_sampler_instrumented",
            "value": 0.44371917402146543,
            "unit": "iter/sec",
            "range": "stddev: 0.007041836136689384",
            "extra": "mean: 2.253677682974376 sec\nrounds: 5"
          },
          {
            "name": "tests/bench_instrumentor/bench.py::test_proxy_instrumented",
            "value": 0.3785035549504128,
            "unit": "iter/sec",
            "range": "stddev: 0.015857548022316495",
            "extra": "mean: 2.641983111971058 sec\nrounds: 5"
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
          "id": "92b79350094cd81db390606cb02367f27d004826",
          "message": "[Support LT-725] Trace Context Manager Entry/Exit Events as `meta_vars`",
          "timestamp": "2024-09-28T21:19:09Z",
          "url": "https://github.com/OrderLab/ml-daikon/pull/97/commits/92b79350094cd81db390606cb02367f27d004826"
        },
        "date": 1727674124918,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/bench_instrumentor/bench.py::test_instrumented",
            "value": 0.006346313992783662,
            "unit": "iter/sec",
            "range": "stddev: 1.0297217519909323",
            "extra": "mean: 157.571781216166 sec\nrounds: 5"
          },
          {
            "name": "tests/bench_instrumentor/bench.py::test_instrumented_with_jit_and_c_tracing_disabled",
            "value": 0.00757620505775561,
            "unit": "iter/sec",
            "range": "stddev: 0.3587578190953122",
            "extra": "mean: 131.99220353418497 sec\nrounds: 5"
          },
          {
            "name": "tests/bench_instrumentor/bench.py::test_instrumented_with_cond_dump_jit_and_c_tracing_disabled",
            "value": 0.0075381004935039225,
            "unit": "iter/sec",
            "range": "stddev: 1.1032843971688855",
            "extra": "mean: 132.65941477720625 sec\nrounds: 5"
          },
          {
            "name": "tests/bench_instrumentor/bench.py::test_sampler_instrumented",
            "value": 0.00166210182477658,
            "unit": "iter/sec",
            "range": "stddev: 1.1013870944510473",
            "extra": "mean: 601.647856402793 sec\nrounds: 5"
          },
          {
            "name": "tests/bench_instrumentor/bench.py::test_proxy_instrumented",
            "value": 0.006891235704751752,
            "unit": "iter/sec",
            "range": "stddev: 0.5195057411724248",
            "extra": "mean: 145.11185552838725 sec\nrounds: 5"
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
          "id": "1e172b0ae1c37a41757abea9182f345d85506ffb",
          "message": "[Support LT-725] Trace Context Manager Entry/Exit Events as `meta_vars`",
          "timestamp": "2024-09-28T21:19:09Z",
          "url": "https://github.com/OrderLab/ml-daikon/pull/97/commits/1e172b0ae1c37a41757abea9182f345d85506ffb"
        },
        "date": 1727681364154,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/bench_instrumentor/bench.py::test_instrumented",
            "value": 0.006374093854408461,
            "unit": "iter/sec",
            "range": "stddev: 0.6979915853183484",
            "extra": "mean: 156.88504481439014 sec\nrounds: 5"
          },
          {
            "name": "tests/bench_instrumentor/bench.py::test_instrumented_with_jit_and_c_tracing_disabled",
            "value": 0.0075224824332279635,
            "unit": "iter/sec",
            "range": "stddev: 0.6573986294985962",
            "extra": "mean: 132.93484017760494 sec\nrounds: 5"
          },
          {
            "name": "tests/bench_instrumentor/bench.py::test_instrumented_with_cond_dump_jit_and_c_tracing_disabled",
            "value": 0.007548443935084906,
            "unit": "iter/sec",
            "range": "stddev: 0.794328628615802",
            "extra": "mean: 132.47763494036627 sec\nrounds: 5"
          },
          {
            "name": "tests/bench_instrumentor/bench.py::test_sampler_instrumented",
            "value": 0.0016691474844256848,
            "unit": "iter/sec",
            "range": "stddev: 0.9377786463007669",
            "extra": "mean: 599.1082329936092 sec\nrounds: 5"
          },
          {
            "name": "tests/bench_instrumentor/bench.py::test_proxy_instrumented",
            "value": 0.40879546693001034,
            "unit": "iter/sec",
            "range": "stddev: 0.015942403775688136",
            "extra": "mean: 2.446211078390479 sec\nrounds: 5"
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
          "id": "8ba9d2c8d1730f04038e78703c7651be9bafa09a",
          "message": "[Support LT-725] Trace Context Manager Entry/Exit Events as `meta_vars`",
          "timestamp": "2024-09-28T21:19:09Z",
          "url": "https://github.com/OrderLab/ml-daikon/pull/97/commits/8ba9d2c8d1730f04038e78703c7651be9bafa09a"
        },
        "date": 1727726292003,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/bench_instrumentor/bench.py::test_instrumented",
            "value": 0.006372687195205627,
            "unit": "iter/sec",
            "range": "stddev: 0.30875655260648843",
            "extra": "mean: 156.91967444319744 sec\nrounds: 5"
          },
          {
            "name": "tests/bench_instrumentor/bench.py::test_instrumented_with_jit_and_c_tracing_disabled",
            "value": 0.007547136946460836,
            "unit": "iter/sec",
            "range": "stddev: 0.45031807125478884",
            "extra": "mean: 132.5005769862095 sec\nrounds: 5"
          },
          {
            "name": "tests/bench_instrumentor/bench.py::test_instrumented_with_cond_dump_jit_and_c_tracing_disabled",
            "value": 0.007555868303382027,
            "unit": "iter/sec",
            "range": "stddev: 0.7607283826540681",
            "extra": "mean: 132.3474629054079 sec\nrounds: 5"
          },
          {
            "name": "tests/bench_instrumentor/bench.py::test_sampler_instrumented",
            "value": 0.0015396376868706415,
            "unit": "iter/sec",
            "range": "stddev: 39.62052471432086",
            "extra": "mean: 649.5034569025971 sec\nrounds: 5"
          },
          {
            "name": "tests/bench_instrumentor/bench.py::test_proxy_instrumented",
            "value": 0.006687564108126825,
            "unit": "iter/sec",
            "range": "stddev: 1.6745347924245895",
            "extra": "mean: 149.53127683438362 sec\nrounds: 5"
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
          "id": "0d18fa73086ff592153b9216b05b2b2f5f1a1bcf",
          "message": "[Support LT-725] Trace Context Manager Entry/Exit Events as `meta_vars`",
          "timestamp": "2024-09-28T21:19:09Z",
          "url": "https://github.com/OrderLab/ml-daikon/pull/97/commits/0d18fa73086ff592153b9216b05b2b2f5f1a1bcf"
        },
        "date": 1727728883835,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/bench_instrumentor/bench.py::test_instrumented",
            "value": 0.03931023238674777,
            "unit": "iter/sec",
            "range": "stddev: 0.32874342557625413",
            "extra": "mean: 25.438669254397972 sec\nrounds: 5"
          },
          {
            "name": "tests/bench_instrumentor/bench.py::test_instrumented_with_jit_and_c_tracing_disabled",
            "value": 0.11254877112685213,
            "unit": "iter/sec",
            "range": "stddev: 0.1766927084003933",
            "extra": "mean: 8.885037037613802 sec\nrounds: 5"
          },
          {
            "name": "tests/bench_instrumentor/bench.py::test_instrumented_with_cond_dump_jit_and_c_tracing_disabled",
            "value": 0.11377101831006804,
            "unit": "iter/sec",
            "range": "stddev: 0.15343142829630566",
            "extra": "mean: 8.789584683813155 sec\nrounds: 5"
          },
          {
            "name": "tests/bench_instrumentor/bench.py::test_sampler_instrumented",
            "value": 0.11230351059516043,
            "unit": "iter/sec",
            "range": "stddev: 0.13287914440524048",
            "extra": "mean: 8.904441140801646 sec\nrounds: 5"
          },
          {
            "name": "tests/bench_instrumentor/bench.py::test_proxy_instrumented",
            "value": 0.11195173105741947,
            "unit": "iter/sec",
            "range": "stddev: 0.22506888842708458",
            "extra": "mean: 8.932421058206819 sec\nrounds: 5"
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
          "id": "a9fa07207eeef13749fafe0700c338401d93dae5",
          "message": "[Support LT-725] Trace Context Manager Entry/Exit Events as `meta_vars`",
          "timestamp": "2024-09-28T21:19:09Z",
          "url": "https://github.com/OrderLab/ml-daikon/pull/97/commits/a9fa07207eeef13749fafe0700c338401d93dae5"
        },
        "date": 1727729824979,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/bench_instrumentor/bench.py::test_instrumented",
            "value": 0.14920908285105783,
            "unit": "iter/sec",
            "range": "stddev: 0.36385968997795853",
            "extra": "mean: 6.702004870562814 sec\nrounds: 5"
          },
          {
            "name": "tests/bench_instrumentor/bench.py::test_instrumented_with_jit_and_c_tracing_disabled",
            "value": 0.15177912529225046,
            "unit": "iter/sec",
            "range": "stddev: 0.11425605489747392",
            "extra": "mean: 6.5885213007684795 sec\nrounds: 5"
          },
          {
            "name": "tests/bench_instrumentor/bench.py::test_instrumented_with_cond_dump_jit_and_c_tracing_disabled",
            "value": 0.15031301054985852,
            "unit": "iter/sec",
            "range": "stddev: 0.10472775354987272",
            "extra": "mean: 6.652784056030214 sec\nrounds: 5"
          },
          {
            "name": "tests/bench_instrumentor/bench.py::test_sampler_instrumented",
            "value": 0.14911011696235021,
            "unit": "iter/sec",
            "range": "stddev: 0.07120599871053389",
            "extra": "mean: 6.706453058798798 sec\nrounds: 5"
          },
          {
            "name": "tests/bench_instrumentor/bench.py::test_proxy_instrumented",
            "value": 0.14973240286054354,
            "unit": "iter/sec",
            "range": "stddev: 0.043439118473654025",
            "extra": "mean: 6.678581128036603 sec\nrounds: 5"
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
          "id": "4045d252dd7e03792ff43c30424af9bb7d9423a5",
          "message": "Functionality: Var preserve invariant",
          "timestamp": "2024-09-28T21:19:09Z",
          "url": "https://github.com/OrderLab/ml-daikon/pull/96/commits/4045d252dd7e03792ff43c30424af9bb7d9423a5"
        },
        "date": 1727774196848,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/bench_instrumentor/bench.py::test_instrumented",
            "value": 0.30284728910652003,
            "unit": "iter/sec",
            "range": "stddev: 0.0641054084712397",
            "extra": "mean: 3.3019942260347306 sec\nrounds: 5"
          },
          {
            "name": "tests/bench_instrumentor/bench.py::test_instrumented_with_jit_and_c_tracing_disabled",
            "value": 0.3002353629794618,
            "unit": "iter/sec",
            "range": "stddev: 0.06321821923303012",
            "extra": "mean: 3.330720239202492 sec\nrounds: 5"
          },
          {
            "name": "tests/bench_instrumentor/bench.py::test_instrumented_with_cond_dump_jit_and_c_tracing_disabled",
            "value": 0.28830856392958726,
            "unit": "iter/sec",
            "range": "stddev: 0.04773272875467745",
            "extra": "mean: 3.468506056047045 sec\nrounds: 5"
          },
          {
            "name": "tests/bench_instrumentor/bench.py::test_sampler_instrumented",
            "value": 0.29875232594610174,
            "unit": "iter/sec",
            "range": "stddev: 0.07001142276041085",
            "extra": "mean: 3.3472542743664233 sec\nrounds: 5"
          },
          {
            "name": "tests/bench_instrumentor/bench.py::test_proxy_instrumented",
            "value": 0.24290158838601533,
            "unit": "iter/sec",
            "range": "stddev: 0.03156089655749349",
            "extra": "mean: 4.11689362199977 sec\nrounds: 5"
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
          "id": "57fb6a759c79b54a51bd6c8f468c651dc9b1aca3",
          "message": "[Support LT-725] Trace Context Manager Entry/Exit Events as `meta_vars`",
          "timestamp": "2024-09-28T21:19:09Z",
          "url": "https://github.com/OrderLab/ml-daikon/pull/97/commits/57fb6a759c79b54a51bd6c8f468c651dc9b1aca3"
        },
        "date": 1727790166683,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/bench_instrumentor/bench.py::test_instrumented",
            "value": 0.15240875066343543,
            "unit": "iter/sec",
            "range": "stddev: 0.1203804995461531",
            "extra": "mean: 6.561303046229296 sec\nrounds: 5"
          },
          {
            "name": "tests/bench_instrumentor/bench.py::test_instrumented_with_jit_and_c_tracing_disabled",
            "value": 0.002619262778335427,
            "unit": "iter/sec",
            "range": "stddev: 1.6338269284844127",
            "extra": "mean: 381.7868173713796 sec\nrounds: 5"
          },
          {
            "name": "tests/bench_instrumentor/bench.py::test_instrumented_with_cond_dump_jit_and_c_tracing_disabled",
            "value": 0.0026213696387223564,
            "unit": "iter/sec",
            "range": "stddev: 2.198895582304303",
            "extra": "mean: 381.4799657508032 sec\nrounds: 5"
          },
          {
            "name": "tests/bench_instrumentor/bench.py::test_sampler_instrumented",
            "value": 0.00102843380671047,
            "unit": "iter/sec",
            "range": "stddev: 31.605599671358135",
            "extra": "mean: 972.3523220211733 sec\nrounds: 5"
          },
          {
            "name": "tests/bench_instrumentor/bench.py::test_proxy_instrumented",
            "value": 0.0018894297514495315,
            "unit": "iter/sec",
            "range": "stddev: 3.0789547134118056",
            "extra": "mean: 529.2602168632205 sec\nrounds: 5"
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
          "id": "e5d26f5deed53586d85ef0f17c98d19d88bfbc11",
          "message": "Functionality: Var preserve invariant",
          "timestamp": "2024-09-28T21:19:09Z",
          "url": "https://github.com/OrderLab/ml-daikon/pull/96/commits/e5d26f5deed53586d85ef0f17c98d19d88bfbc11"
        },
        "date": 1727796548687,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/bench_instrumentor/bench.py::test_instrumented",
            "value": 0.28424250678381185,
            "unit": "iter/sec",
            "range": "stddev: 0.03723037918530445",
            "extra": "mean: 3.5181226457469164 sec\nrounds: 5"
          },
          {
            "name": "tests/bench_instrumentor/bench.py::test_instrumented_with_jit_and_c_tracing_disabled",
            "value": 0.28305877443350047,
            "unit": "iter/sec",
            "range": "stddev: 0.06395363938297138",
            "extra": "mean: 3.532835192978382 sec\nrounds: 5"
          },
          {
            "name": "tests/bench_instrumentor/bench.py::test_instrumented_with_cond_dump_jit_and_c_tracing_disabled",
            "value": 0.2908019552341662,
            "unit": "iter/sec",
            "range": "stddev: 0.12942196065947958",
            "extra": "mean: 3.438766425056383 sec\nrounds: 5"
          },
          {
            "name": "tests/bench_instrumentor/bench.py::test_sampler_instrumented",
            "value": 0.2792985406429665,
            "unit": "iter/sec",
            "range": "stddev: 0.06462641681959375",
            "extra": "mean: 3.580398227996193 sec\nrounds: 5"
          },
          {
            "name": "tests/bench_instrumentor/bench.py::test_proxy_instrumented",
            "value": 0.23297529523760924,
            "unit": "iter/sec",
            "range": "stddev: 0.049943237691338196",
            "extra": "mean: 4.2923006020020695 sec\nrounds: 5"
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
          "id": "aa2b7e37e5267060a893632e0cac2fec6f37a253",
          "message": "Functionality: Var preserve invariant",
          "timestamp": "2024-10-01T15:31:11Z",
          "url": "https://github.com/OrderLab/ml-daikon/pull/96/commits/aa2b7e37e5267060a893632e0cac2fec6f37a253"
        },
        "date": 1727797302364,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/bench_instrumentor/bench.py::test_instrumented",
            "value": 0.2903718110191053,
            "unit": "iter/sec",
            "range": "stddev: 0.06869062732490644",
            "extra": "mean: 3.44386046458967 sec\nrounds: 5"
          },
          {
            "name": "tests/bench_instrumentor/bench.py::test_instrumented_with_jit_and_c_tracing_disabled",
            "value": 0.28823691033204407,
            "unit": "iter/sec",
            "range": "stddev: 0.077794961660875",
            "extra": "mean: 3.469368301401846 sec\nrounds: 5"
          },
          {
            "name": "tests/bench_instrumentor/bench.py::test_instrumented_with_cond_dump_jit_and_c_tracing_disabled",
            "value": 0.29056199401298255,
            "unit": "iter/sec",
            "range": "stddev: 0.11818801115657933",
            "extra": "mean: 3.44160633739084 sec\nrounds: 5"
          },
          {
            "name": "tests/bench_instrumentor/bench.py::test_sampler_instrumented",
            "value": 0.29173029098047143,
            "unit": "iter/sec",
            "range": "stddev: 0.0960630248270102",
            "extra": "mean: 3.427823681384325 sec\nrounds: 5"
          },
          {
            "name": "tests/bench_instrumentor/bench.py::test_proxy_instrumented",
            "value": 0.2447386637242881,
            "unit": "iter/sec",
            "range": "stddev: 0.08092674318117848",
            "extra": "mean: 4.085991092631593 sec\nrounds: 5"
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
          "id": "4fcf557e9c2c366ebd7bb5615f88a6f049261d97",
          "message": "Functionality: Var preserve invariant",
          "timestamp": "2024-10-01T15:31:11Z",
          "url": "https://github.com/OrderLab/ml-daikon/pull/96/commits/4fcf557e9c2c366ebd7bb5615f88a6f049261d97"
        },
        "date": 1727801794932,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/bench_instrumentor/bench.py::test_instrumented",
            "value": 0.3011108983600241,
            "unit": "iter/sec",
            "range": "stddev: 0.045878863551865406",
            "extra": "mean: 3.3210355568211525 sec\nrounds: 5"
          },
          {
            "name": "tests/bench_instrumentor/bench.py::test_instrumented_with_jit_and_c_tracing_disabled",
            "value": 0.29767254123664844,
            "unit": "iter/sec",
            "range": "stddev: 0.026726109137195433",
            "extra": "mean: 3.3593961869832127 sec\nrounds: 5"
          },
          {
            "name": "tests/bench_instrumentor/bench.py::test_instrumented_with_cond_dump_jit_and_c_tracing_disabled",
            "value": 0.30132535555498535,
            "unit": "iter/sec",
            "range": "stddev: 0.07961305957265036",
            "extra": "mean: 3.31867193239741 sec\nrounds: 5"
          },
          {
            "name": "tests/bench_instrumentor/bench.py::test_sampler_instrumented",
            "value": 0.2974102536166904,
            "unit": "iter/sec",
            "range": "stddev: 0.05097471393873591",
            "extra": "mean: 3.362358855619095 sec\nrounds: 5"
          },
          {
            "name": "tests/bench_instrumentor/bench.py::test_proxy_instrumented",
            "value": 0.24886883802005902,
            "unit": "iter/sec",
            "range": "stddev: 0.1231364342186102",
            "extra": "mean: 4.0181808536406605 sec\nrounds: 5"
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
          "id": "c7f58e424e7061c07c889b3b0fa9fed48ad60c81",
          "message": "Functionality: Var preserve invariant",
          "timestamp": "2024-10-01T15:31:11Z",
          "url": "https://github.com/OrderLab/ml-daikon/pull/96/commits/c7f58e424e7061c07c889b3b0fa9fed48ad60c81"
        },
        "date": 1727802088839,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/bench_instrumentor/bench.py::test_instrumented",
            "value": 0.2977023867654436,
            "unit": "iter/sec",
            "range": "stddev: 0.055651883080203045",
            "extra": "mean: 3.3590593977598475 sec\nrounds: 5"
          },
          {
            "name": "tests/bench_instrumentor/bench.py::test_instrumented_with_jit_and_c_tracing_disabled",
            "value": 0.29724498066417465,
            "unit": "iter/sec",
            "range": "stddev: 0.08677488528830794",
            "extra": "mean: 3.3642283807974307 sec\nrounds: 5"
          },
          {
            "name": "tests/bench_instrumentor/bench.py::test_instrumented_with_cond_dump_jit_and_c_tracing_disabled",
            "value": 0.2998424068340711,
            "unit": "iter/sec",
            "range": "stddev: 0.04615724615798311",
            "extra": "mean: 3.3350852888310327 sec\nrounds: 5"
          },
          {
            "name": "tests/bench_instrumentor/bench.py::test_sampler_instrumented",
            "value": 0.3004138155312808,
            "unit": "iter/sec",
            "range": "stddev: 0.09692527122873859",
            "extra": "mean: 3.328741716593504 sec\nrounds: 5"
          },
          {
            "name": "tests/bench_instrumentor/bench.py::test_proxy_instrumented",
            "value": 0.24989874896022488,
            "unit": "iter/sec",
            "range": "stddev: 0.12811398439874788",
            "extra": "mean: 4.001620673015713 sec\nrounds: 5"
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
          "id": "92a34889e2e81e8f8a06853d900b46b42a3e5c3b",
          "message": "Functionality: Var preserve invariant",
          "timestamp": "2024-10-01T15:31:11Z",
          "url": "https://github.com/OrderLab/ml-daikon/pull/96/commits/92a34889e2e81e8f8a06853d900b46b42a3e5c3b"
        },
        "date": 1727802637793,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/bench_instrumentor/bench.py::test_instrumented",
            "value": 0.29974087126622934,
            "unit": "iter/sec",
            "range": "stddev: 0.057593947075761764",
            "extra": "mean: 3.3362150305882095 sec\nrounds: 5"
          },
          {
            "name": "tests/bench_instrumentor/bench.py::test_instrumented_with_jit_and_c_tracing_disabled",
            "value": 0.2994925250697353,
            "unit": "iter/sec",
            "range": "stddev: 0.024521560941346738",
            "extra": "mean: 3.338981498009525 sec\nrounds: 5"
          },
          {
            "name": "tests/bench_instrumentor/bench.py::test_instrumented_with_cond_dump_jit_and_c_tracing_disabled",
            "value": 0.29494626383469624,
            "unit": "iter/sec",
            "range": "stddev: 0.04951460749779237",
            "extra": "mean: 3.3904481006087734 sec\nrounds: 5"
          },
          {
            "name": "tests/bench_instrumentor/bench.py::test_sampler_instrumented",
            "value": 0.3019888374261329,
            "unit": "iter/sec",
            "range": "stddev: 0.08592986692843016",
            "extra": "mean: 3.3113806739449503 sec\nrounds: 5"
          },
          {
            "name": "tests/bench_instrumentor/bench.py::test_proxy_instrumented",
            "value": 0.24699113895174235,
            "unit": "iter/sec",
            "range": "stddev: 0.04031562584291898",
            "extra": "mean: 4.048728242819197 sec\nrounds: 5"
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
          "id": "2227d13ccbded69605f3125dc584f0a817b37ffd",
          "message": "Functionality: Var preserve invariant",
          "timestamp": "2024-10-01T15:31:11Z",
          "url": "https://github.com/OrderLab/ml-daikon/pull/96/commits/2227d13ccbded69605f3125dc584f0a817b37ffd"
        },
        "date": 1727803524911,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/bench_instrumentor/bench.py::test_instrumented",
            "value": 0.29459375380093505,
            "unit": "iter/sec",
            "range": "stddev: 0.055591953815039834",
            "extra": "mean: 3.3945051009999587 sec\nrounds: 5"
          },
          {
            "name": "tests/bench_instrumentor/bench.py::test_instrumented_with_jit_and_c_tracing_disabled",
            "value": 0.296145529073152,
            "unit": "iter/sec",
            "range": "stddev: 0.056591020106631994",
            "extra": "mean: 3.3767182071926074 sec\nrounds: 5"
          },
          {
            "name": "tests/bench_instrumentor/bench.py::test_instrumented_with_cond_dump_jit_and_c_tracing_disabled",
            "value": 0.2998792630630376,
            "unit": "iter/sec",
            "range": "stddev: 0.07402237600419755",
            "extra": "mean: 3.3346753949765118 sec\nrounds: 5"
          },
          {
            "name": "tests/bench_instrumentor/bench.py::test_sampler_instrumented",
            "value": 0.2996195045247879,
            "unit": "iter/sec",
            "range": "stddev: 0.08624594739810218",
            "extra": "mean: 3.337566429749131 sec\nrounds: 5"
          },
          {
            "name": "tests/bench_instrumentor/bench.py::test_proxy_instrumented",
            "value": 0.2496724528747735,
            "unit": "iter/sec",
            "range": "stddev: 0.04526364721546443",
            "extra": "mean: 4.005247629387304 sec\nrounds: 5"
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
          "id": "6da821eb4ccce6b51b6562a5719f6a6925ffdcec",
          "message": "Functionality: Var preserve invariant",
          "timestamp": "2024-10-01T15:31:11Z",
          "url": "https://github.com/OrderLab/ml-daikon/pull/96/commits/6da821eb4ccce6b51b6562a5719f6a6925ffdcec"
        },
        "date": 1727803662230,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/bench_instrumentor/bench.py::test_instrumented",
            "value": 0.3035031561132893,
            "unit": "iter/sec",
            "range": "stddev: 0.09584727215276234",
            "extra": "mean: 3.2948586525628345 sec\nrounds: 5"
          },
          {
            "name": "tests/bench_instrumentor/bench.py::test_instrumented_with_jit_and_c_tracing_disabled",
            "value": 0.29981671998941456,
            "unit": "iter/sec",
            "range": "stddev: 0.07763670971893472",
            "extra": "mean: 3.3353710227878763 sec\nrounds: 5"
          },
          {
            "name": "tests/bench_instrumentor/bench.py::test_instrumented_with_cond_dump_jit_and_c_tracing_disabled",
            "value": 0.29859506064038804,
            "unit": "iter/sec",
            "range": "stddev: 0.0442850819835815",
            "extra": "mean: 3.349017220363021 sec\nrounds: 5"
          },
          {
            "name": "tests/bench_instrumentor/bench.py::test_sampler_instrumented",
            "value": 0.296085220714453,
            "unit": "iter/sec",
            "range": "stddev: 0.08166100615500398",
            "extra": "mean: 3.3774059967836365 sec\nrounds: 5"
          },
          {
            "name": "tests/bench_instrumentor/bench.py::test_proxy_instrumented",
            "value": 0.24625680287179802,
            "unit": "iter/sec",
            "range": "stddev: 0.08466747350264951",
            "extra": "mean: 4.060801522387192 sec\nrounds: 5"
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
          "id": "d1bb26fc2f96d01443d5c3ec7cd1b8736393c9f0",
          "message": "Functionality: Var preserve invariant",
          "timestamp": "2024-10-01T15:31:11Z",
          "url": "https://github.com/OrderLab/ml-daikon/pull/96/commits/d1bb26fc2f96d01443d5c3ec7cd1b8736393c9f0"
        },
        "date": 1727804552439,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/bench_instrumentor/bench.py::test_instrumented",
            "value": 0.29728566496288217,
            "unit": "iter/sec",
            "range": "stddev: 0.0705565148279908",
            "extra": "mean: 3.363767977594398 sec\nrounds: 5"
          },
          {
            "name": "tests/bench_instrumentor/bench.py::test_instrumented_with_jit_and_c_tracing_disabled",
            "value": 0.2939269918876594,
            "unit": "iter/sec",
            "range": "stddev: 0.04031717922927751",
            "extra": "mean: 3.4022054033819584 sec\nrounds: 5"
          },
          {
            "name": "tests/bench_instrumentor/bench.py::test_instrumented_with_cond_dump_jit_and_c_tracing_disabled",
            "value": 0.2935726356440383,
            "unit": "iter/sec",
            "range": "stddev: 0.041973217104068684",
            "extra": "mean: 3.4063120283884927 sec\nrounds: 5"
          },
          {
            "name": "tests/bench_instrumentor/bench.py::test_sampler_instrumented",
            "value": 0.2955454519474892,
            "unit": "iter/sec",
            "range": "stddev: 0.067305477086578",
            "extra": "mean: 3.3835743145784365 sec\nrounds: 5"
          },
          {
            "name": "tests/bench_instrumentor/bench.py::test_proxy_instrumented",
            "value": 0.2475332198988376,
            "unit": "iter/sec",
            "range": "stddev: 0.08466782226872974",
            "extra": "mean: 4.0398618028266355 sec\nrounds: 5"
          }
        ]
      }
    ]
  }
}
window.BENCHMARK_DATA = {
  "lastUpdate": 1720327816968,
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
      }
    ]
  }
}
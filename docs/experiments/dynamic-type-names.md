# Dynamic type-name metadata

## Problem and change

PyTorch `torch._inductor.virtualized.ops` returns callable values for dynamic
`__qualname__` and `__name__` attributes. TrainCheck rejected the first value but
accepted the second, then asserted that its type name was a string. A fresh
MNIST trace-collection attempt failed before training reached its first step.

Validate `__name__` as a nonempty string before using it. Otherwise, use the
existing reported-class fallback. This does not call dynamic metadata, truth-test
non-string values, change valid name precedence, or replace the reported
`__class__` with the object's actual type.

## Validation

An identical seven-test suite was run against the exact parent `typename`
function extracted from commit `9ac61d1140cf8679c24442a174ec12f4de069980` and against
the changed function, with identical module globals. Python 3.10.16 / PyTorch
2.2.2 on macOS arm64:

| Result | Parent | Changed |
| --- | ---: | ---: |
| Tests passed | 4 | 7 |
| Tests failed | 3 | 0 |
| Test errors | 0 | 0 |

The three previously failing tests cover callable dynamic metadata, non-string
names (including an unsafe truth-value implementation), and the real PyTorch
`OpsWrapper`. The remaining tests preserve qualified/simple names, reported-class
proxies, functions, descriptors, tensors, and parameters. Both instrumentation
and runtime modes are exercised for dynamic metadata.

Reproduce the candidate suite with:

```sh
python -m unittest tests.test_dynamic_typename -v
```

The motivating Linux training run used an existing broader temporary workaround.
Its successful collection is not evidence of an end-to-end run of this narrower
production change; the direct real-object regression test is the evidence here.
This is a compatibility fix, with no performance improvement claim.

## Matching Linux runtime follow-up

The exact signed utility change was also tested on Python 3.11.4 and PyTorch
2.16.0.dev20261012+cpu. All seven type-name regressions passed within a **29-test
integration suite** (also covering trace integrity, TorchScript, and precondition
state isolation). A separate 11-test checker-observer suite passed. The real
`OpsWrapper` returned `torch._inductor.virtualized.OpsWrapper` in both modes.

The disposable integration source combined the signed tracing, TorchScript,
cache, sampling, state-isolation, and type-name changes. Its `utils.py` matched
commit `2f034ae5d0fab7e792b7c0a5bdc6a1acfeb8962c` exactly, and all source hashes
were unchanged after testing. Compact evidence includes runtime versions and
log/source hashes. These preflight results do not establish training overhead
or cross-workload invariant generalization.

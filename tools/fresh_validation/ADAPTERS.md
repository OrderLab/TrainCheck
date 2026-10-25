# Bounded trace-impact workloads

`benchmark_workload.py` reads TrainCheck-Benchmarks without modifying it. Run each invocation in a fresh process with the chosen TrainCheck source tree in `PYTHONPATH`:

```sh
PYTHONPATH=/path/to/traincheck-variant python benchmark_workload.py \
  --benchmark-root /path/to/clean/TrainCheck-Benchmarks \
  --workload mnist --steps 5 --seed 42 --batch-size 2 \
  --device cuda --data-root /path/to/MNIST-cache \
  --output /path/to/new/result --mode instrumented --save-checkpoint
```

Use `--mode baseline` for uninstrumented execution. That mode does not import TrainCheck. Modes share source, batch preparation, model initialization, explicit training RNG reset, CUDA synchronization, and training math. The output directory must not already contain a run. No datasets or packages are downloaded.

These are **model micro-workloads adapted from the benchmark**, not full end-to-end benchmark reproductions:

| Name | Exact benchmark definitions | Training/data adaptation |
| --- | --- | --- |
| `mnist` | `instr-overhead/overhead-e2e/mnist/main.py:Net` | Original forward, NLL, Adadelta(lr=1.0). First cached MNIST training samples if `--data-root` is supplied; otherwise explicitly synthetic uniform images and random labels. Original normalization preserved. No epoch scheduler or evaluation. |
| `dcgan` | `instr-overhead/overhead-e2e/dcgan/main.py:Generator,Discriminator,weights_init` | Original discriminator real/fake BCE then generator BCE updates and Adam hyperparameters. ngf/ndf reduced from 64 to 8. Synthetic uniform RGB images and precomputed Gaussian noise; not torchvision FakeData's exact generation. No image visualization. |
| `transformer` | `false-positive/RNN/trainset/word_language_model-RNN_TANH-3epo/model.py:PositionalEncoding,TransformerModel` | Two 32-dimensional encoder layers, 64-dimensional feedforward, 16-token sequences, vocabulary from checked-in WikiText training prefix. NLL/backward/clipped manual SGD. This repo's main loop is RNN-specific; this is an explicit Transformer training adaptation. |

`adaptation.json` identifies exact source paths/SHA256, extracted class source SHA256, dataset hashes and adaptation details, version, model configuration, parameter count, and initial state shapes/hash. Full class/function source spans are saved in `benchmark_definitions.py`; top-level source CLIs are never executed. Original benchmark files retain their original license/copyright.

Instrumented mode calls the public `Instrumentor` programmatically over `torch`, with API-only tracing, normal API coverage (`use_full_instr=False`), no selected API list, no proxy/model tracking, interval 1, and no warmup. It enables tracing only during training bodies via explicit `start_step`; initialization, dataset preparation, metric extraction, timing synchronization, and checkpoint writes are excluded. In both modes the training RNG is reset to `seed + 1` after setup. `TC_ORIGIN_AUDIT=1` enables a separately supplied `origin_audit.install()` before instrumentation, for a distinct diagnostic run.

`steps.jsonl` preserves completed step timing/losses even if a later step fails. `metrics.json` includes completion, error, canonical initial/final model-state hashes and state shapes. `--save-checkpoint` writes the final model state dictionaries after successful training; it does not save optimizer/RNG state and therefore is not by itself a resume-equivalence evaluation. Canonical state hashes cover sorted names, dtypes, shapes, and tensor bytes, avoiding `torch.save` archive metadata differences.

For fresh validation, `--data-steps 20` fixes corpus preparation and model vocabulary across runs that execute different numbers of training steps. `adaptation.json` records both the preparation horizon and actual steps, together with the producer PID, runtime and imported tracing-module path.

# Released CPU/CUDA tensor-hash audit

This is a bounded audit of unchanged TrainCheck v0.2.0 source, not a production
change or an end-to-end training benchmark. The source commit is
`9ba90ac84a1b4efd7e9f65b48b36bbbebf958253`; the retained hash source has SHA-256
`e917bec817df1981c682d8b8645307432c6badb220b9f7c424b4089d9bf97bd7`.

The CUDA dumper flag was changed from true to false by Daniel in commit
`fec69179f6fbfd906789ffa6278c61a627229179` (2026-06-08), whose message is only
`temp fixes`. The same commit adds `to_local()` handling. The commit does not
document why the CUDA backend was disabled; the failures below do not establish
the original motivation.

## Correctness findings

- Of 17 shape/dtype cases, 15 succeed with identical CPU/CUDA hash values and
  unchanged inputs. Tested cases include floating/integer dtypes, rank-three
  tensors, noncontiguous transpose/strided inputs, and empty vectors/columns.
- Scalar float32 is accepted by CPU hashing but CUDA hashing raises `ValueError:
  not enough values to unpack (expected 2, got 0)`. A blanket flag flip changes
  an accepted input into a failure.
- Empty shape `(0, 3)` already fails with CPU `IndexError`; CUDA produces a
  different launch error. Empty-input semantics are an existing limitation,
  not evidence of a successful CUDA fallback.
- Six deliberately delayed nondefault-stream cases return incorrect CUDA hash
  values. The producer write and hash call both execute inside the same
  `with torch.cuda.stream(producer)` context. CUDA returns the old zero-tensor
  hash while CPU hashing on that same current stream returns the expected hash.
  Adding `producer.synchronize()` before the CUDA hash also produces the correct
  result. The explicit control matrix has CUDA-current-stream 0/6 correct,
  CUDA-after-sync 6/6 correct, and CPU-current-stream 6/6 correct. Both int64 and
  float32 are covered; no consumer-outside-producer-stream misuse is involved.
- Both implementations return only the first row's accumulator. Changing a
  nonfirst row does not change either result; changing the first row does. This
  is a separate, pre-existing hash semantics limitation shared by both paths.
  Backend parity is not proof that the hash detects arbitrary tensor mutation.

## Timing results

All inputs below already reside on CUDA. Timings include the entire `tensor_hash`
call (float-to-int conversion, CPU transfer/hash or CUDA launch/result transfer)
and CUDA synchronization. They exclude tracer metadata, training computation,
imports and CUDA/Numba initialization. Two warmups/backend precede seven timed
calls/backend in alternating CPU/CUDA then CUDA/CPU order. These are medians from
one warmed process on a shared host; no confidence interval or end-to-end speedup
is claimed. The explicit synchronization controls are not included in these
default-stream timing cases.

| Float32 input shape | CPU backend | CUDA backend | CPU / CUDA |
|---|---:|---:|---:|
| 16 | 49.53 us | 125.19 us | 0.40x |
| 64 x 64 | 106.31 us | 119.87 us | 0.89x |
| 1024 x 1024 | 11.969 ms | 0.304 ms | 39.40x |
| 1024 | 1.050 ms | 0.202 ms | 5.20x |
| 4096 | 4.062 ms | 0.411 ms | 9.88x |
| 256 x 256 | 0.469 ms | 0.153 ms | 3.06x |
| 4096 x 16 | 0.212 ms | 0.135 ms | 1.57x |
| 1 x 1024 | 1.038 ms | 0.187 ms | 5.56x |
| 1024 x 1 | 33.99 us | 125.18 us | 0.27x |

The last two shapes contain the same number of elements but have different
preferred backends. Input element count alone is insufficient to choose a
threshold. A guarded CUDA path may be useful after stream/shape contracts are
fixed and workload-specific hash-size distributions are measured. Do not enable
CUDA globally on the basis of the large-matrix result.

The [compact evidence](data/tensor-hash-backends-20261026.json) retains all
shape outcomes, timing samples, stream controls, and artifact identities.

## Follow-up on the measured MNIST weight shape

The [API/model-tracer study](tracer-components-20261026.md) identifies `fc1.weight`,
shape `(128, 9216)`, as 98.31% of logical hashed input bytes. A separate matched
microbenchmark of that exact shape, with input already on CUDA, measures
**14.576 ms CPU hashing versus 1.511 ms CUDA hashing (9.64x)** with identical hash
values. The same two warmups and seven alternating timed calls/backend are used;
all raw samples and three verified follow-up artifact hashes are in the compact
evidence. This confirms a relevant optimization opportunity after the scalar and
stream contracts are fixed. It is not a measured speedup for CPU training, model
tracing, or the complete training run. The future stream repair is not timed.

## Environment and evidence

labmach: RTX A2000 12GB, driver 545.23.08, Python 3.10.16, PyTorch 2.2.2+cu121,
Numba 0.61.2. One native CPU thread; GPU was inspected idle before the audit and the additional
timing batch.
`environment.txt` records exact versions and post-run GPU status. The isolated
remote directory is `/home/yuxuan/experiments/codex-hash-backend-audit-20261026-v020`.
No shared environment or production source changed. Local evidence is retained
at `/Users/matt/gitrepos/traincheck/outputs/tracing-components-20261026/hash/`.

`hash-results.json`, `more-shapes-results.json`, and `stream-controls-results.json`
retain exact outcomes and all timing samples. The initial stream stress is in
`stream-results.json`. `verified-artifacts.json` verifies all 13 downloaded
scripts/result/log files against remote hashes. All four audit harnesses passed
strict mypy (`--check-untyped-defs --disallow-untyped-defs
--disallow-any-generics --follow-imports=skip`), Black and isort checks. The
released legacy hash module was not rewritten or type-checked as new code.

To reproduce in a new directory, retain the same released `hash_source.py` and
four audit scripts. Run with the existing environment and these settings:

```sh
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 \
  /home/yuxuan/miniconda3/envs/traincheck/bin/python audit_hash.py \
  --source hash_source.py --output hash-results.json
```

Use the same environment variables for `audit_hash_streams.py`,
`audit_hash_stream_controls.py`, and `audit_hash_more_shapes.py`. Each script
refuses to overwrite prior results. The controller bounded the original runs at
90, 30, 30 and 60 seconds, respectively; all Python runs completed. The first
launch's shell wrapper attempted to assign zsh's read-only `status` variable
after Python completed; the complete saved JSON/log and subsequent checks
confirm the Python audit itself succeeded. Later wrappers use a dedicated exit
code variable.

# Observe model state at a boundary you choose

TrainCheck can capture a model's named parameters and buffers and compare the
result across ranks or execution phases. Use this when it helps answer your
next question; observation is optional. You choose the observation points and
the predicate. No inferred invariant or predefined diagnosis is required.

The snapshot interface reuses TrainCheck's model-state enumeration and existing
tensor fingerprint implementation. It does not replace parameters, install
tensor subclasses, or enable full API tracing. Each snapshot includes every
registered parameter and buffer name, including nonpersistent buffers and
multiple names for an aliased tensor. It records shape, dtype and a SHA-256
fingerprint of every tensor's complete logical contents. Snapshot comparison
checks names and metadata as well as fingerprints.

Importing `traincheck` or `traincheck.observation` does not activate the legacy
instrumentor. Existing public exports such as `Instrumentor` remain available
and load their implementation when requested; explicit instrumentor imports
retain their existing behavior.

## Add one observation

```python
from pathlib import Path
from traincheck.observation import snapshot_model, write_snapshot

# model is your existing torch.nn.Module; rank is its worker's actual rank.
snapshot = snapshot_model(
    model, run_id="my-investigation", rank=rank,
    phase="chosen-boundary", step=step, model_name="policy",
)
output = Path("observations")
output.mkdir(parents=True, exist_ok=True)
write_snapshot(snapshot, output / f"rank{rank}-chosen-boundary-{step}.json")
```

Call this at the actual execution site where the state is useful. Keep the model
quiescent while collecting it: this is a synchronous observation, not an atomic
snapshot against concurrent training or mutation. Use a unique output path for
every observation. `write_snapshot` refuses to overwrite an existing file;
missing parent directories must be created by the caller. In a managed training
service, place files under its supplied output directory and retain the actual
rank in the snapshot. No shared append-only file or logging configuration is
needed.

`run_id` identifies your logical investigation. `model_name` identifies the model
within it. `phase` and nonnegative `step` identify a boundary in your program;
they do not infer training progress. Choose them consistently so independently
written files can be compared. Values are detached observations, so subsequent
training does not change a saved snapshot.

## Compare ranks offline

```python
from pathlib import Path
from traincheck.observation import compare_ranks, read_snapshot

snapshots = [read_snapshot(Path(f"observations/rank{r}-chosen-boundary-0.json"))
             for r in (0, 1)]
result = compare_ranks(snapshots, expected_ranks=[0, 1])
print(result.equal)
for comparison in result.comparisons:
    for difference in comparison.differences:
        print(difference.name, difference.fields)
```

This requires at least two distinct expected ranks and exactly one observation
for every explicitly expected rank, with
the same run, model, phase and step. Missing or duplicate ranks, malformed
records and incompatible labels raise errors. They never become a successful
empty comparison. Every non-reference rank is compared with `reference_rank`
(zero by default). Physical device placement may differ across ranks and does
not itself make the model values different.

Equality is evidence of matching complete fingerprints and metadata; it is not
a mathematical proof that hash collisions are impossible. A difference is an
observation, not automatically a bug. Whether two states should agree depends
on the training contract. Rank-specific buffers or legitimately different model
components need an appropriate predicate instead of unconditional equality.

## Compare two explicitly selected phases

```python
from traincheck.observation import compare_model_snapshots

result = compare_model_snapshots(before, after)
print(result.equal, result.tensors_compared)
for difference in result.differences:
    print(difference.name, difference.fields)
```

`before` and `after` must be snapshots of the same named model in the same
logical investigation. This explicit pair comparison permits different ranks,
phases and steps. For example, an optimizer update may legitimately change model
parameters. The result identifies changed fields; it does not prescribe which
changes are expected. Independently collected observations can use the same
logical run identifier when you deliberately intend to compare them.

## Runnable examples

The repository includes `tools/model_observation/capture_example.py` and
`tools/model_observation/compare.py`. In an exported agent packet, use the
provided copies of these examples from that packet's example directory.

From a checkout with TrainCheck and its dependencies installed:

```sh
python -m tools.model_observation.capture_example \
  --output /tmp/model-observation-demo --run-id demo --steps 2
python -m tools.model_observation.compare pair \
  /tmp/model-observation-demo/rank0-ready-0.json \
  /tmp/model-observation-demo/rank0-update-2.json
```

The small model trains on fixed illustrative data. The pair command normally
reports changed parameter fingerprints and exits with status 1 because the
observed states differ; that is expected after training. An equal comparison
returns 0, while invalid or incomplete inputs return 2 with an error.

On an allocated two-GPU machine, the same example supports ordinary DDP:

```sh
torchrun --standalone --nproc-per-node=2 -m tools.model_observation.capture_example \
  --device cuda --output /tmp/model-observation-ddp --run-id demo-ddp --steps 2
python -m tools.model_observation.compare ranks --expected-ranks 0 1 \
  --snapshots /tmp/model-observation-ddp/rank0-update-2.json \
              /tmp/model-observation-ddp/rank1-update-2.json
```

For a two-process CPU example, omit `--device cuda`. Use fresh output directories
for each invocation. These examples generate their own tiny workload; insert
the observation calls into your actual training code for your investigation.

## Scope and cost

- The API observes registered model parameters and buffers. It does not
  automatically capture optimizer state, gradients, arbitrary locals, source
  provenance, or application-specific objects. Add separate observations when
  those are needed.
- Materialized dense tensors with the supported ordinary dtypes are accepted.
  Sparse, quantized, meta and unsupported tensor subclasses are explicit errors.
  Custom module state-enumeration overrides are also rejected because they can
  hide registered state. These cases do not produce partial passing snapshots.
- It performs no distributed communication or implicit all-gather. If a model
  is sharded, the caller must select and coordinate the appropriate full-model
  observation context supplied by its framework. Comparing local shards as if
  they were replicas is not a valid check. Read the API's supported-tensor errors
  instead of treating an unsupported or absent observation as success.
- Capturing complete fingerprints reads every selected tensor and copies GPU
  data to the CPU. It can synchronize device work. This makes occasional
  observations convenient but does not promise negligible cost or suitability
  for every training step. Choose useful boundaries and measure overhead for
  your workload.
- Fingerprints cannot reconstruct tensor values or measure numerical distance.
  They do distinguish bit patterns such as signed zero and NaN representations;
  they do not apply floating-point tolerances. Matching tensor contents does not
  establish identical storage aliasing or a complete application checkpoint.
- Completeness is relative to the registered model state at each chosen
  boundary and the expected ranks you supplied. Equality alone does not prove
  the intended program location executed; keep the call in the real execution
  path and retain failures as failures.

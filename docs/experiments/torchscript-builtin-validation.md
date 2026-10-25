# TorchScript builtin identity validation

TorchScript records builtin operators by function ID. Replacing `torch.tanh` with a TrainCheck wrapper loses the `aten::tanh` lookup. TorchScript attempts to inspect Python source, follows the wrapper metadata to the C builtin, and raises `TypeError: ... expected, got builtin_function_or_method`.

The compatibility helper looks up the original mapping before the wrapper is published and registers the wrapper with that same operator. It guesses no operator names. Missing private APIs and unregistered Python functions are unchanged. Registered wrappers are retained because the registry stores only IDs; otherwise a later unrelated callable could reuse a stale ID.

## Controls and results

The 2026-10-25 CPU experiment uses a real Python source file containing a representative tanh-based bias-GELU expression. Each control runs in a fresh interpreter, with a 90-second timeout and one native/inter-op thread.

| Control | Compilation | Eager tanh records | Scripted operator |
|---|---|---:|---|
| Unwrapped baseline | Pass | 0 | `aten::tanh` |
| Wrapper without compatibility, no explicit priming | TypeError | 2, one pair | Unregistered wrapper |
| Wrapper without compatibility, explicitly primed | TypeError | 2, one pair | Unregistered wrapper |
| Wrapper plus helper, no explicit priming | Pass | 2, one pair | `aten::tanh` |
| Wrapper plus helper, explicitly primed | Pass | 2, one pair | `aten::tanh` |
| Helper integrated into the actual wrapper | Pass | 2, one pair | `aten::tanh` |

The original study passed all 13 semantic/trace checks and four helper unit tests on macOS with PyTorch 2.2.2 and Linux with PyTorch 2.2.2+cu121; both used Python 3.10.16 and CPU tensors. Candidate eager outputs and both gradients exactly matched the unwrapped baseline within each platform. Successful scripted outputs and both gradients also exactly matched that platform's scripted baseline. Every eager pair retained arguments, return snapshots, and a shared call ID.

The portable checkout test was additionally run on macOS against the trace-integrity source plus this integration: all five test methods passed, comprising the four helper contracts and six fresh-process controls with the same 13 semantic/trace checks. The helper implementation is unchanged from the Linux-tested version apart from its module docstring and formatting. Linux's earlier study is separate evidence; the converted checkout test was not rerun there.

Run the portable tests from the repository root:

```bash
python -m unittest discover -s tests -p test_torchscript_compat.py -v
```

Compact source hashes, control outcomes, and validation details are recorded in [torchscript-builtin-validation.json](torchscript-builtin-validation.json).

## Limits

TrainCheck imports had already initialized the registry in both tested environments. “No explicit priming” therefore does not prove a genuinely uninitialized registry path.

Scripted graphs call `aten::tanh` directly and produce no inner Python tanh records. Eager tracing remains intact; complete compiled-graph tracing is outside this change. Full instrument-before-import Megatron startup and checkpoint-shape compatibility were not validated by this focused fixture.

Registered wrappers remain alive for the process lifetime; repeatedly creating new wrappers retains additional objects. The helper depends on feature-detected private APIs and does not establish compatibility with every future PyTorch version or wrapped operator.

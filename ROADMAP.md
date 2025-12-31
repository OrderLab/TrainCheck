# TrainCheck Roadmap

This document outlines planned directions for the TrainCheck project. The roadmap is aspirational and subject to change as we gather feedback from the community.

## North Star

TrainCheck should be a holistic, production-ready monitoring tool for ML training: low overhead, actionable diagnostics, and flexible enough to integrate with real-world training stacks.

## Near Term (Top Priorities)

- **Overhead & selective tracking** – make selective variable tracking in checking mode production-ready, and tighten micro/macro overhead numbers with clear baselines.
- **Explainability** – generate better invariant descriptions on inference, and present violations with clearer pointers to the triggering API/variable context.
- **Debuggability & flexibility** – support dynamic queries at violation time (e.g., show which variables did not change and their properties) and collect global snapshots early in training to ground debugging.

## Near Term (Supporting Work)

- **Online monitoring** – integrate the checker directly into the collection process so violations are reported immediately during training.
- **Improved distributed support** – better handling of multi-GPU and multi-node runs, including tracing of distributed backends.
- **Stability fixes and tests** – add end-to-end tests for the full instrumentation→inference→checking pipeline and resolve known instrumentation edge cases.
- **Expanded documentation** – guidance on choosing reference runs and diagnosing issues, plus deeper technical docs.

## Medium Term

- **Invariant management** – tooling to filter, group, and suppress benign invariants at scale.
- **Extensible instrumentation** – plugins for third-party libraries and custom frameworks.
- **Performance improvements** – parallel inference and more efficient trace storage formats.
- **Pre-inferred invariant library** – curated, well-tested invariants for common PyTorch and HuggingFace workflows.

## Long Term

- **Automated root-cause analysis** – provide hints or suggested fixes when a violation is detected.
- **Cross-framework support** – expand beyond PyTorch to additional deep learning frameworks.

We welcome contributions in any of these areas. If you have ideas or want to help, please check the [CONTRIBUTING guide](./CONTRIBUTING.md) and open an issue to discuss!

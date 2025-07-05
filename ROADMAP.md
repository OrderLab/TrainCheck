# TrainCheck Roadmap

This document outlines planned directions for the TrainCheck project. The roadmap is aspirational and subject to change as we gather feedback from the community.

## Short Term

- **Online monitoring** – integrate the checker directly into the collection process so violations are reported immediately during training.
- **Pre-inferred invariant library** – ship a curated set of invariants for common PyTorch and HuggingFace workflows to reduce the need for manual inference.
- **Improved distributed support** – better handling of multi-GPU and multi-node runs, including tracing of distributed backends.
- **High-quality invariants** – publish well-tested invariants for PyTorch, DeepSpeed, and Transformers out of the box.
- **Demo assets** – publish a short demo video and GIFs illustrating the TrainCheck workflow.
- **Expanded documentation** – add guidance on choosing reference runs and diagnosing issues, plus deeper technical docs.
- **Stability fixes and tests** – resolve proxy dump bugs and add end-to-end tests for the full instrumentation→inference→checking pipeline.
- **Call graph updates** – document the call-graph generation process and keep graphs in sync with recent PyTorch versions.
- **Repository cleanup** – remove obsolete files and artifacts.

## Medium Term

- **Extensible instrumentation** – allow plugins for third-party libraries and custom frameworks.
- **Smarter invariant filtering** – tooling to help users manage large numbers of invariants and suppress benign ones.
- **Performance improvements** – explore parallel inference and more efficient trace storage formats.

## Long Term

- **Cross-framework support** – expand beyond PyTorch to additional deep learning frameworks.
- **Automated root-cause analysis** – provide hints or suggested fixes when a violation is detected.

We welcome contributions in any of these areas. If you have ideas or want to help, please check the [CONTRIBUTING guide](./CONTRIBUTING.md) and open an issue to discuss!

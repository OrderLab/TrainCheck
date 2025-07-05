# Contributing to TrainCheck

Thank you for your interest in contributing to TrainCheck. This project is actively maintained, and we welcome contributions from the community.

## 🧩 Areas for Contribution

We encourage contributions in the following areas:

- 🛠️ **Core Features**: Implementation of new invariant types, enhanced instrumentation, and improved fault localization.
- ⚙️ **Framework Support**: Expanding compatibility with DeepSpeed, HuggingFace Trainer, and multi-GPU environments.
- 📖 **Documentation**: Creating usage guides, walkthroughs, and clarifications (documentation is currently limited).
- 🔍 **Testing**: Adding realistic traces and increasing coverage for components that are not thoroughly tested.
- 🚧 **Engineering Improvements**: Enhancing log formatting, improving CLI usability, and performing code cleanup.

**For specific tasks and upcoming features where we need assistance, please see our [ROADMAP](./ROADMAP.md) for planned directions and priorities.**

## ⚠️ Important Information for Contributors

**TrainCheck** is structured as a standard Python library. Please follow conventional Python development practices.

- **Code Style**: Install the pre-commit hooks in your development environment:
    ```bash
    pip install pre-commit
    pre-commit install
    pip install isort
    ```
- **Documentation**: The documentation is currently under development. If you have any questions or require clarification, please join our Discord server via the [README](./README.md).
- **Testing**: The test structure is being finalized (see TEST.md (TBD)). In the interim, you are welcome to add unit or end-to-end tests in the `tests/` directory as appropriate.

## 💬 Questions?

Join our Discord server (see [README](./README.md)) — don’t hesitate to ping us there. We’d love to help you get started.



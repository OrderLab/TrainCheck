# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog],
and this project adheres to [Semantic Versioning].

## [Unreleased]

- /

## [0.2.4] - 2024-05-30

### Added

- upload trace-analyzer (only DS-1801 precondition supported)

### Fixed

- fix var_name inconsistency and allow module name to be passed in __call__() function

### Changed

- modify update rate threshold to 10s (around 3 iters) to cater for DS-1801

## [0.2.3] - 2024-05-27

### Added

- support global attributes dumping in meta_vars (e.g. DATA_PARALLEL_GROUP_RANK)

### Fixed

- avoid lower-level meta_vars to be overwritten by higher level ones

## [0.2.2] - 2024-05-26

### Added

- enable blacklist for attributes dumping

### Fixed

- dump original obj instead of str for primitive types

## [0.2.1] - 2024-05-25

### Added

- proxy wrapper logic to support wrapping of `torch.nn.parameters.Parameters` objects
by iterating through `module.named_parameters()`

### Changed

- move the proxy_wrapper configurations from `mldaikon.config.config` to `mldaikon.proxywrapper.config`

### Deprecated

- Remove the argument unproxying functionality in `tracer.global_wrapper`, make proxy_wrapper transparent
to the ml-daikon code

### Fixed

- fix the deprecated `var_name` field with the correct module name

- fix `torch_serialize` functionality

## [0.2.0] - 2024-05-25

### Added

- proxy wrapper logic to support wrapping of `torch.nn.Module` objects

- update rate screening by setting the `proxy_update_limit`, which renders flexibility for coverage-speed trade-off

- update the dumping logic in `dumper.py`, currently a trace could contain the following items: 
`{'process_id', 'thread_id', 'time', 'value', 'var_name', 'attributes', 'meta_vars', 'mode', 'stack_trace'}`
 (stack_trace is mainly for trace analysis purpose and could be removed in later versions)

### Changed

- `torch.nn.Module` type object could be passed down as function arguments

### Deprecated

- intrusive proxying of `__getAttr__` and `__setAttr__` methods


## [0.1.0] - 2024-05-20

- initial release

<!-- Links -->
[keep a changelog]: https://keepachangelog.com/en/1.0.0/
[semantic versioning]: https://semver.org/spec/v2.0.0.html

<!-- Versions -->
[unreleased]: https://github.com/Author/Repository/compare/v0.0.2...HEAD
[0.0.2]: https://github.com/Author/Repository/compare/v0.0.1...v0.0.2
[0.0.1]: https://github.com/Author/Repository/releases/tag/v0.0.1
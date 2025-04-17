# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog],
and this project adheres to [Semantic Versioning].

## [Unreleased]

- /

## [0.4.4] - 2025-02-02

### Added

- Maintain global registry to proxied objects (to access the vars, use `from traincheck.proxy_wrapper.proxy import get_registered_object`)
- Bypass tensor stats/hash computation if it has already been calculated

### Fixed

- Use getattr to access functions to prevent breaking the control flow for property functions

## [0.4.2] - 2024-07-12

### Added

- support pre-check for observer, do not dump it when the tensor is not changed
- support `hashing` functionality for cuda tensors

### Changed

- add proxy-related config to arg_parse of collect_trace, dump the adjusted configs to the instrumented src code
- classify the attributes inside proxy_config into config dicts for clarity
- rename the proxy config file to proxy_config.py

### Fixed

- fix `is_proxied` utility to cater for object without __dict__ attribute


## [0.4.1] - 2024-07-06

### Added

- support delta dumping for proxy trace (the functionality is deprecated and is fixed in proxy-v4.2)
- add the following switches to control the behavior of delta_dumping

    ```python
    delta_dump = True  # only dump the changed part of the object (if this is set to be False, we would dump the whole object no matter what values delta_dump_meta_var and delta_dump_attribute are)
    delta_dump_meta_var = True  # only dump the changed part of the meta_var
    delta_dump_attributes = True  # only dump the changed part of the attribute
    ```

## [0.4.0] - 2024-07-02

### Added

- support automated function observing based on function call graph level
- add function depth info of `torch/optim` library into `traincheck/static_analyzer/func_level` (file generated from modified pyan lib)
- add the following switches to control the behavior of automated function observer

    ```python
    enable_auto_observer = True  # automatically add observer to the function
    enable_auto_observer_depth = 3  # the depth of the function call that we want to observe
    observe_up_to_depth = False  # observe up to the depth of the function call, if False, only observe the function call at the depth
    neglect_hidden_func = (
        True  # neglect the hidden function (function that starts with '_')
    )
    neglect_hidden_module = True  # neglect the hidden module (module that starts with '_')
    observe_then_unproxy = False  # observe the function call and then unproxy the arguments
    ```

## [0.3.5] - 2024-06-29

### Added

- `dump_iter` switch to determine whether or not to dump the variable states from iterator (this would usually generated from e.g. enumerate(self._blocks) function)

### Fixed

- get rid of rubbish trace info due to flawed dumping logic handling

- fix type(Proxy) object handling by replacing with type_handle_traincheck_proxy(..) function using ast

## [0.3.4] - 2024-06-19

support `._version` based var update filtering, disable `__call__` value dumping in default

Use with caution (from Yuxuan): 
- This _version thing is only bumped with in-place ops, such as .add_, ._foreach_add_. Doing things like tensor.data = new_tensor won't bump the version counter.
- This means that doing this _version trick will cause us to lose all not-in-place var updates.
- This behavior is perfectly fine for model parameters as updates to model parameters have to be in-place for memory preservation purposes. 
- For those tensor updates that are not done in an in-place way, they are probably intermediate activations (torch.Tensor types) and thus it indeed might not make sense to keep track of them.

Reason to disable default `__call__` value dumping:
- those are mostly the intermediate results during forward propagation process (data related), would only be of interest if we target numerical issues from the data input side
- the dumping is quite costly as those data tensors are usually quite large

### Added

- Add version based var update filtering, default as follows:

```python
filter_by_tensor_version = True  # only dump the tensor when the version is changed
```

- Add different form of tensor dumping (statistics based dumping), default as follows:

```python
dump_tensor_version = False  # only dump the _version attribute of tensor
dump_tensor_statistics = False  # dump the statistics of tensor {min, max, mean, shape}
```

- Add switch to determine whether or not to dump return value from function call, default as follows:

```python
dump_call_return = False  # dump the return value of the function call
```


## [0.3.3] - 2024-06-12

support manual func_observer, trace supports PT-84911

Solves: #22

### Added

- customizing the proxy_wrapped_modules (initially have to be torch.nn.Modules and torch.nn.Parameters but now could basically be every type)

- realize func_observer to dump the changes in proxied object before and after the function call
 
- generate a depth dict for every functions inside torch library by establish torch function call graph (the depth denotes the shortest number of calls a function would take to trigger a C level function)

### Changed

- integrate the sparse type check together to handled_obj_type in `proxy_handler.py`

### Removed

- remove the deprecated torch.Tensor handling logic from `traincheck/proxy_wrapper/proxy.py:#L231-L295`, since the handling logic for Tensor type is basically the same compared with other types after redesign


## [0.3.2] - 2024-06-09

Main edits: support automated unproxy for C level API invocation and enhance the trace coverage

Solves #26

### Added

- de-proxing args for C-level builtin function call

- type-aware proxy_handler to decide whether an object should be proxied or not

- function wrapping for a callale module attribute

- wrap torch.optim.optimizer._default_to_fused_or_foreach for type() handling

### Changed

- formalize the get_frame_array functionality

- support `torch._C` global wrapping for args de-proxing purpose

### Removed

- remove the var_list variable name tracing in Proxy v2

### Fixed

- pass the proxied object to the `_parameters` dict to solve the insufficient wrapping coverage in PT-84911


## [0.3.1] - 2024-06-01
 
The `DS-1801` precondition inference is fully supported in this version.

### Changed
 
 - dump whole tensor instead of {`min`, `max`, `shape`} attributes

 - update meta_var dumper to filter out files inside the traincheck folder

### Fixed

 - maintain the block list for `meta_var` and `attributes`

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

- move the proxy_wrapper configurations from `traincheck.config.config` to `traincheck.proxywrapper.config`

### Deprecated

- Remove the argument unproxying functionality in `tracer.global_wrapper`, make proxy_wrapper transparent
to the traincheck code

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
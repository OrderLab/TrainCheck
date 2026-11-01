You additionally have TrainCheck's runtime observation facility. Its source is
available in the supplied `traincheck/` package. Use it if it would help your
next decision; it is optional.

TrainCheck can wrap selected Python/PyTorch callables and record API entry,
successful return, or exception events, with process/thread/call identities and
timestamps. You can choose argument fields, enable or suppress recording for
selected scopes, and write your own analysis or checker in any useful format.
There is no requirement to use an invariant inference engine or a predefined
relation language. Selective observation lets you limit the amount of recorded
data; actual cost depends on the observations and should be measured when relevant.

The basic callable interface is:

```python
from traincheck.config import config
from traincheck.instrumentor.tracer import wrapper

# callable_to_observe is a function you selected from your program.
observed = wrapper(
    callable_to_observe,
    is_bound_method=False,
    scan_proxy_in_args=False,
    dump_stack_trace=False,
    dump_ret=False,
    handle_proxy=False,
)
config.DISABLE_WRAPPER = False
# Use observed(...) at the same call site with the original arguments.
```

The training service sets `TRAINCHECK_OUTPUT_DIR` to a unique per-rank directory
before importing your pipeline. Do not share an output file between ranks.
Native JSON lines are written as `trace_API_*.log` and become available after the
worker exits. Inspect actual records before writing a parser; primitive values
have tagged payloads such as `{"int": 3}`. Post records do not repeat arguments.
Preserve identities when joining entry/return records.

For selective argument recording, `dump_args_config` maps a positional argument
index as a string, or a keyword name, to the desired metadata selection; an empty
selection value records that argument normally. Omitted arguments are not
recorded. For example, `{"0": {}}` records the first positional argument.
`dump_ret=False` omits the return value. The current implementation does not
apply the advertised `dump_ret_config` field filter, so do not rely on it.

`traincheck.instrumentor.tracer.suppress_tracing()` excludes recorder-owned work
within a scope. `traincheck.instrumentor.control.start_step()` can maintain step
metadata and apply the configured sampling interval. Arbitrary function locals
are not automatically visible: a small explicit probe or helper at the actual
execution site can expose a value you decide is useful. Live configuration-file
changes do not automatically reconfigure already-installed wrappers.

Recordings should capture actual execution state without changing computation,
RNG, or object lifetimes. A passing predicate needs evidence that the intended
observations actually occurred; missing records, failed calls, or checker errors
are distinct from successful checks. You may analyze raw traces offline or
manage your own checker; no problem-specific rule is supplied.

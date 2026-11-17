# Delaying DDP gradient synchronization to token-window boundaries

Three new matched agent pairs start with the same working streaming DDP loop.
The requested improvement is to delay gradient communication until each token
window ends. The task preserves the numeric trajectory and streaming input
contract, including epoch tails. One arm receives a documented optional
two-line TrainCheck diagnostic. The adapter runs public examples, profiles
native NCCL all-reduces and uses TrainCheck fingerprints for final replica
agreement. This compares access to the complete optional diagnostic workflow,
including its new task adapter. Both arms may write diagnostics and receive
the same GPU request, CLI and decision budgets. The task does not hint at a
pre-existing bug or a preferred implementation.

The operator stores each submitted production file and samples intermediate
edits. Native CLI input, cached input, output and elapsed time are kept apart
from GPU usage. First valid submission means the first frozen decision whose
code later passes every held-out check. Its cost ends before any GPU run
requested by that decision. Agent verification and independent replay are
separate events and GPU costs. Report time, token use and GPU cost for each
pair. Count submitted diagnostic source lines for each agent, including short
entry points that call the optional checker. Audit execution using actual
check output, not filename or documentation reads alone.

The prior [token budget study](token-budget-agent-study-20261118.md) supplies
the workload, independent concatenated-batch reference and frozen numerical
scenarios. The augmented scorer checks pre-step gradients, post-step parameters
and momentum, window boundaries, and repeated or early gradient communication.
It uses CPU profiler records emitted by the underlying NCCL process group with
input shape recording active. The workload uses single-element all-reduces for
token counts. Larger all-reduces carry gradient data. Compare the aggregate
gradient element count with the full trainable gradient payload at each step.
Separate buckets or parameters may be reduced individually. Each reduction
must occur following the final backward marker of that window and precede the
optimizer step. The audit counts launches, not wire traffic or profiler timing.
It applies to this workload's DDP reducer and ordinary tensor all-reduce
implementations.

Native qualification compares the working per-microbatch-sync starter, a
correct once-per-window implementation, and a private variant omitting
synchronization in the final partial window. The optional public check uses
separately specified public cases and prints evidence on each rank. Private
replay uses all six frozen scenarios, and is withheld until the solvers finish.
The low-level collective parser is shared, so controls verify its positive
and negative behavior. This is a small workflow pilot, with no model-training
performance claim.

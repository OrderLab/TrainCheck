Improve the token-budget mode of this working two-rank DDP pipeline in
`training.py`. Synchronize gradients only when a token window ends, once
per full gradient payload per window. Individual buckets or parameter groups
may reduce separately at that boundary. All-reduces for integer token counts
are still allowed at each microbatch. Use DDP's reducer or ordinary
`torch.distributed.all_reduce` for the gradient data.

Preserve the existing behavior: a window consists of whole synchronous
microbatches, closes at or above the global target-token budget, and flushes
its remaining targets at epoch end. Preserve update boundaries, gradient,
parameter and momentum values within floating point tolerance. Keep the
original fixed mode. Do not hold graphs for an entire window, buffer a window
of inputs or require the input iterable to have a length. Each rank has the
same number of microbatches, and each synchronous microbatch contains at
least one nonpadding target globally.

Run the ordinary example as `torchrun --standalone --nproc-per-node=2 train.py
--token-budget 60`. A separate `diagnostic.py` is allowed. The deliverable is
`training.py`. Report when you consider it ready and summarize your evidence.

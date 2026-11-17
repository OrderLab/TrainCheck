The team wants to run this DDP training pipeline on variable-length data with a
global target-token budget per optimizer update. Implement the feature in
`training.py`. The current fixed mode performs one optimizer update per
synchronous microbatch. Preserve that behavior when no budget is supplied.

With a positive budget, accumulate whole synchronous microbatches in input
order. Apply an update as soon as their combined number of non-padding targets
across the training ranks reaches or exceeds the budget. Start the next window
empty and apply the final nonempty partial window at epoch end. For each update,
preserve the objective and gradient of ordinary training on its combined global
batch. Keep the training memory bounded by a single microbatch per rank. The
ranks receive the same number of microbatches and every synchronous microbatch
contains at least one target globally. Keep the existing model and optimizer.

The ordinary entry point is `torchrun --standalone --nproc-per-node=2 train.py`.
Pass `--token-budget 50` to use the feature. Additional diagnostic scripts are
welcome in your workspace; the production deliverable is `training.py`.
Report when you consider the change ready and summarize your evidence.

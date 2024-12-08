# DDP MultiGPU

Adapted from PyTorch's official DDP example for single node multi GPU training.

## Adaptions:
1. Added annotation for init and training stages
2. Annotated the training loop to register the 'step' variable to TrainCheck
3. Fixed a bug in the original pipeline: cross_entropy returns loss always being zero as the output logit is just one single number, changed to `torch.nn.BCEWithLogitsLoss`

Requires 8 GPUs to run.
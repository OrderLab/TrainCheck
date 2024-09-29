## Detected Bugs

| Bug ID | Framework/User | Short Summary | Detected? <Relation>.<number-of-invariants> | Input To Infer Necessary Invariants | Inv Description | NOTE |
| - | - | - | - | - | - | - |
| [PT-FORUM-84911](https://github.com/OrderLab/machine-learning-issues/tree/main/PyTorch-FORUM84911)             | User      | No trainable layers exist                  | APIContain.1        | mnist.py using adam                                                  | `Adam.step` should lead to parameter update       | |
| [DS-1801](https://github.com/OrderLab/machine-learning-issues/tree/main/DeepSpeed-1801)                        | Framework | LayerNorm out-of-sync in tensor parallel   | VarConsistency.1    | Same version of Megatron-Deepspeed using FP16 or BF16 (no grad clip) | Consistency between `parameter.data`              | |
| [stackoverflow-60335387](https://github.com/OrderLab/machine-learning-issues/tree/main/stackoverflow-60335387) | User      | Missing call to `optimizer.zero_grad`      | VarPeriodicChange.1 | mnist.py                                                             | `parameter.grad` periodically `None`              | To be verified @YijunWang1121            |
|                                                                                                                |           |                                            | Lead.1              | mnist.py                                                             | `zero_grad` co-occur with `backward`              | To be verified @Countigo1234             |
| [PT-115607](https://github.com/OrderLab/machine-learning-issues/tree/main/PyTorch-115607)                      | Framework | Missing guards causes obsolete kernel used | APIContain.1        | mnist.py                                                             | `zero_grad` change `grad` from non-zero to `None` | Not verifiable due to no instrumentation |
|                                                                                                                |           |                                            | VarPeriodicChange.1 | mnist.py                                                             | `parameter.grad` periodically `None`              | Not verifiable due to no instrumentation |




## Bugs Potentially Detectable

| Bug ID | Short Summary | Potential Relation | Person Responsible | 
| - | - | - | - |
| [PT-51800](https://github.com/OrderLab/machine-learning-issues/tree/main/PyTorch-51800)   | `forward` should be invoked to init weights prior to `eval` | Lead | @Countigo1234 |
| [PT-124357](https://github.com/OrderLab/machine-learning-issues/tree/main/Pytorch-124357) | `torch.Compile` leading to incorrect kernels                | TBD  | @XuRunhui |
| [TE-1047](https://github.com/NVIDIA/TransformerEngine/issues/1047)                        | TBD | TBD | @essoz |
| [TO related bugs](https://github.com/pytorch/pytorch/issues/84803)                        | TBD | TBD | @keke1022 | 
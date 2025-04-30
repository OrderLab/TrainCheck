import os

import torch

os.environ["ML_DAIKON_OUTPUT_DIR"] = "."
from traincheck.instrumentor.tracer import Instrumentor  # noqa

Instrumentor(
    torch,
    scan_proxy_in_args=True,
    use_full_instr=False,
    funcs_to_instr=[
        "torch.cuda.is_available",
        "torch._VariableFunctionsClass.matmul",
        "torch._VariableFunctionsClass.bmm",
        "torch.nn.modules.transformer.Transformer.forward",
        "torch._VariableFunctionsClass.conv2d",
    ],
    API_dump_stack_trace=False,
).instrument()

num_executions_per_API = 100

# simplest:
for i in range(num_executions_per_API):
    torch.cuda.is_available()

# simple, torch.matmul
for i in range(num_executions_per_API):
    torch.matmul(torch.randn(10, 10), torch.randn(10, 10))

# let's also run the forward pass of a more complicated kernel
a = torch.randn(50, 20, 30)
b = torch.randn(50, 30, 40)
for i in range(num_executions_per_API):
    torch.bmm(a, b)  # Batched matrix multiplication

# super complex, transformer forward pass
transformer = torch.nn.Transformer(nhead=16, num_encoder_layers=12)
src = torch.randn(10, 32, 512)
tgt = torch.randn(20, 32, 512)
for i in range(num_executions_per_API):
    transformer(src, tgt)

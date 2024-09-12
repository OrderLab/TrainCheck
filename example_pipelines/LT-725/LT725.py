import thunder
import torch


def foo(x, w):
    return thunder.prims.linear(x, w, None)


device = torch.device("cuda")
with device:
    # fp32 inputs
    x, w = torch.randn(16, 16), torch.randn(16, 16)
    print(x.dtype, w.dtype)

with torch.autocast("cuda", torch.bfloat16):
    jfoo = thunder.jit(foo)
    jit_out = jfoo(x, w)

print(thunder.last_traces(jfoo)[-1])

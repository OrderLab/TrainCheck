import torch
from torch import Tensor

MULTIPLIER = 6364136223846793005
INCREMENT = 1
MODULUS = 2**64

# Define a fixed constant tensor
FIXED_CONSTANT = torch.tensor([42], dtype=torch.int64)  # Example fixed constant


def tensor_hash(x: Tensor, with_parallel: bool = False) -> int:
    if with_parallel:
        # Ensure the input is a floating-point tensor
        assert x.dtype in [torch.float32, torch.float64]

        # Convert the floating-point tensor to an integer representation
        x = (x * 1e8).to(torch.int64)  # Scale and convert to int64

        # # Expand the fixed constant tensor to match the shape of x
        # constant_tensor = FIXED_CONSTANT.expand_as(x).to

        # # Multiply the tensor with the constant tensor
        # x = x * constant_tensor

        # Ensure the tensor is of integer type
        assert x.dtype == torch.int64

        # Reduce the tensor to a single hash value
        while x.ndim > 0:
            x = _reduce_last_axis(x)
        # convert tensor to value
        return int(x.item())
    else:  # conventional approach using hashlib, use as the baseline to test the accuracy of the parallel approach
        # if on cuda, move to cpu. using hashlib to hash the tensor
        if x.is_cuda:
            x = x.cpu()
        import hashlib

        return (
            int(hashlib.sha256(x.detach().numpy().tobytes()).hexdigest(), 16) % MODULUS
        )


@torch.no_grad()
def _reduce_last_axis(x: Tensor) -> Tensor:
    assert x.dtype == torch.int64
    acc = torch.zeros_like(x[..., 0])
    for i in range(x.shape[-1]):
        acc *= MULTIPLIER
        acc += INCREMENT
        acc += x[..., i]
    return acc

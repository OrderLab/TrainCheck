import time

import numpy as np
import torch
from numba import cuda
from torch import Tensor

MULTIPLIER = 6364136223846793005
INCREMENT = 1
MODULUS = 2**64


@cuda.jit("void(int64[:, :], int64[:], int64, int64)")
def cuda_hash_kernel(data, hash_values, multiplier, increment):
    idx = cuda.grid(1)
    if idx < data.shape[0]:
        hash_value = 0
        for i in range(data.shape[1]):
            hash_value = hash_value * multiplier + data[idx, i] + increment
        hash_values[idx] = hash_value


def hash_tensor_cuda(x):

    # if x is more than 2D, flatten it to 2D
    if x.ndim > 2:
        x = x.flatten(start_dim=0, end_dim=-2)
    else:
        # if x is 1D, add a dimension to make it 2D (n x 1)
        x = x.unsqueeze(0)
    (rows, _) = x.shape

    hash_values = cuda.device_array(rows, dtype=np.int64)

    threads_per_block = 16
    blocks_per_grid = (rows + threads_per_block - 1) // threads_per_block

    cuda_hash_kernel[blocks_per_grid, threads_per_block](
        x, hash_values, MULTIPLIER, INCREMENT
    )

    x = hash_values.copy_to_host()

    return int(x[0])


@torch.no_grad()
def _reduce_last_axis(x: Tensor) -> Tensor:
    assert x.dtype == torch.int64
    acc = torch.zeros_like(x[..., 0])
    for i in range(x.shape[-1]):
        acc *= MULTIPLIER
        acc += INCREMENT
        acc += x[..., i]
    return acc


def tensor_hash(x: Tensor, with_parallel: bool = True, with_cuda: bool = True) -> int:
    if with_parallel:
        if with_cuda and x.is_cuda:
            # Ensure the input is a floating-point tensor
            assert x.dtype in [
                torch.float32,
                torch.float64,
                torch.bfloat16,
                torch.float16,
                torch.float,
            ]

            # Convert the floating-point tensor to an integer representation
            x = (x * 1e8).to(torch.int64)

            # Ensure the tensor is of integer type
            assert x.dtype == torch.int64

            return hash_tensor_cuda(x)
        else:
            # Ensure the input is a floating-point tensor
            assert x.dtype in [
                torch.float32,
                torch.float64,
                torch.bfloat16,
                torch.float16,
                torch.float,
            ]

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

        return int(hashlib.sha256(x.detach().numpy().tobytes()).hexdigest(), 16)


def print_current_memory_usage():
    current_memory = torch.cuda.memory_allocated()
    print(f"Current memory usage: {current_memory}")


def profile_tensor_hashing(tensor, with_parallel: bool = True, with_cuda: bool = True):
    start = time.time()
    print(
        f"Hashing tensor using parallel approach (with cuda): {tensor_hash(tensor, with_parallel=with_parallel, with_cuda=with_cuda)}"
    )
    end = time.time()
    print(f"Time taken: {end-start}")


if __name__ == "__main__":
    # create tensor with different size and different types, test the time of hashing using parallel and non-parallel approach and compare the results
    # size: 10*10, 100*100, 1000*1000, 10000*10000, 10000*10000*10000
    # create a list of tensors

    sizes = [10, 10000]
    tensors = []
    for size in sizes:
        tensor = torch.arange(size * size).reshape(size, size).bfloat16()
        tensors.append(tensor)
    # change the tensor type to bfloat16
    for _ in range(200):
        for tensor in tensors:
            print(f"Tensor size: {tensor.size()}")
            # time
            tensor = tensor.cuda()
            # measure the current cuda memory usage
            print_current_memory_usage()

            profile_tensor_hashing(tensor, with_parallel=True, with_cuda=True)

            print_current_memory_usage()

            profile_tensor_hashing(tensor, with_parallel=True, with_cuda=False)

            print_current_memory_usage()

            print("=============================================")

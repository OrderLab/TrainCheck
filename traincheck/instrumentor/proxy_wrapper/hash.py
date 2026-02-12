import numpy as np
import torch
from numba import cuda
from torch import Tensor

MULTIPLIER = 6364136223846793005
INCREMENT = 1
MODULUS = 2**64

# Define a fixed constant tensor
FIXED_CONSTANT = torch.tensor([42], dtype=torch.int64)  # Example fixed constant

if torch.cuda.is_available():

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
    elif x.ndim == 1:
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


def hash_tensor_cpu(x: torch.Tensor) -> int:
    """Optimized CPU hash for PyTorch tensors using NumPy with correct iterative hashing."""
    if x.ndim > 2:
        x = x.reshape(-1, x.shape[-1])
    elif x.ndim == 1:
        x = x.reshape(1, -1)

    x_np = x.cpu().numpy().astype(np.int64)  # Ensure conversion to NumPy int64

    # Compute cumulative multiplication just like in the loop version
    if x_np.shape == ():  # Handle scalar case
        hash_values = np.zeros(1, dtype=np.int64)
        x_np = np.expand_dims(x_np, axis=(0, 1))  # Convert scalar to 2D array
    elif x_np.ndim == 1:  # Handle vector case
        hash_values = np.zeros(1, dtype=np.int64)  # Single hash value for the vector
        x_np = np.expand_dims(x_np, axis=0)  # Convert vector to 2D array with one row
    else:
        hash_values = np.zeros(x_np.shape[0], dtype=np.int64)  # Hash storage per row

    # Accumulate the hash row-wise, matching the loop behavior
    for i in range(x_np.shape[1]):
        hash_values = (
            hash_values * MULTIPLIER + x_np[:, i] + INCREMENT
        )  # Mimic loop logic

    return int(hash_values[0])  # Return hash for the first row


def efficient_hash_tensor_cpu(x):
    # return int(hash_values[0])
    # Reduce the tensor to a single hash value (It would produce different hash value than the above implementation)
    while x.ndim > 0:
        x = _reduce_last_axis(x)
    # convert tensor to value
    return int(x.item())


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
    if hasattr(x, "_traincheck_tensor_hash"):
        return x._traincheck_tensor_hash
    if with_parallel:
        if x.dtype in [
            torch.float32,
            torch.float64,
            torch.bfloat16,
            torch.float16,
            torch.float,
        ]:
            # Convert the floating-point tensor to an integer representation
            x = (x * 1e8).to(torch.int64)
        else:
            assert x.dtype in [
                torch.int32,
                torch.int64,
                torch.uint8,
                torch.int8,
                torch.int16,
            ], f"Unsupported tensor type for hashing, expected either int or float, got {x.dtype}"
            # Ensure the tensor is of integer type
            x = x.to(torch.int64)

        # Ensure the tensor is of integer type
        assert x.dtype == torch.int64

        if with_cuda and x.is_cuda:
            # Ensure the input is a floating-point tensor
            result = hash_tensor_cuda(x)
            return result
        else:
            result = hash_tensor_cpu(x)
            return result
    else:  # conventional approach using hashlib, use as the baseline to test the accuracy of the parallel approach
        # if on cuda, move to cpu. using hashlib to hash the tensor
        if x.is_cuda:
            x = x.cpu()
        import hashlib

        result = int(hashlib.sha256(x.detach().numpy().tobytes()).hexdigest(), 16)
        return result

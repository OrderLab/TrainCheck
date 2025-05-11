import torch
from traincheck.proxy_wrapper.hash import (
    _reduce_last_axis,
    hash_tensor_cpu,
    hash_tensor_cuda,
)


def test_hash_tensor_cpu_vs_cuda(x):
    cpu_result = hash_tensor_cpu(x)
    gpu_result = hash_tensor_cuda(x)
    # cpu_result = _reduce_last_axis(x)
    print(f"CPU result: {cpu_result}")
    print(f"GPU result: {gpu_result}")
    assert (
        cpu_result == gpu_result
    ), f"CPU result {cpu_result} does not match GPU result {gpu_result}"


if __name__ == "__main__":
    # create tensor of int64
    x = torch.tensor(
        [
            [
                -0.3073,
                0.2444,
                -0.1429,
                0.2990,
                0.1588,
                -0.3054,
                0.0287,
                0.3096,
                -0.2593,
                0.0203,
            ],
            [
                -0.1730,
                -0.2439,
                -0.2617,
                0.1757,
                -0.0990,
                0.1052,
                -0.1954,
                -0.2256,
                0.2092,
                0.2138,
            ],
            [
                -0.1078,
                0.1753,
                -0.1492,
                0.2693,
                -0.2460,
                0.0176,
                -0.0186,
                -0.0495,
                0.1136,
                0.0670,
            ],
            [
                -0.1969,
                0.2628,
                0.0880,
                -0.0736,
                0.0485,
                -0.1396,
                -0.1833,
                -0.1603,
                -0.0694,
                -0.2482,
            ],
        ],
        dtype=torch.float64,
    )
    x = (x * 1e8).to(torch.int64).to("cuda")
    test_hash_tensor_cpu_vs_cuda(x)
    print("Hashing test passed")

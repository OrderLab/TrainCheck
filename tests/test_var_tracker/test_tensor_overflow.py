import torch

# Maximum int64 value
max_val = torch.iinfo(torch.int64).max

# CPU example
a_cpu = torch.tensor([max_val], dtype=torch.int64)
b_cpu = a_cpu + 1

print("CPU:")
print("Tensor a:", a_cpu)
print("Tensor b (a + 1):", b_cpu)

# GPU example (if CUDA is available)
if torch.cuda.is_available():
    a_gpu = a_cpu.to("cuda")
    b_gpu = a_gpu + 1

    print("\nGPU:")
    print("Tensor a:", a_gpu)
    print("Tensor b (a + 1):", b_gpu)
else:
    print("\nCUDA is not available on this device.")

import torch
from traincheck.proxy_wrapper.hash import tensor_hash


def test_model_hash(device):
    # create a small model
    model = torch.nn.Sequential(
        # 1st conv layer
        torch.nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1),
        torch.nn.ReLU(),
        torch.nn.MaxPool2d(kernel_size=2, stride=2, padding=0),
        # 2nd conv layer
        torch.nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
        torch.nn.ReLU(),
        torch.nn.MaxPool2d(kernel_size=2, stride=2, padding=0),
    )
    model.to(device)
    model.cpu()
    model.cuda(device)

    # test model_hash for every tensor in the model
    for name, param in model.named_parameters():
        hash_value = tensor_hash(param, with_parallel=True, with_cuda=True)
        print(f"name: {name}, hash_value: {hash_value}")


def test_DDP_wrapped_model_hash(device):
    # create a small model
    model = torch.nn.Sequential(
        # 1st conv layer
        torch.nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1),
        torch.nn.ReLU(),
        torch.nn.MaxPool2d(kernel_size=2, stride=2, padding=0),
        # 2nd conv layer
        torch.nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
        torch.nn.ReLU(),
        torch.nn.MaxPool2d(kernel_size=2, stride=2, padding=0),
    )
    model.to(device)
    model.cpu()
    model.cuda(device)
    model = torch.nn.parallel.DistributedDataParallel(
        model, device_ids=[device], output_device=device
    )

    # update the tensor of the model
    for param in model.parameters():
        param.data = param.data + 1

    # test model_hash for every tensor in the model
    for name, param in model.named_parameters():
        hash_value = tensor_hash(param, with_parallel=True, with_cuda=True)
        print(
            f"name: {name}, hash_value: {hash_value}, rank: {torch.distributed.get_rank()}"
        )


def test_tensor_hash(device):
    # create tensor
    tensor = torch.tensor(
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
            [
                0.0915,
                0.0344,
                0.2479,
                0.0195,
                -0.2745,
                -0.0254,
                0.0576,
                0.0463,
                -0.2242,
                0.0187,
            ],
            [
                -0.2141,
                0.1785,
                -0.1526,
                -0.0447,
                0.1207,
                0.2580,
                -0.0853,
                -0.0161,
                -0.1084,
                0.0837,
            ],
            [
                0.2043,
                -0.2366,
                -0.3145,
                0.1969,
                -0.2308,
                0.0477,
                0.2222,
                -0.1728,
                0.2560,
                0.0519,
            ],
            [
                0.3159,
                0.0768,
                -0.1080,
                0.1028,
                -0.0691,
                -0.0015,
                -0.1572,
                -0.2224,
                0.1585,
                0.2332,
            ],
            [
                -0.1343,
                -0.0161,
                0.1730,
                0.3076,
                0.0059,
                0.1310,
                0.1867,
                -0.0144,
                0.1348,
                -0.1663,
            ],
            [
                -0.1874,
                -0.0561,
                -0.2298,
                -0.1865,
                -0.1589,
                0.1444,
                -0.1652,
                0.0596,
                0.2129,
                -0.3133,
            ],
        ],
        dtype=torch.float32,
    ).to(device)
    print(f"tensor: {tensor}")

    # test tensor_hash
    hash_value = tensor_hash(tensor, with_parallel=True, with_cuda=True)
    print(f"hash_value: {hash_value}")


# test tensor_hash worked on GPU RANK other than 0
def test_hash_tensor_cuda():
    # test tensor_hash worked on GPU RANK other than 0
    import os

    import numpy as np
    import torch
    import torch.distributed as dist

    # init process group
    dist.init_process_group(backend="nccl", init_method="env://")
    rank = dist.get_rank()
    print(f"rank: {rank}")
    world_size = dist.get_world_size()
    device = torch.device("cuda", rank)
    torch.cuda.set_device(device)
    print(f"rank: {rank}, world_size: {world_size}")

    # test model_hash
    # test_model_hash(device)
    # test DDP wrapped model_hash
    test_DDP_wrapped_model_hash(device)

    # test tensor_hash
    # test_tensor_hash(device)
    # finalize process group
    dist.destroy_process_group()


if __name__ == "__main__":
    test_hash_tensor_cuda()

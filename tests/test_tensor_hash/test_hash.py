import os

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn
import torch.optim as optim
from torch.nn.parallel import DistributedDataParallel as DDP

from mldaikon import annotate_stage
from mldaikon.instrumentor import meta_vars


class ToyModel(nn.Module):
    def __init__(self):
        super(ToyModel, self).__init__()
        self.net1 = torch.nn.Linear(10, 10)
        self.relu = torch.nn.ReLU()
        self.net2 = torch.nn.Linear(10, 5)
        from mldaikon.proxy_wrapper.hash import tensor_hash

        print("Perform tensor hash in init")
        result = tensor_hash(self.net1.weight)
        print("Tensor hash result: ", result)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.relu(self.net1(x))
        # from mldaikon.proxy_wrapper.hash import tensor_hash
        # print("Perform tensor hash in forward pass")
        # result  = tensor_hash(x)
        # print("Tensor hash result: ", result)
        return self.net2(x)


def ddp_run_single_step(move_model_to_cpu_and_back: bool) -> None:
    if move_model_to_cpu_and_back:
        print("Running DDP example with model moved to CPU and back.")
    else:
        print("Running DDP example without moving model to CPU and back.")
    world_size = torch.cuda.device_count()
    mp.spawn(
        _ddp_run_single_step,
        args=(
            world_size,
            move_model_to_cpu_and_back,
        ),
        nprocs=world_size,
        join=True,
    )


def _ddp_run_single_step(
    rank: int, world_size: int, move_model_to_cpu_and_back: bool
) -> None:
    annotate_stage("init")
    print(f"Running basic DDP example on rank {rank}.")
    _ddp_setup(rank, world_size)

    # create model and move it to GPU with id rank
    model = ToyModel()
    from mldaikon.proxy_wrapper.hash import tensor_hash

    print("1. Perform tensor hash in init (before moving to GPU)")
    result = tensor_hash(model.net1.weight)
    print("Tensor hash result: ", result)

    model.to(rank)
    # dump hash of the model
    from mldaikon.proxy_wrapper.hash import tensor_hash

    print("2. Perform tensor hash in init (after moving to GPU)")
    result = tensor_hash(model.net1.weight)
    print("Tensor hash result: ", result)

    loss_fn = nn.MSELoss()
    optimizer = optim.SGD(model.parameters(), lr=0.001)
    ddp_model = DDP(model, device_ids=[rank])

    if move_model_to_cpu_and_back:
        # move the model to CPU and back to GPU
        # this is to test that the gradients are synchronized even if the model is moved to CPU
        ddp_model.cpu()
        ddp_model.cuda(rank)

    # create data
    x = torch.randn(20, 10)
    labels = torch.randn(20, 5)

    annotate_stage("training")
    # forward pass
    x = x.to(rank)
    labels = labels.to(rank)
    outputs = ddp_model(x)

    print(f"Rank {rank}: {outputs}")
    _ddp_cleanup()


# def _check_gradients_between_workers(
#     world_size: int,
#     move_model_to_cpu_and_back: bool,
#     model: nn.Module,
#     verbose: bool = False,
# ) -> None:
#     """Gather gradients from all processes and check that they are equal."""
#     for k, v in model.named_parameters():
#         if v.grad is not None:
#             grad_list = [torch.zeros_like(v.grad) for _ in range(world_size)]
#             dist.all_gather(grad_list, v.grad, async_op=False)
#             for grad in grad_list:
#                 if not torch.allclose(v.grad, grad):
#                     if verbose:
#                         print(
#                             {
#                                 "grad": v.grad.mean(),
#                                 "other_grad": grad.mean(),
#                             }
#                         )
#                     raise RuntimeError(
#                         f"move_model_to_cpu_and_back={move_model_to_cpu_and_back}. "
#                         f"Gradients are not equal across processes (p={k})"
#                     )


def _ddp_setup(rank: int, world_size: int) -> None:
    meta_vars["_DATA_PARALLEL_RANK"] = rank
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12367"
    # dist.init_process_group("gloo", rank=rank, world_size=world_size)
    # change to use nccl
    # dist.init_process_group("nccl", rank=rank, world_size=world_size)
    dist.init_process_group(backend="nccl", init_method="env://")


def _ddp_cleanup() -> None:
    dist.destroy_process_group()


if __name__ == "__main__":
    # ddp_run_single_step(move_model_to_cpu_and_back=True)
    ddp_run_single_step(move_model_to_cpu_and_back=False)

import json
import os
import random

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms

# %%
from efficientnet_pytorch import EfficientNet

# the following import is required for training to be robust to truncated images
from PIL import ImageFile
from torchvision import datasets
from tqdm import tqdm

shape = (224, 224)
log_dir = f"runs/{shape[0]}"
os.makedirs("runs", exist_ok=True)
os.makedirs(log_dir, exist_ok=True)

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

seed = 786
os.environ["PYTHONHASHSEED"] = str(seed)
## Torch RNG
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
## Python RNG
np.random.seed(seed)
random.seed(seed)

## CuDNN determinsim
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
# torch.cuda.empty_cache()

## Specify appropriate transforms, and batch_sizes
data_transform = {
    "train": transforms.Compose(
        [
            transforms.Resize(shape),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.5),
            transforms.RandomRotation(30),
            transforms.ColorJitter(brightness=0, contrast=0.5, saturation=0.5, hue=0.5),
            transforms.ToTensor(),
            transforms.Normalize([0.2829, 0.2034, 0.1512], [0.2577, 0.1834, 0.1411]),
        ]
    ),
    "valid": transforms.Compose(
        [
            transforms.Resize(shape),
            transforms.ToTensor(),
            transforms.Normalize([0.2829, 0.2034, 0.1512], [0.2577, 0.1834, 0.1411]),
        ]
    ),
}


dir_file = "dataset"
train_dir = os.path.join(dir_file, "train")
valid_dir = os.path.join(dir_file, "dev")
train_set = datasets.CIFAR100(
    root="./data", train=True, download=True, transform=data_transform["train"]
)
valid_set = datasets.CIFAR100(
    root="./data", train=False, download=True, transform=data_transform["valid"]
)

batch_size = 64
train_loader = torch.utils.data.DataLoader(
    train_set, batch_size=batch_size, pin_memory=False, num_workers=0, shuffle=False
)
valid_loader = torch.utils.data.DataLoader(
    valid_set, batch_size=1, pin_memory=False, num_workers=0, shuffle=False
)


data_transfer = {"train": train_loader, "valid": valid_loader}

## ML-DAIKON Instrumentation
model_transfer = EfficientNet.from_pretrained("efficientnet-b0")
n_inputs = model_transfer._fc.in_features

num_classes = 100
model_transfer._fc = nn.Linear(n_inputs, num_classes)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
use_cuda = torch.cuda.is_available()

if torch.cuda.device_count() > 1:
    print("Let's use", torch.cuda.device_count(), "GPUs!")
# model_transfer = nn.DataParallel(
#     model_transfer
# )  # moved out from the above if statement

criterion_transfer = nn.CrossEntropyLoss()  # moved out from the above if statement
model_transfer.to(device)


for name, param in model_transfer.named_parameters():
    if "bn" not in name:
        param.requires_grad = False

for param in model_transfer._conv_stem.parameters():
    param.requires_grad = False

for param in model_transfer._fc.parameters():
    param.requires_grad = True

nb_classes = num_classes


ImageFile.LOAD_TRUNCATED_IMAGES = True


def train(n_epochs, loaders, model, optimizer, criterion, use_cuda, save_path):
    """returns trained model"""
    # initialize tracker for minimum validation loss
    os.makedirs(save_path, exist_ok=True)

    valid_loss_min = np.Inf
    res = []
    for epoch in tqdm(range(1, n_epochs + 1), desc="Epochs"):
        # initialize variables to monitor training and validation loss
        ## ML-DAIKON Instrumentation
        train_loss = 0.0
        valid_loss = 0.0
        correct = 0.0
        total = 0.0
        accuracy = 0.0

        model.train()
        iters = 0
        for batch_idx, (data, target) in enumerate(
            tqdm(loaders["train"], desc="Training")
        ):
            iters += 1
            if iters > 50:
                print("ML-DAIKON: Breaking after 10 iterations for testing purposes")
                break
            # move to GPU
            if use_cuda:
                data, target = data.to("cuda", non_blocking=True), target.to(
                    "cuda", non_blocking=True
                )
            optimizer.zero_grad()
            # model = Proxy(model, is_root=True)
            output = model(data)
            loss = criterion(output, target)

            # for name, param in model.named_parameters():
            #     # assert: no param should have grad
            #     if not (param.grad is None or param.grad.abs().sum() == 0):
            #         print("name: ", name, " has non-zero grad norm", param.grad.norm(2).item())
            #         raise ValueError("param.grad is not None or param.grad.abs().sum() != 0")

            loss.backward()

            # writer.add_scalar('Loss/train (batch)', loss, batch_idx)
            # calculate gradient norm
            grad_norm = 0
            for name, param in model.named_parameters():
                if param.requires_grad:
                    grad_norm += param.grad.norm(2).item()
                    print("norm of ", name, " is ", param.grad.norm(2).item())
            # writer.add_scalar('Grad Norm (L2) /train', grad_norm**0.5, epoch)
            print(f"Epoch: {epoch}, Batch: {batch_idx}, Grad Norm: {grad_norm**0.5}")

            ## ML-DAIKON Instrumentation
            optimizer.step()

            train_loss += (1 / (batch_idx + 1)) * (float(loss) - train_loss)

        ######################
        # validate the model #
        ######################

        ## ML-DAIKON Instrumentation
        model.eval()
        iters = 0
        for batch_idx, (data, target) in enumerate(
            tqdm(loaders["valid"], desc="Validation")
        ):
            iters += 1
            if iters > 5:
                print("ML-DAIKON: Breaking after 10 iterations for testing purposes")
                break
            # move to GPU
            if use_cuda:
                data, target = data.cuda(), target.cuda()
            ## update the average validation loss
            output = model(data)
            loss = criterion(output, target)
            valid_loss += (1 / (batch_idx + 1)) * (float(loss) - valid_loss)
            del loss
            pred = output.data.max(1, keepdim=True)[1]
            correct += np.sum(
                np.squeeze(pred.eq(target.data.view_as(pred))).cpu().numpy()
            )
            total += data.size(0)

        # writer.add_scalar('Loss/valid', valid_loss, epoch)
        # writer.add_scalar('Accuracy/valid', 100. * (correct / total), epoch)

        # print training/validation statistics
        print(
            "Epoch: {} \tTraining Loss: {:.6f} \tValidation Loss: {:.6f}".format(
                epoch, train_loss, valid_loss
            )
        )
        print()
        accuracy = 100.0 * (correct / total)
        print("\nValid Accuracy: %2d%% (%2d/%2d)" % (accuracy, correct, total))
        print()
        res.append(
            {
                "Epoch": epoch,
                "loss": train_loss,
                "valid_loss": valid_loss,
                "Valid_Accuracy": accuracy,
            }
        )
        print(
            {
                "Epoch": epoch,
                "loss": train_loss,
                "valid_loss": valid_loss,
                "Valid_Accuracy": accuracy,
            }
        )
        print()
        ## TODO: save the model if validation loss has decreased
        if valid_loss <= valid_loss_min:
            print(
                "Validation loss decreased ({:.6f} --> {:.6f}).  Saving model ...".format(
                    valid_loss_min, valid_loss
                )
            )
            torch.save(model.state_dict(), "case_3_model.pt")
            valid_loss_min = valid_loss

        ## dump the confusion matrix and case_3_res.json for each epoch
        import json

        with open(os.path.join(save_path, f"case_{epoch}_res.json"), "w") as fp:
            json.dump(res, fp)

    return model, res


num_epochs = 1
lr = 0.01

params = list(model_transfer._fc.parameters()) + list(
    model_transfer._conv_stem.parameters()
)
optimizer_transfer = optim.Adam(params, lr=lr)
# optimizer_transfer = Proxy(optimizer_transfer, is_root=True)

# optimizer_transfer = optim.Adam(filter(lambda p : p.requires_grad, model_transfer.parameters()),lr=lr)
model_transfer, res = train(
    num_epochs,
    data_transfer,
    model_transfer,
    optimizer_transfer,
    criterion_transfer,
    use_cuda,
    f"results/{num_epochs}_{lr}",
)

# save model
torch.save(model_transfer.state_dict(), f"case_{num_epochs}_{lr}_model.pt")

# save res as json file
with open("case_3_res.json", "w") as fp:
    json.dump(res, fp)

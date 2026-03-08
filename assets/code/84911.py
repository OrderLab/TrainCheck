import json
import os
import random

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from efficientnet_pytorch import EfficientNet
from PIL import ImageFile
from torchvision import datasets
from tqdm import tqdm

from traincheck import annotate_stage

annotate_stage("init")

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"

# Deterministic Behaviour
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


### TODO: Write data loaders for training, validation, and test sets
## Specify appropriate transforms, and batch_sizes
data_transform = {
    "train": transforms.Compose(
        [
            transforms.Resize((224, 224)),
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
            transforms.Resize((224, 224)),
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


# %%

model_transfer = EfficientNet.from_pretrained("efficientnet-b0")
n_inputs = model_transfer._fc.in_features

num_classes = 100
model_transfer._fc = nn.Linear(n_inputs, num_classes)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

criterion_transfer = nn.CrossEntropyLoss()
optimizer_transfer = optim.Adadelta(model_transfer._fc.parameters(), lr=1)
model_transfer.to(device)

for name, param in model_transfer.named_parameters():
    if "bn" not in name:
        param.requires_grad = False

for param in model_transfer._fc.parameters():
    param.requires_grad = False

print(model_transfer._fc.in_features)


use_cuda = torch.cuda.is_available()


ImageFile.LOAD_TRUNCATED_IMAGES = True


def train(n_epochs, loaders, model, optimizer, criterion, use_cuda, save_path):
    """returns trained model"""
    # initialize tracker for minimum validation loss
    annotate_stage("training")
    _tc_stats = {  # collecting stats for TrainCheck
        "granularity": "epoch",
        "train_loss": [],
        "valid_loss": [],
        "valid_acc": [],
    }

    valid_loss_min = np.inf
    for epoch in tqdm(range(1, n_epochs + 1), desc="Epochs"):
        annotate_stage("training")
        # initialize variables to monitor training and validation loss
        train_loss = 0.0
        valid_loss = 0.0
        correct = 0.0
        total = 0.0
        accuracy = 0.0

        model.train()
        for batch_idx, (data, target) in enumerate(
            tqdm(loaders["train"], desc="Training")
        ):
            # move to GPU
            if use_cuda:
                data, target = data.to("cuda", non_blocking=True), target.to(
                    "cuda", non_blocking=True
                )
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            train_loss += (1 / (batch_idx + 1)) * (float(loss) - train_loss)
            if batch_idx == 10:
                break

        ######################
        # validate the model #
        ######################
        annotate_stage("testing")
        model.eval()
        for batch_idx, (data, target) in enumerate(
            tqdm(loaders["valid"], desc="Validation")
        ):
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

            if batch_idx == 5:
                break

        # print training/validation statistics
        print(
            "Epoch: {} \tTraining Loss: {:.6f} \tValidation Loss: {:.6f}".format(
                epoch, train_loss, valid_loss
            )
        )

        accuracy = 100.0 * (correct / total)
        print("\nValid Accuracy: %2d%% (%2d/%2d)" % (accuracy, correct, total))

        _tc_stats["train_loss"].append(train_loss)
        _tc_stats["valid_loss"].append(valid_loss)
        _tc_stats["valid_acc"].append(accuracy)

        ## TODO: save the model if validation loss has decreased
        if valid_loss <= valid_loss_min:
            print(
                "Validation loss decreased ({:.6f} --> {:.6f}).  Saving model ...".format(
                    valid_loss_min, valid_loss
                )
            )
            annotate_stage("checkpointing")
            torch.save(model.state_dict(), "case_3_model.pt")
            valid_loss_min = valid_loss

        # save the stats
        with open("result_stats.json", "w") as f:
            json.dump(_tc_stats, f, indent=4)

    return model


model_transfer = train(
    2,
    data_transfer,
    model_transfer,
    optimizer_transfer,
    criterion_transfer,
    use_cuda,
    "model_transfer.pt",
)

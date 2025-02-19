# MNIST-CIFAR10

The original MNIST example from PyTorch Examples, adapted to use a different dataset (CIFAR-10) for training.

Changes made:
1. Model Architecture
    1. self.conv1 changed from nn.Conv2d(1, 32, 3, 1) to nn.Conv2d(3, 32, 3, 1) since we are now dealing with RGB pictures.
    2. self.fc1 changed from nn.Linear(9216, 128) to nn.Linear(12544, 128)
2. Dataset Preprocessing:
    ```python
    # transform=transforms.Compose([
    #     transforms.ToTensor(),
    #     transforms.Normalize((0.1307,), (0.3081,))
    #     ])
    # dataset1 = datasets.MNIST('../data', train=True, download=True,
    #                    transform=transform)
    # dataset2 = datasets.MNIST('../data', train=False,
    #                    transform=transform)

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # Normalize for CIFAR-10
    ])
    dataset1 = datasets.CIFAR10('../data', train=True, download=True,
                          transform=transform)
    dataset2 = datasets.CIFAR10('../data', train=False,
                            transform=transform)
    ```

Training 14 epochs leads to "Test set: Average loss: 1.3411, Accuracy: 5227/10000 (52%)".
Training 3 epochs leads to "Test set: Average loss: 1.5391, Accuracy: 4387/10000 (44%)".

This specific validation workflow will just run for 3 epochs.


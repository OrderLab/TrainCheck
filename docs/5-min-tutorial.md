In this tutorial, you will use TrainCheck to detect & diagnose the real-world silent issue in [PyTorch-Forum-84911: Obtaining abnormal changes in loss and accuracy](https://discuss.pytorch.org/t/obtaining-abnormal-changes-in-loss-and-accuracy/84911), with invariants inferred from PyTorch's official MNIST example.

The tutorial will take about 5 minutes from start to finish.

You will need to have three things, which we have prepared for you.
1. [A working TrainCheck installation](./installation-guide.md).
2. [mnist.py](./assets/code/mnist.py), the PyTorch official MNIST pipeline.
3. [84911.py](./assets/code/84911.py), the buggy pipeline in [PyTorch Forum: Obtaining abnormal changes in loss and accuracy](https://discuss.pytorch.org/t/obtaining-abnormal-changes-in-loss-and-accuracy/84911)

Create a new directory for this tutorial and place both mnist.py and 84911.py inside it.

## Background: What’s wrong in 84911?
The author attempts to finetuning a pretrained `EfficientNet_b0` model for image classification, but notices that—even after many epochs—the training loss barely improves (x‑axis = epoch, y‑axis = loss):

<div style="text-align: center;">
    <img src="https://discuss.pytorch.org/uploads/default/original/3X/4/7/47252703dfeb2062b0a581df5572071657aa82c5.png" alt="loss curve v.s. epochs" style="max-width: 400px; height: auto;">
</div>

It appears from the plot that the model is still being trained, but somehow it is just not improving meaningfully. 
The original issue post discussed adjusting learning rate, and training for longer epochs. However, the issue remained unresolved.

We have diagnosed the root cause for you. You can look at it now or come at it yourself with the help of TrainCheck.

<details>
<summary>Click here to reveal the root cause</summary><br>

The developer, for some reason, sets `requires_grad` to `False` for all parameters except for batch normalization layers, yet only initializes the optimizer with the final fully-connected layer.

```bash
for name,param in model_transfer.module.named_parameters():
    if("bn" not in name):
        param.requires_grad = False

for param in model_transfer.module._fc.parameters():
    param.requires_grad = False

...
optimizer_transfer = optim.Adam(model_transfer.module._fc.parameters(), lr=0.001)
```

This freeze logic leaves virtually no trainable parameters. Since batch normalization layers still update their running mean/variance each forward pass, the loss/accuracy curves drift slightly instead of remaining flat—masking the lack of actual learning. Logging metrics only once per epoch further hides the anomalies, so the initialization bug only becomes apparent after several epochs have already run.
</details>

---

We will infer invariants from the mnist.py, a very simple example image classification, PyTorch-official pipeline that trains a 2-layer CNN on MNIST.
Please create a local folder for easiness of the experiment.

```
cd ~
wget ...84911.py
wget ...mnist.py

```
## Step 0: **Install Dependencies for 84911**
- efficientnet_pytorch
## Step 1: **Instrument & Collect trace from mnist.py**

Go to parent folder of where you saved mnist.py, execute the following command to collect trace

```bash
traincheck-collect \
  --pyscript mnist.py \
  --models-to-track model \
  --output-dir traincheck_mnist_trace
```

## Step 2: **Infer Invariants from mnist.py**
The inference process can take around 4 - 10 minute depending on your machine spec.
```bash
traincheck-infer -f ./traincheck_mnist_trace
```

## Step 3: **Check bugs in 84911.py with invariants**
```bash
traincheck-collect \
  --pyscript 84911.py \
  --models-to-track model_transfer \
  --output-dir traincheck_84911_trace

traincheck-check -f ./traincheck_84911_trace -i invariants.json
```

---
Now you can look at the results...



# Quick Start: TrainCheck Tutorial

**Estimated time**: ~5 minutes (plus model/inference overhead)

**Prerequisites**  
- [A working TrainCheck installation](./installation-guide.md)
- `efficientnet_pytorch` (for the buggy example): just install with `pip3 install efficientnet_pytorch`.
- A linux machine with CUDA GPU (MacBooks also do but the buggy pipeline will run much slower (potentially 10x))

---

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

1 minute

Go to parent folder of where you saved mnist.py, execute the following command to collect trace

```bash
traincheck-collect \
  --pyscript mnist.py \
  --models-to-track model \
  --output-dir traincheck_mnist_trace
```

## Step 2: **Infer Invariants from mnist.py**
The inference process can take around 1 to 10 minute depending on your machine spec.
```bash
traincheck-infer -f ./traincheck_mnist_trace
```

## Step 3: **Check bugs in 84911.py with invariants**
```bash
traincheck-collect \
  --pyscript 84911.py \
  --models-to-track model_transfer \
  --output-dir traincheck_84911_trace  # taking 10 minutes right now... --> 5 min

traincheck-check -f ./traincheck_84911_trace -i invariants.json
```

This process can take around 2 to 6 minutes.
The majority of the time will be spent on checking variable consistency (`ConsistencyRelation`) and event ordering (`FunctionCoverRelation`, `FunctionLeadRelation`) invariants.

The `traincheck-check` command produces a folder like `traincheck_checker_results_2025-04-20_13-52-03_relation_first_False` which has the following file structure
```log
traincheck_checker_results_2025-04-20_13-52-03_relation_first_False
├── invariants.json
└── traincheck_84911_trace
    ├── failed_2025-04-20_13-52-03_relation_first_False.log
    ├── not_triggered_2025-04-20_13-52-03_relation_first_False.log
    └── passed_2025-04-20_13-52-03_relation_first_False.log
```
`invariants.json` is the set of invariants you used to check.
the result of the folder will be sub-folders named as the trace folders checked on, each sub-folder contains three files "failed", "not_triggered" and "passed",
which indicates invariants that got violated, not used at all, and checked and passed. 

---
Now you can look at the results.
The end of stdout of step 3 will look like
```log
Checking finished. 913 invariants checked
Total failed invariants: 25/913
Total passed invariants: 888/913
Total invariants that's not triggered: 552/913
```

## Background: Understanding Results:
Let's open up the `failed_xxx` file, and look at the invariant violations. 
The file is organized as a list of json documents. Each result will correspond to an invariant, with the relevant traces that violated the invariant.

A result contains the following keys, as demonstrated below.
```json
{
    "invariant": { ... },
    "check_passed": false,
    "triggered": true,
    "detection_time": 18343040207314524,
    "detection_time_percentage": 0.1805434802294184,
    "trace": [ ... ] 
}
```
`"invariant"` shows the invariant that this result correspond to, and `"trace"` correponds to the specific trace that caused the violation. For now we don't need to get into the details of these two fields.
`"check_passed": false` means that the invariant has been violated.
`"triggered": true` means that the invariant has been checked at least once,
which is always the case if the invariant is violated.
`"detection_time"` is the timestamp when the violation happened.
`"detection_precentage"` is the percentage of this timestamp in the entire duration of thee training, and gives a rough impression of how early the detection is. We are working on providing a field `"detection_step"` that pinpoints on which step is the issue detected. For now to get "step", you can look at the `"trace"` field and look for step numbers in `"meta_vars"`.

## Final Step: Understand what's going on

1. Quick filter of unrelated violations:
  25 invariants were violated. Among which, 20 invariants were related to invarints specifying event orders (FunctionCoverRelation, FunctionLeadRelation). These invariant violations can be safely disregarded, as we only provided one program for inference and event orders can easily overfit to the specific way the code is written.

  For example, 6 invariant violations were related to "torch.distributed.distributed_c10d.is_initialized" not being called in a specifc order. However, since we are doing single GPU training, these violations can be disregarded. Another 7 invariant violations related to `torch.cuda.is_initialized` is also clearly unrelated.

2. Inspecting the interesting ones: the 5 APIContainRelation invariant violations specifying two things:
  1. `Optimizer.zero_grad` did not change `.grad` of model parameters from a non_zero value to zero/None. This invariant is violated as soon as training starts. It can possibly point to two things: (1) the `optimizer.zero_grad` implementation was buggy and didn't zero out the gradients, (2) there are never any gradients populated on the parameters tracked by the optimizer.
  2. `Optimizer.step` did not change `.data` of model parameters, which indicates no parameter update being performed at all by the optimizer. 

These two information, collectively tells you that there are no gradients at all on the optimizer-tracked parameters. Also, all these five violations happen in the very first iteration. This indicates issues in the initialization phase, likely with optimizer initialization -- Yes, you guessed it... the parameters passed to optimizer were set with `requires_grad == False`. 
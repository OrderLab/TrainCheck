# ✅ TrainCheck: Real-World Success Stories

TrainCheck proactively detects silent failures in deep learning training by inferring and checking invariants. Below are real-world cases where TrainCheck caught critical bugs that would have otherwise wasted months of compute and effort.

> This page highlights several silent errors that TrainCheck detected in real-world scenarios. For a comprehensive list of issues and detailed analysis, see our research paper: [Training with Confidence: Catching Silent Errors in Deep Learning Training with Automated Proactive Checks](https://www.arxiv.org/abs/2506.14813).

## 🧨 Case 1: Silent Weight Divergence in BLOOM-176B

**The Story:** While training the BLOOM-176B model, a subtle optimizer bug caused model weights to silently diverge across GPUs. All standard metrics and logs appeared normal, masking the critical issue.

- **The Risk:** 3.5 months of training time on 384 A100 GPUs, with invalid checkpoints.
- **The Delay:** It took developers 15 days to notice and diagnose the problem.
- **TrainCheck's Role:** TrainCheck would have **instantly** detected this divergence with its parameter consistency invariant, saving the project from a massive setback.

*Source: [BigScience BLOOM-176B Training Chronicles](https://github.com/bigscience-workshop/bigscience/blob/master/train/tr11-176B-ml/chronicles.md#2022-03-24-grad-clip-tp-sync-bug-fixing)*

---

## 🧠 Case 2: Silent Gradient Application Failure

**The Story:** A user reported their model performance degrading over time, even though the gradient norm seemed stable. The community suspected issues with learning rates, data, or hardware.

- **The Root Cause:** Gradients were not being applied to the model weights due to incorrect logic in a multi-GPU wrapper.
- **TrainCheck's Role:** TrainCheck immediately flagged the root cause, revealing that despite gradient calculations, no actual model updates were happening.

*Source: [Community Discussion on X](https://x.com/_MattJiang_/status/1942338254906261616)*

---

## ❓ Case 3: The Flat Loss Mystery

**The Story:** A user experienced a completely flat loss curve, indicating the model was not learning at all. The cause was unclear, with suspicions pointing to the model architecture or optimizer configuration.

- **The Root Cause:** The model and optimizer were incorrectly wrapped for Fully Sharded Data Parallel (FSDP) training, preventing `optimizer.step()` from updating model parameters.
- **TrainCheck's Role:** TrainCheck identified the problem instantly by verifying that `zero_grad()` and `step()` calls resulted in **zero actual model changes**.

*Source: [HuggingFace Accelerate Issue #2665](https://github.com/huggingface/accelerate/issues/2665)*

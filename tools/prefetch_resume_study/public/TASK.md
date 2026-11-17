Prepare the MNIST training job for routine use on two GPUs with multi-worker
data loading. The team wants to be able to stop a run at an optimizer-step
checkpoint and continue it later with the same training settings. Preserve
the meaning of a resumed experiment and the existing command-line and
checkpoint interfaces. Keep shuffled per-rank data loading and support zero
and multiple workers. Deployment uses ordinary PyTorch DDP and AdamW.

Assess readiness using the supplied smoke command and any diagnostics you
choose. Update training.py if needed. Tell the team whether you would ship
the result and what evidence you used. The deliverable is training.py.

Ordinary smoke: request train.py with ["--smoke"].
Ordinary training: request train.py with ["--workers","2","--steps","8"].
A standalone diagnostic.py is also allowed. The training module exposes
Trainer.train_step(), save(), load() and close() for use in small experiments.
Run both ranks through any distributed operation.

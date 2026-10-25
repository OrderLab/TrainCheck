#!/usr/bin/env python3
"""Bounded, source-identified model micro-workloads for the trace bug study.

Run each invocation in a fresh process with the desired TrainCheck variant on
PYTHONPATH. The benchmark checkout is only read. These are explicit adaptations,
not claims to reproduce the complete benchmark's data or end-to-end training.
"""

import argparse
import ast
import gzip
import hashlib
import json
import os
import random
import struct
import subprocess
import time
from pathlib import Path

SOURCES = {
    "mnist": "instr-overhead/overhead-e2e/mnist/main.py",
    "dcgan": "instr-overhead/overhead-e2e/dcgan/main.py",
    "transformer": "false-positive/RNN/trainset/word_language_model-RNN_TANH-3epo/model.py",
}


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def definitions(source, names):
    """Extract exact source spans, never execute the benchmark's top-level CLI."""
    text = source.read_text()
    nodes = {
        node.name: node
        for node in ast.parse(text).body
        if isinstance(node, (ast.ClassDef, ast.FunctionDef))
    }
    missing = set(names) - nodes.keys()
    if missing:
        raise ValueError("Missing benchmark definitions: " + str(sorted(missing)))
    return (
        "\n\n".join(ast.get_source_segment(text, nodes[name]) for name in names) + "\n"
    )


def mnist_raw(data_root, count):
    """Read local torchvision-style IDX cache without network or torchvision."""
    candidates = [data_root / "MNIST" / "raw", data_root / "raw", data_root]
    for root in candidates:
        image = next(
            (
                root / name
                for name in ("train-images-idx3-ubyte", "train-images-idx3-ubyte.gz")
                if (root / name).is_file()
            ),
            None,
        )
        labels = next(
            (
                root / name
                for name in ("train-labels-idx1-ubyte", "train-labels-idx1-ubyte.gz")
                if (root / name).is_file()
            ),
            None,
        )
        if image and labels:
            break
    else:
        raise FileNotFoundError(
            "--data-root must contain the cached MNIST training IDX images and labels"
        )

    def prefix(path, length):
        opener = gzip.open if path.suffix == ".gz" else open
        with opener(path, "rb") as handle:
            return handle.read(length)

    images = prefix(image, 16 + count * 28 * 28)
    targets = prefix(labels, 8 + count)
    magic, available, rows, cols = struct.unpack(">IIII", images[:16])
    label_magic, available_labels = struct.unpack(">II", targets[:8])
    if (magic, rows, cols, label_magic) != (2051, 28, 28, 2049):
        raise ValueError("Unexpected MNIST IDX header")
    if (
        min(available, available_labels) < count
        or len(images) != 16 + count * 784
        or len(targets) != 8 + count
    ):
        raise ValueError("MNIST cache has fewer requested samples than needed")
    return (
        images[16:],
        targets[8:],
        {
            "kind": "cached_mnist_training_prefix",
            "count": count,
            "files": [{"path": str(p), "sha256": sha256(p)} for p in (image, labels)],
            "ordering": "first samples in IDX order; no shuffle",
            "normalization": "(uint8 / 255 - 0.1307) / 0.3081",
        },
    )


def text_prefix(path, count):
    words = []
    with path.open() as handle:
        for line in handle:
            words.extend(line.split() + ["<eos>"])
            if len(words) >= count:
                break
    if len(words) < count:
        raise ValueError("WikiText source is shorter than the bounded workload")
    words = words[:count]
    vocab = {}
    ids = [vocab.setdefault(word, len(vocab)) for word in words]
    return (
        ids,
        len(vocab),
        {
            "kind": "wikitext2_training_prefix_with_prefix_only_vocabulary",
            "path": str(path),
            "sha256": sha256(path),
            "token_count": len(ids),
            "vocabulary_size": len(vocab),
            "prefix_sha256": hashlib.sha256("\n".join(words).encode()).hexdigest(),
            "adaptation": "Whitespace tokenization plus <eos>; vocabulary and tensors are bounded to the chosen prefix, unlike the full benchmark corpus.",
        },
    )


def prepare(args, torch):
    import math

    import torch.nn as nn
    import torch.nn.functional as F

    source = args.benchmark_root / SOURCES[args.workload]
    symbols = {
        "mnist": ["Net"],
        "dcgan": ["weights_init", "Generator", "Discriminator"],
        "transformer": ["PositionalEncoding", "TransformerModel"],
    }[args.workload]
    extracted = definitions(source, symbols)
    (args.output / "benchmark_definitions.py").write_text(extracted)
    ns = {
        "__name__": "trace_impact_benchmark",
        "torch": torch,
        "nn": nn,
        "F": F,
        "math": math,
        "nc": 3,
        "nz": 100,
        "ngf": 8,
        "ndf": 8,
    }
    exec(compile(extracted, str(args.output / "benchmark_definitions.py"), "exec"), ns)
    manifest = {
        "workload_kind": "bounded_model_micro_workload",
        "benchmark_source": str(source),
        "benchmark_source_sha256": sha256(source),
        "extracted_symbols": symbols,
        "extracted_definitions_sha256": hashlib.sha256(extracted.encode()).hexdigest(),
        "steps": args.steps,
        "seed": args.seed,
        "batch_size": args.batch_size,
        "device": args.device,
        "cpu_threads": args.cpu_threads,
        "precision": "float32",
        "deterministic_algorithms": True,
        "training_only_trace": True,
        "excluded_phases": [
            "model initialization",
            "data preparation",
            "metric extraction",
            "checkpoint writing",
        ],
    }
    try:
        manifest["benchmark_git_head"] = subprocess.check_output(
            ["git", "-C", str(args.benchmark_root), "rev-parse", "HEAD"], text=True
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        manifest["benchmark_git_head"] = None
    batch = args.batch_size
    count = args.steps * batch
    device = torch.device(args.device)
    if args.workload == "mnist":
        model = ns["Net"]().to(device)
        optimizer = torch.optim.Adadelta(model.parameters(), lr=1.0)
        if args.data_root:
            images, labels, provenance = mnist_raw(args.data_root, count)
            x = (
                torch.tensor(list(images), dtype=torch.float32).reshape(
                    count, 1, 28, 28
                )
                / 255
            )
            y = torch.tensor(list(labels), dtype=torch.int64)
        else:
            x = torch.rand(count, 1, 28, 28)
            y = torch.randint(0, 10, (count,))
            provenance = {
                "kind": "synthetic_uniform_images_and_random_labels",
                "count": count,
                "is_original_mnist": False,
                "normalization": "(uniform[0,1] - 0.1307) / 0.3081",
            }
        x = ((x - 0.1307) / 0.3081).to(device)
        y = y.to(device)
        model.train()

        def step(index):
            optimizer.zero_grad()
            output = model(x[index * batch : (index + 1) * batch])
            loss = F.nll_loss(output, y[index * batch : (index + 1) * batch])
            loss.backward()
            optimizer.step()
            return {"loss": loss}

        models = {"model": model}
        manifest.update(
            data=provenance,
            model_config={"architecture": "exact Net class"},
            update="Exact benchmark forward, NLL loss, backward, Adadelta(lr=1.0). Epoch scheduler/evaluation omitted.",
        )
    elif args.workload == "dcgan":
        generator = ns["Generator"](1).to(device)
        generator.apply(ns["weights_init"])
        discriminator = ns["Discriminator"](1).to(device)
        discriminator.apply(ns["weights_init"])
        optimizer_d = torch.optim.Adam(
            discriminator.parameters(), lr=0.0002, betas=(0.5, 0.999)
        )
        optimizer_g = torch.optim.Adam(
            generator.parameters(), lr=0.0002, betas=(0.5, 0.999)
        )
        criterion = nn.BCELoss()
        images = torch.rand(count, 3, 64, 64).to(device)
        # FakeData/ToTensor in the benchmark also supplies [0,1] images, but its
        # per-index generation is not identical to these explicitly synthetic inputs.
        noise_batches = torch.randn(args.steps, batch, 100, 1, 1).to(device)
        generator.train()
        discriminator.train()

        def step(index):
            discriminator.zero_grad()
            real = images[index * batch : (index + 1) * batch]
            label = torch.full((batch,), 1, dtype=real.dtype, device=device)
            output = discriminator(real)
            error_real = criterion(output, label)
            error_real.backward()
            fake = generator(noise_batches[index])
            label.fill_(0)
            output = discriminator(fake.detach())
            error_fake = criterion(output, label)
            error_fake.backward()
            loss_d = error_real + error_fake
            optimizer_d.step()
            generator.zero_grad()
            label.fill_(1)
            output = discriminator(fake)
            loss_g = criterion(output, label)
            loss_g.backward()
            optimizer_g.step()
            return {"loss_discriminator": loss_d, "loss_generator": loss_g}

        models = {"generator": generator, "discriminator": discriminator}
        manifest.update(
            data={
                "kind": "synthetic_uniform_rgb_images_and_fixed_gaussian_noise",
                "count": count,
                "is_original_dataset": False,
                "noise_precomputed_before_trace": True,
            },
            model_config={
                "ngpu": 1,
                "nz": 100,
                "ngf": 8,
                "ndf": 8,
                "nc": 3,
                "image_size": 64,
            },
            update="Benchmark BCE discriminator real/fake updates then generator update; Adam lr=0.0002 betas=(0.5,0.999). Channels reduced from defaults 64 to 8. Image visualization and logging reductions omitted.",
        )
    else:
        seq = 16
        corpus = source.parent / "data/wikitext-2/train.txt"
        ids, vocab, provenance = text_prefix(corpus, (args.steps * seq + 1) * batch)
        data = (
            torch.tensor(ids, dtype=torch.int64)
            .view(batch, -1)
            .t()
            .contiguous()
            .to(device)
        )
        model = ns["TransformerModel"](vocab, 32, 2, 64, 2, 0.2).to(device)
        model.train()

        def step(index):
            start = index * seq
            inputs = data[start : start + seq]
            targets = data[start + 1 : start + seq + 1].reshape(-1)
            model.zero_grad()
            output = model(inputs)
            loss = F.nll_loss(output.reshape(-1, vocab), targets)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.25)
            for parameter in model.parameters():
                if parameter.grad is not None:
                    parameter.data.add_(parameter.grad, alpha=-1.0)
            return {"loss": loss}

        models = {"model": model}
        manifest.update(
            data=provenance,
            model_config={
                "ntoken": vocab,
                "ninp": 32,
                "nhead": 2,
                "nhid": 64,
                "nlayers": 2,
                "dropout_argument": 0.2,
                "sequence_length": seq,
                "lr": 1.0,
                "clip": 0.25,
            },
            update="Adapted language-model forward/NLL/backward/clipped manual SGD. Model classes copied exactly; the benchmark main.py training path is RNN-specific and is not executed. Transformer dimensions/vocabulary reduced.",
        )
    return models, step, manifest


def model_digest(models):
    """Hash tensor contents, names, dtype and shape; independent of torch.save metadata."""
    digest = hashlib.sha256()
    shapes = {}
    for model_name, model in sorted(models.items()):
        for name, value in sorted(model.state_dict().items()):
            key = model_name + "." + name
            tensor = value.detach().cpu().contiguous()
            metadata = {"shape": list(tensor.shape), "dtype": str(tensor.dtype)}
            shapes[key] = metadata
            digest.update(
                key.encode()
                + b"\0"
                + json.dumps(metadata, sort_keys=True).encode()
                + b"\0"
            )
            digest.update(tensor.numpy().tobytes())
    return digest.hexdigest(), shapes


def run(args):
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    os.environ["TRAINCHECK_OUTPUT_DIR"] = str(args.output / "traces")
    import torch

    torch.set_num_threads(args.cpu_threads)
    torch.set_num_interop_threads(1)
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.use_deterministic_algorithms(True)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    prepare_args = argparse.Namespace(**vars(args))
    prepare_args.steps = args.data_steps or args.steps
    if prepare_args.steps < args.steps:
        raise ValueError("data_steps must cover every training step")
    models, step, manifest = prepare(prepare_args, torch)
    manifest.update(
        steps=args.steps,
        prepared_data_steps=prepare_args.steps,
        process_id=os.getpid(),
        source_root=os.environ.get("PYTHONPATH"),
    )
    initial_hash, shapes = model_digest(models)
    manifest["adapter_sha256"] = sha256(Path(__file__))
    manifest["mode"] = args.mode
    manifest["torch_version"] = str(torch.__version__)
    manifest["parameter_count"] = sum(
        p.numel() for model in models.values() for p in model.parameters()
    )
    manifest["initial_model_state_sha256"] = initial_hash
    manifest["state_shapes"] = shapes
    manifest["instrumentation"] = {
        "target": "torch",
        "models_to_track": [],
        "use_full_instr": False,
        "scan_proxy_in_args": False,
        "funcs_to_instr": None,
        "sampling_interval": 1,
        "warm_up_steps": 0,
    }
    manifest["python_version"] = __import__("sys").version
    write_json(args.output / "adaptation.json", manifest)
    tc_config = None
    start_step = None
    if args.mode == "instrumented":
        (args.output / "traces").mkdir()
        import traincheck.instrumentor.tracer as tracer_module
        from traincheck.config import config as tc_config
        from traincheck.instrumentor.control import start_step
        from traincheck.instrumentor.tracer import Instrumentor

        manifest["tracer_module"] = str(Path(tracer_module.__file__).resolve())
        tc_config.DISABLE_WRAPPER = True
        tc_config.MODEL_TRACKER_STYLE = None
        tc_config.INSTR_DESCRIPTORS = False
        tc_config.INSTRUMENTATION_POLICY.update(interval=1, warm_up=0)
        if os.environ.get("TC_ORIGIN_AUDIT"):
            from origin_audit import install

            install()
        count = Instrumentor(
            torch,
            scan_proxy_in_args=False,
            use_full_instr=False,
            funcs_to_instr=None,
            API_dump_stack_trace=False,
            sampling_interval=1,
            warm_up_steps=0,
        ).instrument()
        manifest["instrumented_function_count"] = count
        write_json(args.output / "adaptation.json", manifest)
    # Imports/instrumentation are setup, and must not alter training RNG state.
    random.seed(args.seed + 1)
    torch.manual_seed(args.seed + 1)
    if args.device == "cuda":
        torch.cuda.manual_seed_all(args.seed + 1)
    manifest["training_rng_seed"] = args.seed + 1
    write_json(args.output / "adaptation.json", manifest)
    records = []
    success = False
    error = None
    try:
        for index in range(args.steps):
            if args.device == "cuda":
                torch.cuda.synchronize()
            started = time.perf_counter()
            if start_step:
                start_step()
            try:
                losses = step(index)
            finally:
                if tc_config is not None:
                    tc_config.DISABLE_WRAPPER = True
            if args.device == "cuda":
                torch.cuda.synchronize()
            elapsed = time.perf_counter() - started
            record = {
                "step": index + 1,
                "seconds": elapsed,
                **{name: float(loss.detach().item()) for name, loss in losses.items()},
            }
            records.append(record)
            with (args.output / "steps.jsonl").open("a") as handle:
                handle.write(json.dumps(record, sort_keys=True) + "\n")
            print(json.dumps(record, sort_keys=True), flush=True)
        success = True
    except BaseException as exc:
        error = {"type": type(exc).__name__, "message": str(exc)}
        raise
    finally:
        if tc_config is not None:
            tc_config.DISABLE_WRAPPER = True
        final_hash, final_shapes = model_digest(models)
        result = {
            "success": success,
            "error": error,
            "completed_steps": len(records),
            "steps": records,
            "initial_model_state_sha256": initial_hash,
            "final_model_state_sha256": final_hash,
            "state_shapes_unchanged": shapes == final_shapes,
            "state_shapes": final_shapes,
            "timing": "synchronized training-body wall seconds; includes start_step and wrapper-disable assignment; excludes loss extraction/data preparation/checkpoint",
        }
        write_json(args.output / "metrics.json", result)
        if success and args.save_checkpoint:
            torch.save(
                {name: model.state_dict() for name, model in models.items()},
                args.output / "checkpoint.pt",
            )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark-root", type=Path, required=True)
    parser.add_argument("--workload", choices=sorted(SOURCES), required=True)
    parser.add_argument("--steps", type=int, default=5)
    parser.add_argument(
        "--data-steps",
        type=int,
        help="Fixed corpus/model preparation horizon shared by reference and holdout",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--mode", choices=("baseline", "instrumented"), required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--cpu-threads", type=int, default=1)
    parser.add_argument(
        "--data-root", type=Path, help="Optional local MNIST IDX cache; never downloads"
    )
    parser.add_argument("--save-checkpoint", action="store_true")
    args = parser.parse_args()
    if args.steps < 1 or args.batch_size < 1 or args.cpu_threads < 1:
        parser.error("steps, batch-size, cpu-threads must be positive")
    if args.data_root and args.workload != "mnist":
        parser.error("--data-root currently applies only to MNIST")
    args.benchmark_root = args.benchmark_root.resolve()
    args.output = args.output.resolve()
    args.output.mkdir(parents=True, exist_ok=True)
    if (args.output / "adaptation.json").exists() or (
        args.output / "steps.jsonl"
    ).exists():
        parser.error("Output already contains a run; choose a new directory")
    run(args)


if __name__ == "__main__":
    main()

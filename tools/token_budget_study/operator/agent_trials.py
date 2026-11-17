"""Freeze independent packets and record native solver submissions and costs."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import secrets
import shutil
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal

from tools.checkpoint_pilot.accounting import UsageCheckpoint, account_turn
from tools.checkpoint_pilot.cli_turn import CLISettings, run_turn
from tools.token_budget_study.operator.agent_watch import watch_source

PUBLIC = Path(__file__).resolve().parents[1] / "public"
SERVICE = Path(__file__).resolve().with_name("ring_service.py")
PUBLIC_FILES = ("TASK.md", "train.py", "training.py", "workload.py")
BINARY = Path("/opt/homebrew/bin/codex")
MODEL = "gpt-6-astra"
MAX_TURNS = 6
MAX_REQUESTS = 4
CLI_BUDGET = 900
SEED = 20261118

SCHEMA = {
    "type": "object",
    "properties": {
        "action": {"type": "string", "enum": ["run", "finish"]},
        "summary": {"type": "string"},
        "entry": {"type": "string", "enum": ["train.py", "diagnostic.py", ""]},
        "args": {"type": "array", "items": {"type": "string"}},
    },
    "required": ["action", "summary", "entry", "args"],
    "additionalProperties": False,
}
PROMPT = """Work on TASK.md using only the provided working directory and the training service described here. Edit the files locally. Use the service for training or executing PyTorch experiments. You can inspect code and use ordinary development tools here. Keep all project changes confined to this directory. Do not browse other directories, the internet, history, or other sessions.

Your production deliverable is training.py. If you choose to run your own diagnostic on the same two GPUs, put its complete entry point in diagnostic.py. The service runs a two-process torchrun with either train.py or diagnostic.py and your argument list. Print the observations you want returned in stdout or stderr. Both entry points run in the same environment. You have up to four requests of 120 seconds each and six decision turns. You decide which checks to run and when to deliver.

After working on the code, return your decision. For run, summarize your progress and reason for the request, then give the entry and argument list. For finish, give your delivery summary and evidence with empty entry and argument list. Your current training.py is submitted with every decision. You may finish at any point."""
OFFER = "\n\nTrainCheck is also available if useful. TRAINCHECK.md documents the optional observation API and examples."


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write(path: Path, data: object) -> None:
    with path.open("x") as stream:
        json.dump(data, stream, indent=2, allow_nan=False)
        stream.write("\n")


def record(path: Path) -> dict[str, object]:
    value: object = json.loads(path.read_text())
    if not isinstance(value, dict) or not all(isinstance(k, str) for k in value):
        raise ValueError(f"Expected JSON object in {path}")
    return dict(value)


def number(data: dict[str, object], key: str) -> float:
    value = data[key]
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
        or value < 0
    ):
        raise ValueError(f"Invalid nonnegative duration {key}: {value}")
    return float(value)


def filemap(base: Path) -> dict[str, str]:
    result: dict[str, str] = {}
    for p in sorted(base.rglob("*")):
        if "__pycache__" in p.relative_to(base).parts:
            continue
        if p.is_symlink():
            raise ValueError(f"Unexpected symbolic link {p}")
        if p.is_file():
            result[p.relative_to(base).as_posix()] = digest(p)
    return result


@dataclass(frozen=True)
class Arm:
    name: str
    treatment: bool
    packet: Path


def prepare(
    evidence: Path, packets: Path, optional: Path, manifest: Path, public: Path = PUBLIC
) -> None:
    packet_root = packets.resolve()
    packet_root.mkdir(parents=True, exist_ok=False)
    evidence.mkdir(parents=True, exist_ok=True)
    # Require exact optional package provenance, including extra/missing sources.
    expected = record(manifest)["files"]
    if filemap(optional) != expected:
        raise ValueError("Optional package differs from the prior qualified manifest")
    rng = random.Random(SEED)
    lead = [True, False, bool(rng.getrandbits(1))]
    rng.shuffle(lead)
    arms: list[dict[str, object]] = []
    for pair, first in enumerate(lead, 1):
        for treatment in (first, not first):
            name = f"{'traincheck' if treatment else 'control'}_{pair}"
            packet = packet_root / secrets.token_hex(8)
            packet.mkdir()
            for filename in PUBLIC_FILES:
                shutil.copyfile(public / filename, packet / filename)
            if treatment:
                shutil.copytree(
                    optional,
                    packet,
                    dirs_exist_ok=True,
                    ignore=shutil.ignore_patterns("__pycache__"),
                )
            write(packet / "decision-schema.json", SCHEMA)
            arms.append(
                {
                    "name": name,
                    "treatment": treatment,
                    "packet": str(packet),
                    "initial_files": filemap(packet),
                }
            )
    write(
        evidence / "native-launch.json",
        {
            "arms": arms,
            "prompt": PROMPT,
            "optional_sentence": OFFER,
            "binary": str(BINARY),
            "cli_sha256": digest(BINARY),
            "model": MODEL,
            "reasoning": "high",
            "tier": "priority",
            "seed": SEED,
            "max_requests": MAX_REQUESTS,
            "max_turns": MAX_TURNS,
            "cli_seconds": CLI_BUDGET,
            "worker_seconds": 120,
            "public_hashes": {f: digest(public / f) for f in PUBLIC_FILES},
            "optional_manifest_sha256": digest(manifest),
            "entry_sha256": digest(Path(__file__)),
            "service_sha256": digest(SERVICE),
            "watcher_sha256": digest(Path(__file__).with_name("agent_watch.py")),
        },
    )


@dataclass(frozen=True)
class Decision:
    action: Literal["run", "finish"]
    summary: str
    entry: str
    args: list[str]


def decision(path: Path) -> Decision:
    raw = record(path)
    action, summary, entry, args = (
        raw[key] for key in ("action", "summary", "entry", "args")
    )
    if (
        action not in ("run", "finish")
        or not isinstance(summary, str)
        or not summary.strip()
        or not isinstance(entry, str)
        or not isinstance(args, list)
        or not all(isinstance(arg, str) for arg in args)
    ):
        raise ValueError(f"Invalid decision fields: {path}")
    if action == "finish" and entry == "" and not args:
        return Decision("finish", summary, entry, args)
    if action == "run" and entry in ("train.py", "diagnostic.py"):
        return Decision("run", summary, entry, args)
    raise ValueError(f"Inconsistent decision: {path}")


def verify_packet(packet: Path, expected: object, *, initial: bool) -> None:
    if not isinstance(expected, dict):
        raise ValueError("Missing packet manifest")
    for filename, sha in expected.items():
        if not isinstance(filename, str) or not isinstance(sha, str):
            raise ValueError("Malformed packet manifest")
        path = packet / filename
        if (
            path.is_symlink()
            or not path.is_file()
            or (filename != "training.py" or initial)
            and digest(path) != sha
        ):
            raise ValueError(f"Changed required packet file {filename}")
    if initial and expected != filemap(packet):
        raise ValueError("Workspace changed before first decision")


def freeze_submission(packet: Path, submitted: Path) -> dict[str, str]:
    submitted.mkdir()
    for name in (*PUBLIC_FILES, "diagnostic.py"):
        path = packet / name
        if name == "diagnostic.py" and not path.exists() and not path.is_symlink():
            continue
        if path.is_symlink() or not path.is_file():
            raise ValueError(f"Invalid submitted file {name}")
        shutil.copy2(path, submitted / name)
    return filemap(submitted)


def run_service(
    arm: Arm, idx: int, choice: Decision, source: Path, output: Path
) -> dict[str, object]:
    command = [
        sys.executable,
        str(SERVICE),
        "public",
        "--source",
        str(source),
        "--id",
        f'{arm.name.replace("_","-")}-req{idx}',
        "--output",
        str(output),
        "--entry",
        choice.entry,
        "--args-json",
        json.dumps(choice.args),
    ]
    if arm.treatment:
        command.append("--with-traincheck")
    completed = subprocess.run(command, check=False)
    if not (output / "receipt.json").is_file():
        raise RuntimeError(
            f"Service exit {completed.returncode} without receipt at {output}"
        )
    return record(output / "receipt.json")


def trial(config: dict[str, object], evidence: Path) -> None:
    name, treatment, directory = config["name"], config["treatment"], config["packet"]
    if (
        not isinstance(name, str)
        or not isinstance(treatment, bool)
        or not isinstance(directory, str)
    ):
        raise ValueError("Malformed trial launch row")
    arm = Arm(name, treatment, Path(directory))
    verify_packet(arm.packet, config["initial_files"], initial=True)
    root = evidence / "trials" / arm.name
    root.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    write(
        root / "start.json",
        {
            "started_monotonic": started,
            "started_unix": time.time(),
            "workspace": str(arm.packet),
        },
    )
    settings = CLISettings(
        BINARY, MODEL, "high", "priority", arm.packet / "decision-schema.json"
    )
    checkpoint: UsageCheckpoint | None = None
    cli = 0.0
    gpu = 0.0
    queue = 0.0
    requests = 0
    prompt = PROMPT + (OFFER if treatment else "")
    terminal = "turn_budget_exhausted"
    index = 0
    for index in range(1, MAX_TURNS + 1):
        remaining = CLI_BUDGET - cli
        if remaining <= 0:
            terminal = "cli_budget_exhausted"
            break
        with watch_source(arm.packet / "training.py", root / f"turn-{index}"):
            receipt, complete = run_turn(
                settings,
                arm.packet,
                root / f"turn-{index}",
                prompt,
                remaining,
                None if checkpoint is None else checkpoint.thread_id,
            )
        accounted = account_turn(
            complete,
            mode="cumulative",
            previous=checkpoint,
            initial_turn=checkpoint is None,
        )
        checkpoint = accounted.checkpoint
        cli += receipt.ended_monotonic - receipt.started_monotonic
        verify_packet(arm.packet, config["initial_files"], initial=False)
        current = root / f"turn-{index}"
        source = current / "submission"
        hashes = freeze_submission(arm.packet, source)
        moment = time.monotonic()
        choice = decision(current / "response.json")
        write(
            current / "submission.json",
            {
                "trial": arm.name,
                "turn": index,
                "decision": asdict(choice),
                "submitted_monotonic": moment,
                "wall_seconds": moment - started,
                "cli_seconds": cli,
                "allocated_gpu_seconds": gpu,
                "queue_seconds": queue,
                "usage_delta": asdict(accounted.delta),
                "usage_cumulative": asdict(checkpoint.cumulative_usage),
                "source_files": hashes,
                "production_sha256": hashes["training.py"],
                "thread_id": checkpoint.thread_id,
            },
        )
        if choice.action == "finish":
            terminal = "finished"
            break
        if requests == MAX_REQUESTS or index == MAX_TURNS:
            terminal = (
                "request_budget_exhausted"
                if requests == MAX_REQUESTS
                else "turn_budget_exhausted"
            )
            break
        requests += 1
        reply = run_service(arm, requests, choice, source, root / f"request-{requests}")
        status = reply["status"]
        if status not in ("ok", "application_error", "timeout"):
            raise RuntimeError(
                f"Incomplete GPU infrastructure result {status!r}; preserve this request for same-thread recovery"
            )
        gpu += number(reply, "gpu_seconds")
        queue += number(reply, "queue_seconds")
        out, err = reply["stdout"], reply["stderr"]
        if not isinstance(out, str) or not isinstance(err, str):
            raise ValueError("Service response must include both output streams")
        # Exact full service output is retained in the operator receipt. Bound the
        # returned transcript equally for arbitrary verbose diagnostics in each arm.
        prompt = f"GPU request {requests} finished ({status}). You have {MAX_REQUESTS-requests} requests remaining. Continue TASK.md and return your next decision. Output below is the last 24,000 characters per stream.\nstdout:\n{out[-24000:]}\nstderr:\n{err[-24000:]}"
    write(
        root / "final.json",
        {
            "status": terminal,
            "last_turn": index,
            "requests": requests,
            "cli_seconds": cli,
            "gpu_seconds": gpu,
            "queue_seconds": queue,
            "elapsed_seconds": time.monotonic() - started,
        },
    )


def run(evidence: Path) -> None:
    frozen = record(evidence / "native-launch.json")
    if (
        frozen["cli_sha256"] != digest(BINARY)
        or frozen["entry_sha256"] != digest(Path(__file__))
        or frozen["service_sha256"] != digest(SERVICE)
        or frozen["watcher_sha256"]
        != digest(Path(__file__).with_name("agent_watch.py"))
    ):
        raise ValueError("Frozen launcher or executable changed")
    arms = frozen["arms"]
    if (
        not isinstance(arms, list)
        or len(arms) != 6
        or not all(isinstance(a, dict) for a in arms)
    ):
        raise ValueError("Expected six frozen trial packets")
    # Both arms of one pair can draft next decisions while the Ring18 queue serializes GPU workers.
    for pair in range(0, 6, 2):
        with ThreadPoolExecutor(max_workers=2) as pool:
            jobs = [pool.submit(trial, arm, evidence) for arm in arms[pair : pair + 2]]
            for job in jobs:
                job.result()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["prepare", "run"])
    parser.add_argument("--evidence", type=Path, required=True)
    parser.add_argument("--packets", type=Path)
    parser.add_argument("--optional", type=Path)
    parser.add_argument("--optional-manifest", type=Path)
    parser.add_argument("--public", type=Path, default=PUBLIC)
    args = parser.parse_args()
    if args.action == "prepare":
        if (
            args.packets is None
            or args.optional is None
            or args.optional_manifest is None
        ):
            parser.error(
                "prepare requires --packets, --optional and --optional-manifest"
            )
        prepare(
            args.evidence,
            args.packets,
            args.optional,
            args.optional_manifest,
            args.public,
        )
    else:
        run(args.evidence)


if __name__ == "__main__":
    main()

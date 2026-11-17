"""Operator-only, serialized two-GPU public execution and private evaluation on Ring18."""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
import re
import signal
import subprocess
import sys
import time
from pathlib import Path
from typing import TypedDict

DEFAULT_REMOTE = Path("/home/yuxuan/experiments/traincheck-token-budget-20261118")
REMOTE = DEFAULT_REMOTE
PYTHON = (
    "/home/yuxuan/experiments/traincheck-token-normalization-20261102/venv/bin/python"
)
DEVICES = (4, 5)
SSH = [
    "ssh",
    "-T",
    "-o",
    "BatchMode=yes",
    "-o",
    "ConnectTimeout=15",
    "-o",
    "ServerAliveInterval=30",
    "-o",
    "ServerAliveCountMax=4",
    "-o",
    "ControlMaster=auto",
    "-o",
    "ControlPersist=120",
    "-o",
    "ControlPath=/tmp/traincheck-ring18-1118-%C",
    "ring18",
]


class Request(TypedDict):
    id: str
    kind: str
    entry: str
    argv: list[str]
    files: dict[str, str]
    with_traincheck: bool
    worker_sha256: str


def parse_request(raw: object) -> Request:
    if not isinstance(raw, dict):
        raise ValueError("Expected request object")
    uid, kind, entry, argv, files, treatment, digest = (
        raw[k]
        for k in (
            "id",
            "kind",
            "entry",
            "argv",
            "files",
            "with_traincheck",
            "worker_sha256",
        )
    )
    if not isinstance(uid, str) or not re.fullmatch("[A-Za-z0-9_-]{1,96}", uid):
        raise ValueError("Invalid request ID")
    if (
        kind not in ("public", "grade")
        or entry not in ("train.py", "diagnostic.py")
        or not isinstance(treatment, bool)
    ):
        raise ValueError("Invalid service request")
    if (
        not isinstance(argv, list)
        or len(argv) > 32
        or not all(
            isinstance(x, str) and len(x) <= 256 and "\x00" not in x for x in argv
        )
    ):
        raise ValueError("Invalid worker arguments")
    if (
        not isinstance(files, dict)
        or not {"training.py"} <= set(files) <= {"training.py", "diagnostic.py"}
        or not all(
            isinstance(v, str) and len(v.encode()) <= 262144 for v in files.values()
        )
    ):
        raise ValueError("Invalid editable file set")
    if entry == "diagnostic.py" and entry not in files:
        raise ValueError("Missing diagnostic entry")
    if (
        not isinstance(digest, str)
        or digest != hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    ):
        raise ValueError("Service version mismatch")
    return Request(
        id=uid,
        kind=kind,
        entry=entry,
        argv=argv,
        files=files,
        with_traincheck=treatment,
        worker_sha256=digest,
    )


def gpu_state() -> dict[int, int]:
    output = subprocess.check_output(
        [
            "nvidia-smi",
            "--query-gpu=index,memory.used",
            "--format=csv,noheader,nounits",
        ],
        text=True,
        timeout=15,
    )
    inventory = {
        int(i): int(mem) for line in output.splitlines() for i, mem in [line.split(",")]
    }
    return {i: inventory[i] for i in DEVICES}


def owned_pids(uid: str) -> list[int]:
    marker = ("TRAINCHECK_STUDY_E_REQUEST=" + uid).encode()
    result: list[int] = []
    for path in Path("/proc").glob("[0-9]*/environ"):
        try:
            if path.stat().st_uid == os.getuid() and marker in path.read_bytes().split(
                b"\0"
            ):
                result.append(int(path.parent.name))
        except (FileNotFoundError, ProcessLookupError, PermissionError):
            continue  # Process exited or belongs to another user/restricted subsystem.
    return result


def cleanup(uid: str) -> None:
    for sig in (signal.SIGTERM, signal.SIGKILL):
        for pid in owned_pids(uid):
            try:
                os.kill(pid, sig)
            except ProcessLookupError:
                pass  # The tagged process exited before signalling.
        if not owned_pids(uid):
            return
        time.sleep(0.5)
    if owned_pids(uid):
        raise RuntimeError("Owned worker descendants remain")


def tail(path: Path) -> str:
    with path.open("rb") as f:
        f.seek(max(0, path.stat().st_size - 32768))
        return f.read().decode(errors="replace")


def run_worker(req: Request, run: Path, source: Path) -> dict[str, object]:
    public = req["kind"] == "public"
    if public:
        command = [
            PYTHON,
            "-m",
            "torch.distributed.run",
            "--standalone",
            "--nproc-per-node=2",
            req["entry"],
            *req["argv"],
        ]
    else:
        command = [
            PYTHON,
            str(REMOTE / "private/grade.py"),
            "--source",
            str(source),
            "--output",
            str(run / "grade.json"),
            "--device",
            "cuda",
            "--timeout",
            "85",
        ]
        if req["with_traincheck"]:
            command.extend(["--optional-package", str(REMOTE / "optional/traincheck")])
        collective_audit = REMOTE / "private/collectives.py"
        if collective_audit.is_file():
            command.extend(["--collective-audit", str(collective_audit)])
    env = os.environ.copy()
    env.update(
        TRAINCHECK_STUDY_E_REQUEST=req["id"],
        CUDA_VISIBLE_DEVICES="4,5",
        OMP_NUM_THREADS="1",
        PYTHONDONTWRITEBYTECODE="1",
        PYTHONNOUSERSITE="1",
        PYTHONPATH=str(REMOTE / ("optional" if req["with_traincheck"] else "control"))
        + os.pathsep
        + str(source),
    )
    with (run / "stdout.log").open("xb") as out, (run / "stderr.log").open("xb") as err:
        start = time.monotonic()
        (run / "started.json").write_text(
            json.dumps(
                {
                    "unix_time": time.time(),
                    "devices": DEVICES,
                    "source_sha256": hashlib.sha256(
                        (source / "training.py").read_bytes()
                    ).hexdigest(),
                }
            )
        )
        process = subprocess.Popen(
            command, cwd=source, env=env, stdout=out, stderr=err, start_new_session=True
        )
        try:
            code = process.wait(timeout=120)
            status = "ok" if code == 0 else "application_error"
        except subprocess.TimeoutExpired:
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            code = process.wait(timeout=10)
            status = "timeout"
        finally:
            cleanup(req["id"])
        until = time.monotonic() + 8
        post = gpu_state()
        while any(post.values()) and time.monotonic() < until:
            time.sleep(0.25)
            post = gpu_state()
        elapsed = time.monotonic() - start
    if any(post.values()):
        raise RuntimeError(
            "GPU release not verified; inspect durable logs and started receipt"
        )
    result: dict[str, object] = {
        "status": status,
        "exit_code": code,
        "worker_seconds": elapsed,
        "gpu_seconds": elapsed * 2,
        "devices": DEVICES,
        "gpu_release": post,
        "stdout": tail(run / "stdout.log"),
        "stderr": tail(run / "stderr.log"),
    }
    if not public and (run / "grade.json").is_file():
        result["grade"] = json.loads((run / "grade.json").read_text())
    return result


def serve(req: Request) -> dict[str, object]:
    run = REMOTE / "runs" / req["id"]
    run.mkdir(mode=0o700, exist_ok=False)
    (run / "request.json").write_text(json.dumps(req))
    source = run / "source"
    source.mkdir()
    for name in ("train.py", "workload.py"):
        (source / name).write_bytes((REMOTE / "public" / name).read_bytes())
    for name, content in req["files"].items():
        (source / name).write_bytes(content.encode())
    start = time.monotonic()
    with (REMOTE / "service.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        queue = time.monotonic() - start
        pre = gpu_state()
        if any(pre.values()):
            receipt: dict[str, object] = {
                "status": "busy",
                "worker_seconds": 0.0,
                "gpu_seconds": 0.0,
                "stdout": "",
                "stderr": "Selected GPUs occupied; no launch.",
            }
        else:
            receipt = run_worker(req, run, source)
        receipt.update(
            queue_seconds=queue,
            initial_gpu_memory_mib=pre,
            id=req["id"],
            kind=req["kind"],
            training_sha256=hashlib.sha256(
                req["files"]["training.py"].encode()
            ).hexdigest(),
        )
        (run / "receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")
    return receipt


def client() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("kind", choices=("public", "grade"))
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--id", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--entry", choices=("train.py", "diagnostic.py"), default="train.py"
    )
    parser.add_argument("--args-json", default="[]")
    parser.add_argument("--with-traincheck", action="store_true")
    args = parser.parse_args()
    remote = Path(os.environ.get("TRAINCHECK_STUDY_ROOT", str(DEFAULT_REMOTE)))
    if remote.parent != DEFAULT_REMOTE.parent or not re.fullmatch(
        r"traincheck-[a-z0-9-]+", remote.name
    ):
        raise ValueError("Expected a named TrainCheck experiment directory")
    files: dict[str, str] = {}
    for name in ("training.py", "diagnostic.py"):
        path = args.source / name
        if path.is_symlink():
            raise ValueError("Expected regular source file")
        if path.is_file():
            files[name] = path.read_bytes().decode()
    req = parse_request(
        dict(
            id=args.id,
            kind=args.kind,
            entry=args.entry,
            argv=json.loads(args.args_json),
            files=files,
            with_traincheck=args.with_traincheck,
            worker_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        )
    )
    args.output.mkdir(parents=True, exist_ok=False)
    payload = json.dumps(req).encode()
    (args.output / "request.json").write_bytes(payload)
    since = time.monotonic()
    proc = subprocess.run(
        [*SSH, PYTHON, str(remote / "ring_service.py"), "serve"],
        input=payload,
        capture_output=True,
        timeout=1800,
    )
    duration = time.monotonic() - since
    (args.output / "transport.stderr").write_bytes(proc.stderr)
    if proc.returncode:
        raise RuntimeError(
            f"SSH worker failed ({proc.returncode}): {proc.stderr[-2000:]!r}; inspect remote request before retrying"
        )
    receipt = json.loads(proc.stdout)
    if not isinstance(receipt, dict) or receipt["id"] != req["id"]:
        raise ValueError("Unbound worker receipt")
    receipt["service_seconds"] = duration
    receipt["transfer_seconds"] = (
        duration - receipt["queue_seconds"] - receipt["worker_seconds"]
    )
    (args.output / "receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")
    print(
        json.dumps(
            {
                key: receipt[key]
                for key in ("id", "status", "gpu_seconds", "queue_seconds")
            }
        )
    )


if __name__ == "__main__":
    if sys.argv[1:] == ["serve"]:
        REMOTE = Path(__file__).resolve().parent
        print(json.dumps(serve(parse_request(json.load(sys.stdin)))))
    else:
        client()

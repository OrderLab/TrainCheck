"""Observe candidate edits without instrumenting solver code or returning grades."""

from __future__ import annotations

import hashlib
import json
import os
import stat
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator


def _regular_bytes(path: Path) -> bytes | None:
    # Replacements can make a candidate temporarily absent. Never follow a
    # link created by an editor; compare the identity of the opened file.
    if not path.is_file() or path.is_symlink():
        return None
    try:
        fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
    except FileNotFoundError:
        return None
    with os.fdopen(fd, "rb") as stream:
        before = os.fstat(stream.fileno())
        if not stat.S_ISREG(before.st_mode):
            return None
        data = stream.read()
        after = os.fstat(stream.fileno())
        if before.st_size != after.st_size or before.st_mtime_ns != after.st_mtime_ns:
            return None
        return data


def _watch(
    source: Path,
    native: Path,
    directory: Path,
    stop: threading.Event,
    ready: threading.Event,
) -> None:
    blobs = directory / "blobs"
    blobs.mkdir(parents=True, exist_ok=False)
    previous: tuple[str | None, int] | None = None
    with (directory / "index.jsonl").open("x") as index:
        while True:
            ending = stop.is_set()
            raw = _regular_bytes(source)
            stamp = time.monotonic()
            try:
                received_bytes = native.stat().st_size
            except FileNotFoundError:
                # The native collector creates its transcript after we start.
                received_bytes = 0
            sha = None if raw is None else hashlib.sha256(raw).hexdigest()
            current = (sha, received_bytes)
            if current != previous or ending:
                if sha is not None:
                    target = blobs / f"{sha}.py"
                    if not target.exists():
                        assert raw is not None
                        target.write_bytes(raw)
                index.write(
                    json.dumps(
                        {
                            "observed_monotonic": stamp,
                            "sha256": sha,
                            "native_received_bytes": received_bytes,
                            "final": ending,
                        }
                    )
                    + "\n"
                )
                index.flush()
                previous = current
            ready.set()
            if ending:
                return
            stop.wait(0.05)


@contextmanager
def watch_source(source: Path, turn_dir: Path) -> Iterator[None]:
    """Preserve observed source versions and approximate native event boundaries.

    The sampling interval is 50ms. A version that exists only between samples
    may be missed; private grading at actual submitted boundaries is separate.
    """
    stop = threading.Event()
    ready = threading.Event()
    directory = turn_dir.with_name(turn_dir.name + ".watch")
    with ThreadPoolExecutor(max_workers=1) as pool:
        observer = pool.submit(
            _watch, source, turn_dir / "received.jsonl", directory, stop, ready
        )
        try:
            while not ready.wait(0.05):
                if observer.done():
                    observer.result()
            yield
        finally:
            stop.set()
            observer.result()

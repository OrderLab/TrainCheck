"""Build the explicitly composed, disposable validation source without changing Git."""

import argparse
import hashlib
import io
import json
import subprocess
import tarfile
from pathlib import Path

BASE = "99d16905296d83495570fdcafa91370b242d2528"
INTEGRITY = "7d6c683db92d83ffd6db9fc9c11fbc0ebdd2c5d4"
TORCHSCRIPT = "9ac61d1140cf8679c24442a174ec12f4de069980"
STATE = "1b5aedffbeefd4cdd9d30036dbb0a833521526cf"
TYPENAME = "2f034ae5d0fab7e792b7c0a5bdc6a1acfeb8962c"


def digest(data):
    return hashlib.sha256(data).hexdigest()


def build(repo, destination):
    def git(*args):
        return subprocess.check_output(["git", "-C", str(repo), *args])

    destination.mkdir(parents=True, exist_ok=False)
    with tarfile.open(
        fileobj=io.BytesIO(git("archive", BASE, "traincheck", "tests"))
    ) as tar:
        for member in tar:
            if member.isfile():
                path = destination / member.name
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(tar.extractfile(member).read())
    old_hashes = {
        str(p.relative_to(destination)): digest(p.read_bytes())
        for p in destination.rglob("*.py")
    }
    # These instrumentor files have no changes on the inference/cache stack.
    for path in (
        "traincheck/instrumentor/tracer.py",
        "traincheck/instrumentor/dumper.py",
    ):
        assert git("show", f"{INTEGRITY}^:{path}") == (destination / path).read_bytes()
    assert (
        git("show", f"{TYPENAME}^:traincheck/utils.py")
        == (destination / "traincheck/utils.py").read_bytes()
    )
    replacements = {
        "traincheck/instrumentor/tracer.py": TORCHSCRIPT,
        "traincheck/instrumentor/dumper.py": INTEGRITY,
        "traincheck/instrumentor/torchscript_compat.py": TORCHSCRIPT,
        "tests/test_trace_integrity.py": INTEGRITY,
        "tests/test_torchscript_compat.py": TORCHSCRIPT,
        "tests/test_precondition_state_isolation.py": STATE,
        "traincheck/utils.py": TYPENAME,
        "tests/test_dynamic_typename.py": TYPENAME,
    }
    for path, commit in replacements.items():
        (destination / path).write_bytes(git("show", f"{commit}:{path}"))
    path = destination / "traincheck/invariant/precondition.py"
    source = path.read_text()
    old = "_pruned_clauses: set[PreconditionClause] = set(),"
    new = "_pruned_clauses: set[PreconditionClause] | None = None,"
    assert source.count(old) == 1
    source = source.replace(old, new)
    # Copy only the two approved state hunks, preserving the base RNG isolation.
    state_source = git("show", f"{STATE}:traincheck/invariant/precondition.py").decode()
    comment = "    # Recursive children share pruning state; independent solves start fresh.\n"
    insertion = (
        comment + "    if _pruned_clauses is None:\n        _pruned_clauses = set()\n"
    )
    assert insertion in state_source
    anchor = state_source.split(insertion)[1][:100]
    assert source.count(anchor) == 1
    source = source.replace(anchor, insertion + anchor, 1)
    assert "group_rng = random.Random(42)" in source
    path.write_text(source)
    hashes = {
        str(p.relative_to(destination)): digest(p.read_bytes())
        for p in destination.rglob("*.py")
    }
    changed = sorted(p for p, h in hashes.items() if h != old_hashes.get(p))
    manifest = {
        "base": BASE,
        "trace_integrity": INTEGRITY,
        "torchscript": TORCHSCRIPT,
        "state_fix_hunks_from": STATE,
        "changed_python_files": changed,
        "dynamic_typename_fix": TYPENAME,
        "source_sha256": hashes,
    }
    (destination / "source-manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n"
    )
    return manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(build(args.repo.resolve(), args.output.resolve()), indent=2))

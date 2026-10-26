"""Validate an installed release in its Python environment, outside the checkout.

Run this file with the Python executable from a venv containing the candidate
wheel. Source/editable installations deliberately fail the prefix check.
"""

import argparse
import importlib.metadata
import json
import subprocess
import sys
from pathlib import Path
from typing import TypedDict

CONSOLE_COMMANDS = frozenset(
    {
        "traincheck-collect",
        "traincheck-infer",
        "traincheck-check",
        "traincheck-onlinecheck",
    }
)


class CommandCheck(TypedDict):
    name: str
    target: str


class VerificationReport(TypedDict):
    version: str
    environment_prefix: str
    package_path: str
    commands: list[CommandCheck]


# Each command gets a separate interpreter: an earlier command's imports must
# not make a malformed dotted attribute path appear loadable.
_LOAD_ENTRY_POINT = """
import importlib.metadata
from pathlib import Path
import sys

distribution = importlib.metadata.distribution("traincheck")
name = sys.argv[1]
expected_version = sys.argv[2]
expected_package = Path(sys.argv[3]).resolve()
if distribution.version != expected_version:
    raise RuntimeError("Child process resolved an unexpected package version")
matches = [
    entry for entry in distribution.entry_points
    if entry.group == "console_scripts" and entry.name == name
]
if len(matches) != 1:
    raise RuntimeError(f"Expected one console entry point for {name!r}")
entry = matches[0]
loaded = entry.load()
if not callable(loaded):
    raise TypeError(f"Console entry point {name!r} did not load a callable")
import traincheck
module_file = traincheck.__file__
if not isinstance(module_file, str):
    raise TypeError("Installed traincheck has no filesystem module path")
if Path(module_file).resolve() != expected_package:
    raise RuntimeError(f"Unexpected imported package location: {module_file}")
"""


def console_entries(
    distribution: importlib.metadata.Distribution,
) -> dict[str, importlib.metadata.EntryPoint]:
    entries: dict[str, importlib.metadata.EntryPoint] = {}
    for entry in distribution.entry_points:
        if entry.group != "console_scripts":
            continue
        if entry.name in entries:
            raise ValueError(f"Duplicate console entry point: {entry.name!r}")
        entries[entry.name] = entry
    if entries.keys() != CONSOLE_COMMANDS:
        raise ValueError(
            "Unexpected console command set: "
            f"expected {sorted(CONSOLE_COMMANDS)!r}, found {sorted(entries)!r}"
        )
    return entries


def verify_installed(expected_version: str) -> VerificationReport:
    distribution = importlib.metadata.distribution("traincheck")
    if distribution.version != expected_version:
        raise ValueError(
            f"Expected traincheck {expected_version}, found {distribution.version}"
        )
    prefix = Path(sys.prefix).resolve()
    # Distribution.locate_file returns a SimplePath; its string representation
    # supplies the filesystem path for this installed-wheel-only validator.
    package_path = Path(str(distribution.locate_file("traincheck/__init__.py")))
    package_path = package_path.resolve(strict=True)
    if not package_path.is_relative_to(prefix):
        raise ValueError(
            f"Package is outside the selected Python environment: {package_path}"
        )
    commands: list[CommandCheck] = []
    for name, entry in sorted(console_entries(distribution).items()):
        result = subprocess.run(
            [
                sys.executable,
                "-I",
                "-B",
                "-c",
                _LOAD_ENTRY_POINT,
                name,
                expected_version,
                str(package_path),
            ],
            cwd=prefix,
            capture_output=True,
            text=True,
            timeout=60,
            check=False,
        )
        if result.returncode != 0:
            raise RuntimeError(
                f"Console entry point {name!r} failed in a fresh interpreter "
                f"(exit {result.returncode}).\n{result.stdout}{result.stderr}"
            )
        commands.append({"name": name, "target": entry.value})
    return {
        "version": distribution.version,
        "environment_prefix": str(prefix),
        "package_path": str(package_path),
        "commands": commands,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--expected-version", required=True)
    args = parser.parse_args()
    expected_version: object = args.expected_version
    if not isinstance(expected_version, str) or not expected_version:
        parser.error("--expected-version must be a nonempty string")
    print(json.dumps(verify_installed(expected_version), indent=2))


if __name__ == "__main__":
    main()

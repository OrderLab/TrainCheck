"""Exercise import availability in fresh processes without changing pytest state."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]


def worker(tmp_path: Path, statements: str) -> dict[str, object]:
    package = tmp_path / "traincheck"
    package.mkdir()
    (package / "__init__.py").write_text("IMPORTED = True\n")
    (package / "child.py").write_text("IMPORTED = True\n")
    (tmp_path / "trainchecker.py").write_text("VALUE = 17\n")
    source = (
        "import importlib, json, sys\n"
        "from types import ModuleType\n"
        "sys.path.insert(0, sys.argv[1])\n"
        "from tools.runtime_pilot.access import install_traincheck_blocker\n"
        "sys.path.insert(0, sys.argv[2])\n" + statements
    )
    completed = subprocess.run(
        [sys.executable, "-I", "-c", source, str(REPO), str(tmp_path)],
        check=True,
        capture_output=True,
        text=True,
        timeout=20,
    )
    raw: object = json.loads(completed.stdout)
    assert isinstance(raw, dict)
    result: dict[str, object] = {}
    for name, value in raw.items():
        assert isinstance(name, str)
        result[name] = value
    return result


def test_installed_package_blocked_then_owned_finder_cleanup(tmp_path: Path) -> None:
    result = worker(
        tmp_path,
        "original = tuple(sys.meta_path)\n"
        "finder = install_traincheck_blocker()\n"
        "try:\n"
        "    importlib.import_module('traincheck')\n"
        "except ModuleNotFoundError as error:\n"
        "    missing = error.name\n"
        "else:\n"
        "    raise AssertionError('Installed TrainCheck was not blocked')\n"
        "assert 'traincheck' not in sys.modules\n"
        "sys.meta_path.remove(finder)\n"
        "assert tuple(sys.meta_path) == original\n"
        "package = importlib.import_module('traincheck')\n"
        "print(json.dumps({'missing': missing, 'imported_after_cleanup': package.IMPORTED}))\n",
    )
    assert result == {"missing": "traincheck", "imported_after_cleanup": True}


@pytest.mark.parametrize("name", ["traincheck.child", "traincheck.instrumentor.tracer"])
def test_submodule_namespace_blocked(tmp_path: Path, name: str) -> None:
    result = worker(
        tmp_path,
        "finder = install_traincheck_blocker()\n"
        f"name = {name!r}\n"
        "try:\n"
        "    finder.find_spec(name, [])\n"
        "except ModuleNotFoundError as error:\n"
        "    assert error.name == name\n"
        "else:\n"
        "    raise AssertionError('Submodule was not blocked')\n"
        "try:\n"
        "    importlib.import_module(name)\n"
        "except ModuleNotFoundError as error:\n"
        "    assert error.name == 'traincheck'\n"
        "else:\n"
        "    raise AssertionError('Normal submodule import was not blocked')\n"
        "sys.meta_path.remove(finder)\n"
        "print(json.dumps({'blocked': name}))\n",
    )
    assert result == {"blocked": name}


def test_unrelated_imports_and_similar_prefix_unchanged(tmp_path: Path) -> None:
    result = worker(
        tmp_path,
        "finder = install_traincheck_blocker()\n"
        "module = importlib.import_module('trainchecker')\n"
        "decimal = importlib.import_module('decimal')\n"
        "assert finder.find_spec('other.traincheck', None) is None\n"
        "sys.meta_path.remove(finder)\n"
        "print(json.dumps({'prefix': module.VALUE, 'decimal': str(decimal.Decimal('1.25'))}))\n",
    )
    assert result == {"prefix": 17, "decimal": "1.25"}


@pytest.mark.parametrize("name", ["traincheck", "traincheck.child"])
@pytest.mark.parametrize("placeholder", [False, True])
def test_preloaded_namespace_rejected_without_removal(
    tmp_path: Path, name: str, placeholder: bool
) -> None:
    result = worker(
        tmp_path,
        f"name = {name!r}\n"
        f"existing = None if {placeholder!r} else ModuleType(name)\n"
        "sys.modules[name] = existing\n"
        "original = tuple(sys.meta_path)\n"
        "try:\n"
        "    install_traincheck_blocker()\n"
        "except RuntimeError as error:\n"
        "    assert name in str(error)\n"
        "else:\n"
        "    raise AssertionError('Preloaded namespace was accepted')\n"
        "assert name in sys.modules and sys.modules[name] is existing\n"
        "assert tuple(sys.meta_path) == original\n"
        "print(json.dumps({'preserved': name}))\n",
    )
    assert result == {"preserved": name}


def test_repeat_install_restores_precedence_without_duplicate(tmp_path: Path) -> None:
    result = worker(
        tmp_path,
        "original = tuple(sys.meta_path)\n"
        "finder = install_traincheck_blocker()\n"
        "assert install_traincheck_blocker() is finder\n"
        "sys.meta_path.remove(finder)\n"
        "sys.meta_path.append(finder)\n"
        "assert install_traincheck_blocker() is finder\n"
        "assert sys.meta_path[0] is finder\n"
        "assert sum(item is finder for item in sys.meta_path) == 1\n"
        "sys.meta_path.remove(finder)\n"
        "assert tuple(sys.meta_path) == original\n"
        "print(json.dumps({'idempotent': True}))\n",
    )
    assert result == {"idempotent": True}


def test_repeat_install_does_not_hide_later_namespace_injection(tmp_path: Path) -> None:
    result = worker(
        tmp_path,
        "finder = install_traincheck_blocker()\n"
        "existing = ModuleType('traincheck')\n"
        "sys.modules['traincheck'] = existing\n"
        "try:\n"
        "    install_traincheck_blocker()\n"
        "except RuntimeError:\n"
        "    pass\n"
        "else:\n"
        "    raise AssertionError('Later namespace injection was hidden')\n"
        "assert sys.modules['traincheck'] is existing\n"
        "assert sum(item is finder for item in sys.meta_path) == 1\n"
        "sys.meta_path.remove(finder)\n"
        "print(json.dumps({'rejected': True}))\n",
    )
    assert result == {"rejected": True}

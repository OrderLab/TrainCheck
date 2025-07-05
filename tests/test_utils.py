import importlib.util
from pathlib import Path

import torch

# Load utils module without triggering traincheck package __init__
UTILS_PATH = Path(__file__).resolve().parents[1] / "traincheck" / "utils.py"
_spec = importlib.util.spec_from_file_location("tc_utils", UTILS_PATH)
_utils = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_utils)

typename = _utils.typename


def test_typename_builtin_function():
    assert typename(len) == "len"
    assert typename(print) == "print"

def test_typename_tensor_and_parameter():
    t = torch.tensor([1.0])
    assert typename(t) == t.type()
    p = torch.nn.Parameter(torch.zeros(1))
    assert typename(p) == "torch.nn.Parameter"

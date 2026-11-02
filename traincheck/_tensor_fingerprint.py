"""Exact tensor hashing shared by tracker dumping and explicit observations."""

import hashlib

from torch import Tensor


def exact_tensor_hash(tensor: Tensor) -> int:
    """Preserve the tracker's conventional SHA-256 implementation.

    Callers choose the input representation. The legacy caller passes a native
    tensor, retaining its existing NumPy dtype limitations. Model snapshots pass
    a contiguous uint8 view to cover all bytes of supported logical dtypes.
    This module imports no instrumentor package and activates no instrumentation.
    """
    if tensor.is_cuda:
        tensor = tensor.cpu()
    return int(hashlib.sha256(tensor.detach().numpy().tobytes()).hexdigest(), 16)

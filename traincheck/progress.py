"""Thin tqdm wrapper that can be silenced during invariant checking.

Import this instead of tqdm in relation and trace modules:

    from traincheck.progress import tqdm

When traincheck.utils._suppress_inner_progress is True (set by
check_engine() while the outer checking bar is active), all bars
created via this wrapper are disabled so only the single top-level
progress bar is visible.
"""

from tqdm import tqdm as _tqdm_orig


def tqdm(iterable=None, *args, **kwargs):  # type: ignore[override]
    from traincheck import utils as _utils

    if _utils._suppress_inner_progress and "disable" not in kwargs:
        kwargs["disable"] = True
    if iterable is not None:
        return _tqdm_orig(iterable, *args, **kwargs)
    return _tqdm_orig(*args, **kwargs)

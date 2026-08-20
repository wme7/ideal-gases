# SPDX-License-Identifier: MIT
# Copyright (c) 2014 Manuel A. Diaz

"""Optional tqdm progress bars for NSF time loops."""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from typing import Protocol

__all__ = [
    "DEFAULT_PROGRESS_EVERY",
    "_validate_progress_every",
    "nsf_pbar",
]

DEFAULT_PROGRESS_EVERY = 100


class ProgressBar(Protocol):
    n: float
    total: float

    def update(self, n: float = 1.0) -> object: ...


class _NoOpBar:
    __slots__ = ("n", "total")

    def __init__(self, total: float = 0.0) -> None:
        self.n = 0.0
        self.total = total

    def update(self, n: float = 1.0) -> None:
        self.n += n


def _require_tqdm():
    try:
        from tqdm import tqdm
    except ImportError as exc:
        msg = "Progress bars require tqdm. Install with: pip install ideal-gases[progress]"
        raise RuntimeError(msg) from exc
    return tqdm


def _validate_progress_every(progress_every) -> int:
    every = int(progress_every)
    if every < 1 or float(progress_every) != float(every):
        raise ValueError(
            f"progress_every = {progress_every} invalid; expected a positive integer"
        )
    return every


@contextmanager
def nsf_pbar(enabled: bool, *, total: float, desc: str) -> Iterator[ProgressBar]:
    if not enabled or total <= 0:
        yield _NoOpBar(total)
        return
    tqdm = _require_tqdm()
    with tqdm(
        total=total,
        unit="s",
        desc=desc,
        bar_format=(
            "{l_bar}{bar}| {n:.4f}/{total:.4f}s "
            "[{elapsed}<{remaining}, {rate_fmt}{postfix}]"
        ),
    ) as bar:
        yield bar

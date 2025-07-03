"""Utilities to handle optional dependencies in dltype."""

from functools import cache
from collections.abc import Callable
from typing import TypeVar, ParamSpec, NoReturn

Ret = TypeVar("Ret")
P = ParamSpec("P")


def _empty_wrapper(fn: Callable[P, Ret]) -> Callable[P, Ret]:
    """A no-op function used as a placeholder for optional dependencies."""
    return fn


# import these first and avoid runtime penalties if they are not available
try:
    import torch

    torch_jit_unused = torch.jit.unused  # re-export for compatibility
except ImportError:
    torch_jit_unused = _empty_wrapper
    torch = None

try:
    import numpy as np
except ImportError:
    np = None


@cache
def is_torch_available() -> bool:
    """Check if the torch library is available."""
    return torch is not None


@cache
def is_numpy_available() -> bool:
    """Check if the numpy library is available."""
    return np is not None


def raise_for_missing_dependency() -> NoReturn:
    """Raise an ImportError if neither torch nor numpy is available."""
    if not is_torch_available() and not is_numpy_available():
        raise ImportError(
            "Neither torch nor numpy is available. Please install one of them to use dltype."
        )
    assert False, (
        "Improper use of raise_for_missing_dependency, should only be called when both dependencies are missing."
    )


def is_torch_scripting() -> bool:
    """Check if the torch library is in scripting mode."""
    if not is_torch_available():
        return False

    return torch.jit.is_scripting()

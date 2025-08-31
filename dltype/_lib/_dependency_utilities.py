"""Utilities to handle optional dependencies in dltype."""

from collections.abc import Callable
from functools import cache
from typing import NoReturn, ParamSpec, TypeVar

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


@cache
def is_np_float128_available() -> bool:
    _FLOAT128_AVAILABLE = False
    if is_numpy_available():
        try:
            # Check if float128 is available (may not be supported on all platforms)
            _ = np.float128
            _FLOAT128_AVAILABLE = True
        except AttributeError:
            pass
    return _FLOAT128_AVAILABLE


@cache
def is_np_longdouble_available() -> bool:
    _LONGDOUBLE_AVAILABLE = False
    if is_numpy_available():
        try:
            # Check if longdouble is available (may not be supported on all platforms)
            _ = np.longdouble
            _LONGDOUBLE_AVAILABLE = True
        except AttributeError:
            pass
    return _LONGDOUBLE_AVAILABLE


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

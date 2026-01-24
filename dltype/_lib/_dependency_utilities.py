"""Utilities to handle optional dependencies in dltype."""

from __future__ import annotations

import typing
from functools import cache

if typing.TYPE_CHECKING:
    from collections.abc import Callable

Ret = typing.TypeVar("Ret")
P = typing.ParamSpec("P")


def _empty_wrapper(fn: Callable[P, Ret]) -> Callable[P, Ret]:
    """A no-op function used as a placeholder for optional dependencies."""
    return fn


# import these first and avoid runtime penalties if they are not available
try:
    import torch

    # re-export for compatibility
    torch_jit_unused = torch.jit.unused
except ImportError:
    torch_jit_unused = _empty_wrapper
    torch = None


try:
    import numpy as np
except ImportError:
    np = None

try:
    import jax
except ImportError:
    jax = None


@cache
def is_torch_available() -> bool:
    """Check if the torch library is available."""
    return torch is not None


@cache
def is_numpy_available() -> bool:
    """Check if the numpy library is available."""
    return np is not None


@cache
def is_jax_available() -> bool:
    """Check if jax is available."""
    return jax is not None


@cache
def is_np_float128_available() -> bool:
    float_128_available = False
    if is_numpy_available():
        try:
            # Check if float128 is available (may not be supported on all platforms)
            _ = np.float128  # pyright: ignore[reportOptionalMemberAccess]
            float_128_available = True
        except AttributeError:
            pass
    return float_128_available


@cache
def is_np_longdouble_available() -> bool:
    longdouble_available = False
    if is_numpy_available():
        try:
            # Check if longdouble is available (may not be supported on all platforms)
            _ = np.longdouble  # pyright: ignore[reportOptionalMemberAccess]
            longdouble_available = True
        except AttributeError:
            pass
    return longdouble_available


def raise_for_missing_dependency() -> typing.NoReturn:
    """Raise an ImportError if neither torch nor numpy is available."""
    if not is_torch_available() and not is_numpy_available():
        msg = "Neither torch nor numpy is available. Please install one of them to use dltype."
        raise ImportError(msg)

    msg = "Improper use of raise_for_missing_dependency, should only be called when both dependencies are missing."
    raise AssertionError(msg)


def is_torch_scripting() -> bool:
    """Check if the torch library is in scripting mode."""
    if not is_torch_available():
        return False

    return torch.jit.is_scripting()  # pyright: ignore[reportOptionalMemberAccess, reportPrivateImportUsage]

"""Test that dltype can operate with either numpy or torch installed."""

import pytest

from unittest.mock import patch
from importlib import reload
from collections.abc import Iterator
import dltype
import sys

import numpy
import torch


@pytest.fixture(autouse=True)
def clear_cached_available_fns() -> Iterator[None]:
    """Clear cached functions to ensure fresh imports."""
    # Clear the cache for the dependency utilities
    from dltype._lib._dependency_utilities import is_torch_available, is_numpy_available

    is_torch_available.cache_clear()
    is_numpy_available.cache_clear()
    yield


@pytest.fixture(autouse=True)
def reset_modules() -> Iterator[None]:
    # Store a copy of the initial sys.modules state
    initial_modules = sys.modules.copy()
    yield
    # Restore sys.modules to its initial state after the test
    sys.modules.clear()
    sys.modules.update(initial_modules)


@pytest.fixture()
def mock_missing_numpy() -> Iterator[None]:
    """Mock numpy as missing."""
    with patch("dltype._lib._dependency_utilities.np", None):
        yield


@pytest.fixture()
def mock_missing_torch() -> Iterator[None]:
    """Mock torch as missing."""
    with patch("dltype._lib._dependency_utilities.torch", None):
        yield


def test_dltype_imports_without_torch(mock_missing_torch: None):
    """Test that dltype can be imported without torch."""
    del sys.modules["torch"]

    with pytest.raises(ImportError):
        reload(torch)

    _reloaded_dltype = reload(dltype)

    assert _reloaded_dltype.BoolTensor.DTYPES == (numpy.bool_,)


def test_dltype_imports_without_numpy(mock_missing_numpy: None):
    """Test that dltype can be imported without numpy."""
    del sys.modules["numpy"]

    with pytest.raises(ImportError):
        reload(numpy)

    _reloaded_dltype = reload(dltype)

    assert _reloaded_dltype.BoolTensor.DTYPES == (torch.bool,)


def test_dltype_imports_with_both():
    """Test that dltype can be imported with both torch and numpy."""
    _reloaded_dltype = reload(dltype)
    assert _reloaded_dltype.BoolTensor.DTYPES == (
        torch.bool,
        numpy.bool_,
    )


def test_dltype_asserts_import_error_with_neither(
    mock_missing_numpy: None, mock_missing_torch: None
):
    """Test that dltype raises ImportError if neither torch nor numpy is available."""

    with pytest.raises(ImportError, match="Neither torch nor numpy is available"):
        reload(dltype)

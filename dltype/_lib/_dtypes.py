"""Supported datatypes."""

import typing

from dltype._lib import _dependency_utilities as _deps

# NOTE: the order of these is important, pyright assumes the last branch is taken
# so we get proper union type hint checking
if _deps.is_numpy_available() and not _deps.is_torch_available() and not _deps.is_jax_available():
    # numpy is here, jax and torch are not
    import numpy as np
    import numpy.typing as npt

    DLtypeTensorT: typing.TypeAlias = npt.NDArray[typing.Any]  # pyright: ignore[reportRedeclaration]
    DLtypeDtypeT: typing.TypeAlias = npt.DTypeLike  # pyright: ignore[reportRedeclaration]
    SUPPORTED_TENSOR_TYPES: typing.Final = {np.ndarray}
elif not _deps.is_numpy_available() and _deps.is_torch_available():
    # If numpy is not available jax cannot be installed so we have only torch.
    import torch

    DLtypeTensorT: typing.TypeAlias = torch.Tensor  # pyright: ignore[reportRedeclaration]
    DLtypeDtypeT: typing.TypeAlias = torch.dtype  # pyright: ignore[reportRedeclaration]
    SUPPORTED_TENSOR_TYPES: typing.Final = {torch.Tensor}  # pyright: ignore[reportGeneralTypeIssues, reportConstantRedefinition]
elif _deps.is_numpy_available() and not _deps.is_torch_available() and _deps.is_jax_available():
    # we have numpy and jax but not torch
    import jax
    import numpy as np
    import numpy.typing as npt

    DLtypeTensorT: typing.TypeAlias = jax.Array | npt.NDArray[typing.Any]  # pyright: ignore[reportRedeclaration]
    DLtypeDtypeT: typing.TypeAlias = npt.DTypeLike  # pyright: ignore[reportRedeclaration]
    SUPPORTED_TENSOR_TYPES: typing.Final = {np.ndarray, jax.Array}  # pyright: ignore[reportGeneralTypeIssues, reportConstantRedefinition]
elif _deps.is_numpy_available() and _deps.is_torch_available() and not _deps.is_jax_available():
    # we have torch and numpy but not jax
    import numpy as np
    import numpy.typing as npt
    import torch

    DLtypeTensorT: typing.TypeAlias = torch.Tensor | npt.NDArray[typing.Any]  # pyright: ignore[reportRedeclaration]
    DLtypeDtypeT: typing.TypeAlias = torch.dtype | npt.DTypeLike  # pyright: ignore[reportRedeclaration]
    SUPPORTED_TENSOR_TYPES: typing.Final = {torch.Tensor, np.ndarray}  # pyright: ignore[reportGeneralTypeIssues, reportConstantRedefinition]
elif _deps.is_numpy_available() and _deps.is_torch_available() and _deps.is_jax_available():
    # we have all three
    import jax
    import numpy as np
    import numpy.typing as npt
    import torch

    DLtypeTensorT: typing.TypeAlias = jax.Array | npt.NDArray[typing.Any] | torch.Tensor  # pyright: ignore[reportRedeclaration]
    DLtypeDtypeT: typing.TypeAlias = npt.DTypeLike | torch.dtype  # pyright: ignore[reportRedeclaration]
    SUPPORTED_TENSOR_TYPES: typing.Final = {np.ndarray, jax.Array, torch.Tensor}  # pyright: ignore[reportGeneralTypeIssues, reportConstantRedefinition]
else:
    _deps.raise_for_missing_dependency()

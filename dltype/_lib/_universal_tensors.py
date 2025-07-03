"""Tensor types work with either torch or numpy (maybe extended later)."""

from dltype._lib import _tensor_type_base
from dltype._lib._dependency_utilities import (
    is_torch_available,
    is_numpy_available,
    raise_for_missing_dependency,
)

if is_numpy_available():
    from dltype._lib._numpy_tensors import (
        IntTensor as NumPyIntTensor,
        FloatTensor as NumPyFloatTensor,
        BoolTensor as NumPyBoolTensor,
        DoubleTensor as NumPyDoubleTensor,
    )

if is_torch_available():
    from dltype._lib._torch_tensors import (
        IntTensor as TorchIntTensor,
        FloatTensor as TorchFloatTensor,
        BoolTensor as TorchBoolTensor,
        DoubleTensor as TorchDoubleTensor,
    )


class IntTensor(_tensor_type_base.TensorTypeBase):
    """A class to represent an integer tensor type."""

    DTYPES = (
        (*TorchIntTensor.DTYPES, *NumPyIntTensor.DTYPES)
        if is_torch_available() and is_numpy_available()
        else (
            TorchIntTensor.DTYPES
            if is_torch_available()
            else NumPyIntTensor.DTYPES
            if is_numpy_available()
            else raise_for_missing_dependency()
        )
    )


class FloatTensor(_tensor_type_base.TensorTypeBase):
    """A class to represent a float tensor type."""

    DTYPES = (
        (*TorchFloatTensor.DTYPES, *NumPyFloatTensor.DTYPES)
        if is_torch_available() and is_numpy_available()
        else (
            TorchFloatTensor.DTYPES
            if is_torch_available()
            else NumPyFloatTensor.DTYPES
            if is_numpy_available()
            else raise_for_missing_dependency()
        )
    )


class BoolTensor(_tensor_type_base.TensorTypeBase):
    """A class to represent a boolean tensor type."""

    DTYPES = (
        (*TorchBoolTensor.DTYPES, *NumPyBoolTensor.DTYPES)
        if is_torch_available() and is_numpy_available()
        else (
            TorchBoolTensor.DTYPES
            if is_torch_available()
            else NumPyBoolTensor.DTYPES
            if is_numpy_available()
            else raise_for_missing_dependency()
        )
    )


class DoubleTensor(_tensor_type_base.TensorTypeBase):
    """A class to represent a double tensor type."""

    DTYPES = (
        (*TorchDoubleTensor.DTYPES, *NumPyDoubleTensor.DTYPES)
        if is_torch_available() and is_numpy_available()
        else (
            TorchDoubleTensor.DTYPES
            if is_torch_available()
            else NumPyDoubleTensor.DTYPES
            if is_numpy_available()
            else raise_for_missing_dependency()
        )
    )


__all__ = [
    "IntTensor",
    "FloatTensor",
    "BoolTensor",
    "DoubleTensor",
]

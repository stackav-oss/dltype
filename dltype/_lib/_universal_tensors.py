"""Tensor types work with either torch or numpy (maybe extended later)."""

from dltype._lib import _tensor_type_base
from dltype._lib._dependency_utilities import (
    is_torch_available,
    is_numpy_available,
    raise_for_missing_dependency,
)

if is_numpy_available():
    from dltype._lib._numpy_tensors import (
        Int8Tensor as NumPyInt8Tensor,
        Int16Tensor as NumPyInt16Tensor,
        Int32Tensor as NumPyInt32Tensor,
        Int64Tensor as NumPyInt64Tensor,
        IntTensor as NumPyIntTensor,
        HalfFloatTensor as NumPyHalfFloatTensor,
        Float16Tensor as NumPyFloat16Tensor,
        Float32Tensor as NumPyFloat32Tensor,
        Float64Tensor as NumPyFloat64Tensor,
        FloatTensor as NumpyFloatTensor,
        BoolTensor as NumPyBoolTensor,
    )

if is_torch_available():
    from dltype._lib._torch_tensors import (
        Int8Tensor as TorchInt8Tensor,
        Int16Tensor as TorchInt16Tensor,
        Int32Tensor as TorchInt32Tensor,
        Int64Tensor as TorchInt64Tensor,
        IntTensor as TorchIntTensor,
        HalfFloatTensor as TorchHalfFloatTensor,
        BFloat16Tensor as TorchBFloat16Tensor,
        Float16Tensor as TorchFloat16Tensor,
        Float32Tensor as TorchFloat32Tensor,
        Float64Tensor as TorchFloat64Tensor,
        FloatTensor as TorchFloatTensor,
        BoolTensor as TorchBoolTensor,
    )


class Int8Tensor(_tensor_type_base.TensorTypeBase):
    """A class to represent an 8-bit integer tensor type."""

    DTYPES = (
        (*TorchInt8Tensor.DTYPES, *NumPyInt8Tensor.DTYPES)
        if is_torch_available() and is_numpy_available()
        else (
            TorchInt8Tensor.DTYPES
            if is_torch_available()
            else NumPyInt8Tensor.DTYPES
            if is_numpy_available()
            else raise_for_missing_dependency()
        )
    )


class Int16Tensor(_tensor_type_base.TensorTypeBase):
    """A class to represent a 16-bit integer tensor type."""

    DTYPES = (
        (*TorchInt16Tensor.DTYPES, *NumPyInt16Tensor.DTYPES)
        if is_torch_available() and is_numpy_available()
        else (
            TorchInt16Tensor.DTYPES
            if is_torch_available()
            else NumPyInt16Tensor.DTYPES
            if is_numpy_available()
            else raise_for_missing_dependency()
        )
    )


class Int32Tensor(_tensor_type_base.TensorTypeBase):
    """A class to represent a 32-bit integer tensor type."""

    DTYPES = (
        (*TorchInt32Tensor.DTYPES, *NumPyInt32Tensor.DTYPES)
        if is_torch_available() and is_numpy_available()
        else (
            TorchInt32Tensor.DTYPES
            if is_torch_available()
            else NumPyInt32Tensor.DTYPES
            if is_numpy_available()
            else raise_for_missing_dependency()
        )
    )


class Int64Tensor(_tensor_type_base.TensorTypeBase):
    """A class to represent a 64-bit integer tensor type."""

    DTYPES = (
        (*TorchInt64Tensor.DTYPES, *NumPyInt64Tensor.DTYPES)
        if is_torch_available() and is_numpy_available()
        else (
            TorchInt64Tensor.DTYPES
            if is_torch_available()
            else NumPyInt64Tensor.DTYPES
            if is_numpy_available()
            else raise_for_missing_dependency()
        )
    )


class IntTensor(_tensor_type_base.TensorTypeBase):
    """A class to represent an integer tensor of any precision (8 bit, 16 bit, 32 bit and 64 bit)."""

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


class HalfFloatTensor(_tensor_type_base.TensorTypeBase):
    """A class to represent half precision tensor types. Does not include special types such as bfloat16.

    Use this type if bfloat16 causes issues for some reason and you need to prohibit its use."""

    DTYPES = (
        (*TorchHalfFloatTensor.DTYPES, *NumPyHalfFloatTensor.DTYPES)
        if is_torch_available() and is_numpy_available()
        else (
            TorchHalfFloatTensor.DTYPES
            if is_torch_available()
            else NumPyHalfFloatTensor.DTYPES
            if is_numpy_available()
            else raise_for_missing_dependency()
        )
    )


class BFloat16Tensor(_tensor_type_base.TensorTypeBase):
    """A tensor that can only be bfloat16."""

    DTYPES = TorchBFloat16Tensor.DTYPES if is_torch_available() else ()


class Float16Tensor(_tensor_type_base.TensorTypeBase):
    """A class to represent any 16-bit float tensor types (includes bfloat16)."""

    DTYPES = (
        (*TorchFloat16Tensor.DTYPES, *NumPyFloat16Tensor.DTYPES)
        if is_torch_available() and is_numpy_available()
        else (
            TorchFloat16Tensor.DTYPES
            if is_torch_available()
            else NumPyFloat16Tensor.DTYPES
            if is_numpy_available()
            else raise_for_missing_dependency()
        )
    )


class Float32Tensor(_tensor_type_base.TensorTypeBase):
    """A class to represent a 32-bit float tensor type."""

    DTYPES = (
        (*TorchFloat32Tensor.DTYPES, *NumPyFloat32Tensor.DTYPES)
        if is_torch_available() and is_numpy_available()
        else (
            TorchFloat32Tensor.DTYPES
            if is_torch_available()
            else NumPyFloat32Tensor.DTYPES
            if is_numpy_available()
            else raise_for_missing_dependency()
        )
    )


class Float64Tensor(_tensor_type_base.TensorTypeBase):
    """A class to represent a double tensor type."""

    DTYPES = (
        (*TorchFloat64Tensor.DTYPES, *NumPyFloat64Tensor.DTYPES)
        if is_torch_available() and is_numpy_available()
        else (
            TorchFloat64Tensor.DTYPES
            if is_torch_available()
            else NumPyFloat64Tensor.DTYPES
            if is_numpy_available()
            else raise_for_missing_dependency()
        )
    )


DoubleTensor = Float64Tensor


class FloatTensor(_tensor_type_base.TensorTypeBase):
    """A class to represent the superset of any floating point type of any precision.

    This includes 16 bit, 32 bit, 64 bit, and optionally numpy's 128 bit if it is supported."""

    DTYPES = (
        (*TorchFloatTensor.DTYPES, *NumpyFloatTensor.DTYPES)
        if is_torch_available() and is_numpy_available()
        else (
            TorchFloatTensor.DTYPES
            if is_torch_available()
            else NumpyFloatTensor.DTYPES
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


__all__ = [
    "Int8Tensor",
    "Int16Tensor",
    "Int32Tensor",
    "Int64Tensor",
    "IntTensor",
    "HalfFloatTensor",
    "BFloat16Tensor",
    "Float16Tensor",
    "Float32Tensor",
    "Float64Tensor",
    "DoubleTensor",
    "FloatTensor",
    "BoolTensor",
]

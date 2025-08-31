"""Tensor types work with either torch or numpy (maybe extended later)."""

from dltype._lib import _tensor_type_base
from dltype._lib._dependency_utilities import (
    is_numpy_available,
    is_torch_available,
    raise_for_missing_dependency,
)

if is_numpy_available():
    from dltype._lib._numpy_tensors import (
        BoolTensor as NumPyBoolTensor,
    )
    from dltype._lib._numpy_tensors import (
        Float16Tensor as NumPyFloat16Tensor,
    )
    from dltype._lib._numpy_tensors import (
        Float32Tensor as NumPyFloat32Tensor,
    )
    from dltype._lib._numpy_tensors import (
        Float64Tensor as NumPyFloat64Tensor,
    )
    from dltype._lib._numpy_tensors import (
        FloatTensor as NumpyFloatTensor,
    )
    from dltype._lib._numpy_tensors import (
        IEEE754HalfFloatTensor as NumPyIEEE754HalfFloatTensor,
    )
    from dltype._lib._numpy_tensors import (
        Int8Tensor as NumPyInt8Tensor,
    )
    from dltype._lib._numpy_tensors import (
        Int16Tensor as NumPyInt16Tensor,
    )
    from dltype._lib._numpy_tensors import (
        Int32Tensor as NumPyInt32Tensor,
    )
    from dltype._lib._numpy_tensors import (
        Int64Tensor as NumPyInt64Tensor,
    )
    from dltype._lib._numpy_tensors import (
        IntTensor as NumPyIntTensor,
    )
    from dltype._lib._numpy_tensors import (
        SignedIntTensor as NumPySignedIntTensor,
    )
    from dltype._lib._numpy_tensors import (
        UInt8Tensor as NumPyUInt8Tensor,
    )
    from dltype._lib._numpy_tensors import (
        UInt16Tensor as NumPyUInt16Tensor,
    )
    from dltype._lib._numpy_tensors import (
        UInt32Tensor as NumPyUInt32Tensor,
    )
    from dltype._lib._numpy_tensors import (
        UInt64Tensor as NumPyUInt64Tensor,
    )
    from dltype._lib._numpy_tensors import (
        UnsignedIntTensor as NumPyUnsignedIntTensor,
    )

if is_torch_available():
    from dltype._lib._torch_tensors import (
        BFloat16Tensor as TorchBFloat16Tensor,
    )
    from dltype._lib._torch_tensors import (
        BoolTensor as TorchBoolTensor,
    )
    from dltype._lib._torch_tensors import (
        Float16Tensor as TorchFloat16Tensor,
    )
    from dltype._lib._torch_tensors import (
        Float32Tensor as TorchFloat32Tensor,
    )
    from dltype._lib._torch_tensors import (
        Float64Tensor as TorchFloat64Tensor,
    )
    from dltype._lib._torch_tensors import (
        FloatTensor as TorchFloatTensor,
    )
    from dltype._lib._torch_tensors import (
        IEEE754HalfFloatTensor as TorchIEEE754HalfFloatTensor,
    )
    from dltype._lib._torch_tensors import (
        Int8Tensor as TorchInt8Tensor,
    )
    from dltype._lib._torch_tensors import (
        Int16Tensor as TorchInt16Tensor,
    )
    from dltype._lib._torch_tensors import (
        Int32Tensor as TorchInt32Tensor,
    )
    from dltype._lib._torch_tensors import (
        Int64Tensor as TorchInt64Tensor,
    )
    from dltype._lib._torch_tensors import (
        IntTensor as TorchIntTensor,
    )
    from dltype._lib._torch_tensors import (
        SignedIntTensor as TorchSignedIntTensor,
    )
    from dltype._lib._torch_tensors import (
        UInt8Tensor as TorchUInt8Tensor,
    )
    from dltype._lib._torch_tensors import (
        UInt16Tensor as TorchUInt16Tensor,
    )
    from dltype._lib._torch_tensors import (
        UInt32Tensor as TorchUInt32Tensor,
    )
    from dltype._lib._torch_tensors import (
        UInt64Tensor as TorchUInt64Tensor,
    )
    from dltype._lib._torch_tensors import (
        UnsignedIntTensor as TorchUnsignedIntTensor,
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


class UInt8Tensor(_tensor_type_base.TensorTypeBase):
    """A class to represent an unsigned 8-bit integer tensor type."""

    DTYPES = (
        (*TorchUInt8Tensor.DTYPES, *NumPyUInt8Tensor.DTYPES)
        if is_torch_available() and is_numpy_available()
        else (
            TorchUInt8Tensor.DTYPES
            if is_torch_available()
            else NumPyUInt8Tensor.DTYPES
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


class UInt16Tensor(_tensor_type_base.TensorTypeBase):
    """A class to represent an unsigned 16-bit integer tensor type."""

    DTYPES = (
        (*TorchUInt16Tensor.DTYPES, *NumPyUInt16Tensor.DTYPES)
        if is_torch_available() and is_numpy_available()
        else (
            TorchUInt16Tensor.DTYPES
            if is_torch_available()
            else NumPyUInt16Tensor.DTYPES
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


class UInt32Tensor(_tensor_type_base.TensorTypeBase):
    """A class to represent an unsigned 32-bit integer tensor type."""

    DTYPES = (
        (*TorchUInt32Tensor.DTYPES, *NumPyUInt32Tensor.DTYPES)
        if is_torch_available() and is_numpy_available()
        else (
            TorchUInt32Tensor.DTYPES
            if is_torch_available()
            else NumPyUInt32Tensor.DTYPES
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


class UInt64Tensor(_tensor_type_base.TensorTypeBase):
    """A class to represent an unsigned 64-bit integer tensor type."""

    DTYPES = (
        (*TorchUInt64Tensor.DTYPES, *NumPyUInt64Tensor.DTYPES)
        if is_torch_available() and is_numpy_available()
        else (
            TorchUInt64Tensor.DTYPES
            if is_torch_available()
            else NumPyUInt64Tensor.DTYPES
            if is_numpy_available()
            else raise_for_missing_dependency()
        )
    )


class SignedIntTensor(_tensor_type_base.TensorTypeBase):
    """A class to represent an integer tensor of any precision (8 bit, 16 bit, 32 bit and 64 bit)."""

    DTYPES = (
        (*TorchSignedIntTensor.DTYPES, *NumPySignedIntTensor.DTYPES)
        if is_torch_available() and is_numpy_available()
        else (
            TorchIntTensor.DTYPES
            if is_torch_available()
            else NumPySignedIntTensor.DTYPES
            if is_numpy_available()
            else raise_for_missing_dependency()
        )
    )


class UnsignedIntTensor(_tensor_type_base.TensorTypeBase):
    """A class to represent an unsigned integer tensor of any precision (8 bit, 16 bit, 32 bit and 64 bit)."""

    DTYPES = (
        (*TorchUnsignedIntTensor.DTYPES, *NumPyUnsignedIntTensor.DTYPES)
        if is_torch_available() and is_numpy_available()
        else (
            TorchUnsignedIntTensor.DTYPES
            if is_torch_available()
            else NumPyUnsignedIntTensor.DTYPES
            if is_numpy_available()
            else raise_for_missing_dependency()
        )
    )


class IntTensor(_tensor_type_base.TensorTypeBase):
    """A class to represent an integer tensor (signed or unsigned) of any precision (8 bit, 16 bit, 32 bit and 64 bit)."""

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


class IEEE754HalfFloatTensor(_tensor_type_base.TensorTypeBase):
    """A class to represent half precision tensor types. Does not include special types such as bfloat16.

    Use this type if bfloat16 causes issues for some reason and you need to prohibit its use."""

    DTYPES = (
        (*TorchIEEE754HalfFloatTensor.DTYPES, *NumPyIEEE754HalfFloatTensor.DTYPES)
        if is_torch_available() and is_numpy_available()
        else (
            TorchIEEE754HalfFloatTensor.DTYPES
            if is_torch_available()
            else NumPyIEEE754HalfFloatTensor.DTYPES
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
    "UInt8Tensor",
    "UInt16Tensor",
    "UInt32Tensor",
    "UInt64Tensor",
    "Int8Tensor",
    "Int16Tensor",
    "Int32Tensor",
    "Int64Tensor",
    "UnsignedIntTensor",
    "SignedIntTensor",
    "IntTensor",
    "IEEE754HalfFloatTensor",
    "BFloat16Tensor",
    "Float16Tensor",
    "Float32Tensor",
    "Float64Tensor",
    "DoubleTensor",
    "FloatTensor",
    "BoolTensor",
]

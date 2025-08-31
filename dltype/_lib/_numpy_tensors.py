"""Numpy-only tensor types for DLType."""

import numpy as np

from dltype._lib._dependency_utilities import (
    is_np_float128_available,
    is_np_longdouble_available,
)
from dltype._lib._tensor_type_base import TensorTypeBase


class UInt8Tensor(TensorTypeBase):
    """A class to represent an 8-bit unsigned integer tensor type."""

    DTYPES = (np.uint8,)


class UInt16Tensor(TensorTypeBase):
    """A class to represent an 16-bit unsigned integer tensor type."""

    DTYPES = (np.uint16,)


class UInt32Tensor(TensorTypeBase):
    """A class to represent an 32-bit unsigned integer tensor type."""

    DTYPES = (np.uint32,)


class UInt64Tensor(TensorTypeBase):
    """A class to represent an 64-bit unsigned integer tensor type."""

    DTYPES = (np.uint64,)


class Int8Tensor(TensorTypeBase):
    """A class to represent an 8-bit integer tensor type."""

    DTYPES = (np.int8,)


class Int16Tensor(TensorTypeBase):
    """A class to represent a 16-bit integer tensor type."""

    DTYPES = (np.int16,)


class Int32Tensor(TensorTypeBase):
    """A class to represent a 32-bit integer tensor type."""

    DTYPES = (np.int32,)


class Int64Tensor(TensorTypeBase):
    """A class to represent a 64-bit integer tensor type."""

    DTYPES = (np.int64,)


class SignedIntTensor(TensorTypeBase):
    """A class to represent any signed integer tensor type of any size (8 bit, 16 bit, 32 bit, and 64 bit)."""

    DTYPES = (
        *Int8Tensor.DTYPES,
        *Int16Tensor.DTYPES,
        *Int32Tensor.DTYPES,
        *Int64Tensor.DTYPES,
    )


class UnsignedIntTensor(TensorTypeBase):
    """A class to represent any unsigned integer tensor type of any size (8 bit, 16 bit, 32 bit, and 64 bit)."""

    DTYPES = (
        *UInt8Tensor.DTYPES,
        *UInt16Tensor.DTYPES,
        *UInt32Tensor.DTYPES,
        *UInt64Tensor.DTYPES,
    )


class IntTensor(TensorTypeBase):
    """A class to represent any integer tensor type of any size (8 bit, 16 bit, 32 bit, and 64 bit)."""

    DTYPES = (
        *SignedIntTensor.DTYPES,
        *UnsignedIntTensor.DTYPES,
    )


class IEEE754HalfFloatTensor(TensorTypeBase):
    """A dtype for 16 bit half-precision floats that comply with the IEE 754 specification."""

    DTYPES = (np.float16,)


# Note that numpy does not support non IEEE754 compliant 16 bit floating types such as bfloat16
Float16Tensor = IEEE754HalfFloatTensor


class Float32Tensor(TensorTypeBase):
    """A class to represent any 32 bit floating point type."""

    DTYPES = (np.float32,)


class Float64Tensor(TensorTypeBase):
    """A class to represent a double tensor type."""

    DTYPES = (np.float64,)


DoubleTensor = Float64Tensor


class Float128Tensor(TensorTypeBase):
    """A class to represent 128 bit floating point type."""

    DTYPES = (np.float128,) if is_np_float128_available() else ()


class LongDoubleTensor(TensorTypeBase):
    """A class to represent long double floating point type."""

    DTYPES = (np.longdouble,) if is_np_longdouble_available() else ()


class FloatTensor(TensorTypeBase):
    """A class to represent any floating point tensor type of any size (16 bit, 32 bit, 64 bit, and optionally 128 bit)."""

    # Build DTYPES list based on available types
    DTYPES = (
        # Add standard floating point types (16, 32, 64 bits)
        *Float16Tensor.DTYPES,
        *Float32Tensor.DTYPES,
        *DoubleTensor.DTYPES,
        # Add optional 128-bit and longdouble types if available
        *Float128Tensor.DTYPES,
        *LongDoubleTensor.DTYPES,
    )


class BoolTensor(TensorTypeBase):
    """A class to represent a boolean tensor type."""

    DTYPES = (np.bool_,)

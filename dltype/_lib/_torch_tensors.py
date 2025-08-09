"""Torch tensors for DLType."""

import torch

from dltype._lib._tensor_type_base import TensorTypeBase


class Int8Tensor(TensorTypeBase):
    """A class to represent an 8-bit integer tensor type."""

    DTYPES = (torch.int8,)


class Int16Tensor(TensorTypeBase):
    """A class to represent a 16-bit integer tensor type."""

    DTYPES = (torch.int16,)


class Int32Tensor(TensorTypeBase):
    """A class to represent a 32-bit integer tensor type."""

    DTYPES = (torch.int32,)


class Int64Tensor(TensorTypeBase):
    """A class to represent a 64-bit integer tensor type."""

    DTYPES = (torch.int64,)


class IntTensor(TensorTypeBase):
    """A class to represent any integer tensor type of any size (8 bit, 16 bit, 32 bit, and 64 bit)."""

    DTYPES = (
        *Int8Tensor.DTYPES,
        *Int16Tensor.DTYPES,
        *Int32Tensor.DTYPES,
        *Int64Tensor.DTYPES,
    )


class BFloat16Tensor(TensorTypeBase):
    """A class to represent bfloat16."""

    DTYPES = (torch.bfloat16,)


class HalfFloatTensor(TensorTypeBase):
    """A dtype for standard 16 bit half-floats."""

    DTYPES = (
        torch.half,
        torch.float16,
    )


class Float16Tensor(TensorTypeBase):
    """A class to represent any 16 bit floating point type (includes regular float16 as well as bfloat16)."""

    DTYPES = (*HalfFloatTensor.DTYPES, *BFloat16Tensor.DTYPES)


class Float32Tensor(TensorTypeBase):
    """A class to represent any 32 bit floating point type."""

    DTYPES = (
        torch.float,
        torch.float32,
    )


class DoubleTensor(TensorTypeBase):
    """A class to represent a double tensor type."""

    DTYPES = (torch.double, torch.float64)


Float64Tensor = DoubleTensor


class FloatTensor(TensorTypeBase):
    """A class to represent any floating point tensor type of any size (16 bit, 32 bit, and 64 bit)."""

    DTYPES = (
        *Float16Tensor.DTYPES,
        *Float32Tensor.DTYPES,
        *Float64Tensor.DTYPES,
    )


class BoolTensor(TensorTypeBase):
    """A class to represent a boolean tensor type."""

    DTYPES = (torch.bool,)

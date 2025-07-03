"""Torch tensors for DLType."""

import torch

from dltype._lib._tensor_type_base import TensorTypeBase


class IntTensor(TensorTypeBase):
    """A class to represent an integer tensor type."""

    DTYPES = (torch.int, torch.int8, torch.int16, torch.int32, torch.int64)


class FloatTensor(TensorTypeBase):
    """A class to represent a float tensor type."""

    DTYPES = (
        torch.float,
        torch.float16,
        torch.float32,
        torch.float64,
        torch.half,
        torch.bfloat16,
        torch.double,
    )


class BoolTensor(TensorTypeBase):
    """A class to represent a boolean tensor type."""

    DTYPES = (torch.bool,)


class DoubleTensor(TensorTypeBase):
    """A class to represent a double tensor type."""

    DTYPES = (torch.double, torch.float64)

"""Numpy-only tensor types for DLType."""

import numpy as np

from dltype._lib._tensor_type_base import TensorTypeBase


class IntTensor(TensorTypeBase):
    """A class to represent an integer tensor type."""

    DTYPES = (np.int_, np.int8, np.int16, np.int32, np.int64)


class FloatTensor(TensorTypeBase):
    """A class to represent a float tensor type."""

    DTYPES = (
        np.float64,
        np.float16,
        np.float32,
        np.float64,
        np.float128,
        np.half,
        np.longdouble,
    )


class BoolTensor(TensorTypeBase):
    """A class to represent a boolean tensor type."""

    DTYPES = (np.bool_,)


class DoubleTensor(TensorTypeBase):
    """A class to represent a double tensor type."""

    DTYPES = (np.float64,)

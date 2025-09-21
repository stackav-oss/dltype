"""A fast, lightweight runtime type checker for torch tensors and numpy arrays."""

from dltype._lib._constants import (
    DEBUG_MODE,
    MAX_ACCEPTABLE_EVALUATION_TIME_NS,
)
from dltype._lib._core import (
    DLTypeScopeProvider,
    dltyped,
    dltyped_dataclass,
    dltyped_namedtuple,
)
from dltype._lib._dependency_utilities import (
    is_numpy_available,
    is_torch_available,
    raise_for_missing_dependency,
)
from dltype._lib._dtypes import SUPPORTED_TENSOR_TYPES
from dltype._lib._errors import (
    DLTypeDtypeError,
    DLTypeDuplicateError,
    DLTypeError,
    DLTypeInvalidReferenceError,
    DLTypeNDimsError,
    DLTypeScopeProviderError,
    DLTypeShapeError,
    DLTypeUnsupportedTensorTypeError,
)
from dltype._lib._tensor_type_base import (
    TensorTypeBase,
)

if is_torch_available() and is_numpy_available():
    from dltype._lib._universal_tensors import (
        BFloat16Tensor,
        BoolTensor,
        DoubleTensor,
        Float16Tensor,
        Float32Tensor,
        Float64Tensor,
        FloatTensor,
        IEEE754HalfFloatTensor,
        Int8Tensor,
        Int16Tensor,
        Int32Tensor,
        Int64Tensor,
        IntTensor,
        SignedIntTensor,
        UInt8Tensor,
        UInt16Tensor,
        UInt32Tensor,
        UInt64Tensor,
        UnsignedIntTensor,
    )
elif is_torch_available():
    from dltype._lib._torch_tensors import (
        BFloat16Tensor,
        BoolTensor,
        DoubleTensor,
        Float16Tensor,
        Float32Tensor,
        Float64Tensor,
        FloatTensor,
        IEEE754HalfFloatTensor,
        Int8Tensor,
        Int16Tensor,
        Int32Tensor,
        Int64Tensor,
        IntTensor,
        SignedIntTensor,
        UInt8Tensor,
        UInt16Tensor,
        UInt32Tensor,
        UInt64Tensor,
        UnsignedIntTensor,
    )
elif is_numpy_available():
    from dltype._lib._numpy_tensors import (
        BoolTensor,
        DoubleTensor,
        Float16Tensor,
        Float32Tensor,
        Float64Tensor,
        FloatTensor,
        IEEE754HalfFloatTensor,
        Int8Tensor,
        Int16Tensor,
        Int32Tensor,
        Int64Tensor,
        IntTensor,
        SignedIntTensor,
        UInt8Tensor,
        UInt16Tensor,
        UInt32Tensor,
        UInt64Tensor,
        UnsignedIntTensor,
    )

    BFloat16Tensor = None
else:
    raise_for_missing_dependency()


__all__ = [
    "DEBUG_MODE",
    "MAX_ACCEPTABLE_EVALUATION_TIME_NS",
    "SUPPORTED_TENSOR_TYPES",
    "BFloat16Tensor",
    "BFloat16Tensor",
    "BoolTensor",
    "DLTypeDtypeError",
    "DLTypeDuplicateError",
    "DLTypeError",
    "DLTypeInvalidReferenceError",
    "DLTypeNDimsError",
    "DLTypeScopeProvider",
    "DLTypeScopeProviderError",
    "DLTypeShapeError",
    "DLTypeUnsupportedTensorTypeError",
    "DoubleTensor",
    "Float16Tensor",
    "Float32Tensor",
    "Float64Tensor",
    "FloatTensor",
    "IEEE754HalfFloatTensor",
    "Int8Tensor",
    "Int16Tensor",
    "Int32Tensor",
    "Int64Tensor",
    "IntTensor",
    "SignedIntTensor",
    "TensorTypeBase",
    "UInt8Tensor",
    "UInt16Tensor",
    "UInt32Tensor",
    "UInt64Tensor",
    "UnsignedIntTensor",
    "dltyped",
    "dltyped_dataclass",
    "dltyped_namedtuple",
]

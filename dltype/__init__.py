from dltype._lib._core import (
    DLTypeScopeProvider,
    dltyped,
    dltyped_namedtuple,
    dltyped_dataclass,
)

from dltype._lib._constants import (
    DEBUG_MODE,
    MAX_ACCEPTABLE_EVALUATION_TIME_NS,
)

from dltype._lib._tensor_type_base import (
    SUPPORTED_TENSOR_TYPES,
    TensorTypeBase,
)

from dltype._lib._errors import (
    DLTypeError,
    DLTypeShapeError,
    DLTypeScopeProviderError,
    DLTypeInvalidReferenceError,
    DLTypeDuplicateError,
    DLTypeDtypeError,
)
from dltype._lib._dependency_utilities import (
    is_torch_available,
    is_numpy_available,
    raise_for_missing_dependency,
)

if is_torch_available() and is_numpy_available():
    from dltype._lib._universal_tensors import (
        UInt8Tensor,
        UInt16Tensor,
        UInt32Tensor,
        UInt64Tensor,
        Int8Tensor,
        Int16Tensor,
        Int32Tensor,
        Int64Tensor,
        SignedIntTensor,
        UnsignedIntTensor,
        IntTensor,
        Float16Tensor,
        IEEE754HalfFloatTensor,
        BFloat16Tensor,
        Float32Tensor,
        Float64Tensor,
        FloatTensor,
        BoolTensor,
        DoubleTensor,
    )
elif is_torch_available():
    from dltype._lib._torch_tensors import (
        UInt8Tensor,
        UInt16Tensor,
        UInt32Tensor,
        UInt64Tensor,
        Int8Tensor,
        Int16Tensor,
        Int32Tensor,
        Int64Tensor,
        SignedIntTensor,
        UnsignedIntTensor,
        IntTensor,
        Float16Tensor,
        IEEE754HalfFloatTensor,
        BFloat16Tensor,
        Float32Tensor,
        Float64Tensor,
        FloatTensor,
        BoolTensor,
        DoubleTensor,
    )
elif is_numpy_available():
    from dltype._lib._numpy_tensors import (
        UInt8Tensor,
        UInt16Tensor,
        UInt32Tensor,
        UInt64Tensor,
        Int8Tensor,
        Int16Tensor,
        Int32Tensor,
        Int64Tensor,
        SignedIntTensor,
        UnsignedIntTensor,
        IntTensor,
        Float16Tensor,
        IEEE754HalfFloatTensor,
        Float32Tensor,
        Float64Tensor,
        FloatTensor,
        BoolTensor,
        DoubleTensor,
    )

    BFloat16Tensor = None
else:
    raise_for_missing_dependency()


__all__ = [
    "DEBUG_MODE",
    "MAX_ACCEPTABLE_EVALUATION_TIME_NS",
    "SUPPORTED_TENSOR_TYPES",
    "DLTypeError",
    "DLTypeShapeError",
    "DLTypeScopeProviderError",
    "DLTypeInvalidReferenceError",
    "DLTypeDuplicateError",
    "DLTypeDtypeError",
    "DLTypeScopeProvider",
    "TensorTypeBase",
    "dltyped",
    "dltyped_namedtuple",
    "dltyped_dataclass",
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
    "Float16Tensor",
    "BFloat16Tensor",
    "IEEE754HalfFloatTensor",
    "BFloat16Tensor",
    "Float32Tensor",
    "Float64Tensor",
    "FloatTensor",
    "BoolTensor",
    "DoubleTensor",
]

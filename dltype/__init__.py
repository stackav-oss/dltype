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
from dltype._lib._errors import (
    DLTypeDtypeError,
    DLTypeDuplicateError,
    DLTypeError,
    DLTypeInvalidReferenceError,
    DLTypeScopeProviderError,
    DLTypeShapeError,
    DLTypeNDimsError,
)
from dltype._lib._tensor_type_base import (
    TensorTypeBase,
)
from dltype._lib._dtypes import SUPPORTED_TENSOR_TYPES

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
    "DLTypeError",
    "DLTypeShapeError",
    "DLTypeScopeProviderError",
    "DLTypeInvalidReferenceError",
    "DLTypeDuplicateError",
    "DLTypeDtypeError",
    "DLTypeNDimsError",
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

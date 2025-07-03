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
    print("BOTH TORCH AND NUMPY ARE AVAILABLE")
    from dltype._lib._universal_tensors import (
        IntTensor,
        FloatTensor,
        BoolTensor,
        DoubleTensor,
    )
elif is_numpy_available():
    print("ONLY NUMPY IS AVAILABLE")
    from dltype._lib._numpy_tensors import (
        IntTensor,
        FloatTensor,
        BoolTensor,
        DoubleTensor,
    )
elif is_torch_available():
    print("ONLY TORCH IS AVAILABLE")
    from dltype._lib._torch_tensors import (
        IntTensor,
        FloatTensor,
        BoolTensor,
        DoubleTensor,
    )
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
    "IntTensor",
    "FloatTensor",
    "BoolTensor",
    "DoubleTensor",
]

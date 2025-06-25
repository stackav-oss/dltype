"""A module to assist with using Annotated[torch.Tensor] in type hints."""

from __future__ import annotations

import inspect
import logging
import time
import warnings
from collections import deque
from functools import wraps
from typing import (
    TYPE_CHECKING,
    Annotated,
    Any,
    ClassVar,
    Final,
    Literal,
    NamedTuple,
    ParamSpec,
    Protocol,
    TypeAlias,
    TypeVar,
    Union,  # pyright: ignore[reportDeprecated] # TODO(DX-2313): Address pyright errors ignored to migrate from mypy # fmt: skip
    cast,
    get_args,
    get_origin,
    get_type_hints,
    runtime_checkable,
)

import numpy as np
import numpy.typing as npt
import torch
from pydantic_core import core_schema
from typing_extensions import override

from dltype._lib._parser import (
    DLTypeDimensionExpression,
    DLTypeModifier,
    expression_from_string,
)

if TYPE_CHECKING:
    from collections.abc import Callable

    from pydantic import GetCoreSchemaHandler, ValidationInfo

_logger: Final = logging.getLogger(__name__)

P = ParamSpec("P")
R = TypeVar("R")

# Constants
PYDANTIC_INFO_KEY: Final = "__dltype__"
EvaluatedDimensionT: TypeAlias = dict[str, int]
DEBUG_MODE: Final = False
MAX_ACCEPTABLE_EVALUATION_TIME_NS: Final = int(5e9)  # 5ms
DLtypeTensorT: TypeAlias = torch.Tensor | npt.NDArray[Any]
SUPPORTED_TENSOR_TYPES: Final = {torch.Tensor, np.ndarray}


def _maybe_warn_runtime(runtime_ns: int) -> None:
    return runtime_ns > MAX_ACCEPTABLE_EVALUATION_TIME_NS


class DLTypeError(TypeError):
    """An error raised when a type assertion is hit."""


class DLTypeShapeError(DLTypeError):
    """An error raised when a shape assertion is hit."""


class DLTypeDtypeError(DLTypeError):
    """An error raised when a dtype assertion is hit."""


class DLTypeDuplicateError(DLTypeError):
    """An error raised when a duplicate tensor name is hit."""


class DLTypeInvalidReferenceError(DLTypeError):
    """An error raised when an invalid reference is hit."""


class DLTypeScopeProviderError(DLTypeError):
    """An error raised when an invalid scope provider is hit."""


class _DLTypeAnnotation(NamedTuple):
    """A class representing a type annotation for a tensor."""

    tensor_type_hint: type[DLtypeTensorT]
    dltype_annotation: TensorTypeBase

    @classmethod
    def from_hint(
        cls, hint: type | None, optional: bool = False
    ) -> _DLTypeAnnotation | None:
        """Create a new _DLTypeAnnotation from a type hint."""
        if hint is None:
            return None

        _logger.debug("Creating DLType from hint %r", hint)
        n_expected_args = len(cls._fields)
        origin = get_origin(hint)
        args = get_args(hint)

        # Handle Optional[T] types (Union[T, None] or Union[T, NoneType])
        if origin is Union:  # pyright: ignore[reportDeprecated] # TODO(DX-2313): Address pyright errors ignored to migrate from mypy # fmt: skip
            # Get the non-None type from the Union
            non_none_types = [t for t in args if t not in {type(None), None}]

            # Only support Optional[T], not general Union types
            if len(non_none_types) != 1:
                msg = f"Only Optional tensor types are supported, not general Union types. Got: {hint}"
                raise TypeError(msg)

            # Recursively process the non-None type with optional=True
            return cls.from_hint(non_none_types[0], optional=True)

        # Only process Annotated types
        if origin is not Annotated:
            return None

        # Ensure the annotation is a TensorTypeBase
        if len(args) < n_expected_args or not isinstance(args[1], TensorTypeBase):
            _logger.warning(
                "Invalid annotated dltype hint: %r",
                args[1:] if len(args) >= n_expected_args else None,
            )
            return None

        # Ensure the base type is a supported tensor type
        tensor_type, dltype_hint = args[0], args[1]
        if not any(T in tensor_type.mro() for T in SUPPORTED_TENSOR_TYPES):
            msg = f"Invalid base type=<{tensor_type}> in DLType hint, expected a subtype of {SUPPORTED_TENSOR_TYPES}"
            raise TypeError(msg)

        dltype_hint.optional = optional
        return cls(tensor_type_hint=tensor_type, dltype_annotation=dltype_hint)


class _ConcreteType(NamedTuple):
    """A class containing a tensor name, a tensor value, and its type."""

    tensor_arg_name: str
    tensor: DLtypeTensorT
    dltype_annotation: TensorTypeBase

    def get_expected_shape(
        self, tensor: DLtypeTensorT
    ) -> tuple[DLTypeDimensionExpression, ...]:
        """Get the expected shape of the tensor.

        We handle multi-axis dimensions by replacing the multi-axis placeholder with the actual shape.
        """
        expected_shape = list(self.dltype_annotation.expected_shape)

        if self.dltype_annotation.multiaxis_index is not None:
            _logger.debug(
                "Replacing multiaxis dimension %r with actual shape %r",
                self.dltype_annotation.multiaxis_index,
                tensor.shape,
            )
            # for multiaxis dimensions, we replace the values in the expected shape with the actual shape for every
            # dimension that is not the multiaxis dimension
            actual_shape = tensor.shape

            multi_axis_offset = len(actual_shape) - len(expected_shape) + 1

            expected_shape.pop(self.dltype_annotation.multiaxis_index)
            for i in range(multi_axis_offset):
                expected_shape.insert(
                    self.dltype_annotation.multiaxis_index + i,
                    DLTypeDimensionExpression.from_multiaxis_literal(
                        f"{self.dltype_annotation.multiaxis_name}[{i}]",
                        actual_shape[self.dltype_annotation.multiaxis_index + i],
                        is_anonymous=self.dltype_annotation.anonymous_multiaxis,
                    ),
                )

        return tuple(expected_shape)


@runtime_checkable
class DLTypeScopeProvider(Protocol):
    """A protocol for classes that provide a scope for DLTypeDimensionExpression evaluation."""

    def get_dltype_scope(self) -> EvaluatedDimensionT:
        """Get the current scope of variables for the DLTypeDimensionExpression evaluation."""
        ...


class _DLTypeContext:
    """A class representing the current context for type hints.

    Keeps track of a simple mapping of names to expected shapes and types.

    This context can be evaluated at any time to check if the current actual shapes and types match the expected ones.

    We evaluate in a first-come-first-correct manner where the first tensor of a given name is considered the correct one.
    """

    def __init__(self) -> None:
        """Create a new DLTypeContext."""
        self._hinted_tensors: deque[_ConcreteType] = deque()
        # mapping of dimension -> shape
        self.tensor_shape_map: EvaluatedDimensionT = {}
        # mapping of tensor name -> tensor type, used to check for duplicates
        self.registered_tensor_dtypes: dict[str, torch.dtype | npt.DTypeLike] = {}

    def add(self, name: str, tensor: Any, dltype_annotation: TensorTypeBase) -> None:  # noqa: ANN401
        """Add a tensor to the context."""
        if dltype_annotation.optional and tensor is None:
            # skip optional tensors
            return
        if not isinstance(tensor, torch.Tensor | np.ndarray):
            msg = f"Invalid type {type(tensor)}"
            raise DLTypeError(msg)
        self._hinted_tensors.append(_ConcreteType(name, tensor, dltype_annotation))

    def assert_context(self) -> None:
        """Considering the current context, check if all tensors match their expected types."""
        __tracebackhide__ = not DEBUG_MODE

        start_t = time.perf_counter_ns()

        try:
            while self._hinted_tensors:
                tensor_context = self._hinted_tensors.popleft()
                # first check if the tensor could possibly have the right shape
                tensor_context.dltype_annotation.check(tensor_context.tensor)

                if tensor_context.tensor_arg_name in self.registered_tensor_dtypes:
                    msg = f"[tensor={tensor_context.tensor_arg_name=}] Duplicate tensor name in type checking context!"
                    raise DLTypeDuplicateError(msg)

                self.registered_tensor_dtypes[tensor_context.tensor_arg_name] = (
                    tensor_context.tensor.dtype
                )
                expected_shape = tensor_context.get_expected_shape(
                    tensor_context.tensor
                )
                self._assert_tensor_shape(
                    tensor_context.tensor_arg_name,
                    expected_shape,
                    tensor_context.tensor,
                )

        finally:
            end_t = time.perf_counter_ns()
            runtime_ns = end_t - start_t
            _logger.debug("Context evaluation took %d ns", runtime_ns)
            if _maybe_warn_runtime(runtime_ns):
                warnings.warn(
                    f"Type checking took longer than expected {(runtime_ns) / 1e6:.2f}ms > {MAX_ACCEPTABLE_EVALUATION_TIME_NS / 1e6}ms",
                    UserWarning,
                    stacklevel=2,
                )

    def _assert_tensor_shape(
        self,
        tensor_arg_name: str,
        expected_shape: tuple[DLTypeDimensionExpression, ...],
        tensor: DLtypeTensorT,
    ) -> None:
        """Check if the tensor shape matches the expected shape."""
        __tracebackhide__ = not DEBUG_MODE
        actual_shape = tuple(tensor.shape)

        for dim_idx, dimension_expression in enumerate(expected_shape):
            if dimension_expression.is_anonymous:
                # we don't need to check anonymous dimensions
                continue

            if (
                dimension_expression.is_literal
                and dimension_expression.identifier not in self.tensor_shape_map
            ):
                # handled by the check method above
                _logger.debug(
                    "Skipping literal dimension %r (%s)",
                    dimension_expression,
                    self.tensor_shape_map,
                )
                continue

            if (
                dimension_expression.is_identifier
                and dimension_expression.identifier not in self.tensor_shape_map
            ):
                _logger.debug(
                    "establishing %r with %r",
                    dimension_expression.identifier,
                    actual_shape[dim_idx],
                )
                self.tensor_shape_map[dimension_expression.identifier] = actual_shape[
                    dim_idx
                ]
                continue

            _logger.debug(
                "Checking dimension %r with scope=%s",
                dimension_expression,
                self.tensor_shape_map,
            )

            try:
                expected_result = dimension_expression.evaluate(self.tensor_shape_map)
            except KeyError as e:
                missing_ref = e.args[0]
                msg = f"[tensor={tensor_arg_name}] Missing reference '{missing_ref}' in {self.tensor_shape_map=}"
                raise DLTypeInvalidReferenceError(msg) from e

            if expected_result != actual_shape[dim_idx]:
                msg = f"[tensor={tensor_arg_name}] Invalid shape at {dim_idx=} actual={actual_shape[dim_idx]} expected {dimension_expression}={expected_result}"
                raise DLTypeShapeError(msg)

            if dimension_expression.identifier not in self.tensor_shape_map:
                self.tensor_shape_map[dimension_expression.identifier] = actual_shape[
                    dim_idx
                ]


def _resolve_numpy_dtype(np_array_t: type[npt.NDArray[Any]]) -> list[npt.DTypeLike]:
    """Resolve the numpy dtype of a numpy array."""
    maybe_dtype_arg = get_args(np_array_t)[1]
    maybe_dtype = get_args(maybe_dtype_arg)

    # if the dtype is a union of types, we need to resolve it
    return [
        cast("npt.DTypeLike", dtype)
        for maybe_union in maybe_dtype
        for dtype in get_args(maybe_union) or [maybe_union]
    ]


class TensorTypeBase:
    """A class to represent a tensor type.

    A tensor type is expected to validate the shape of any literal integers present in the type hint.
    It may also choose to validate the datatype of the tensor.
    """

    TORCH_DTYPES: ClassVar[tuple[torch.dtype, ...]] = ()
    """The torch dtypes that this tensor type asserts to contain. (empty for any dtype)."""
    NP_DTYPES: ClassVar[tuple[npt.DTypeLike, ...]] = ()
    """The numpy dtypes that this tensor type asserts to contain. (empty for any dtype)."""

    def __init__(self, shape: str | None, optional: bool = False) -> None:
        """Create a new tensor type object."""
        self.multiaxis_index: int | None = None
        self.anonymous_multiaxis: bool = False
        self.multiaxis_name: str | None = None
        self.optional = optional
        self.expected_shape = self._parse_shape_string(shape)
        # only include literal dimensions that aren't multiaxis
        self._literal_dims = tuple(
            (idx, dim.evaluate({}))
            for idx, dim in enumerate(self.expected_shape)
            if dim.is_literal and idx != self.multiaxis_index  # pyright: ignore[reportUnnecessaryComparison] # TODO(DX-2313): Address pyright errors ignored to migrate from mypy # fmt: skip
        )

    @override
    def __repr__(self) -> str:
        """Get the string representation of the tensor type."""
        return f"{self.__class__.__name__}[{self.expected_shape}]"

    def _parse_shape_string(
        self, shape_string: str | None
    ) -> tuple[DLTypeDimensionExpression, ...]:
        """Parse the shape string into a list of dimension expressions."""
        if shape_string is None:
            return ()

        split_shape = shape_string.split()

        if not split_shape:
            msg = f"Invalid shape {shape_string=}"
            raise SyntaxError(msg)

        # Process shape specification, looking for multiaxis modifiers
        processed_shapes: list[DLTypeDimensionExpression] = []
        modifiers: dict[int, DLTypeModifier | None] = {}

        for i, dim_str in enumerate(split_shape):
            modifiers[i] = None
            for modifier in DLTypeModifier:
                if dim_str.startswith(modifier.value):
                    modifiers[i] = modifier
                    break

            this_dimension_modifier = modifiers[i]
            if this_dimension_modifier in {
                DLTypeModifier.NAMED_MULTIAXIS,
                DLTypeModifier.ANONYMOUS_MULTIAXIS,
            }:
                if self.multiaxis_index is not None:
                    msg = f"Multiple multiaxis modifiers not allowed in {shape_string=}"
                    raise SyntaxError(msg)

                self.multiaxis_index = i
                self.multiaxis_name = dim_str[len(this_dimension_modifier.value) :]
                self.anonymous_multiaxis = (
                    this_dimension_modifier == DLTypeModifier.ANONYMOUS_MULTIAXIS
                )

            processed_shapes.append(expression_from_string(dim_str))

        return tuple(processed_shapes)

    @classmethod
    def __class_getitem__(cls, shape_string: str) -> TensorTypeBase:
        """Get the type of the tensor."""
        return cls(shape_string)

    def __get_pydantic_core_schema__(
        self, source_type: type, handler: GetCoreSchemaHandler
    ) -> core_schema.CoreSchema:
        """Get the Pydantic core schema for this type."""

        def validate_tensor(
            tensor: DLtypeTensorT, info: ValidationInfo
        ) -> DLtypeTensorT:
            """Validate the tensor."""
            __tracebackhide__ = not DEBUG_MODE
            self.check(tensor)

            if PYDANTIC_INFO_KEY not in info.data:
                info.data[PYDANTIC_INFO_KEY] = _DLTypeContext()

            dl_context = cast("_DLTypeContext", info.data[PYDANTIC_INFO_KEY])
            dl_context.add(info.field_name or "_unknown_", tensor, self)
            dl_context.assert_context()

            return tensor

        if get_origin(source_type) is np.ndarray:
            dtypes = _resolve_numpy_dtype(source_type)
            if self.NP_DTYPES and any(dtype not in self.NP_DTYPES for dtype in dtypes):
                msg = f"Invalid numpy array dtype=<{dtypes}> expected ({'|'.join(map(str, self.NP_DTYPES))})"
                raise DLTypeDtypeError(msg)
            # numpy arrays don't implement isinstance() because the type is actually a
            # parameterized generic alias and not a concrete type. We need to check the origin instead.
            # This is a bit of a hack, but we still get the correct type hint in the end because we check against the dtype of the tensor first.
            source_type = np.ndarray

        return core_schema.with_info_after_validator_function(
            validate_tensor,
            schema=core_schema.is_instance_schema(source_type),
            field_name=handler.field_name,
        )

    def check(self, tensor: npt.NDArray[Any] | torch.Tensor) -> None:
        """Check if the tensor matches this type."""
        # Basic validation for multi-axis dimensions
        __tracebackhide__ = not DEBUG_MODE
        if self.multiaxis_index is not None:
            # Min required dimensions = expected shape length + extra dimensions - 1 (the multi-axis placeholder)
            min_required_dims = len(self.expected_shape) - 1
            if len(tensor.shape) < min_required_dims:
                msg = f"Invalid number of dimensions: {tensor.shape=}, expected at least {min_required_dims}"
                raise DLTypeShapeError(msg)

        # Standard case: exact dimension count match
        elif len(tensor.shape) != len(self.expected_shape):
            msg = f"Invalid number of dimensions {tensor.shape=} {self.expected_shape=}"
            raise DLTypeShapeError(msg)

        if (
            isinstance(tensor, torch.Tensor)
            and self.TORCH_DTYPES
            and tensor.dtype not in self.TORCH_DTYPES
        ):
            msg = f"Invalid dtype {tensor.dtype} expected {self.TORCH_DTYPES}"
            raise DLTypeDtypeError(msg)

        if (
            isinstance(tensor, np.ndarray)
            and self.NP_DTYPES
            and tensor.dtype not in self.NP_DTYPES
        ):
            msg = f"Invalid dtype {tensor.dtype} expected {self.NP_DTYPES}"
            raise DLTypeDtypeError(msg)

        for idx, dim in self._literal_dims:
            # Adjust index if multiaxis exists and is before this dimension
            adjusted_idx = idx
            if self.multiaxis_index is not None and idx > self.multiaxis_index:
                # Adjust by the difference between actual and expected dimensions
                adjusted_idx += len(tensor.shape) - len(self.expected_shape)

            if tensor.shape[adjusted_idx] != dim:
                msg = f"[tensor=tensor] Invalid shape at dim_idx={adjusted_idx} actual={tensor.shape[adjusted_idx]} expected {dim}"
                raise DLTypeShapeError(msg)


class IntTensor(TensorTypeBase):
    """A class to represent an integer tensor type."""

    TORCH_DTYPES = (torch.int, torch.int8, torch.int16, torch.int32, torch.int64)
    NP_DTYPES = (np.int_, np.int8, np.int16, np.int32, np.int64)


class FloatTensor(TensorTypeBase):
    """A class to represent a float tensor type."""

    TORCH_DTYPES = (
        torch.float,
        torch.float16,
        torch.float32,
        torch.float64,
        torch.half,
        torch.bfloat16,
        torch.double,
    )
    NP_DTYPES = (
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

    TORCH_DTYPES = (torch.bool,)
    NP_DTYPES = (np.bool_,)


class DoubleTensor(TensorTypeBase):
    """A class to represent a double tensor type."""

    TORCH_DTYPES = (torch.double,)
    NP_DTYPES = (np.float64,)


def _maybe_get_type_hints(
    existing_hints: dict[str, _DLTypeAnnotation | None] | None, func: Callable[P, R]
) -> dict[str, _DLTypeAnnotation | None] | None:
    """Get the type hints for a function, or return an empty dict if not available."""
    if existing_hints is not None:
        return existing_hints
    try:
        return {
            name: _DLTypeAnnotation.from_hint(hint)
            for name, hint in get_type_hints(func, include_extras=True).items()
        }
    except NameError:
        return None


def _maybe_get_signature(
    existing: inspect.Signature | None, func: Callable[P, R]
) -> inspect.Signature | None:
    """Get the signature of a function, or return an empty signature if not available."""
    if existing is not None:
        return existing
    try:
        return inspect.signature(func)
    except TypeError:
        return None


def dltyped(  # noqa: C901, PLR0915
    scope_provider: DLTypeScopeProvider | Literal["self"] | None = None,
) -> Callable[[Callable[P, R]], Callable[P, R]]:
    """Apply type checking to the decorated function.

    Args:
        scope_provider: An optional scope provider to use for type checking, if None, no scope provider is used, if 'self'
            is used, the first argument of the function is expected to be a DLTypeScopeProvider and the function must be a method.

    Returns:
        A wrapper function with type checking
    """

    def _inner_dltyped(func: Callable[P, R]) -> Callable[P, R]:  # noqa: C901, PLR0915
        if torch.jit.is_scripting():
            # jit script doesn't support annotated type hints at all, we have no choice but to skip the type checking
            return func

        # Handle regular functions
        signature = _maybe_get_signature(None, func)
        # assume that if signature is None, we are dealing with a function with a forward reference, which is almost certainly a classmethod or staticmethod
        # we can't check the signature in this case, so we just assume it's a method for now to avoid raising a false positive error
        # if it _isn't_ a method but we specified "self", later on when we check if the scope provider is a DLTypeScopeProvider, we'll raise an error
        is_method = (
            bool("self" in signature.parameters or "cls" in signature.parameters)
            if signature
            else True
        )
        if scope_provider == "self" and not is_method:
            msg = "Scope provider types can only be used with methods."
            raise TypeError(msg)
        _return_key = "return"
        dltype_hints = _maybe_get_type_hints(None, func)

        # if we added dltype to a method where it will have no effect, warn the user
        if dltype_hints is not None and all(v is None for v in dltype_hints.values()):
            _logger.warning("dltype_hints=%r", dltype_hints)
            warnings.warn(
                "No DLType hints found, skipping type checking",
                UserWarning,
                stacklevel=2,
            )
            return func

        @wraps(func)
        @torch.jit.unused
        def wrapper(*args: P.args, **kwargs: P.kwargs) -> R:  # noqa: C901, PLR0912
            __tracebackhide__ = not DEBUG_MODE
            nonlocal signature
            nonlocal dltype_hints

            dltype_hints = _maybe_get_type_hints(dltype_hints, func)
            signature = _maybe_get_signature(signature, func)
            if signature is None or dltype_hints is None:
                warnings.warn(
                    "Unable to determine signature of dltyped function, type checking will be skipped. (Inner classes with forward references are  not supported.)",
                    UserWarning,
                    stacklevel=2,
                )
                return func(*args, **kwargs)

            bound_args = signature.bind(*args, **kwargs)
            bound_args.apply_defaults()

            _actual_args = bound_args.arguments

            ctx = _DLTypeContext()

            if scope_provider == "self" and isinstance(
                _actual_args[str(scope_provider)], DLTypeScopeProvider
            ):
                ctx.tensor_shape_map = _actual_args[
                    str(scope_provider)
                ].get_dltype_scope()
                _logger.debug("Using self as scope provider %s", ctx.tensor_shape_map)
            elif scope_provider is not None and isinstance(
                scope_provider, DLTypeScopeProvider
            ):
                ctx.tensor_shape_map = scope_provider.get_dltype_scope()
                _logger.debug("Using unbound scope provider %s", ctx.tensor_shape_map)
            elif scope_provider is not None:
                msg = f"Invalid scope provider {scope_provider=} expected DLTypeScopeProvider or 'self'"
                raise DLTypeScopeProviderError(msg)

            for name in dltype_hints:
                if name == _return_key:
                    # special handling of the return value, we don't want to evaluate the function before the arguments are checked
                    continue

                if name in {"self", "cls"}:
                    # if we have an argument called self or cls with a type hint, we would skip a parameter incorrectly
                    # just disallow the behavior
                    msg = f"Invalid argument {name=} is not a supported argument name for dltype."
                    raise TypeError(msg)

                if maybe_annotation := dltype_hints.get(name):
                    tensor = _actual_args[name]
                    ctx.add(name, tensor, maybe_annotation.dltype_annotation)
                elif isinstance(_actual_args[name], torch.Tensor | np.ndarray):
                    warnings.warn(
                        f"[argument={name}] is missing a DLType hint",
                        UserWarning,
                        stacklevel=2,
                    )
                else:
                    _logger.debug("No DLType hint for %r", name)

            try:
                ctx.assert_context()
                retval = func(*args, **kwargs)
                if maybe_return_annotation := dltype_hints.get(_return_key):
                    ctx.add(
                        _return_key, retval, maybe_return_annotation.dltype_annotation
                    )
                    ctx.assert_context()
                elif isinstance(retval, torch.Tensor | np.ndarray):
                    warnings.warn(
                        f"[{_return_key}] is missing a DLType hint",
                        UserWarning,
                        stacklevel=2,
                    )
            except DLTypeError as e:
                # include the full function signature in the error message
                msg = f"Error in {func.__name__}{signature}: {e}\n"
                raise e.__class__(msg) from e
            return retval

        return wrapper

    return _inner_dltyped


# Add this with your other TypeVar definitions
NT = TypeVar("NT", bound=NamedTuple)


def dltyped_namedtuple() -> Callable[[type[NT]], type[NT]]:  # noqa: C901
    """Apply type checking to a NamedTuple class.

    Returns:
        A modified NamedTuple class with type checking on construction
    """

    def _inner_dltyped_namedtuple(cls: type[NT]) -> type[NT]:
        # NOTE: NamedTuple isn't actually a class, it's a factory function that returns a new class so we can't use issubclass here
        if not (isinstance(cls, type) and hasattr(cls, '_fields') and issubclass(cls, tuple)):  # pyright: ignore[reportUnnecessaryIsInstance] # TODO(DX-2313): Address pyright errors ignored to migrate from mypy # fmt: skip
            msg = f"Expected a NamedTuple class, got {cls}"
            raise TypeError(msg)

        # Get the field annotations from the NamedTuple
        field_hints = get_type_hints(cls, include_extras=True)

        # Check for fields with DLType annotations
        dltype_fields: dict[str, _DLTypeAnnotation] = {}
        for field_name in cls._fields:
            if field_name in field_hints:
                hint = field_hints[field_name]
                dltype_annotation = _DLTypeAnnotation.from_hint(hint)
                if dltype_annotation is not None:
                    dltype_fields[field_name] = dltype_annotation

        # If no fields need validation, return the original class
        if not dltype_fields:
            return cls

        # Create a new __new__ method that validates on construction
        original_new = cls.__new__

        def validated_new(cls_inner: type[NT], *args: Any, **kwargs: Any) -> NT:  # noqa: ANN401 (these actually can be any type)
            """A new __new__ method that validates the fields upon construction."""
            # First create the instance using the original __new__
            instance = original_new(cls_inner, *args, **kwargs)

            # Then validate all fields with DLType annotations
            ctx = _DLTypeContext()
            for field_name, annotation in dltype_fields.items():
                field_index = cls._fields.index(field_name)
                value = instance[field_index]
                if not (annotation.dltype_annotation.optional and value is None):
                    ctx.add(field_name, value, annotation.dltype_annotation)

            # Assert that all fields are valid
            try:
                ctx.assert_context()
            except DLTypeError as e:
                msg = f"Error in {cls.__name__} constructor: {e}\n"
                raise e.__class__(msg) from e

            return instance

        # Create the new class with our modified __new__ method
        return cast("type[NT]", type(cls.__name__, (cls,), {"__new__": validated_new}))

    return _inner_dltyped_namedtuple


DataclassT = TypeVar("DataclassT")


def dltyped_dataclass() -> Callable[[type[DataclassT]], type[DataclassT]]:
    """Apply type checking to a dataclass.

    This will validate all fields with DLType annotations during object construction.
    Works with both regular and frozen dataclasses.

    Returns:
        A modified dataclass with type checking on initialization
    """

    def _inner_dltyped_dataclass(cls: type[DataclassT]) -> type[DataclassT]:
        if torch.jit.is_scripting():
            return cls

        # check that we are a dataclass, raise an error if not
        if not hasattr(cls, "__dataclass_fields__"):
            msg = f"Class {cls.__name__} is not a dataclass, apply @dataclass below dltyped_dataclass."
            raise TypeError(msg)

        # Store original __init__ to ensure we run after the dataclass initialization
        original_init = cls.__init__
        # Get field annotations
        field_hints = get_type_hints(cls, include_extras=True)
        _dltype_hints = {
            name: _DLTypeAnnotation.from_hint(hint)
            for name, hint in field_hints.items()
        }

        def new_init(self: DataclassT, *args: Any, **kwargs: Any) -> None:  # noqa: ANN401
            """A new __init__ method that validates the fields after initialization."""
            # First call the original __init__
            original_init(self, *args, **kwargs)

            # Validate fields with DLType annotations
            ctx = _DLTypeContext()

            for field_name in field_hints:
                annotation = _dltype_hints.get(field_name)
                if annotation is not None:
                    # Get the field value
                    value = getattr(self, field_name, None)

                    # Skip None values for optional fields
                    if annotation.dltype_annotation.optional and value is None:
                        continue

                    # Add to validation context
                    ctx.add(field_name, value, annotation.dltype_annotation)

            # Assert that all fields are valid
            try:
                ctx.assert_context()
            except DLTypeError as e:
                msg = f"Error in {cls.__name__} field validation: {e}"
                raise e.__class__(msg) from e

        # Replace the __init__ method
        cls.__init__ = new_init

        return cls

    return _inner_dltyped_dataclass

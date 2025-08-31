"""A module to assist with using Annotated[torch.Tensor] in type hints."""

from __future__ import annotations

import inspect
import logging
import warnings
from functools import wraps
from typing import (
    TYPE_CHECKING,
    Annotated,
    Any,
    Final,
    Literal,
    NamedTuple,
    ParamSpec,
    Protocol,
    TypeVar,
    Union,
    cast,
    get_args,
    get_origin,
    get_type_hints,
    runtime_checkable,
)

from dltype._lib import (
    _constants,
    _dependency_utilities,
    _dltype_context,
    _errors,
    _tensor_type_base,
)

if TYPE_CHECKING:
    from collections.abc import Callable


_logger: Final = logging.getLogger(__name__)

P = ParamSpec("P")
R = TypeVar("R")


class _DLTypeAnnotation(NamedTuple):
    """A class representing a type annotation for a tensor."""

    tensor_type_hint: type[_tensor_type_base.DLtypeTensorT]
    dltype_annotation: _tensor_type_base.TensorTypeBase

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
        if origin is Union:
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
        if len(args) < n_expected_args or not isinstance(
            args[1], _tensor_type_base.TensorTypeBase
        ):
            _logger.warning(
                "Invalid annotated dltype hint: %r",
                args[1:] if len(args) >= n_expected_args else None,
            )
            return None

        # Ensure the base type is a supported tensor type
        tensor_type, dltype_hint = args[0], args[1]
        if not any(
            T in tensor_type.mro() for T in _tensor_type_base.SUPPORTED_TENSOR_TYPES
        ):
            msg = f"Invalid base type=<{tensor_type}> in DLType hint, expected a subtype of {_tensor_type_base.SUPPORTED_TENSOR_TYPES}"
            raise TypeError(msg)

        dltype_hint.optional = optional
        return cls(tensor_type_hint=tensor_type, dltype_annotation=dltype_hint)


@runtime_checkable
class DLTypeScopeProvider(Protocol):
    """A protocol for classes that provide a scope for DLTypeDimensionExpression evaluation."""

    def get_dltype_scope(self) -> _dltype_context.EvaluatedDimensionT:
        """Get the current scope of variables for the DLTypeDimensionExpression evaluation."""
        ...


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
        if _dependency_utilities.is_torch_scripting():
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
        @_dependency_utilities.torch_jit_unused  # pyright: ignore[reportUnknownMemberType]
        def wrapper(*args: P.args, **kwargs: P.kwargs) -> R:  # noqa: C901, PLR0912
            __tracebackhide__ = not _constants.DEBUG_MODE
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

            ctx = _dltype_context.DLTypeContext()

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
                raise _errors.DLTypeScopeProviderError(msg)

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
                elif any(
                    isinstance(_actual_args[name], T)
                    for T in _tensor_type_base.SUPPORTED_TENSOR_TYPES
                ):
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
                elif any(
                    isinstance(retval, T)
                    for T in _tensor_type_base.SUPPORTED_TENSOR_TYPES
                ):
                    warnings.warn(
                        f"[{_return_key}] is missing a DLType hint",
                        UserWarning,
                        stacklevel=2,
                    )
            except _errors.DLTypeError as e:
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
        if not (
            isinstance(cls, type) and hasattr(cls, "_fields") and issubclass(cls, tuple)  # pyright: ignore[reportUnnecessaryIsInstance]
        ):
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
            ctx = _dltype_context.DLTypeContext()
            for field_name, annotation in dltype_fields.items():
                field_index = cls._fields.index(field_name)
                value = instance[field_index]
                if not (annotation.dltype_annotation.optional and value is None):
                    ctx.add(field_name, value, annotation.dltype_annotation)

            # Assert that all fields are valid
            try:
                ctx.assert_context()
            except _errors.DLTypeError as e:
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
        if _dependency_utilities.is_torch_scripting():
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
            ctx = _dltype_context.DLTypeContext()

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
            except _errors.DLTypeError as e:
                msg = f"Error in {cls.__name__} field validation: {e}"
                raise e.__class__(msg) from e

        # Replace the __init__ method
        cls.__init__ = new_init

        return cls

    return _inner_dltyped_dataclass

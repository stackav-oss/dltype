"""Errors for the dltype library."""

from abc import ABC, abstractmethod
from collections import abc
import typing

from dltype._lib._dtypes import DLtypeDtypeT, SUPPORTED_TENSOR_TYPES


class DLTypeError(TypeError, ABC):
    """An error raised when a type assertion is hit."""

    def __init__(self, error_ctx: str | None) -> None:
        self._ctx = error_ctx
        super().__init__()

    def set_context(self, error_ctx: str) -> None:
        self._ctx = error_ctx

    @abstractmethod
    def __str__(self) -> str:
        if self._ctx is not None:
            return f"[{self._ctx}] {self!s}"
        return super().__str__()


class DLTypeUnsupportedTensorTypeError(DLTypeError):
    """An error raised when dltype is attempted to be used on an unsupported tensor type."""

    def __init__(self, actual_type: type[typing.Any]) -> None:
        self._actual = actual_type

    def __str__(self) -> str:
        return f"Invalid tensor type, expected one of {SUPPORTED_TENSOR_TYPES}, actual={self._actual}"


class DLTypeShapeError(DLTypeError):
    """An error raised when a shape assertion is hit."""

    def __init__(
        self,
        index: int,
        expected_shape: int,
        actual: int,
        tensor_name: str,
        error_ctx: str | None = None,
    ) -> None:
        self._tensor_name = tensor_name or "anonymous"
        self._index = index
        self._expected = expected_shape
        self._actual = actual
        super().__init__(error_ctx=error_ctx)

    def __str__(self) -> str:
        return f"Invalid tensor shape, tensor={self._tensor_name} dim={self._index} expected={self._expected} actual={self._actual}"


class DLTypeNDimsError(DLTypeError):
    """An error raised when a tensor does not have the expected number of dimensions."""

    def __init__(
        self,
        expected: int,
        actual: int,
        tensor_name: str,
        error_ctx: str | None = None,
    ) -> None:
        self._tensor_name = tensor_name or "anonymous"
        self._expected = expected
        self._actual = actual
        super().__init__(error_ctx=error_ctx)

    def __str__(self) -> str:
        return f"Invalid number of dimensions, tensor={self._tensor_name} expected ndims={self._expected} actual={self._actual}"


class DLTypeDtypeError(DLTypeError):
    """An error raised when a dtype assertion is hit."""

    def __init__(
        self,
        tensor_name: str | None,
        expected: abc.Iterable[DLtypeDtypeT] | None,
        received: abc.Iterable[DLtypeDtypeT] | None,
        error_ctx: str | None = None,
    ) -> None:
        """Raise an error regarding an invalid shape."""
        self._tensor_name = tensor_name or "anonymous"
        self._expected = expected or set()
        self._received = received or set()
        super().__init__(error_ctx=error_ctx)

    def __str__(self) -> str:
        return f"Invalid dtype, tensor={self._tensor_name} expected one of ({', '.join(sorted(map(str, self._expected)))}) got={', '.join(sorted(map(str, self._received)))}"


class DLTypeDuplicateError(DLTypeError):
    """An error raised when a duplicate tensor name is hit."""

    def __init__(
        self,
        tensor_name: str | None,
        error_ctx: str | None = None,
    ) -> None:
        self._tensor_name = tensor_name
        super().__init__(error_ctx=error_ctx)

    def __str__(self) -> str:
        return f"Invalid duplicate tensor, tensor={self._tensor_name}"


class DLTypeInvalidReferenceError(DLTypeError):
    """An error raised when an invalid reference is hit."""

    def __init__(
        self,
        tensor_name: str | None,
        missing_ref: str | None,
        current_context: dict[str, int] | None,
        error_ctx: str | None = None,
    ) -> None:
        self._tensor_name = tensor_name or "?"
        self._missing_ref = missing_ref or "?"
        self._context = current_context or {}
        super().__init__(error_ctx=error_ctx)

    def __str__(self) -> str:
        return f"Invalid axis referenced before assignment tensor={self._tensor_name} missing_ref={self._missing_ref} valid_refs={', '.join(self._context.keys())}"


class DLTypeScopeProviderError(DLTypeError):
    """An error raised when an invalid scope provider is hit."""

    def __init__(
        self,
        bad_scope_provider: str,
        error_ctx: str | None = None,
    ) -> None:
        self._bad_scope_provider = bad_scope_provider
        super().__init__(error_ctx=error_ctx)

    def __str__(self) -> str:
        return f"Invalid scope provider {self._bad_scope_provider}, expected 'self' or a DLTypeScopeProvider"

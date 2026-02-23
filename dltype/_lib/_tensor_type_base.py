"""The base class for all dltype supported tensor annotations."""

from __future__ import annotations

import typing

from pydantic_core import core_schema
from typing_extensions import override

from dltype._lib import (
    _constants,
    _dltype_context,
    _dtypes,
    _errors,
    _parser,
    _symbolic_expressions,
)
from dltype._lib import _dependency_utilities as _deps

if typing.TYPE_CHECKING:
    from pydantic import GetCoreSchemaHandler, ValidationInfo

if _deps.is_numpy_available():
    import numpy as np
    import numpy.typing as npt


def _resolve_numpy_dtype(
    np_array_t: type[npt.NDArray[typing.Any]],
) -> list[npt.DTypeLike]:
    """Resolve the numpy dtype of a numpy array."""
    maybe_dtype_arg = typing.get_args(np_array_t)[1]
    maybe_dtype = typing.get_args(maybe_dtype_arg)

    # if the dtype is a union of types, we need to resolve it
    return [
        typing.cast("npt.DTypeLike", dtype)
        for maybe_union in maybe_dtype
        for dtype in typing.get_args(maybe_union) or [maybe_union]
    ]


class TensorTypeBase:
    """
    A class to represent a tensor type.

    A tensor type is expected to validate the shape of any literal integers present in the type hint.
    It may also choose to validate the datatype of the tensor.
    """

    DTYPES: typing.ClassVar[tuple[_dtypes.DLtypeDtypeT, ...]] = ()
    """The torch dtypes that this tensor type asserts to contain. (empty for any dtype)."""

    def __init__(self, shape: str | None, *, optional: bool = False) -> None:
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
            if dim.is_literal and idx != self.multiaxis_index  # pyright: ignore[reportUnnecessaryComparison]
        )

    @override
    def __repr__(self) -> str:
        """Get the string representation of the tensor type."""
        return f"{self.__class__.__name__}[{self.expected_shape}]"

    def _parse_shape_string(
        self,
        shape_string: str | None,
    ) -> tuple[_parser.DLTypeDimensionExpression, ...]:
        """Parse the shape string into a list of dimension expressions."""
        if shape_string is None:
            return ()

        split_shape = shape_string.split()

        if not split_shape:
            msg = f"Invalid shape {shape_string=}"
            raise SyntaxError(msg)

        # Process shape specification, looking for multiaxis modifiers
        processed_shapes: list[_parser.DLTypeDimensionExpression] = []
        _multiaxis_parsed: set[int] = set()

        for i, dim_str in enumerate(split_shape):
            expression = _parser.expression_from_string(dim_str)
            if expression.is_named_multiaxis or expression.is_anonymous:
                _multiaxis_parsed.add(i)
                self.multiaxis_name = expression.identifier if expression.is_named_multiaxis else None
                self.multiaxis_index = i
            self.anonymous_multiaxis |= expression.is_anonymous

            processed_shapes.append(expression)

        if len(_multiaxis_parsed) > 1:
            msg = f"Multiple multiaxis modifiers not allowed in {shape_string=}"
            raise SyntaxError(msg)

        return tuple(processed_shapes)

    @classmethod
    def __class_getitem__(cls, shape_string: str | None | _symbolic_expressions.Shape) -> TensorTypeBase:
        """Get the type of the tensor."""
        return cls(shape_string if isinstance(shape_string, str | None) else str(shape_string))

    def __get_pydantic_core_schema__(
        self,
        source_type: type,
        handler: GetCoreSchemaHandler,
    ) -> core_schema.CoreSchema:
        """Get the Pydantic core schema for this type."""

        def validate_tensor(
            tensor: _dtypes.DLtypeTensorT,
            info: ValidationInfo,
        ) -> _dtypes.DLtypeTensorT:
            """Validate the tensor."""
            __tracebackhide__ = not _constants.DEBUG_MODE
            self.check(tensor, info.field_name or "anonymous")

            if _constants.PYDANTIC_INFO_KEY not in info.data:
                info.data[_constants.PYDANTIC_INFO_KEY] = _dltype_context.DLTypeContext()

            dl_context = typing.cast(
                "_dltype_context.DLTypeContext",
                info.data[_constants.PYDANTIC_INFO_KEY],
            )
            dl_context.add(info.field_name or "_unknown_", (tensor,), (self,))
            dl_context.assert_context()

            return tensor

        if _deps.is_numpy_available() and typing.get_origin(source_type) is np.ndarray:  # pyright: ignore[reportPossiblyUnboundVariable]
            dtypes = _resolve_numpy_dtype(source_type)
            if self.DTYPES and any(dtype not in self.DTYPES for dtype in dtypes):
                raise _errors.DLTypeDtypeError(
                    tensor_name=handler.field_name,
                    expected=self.DTYPES,
                    received=dtypes,
                )
            # numpy arrays don't implement isinstance() because the type is actually a
            # parameterized generic alias and not a concrete type. We need to check the origin instead.
            # This is a bit of a hack, but we still get the correct type hint in
            # the end because we check against the dtype of the tensor first.
            source_type = np.ndarray  # pyright: ignore[reportPossiblyUnboundVariable]

        return core_schema.with_info_after_validator_function(
            validate_tensor,
            schema=core_schema.is_instance_schema(source_type),
            field_name=handler.field_name,
        )

    def check(
        self,
        tensor: _dtypes.DLtypeTensorT,
        tensor_name: str = "anonymous",
    ) -> None:
        """Check if the tensor matches this type."""
        # Basic validation for multi-axis dimensions
        __tracebackhide__ = not _constants.DEBUG_MODE
        if self.multiaxis_index is not None:
            # Min required dimensions = expected shape length + extra dimensions - 1 (the multi-axis placeholder)
            min_required_dims = len(self.expected_shape) - 1
            if len(tensor.shape) < min_required_dims:
                raise _errors.DLTypeNDimsError(
                    expected=min_required_dims,
                    actual=tensor.ndim,
                    tensor_name=tensor_name,
                )

        # Standard case: exact dimension count match
        elif len(tensor.shape) != len(self.expected_shape):
            raise _errors.DLTypeNDimsError(
                expected=len(self.expected_shape),
                actual=tensor.ndim,
                tensor_name=tensor_name,
            )

        if self.DTYPES and tensor.dtype not in self.DTYPES:
            raise _errors.DLTypeDtypeError(
                expected=self.DTYPES,
                received={tensor.dtype},
                tensor_name=tensor_name,
            )

        for idx, dim in self._literal_dims:
            # Adjust index if multiaxis exists and is before this dimension
            adjusted_idx = idx
            if self.multiaxis_index is not None and idx > self.multiaxis_index:
                # Adjust by the difference between actual and expected dimensions
                adjusted_idx += len(tensor.shape) - len(self.expected_shape)

            if tensor.shape[adjusted_idx] != dim:
                raise _errors.DLTypeShapeError(
                    tensor_name=tensor_name,
                    index=adjusted_idx,
                    expected_shape=dim,
                    actual=tensor.shape[adjusted_idx],
                )

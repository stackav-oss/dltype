"""The base class for all dltype supported tensor annotations."""

from __future__ import annotations

import typing

import numpy as np
import numpy.typing as npt
import torch
from pydantic_core import core_schema
from typing_extensions import override

from dltype._lib import (
    _parser,
    _errors,
    _constants,
    _dltype_context,
    _dependency_utilities as _deps,
)

if typing.TYPE_CHECKING:
    from pydantic import GetCoreSchemaHandler, ValidationInfo

if _deps.is_numpy_available() and _deps.is_torch_available():
    import torch
    import numpy.typing as npt

    DLtypeTensorT: typing.TypeAlias = torch.Tensor | npt.NDArray[typing.Any]
    DLtypeDtypeT: typing.TypeAlias = torch.dtype | npt.DTypeLike
    SUPPORTED_TENSOR_TYPES: typing.Final = {torch.Tensor, np.ndarray}
elif _deps.is_numpy_available():
    import numpy as np
    import numpy.typing as npt

    DLtypeTensorT: typing.TypeAlias = npt.NDArray[typing.Any]
    DLtypeDtypeT: typing.TypeAlias = npt.DTypeLike
    SUPPORTED_TENSOR_TYPES: typing.Final = {np.ndarray}
elif _deps.is_torch_available():
    import torch

    DLtypeTensorT: typing.TypeAlias = torch.Tensor
    DLtypeDtypeT: typing.TypeAlias = torch.dtype
    SUPPORTED_TENSOR_TYPES: typing.Final = {torch.Tensor}
else:
    _deps.raise_for_missing_dependency()


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
    """A class to represent a tensor type.

    A tensor type is expected to validate the shape of any literal integers present in the type hint.
    It may also choose to validate the datatype of the tensor.
    """

    DTYPES: typing.ClassVar[tuple[DLtypeDtypeT, ...]] = ()
    """The torch dtypes that this tensor type asserts to contain. (empty for any dtype)."""

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
        modifiers: dict[int, _parser.DLTypeModifier | None] = {}

        for i, dim_str in enumerate(split_shape):
            modifiers[i] = None
            for modifier in _parser.DLTypeModifier:
                if dim_str.startswith(modifier.value):
                    modifiers[i] = modifier
                    break

            this_dimension_modifier = modifiers[i]
            if this_dimension_modifier in {
                _parser.DLTypeModifier.NAMED_MULTIAXIS,
                _parser.DLTypeModifier.ANONYMOUS_MULTIAXIS,
            }:
                if self.multiaxis_index is not None:
                    msg = f"Multiple multiaxis modifiers not allowed in {shape_string=}"
                    raise SyntaxError(msg)

                self.multiaxis_index = i
                self.multiaxis_name = dim_str[len(this_dimension_modifier.value) :]
                self.anonymous_multiaxis = (
                    this_dimension_modifier
                    == _parser.DLTypeModifier.ANONYMOUS_MULTIAXIS
                )

            processed_shapes.append(_parser.expression_from_string(dim_str))

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
            __tracebackhide__ = not _constants.DEBUG_MODE
            self.check(tensor)

            if _constants.PYDANTIC_INFO_KEY not in info.data:
                info.data[_constants.PYDANTIC_INFO_KEY] = (
                    _dltype_context.DLTypeContext()
                )

            dl_context = typing.cast(
                "_dltype_context.DLTypeContext", info.data[_constants.PYDANTIC_INFO_KEY]
            )
            dl_context.add(info.field_name or "_unknown_", tensor, self)
            dl_context.assert_context()

            return tensor

        if _deps.is_numpy_available() and typing.get_origin(source_type) is np.ndarray:
            dtypes = _resolve_numpy_dtype(source_type)
            if self.DTYPES and any(dtype not in self.DTYPES for dtype in dtypes):
                msg = f"Invalid numpy array dtype=<{dtypes}> expected ({'|'.join(map(str, self.DTYPES))})"
                raise _errors.DLTypeDtypeError(msg)
            # numpy arrays don't implement isinstance() because the type is actually a
            # parameterized generic alias and not a concrete type. We need to check the origin instead.
            # This is a bit of a hack, but we still get the correct type hint in the end because we check against the dtype of the tensor first.
            source_type = np.ndarray

        return core_schema.with_info_after_validator_function(
            validate_tensor,
            schema=core_schema.is_instance_schema(source_type),
            field_name=handler.field_name,
        )

    def check(self, tensor: DLtypeTensorT) -> None:
        """Check if the tensor matches this type."""
        # Basic validation for multi-axis dimensions
        __tracebackhide__ = not _constants.DEBUG_MODE
        if self.multiaxis_index is not None:
            # Min required dimensions = expected shape length + extra dimensions - 1 (the multi-axis placeholder)
            min_required_dims = len(self.expected_shape) - 1
            if len(tensor.shape) < min_required_dims:
                msg = f"Invalid number of dimensions: {tensor.shape=}, expected at least {min_required_dims}"
                raise _errors.DLTypeShapeError(msg)

        # Standard case: exact dimension count match
        elif len(tensor.shape) != len(self.expected_shape):
            msg = f"Invalid number of dimensions {tensor.shape=} {self.expected_shape=}"
            raise _errors.DLTypeShapeError(msg)

        if self.DTYPES and tensor.dtype not in self.DTYPES:
            msg = f"Invalid dtype {tensor.dtype} expected {self.DTYPES}"
            raise _errors.DLTypeDtypeError(msg)

        for idx, dim in self._literal_dims:
            # Adjust index if multiaxis exists and is before this dimension
            adjusted_idx = idx
            if self.multiaxis_index is not None and idx > self.multiaxis_index:
                # Adjust by the difference between actual and expected dimensions
                adjusted_idx += len(tensor.shape) - len(self.expected_shape)

            if tensor.shape[adjusted_idx] != dim:
                msg = f"[tensor=tensor] Invalid shape at dim_idx={adjusted_idx} actual={tensor.shape[adjusted_idx]} expected {dim}"
                raise _errors.DLTypeShapeError(msg)

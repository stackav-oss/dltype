"""A module to assist with using Annotated[torch.Tensor] in type hints."""

from __future__ import annotations

import logging
import time
import warnings
from collections import deque
from typing import Any, Final, NamedTuple, TypeAlias

from dltype._lib import _constants, _errors, _parser, _tensor_type_base

_logger: Final = logging.getLogger(__name__)

EvaluatedDimensionT: TypeAlias = dict[str, int]


def _maybe_warn_runtime(runtime_ns: int) -> None:
    return runtime_ns > _constants.MAX_ACCEPTABLE_EVALUATION_TIME_NS


class _ConcreteType(NamedTuple):
    """A class containing a tensor name, a tensor value, and its type."""

    tensor_arg_name: str
    tensor: _tensor_type_base.DLtypeTensorT
    dltype_annotation: _tensor_type_base.TensorTypeBase

    def get_expected_shape(
        self, tensor: _tensor_type_base.DLtypeTensorT
    ) -> tuple[_parser.DLTypeDimensionExpression, ...]:
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
                    _parser.DLTypeDimensionExpression.from_multiaxis_literal(
                        f"{self.dltype_annotation.multiaxis_name}[{i}]",
                        actual_shape[self.dltype_annotation.multiaxis_index + i],
                        is_anonymous=self.dltype_annotation.anonymous_multiaxis,
                    ),
                )

        return tuple(expected_shape)


class DLTypeContext:
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
        self.registered_tensor_dtypes: dict[str, _tensor_type_base.DLtypeDtypeT] = {}

    def add(
        self,
        name: str,
        tensor: Any,
        dltype_annotation: _tensor_type_base.TensorTypeBase,
    ) -> None:  # noqa: ANN401
        """Add a tensor to the context."""
        if dltype_annotation.optional and tensor is None:
            # skip optional tensors
            return
        if not any(
            isinstance(tensor, T) for T in _tensor_type_base.SUPPORTED_TENSOR_TYPES
        ):
            msg = f"Invalid type {type(tensor)}"
            raise _errors.DLTypeError(msg)
        self._hinted_tensors.append(_ConcreteType(name, tensor, dltype_annotation))

    def assert_context(self) -> None:
        """Considering the current context, check if all tensors match their expected types."""
        __tracebackhide__ = not _constants.DEBUG_MODE

        start_t = time.perf_counter_ns()

        try:
            while self._hinted_tensors:
                tensor_context = self._hinted_tensors.popleft()
                # first check if the tensor could possibly have the right shape
                tensor_context.dltype_annotation.check(tensor_context.tensor)

                if tensor_context.tensor_arg_name in self.registered_tensor_dtypes:
                    msg = f"[tensor={tensor_context.tensor_arg_name=}] Duplicate tensor name in type checking context!"
                    raise _errors.DLTypeDuplicateError(msg)

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
                    f"Type checking took longer than expected {(runtime_ns) / 1e6:.2f}ms > {_constants.MAX_ACCEPTABLE_EVALUATION_TIME_NS / 1e6}ms",
                    UserWarning,
                    stacklevel=2,
                )

    def _assert_tensor_shape(
        self,
        tensor_arg_name: str,
        expected_shape: tuple[_parser.DLTypeDimensionExpression, ...],
        tensor: _tensor_type_base.DLtypeTensorT,
    ) -> None:
        """Check if the tensor shape matches the expected shape."""
        __tracebackhide__ = not _constants.DEBUG_MODE
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
                raise _errors.DLTypeInvalidReferenceError(msg) from e

            if expected_result != actual_shape[dim_idx]:
                msg = f"[tensor={tensor_arg_name}] Invalid shape at {dim_idx=} actual={actual_shape[dim_idx]} expected {dimension_expression}={expected_result}"
                raise _errors.DLTypeShapeError(msg)

            if dimension_expression.identifier not in self.tensor_shape_map:
                self.tensor_shape_map[dimension_expression.identifier] = actual_shape[
                    dim_idx
                ]

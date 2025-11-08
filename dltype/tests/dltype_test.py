# pyright: reportPrivateUsage=false, reportUnknownMemberType=false
"""Tests for common types used in deep learning."""

import re
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Annotated, Final, NamedTuple, TypeAlias
from unittest.mock import patch

import numpy as np
import numpy.typing as npt
import pytest
import torch
from pydantic import BaseModel

import dltype

np_rand = np.random.RandomState(42).rand
NPFloatArrayT: TypeAlias = npt.NDArray[np.float32 | np.float64]
NPIntArrayT: TypeAlias = npt.NDArray[np.int32 | np.uint16 | np.uint32 | np.uint8]


class _RaisesInfo(NamedTuple):
    exception_type: type[Exception] | None = None
    regex: str | None = None
    value: torch.Tensor | None = None


@dltype.dltyped()
def bad_function(
    tensor: Annotated[torch.Tensor, dltype.TensorTypeBase["b c h w"]],
) -> Annotated[torch.Tensor, dltype.TensorTypeBase["h w b c"]]:
    """A function that takes a tensor and returns a tensor."""
    return tensor


@dltype.dltyped()
def good_function(
    tensor: Annotated[torch.Tensor, dltype.TensorTypeBase["b c h w"]],
) -> Annotated[torch.Tensor, dltype.TensorTypeBase["h w b c"]]:
    """A function that takes a tensor and returns a tensor."""
    return tensor.permute(2, 3, 0, 1)


@dltype.dltyped()
def incomplete_annotated_function(
    tensor: Annotated[torch.Tensor, dltype.TensorTypeBase["b c h w"]],
) -> torch.Tensor:
    """A function that takes a tensor and returns a tensor."""
    return tensor


@dltype.dltyped()
def incomplete_return_function(
    tensor: torch.Tensor,
) -> Annotated[torch.Tensor, dltype.TensorTypeBase["h w b c"]]:
    """A function that takes a tensor and returns a tensor."""
    return tensor.permute(2, 3, 0, 1)


@dltype.dltyped()
def inconsistent_shape_function(
    tensor: Annotated[torch.Tensor, dltype.TensorTypeBase["b c h w"]],
) -> Annotated[torch.Tensor, dltype.TensorTypeBase["h w b"]]:
    """A function that takes a tensor and returns a tensor."""
    if tensor.shape[0] == 1:
        return tensor.permute(2, 3, 0)
    return tensor.permute(2, 3, 0, 1)


BAD_DIMENSION_RX: Final = r"Invalid number of dimensions"


def bad_dimension_error(
    tensor_name: str,
    *,
    idx: int,
    expected: int,
    actual: int,
) -> str:
    return re.escape(
        f"Invalid tensor shape, tensor={tensor_name} dim={idx} expected={expected} actual={actual}",
    )


def bad_ndim_error(tensor_name: str, *, expected: int, actual: int) -> str:
    return re.escape(
        f"Invalid number of dimensions, tensor={tensor_name} expected ndims={expected} actual={actual}",
    )


@pytest.mark.parametrize(
    ("input_tensor", "func", "expected"),
    [
        pytest.param(
            torch.ones(1, 1, 1, 1),
            bad_function,
            _RaisesInfo(value=torch.ones(1, 1, 1, 1)),
            id="bad_func trivial",
        ),
        pytest.param(
            torch.rand(1, 2, 3, 4),
            bad_function,
            _RaisesInfo(
                exception_type=dltype.DLTypeShapeError,
                regex=bad_dimension_error("return", expected=3, idx=0, actual=1),
            ),
            id="bad_func_4D",
        ),
        pytest.param(
            torch.rand(1, 2, 3),
            bad_function,
            _RaisesInfo(
                exception_type=dltype.DLTypeNDimsError,
                regex=bad_ndim_error("tensor", expected=4, actual=3),
            ),
            id="bad_func_3D",
        ),
        pytest.param(
            torch.rand(1, 2, 3, 4, 5),
            bad_function,
            _RaisesInfo(
                exception_type=dltype.DLTypeNDimsError,
                regex=bad_ndim_error("tensor", expected=4, actual=5),
            ),
            id="bad_func_5D",
        ),
        pytest.param(
            torch.ones(1, 2, 3, 4),
            good_function,
            _RaisesInfo(value=torch.ones(3, 4, 1, 2)),
            id="good_func_4D",
        ),
        pytest.param(
            torch.rand(1, 2, 3),
            good_function,
            _RaisesInfo(
                exception_type=dltype.DLTypeNDimsError,
                regex=bad_ndim_error("tensor", expected=4, actual=3),
            ),
            id="good_func_3D",
        ),
        pytest.param(
            torch.ones(1, 2, 3, 4),
            incomplete_annotated_function,
            _RaisesInfo(value=torch.ones(1, 2, 3, 4)),
            id="incomplete_annotated_4D",
        ),
        pytest.param(
            torch.ones(1, 2, 3, 4),
            incomplete_return_function,
            _RaisesInfo(value=torch.ones(3, 4, 1, 2)),
            id="incomplete_return_4D",
        ),
        pytest.param(
            torch.ones(1, 2, 3),
            incomplete_return_function,
            _RaisesInfo(
                exception_type=RuntimeError,
                regex=r"number of dimensions in the tensor input does not match*",
            ),
            id="invalid arg no type hint",
        ),
    ],
)
def test_single_in_single_out(
    input_tensor: torch.Tensor,
    func: Callable[[torch.Tensor], torch.Tensor],
    expected: _RaisesInfo,
) -> None:
    # test both positional and keyword arguments
    if expected.exception_type is not None:
        with pytest.raises(expected.exception_type, match=expected.regex):
            func(input_tensor)
        with pytest.raises(expected.exception_type, match=expected.regex):
            func(tensor=input_tensor)  # pyright: ignore[reportCallIssue]
    else:
        torch.testing.assert_close(func(input_tensor), expected.value)
        torch.testing.assert_close(func(tensor=input_tensor), expected.value)  # pyright: ignore[reportCallIssue]


class _TestBaseModel(BaseModel, frozen=True):
    tensor: Annotated[torch.Tensor, dltype.TensorTypeBase("b c h w")]
    tensor_2: Annotated[torch.Tensor, dltype.TensorTypeBase("b c h w")]


class _TestBaseModel2(BaseModel, frozen=True):
    tensor: Annotated[torch.Tensor, dltype.TensorTypeBase("b h c w")]
    tensor_2: Annotated[torch.Tensor, dltype.TensorTypeBase("b a c d")]


class _TestBaseModelWithNumpy(BaseModel, frozen=True):
    tensor: Annotated[NPFloatArrayT, dltype.TensorTypeBase("b c h w")]
    tensor_2: Annotated[NPFloatArrayT, dltype.FloatTensor("b c h w")]


@pytest.mark.parametrize(
    ("tensor", "tensor_2", "model", "expected"),
    [
        pytest.param(
            torch.rand(1, 2, 3, 4),
            torch.rand(1, 2, 3, 4),
            _TestBaseModel,
            _RaisesInfo(),
            id="good_tensors",
        ),
        pytest.param(
            torch.rand(1, 2, 3, 4),
            torch.rand(1, 2, 3),
            _TestBaseModel,
            _RaisesInfo(
                exception_type=dltype.DLTypeNDimsError,
                regex=bad_ndim_error("tensor_2", expected=4, actual=3),
            ),
            id="bad_tensors",
        ),
        pytest.param(
            torch.rand(1, 2, 3, 4),
            torch.rand(1, 2, 3, 5),
            _TestBaseModel,
            _RaisesInfo(
                exception_type=dltype.DLTypeShapeError,
                regex=bad_dimension_error("tensor_2", idx=3, expected=4, actual=5),
            ),
            id="bad_tensors_2",
        ),
        pytest.param(
            torch.rand(2, 2, 2, 2),
            torch.rand(2, 2, 2, 2),
            _TestBaseModel,
            _RaisesInfo(),
            id="good_tensors_2",
        ),
        pytest.param(
            torch.rand(1, 2, 3, 4),
            torch.rand(1, 2, 3, 4),
            _TestBaseModel2,
            _RaisesInfo(),
            id="good_tensors_3",
        ),
        pytest.param(
            torch.rand(1, 2, 3, 4),
            torch.rand(1, 5, 3, 7),
            _TestBaseModel2,
            _RaisesInfo(),
            id="bad_tensors_3",
        ),
        pytest.param(
            torch.rand(4, 3, 2, 1),
            torch.rand(1, 2, 3, 4),
            _TestBaseModel2,
            _RaisesInfo(
                exception_type=dltype.DLTypeShapeError,
                regex=bad_dimension_error("tensor_2", idx=0, expected=4, actual=1),
            ),
            id="bad_tensors_4",
        ),
        pytest.param(
            torch.rand(2, 2, 2, 2),
            torch.rand(2, 2, 1, 2),
            _TestBaseModel2,
            _RaisesInfo(
                exception_type=dltype.DLTypeShapeError,
                regex=bad_dimension_error("tensor_2", idx=2, expected=2, actual=1),
            ),
            id="good_tensors_4",
        ),
        pytest.param(
            np_rand(1, 2, 3, 4),
            np_rand(1, 2, 3, 4),
            _TestBaseModelWithNumpy,
            _RaisesInfo(),
            id="good_tensors_numpy",
        ),
        pytest.param(
            np_rand(1, 2, 3, 4),
            np_rand(1, 2, 3),
            _TestBaseModelWithNumpy,
            _RaisesInfo(
                exception_type=dltype.DLTypeNDimsError,
                regex=bad_ndim_error("tensor_2", expected=4, actual=3),
            ),
            id="bad_tensors_numpy",
        ),
        pytest.param(
            np_rand(1, 2, 3, 4),
            np_rand(1, 2, 3, 5),
            _TestBaseModelWithNumpy,
            _RaisesInfo(
                exception_type=dltype.DLTypeShapeError,
                regex=bad_dimension_error("tensor_2", idx=3, expected=4, actual=5),
            ),
            id="bad_tensors_numpy_2",
        ),
        pytest.param(
            np_rand(2, 2, 2, 2),
            np_rand(2, 2, 2, 2).astype(np.int32),
            _TestBaseModelWithNumpy,
            _RaisesInfo(
                exception_type=dltype.DLTypeDtypeError,
                regex=r"Invalid dtype.*got=int32",
            ),
            id="bad_dtype",
        ),
    ],
)
def test_dltype_pydantic(
    tensor: torch.Tensor,
    tensor_2: torch.Tensor,
    model: type[BaseModel],
    expected: _RaisesInfo,
) -> None:
    if expected.exception_type:
        with pytest.raises(expected.exception_type, match=expected.regex):
            model(tensor=tensor, tensor_2=tensor_2)
    else:
        model(tensor=tensor, tensor_2=tensor)


@dltype.dltyped()
def numpy_function(
    tensor: Annotated[NPFloatArrayT, dltype.TensorTypeBase["b c h w"]],
) -> Annotated[torch.Tensor, dltype.TensorTypeBase["h w b c"]]:
    """A function that takes a tensor and returns a tensor."""
    return torch.from_numpy(tensor).permute(2, 3, 0, 1)


@pytest.mark.parametrize(
    ("tensor", "expected"),
    [
        pytest.param(
            np.ones((1, 2, 3, 4), dtype=np.float32),
            _RaisesInfo(value=torch.ones(3, 4, 1, 2)),
            id="good_tensors",
        ),
        pytest.param(
            np.zeros((1, 2, 3)),
            _RaisesInfo(
                exception_type=dltype.DLTypeNDimsError,
                regex=bad_ndim_error("tensor", expected=4, actual=3),
            ),
            id="bad_tensors",
        ),
        pytest.param(
            np.zeros((1, 2, 3, 5, 6)),
            _RaisesInfo(
                exception_type=dltype.DLTypeNDimsError,
                regex=bad_ndim_error("tensor", expected=4, actual=5),
            ),
            id="bad_tensors_2",
        ),
    ],
)
def test_numpy_mixed(tensor: NPFloatArrayT, expected: _RaisesInfo) -> None:
    if expected.exception_type:
        with pytest.raises(expected.exception_type, match=expected.regex):
            numpy_function(tensor)
    else:
        torch.testing.assert_close(numpy_function(tensor), expected.value)


@pytest.mark.parametrize(
    ("tensor_type", "tensor", "expected"),
    [
        pytest.param(
            dltype.IntTensor("b c h w"),
            torch.rand(1, 2, 3, 4),
            _RaisesInfo(exception_type=dltype.DLTypeDtypeError),
            id="int_tensor",
        ),
        pytest.param(
            dltype.IntTensor("b c h w"),
            torch.rand(1, 2, 3, 4).int(),
            _RaisesInfo(),
            id="int_tensor_2",
        ),
        pytest.param(
            dltype.FloatTensor("b c h w"),
            torch.rand(1, 2, 3, 4).int(),
            _RaisesInfo(exception_type=dltype.DLTypeDtypeError),
            id="float_tensor",
        ),
        pytest.param(
            dltype.FloatTensor("b c h w"),
            torch.rand(1, 2, 3, 4),
            _RaisesInfo(),
            id="float_tensor_2",
        ),
        pytest.param(
            dltype.FloatTensor("b c h w"),
            np_rand(1, 2, 3, 4).astype(np.double),
            _RaisesInfo(),
            id="float_tensor_3",
        ),
        pytest.param(
            dltype.IntTensor("b c h w"),
            np_rand(1, 2, 3, 4),
            _RaisesInfo(exception_type=dltype.DLTypeDtypeError),
            id="int_tensor_2",
        ),
        pytest.param(
            dltype.DoubleTensor("b c h w"),
            np_rand(1, 2, 3, 4).astype(np.float32),
            _RaisesInfo(exception_type=dltype.DLTypeDtypeError),
            id="double_tensor",
        ),
        pytest.param(
            dltype.DoubleTensor("b c h w"),
            np_rand(1, 2, 3, 4).astype(np.double),
            _RaisesInfo(),
            id="double_tensor_2",
        ),
        pytest.param(
            dltype.DoubleTensor("b c h w"),
            np_rand(1, 2, 3, 4).astype(np.float64),
            _RaisesInfo(),
            id="double_tensor_3",
        ),
    ],
)
def test_types(
    tensor_type: dltype.TensorTypeBase,
    tensor: torch.Tensor | NPFloatArrayT,
    expected: _RaisesInfo,
) -> None:
    if expected.exception_type:
        with pytest.raises(expected.exception_type, match=expected.regex):
            tensor_type.check(tensor)
    else:
        tensor_type.check(tensor)


@pytest.mark.parametrize(
    ("tensor_type", "tensor", "expected"),
    [
        pytest.param(
            dltype.FloatTensor("1 2 3 4"),
            torch.rand(1, 2, 3, 4),
            _RaisesInfo(),
            id="int_tensor",
        ),
        pytest.param(
            dltype.FloatTensor("1 2 3 4"),
            np_rand(1, 2, 3, 4),
            _RaisesInfo(),
            id="int_tensor_2",
        ),
        pytest.param(
            dltype.FloatTensor("b c 3 4"),
            torch.rand(1, 2, 3, 4),
            _RaisesInfo(),
            id="mixed literal dims",
        ),
        pytest.param(
            dltype.FloatTensor("1 c h 4"),
            torch.rand(1, 2, 3, 4),
            _RaisesInfo(),
            id="dims before and after",
        ),
        pytest.param(
            dltype.FloatTensor("1 2 3 w"),
            torch.rand(2, 2, 3, 9),
            _RaisesInfo(
                exception_type=dltype.DLTypeShapeError,
                regex=bad_dimension_error("anonymous", idx=0, expected=1, actual=2),
            ),
            id="bad literal dim",
        ),
        pytest.param(
            dltype.FloatTensor("*batch c 2 3"),
            torch.rand(1, 2, 2, 3),
            _RaisesInfo(),
            id="wildcard dim",
        ),
        pytest.param(
            dltype.FloatTensor("*batch c 2 3"),
            torch.rand(3, 2, 3, 2),
            _RaisesInfo(
                exception_type=dltype.DLTypeShapeError,
                regex=bad_dimension_error("anonymous", idx=2, expected=2, actual=3),
            ),
            id="wildcard dim_2",
        ),
    ],
)
def test_literal_shapes(
    tensor_type: dltype.TensorTypeBase,
    tensor: torch.Tensor | NPFloatArrayT,
    expected: _RaisesInfo,
) -> None:
    if expected.exception_type is not None:
        with pytest.raises(expected.exception_type, match=expected.regex):
            tensor_type.check(tensor)
    else:
        tensor_type.check(tensor)


def test_onnx_export() -> None:
    class _DummyModule(torch.nn.Module):
        @dltype.dltyped()
        def forward(
            self,
            x: Annotated[torch.Tensor, dltype.FloatTensor("b c h w")],
        ) -> Annotated[torch.Tensor, dltype.FloatTensor("b c h w")]:
            return torch.multiply(x, 2)

    with NamedTemporaryFile() as f:
        torch.onnx.export(
            _DummyModule(),
            (torch.rand(1, 2, 3, 4),),
            f.name,
            input_names=["input"],
            output_names=["output"],
        )

        assert Path(f.name).exists()
        assert Path(f.name).stat().st_size > 0

        with pytest.raises(TypeError):
            torch.onnx.export(
                _DummyModule(),
                (torch.rand(1, 2, 3),),
                f.name,
                input_names=["input"],
                output_names=["output"],
            )


def test_torch_compile() -> None:
    class _DummyModule(torch.nn.Module):
        @dltype.dltyped()
        def forward(
            self,
            x: Annotated[torch.Tensor, dltype.FloatTensor("b c h w")],
        ) -> Annotated[torch.Tensor, dltype.FloatTensor("b c h w")]:
            return torch.multiply(x, 2)

    _DummyModule().forward(torch.rand(1, 2, 3, 4))

    with pytest.raises(TypeError):
        _DummyModule().forward(torch.rand(1, 2, 3))

    module = torch.compile(_DummyModule())

    module(torch.rand(1, 2, 3, 4))

    with pytest.raises(TypeError):
        module(torch.rand(1, 2, 3))

    torch.jit.trace(_DummyModule(), torch.rand(1, 2, 3, 4))

    scripted_module = torch.jit.script(_DummyModule())

    scripted_module(torch.rand(1, 2, 3, 4))

    with pytest.raises(TypeError):
        scripted_module(torch.rand(1, 2, 3))


@dltype.dltyped()
def mixed_func(  # noqa: PLR0913
    tensor: Annotated[torch.Tensor, dltype.TensorTypeBase["b c h w"]],
    array: Annotated[NPFloatArrayT, dltype.TensorTypeBase["b c h w"]],
    number: int,
    other_tensor: torch.Tensor,
    other_array: NPFloatArrayT,
    other_number: float,
    other_annotated_tensor: Annotated[torch.Tensor, dltype.TensorTypeBase["c c c"]],
) -> Annotated[torch.Tensor, dltype.TensorTypeBase["h w b c"]]:
    return tensor.permute(2, 3, 0, 1)


def test_mixed_typing() -> None:
    mixed_func(
        torch.rand(1, 2, 3, 4),
        np_rand(1, 2, 3, 4),
        1,
        torch.rand(1, 1, 1, 1),
        np_rand(1, 2, 3, 4),
        1.0,
        torch.rand(2, 2, 2),
    )

    with pytest.raises(TypeError):
        mixed_func(
            torch.rand(1, 2, 3, 4),
            np_rand(1, 2, 3, 4),
            1,
            torch.rand(1, 1, 1, 1),
            np_rand(1, 2, 3, 4),
            1.0,
            torch.rand(1, 2, 2),
        )


def test_bad_argument_name() -> None:
    @dltype.dltyped()
    def bad_function(
        self: Annotated[torch.Tensor, dltype.TensorTypeBase["b c h w"]],
    ) -> Annotated[torch.Tensor, dltype.TensorTypeBase["h w b c"]]:
        return self

    with pytest.raises(TypeError):
        bad_function(torch.rand(1, 2, 3, 4))

    @dltype.dltyped()
    def other_bad_function(
        self: int,
        tensor: Annotated[torch.Tensor, dltype.TensorTypeBase["b c h w"]],
    ) -> Annotated[torch.Tensor, dltype.TensorTypeBase["h w b c"]]:
        return tensor

    with pytest.raises(TypeError):
        other_bad_function(1, torch.rand(1, 2, 3, 4))


def test_bad_dimension_name() -> None:
    with pytest.raises(SyntaxError):

        def bad_function(  # pyright: ignore[reportUnusedFunction]
            tensor: Annotated[torch.Tensor, dltype.TensorTypeBase["b?"]],
        ) -> None:
            print(tensor)


@dltype.dltyped()
def func_with_expression(
    input_tensor: Annotated[torch.Tensor, dltype.FloatTensor["batch channels dim"]],
) -> Annotated[torch.Tensor, dltype.FloatTensor["batch channels dim-1"]]:
    return input_tensor[..., :-1]


@dltype.dltyped()
def func_with_min(
    input_tensor: Annotated[torch.Tensor, dltype.FloatTensor["batch channels dim"]],
) -> Annotated[torch.Tensor, dltype.FloatTensor["batch channels min(10,max(1,dim-1))"]]:
    if input_tensor.shape[2] == 1:
        return input_tensor[...]
    if input_tensor.shape[2] > 10:
        return input_tensor[..., :10]
    return input_tensor[..., :-1]


@pytest.mark.parametrize(
    ("tensor", "function", "expected"),
    [
        pytest.param(
            torch.rand(1, 2, 3),
            func_with_expression,
            _RaisesInfo(),
            id="basic_expression",
        ),
        pytest.param(
            torch.rand(1, 2, 3, 4),
            func_with_expression,
            _RaisesInfo(
                exception_type=dltype.DLTypeNDimsError,
                regex=bad_ndim_error("input_tensor", expected=3, actual=4),
            ),
            id="bad_expression",
        ),
        pytest.param(
            torch.rand(1, 2, 3),
            func_with_min,
            _RaisesInfo(),
            id="min_expression",
        ),
        pytest.param(
            torch.rand(1, 2, 1),
            func_with_min,
            _RaisesInfo(),
            id="min_expression_2",
        ),
        pytest.param(
            torch.rand(1, 2, 11),
            func_with_min,
            _RaisesInfo(),
            id="min_expression_3",
        ),
        pytest.param(
            torch.rand(1, 2),
            func_with_min,
            _RaisesInfo(
                exception_type=dltype.DLTypeNDimsError,
                regex=bad_ndim_error("input_tensor", expected=3, actual=2),
            ),
            id="min_expression_4",
        ),
    ],
)
def test_typing_expressions(
    tensor: torch.Tensor,
    function: Callable[[torch.Tensor], torch.Tensor],
    expected: _RaisesInfo,
) -> None:
    if expected.exception_type:
        with pytest.raises(expected.exception_type, match=expected.regex):
            function(tensor)
    else:
        function(tensor)


def test_expression_syntax_errors() -> None:
    with pytest.raises(SyntaxError):

        @dltype.dltyped()
        def func_with_bad_expression(  # pyright: ignore[reportUnusedFunction]
            _: Annotated[torch.Tensor, dltype.FloatTensor["batch channels dim+"]],
        ) -> None:
            return None

    with pytest.raises(SyntaxError):

        @dltype.dltyped()
        def func_with_bad_expression(  # pyright: ignore[reportUnusedFunction]
            _: Annotated[torch.Tensor, dltype.FloatTensor["+ - * dim"]],
        ) -> None:
            return None

    with pytest.raises(SyntaxError):
        # don't allow multiple min/max calls
        @dltype.dltyped()
        def func_with_bad_expression(  # pyright: ignore[reportUnusedFunction]
            _: Annotated[
                torch.Tensor,
                dltype.FloatTensor["batch channels min(1,channels-1)+max(channels,dim)"],
            ],
        ) -> None:
            return None

    with pytest.raises(SyntaxError):
        # don't allow multiple operators in a row
        @dltype.dltyped()
        def func_with_bad_expression(  # pyright: ignore[reportUnusedFunction]
            _: Annotated[torch.Tensor, dltype.FloatTensor["dim dim-*1"]],
        ) -> None:
            return None


@dltype.dltyped()
def func_with_axis_wildcard(
    input_tensor: Annotated[torch.Tensor, dltype.FloatTensor["*batch channels h w"]],
    _: Annotated[torch.Tensor, dltype.FloatTensor["*batch channels w h"]],
) -> Annotated[torch.Tensor, dltype.FloatTensor["*batch channels*h*w"]]:
    # reshape along the last two dimensions but preserve the first N dimensions and the channel dimension
    return input_tensor.view(*input_tensor.shape[:-3], -1)


# function with a wildcard in the middle of the dimensions
@dltype.dltyped()
def func_with_mid_tensor_wildcard(
    input_tensor: Annotated[torch.Tensor, dltype.FloatTensor["batch *channels h w"]],
    fail: bool = False,
) -> Annotated[torch.Tensor, dltype.FloatTensor["batch *channels h*w"]]:
    # reshape the tensor to preserve the batch dimension, retain the
    # channel dimensions, and flatten the spatial dimensions
    if fail:
        return input_tensor.view(input_tensor.shape[0] + 1, -1)
    return input_tensor.view(*input_tensor.shape[:-2], -1)


# function with a wildcard in the middle of the dimensions
@dltype.dltyped()
def func_with_anon_wildcard(
    input_tensor: Annotated[torch.Tensor, dltype.FloatTensor["... h w"]],
    fail: bool = False,
) -> Annotated[torch.Tensor, dltype.FloatTensor["N h*w"]]:
    # take any number of dimensions > 2 and preserve the last two dimensions but flatten the rest
    if fail:
        return input_tensor.view(-1, input_tensor.shape[-1], input_tensor.shape[-2])
    return input_tensor.view(-1, input_tensor.shape[-2] * input_tensor.shape[-1])


@pytest.mark.parametrize(
    ("tensor", "maybe_tensor_b", "func", "expected"),
    [
        pytest.param(
            torch.rand(1, 2, 3, 4, 5),
            torch.rand(1, 2, 3, 5, 4),
            func_with_axis_wildcard,
            _RaisesInfo(),
            id="wildcard_axis",
        ),
        pytest.param(
            torch.rand(1, 2, 3, 4),
            torch.rand(1, 2, 4, 3),
            func_with_axis_wildcard,
            _RaisesInfo(),
            id="wildcard_axis_2",
        ),
        pytest.param(
            torch.rand(1, 2, 3),
            torch.rand(1, 2, 3),
            func_with_axis_wildcard,
            _RaisesInfo(
                exception_type=dltype.DLTypeShapeError,
                regex=bad_dimension_error("_", idx=1, expected=3, actual=2),
            ),
            id="wildcard_axis_3",
        ),
        pytest.param(
            torch.rand(1, 2, 3),
            None,
            func_with_mid_tensor_wildcard,
            _RaisesInfo(),
            id="mid_tensor_wildcard",
        ),
        pytest.param(
            torch.rand(1, 2, 3, 4),
            None,
            func_with_mid_tensor_wildcard,
            _RaisesInfo(),
            id="mid_tensor_wildcard_2",
        ),
        pytest.param(
            torch.rand(1, 2, 3, 4, 5),
            None,
            func_with_mid_tensor_wildcard,
            _RaisesInfo(),
            id="mid_tensor_wildcard_2",
        ),
        pytest.param(
            torch.rand(1, 2),
            None,
            func_with_mid_tensor_wildcard,
            _RaisesInfo(
                exception_type=dltype.DLTypeNDimsError,
                regex=bad_ndim_error("input_tensor", expected=3, actual=2),
            ),
            id="mid-tensor not enough dims",
        ),
        pytest.param(
            torch.rand(1, 2, 3, 4),
            True,
            func_with_mid_tensor_wildcard,
            _RaisesInfo(
                exception_type=dltype.DLTypeShapeError,
                regex=bad_dimension_error("return", idx=0, actual=2, expected=1),
            ),
            id="mid-tensor bad function impl",
        ),
        pytest.param(
            torch.rand(1, 2, 3),
            None,
            func_with_anon_wildcard,
            _RaisesInfo(),
            id="anon_wildcard",
        ),
        pytest.param(
            torch.rand(1, 2, 3, 4, 5, 6),
            None,
            func_with_anon_wildcard,
            _RaisesInfo(),
            id="anon_wildcard_2",
        ),
        pytest.param(
            torch.rand(1, 2),
            None,
            func_with_anon_wildcard,
            _RaisesInfo(),
            id="anon_wildcard_3",
        ),
        pytest.param(
            torch.rand(1, 2, 3, 4),
            True,
            func_with_anon_wildcard,
            _RaisesInfo(
                exception_type=dltype.DLTypeNDimsError,
                regex=bad_ndim_error("return", expected=2, actual=3),
            ),
            id="anon wildcard fail",
        ),
    ],
)
def test_multiaxis_support(
    tensor: torch.Tensor,
    maybe_tensor_b: torch.Tensor | bool | None,
    func: Callable[[torch.Tensor, torch.Tensor | bool | None], torch.Tensor],
    expected: _RaisesInfo,
) -> None:
    if expected.exception_type:
        with pytest.raises(expected.exception_type, match=expected.regex):
            func(tensor, maybe_tensor_b)
    else:
        func(tensor, maybe_tensor_b)


# function with a wildcard in the middle of the dimensions
@dltype.dltyped()
def func_with_two_anon_wildcards(
    input_tensor: Annotated[torch.Tensor, dltype.FloatTensor["... h w"]],
    _: None = None,
) -> Annotated[torch.Tensor, dltype.FloatTensor["... h w"]]:
    # concatenate the tensor with itself along the batch dimension
    return torch.cat([input_tensor, input_tensor], dim=0)


@dltype.dltyped()
def bad_func_with_two_named_wildcards(
    input_tensor: Annotated[torch.Tensor, dltype.FloatTensor["*batch h w"]],
    _: None = None,
) -> Annotated[torch.Tensor, dltype.FloatTensor["*batch h w"]]:
    # concatenate the tensor with itself along the batch dimension, this should fail every time because the
    # batch dimension is different
    return torch.cat([input_tensor, input_tensor], dim=0)


@dltype.dltyped()
def func_with_named_wildcard_followed_by_literal(
    input_tensor: Annotated[torch.Tensor, dltype.FloatTensor["*batch 1 h w"]],
    _: None = None,
) -> None:
    # this should work because the batch dimension is the same
    return None


def test_anonymous_wildcard_arg_and_return() -> None:
    func_with_two_anon_wildcards(torch.rand(1, 2, 3))
    input_t = torch.rand(1, 2, 3, 4, 5, 6)
    input_shape = input_t.shape
    result = func_with_two_anon_wildcards(input_t)
    # test that anonymous dimensions aren't matched
    assert result.shape[0] != input_shape[0]

    with pytest.raises(TypeError):
        bad_func_with_two_named_wildcards(torch.rand(1, 2, 3))

    with pytest.raises(TypeError):
        bad_func_with_two_named_wildcards(torch.rand(1, 2, 3, 4, 5, 6))

    func_with_named_wildcard_followed_by_literal(torch.rand(1, 1, 1, 3, 4))
    func_with_named_wildcard_followed_by_literal(torch.rand(4, 3, 2, 1, 4, 5))
    func_with_named_wildcard_followed_by_literal(torch.rand(1, 2, 3))

    with pytest.raises(TypeError):
        func_with_named_wildcard_followed_by_literal(torch.rand(4, 3, 2, 2, 4, 5))

    with pytest.raises(TypeError):
        func_with_named_wildcard_followed_by_literal(torch.rand(3, 2, 1))


def test_multiaxis_syntax() -> None:
    # fail if we have multiple wildcards
    with pytest.raises(SyntaxError):
        dltype.FloatTensor("batch *channels *h w")

    with pytest.raises(SyntaxError):
        dltype.FloatTensor("... *channels h w")

    with pytest.raises(SyntaxError):
        dltype.FloatTensor("...... h w")

    with pytest.raises(SyntaxError):
        dltype.FloatTensor("batch *channels h w *channels")

    with pytest.raises(SyntaxError):
        dltype.FloatTensor("*... h w")

    with pytest.raises(SyntaxError):
        dltype.FloatTensor("...batch h w")


def test_named_axis() -> None:
    dltype.FloatTensor("batch channels dim=1")
    dltype.FloatTensor("batch channels dim=min(batch,channels)")
    dltype.FloatTensor("batch channels dim=max(batch-1,channels)")

    dltype.FloatTensor("batch channels dim=channels-1")
    with pytest.raises(SyntaxError):
        dltype.FloatTensor("batch channels=channels-1")

    with pytest.raises(SyntaxError):
        dltype.FloatTensor("batch channels dim=min(dim,channels)")

    @dltype.dltyped()
    def func_with_misordered_identifier(
        tensor: Annotated[
            torch.Tensor,
            dltype.FloatTensor["batch dim=channels channels=4"],
        ],
    ) -> None:
        return None

    with pytest.raises(dltype.DLTypeNDimsError):
        func_with_misordered_identifier(torch.rand(1, 2, 3, 4))

    with pytest.raises(dltype.DLTypeInvalidReferenceError):
        func_with_misordered_identifier(torch.rand(1, 3, 4))

    with pytest.raises(dltype.DLTypeInvalidReferenceError):
        func_with_misordered_identifier(torch.rand(1, 4, 4))


# mock max_acceptable_eval_time to zero to ensure we issue a warning if the context takes too long
def test_warn_on_function_evaluation() -> None:
    with patch("dltype._lib._dltype_context._maybe_warn_runtime", return_value=True):

        @dltype.dltyped()
        def dummy_function(
            tensor: Annotated[torch.Tensor, dltype.FloatTensor["batch channels h w"]],
        ) -> Annotated[torch.Tensor, dltype.FloatTensor["batch channels h w"]]:
            return tensor

        with pytest.warns(UserWarning, match="Type checking took longer than expected"):
            dummy_function(torch.rand(1, 2, 3, 4))


def test_debug_mode_not_enabled() -> None:
    if dltype.DEBUG_MODE:
        pytest.fail("DEBUG_MODE should not be enabled by default")


def test_incompatible_tensor_type() -> None:
    with pytest.raises(TypeError):

        @dltype.dltyped()
        def bad_function(  # pyright: ignore[reportUnusedFunction]
            tensor: Annotated[list[int], dltype.IntTensor["b c h w"]],
        ) -> list[int]:
            return tensor


def test_dimension_name_with_underscores() -> None:
    @dltype.dltyped()
    def good_function(  # pyright: ignore[reportUnusedFunction]
        tensor: Annotated[
            torch.Tensor,
            dltype.IntTensor["batch channels_in channels_out"],
        ],
    ) -> torch.Tensor:
        return tensor


def test_dimension_with_external_scope() -> None:
    class Provider:
        def get_dltype_scope(self) -> dict[str, int]:
            return {"channels_in": 3, "channels_out": 4}

        @dltype.dltyped(scope_provider="self")
        def forward(
            self,
            tensor: Annotated[
                torch.Tensor,
                dltype.FloatTensor["batch channels_in channels_out"],
            ],
        ) -> torch.Tensor:
            return tensor

    @dltype.dltyped(scope_provider=Provider())
    def good_function(
        tensor: Annotated[
            torch.Tensor,
            dltype.IntTensor["batch channels_in channels_out"],
        ],
    ) -> torch.Tensor:
        return tensor

    good_function(torch.ones(1, 3, 4).int())
    good_function(torch.ones(4, 3, 4).int())

    with pytest.raises(dltype.DLTypeShapeError):
        good_function(torch.ones(1, 3, 5).int())
    with pytest.raises(dltype.DLTypeShapeError):
        good_function(torch.ones(1, 2, 4).int())

    provider = Provider()

    provider.forward(torch.ones(1, 3, 4))
    provider.forward(torch.ones(4, 3, 4))

    with pytest.raises(dltype.DLTypeShapeError):
        provider.forward(torch.ones(1, 3, 5))
    with pytest.raises(dltype.DLTypeShapeError):
        provider.forward(torch.ones(1, 2, 4))


def test_optional_type_handling() -> None:
    """Test that dltyped correctly handles Optional tensor types."""

    # Test with a function with optional parameter
    @dltype.dltyped()
    def optional_tensor_func(
        tensor: Annotated[torch.Tensor, dltype.FloatTensor["b c h w"]] | None,
    ) -> torch.Tensor:
        if tensor is None:
            return torch.zeros(1, 3, 5, 5)
        return tensor

    # Should work with None
    result = optional_tensor_func(None)
    assert result.shape == (1, 3, 5, 5)

    # Should work with correct tensor
    input_tensor = torch.rand(2, 3, 4, 4)
    torch.testing.assert_close(optional_tensor_func(input_tensor), input_tensor)

    # Should fail with incorrect shape
    with pytest.raises(dltype.DLTypeNDimsError):
        optional_tensor_func(torch.rand(2, 3, 4))

    # Test with a function that returns an optional tensor
    @dltype.dltyped()
    def return_optional_tensor(
        *,
        return_none: bool,
    ) -> Annotated[torch.Tensor, dltype.FloatTensor["b c h w"]] | None:
        if return_none:
            return None
        return torch.rand(2, 3, 4, 4)

    # Should work with either None or tensor
    assert return_optional_tensor(return_none=True) is None
    assert return_optional_tensor(return_none=False) is not None

    # Test rejection of non-Optional unions
    with pytest.raises(TypeError, match="Only Optional tensor types are supported"):

        @dltype.dltyped()
        def union_tensor_func(  # pyright: ignore[reportUnusedFunction]
            tensor: Annotated[torch.Tensor, dltype.FloatTensor["b c h w"]]
            | Annotated[torch.Tensor, dltype.IntTensor["b c"]],
        ) -> None:
            pass

    # Test optional with classes
    class ModelWithOptional(torch.nn.Module):
        @dltype.dltyped()
        def forward(
            self,
            x: Annotated[torch.Tensor, dltype.FloatTensor["b c h w"]] | None,
            mask: Annotated[torch.Tensor, dltype.BoolTensor["b h w"]] | None = None,
        ) -> Annotated[torch.Tensor, dltype.FloatTensor["b c h w"]] | None:
            if x is None:
                return None
            if mask is not None:
                return x * mask.unsqueeze(1)
            return x

    model = ModelWithOptional()

    # Test with various input combinations
    x = torch.rand(2, 3, 4, 4)
    mask = torch.randint(0, 2, (2, 4, 4)).bool()

    torch.testing.assert_close(model(x), x)
    torch.testing.assert_close(model(x, mask), x * mask.unsqueeze(1))
    assert model(None) is None
    assert model(None, mask) is None

    # Should still validate shapes when tensors are provided
    with pytest.raises(dltype.DLTypeNDimsError):
        model(torch.rand(2, 3, 4), None)

    with pytest.raises(dltype.DLTypeShapeError):
        model(x, torch.randint(0, 2, (2, 5, 5)).bool())


def test_named_tuple_handling() -> None:
    @dltype.dltyped_namedtuple()
    class MyNamedTuple(NamedTuple):
        tensor: Annotated[torch.Tensor, dltype.FloatTensor["b c h w"]]
        mask: Annotated[torch.Tensor, dltype.IntTensor["b h w"]]
        other: int

    MyNamedTuple(torch.rand(2, 3, 4, 4), torch.randint(0, 2, (2, 4, 4)), 1)

    with pytest.raises(dltype.DLTypeNDimsError):
        MyNamedTuple(torch.rand(2, 3, 4), torch.randint(0, 2, (2, 4, 4)), 1)

    with pytest.raises(dltype.DLTypeDtypeError):
        MyNamedTuple(torch.rand(2, 3, 4, 4), torch.randint(0, 2, (2, 4, 4)).bool(), 1)

    # test a named tuple with an optional field

    @dltype.dltyped_namedtuple()
    class MyOptionalNamedTuple(NamedTuple):
        tensor: Annotated[torch.Tensor, dltype.FloatTensor["b c h w"]]
        mask: Annotated[torch.Tensor, dltype.IntTensor["b h w"]] | None

    MyOptionalNamedTuple(torch.rand(2, 3, 4, 4), None)
    MyOptionalNamedTuple(torch.rand(2, 3, 4, 4), torch.randint(0, 2, (2, 4, 4)))

    with pytest.raises(dltype.DLTypeNDimsError):
        MyOptionalNamedTuple(torch.rand(2, 3, 4), None)

    with pytest.raises(dltype.DLTypeDtypeError):
        MyOptionalNamedTuple(
            torch.rand(2, 3, 4, 4),
            torch.randint(0, 2, (2, 4, 4)).bool(),
        )


def test_annotated_dataclass() -> None:
    """Test that dltyped correctly handles Annotated dataclasses."""

    # Test with a function with annotated dataclass
    @dltype.dltyped_dataclass()
    @dataclass(frozen=True, kw_only=True, slots=True)
    class AnnotatedDataclass:
        tensor: Annotated[torch.Tensor, dltype.FloatTensor["b c h w"]]
        tensor_2: Annotated[torch.Tensor, dltype.IntTensor["b c h w"]]
        other_thing: int
        un_annotated_tensor: torch.Tensor

    AnnotatedDataclass(
        tensor=torch.rand(1, 2, 3, 4),
        tensor_2=torch.randint(0, 10, (1, 2, 3, 4)),
        other_thing=5,
        un_annotated_tensor=torch.rand(1, 2, 3, 4),
    )

    with pytest.raises(dltype.DLTypeShapeError):
        AnnotatedDataclass(
            tensor=torch.rand(1, 2, 3, 4),
            tensor_2=torch.randint(0, 10, (1, 2, 3, 5)),
            other_thing=5,
            un_annotated_tensor=torch.rand(1, 2, 3, 4),
        )

    # test that the order of the decorator matters

    with pytest.raises(
        TypeError,
        match=r"Class AnnotatedDataclass2 is not a dataclass, apply @dataclass below dltyped_dataclass.",
    ):

        @dataclass(frozen=True, kw_only=True, slots=True)
        @dltype.dltyped_dataclass()
        class AnnotatedDataclass2:  # pyright: ignore[reportUnusedClass]
            pass

    # test with optional fields

    @dltype.dltyped_dataclass()
    @dataclass(frozen=True, kw_only=True, slots=True)
    class OptionalAnnotatedDataclass:
        tensor: Annotated[torch.Tensor, dltype.FloatTensor["b c h w"]] | None
        tensor_2: Annotated[torch.Tensor, dltype.IntTensor["b c h w"]] | None = None
        other_thing: int = 0

    OptionalAnnotatedDataclass(tensor=None)
    OptionalAnnotatedDataclass(tensor=None, tensor_2=None)
    OptionalAnnotatedDataclass(tensor=torch.rand(1, 2, 3, 4))
    OptionalAnnotatedDataclass(
        tensor=torch.rand(1, 2, 3, 4),
        tensor_2=torch.randint(0, 10, (1, 2, 3, 4)),
    )

    with pytest.raises(dltype.DLTypeNDimsError):
        OptionalAnnotatedDataclass(tensor=torch.rand(1, 2, 3))

    with pytest.raises(dltype.DLTypeDtypeError):
        OptionalAnnotatedDataclass(
            tensor=torch.rand(1, 2, 3, 4),
            tensor_2=torch.rand(1, 2, 3, 4).bool(),
        )


def test_improper_base_model_construction() -> None:
    """Test that improper construction of BaseModel raises an error."""
    with pytest.raises(dltype.DLTypeDtypeError, match=r"Invalid dtype"):

        class _BadModel(BaseModel):  # pyright: ignore[reportUnusedClass]
            tensor: Annotated[npt.NDArray[np.float32], dltype.IntTensor["b c h w"]]

    with pytest.raises(dltype.DLTypeDtypeError, match=r"Invalid dtype"):

        class _BadModel2(BaseModel):  # pyright: ignore[reportUnusedClass]
            tensor: Annotated[
                npt.NDArray[np.float32 | np.float64],
                dltype.IntTensor["b c h w"],
            ]

    class _GoodModel(BaseModel):  # pyright: ignore[reportUnusedClass]
        tensor: Annotated[npt.NDArray[np.int32], dltype.IntTensor["b c h w"]]

    class _GoodModel2(BaseModel):  # pyright: ignore[reportUnusedClass]
        tensor: Annotated[npt.NDArray[np.int8 | np.int32], dltype.IntTensor["b c h w"]]


class _MyClass:
    def __init__(self, tensor: torch.Tensor) -> None:
        self.tensor = tensor

    @classmethod
    @dltype.dltyped()
    def create(
        cls,
        tensor: Annotated[torch.Tensor, dltype.FloatTensor["b c h w"]],
    ) -> "_MyClass":
        return cls(tensor)


def test_class_with_forward_reference() -> None:
    """Test that a class with a forward reference to itself raises an error."""
    _MyClass.create(torch.rand(1, 2, 3, 4))

    with pytest.raises(dltype.DLTypeNDimsError):
        _MyClass.create(torch.rand(1, 2, 3))

    # inner classes do not evaluate the forward reference correctly, warn the user

    class _InnerClass:
        def __init__(self, tensor: torch.Tensor) -> None:
            self.tensor = tensor

        @classmethod
        @dltype.dltyped()
        def create(
            cls,
            tensor: Annotated[torch.Tensor, dltype.FloatTensor["b c h w"]],
        ) -> "_InnerClass":
            return cls(tensor)

    with pytest.warns(
        UserWarning,
        match="Unable to determine signature of dltyped function, type checking will be skipped.",
    ):
        _InnerClass.create(torch.rand(1, 2, 3, 4))


def test_warning_if_decorator_has_no_annotations_to_check() -> None:
    with pytest.warns(
        UserWarning,
        match="No DLType hints found, skipping type checking",
    ):

        @dltype.dltyped()
        def no_annotations(tensor: torch.Tensor) -> torch.Tensor:  # pyright: ignore[reportUnusedFunction]
            return tensor

    # should warn if some tensors are untyped
    @dltype.dltyped()
    def some_annotations(
        tensor: Annotated[torch.Tensor, dltype.FloatTensor["1 2 3"]],
    ) -> torch.Tensor:
        return tensor

    with pytest.warns(
        UserWarning,
        match=re.escape("[return] is missing a DLType hint"),
    ):
        some_annotations(torch.rand(1, 2, 3))


def test_scalar() -> None:
    """Test that dltyped correctly handles scalar types."""

    @dltype.dltyped()
    def scalar_func(
        x: Annotated[torch.Tensor, dltype.FloatTensor[None]],
    ) -> Annotated[torch.Tensor, dltype.FloatTensor[None]]:
        return x

    # Should work with a scalar tensor
    scalar_func(torch.tensor(3.14))

    # Should fail with a non-scalar tensor
    with pytest.raises(dltype.DLTypeNDimsError):
        scalar_func(torch.tensor([3.14]))

    with pytest.raises(SyntaxError, match="Invalid shape shape_string=''"):
        Annotated[torch.Tensor, dltype.FloatTensor[""]]


def test_signed_vs_unsigned() -> None:
    """Test signed vs unsigned errors are handled correctly."""

    @dltype.dltyped()
    def signed_vs_unsigned(
        x: Annotated[NPIntArrayT, dltype.SignedIntTensor["x"]],
        y: Annotated[NPIntArrayT, dltype.UnsignedIntTensor["x"]],
    ) -> Annotated[torch.Tensor, dltype.IntTensor["x"]]:
        return torch.from_numpy((x * y).astype(np.uint8))

    # should work nominally

    np.testing.assert_allclose(
        signed_vs_unsigned(
            np.array([6], dtype=np.int32),  # pyright: ignore[reportUnknownArgumentType]
            np.array([8], dtype=np.uint32),
        ).numpy(),
        np.array([48], dtype=np.uint8),
    )

    # Should fail with a bad signed tensor
    with pytest.raises(dltype.DLTypeDtypeError):
        signed_vs_unsigned(
            np.array([6], dtype=np.uint32),
            np.array([8], dtype=np.uint32),
        )

    with pytest.raises(dltype.DLTypeDtypeError):
        signed_vs_unsigned(np.array([6], dtype=np.int32), np.array([8], dtype=np.int32))


def test_bit_widths() -> None:
    """Test bit width errors are handled correctly."""

    @dltype.dltyped()
    def various_bit_widths(
        x: Annotated[NPIntArrayT, dltype.UInt16Tensor["x"]],
        y: Annotated[torch.Tensor, dltype.Int64Tensor["x"]],
    ) -> Annotated[NPIntArrayT, dltype.UInt8Tensor["x"]]:
        return (x + y.numpy()).astype(np.uint8)

    # should work nominally

    np.testing.assert_allclose(
        various_bit_widths(
            np.array([6], dtype=np.uint16),
            torch.tensor([8], dtype=torch.int64),
        ),
        np.array([14], dtype=np.uint8),
    )

    # Should fail with a bad width on a numpy tensor
    with pytest.raises(dltype.DLTypeDtypeError):
        various_bit_widths(
            np.array([6], dtype=np.uint32),
            torch.tensor([8], dtype=torch.int64),
        )

    # Should fail with a bad width on a torch tensor
    with pytest.raises(dltype.DLTypeDtypeError):
        various_bit_widths(
            np.array([6], dtype=np.uint16),
            torch.tensor([8], dtype=torch.int32),
        )


def test_invalid_tensor_type_handling() -> None:
    with pytest.raises(dltype.DLTypeUnsupportedTensorTypeError):
        good_function([1, 2, 3])  # type: ignore (intentionally bypass static type checking)


ShapedTensorT: TypeAlias = Annotated[torch.Tensor, dltype.Float16Tensor["1 2 3"]]


def test_type_alias() -> None:
    @dltype.dltyped()
    def function(tensor: ShapedTensorT) -> None:
        print(tensor)

    function(torch.empty((1, 2, 3), dtype=torch.float16))

    with pytest.raises(dltype.DLTypeDtypeError):
        function(torch.empty(1, 2, 3, dtype=torch.float32))


ShapeT: TypeAlias = dltype.Shape[1, 2, ..., dltype.VariableAxis("last")]
RGB: Final = dltype.ConstantAxis("RGB", 3)
Batch: Final = dltype.AnonymousAxis("batch")
ImgH: Final = dltype.VariableAxis("ImgH")
ImgW: Final = dltype.VariableAxis("ImgW")

ImgBatch: TypeAlias = dltype.Shape[Batch, RGB, ImgH, ImgW]


def test_shaped_tensor() -> None:
    @dltype.dltyped()
    def func(
        tensor: Annotated[torch.Tensor, dltype.FloatTensor[ShapeT]],
        fail: bool = False,
    ) -> Annotated[torch.Tensor, dltype.FloatTensor["1 2 ... last"]]:
        return tensor if not fail else torch.empty(1, 2, 99)

    assert str(ImgBatch) == "*batch RGB=3 ImgH ImgW"

    @dltype.dltyped()
    def func2(
        arg: Annotated[torch.Tensor, dltype.UInt8Tensor[ImgBatch]],
    ) -> Annotated[
        torch.Tensor,
        dltype.UInt8Tensor[
            dltype.Shape[
                Batch,
                ImgH * ImgW,
                RGB,
            ]
        ],
    ]:
        if arg.ndim > 3:
            return arg.view(*arg.shape[:-3], arg.shape[-2] * arg.shape[-1], 3)

        return arg.view(arg.shape[-2] * arg.shape[-1], 3)

    func(torch.empty(1, 2, 9))
    func(torch.empty(1, 2, 3, 9))

    func2(torch.empty(3, 4, 5, dtype=torch.uint8))
    func2(torch.empty(1, 3, 4, 5, dtype=torch.uint8))
    func2(torch.empty(1, 2, 3, 4, 5, dtype=torch.uint8))

    with pytest.raises(dltype.DLTypeShapeError):
        func(torch.empty(0, 2, 8))

    with pytest.raises(dltype.DLTypeShapeError):
        func(torch.empty(1, 2, 4), True)

    class PydanticObj(BaseModel):
        tensor_a: Annotated[torch.Tensor, dltype.FloatTensor[dltype.Shape[4, 4]]]

    PydanticObj(tensor_a=torch.zeros((4, 4)))

    with pytest.raises(dltype.DLTypeShapeError):
        PydanticObj(tensor_a=torch.ones((3, 3)))

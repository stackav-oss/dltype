"""Example usages."""

from __future__ import annotations

from collections.abc import Iterator  # noqa: TC003
from contextlib import contextmanager
from typing import Annotated

import numpy as np

import dltype


@contextmanager
def _hide_internal_dltype_stacktrace(name: str) -> Iterator[None]:
    try:
        yield
        msg = "Expected block to raise"
        raise RuntimeError(msg)
    except dltype.DLTypeError as e:
        print(f"{name}: {e.__class__.__name__}: {e}")  # noqa: T201


"""
Basic usage.
"""


@dltype.dltyped()
def cat_1d(
    arr1: Annotated[np.ndarray, dltype.FloatTensor["len1"]],
    arr2: Annotated[np.ndarray, dltype.FloatTensor["len2"]],
) -> Annotated[np.ndarray, dltype.FloatTensor["len1+len2"]]:
    """Concatenate 2 arrays together on the first axis."""
    return np.concatenate((arr1, arr2), axis=0)


@dltype.dltyped()
def fixed_size_crop(
    arr1: Annotated[np.ndarray, dltype.FloatTensor["batch channels=3 height width"]],
) -> Annotated[np.ndarray, dltype.FloatTensor["batch channels min(768,height) min(1024,width)"]]:
    """Crop the top 1024x768 pixels."""
    return arr1[..., :768, :1024]


@dltype.dltyped()
def warning_for_missing_annotation(
    # >>> UserWarning: [no_annotation] is missing a DLType hint
    no_annotation: np.ndarray,
) -> Annotated[np.ndarray, dltype.FloatTensor["batch channels w h"]]:
    """Crop the top 1024x768 pixels."""
    return no_annotation


B = dltype.VariableAxis("batch")
C = dltype.ConstantAxis("channels", 3)
W = dltype.VariableAxis("width")
H = dltype.VariableAxis("height")
N = dltype.AnonymousAxis("ndims")

# Saving an annotation as a type alias for later use
ImgShape = dltype.Shape[B, C, W, H]
Uint8Img = dltype.UInt8Tensor[ImgShape]
NPImgArr = Annotated[np.ndarray, Uint8Img]


@dltype.dltyped()
def static_shape_stack(
    arr: Annotated[np.ndarray, dltype.IntTensor[dltype.Shape[B, C, N]]],
    # note the B*2, resolves to 2x the input batch dimension
) -> Annotated[np.ndarray, dltype.IntTensor[dltype.Shape[B * 2, C, N]]]:
    """
    Stack an array on top of itself.

    Examples of using statically defined shapes.
    Static analyzers will catch invalid shape expressions.
    In addition to built in operators we also support ISQRT, min, and max (imported through dltype, not the python builtin).
    """
    return np.concatenate((arr, arr), axis=0)


if __name__ == "__main__":
    assert cat_1d(np.zeros((1)), np.ones((2))).shape == (3,)

    with _hide_internal_dltype_stacktrace("bad dims"):
        # >>> DLTypeNDimsError: Invalid number of dimensions, tensor=arr2 expected ndims=1 actual=2
        cat_1d(np.zeros((1,)), np.zeros((1, 2)))

    with _hide_internal_dltype_stacktrace("bad dtype"):
        # >>> DLTypeDtypeError: Invalid dtype, tensor=arr1 expected one of (...supported float types) got=int32
        cat_1d(np.zeros((1,), dtype=np.int32), np.zeros((1,)))

    img = np.zeros((1, 3, 800, 2048))
    fixed_size_crop(img)

    with _hide_internal_dltype_stacktrace("bad channels"):
        # >>> DLTypeShapeError: Invalid tensor shape, tensor=arr1 dim=1 expected=3 actual=1
        fixed_size_crop(img[:, :1, ...])

    fixed_size_crop(img[..., :100])

    static_shape_stack(np.zeros((1, 3, 5, 5, 9), dtype=np.int32))

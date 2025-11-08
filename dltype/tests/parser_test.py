# pyright: reportPrivateUsage=false
"""Tests for expression parsing."""

import pytest

from dltype import AnonymousAxis, ConstantAxis, LiteralAxis, Max, Min, Shape, VariableAxis
from dltype._lib import _parser


@pytest.mark.parametrize(
    ("expression", "scope", "expected"),
    [
        ("a", {"a": 1}, 1),
        ("a=1", {}, 1),
        ("1+2", {}, 3),
        ("1+2*3", {}, 7),
        ("3*3", {}, 9),
        ("3*3", {"x": 1}, 9),
        ("3*x", {"x": 3}, 9),
        ("x+y*x", {"x": 3, "y": 4}, 15),
        ("min(55,1-2)", {}, -1),
        ("max(55,1-2)", {}, 55),
        ("min(max(0,x),y)", {"x": 3, "y": 4}, 3),
        ("min(max(0,x),y)", {"x": -3, "y": 4}, 0),
        ("max(x-y,0)", {"x": -3, "y": -4}, 1),
        ("max(x-y,0)", {"x": 3, "y": 4}, 0),
        ("max(x-y,0)", {"x": 3, "y": 2}, 1),
        ("3^2", {}, 9),
        ("3^2^2", {}, 81),
        ("min(3^x,max(3-y,99))", {"x": 2, "y": 4}, 9),
        ("max(3^x,min(3-y,99))", {"x": 2, "y": 4}, 9),
        ("min(3^x,3-y)", {"x": 2, "y": 4}, -1),
        ("variable_name_with_underscores", {"variable_name_with_underscores": 1}, 1),
    ],
)
def test_parse_expression(
    expression: str,
    scope: dict[str, int],
    expected: int,
) -> None:
    assert _parser.expression_from_string(expression).evaluate(scope) == expected


@pytest.mark.parametrize(
    ("expression", "scope"),
    [
        ("1 + 2", {}),
        ("a+2", {}),
        ("a=a-1", {}),
        ("a", {"b": 1}),
        ("min(1,2", {}),
        ("*batch", {}),
        ("3**2", {}),
        ("^", {}),
    ],
)
def test_parse_invalid_expression(expression: str, scope: dict[str, int]) -> None:
    with pytest.raises((SyntaxError, KeyError)):
        _parser.expression_from_string(expression).evaluate(scope)


@pytest.mark.parametrize(
    ("expression", "expected"),
    [
        (Shape[1 + 2], "3"),
        (Shape[Max(1, VariableAxis("imageh"))], "max(1,imageh)"),
        (Shape[Min(1, 2)], "1"),
        (Shape[ConstantAxis("RGB", 4)], "RGB=4"),
        (Shape[..., VariableAxis("c"), VariableAxis("h"), VariableAxis("w")], "... c h w"),
        (
            Shape[AnonymousAxis("batch"), VariableAxis("c"), VariableAxis("h"), VariableAxis("w")],
            "*batch c h w",
        ),
        (Shape[LiteralAxis(4), VariableAxis("r")], "4 r"),
        (Shape[Min(4 + VariableAxis("image_w"), VariableAxis("imageh"))], "min(4+image_w,imageh)"),
    ],
)
def test_parse_symbolic(expression: Shape, expected: str) -> None:
    assert str(expression) == expected


def test_raises() -> None:
    rgb = ConstantAxis("RGB", 3)

    # NOTE: we disallow adding anything to constants because it isn't clear what the intent
    # would be.

    # For example, if we have rgb=3 we clearly have an axis that is meant for
    # rgb channels and would have a shape of 3.

    # however, what does rgb+1 mean? Do we want the axis of rgb to be 4 instead? in which case, is it even
    # referring to the same axis anymore? Or do we want a new axis for rgba, but how do we change the name in an addition operation?

    with pytest.raises(TypeError):
        _ = rgb + 4  # pyright: ignore[reportUnknownVariableType, reportOperatorIssue]

    with pytest.raises(TypeError):
        _ = 4 + rgb  # pyright: ignore[reportUnknownVariableType, reportOperatorIssue]

    # Similarly operating on anonymous axes doesn't make sense in general as it isn't clear what the intent would be
    with pytest.raises(TypeError):
        _ = AnonymousAxis("*batch") + 4  # pyright: ignore[reportUnknownVariableType, reportOperatorIssue]

    with pytest.raises(TypeError):
        _ = 4 + AnonymousAxis("*batch")  # pyright: ignore[reportUnknownVariableType, reportOperatorIssue]

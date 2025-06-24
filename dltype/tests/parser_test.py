# pyright: reportPrivateUsage=false
"""Tests for expression parsing."""

import pytest

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
    expression: str, scope: dict[str, int], expected: int
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

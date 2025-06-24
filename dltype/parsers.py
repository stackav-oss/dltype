"""A module for parsing dimension shape expressions for dltype."""

from __future__ import annotations

import enum
import logging
import re
from typing import Final

from typing_extensions import override

_logger: Final = logging.getLogger(__name__)


class DLTypeSpecifier(enum.Enum):
    """An enum representing a way to specify a name for a dimension expression or literal."""

    EQUALS = "="


class DLTypeModifier(enum.Enum):
    """An enum representing a modifier that can be applied to a dimension expression."""

    ANONYMOUS_MULTIAXIS = "..."
    NAMED_MULTIAXIS = "*"


class _DLTypeOperator(enum.Enum):
    """An enum representing a mathematical operator for a dimension expression."""

    ADD = "+"
    SUB = "-"
    MUL = "*"
    EXP = "^"
    DIV = "/"
    MIN = "min"
    MAX = "max"

    def evaluate(self, a: int, b: int) -> int:  # noqa: PLR0911
        """Evaluate the operator."""
        if self is _DLTypeOperator.ADD:
            return a + b
        if self is _DLTypeOperator.SUB:
            return a - b
        if self is _DLTypeOperator.MUL:
            return a * b
        if self is _DLTypeOperator.EXP:
            return int(a**b)
        if self is _DLTypeOperator.DIV:
            return a // b
        if self is _DLTypeOperator.MIN:
            return min(a, b)
        if self is _DLTypeOperator.MAX:
            return max(a, b)
        raise NotImplementedError(self)


_op_precedence: Final = {
    _DLTypeOperator.ADD: 1,
    _DLTypeOperator.SUB: 1,
    _DLTypeOperator.MUL: 2,
    _DLTypeOperator.DIV: 2,
    _DLTypeOperator.EXP: 3,
    _DLTypeOperator.MIN: 0,
    _DLTypeOperator.MAX: 0,
}

_functional_operators: Final = frozenset({_DLTypeOperator.MIN, _DLTypeOperator.MAX})
_valid_operators: frozenset[str] = frozenset(
    {op.value for op in _DLTypeOperator if op not in _functional_operators}
)
_valid_modifiers: frozenset[str] = frozenset({mod.value for mod in DLTypeModifier})

INFIX_EXPRESSION_SPLIT_RX: Final = re.compile(
    f"({'|'.join(map(re.escape, _valid_operators))})"
)
VALID_EXPRESSION_RX: Final = re.compile(
    f"^[a-zA-Z0-9_{''.join(map(re.escape, _valid_operators.union(_valid_modifiers)))}]+$"
)
VALID_IDENTIFIER_RX: Final = re.compile(r"^[a-zA-Z][a-zA-Z0-9\_]*$")


class DLTypeDimensionExpression:
    """A class representing a dimension that depends on other dimensions."""

    def __init__(
        self,
        identifier: str,
        postfix_expression: list[str | int | _DLTypeOperator],
        *,
        is_multiaxis_literal: bool = False,
        is_anonymous: bool = False,
    ) -> None:
        """Create a new dimension expression."""
        self.identifier = identifier
        self.parsed_expression = postfix_expression
        # multiaxis literals cannot be evaluated until the actual shape is known, so we don't consider this to be a true literal
        # for the purposes of evaluating the expression
        self.is_literal = not is_multiaxis_literal and all(
            isinstance(token, int) for token in postfix_expression
        )
        self.is_identifier = is_multiaxis_literal or (
            postfix_expression == [identifier]
        )
        # this is an expression if it's not a literal value, if it's an identifier that points to another dimension, or if it's an identifier that doesn't just point to itself
        self.is_expression = not self.is_literal and (
            len(postfix_expression) > 1 or self.identifier not in postfix_expression
        )
        self.is_multiaxis_literal = is_multiaxis_literal
        self.is_anonymous = is_anonymous
        _logger.debug(
            "Created new %s dimension expression %r",
            "multiaxis" if self.is_multiaxis_literal else "",
            self,
        )

        # ensure we don't have any self-referential expressions
        if self.is_expression and self.identifier in self.parsed_expression:
            msg = f"Self-referential expression {self=}"
            raise SyntaxError(msg)

    @override
    def __repr__(self) -> str:
        """Get a string representation of the dimension expression."""
        if self.is_anonymous:
            return f"Anonymous<{self.identifier}>"
        if self.is_literal:
            return f"Literal<{self.identifier}={self.parsed_expression}>"
        return f"Identifier<{self.identifier}={self.parsed_expression}>"

    @classmethod
    def from_multiaxis_literal(
        cls, identifier: str, literal: int, is_anonymous: bool = False
    ) -> DLTypeDimensionExpression:
        """Create a new dimension expression from a multi-axis literal.

        This is a special case where the expression is a single literal that is repeated across all axes.
        Anonymous axes are a special case where the actual value of the literal is irrelevant.
        """
        return cls(
            identifier, [literal], is_multiaxis_literal=True, is_anonymous=is_anonymous
        )

    def evaluate(self, scope: dict[str, int]) -> int:
        """Evaluate the expression."""
        _logger.debug("Evaluating expression %s with scope %s", self, scope)
        stack: list[int] = []

        if self.is_anonymous:
            msg = "Cannot evaluate an anonymous axis"
            raise ValueError(msg)

        if self.identifier in scope:
            # if the identifier is in the scope, we return the value directly
            # however if we're an anonymous axis, we don't want to return the value directly as the prior scoped value is irrelevant
            return scope[self.identifier]

        for token in self.parsed_expression:
            if isinstance(token, int):
                # literal integer
                stack.append(token)
            elif isinstance(token, str):
                # intentionally allow KeyError to be raised if the identifier is not in the scope
                stack.append(scope[token])
            elif isinstance(token, _DLTypeOperator):  # pyright: ignore[reportUnnecessaryIsInstance] # TODO(DX-2313): Address pyright errors ignored to migrate from mypy # fmt: skip
                b = stack.pop()
                a = stack.pop()
                stack.append(token.evaluate(a, b))
            else:
                msg = f"Invalid token {token=}"
                raise TypeError(msg)

        if len(stack) != 1:
            msg = f"Invalid stack {stack=}"
            raise ValueError(msg)

        _logger.debug("Evaluated expression %r to %r", self, stack[0])
        return stack[0]


def _postfix_from_infix(identifier: str, expression: str) -> DLTypeDimensionExpression:
    """Extract a postfix expression from an infix expression.

    Notably, this function does not handle function calls, which are handled separately.
    """
    _logger.debug("Parsing infix expression %r", expression)

    # this is a modified expression, so we need to handle it differently
    if expression == DLTypeModifier.ANONYMOUS_MULTIAXIS.value:
        return DLTypeDimensionExpression(expression, [], is_anonymous=True)
    if expression.startswith(DLTypeModifier.NAMED_MULTIAXIS.value):
        stripped_expression = expression[len(DLTypeModifier.NAMED_MULTIAXIS.value) :]
        if not VALID_IDENTIFIER_RX.match(stripped_expression):
            msg = f"Invalid identifier {stripped_expression=}"
            raise SyntaxError(msg)
        return DLTypeDimensionExpression(stripped_expression, [stripped_expression])

    split_expression = INFIX_EXPRESSION_SPLIT_RX.split(expression)

    # Convert infix to postfix using shunting yard algorithm
    stack: list[str | _DLTypeOperator] = []
    postfix: list[str | int | _DLTypeOperator] = []
    for token in split_expression:
        if token.isdigit():
            postfix.append(int(token))
        elif token in _valid_operators:
            current_op = _DLTypeOperator(token)

            # Pop operators with higher or equal precedence
            while (
                stack
                and isinstance(stack[-1], _DLTypeOperator)
                and _op_precedence.get(stack[-1], 0)
                >= _op_precedence.get(current_op, 0)
            ):
                postfix.append(stack.pop())

            stack.append(current_op)
        elif VALID_IDENTIFIER_RX.match(token):
            # It's a variable name
            postfix.append(token)
        else:
            msg = f"Invalid expression {expression=}"
            raise SyntaxError(msg)

    # Pop any remaining operators
    while stack:
        postfix.append(stack.pop())

    _logger.debug("Parsed infix expression %r to postfix %r", expression, postfix)
    return DLTypeDimensionExpression(identifier, postfix)


def _maybe_parse_functional_expression(
    identifier: str, expression: str, function: _DLTypeOperator
) -> DLTypeDimensionExpression | None:
    """Parse a function-like expression such as min(a,b) or max(x,y).

    Args:
        identifier: The identifier for the expression (e.g. the name of the dimension)
        expression: The expression to parse
        function: The function operator (_DLTypeOperator.MIN or _DLTypeOperator.MAX)

    Returns:
        A parsed dimension expression if the expression is a valid function call, None otherwise
    """
    if not expression.startswith(f"{function.value}("):
        return None

    # Find balanced closing parenthesis
    # Strip function name and opening parenthesis
    content = expression[len(function.value) + 1 :]

    # Must end with closing parenthesis
    if not content.endswith(")"):
        msg = f"Invalid function expression: {expression}, missing closing parenthesis"
        raise SyntaxError(msg)

    # Remove closing parenthesis
    content = content[:-1]

    # Find the comma that separates arguments (accounting for nesting)
    depth = 0
    comma_index = -1

    for i, char in enumerate(content):
        if char == "(":
            depth += 1
        elif char == ")":
            depth -= 1
            if depth < 0:
                msg = f"Unbalanced parentheses in function expression: {expression}"
                raise SyntaxError(msg)
        elif char == "," and depth == 0:
            comma_index = i
            break

    if comma_index == -1:
        msg = f"Invalid function expression: {expression}, expected two arguments separated by comma"
        raise SyntaxError(msg)

    # Split arguments and parse recursively
    arg1 = content[:comma_index].strip()
    arg2 = content[comma_index + 1 :].strip()

    # Recursively parse both arguments
    expr_1 = expression_from_string(arg1)
    expr_2 = expression_from_string(arg2)

    # Build postfix expression: [arg1 tokens, arg2 tokens, function]
    return DLTypeDimensionExpression(
        identifier, [*expr_1.parsed_expression, *expr_2.parsed_expression, function]
    )


def expression_from_string(expression: str) -> DLTypeDimensionExpression:
    """Parse a dimension expression from a string and return a parsed expression.

    Examples:
        >>> expression_from_string("a+b")
        Identifier<a + b=[a, b, +]>

        >>> expression_from_string("min(a,b)")
        Identifier<min(a, b)=[a, b, min]>

        # literals
        >>> expression_from_string("10")
        Literal<10=[10]>

        >>> expression_from_string("...")
        Anonymous<...>

        >>> expression_from_string("a*10")
        Identifier<a * 10=[a, 10, *]>

        >>> expression_from_string("a*10+b")
        Identifier<a * 10 + b=[a, 10, *, b, +]>

    Args:
        identifier: The identifier for the expression (e.g. the name of the dimension)
        expression: The expression to parse

    Returns:
        A parsed dimension expression.
    """
    if not expression:
        msg = f"Empty expression {expression=}"
        raise SyntaxError(msg)

    # split the expression into the identifier and the expression if it has a specifier
    identifier = expression
    if DLTypeSpecifier.EQUALS.value in expression:
        identifier, expression = expression.split(DLTypeSpecifier.EQUALS.value)

    for function in _functional_operators:
        if result := _maybe_parse_functional_expression(
            identifier, expression, function
        ):
            _logger.debug("Parsed function expression %r", result)
            return result

    if not VALID_EXPRESSION_RX.match(expression):
        msg = f"Invalid {expression=} {VALID_EXPRESSION_RX=}"
        raise SyntaxError(msg)

    # split the expression into tokens using the operators from the enum as delimiters
    return _postfix_from_infix(identifier, expression)

"""A module for parsing dimension shape expressions for dltype."""

from __future__ import annotations

import contextlib
import enum
import math
import re
import typing

from typing_extensions import override

from dltype._lib import _log_utils

_logger: typing.Final = _log_utils.get_logger(__name__)


class _DLTypeSpecifier(enum.Enum):
    """An enum representing a way to specify a name for a dimension expression or literal."""

    EQUALS = "="

    def __repr__(self) -> str:
        return self.value


class _DLTypeGroupToken(enum.Enum):
    """An enum for grouping tokens."""

    LPAREN = "("
    RPAREN = ")"
    COMMA = ","

    def __repr__(self) -> str:
        return self.value


class _DLTypeModifier(enum.Enum):
    """An enum representing a modifier that can be applied to a dimension expression."""

    ANONYMOUS_MULTIAXIS = "..."
    NAMED_MULTIAXIS = "*"

    def __repr__(self) -> str:
        return self.value


class _DLTypeOperator(enum.Enum):
    """
    An enum representing a mathematical operator for a dimension expression.

    NOTE: the ordering here has to be maintained such that any operator that contains another operator appears first.
    """

    MIN = "min"
    MAX = "max"
    ISQRT = "isqrt"
    EQ = "=="
    NE = "!="
    GE = ">="
    LE = "<="
    ADD = "+"
    SUB = "-"
    MUL = "*"
    EXP = "^"
    DIV = "/"
    GT = ">"
    LT = "<"
    MOD = "%"

    def __repr__(self) -> str:
        return self.value

    def evaluate_unary(self, a: int) -> int:
        """Evaluate the unary operator."""
        if self is _DLTypeOperator.ISQRT:
            return math.isqrt(a)
        raise NotImplementedError(self)

    def evaluate(self, a: int, b: int) -> int:  # noqa: C901, PLR0911, PLR0912
        """Evaluate the operator."""
        match self:
            case _DLTypeOperator.GT:
                return int(a > b)
            case _DLTypeOperator.GE:
                return int(a >= b)
            case _DLTypeOperator.LT:
                return int(a < b)
            case _DLTypeOperator.LE:
                return int(a <= b)
            case _DLTypeOperator.EQ:
                return int(a == b)
            case _DLTypeOperator.NE:
                return int(a != b)
            case _DLTypeOperator.ADD:
                return a + b
            case _DLTypeOperator.SUB:
                return a - b
            case _DLTypeOperator.MUL:
                return a * b
            case _DLTypeOperator.EXP:
                return int(a**b)
            case _DLTypeOperator.DIV:
                return a // b
            case _DLTypeOperator.MIN:
                return min(a, b)
            case _DLTypeOperator.MAX:
                return max(a, b)
            case _DLTypeOperator.MOD:
                return a % b
            case _DLTypeOperator.ISQRT:
                msg = f"Invalid unary operator {self=}"
                raise AssertionError(msg)
        raise NotImplementedError(self)


_op_precedence: typing.Final = {
    _DLTypeOperator.EQ: 0,
    _DLTypeOperator.NE: 0,
    _DLTypeOperator.ADD: 1,
    _DLTypeOperator.SUB: 1,
    _DLTypeOperator.MUL: 2,
    _DLTypeOperator.DIV: 2,
    _DLTypeOperator.MOD: 2,
    _DLTypeOperator.EXP: 3,
    _DLTypeOperator.MIN: 4,
    _DLTypeOperator.MAX: 4,
    _DLTypeOperator.GT: 5,
    _DLTypeOperator.GE: 5,
    _DLTypeOperator.LT: 5,
    _DLTypeOperator.LE: 5,
    _DLTypeOperator.ISQRT: 5,
    _DLTypeGroupToken.LPAREN: 6,
}

_unary_functions: typing.Final = frozenset({_DLTypeOperator.ISQRT})
_binary_functions: typing.Final = frozenset({_DLTypeOperator.MIN, _DLTypeOperator.MAX})
_functional_operators: typing.Final = frozenset(_unary_functions.union(_binary_functions))
_infix_operators: typing.Final = frozenset(
    {
        _DLTypeOperator.ADD,
        _DLTypeOperator.SUB,
        _DLTypeOperator.MUL,
        _DLTypeOperator.DIV,
        _DLTypeOperator.EXP,
        _DLTypeOperator.EQ,
        _DLTypeOperator.NE,
        _DLTypeOperator.GT,
        _DLTypeOperator.GE,
        _DLTypeOperator.LT,
        _DLTypeOperator.LE,
        _DLTypeOperator.MOD,
    }
)
_VALID_IDENTIFIER_RX: typing.Final = re.compile(r"^[a-zA-Z][a-zA-Z0-9\_]*$")


class DLTypeDimensionExpression:
    """A class representing a dimension that depends on other dimensions."""

    def __init__(
        self,
        identifier: str,
        postfix_expression: list[str | int | _DLTypeOperator],
        *,
        is_multiaxis_literal: bool = False,
        is_anonymous: bool = False,
        is_named_multiaxis: bool = False,
    ) -> None:
        """Create a new dimension expression."""
        self.identifier = identifier
        self.parsed_expression = postfix_expression
        # multiaxis literals cannot be evaluated until
        # the actual shape is known, so we don't consider this to be a true literal
        # for the purposes of evaluating the expression
        self.is_literal = not is_multiaxis_literal and all(
            isinstance(token, int) for token in postfix_expression
        )

        self.is_identifier = (
            is_multiaxis_literal or is_named_multiaxis or (postfix_expression == [identifier])
        )
        # this is an expression if it's not a literal value, if it's
        # an identifier that points to another dimension, or if it's an
        # identifier that doesn't just point to itself
        self.is_expression = not (self.is_identifier and self.is_literal) and (
            len(postfix_expression) > 1 or self.identifier not in postfix_expression
        )
        self.is_multiaxis_literal = is_multiaxis_literal
        self.is_anonymous = is_anonymous
        self.is_named_multiaxis = is_named_multiaxis

        if (
            not self.is_literal
            and not self.is_anonymous
            and not self.is_named_multiaxis
            and not self.is_multiaxis_literal
        ):
            with contextlib.suppress(KeyError):
                self.evaluate({})
                self.is_literal = True

        _logger.debug(
            "Created new %s dimension expression %r", "multiaxis" if self.is_multiaxis_literal else "", self
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
        cls,
        identifier: str,
        literal: int,
        *,
        is_anonymous: bool = False,
    ) -> DLTypeDimensionExpression:
        """
        Create a new dimension expression from a multi-axis literal.

        This is a special case where the expression is a single literal that is repeated across all axes.
        Anonymous axes are a special case where the actual value of the literal is irrelevant.
        """
        return cls(
            identifier,
            [literal],
            is_multiaxis_literal=True,
            is_anonymous=is_anonymous,
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
            # however if we're an anonymous axis, we don't want to
            # return the value directly as the prior scoped value is irrelevant
            return scope[self.identifier]

        for token in self.parsed_expression:
            if isinstance(token, int):
                # literal integer
                stack.append(token)
            elif isinstance(token, str):
                # intentionally allow KeyError to be raised if the identifier is not in the scope
                stack.append(scope[token])
            elif isinstance(token, _DLTypeOperator):  # pyright: ignore[reportUnnecessaryIsInstance]
                b = stack.pop()
                if token in _unary_functions:
                    stack.append(token.evaluate_unary(b))
                    continue
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


def _maybe_multiaxis(
    identifier: str,
    expression: list[TokenT | str | int],
) -> DLTypeDimensionExpression | None:
    # this is a modified expression, so we need to handle it differently
    if len(expression) == 1 and expression[0] == _DLTypeModifier.ANONYMOUS_MULTIAXIS.value:
        return DLTypeDimensionExpression(identifier, [], is_anonymous=True)

    if len(expression) == 2 and expression[0] == _DLTypeOperator.MUL and isinstance(expression[1], str):  # noqa: PLR2004
        if not _VALID_IDENTIFIER_RX.match(expression[1]):
            msg = f"{expression[1]} is not a valid multiaxis identifier"
            raise SyntaxError(msg)
        return DLTypeDimensionExpression(expression[1], [expression[1]], is_named_multiaxis=True)

    return None


def _flush_op_by_precedence(
    stack: list[str | _DLTypeOperator],
    postfix: list[str | int | _DLTypeOperator],
    current_op: _DLTypeOperator | _DLTypeGroupToken,
) -> None:
    # Pop operators with higher or equal precedence
    while (
        stack
        and (isinstance(stack[-1], _DLTypeOperator | _DLTypeGroupToken))
        and _op_precedence.get(stack[-1], 0) >= _op_precedence.get(current_op, 0)
    ):
        postfix.append(stack.pop())


def _postfix_from_infix(identifier: str, expression: list[TokenT | str | int]) -> DLTypeDimensionExpression:  # noqa: C901, PLR0912, PLR0915
    """
    Extract a postfix expression from an infix expression.

    Notably, this function does not handle function calls, which are handled separately.
    """
    _logger.debug("Parsing infix expression %r", expression)

    if not expression:
        msg = f"Argument list empty ({identifier})"
        raise SyntaxError(msg)

    if maybe_multiaxis := _maybe_multiaxis(identifier, expression):
        return maybe_multiaxis

    # Convert infix to postfix using shunting yard algorithm
    scope_vars: set[str] = set()
    stack: list[str | _DLTypeOperator] = []
    postfix: list[str | int | _DLTypeOperator] = []
    current_index = 0

    while current_index < len(expression):
        token = expression[current_index]

        if isinstance(token, int):
            postfix.append(token)
            current_index += 1
        elif token in _infix_operators:
            current_op = token
            _flush_op_by_precedence(stack, postfix, current_op)

            stack.append(current_op)
            current_index += 1
        elif token in _functional_operators or token == _DLTypeGroupToken.LPAREN:
            current_op = token
            assert isinstance(current_op, _DLTypeGroupToken | _DLTypeOperator)
            _flush_op_by_precedence(stack, postfix, current_op)

            lparen, comma_indices, rparen = _get_group_indices(expression[current_index:], current_index)
            if token in _binary_functions and len(comma_indices) != 1:
                msg = f"{token.value} requires two arguments, received {len(comma_indices) + 1}"
                raise SyntaxError(msg)
            if token in _unary_functions and len(comma_indices) != 0:
                msg = f"{token.value} requires one argument, received {len(comma_indices) + 1}"
                raise SyntaxError(msg)
            if token == _DLTypeGroupToken.LPAREN and len(comma_indices) != 0:
                msg = "Group received invalid comma separator"
                raise SyntaxError(msg)

            lhs = lparen
            for arg_idx in [*comma_indices, rparen]:
                inner_expr = _postfix_from_infix(f"{identifier}[{arg_idx}]", expression[lhs + 1 : arg_idx])
                postfix.extend(inner_expr.parsed_expression)
                scope_vars.update(exp for exp in inner_expr.parsed_expression if isinstance(exp, str))
                lhs = arg_idx

            if current_op in _functional_operators:
                stack.append(current_op)
            current_index = rparen + 1
        elif isinstance(token, str) and _VALID_IDENTIFIER_RX.match(token):
            postfix.append(token)
            scope_vars.add(token)
            current_index += 1
        else:
            msg = f"Invalid expression={identifier} [{token=}] pos={current_index}/{len(expression)}"
            raise SyntaxError(msg)

    # Pop any remaining operators
    while stack:
        postfix.append(stack.pop())

    _logger.debug("Parsed infix expression %r to postfix %r", identifier, postfix)
    return DLTypeDimensionExpression(identifier, postfix)


TokenT: typing.TypeAlias = _DLTypeSpecifier | _DLTypeGroupToken | _DLTypeOperator


def _span_to_str_or_int(span: str) -> str | int:
    if span.isnumeric():
        return int(span)
    return span


def _assert_token_list_valid(tokenized: list[str | int | TokenT]) -> None:
    if len(tokenized) == 0:
        msg = "Empty expression"
        raise SyntaxError(msg)

    if len(tokenized) == 1 and isinstance(tokenized[0], str | int):
        # special case where the only token is an identifier
        return

    if len(tokenized) == 2 and tokenized[0] == _DLTypeOperator.MUL and isinstance(tokenized[1], str):  # noqa: PLR2004
        # is an anonymous named axis
        return

    # other than the special case above, fold the iterated list in to make sure the operators are balanced
    n_expected_args = 1  # expected at least one expression
    n_actual_args = 0

    for tok in reversed(tokenized):
        if tok in _unary_functions:
            n_expected_args += 1
            n_actual_args += 1
        elif tok in _binary_functions | _infix_operators:
            # all operators are binary
            n_expected_args += 2
            n_actual_args += 1
        elif isinstance(tok, str | int):
            n_actual_args += 1
        elif isinstance(tok, _DLTypeGroupToken):
            continue
        else:
            raise SyntaxError(tok)

    if n_expected_args != n_actual_args:
        raise SyntaxError("Invalid expression syntax: " + "".join(map(repr, tokenized)))


def _tokenize_string_expr(  # noqa: C901, PLR0912
    expression: str,
) -> list[str | int | TokenT]:
    return_list: list[str] = []
    current_span = ""

    # first pass just split by character, then we can combine repeated tokens like "==" or ">=" into a single token
    for character in expression:
        if character == " ":
            msg = "Spaces not permitted in dimension expressions"
            raise SyntaxError(msg)

        return_list.append(character)

    # second pass combine repeated tokens like "==" or ">=" into a single token
    second_pass_return_list: list[str | TokenT] = [""] * len(return_list)
    current_span = ""
    offset = 0
    found_operator = False

    for _idx in range(len(return_list)):
        idx = _idx + offset
        if idx >= len(return_list):
            second_pass_return_list = second_pass_return_list[: len(return_list) - offset]
            break
        character = return_list[idx]
        found_operator = False
        assert isinstance(character, str)
        for op in _DLTypeOperator:
            if tuple(expression[idx : idx + len(op.value)]) == tuple(op.value):
                second_pass_return_list[idx - offset] = op
                offset += len(op.value) - 1
                found_operator = True
                break
        if not found_operator:
            for spec in _DLTypeSpecifier:
                if tuple(expression[idx : idx + len(spec.value)]) == tuple(spec.value):
                    second_pass_return_list[idx - offset] = spec
                    offset += len(spec.value) - 1
                    found_operator = True
                    break
        if not found_operator:
            for group in _DLTypeGroupToken:
                if tuple(expression[idx : idx + len(group.value)]) == tuple(group.value):
                    second_pass_return_list[idx - offset] = group
                    offset += len(group.value) - 1
                    found_operator = True
                    break
        if not found_operator:
            second_pass_return_list[idx - offset] = character
    second_pass_return_list = [tok for tok in second_pass_return_list if tok]

    # third pass convert to tokens or strings/ints
    final_return_list: list[str | int | TokenT] = []
    current_span = ""
    for token in second_pass_return_list:
        if isinstance(token, _DLTypeOperator | _DLTypeSpecifier | _DLTypeGroupToken):
            if current_span:
                final_return_list.append(_span_to_str_or_int(current_span))
                current_span = ""
            final_return_list.append(token)
        else:
            current_span += token
    if current_span:
        final_return_list.append(_span_to_str_or_int(current_span))

    _assert_token_list_valid(final_return_list)
    return final_return_list


def _get_group_indices(expr: list[TokenT | int | str], offset: int) -> tuple[int, list[int], int]:
    lparen_idx: int | None = None
    comma_idx: list[int] = []
    rparen_idx: int | None = None
    nesting_depth = 0

    for idx, tok in enumerate(expr):
        if tok == _DLTypeGroupToken.LPAREN:
            nesting_depth += 1
            lparen_idx = idx + offset if nesting_depth == 1 else lparen_idx
        elif tok == _DLTypeGroupToken.COMMA and nesting_depth == 1:
            comma_idx.append(idx + offset)
        elif tok == _DLTypeGroupToken.RPAREN:
            rparen_idx = idx + offset if nesting_depth == 1 else rparen_idx
            nesting_depth -= 1

        if rparen_idx:
            break

    if lparen_idx is None:
        msg = f"Invalid function syntax {expr=}, missing ("
        raise SyntaxError(msg)
    if rparen_idx is None:
        msg = f"Invalid function syntax {expr=}, missing )"
        raise SyntaxError(msg)

    if lparen_idx > rparen_idx or any(c_idx < lparen_idx or c_idx > rparen_idx for c_idx in comma_idx):
        msg = f"Unbalanced parenthesis in expression {''.join(map(repr, expr))}"
        raise SyntaxError(msg)

    return lparen_idx, comma_idx, rparen_idx


def expression_from_string(expression: str) -> DLTypeDimensionExpression:
    """Parse a dimension expression from a string and return a parsed expression."""
    if not expression:
        msg = f"Empty expression {expression=}"
        raise SyntaxError(msg)

    # split the expression into the identifier and the expression if it has a specifier
    identifier = expression
    if _DLTypeSpecifier.EQUALS.value in expression:
        _identifier, _expression = expression.split(_DLTypeSpecifier.EQUALS.value, maxsplit=1)
        if _VALID_IDENTIFIER_RX.match(_identifier) and _expression[0].isalnum():
            # we had an assignment expression, so we can use the identifier and expression as-is
            identifier = _identifier
            expression = _expression
    tokenized = _tokenize_string_expr(expression)
    return _postfix_from_infix(identifier, tokenized)

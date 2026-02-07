"""Allows specifying expressions as symbolic types rather than strings."""

from __future__ import annotations

import math
import typing
from abc import ABC, abstractmethod
from types import EllipsisType

from typing_extensions import override


class OperableAxis(ABC):
    @abstractmethod
    def __str__(self) -> str: ...

    def __repr__(self) -> str:
        return self.__str__()

    def __resolve_expr_sides(
        self,
        other: OperableAxisT,
        *,
        reverse: bool = False,
    ) -> tuple[OperableAxisT | OperableAxis, OperableAxisT | OperableAxis]:
        lhs = other if reverse else self
        rhs = self if reverse else other

        return lhs, rhs

    def __add__(self, other: OperableAxisT) -> ComputedAxis:
        return ComputedAxis(Add(*self.__resolve_expr_sides(other)))

    def __radd__(self, other: OperableAxisT) -> ComputedAxis:
        return ComputedAxis(Add(*self.__resolve_expr_sides(other, reverse=True)))

    def __sub__(self, other: OperableAxisT) -> ComputedAxis:
        return ComputedAxis(Subtract(*self.__resolve_expr_sides(other)))

    def __rsub__(self, other: OperableAxisT) -> ComputedAxis:
        return ComputedAxis(Subtract(*self.__resolve_expr_sides(other, reverse=True)))

    def __mul__(self, other: OperableAxisT) -> ComputedAxis:
        return ComputedAxis(Multiply(*self.__resolve_expr_sides(other)))

    def __rmul__(self, other: OperableAxisT) -> ComputedAxis:
        return ComputedAxis(Multiply(*self.__resolve_expr_sides(other, reverse=True)))

    def __floordiv__(self, other: OperableAxisT) -> ComputedAxis:
        return ComputedAxis(Divide(*self.__resolve_expr_sides(other)))

    def __rfloordiv__(self, other: OperableAxisT) -> ComputedAxis:
        return ComputedAxis(Divide(*self.__resolve_expr_sides(other, reverse=True)))

    def __pow__(self, other: OperableAxisT) -> ComputedAxis:
        return ComputedAxis(Exp(*self.__resolve_expr_sides(other)))

    def __rpow__(self, other: OperableAxisT) -> ComputedAxis:
        return ComputedAxis(Exp(*self.__resolve_expr_sides(other, reverse=True)))


class AxisOperationBase(OperableAxis, ABC):
    @abstractmethod
    def __init__(self) -> None:
        pass


class UnaryAxisOperationBase(AxisOperationBase):
    def __init__(self, axis: Group | OperableAxis | ComputedAxis | int) -> None:
        self._axis = axis if isinstance(axis, OperableAxis | ComputedAxis | Group) else LiteralAxis(axis)


class ISqrt(UnaryAxisOperationBase):
    def __str__(self) -> str:
        if isinstance(self._axis, LiteralAxis):
            return f"{math.isqrt(self._axis.value)}"
        return f"isqrt({self._axis})"


class Group(UnaryAxisOperationBase):
    def __init__(
        self,
        grouped_op: OperableAxis | ComputedAxis | int,
    ) -> None:
        self._operators = grouped_op

    def __str__(self) -> str:
        return f"({self._operators})"


class BinaryAxisOperationBase(AxisOperationBase):
    def __init__(
        self,
        lhs: Group | OperableAxis | ComputedAxis | int,
        rhs: Group | OperableAxis | ComputedAxis | int,
    ) -> None:
        self._lhs = lhs if isinstance(lhs, OperableAxis | ComputedAxis | Group) else LiteralAxis(lhs)
        self._rhs = rhs if isinstance(rhs, OperableAxis | ComputedAxis | Group) else LiteralAxis(rhs)


class Add(BinaryAxisOperationBase):
    def __str__(self) -> str:
        if isinstance(self._lhs, LiteralAxis) and isinstance(self._rhs, LiteralAxis):
            return f"{self._lhs.value + self._rhs.value}"
        return f"{self._lhs}+{self._rhs}"


class Subtract(BinaryAxisOperationBase):
    def __str__(self) -> str:
        if isinstance(self._lhs, LiteralAxis) and isinstance(self._rhs, LiteralAxis):
            return f"{self._lhs.value - self._rhs.value}"
        return f"{self._lhs}-{self._rhs}"


class Divide(BinaryAxisOperationBase):
    def __str__(self) -> str:
        if isinstance(self._lhs, LiteralAxis) and isinstance(self._rhs, LiteralAxis):
            return f"{self._lhs.value // self._rhs.value}"
        return f"{self._lhs}/{self._rhs}"


class Multiply(BinaryAxisOperationBase):
    def __str__(self) -> str:
        if isinstance(self._lhs, LiteralAxis) and isinstance(self._rhs, LiteralAxis):
            return f"{self._lhs.value * self._rhs.value}"
        return f"{self._lhs}*{self._rhs}"


class Exp(BinaryAxisOperationBase):
    def __str__(self) -> str:
        if isinstance(self._lhs, LiteralAxis) and isinstance(self._rhs, LiteralAxis):
            return f"{self._lhs.value**self._rhs.value}"
        return f"{self._lhs}^{self._rhs}"


class Max(BinaryAxisOperationBase):
    def __str__(self) -> str:
        if isinstance(self._lhs, LiteralAxis) and isinstance(self._rhs, LiteralAxis):
            return f"{max(self._lhs.value, self._rhs.value)}"
        return f"max({self._lhs},{self._rhs})"


class Min(BinaryAxisOperationBase):
    def __str__(self) -> str:
        if isinstance(self._lhs, LiteralAxis) and isinstance(self._rhs, LiteralAxis):
            return f"{min(self._lhs.value, self._rhs.value)}"
        return f"min({self._lhs},{self._rhs})"


class LiteralAxis(OperableAxis):
    def __init__(self, value: int) -> None:
        """Initialize an axis with a literal integer value."""
        self._value = value

    @property
    def value(self) -> int:
        return self._value

    def __str__(self) -> str:
        return str(self._value)


class VariableAxis(OperableAxis):
    def __init__(self, identifier: str) -> None:
        """Initialize an axis with an identifier string."""
        self._identifier = identifier

    def __str__(self) -> str:
        return str(self._identifier)


class ComputedAxis(OperableAxis):
    def __init__(self, computation: AxisOperationBase) -> None:
        self._computation = computation

    def __str__(self) -> str:
        return f"{self._computation}"

    def __repr__(self) -> str:
        return self.__str__()


class NamedComputedAxis(ComputedAxis):
    def __init__(self, identifier: str, computation: AxisOperationBase) -> None:
        super().__init__(computation)
        self._identifier = identifier

    @property
    def identifier(self) -> str:
        return self._identifier

    @override
    def __str__(self) -> str:
        return f"{self._identifier}={self._computation}"


OperableAxisT: typing.TypeAlias = Group | LiteralAxis | VariableAxis | ComputedAxis | int


class ConstantAxis:
    def __init__(self, identifier: str, value: int) -> None:
        """Initialize a symbol with an identifier string equal to a constant value i.e. batch=3."""
        self._identifier = str(identifier)
        self._value = value

    @property
    def value(self) -> int:
        return self._value

    @property
    def identifier(self) -> str:
        return self._identifier

    def __str__(self) -> str:
        return f"{self._identifier}={self._value}"


class AnonymousAxis:
    def __init__(self, maybe_name: str | EllipsisType) -> None:
        """Initialize an axis or set of axes with zero or more values, optionally give a name."""
        self._identifier = maybe_name

    def __str__(self) -> str:
        return ("*" + str(self._identifier)) if isinstance(self._identifier, str) else "..."

    def __repr__(self) -> str:
        return self.__str__()


AxisT: typing.TypeAlias = OperableAxisT | AnonymousAxis | ConstantAxis
ExpressionComponentT = AxisT | EllipsisType


class Shape:
    """The expression of tensor shape as a sequence of expressions."""

    def __init__(self, symbols: tuple[ExpressionComponentT, ...] | ExpressionComponentT) -> None:
        _symbols = symbols if isinstance(symbols, tuple) else (symbols,)
        self._raveled_expressions = [
            (AnonymousAxis(...) if isinstance(symbol, EllipsisType) else symbol) for symbol in _symbols
        ]

    def __str__(self) -> str:
        return " ".join(map(str, self._raveled_expressions))

    def __repr__(self) -> str:
        return self.__str__()

    @classmethod
    def __class_getitem__(cls, args: tuple[ExpressionComponentT, ...]) -> Shape:
        return cls(args)

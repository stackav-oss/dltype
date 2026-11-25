"""Tests for torch.compile compatibility."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Annotated, NamedTuple

import pytest
import torch

import dltype


@dltype.dltyped_namedtuple()
class _TorchCompileNamedTuple(NamedTuple):
    xyz: Annotated[torch.Tensor, dltype.FloatTensor["batch features"]]


@dltype.dltyped_dataclass()
@dataclass
class _TorchCompileDataclass:
    xyz: Annotated[torch.Tensor, dltype.FloatTensor["batch features"]]


def _assert_compiled_matches_eager(
    eager_fn: Callable[[torch.Tensor], torch.Tensor],
    compiled_fn: Callable[[torch.Tensor], torch.Tensor],
) -> None:
    eager_input = torch.randn(2, 3, requires_grad=True)
    compiled_input = eager_input.detach().clone().requires_grad_()

    eager_output = eager_fn(eager_input)
    compiled_output = compiled_fn(compiled_input)
    eager_gradient = torch.autograd.grad(eager_output.sum(), eager_input)[0]
    compiled_gradient = torch.autograd.grad(compiled_output.sum(), compiled_input)[0]

    torch.testing.assert_close(compiled_output, eager_output)
    torch.testing.assert_close(compiled_gradient, eager_gradient)


def test_torch_compile_fullgraph_dltyped_function() -> None:
    @dltype.dltyped()
    def square(
        tensor: Annotated[torch.Tensor, dltype.FloatTensor["batch features"]],
    ) -> Annotated[torch.Tensor, dltype.FloatTensor["batch features"]]:
        return tensor.square()

    eager_tensor = torch.randn(2, 3, requires_grad=True)
    square(eager_tensor)
    with pytest.raises(dltype.DLTypeNDimsError):
        square(torch.randn(2, 3, 1))

    # fullgraph=True makes unsupported warning handling fail instead of silently
    # introducing a graph break.
    compiled_square = torch.compile(square, fullgraph=True, backend="eager")
    _assert_compiled_matches_eager(square, compiled_square)


def test_torch_compile_fullgraph_dltyped_self_scope() -> None:
    class ScopedSquare(torch.nn.Module):
        def get_dltype_scope(self) -> dict[str, int]:
            return {"features": 3}

        @dltype.dltyped("self")
        def forward(
            self,
            tensor: Annotated[torch.Tensor, dltype.FloatTensor["batch features"]],
        ) -> Annotated[torch.Tensor, dltype.FloatTensor["batch features"]]:
            return tensor.square()

    eager_module = ScopedSquare()
    eager_module(torch.randn(2, 3))
    with pytest.raises(dltype.DLTypeShapeError):
        eager_module(torch.randn(2, 4))

    # fullgraph=True makes unsupported Protocol checks fail instead of silently
    # introducing a graph break.
    compiled_module = torch.compile(eager_module, fullgraph=True, backend="eager")
    _assert_compiled_matches_eager(eager_module, compiled_module)


def test_torch_compile_fullgraph_dltyped_namedtuple() -> None:
    def scale(tensor: torch.Tensor) -> torch.Tensor:
        return _TorchCompileNamedTuple(xyz=tensor).xyz * 2

    compiled_scale = torch.compile(scale, fullgraph=True, backend="eager")
    _assert_compiled_matches_eager(scale, compiled_scale)


def test_torch_compile_fullgraph_dltyped_dataclass() -> None:
    def scale(tensor: torch.Tensor) -> torch.Tensor:
        return _TorchCompileDataclass(xyz=tensor).xyz * 2

    compiled_scale = torch.compile(scale, fullgraph=True, backend="eager")
    _assert_compiled_matches_eager(scale, compiled_scale)

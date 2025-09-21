"""Benchmark dltype vs. beartype vs. manual checking vs. baseline."""

from contextlib import suppress
from enum import Enum, auto
from inspect import signature
from typing import Annotated, Final, NamedTuple

import torch
from torch.utils.benchmark import Measurement, Timer

import dltype


class BenchmarkMode(str, Enum):
    """What conditions to apply to the benchmark arguments."""

    correct = auto()
    incorrect_shape = auto()
    incorrect_datatype = auto()
    incorrect_shape_and_datatype = auto()


class SetupTensors(NamedTuple):
    """Collection of tensors that will become the function arguments for benchmark code."""

    tensor_a: torch.Tensor | None
    tensor_b: torch.Tensor | None
    tensor_c: torch.Tensor | None


def setup_code(mode: BenchmarkMode) -> SetupTensors:
    """Set up tensors for the benchmark code."""
    match mode:
        case BenchmarkMode.correct:
            return SetupTensors(
                tensor_a=torch.rand(8, 2, 3, 4),
                tensor_b=torch.rand(8, 2, 3, 4),
                tensor_c=torch.rand(8, 2, 3, 4),
            )
        case BenchmarkMode.incorrect_shape:
            return SetupTensors(
                tensor_a=torch.rand(8, 2, 3, 4),
                tensor_b=torch.rand(7, 2, 3, 4),
                tensor_c=torch.rand(8, 2, 3),
            )
        case BenchmarkMode.incorrect_datatype:
            return SetupTensors(
                tensor_a=torch.rand(8, 2, 3, 4).int(),
                tensor_b=torch.rand(8, 2, 3, 4).int(),
                tensor_c=torch.rand(8, 2, 3, 4).int(),
            )
        case BenchmarkMode.incorrect_shape_and_datatype:
            return SetupTensors(
                tensor_a=torch.rand(8, 2, 3, 4).int(),
                tensor_b=None,
                tensor_c=torch.rand(8, 2, 3).int(),
            )


class BenchmarkParams(NamedTuple):
    """Parameters for a benchmark run."""

    mode: BenchmarkMode
    function_name: str
    function_args: tuple[str, ...] | None
    add_decorator: bool
    expected_error: type[Exception] | None


class BenchmarkResult(NamedTuple):
    """Result of a benchmark run."""

    params: BenchmarkParams
    measurement: Measurement


class BenchmarkFunc:
    """A dltype benchmark function taking params and returning a result of that benchmark when called."""

    def __init__(self, params: BenchmarkParams) -> None:
        """Create a new benchmark function."""
        suppressed_prefix = (
            f"with {suppress.__name__}({params.expected_error.__name__}): " if params.expected_error else ""
        )
        tensor_args = ", ".join(SetupTensors._fields)
        maybe_decorated_function = (
            f"dltype.dltyped()({params.function_name})" if params.add_decorator else f"{params.function_name}"
        )
        bench = (
            f"{maybe_decorated_function}({', '.join(params.function_args) if params.function_args else ''})"
        )

        self._timer = Timer(
            setup=f"{tensor_args} = setup_code(BenchmarkMode.{params.mode.name})",
            stmt=f"{suppressed_prefix}{bench}",
            globals=globals()
            | ({params.expected_error.__name__: params.expected_error} if params.expected_error else {}),
        )
        self._params = params

    def __call__(self) -> BenchmarkResult:
        """Run the benchmark and return the result."""
        print(f"running bench={self._params.mode=} {self._params.function_name=}")  # noqa: T201
        return BenchmarkResult(self._params, self._timer.adaptive_autorange())


def baseline(
    tensor_a: torch.Tensor,
    tensor_b: torch.Tensor,
    tensor_c: torch.Tensor,
) -> torch.Tensor:
    """A function that takes a tensor and returns a tensor."""
    return (tensor_a * tensor_b + tensor_c).permute(2, 3, 0, 1)


def dltype_function(
    tensor_a: Annotated[torch.Tensor, dltype.FloatTensor["b c h w"]],
    tensor_b: Annotated[torch.Tensor, dltype.FloatTensor["b c h w"]],
    tensor_c: Annotated[torch.Tensor, dltype.FloatTensor["b c h w"]],
) -> Annotated[torch.Tensor, dltype.FloatTensor["h w b c"]]:
    """A function that takes a tensor and returns a tensor."""
    return (tensor_a * tensor_b + tensor_c).permute(2, 3, 0, 1)


@dltype.dltyped()
def dltype_decorated(
    tensor_a: Annotated[torch.Tensor, dltype.FloatTensor["b c h w"]],
    tensor_b: Annotated[torch.Tensor, dltype.FloatTensor["b c h w"]],
    tensor_c: Annotated[torch.Tensor, dltype.FloatTensor["b c h w"]],
) -> Annotated[torch.Tensor, dltype.FloatTensor["h w b c"]]:
    """A function that takes a tensor and returns a tensor."""
    return (tensor_a * tensor_b + tensor_c).permute(2, 3, 0, 1)


def manual_shape_check(
    tensor_a: torch.Tensor,
    tensor_b: torch.Tensor,
    tensor_c: torch.Tensor,
) -> torch.Tensor:
    """A function that takes a tensor and returns a tensor."""
    if not all(
        isinstance(tensor, torch.Tensor)  # pyright: ignore[reportUnnecessaryIsInstance]
        for tensor in (tensor_a, tensor_b, tensor_c)
    ):
        msg = "Tensors must have type=torch.Tensor."
        raise TypeError(msg)
    shapes = (tensor_a.shape, tensor_b.shape, tensor_c.shape)
    if not all(tensor.dtype == torch.float32 for tensor in (tensor_a, tensor_b, tensor_c)):
        msg = "Tensors must have dtype=torch.float32."
        raise TypeError(msg)
    if {len(shape) for shape in shapes} != {4}:
        msg = "Shapes must have the same number of dimensions=4."
        raise TypeError(msg)
    if all(shape == shapes[0] for shape in shapes):
        return (tensor_a * tensor_b + tensor_c).permute(2, 3, 0, 1)
    msg = "Shapes must be equal."
    raise TypeError(msg)


# a dltyped function with an expression equivalent to the jaxtyped_with_expression function
@dltype.dltyped()
def dltyped_with_expression(
    tensor_a: Annotated[torch.Tensor, dltype.FloatTensor["b c h w"]],
    tensor_b: Annotated[torch.Tensor, dltype.FloatTensor["b1 c1 h1 w1"]],
) -> Annotated[torch.Tensor, dltype.FloatTensor["b max(c-1,0) w+h1"]]:
    """A function that takes a tensor and returns a tensor."""
    # Use c-1 dimension as specified in return type
    reduced_c = max(tensor_a.shape[1] - 1, 0)

    w_plus_h = tensor_a.shape[3] + tensor_b.shape[2]

    return torch.zeros(
        tensor_a.shape[0],  # b
        reduced_c,  # c-1
        w_plus_h,  # w+h1
    )


def expression_baseline(
    tensor_a: torch.Tensor,
    tensor_b: torch.Tensor,
) -> torch.Tensor:
    """A function that takes a tensor and returns a tensor."""
    # Use c-1 dimension as specified in return type
    reduced_c = max(tensor_a.shape[1] - 1, 0)

    w_plus_h = tensor_a.shape[3] + tensor_b.shape[2]

    return torch.zeros(
        tensor_a.shape[0],  # b
        reduced_c,  # c-1
        w_plus_h,  # w+h1
    )


def anonymous_axis_baseline(
    tensor_a: torch.Tensor,
) -> torch.Tensor:
    """A function that takes a tensor and cats it."""
    return torch.stack([tensor_a, tensor_a], dim=1)


@dltype.dltyped()
def dltyped_anonymous_axis(
    tensor_a: Annotated[torch.Tensor, dltype.FloatTensor["*batch h w"]],
) -> Annotated[torch.Tensor, dltype.FloatTensor["*batch 2 h w"]]:
    """A function that takes a tensor and adds a second dimension to it."""
    return torch.stack([tensor_a, tensor_a], dim=1)


if __name__ == "__main__":
    all_functions: Final = [
        baseline,
        manual_shape_check,
        dltype_function,
        dltype_decorated,
        expression_baseline,
        dltyped_with_expression,
        anonymous_axis_baseline,
        dltyped_anonymous_axis,
    ]
    needs_decorator = frozenset({dltype_function})
    error_override = {
        manual_shape_check.__name__: {
            BenchmarkMode.incorrect_shape: TypeError,
            BenchmarkMode.incorrect_datatype: TypeError,
            BenchmarkMode.incorrect_shape_and_datatype: TypeError,
        },
        baseline.__name__: {
            BenchmarkMode.incorrect_shape: RuntimeError,
            BenchmarkMode.incorrect_datatype: RuntimeError,
            BenchmarkMode.incorrect_shape_and_datatype: TypeError,
        },
        expression_baseline.__name__: {
            BenchmarkMode.incorrect_shape_and_datatype: AttributeError,
        },
        anonymous_axis_baseline.__name__: {
            BenchmarkMode.incorrect_shape_and_datatype: TypeError,
        },
    }

    all_benchmarks: dict[BenchmarkMode, dict[str, BenchmarkFunc]] = {}

    for mode in BenchmarkMode:
        for func in all_functions:
            expected_error = None
            match mode:
                case BenchmarkMode.correct:
                    expected_error = None
                case BenchmarkMode.incorrect_shape:
                    expected_error = dltype.DLTypeShapeError
                case BenchmarkMode.incorrect_datatype:
                    expected_error = dltype.DLTypeDtypeError
                case BenchmarkMode.incorrect_shape_and_datatype:
                    expected_error = dltype.DLTypeError

            if func.__name__ in error_override:
                expected_error = error_override[func.__name__].get(mode, expected_error)

            all_benchmarks.setdefault(mode, {})[func.__name__] = BenchmarkFunc(
                BenchmarkParams(
                    mode=mode,
                    function_name=func.__name__,
                    function_args=tuple(map(str, signature(func).parameters.keys())),
                    add_decorator=func in needs_decorator,
                    expected_error=expected_error,
                ),
            )

    summary_results: dict[BenchmarkMode, dict[str, BenchmarkResult]] = {}

    for mode, mode_runs in all_benchmarks.items():
        for func_name, benchmark in mode_runs.items():
            summary_results.setdefault(mode, {})[func_name] = benchmark()

    for results in summary_results.values():
        for result in results.values():
            print("-" * 10)  # noqa: T201
            print(f"Function: {result.params.function_name} Setup: {result.params.mode}")  # noqa: T201
            print(result.measurement)  # noqa: T201
            print("-" * 10)  # noqa: T201

    max_func_length = max(len(func.__name__) + 3 for func in all_functions)
    max_mode_length = max(len(mode.name) + 3 for mode in BenchmarkMode)

    print(f"{'Benchmark':<{max_func_length}}", end="")  # noqa: T201
    for mode in BenchmarkMode:
        print(f"{mode.name:>{max_mode_length}} ", end="")  # noqa: T201

    for func in all_functions:
        print()  # noqa: T201
        print(f"{func.__name__:<{max_func_length}}", end="")  # noqa: T201
        for mode in BenchmarkMode:
            result = summary_results[mode][func.__name__]
            value = f"{result.measurement.mean * 1e6:.2f} uS"
            print(f"{value:>{max_mode_length}}", end="")  # noqa: T201

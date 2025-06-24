"""Benchmark dltype vs. beartype vs. manual checking vs. baseline."""

from contextlib import suppress
from typing import Annotated

import torch
from torch.utils.benchmark import Measurement, Timer

from dltype import dltype


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
    if not all(isinstance(tensor, torch.Tensor) for tensor in (tensor_a, tensor_b, tensor_c)):  # pyright: ignore[reportUnnecessaryIsInstance] # TODO(DX-2313): Address pyright errors ignored to migrate from mypy # fmt: skip
        msg = "Tensors must have type=torch.Tensor."
        raise TypeError(msg)
    shapes = (tensor_a.shape, tensor_b.shape, tensor_c.shape)
    if not all(
        tensor.dtype == torch.float32 for tensor in (tensor_a, tensor_b, tensor_c)
    ):
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
) -> Annotated[torch.Tensor, dltype.FloatTensor["*batch 1 h w"]]:
    """A function that takes a tensor and adds a second dimension to it."""
    return torch.stack([tensor_a, tensor_a], dim=1)


if __name__ == "__main__":
    setup_codes = [
        (
            "correct",
            """
        tensor_a = torch.rand(8, 2, 3, 4)
        tensor_b = torch.rand(8, 2, 3, 4)
        tensor_c = torch.rand(8, 2, 3, 4)
        """,
        ),
        (
            "incorrect shape",
            """
        tensor_a = torch.rand(8, 2, 3, 4)
        tensor_b = torch.rand(7, 2, 3, 4)
        tensor_c = torch.rand(8, 2, 3)
        """,
        ),
        (
            "incorrect data type",
            """
        tensor_a = torch.rand(8, 2, 3, 4).int()
        tensor_b = torch.rand(8, 2, 3, 4).int()
        tensor_c = torch.rand(8, 2, 3, 4).int()
        """,
        ),
        (
            "incorrect shape and data type",
            """
        tensor_a = torch.rand(8, 2, 3, 4).int()
        tensor_b = None
        tensor_c = torch.rand(8, 2, 3).int()
        """,
        ),
    ]

    summary_results: list[dict[str, Measurement]] = []

    for bench_name, setup_code in setup_codes:
        all_benchmarks: dict[str, Measurement] = {}
        # Benchmark basic function using torch.benchmark
        all_benchmarks["baseline"] = Timer(
            setup=setup_code,
            stmt="with suppress(RuntimeError, TypeError): baseline(tensor_a, tensor_b, tensor_c)",
            globals=globals(),
        ).adaptive_autorange()

        # # Benchmark manual checking using torch.benchmark
        all_benchmarks["manual_shape_check"] = Timer(
            setup=setup_code,
            stmt="with suppress(TypeError): manual_shape_check(tensor_a, tensor_b, tensor_c)",
            globals=globals(),
        ).adaptive_autorange()

        # # Benchmark dltype function using torch.benchmark
        all_benchmarks["dltype_with_overhead"] = Timer(
            setup=setup_code,
            stmt="with suppress(TypeError): dltype.dltyped()(dltype_function)(tensor_a, tensor_b, tensor_c)",
            globals=globals(),
        ).adaptive_autorange()

        ## Benchmark pre-annotated dltype function using torch.benchmark
        all_benchmarks["dltype_decorated"] = Timer(
            setup=setup_code,
            stmt="with suppress(TypeError): dltype_decorated(tensor_a, tensor_b, tensor_c)",
            globals=globals(),
        ).adaptive_autorange()

        all_benchmarks["expression_baseline"] = Timer(
            setup=setup_code,
            stmt="with suppress(Exception): expression_baseline(tensor_a, tensor_b)",
            globals=globals(),
        ).adaptive_autorange()

        all_benchmarks["dltyped_with_expression"] = Timer(
            setup=setup_code,
            stmt="with suppress(TypeError): dltyped_with_expression(tensor_a, tensor_b)",
            globals=globals(),
        ).adaptive_autorange()

        all_benchmarks["anonymous_axis_baseline"] = Timer(
            setup=setup_code,
            stmt="with suppress(Exception): anonymous_axis_baseline(tensor_a)",
            globals=globals(),
        ).adaptive_autorange()

        all_benchmarks["dltyped_anonymous_axis"] = Timer(
            setup=setup_code,
            stmt="with suppress(TypeError): dltyped_anonymous_axis(tensor_a)",
            globals=globals(),
        ).adaptive_autorange()

        print("================================================================")
        for name, benchmark in all_benchmarks.items():
            print("----------------------------------------------------------------")
            print(f"Function: {name} Setup: {bench_name}")
            print(benchmark)
        print("================================================================")
        summary_results.append(all_benchmarks)

    # print a table with the function benchmarked on the top row and setup on the left column

    # Extract function names and setup names
    functions = list(summary_results[0].keys())
    setups = [setup[0] for setup in setup_codes]

    # Define a consistent column width
    func_col_width = 25
    setup_col_width = 20

    # Print header row
    print(f"{'Function':<{func_col_width}}", end="")
    for setup in setups:
        print(f"{setup:<{setup_col_width}}", end="")
    print()

    # Print separator
    print("-" * func_col_width, end="")
    for _ in setups:
        print("-" * setup_col_width, end="")
    print()

    # Print data rows (now each row is a function)
    for func in functions:
        print(f"{func:<{func_col_width}}", end="")
        for i, _ in enumerate(setups):
            # Get the measurement for this function and setup
            measurement = summary_results[i][func]
            mean_time_us = measurement.mean * 1e6  # Convert to microseconds
            # Format the value with consistent width
            value = f"{mean_time_us:.3f} uS"
            print(f"{value:<{setup_col_width}}", end="")
        print()

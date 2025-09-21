# DL Type (Deep Learning Type Library)

This typing library is intended to replace jaxtyping for runtime type checking of torch tensors and numpy arrays.

In particular, we support two functions that beartype/jaxtype do not:

1. Support for torch.jit.script/torch.compile/torch.jit.trace
2. Pydantic model type annotations for torch tensors.

## Features

- Shape and Type Validation: Validate tensor shapes and types at runtime with symbolic dimension support.
- Pydantic Integration: First-class support for tensor validation in Pydantic models.
- Context-Aware Validation: Ensures consistency across multiple tensors in the same context.
- ONNX/torch.compile Compatible: Works seamlessly with model export and compilation workflows.
- Symbolic Dimensions: Support for named dimensions that enforce consistency.

## Installation

Install dltype through pip
```bash
pip3 install dltype
```

> [!NOTE]
> dltype does not depend explicitly on torch or numpy, but you must have at least one of them installed at import time otherwise the import will fail.

## Usage

Type hints are evaluated in a context in source-code order, so any references to dimension symbols must exist before an expression is evaluated.

## Supported syntax

DL Type supports four types of dimension specifications:

### Scalars

Single element tensors with no shape

```python
IntTensor[None] # An integer tensor with a single value and no axes
```

### Literal Dimensions

Simple integer dimensions with fixed sizes:

```python
FloatTensor["3 5"]  # A tensor with shape (3, 5)
FloatTensor["batch channels=3 height width"] # identifiers set to dimensions for documentation
```

### Expressions

Mathematical expressions combining literals and symbols.

```python
FloatTensor["batch channels*2"]  # If channels=64, shape would be (batch, 128)
FloatTensor["batch-1"]  # One less than the batch dimension
FloatTensor["features/2"]  # Half the features dimension
```

#### Supported Operators and Functions

> [!NOTE]
> Expressions _must_ never have spaces.

##### Operators

- `+` Addition
- `-` Subtraction
- `*` Multiplication
- `/` Integer division
- `^` Exponentiation

##### Functions

- `min(a,b)` Minimum of two expressions
- `max(a,b)` Maximum of two expressions

> [!WARNING]
> While nested function calls like `min(max(a,b),c)` are supported,
> combining function calls with other operators in the same expression
> (e.g., `min(1,batch)+max(2,channels)`) is not supported to simplify parsing.

### Symbolic Dimensions

Symbolic Dimensions
Named dimensions that ensure consistency across tensors:

```python
FloatTensor["batch channels"]  # A tensor with two dimensions
```

### Multi Dimensions

Named or anonymous dimension identifiers that may cover zero or more dimensions in the actual tensors.
Only one multi-dimension identifier is allowed per type hint.

```python
FloatTensor["... channels h w"] # anonymous dimension will not be matched across tensors
DoubleTensor["batch *channels features"] # named dimension which can be matched across tensors
```

## Argument and return typing

```python
from typing import Annotated
import torch
from dltype import FloatTensor, dltyped

@dltyped()
def add_tensors(
    x: Annotated[torch.Tensor, FloatTensor["batch features"]],
    y: Annotated[torch.Tensor, FloatTensor["batch features"]]
) -> Annotated[torch.Tensor, FloatTensor["batch features"]]:
    return x + y
```

## Pydantic model typing

```python
from typing import Annotated
from pydantic import BaseModel
import torch
from dltype import FloatTensor, IntTensor

class ImageBatch(BaseModel):
    # note the parenthesis instead of brackets for pydantic models
    images: Annotated[torch.Tensor, FloatTensor("batch 3 height width")]
    labels: Annotated[torch.Tensor, IntTensor("batch")]

    # All tensor validations happen automatically
    # Shape consistency is enforced across fields
```

## NamedTuple typing

We expose `@dltyped_namedtuple()` for NamedTuples.
`NamedTuples` are validated upon construction, beware that assignments or manipulations after construction are unchecked.

```python
@dltype.dltyped_namedtuple()
class MyNamedTuple(NamedTuple):
    tensor: Annotated[torch.Tensor, dltype.FloatTensor["b c h w"]]
    mask: Annotated[torch.Tensor, dltype.IntTensor["b h w"]]
    other: int
```

## @dataclass support

Similar to `NamedTuples` and pydantic `BaseModels`, `@dataclasses` may be decorated and validated.
The normal caveats apply in that we only validate at construction and not on assignment.
Therefore, we recommend using frozen `@dataclasses` when possible.

```python
from typing import Annotated
import torch
from dltype import FloatTensor, IntTensor, dltyped_dataclass

# order is important, we raise an error if dltyped_dataclass is applied below dataclass
# this is because the @dataclass decorator applies a bunch of source code modification that we don't want to have to hack around
@dltyped_dataclass()
@dataclass(frozen=True, slots=True)
class MyDataclass:
    images: Annotated[torch.Tensor, FloatTensor["batch 3 height width"]]
    labels: Annotated[torch.Tensor, IntTensor["batch"]]
```

## Optionals

We have no support for general unions of types to prevent confusing behavior when using runtime shape checking.
DLType only supports optional types (i.e. `Type | None`).
To annotate a tensor as being optional, see the example below.

```python
@dltype.dltyped()
def optional_tensor_func(tensor: Annotated[torch.Tensor, dltype.FloatTensor["b c h w"]] | None) -> torch.Tensor:
    if tensor is None:
        return torch.zeros(1, 3, 5, 5)
    return tensor
```

## Numpy and Tensor Mixing

```python
from typing import Annotated
import torch
import numpy as np
from dltype import FloatTensor, dltyped

@dltyped()
def transform_tensors(
    points: Annotated[np.ndarray, FloatTensor["N 3"]]
    transform: Annotated[torch.Tensor, FloatTensor["3 3"]]
) -> Annotated[torch.Tensor, FloatTensor["N 3"]]:
    return torch.from_numpy(points) @ transform
```

## Providing External Scope

There are situations that a runtime variable may influence the expected shape of a tensor.
To provide external scope to be used by dltype, you may implement the `DLTypeScopeProvider` protocol.
There are two flavors of this, one for methods, the other for free functions, both are shown below.
Using external scope providers for free functions is not an encouraged use case as it encourages keeping global state.
Additionally, free functions are generally stateless but this makes the type checking logic stateful and thus
makes the execution of the function impure.
We support this because there are certain scenarios where loading a configuration from a file and providing it as an expected dimension for some typed function may be useful and necessary.

```python

# Using `self` as the DLTypeScopeProvider in an object (this is the primary use case)
class MyModule(nn.Module):
    # ... some implementation details
    def __init__(self, config: MyConfig) -> None:
        self.cfg = config

    # the DLTypeScopeProvider protocol requires this function to be specified.
    def get_dltype_scope(self) -> dict[str, int]:
        """Return the DLType scope which is simply a dictionary of 'axis-name' -> dimension size."""
        return {"in_channel": self.cfg.in_channel}

    # "self" is a literal, not a string -- pyright will yell at you if this is wrong.
    # The first argument of the decorated function will be checked to obey the protocol before calling `get_dltype_scope`.
    @dltyped("self")
    def forward(
        self,
        tensor_1: Annotated[torch.Tensor, FloatTensor["batch num_voxel_features z y x"]],
        # NOTE: in_channel comes from the external scope and is used in the expression below to evaluate the 'channels' expected dimension
        tensor_2: Annotated[torch.Tensor, FloatTensor["batch channels=in_channel-num_voxel_features z y x"]]
    ) -> torch.Tensor:

## Using a scope provider for a free function

class MyProvider:
    def get_dltype_scope(self) -> dict[str, int]:
        # load some_value from a config file in the constructor
        # or fetch it from a singleton
        return {
            "dim1": self.some_value
        }

@dltyped(provider=MyProvider())
def free_function(tensor: FloatTensor["batch dim1"]) -> None:
    # ... implementation details, dim1 provided by the external scope
```

## Supported Types

- `FloatTensor`: For any precision floating point tensor. Is a superset of the following:
    - `Float16Tensor`: For any 16 bit floating point type. Is a superset of the following:
        - `IEEE754HalfFloatTensor`: For 16 bit floating point types that comply with the IEE 754 half-precision specification (notably, does not include `bfloat16`). For numpy tensors `Float16Tensor` is equal to `IEEE754HalfFloatTensor`. Use if you need to forbid usage of `bfloat16` for some reason. Otherwise prefer the `Float16Tensor` type for usage with mixed precision codebases.
        - `BFloat16Tensor`: For 16 bit floating point tensors following the [`bfloat16` format](https://en.wikipedia.org/wiki/Bfloat16_floating-point_format). Is not IEEE 754 compliant and is not supported by NumPy. Use if you need to write code that is `bfloat16` specific, otherwise prefer `Float16Tensor` for usage with a mixed precision instruction scope (such as `torch.amp`).
    - `Float32Tensor`: For single precision 32 bit floats.
    - `Float64Tensor`: For double precision 64 bit floats. Aliases to `DoubleTensor`.
    - Note that `np.float128` and `np.longdouble` will be considered as `FloatTensors` BUT do not exist as standalone types to be used by `dltype` ie. there is no `Float128Tensor` type. These types are not supported by torch, and only supported by numpy on certain platforms, thus we only "support" them insofar as they are considered floating point types.
- `IntTensor`: For integer tensors of any precision. Is a superset of the following:
    - `Int8Tensor`
    - `Int16Tensor`
    - `Int32Tensor`
    - `Int64Tensor`
- `BoolTensor`: For boolean tensors
- `TensorTypeBase`: Base class for any tensor which does not enforce any specific datatype, feel free to add custom validation logic by overriding the `check` method.

## Limitations

- In the current implementation, _every_ call will be checked, which may or may not be slow depending on how big the context is (it shouldn't be that slow).
- Pydantic default values are not checked.
- Only symbolic, literal, and expressions are allowed for dimension specifiers, f-string syntax from `jaxtyping` is not supported.
- Only torch tensors and numpy arrays are supported for now.
- Static checking is not supported, only runtime checks, though some errors will be caught statically by construction.
- We do not support container types (i.e. `list[TensorTypeBase]`) and we probably never will because parsing arbitrarily nested containers is very slow to do at runtime.
- We do not support union types, but we do support optionals.

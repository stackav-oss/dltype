# Contributing
Contributions are welcome & instructions on how to set up the package for local development are below.

## Development Setup
The development environment is managed via `uv`, please install it before proceeding.

Create the uv project's virtual environment with:

```bash
uv sync
```

And then install the `dltype` package as a local editable extension using:

```bash
uv pip install -e .
```

The unit tests can be run with:

```bash
uv run pytest
```

and the benchmark with:

```bash
uv run benchmark.py
```

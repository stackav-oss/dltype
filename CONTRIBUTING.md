# Contributing
Contributions are welcome & instructions on how to set up the package for local development are below.

## Development Setup
The development environment is managed via `uv`.
Plase run the setup script to install it and sync the project dependencies before developing.

```bash
bash ./setup.sh
```


The unit tests can be run with:

```bash
uv run pytest
```

and the benchmark with:

```bash
uv run benchmark.py
```

#!/bin/bash

# This script sets up the environment for the project

curl -LsSf https://astral.sh/uv/install.sh | sh

uv python install
uv sync

if ! command -v pre-commit >/dev/null 2>&1
then
    echo "WARNING: pre-commit not found, please install it for a better dev experience"
    echo "pip install pre-commit --break-system-packages"
    echo "pre-commit install --install-hooks"
else
    pre-commit install --install-hooks
fi

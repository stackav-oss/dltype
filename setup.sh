#!/bin/bash

# This script sets up the environment for the project

if ! command -v uv >/dev/null 2>&1
then
    echo "uv not found, installing..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
fi

uv python install
uv tool install prek
uv sync

prek install --install-hooks

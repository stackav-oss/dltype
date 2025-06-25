#!/bin/bash

# This script sets up the environment for the project

curl -LsSf https://astral.sh/uv/install.sh | sh

uv python install
uv sync
uv run pre-commit install --install-hooks

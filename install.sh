#!/bin/bash
set -eu

# Run the locked install from this repository, even when invoked elsewhere.
cd "$(dirname "$0")"

if ! command -v uv >/dev/null 2>&1; then
    echo "uv is required; install it from https://docs.astral.sh/uv/getting-started/installation/" >&2
    exit 1
fi

uv sync --locked
printf '\nInstallation complete. To play:\n  uv run python -m chess.engine\n'

#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
python3 -m venv .venv
# Python builds may default to clang even when only GCC is installed.
CC="${CC:-gcc}" .venv/bin/python -m pip install -r requirements.txt
.venv/bin/python -m pip install --no-deps -e .

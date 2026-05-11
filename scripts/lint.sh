#!/usr/bin/env bash
# Run Ruff + mypy on src/ (optional; requires requirements-dev.txt).
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
export PYTHONPATH="${ROOT}/src${PYTHONPATH:+:$PYTHONPATH}"

if ! command -v ruff >/dev/null 2>&1; then
  echo "ruff not found. Install: pip install -r requirements-dev.txt" >&2
  exit 1
fi

echo "========== ruff =========="
ruff check src tests

echo "========== mypy (may be slow first run; follow_imports=silent in pyproject) =========="
if command -v mypy >/dev/null 2>&1; then
  mypy src
else
  echo "mypy not found; skip (pip install -r requirements-dev.txt)" >&2
fi

echo "lint.sh: OK"

#!/usr/bin/env bash
# Full local verification: entire pytest suite, optional short Ray training smoke.
#
# Usage (from assignment2/):
#   chmod +x scripts/verify_all.sh   # once
#   ./scripts/verify_all.sh
#   ./scripts/verify_all.sh --train
#   ./scripts/verify_all.sh --lint   # requires: pip install -r requirements-dev.txt
#
# Environment:
#   VERIFY_TRAIN_ITERATIONS  iterations for --train (default: 2)
#   PYTEST_ARGS              extra args for pytest (e.g. '-x' to stop on first failure)

set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
export PYTHONPATH="${ROOT}/src${PYTHONPATH:+:$PYTHONPATH}"

RUN_TRAIN=0
RUN_LINT=0
for arg in "$@"; do
  case "$arg" in
    --train|--smoke) RUN_TRAIN=1 ;;
    --lint) RUN_LINT=1 ;;
    -h|--help)
      cat <<'EOF'
Usage: verify_all.sh [--train] [--lint]

  (default)   Run: python -m pytest tests/ -v --tb=short  (+ $PYTEST_ARGS)
  --train     After pytest, run a short RLlib training job (Ray). Slow; needs resources.
  --lint      After pytest, run ruff + mypy (needs requirements-dev.txt).

Environment:
  VERIFY_TRAIN_ITERATIONS   default 2
  PYTEST_ARGS               extra pytest arguments (quoted)
EOF
      exit 0
      ;;
  esac
done

TRAIN_ITERS="${VERIFY_TRAIN_ITERATIONS:-2}"

echo "========== pytest (full suite) =========="
# shellcheck disable=SC2086
python -m pytest tests/ -v --tb=short ${PYTEST_ARGS:-}

if [[ "$RUN_LINT" -eq 1 ]]; then
  echo ""
  echo "========== lint (ruff + mypy) =========="
  if command -v ruff >/dev/null 2>&1; then
    ruff check src tests
  else
    echo "ruff not installed; skip (pip install -r requirements-dev.txt)" >&2
  fi
  if command -v mypy >/dev/null 2>&1; then
    (cd "$ROOT" && mypy)
  else
    echo "mypy not installed; skip (pip install -r requirements-dev.txt)" >&2
  fi
fi

if [[ "$RUN_TRAIN" -eq 1 ]]; then
  echo ""
  echo "========== 2/2 training smoke (Ray, ${TRAIN_ITERS} iterations, --fast) =========="
  echo "Uses smaller CNN + lighter PPO for quicker smoke; full-quality runs omit --fast."
  python "${ROOT}/src/train.py" \
    --iterations "${TRAIN_ITERS}" \
    --checkpoint-dir checkpoints \
    --seed 0 \
    --no-early-stop \
    --fast
fi

echo ""
echo "verify_all.sh: OK"

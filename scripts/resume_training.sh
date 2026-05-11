#!/usr/bin/env bash
# Resume PPO training from the latest scheduled checkpoint.
#
# Usage:
#   bash scripts/resume_training.sh                                  # auto-pick newest 5-step checkpoint, up to iter 2400
#   bash scripts/resume_training.sh 2600                             # different absolute target iteration
#   bash scripts/resume_training.sh 2400 checkpoints/checkpoint_001805.pt   # explicit checkpoint
#   ITERATIONS=2600 RESTORE=... bash scripts/resume_training.sh
#
# --iterations is the ABSOLUTE target iteration (not delta). Checkpoints are saved
# every 5 iters, so default auto-picks the newest checkpoint whose iteration is a
# multiple of 5 (ignoring stray 1-off files from interrupts).

set -euo pipefail

cd "$(dirname "$0")/.."
export PYTHONPATH=src

pick_latest_checkpoint() {
    local best=""
    local best_num=-1
    for f in checkpoints/checkpoint_*.pt; do
        [ -f "$f" ] || continue
        local stem="${f##*/checkpoint_}"
        local num="${stem%.pt}"
        num=$((10#$num))            # strip leading zeros for arithmetic
        if (( num % 5 == 0 )) && (( num > best_num )); then
            best="$f"
            best_num=$num
        fi
    done
    if [ -z "$best" ]; then
        echo "no checkpoint with iter%5==0 found under checkpoints/" >&2
        exit 1
    fi
    echo "$best"
}

ITERATIONS="${1:-${ITERATIONS:-2400}}"
RESTORE="${2:-${RESTORE:-$(pick_latest_checkpoint)}}"

echo "→ restoring from: $RESTORE"
echo "→ target iterations (absolute): $ITERATIONS"
echo "→ eval every 10 iters, plots/checkpoints every 5 iters"
echo

exec .venv/bin/python -u src/train.py \
    --restore "$RESTORE" \
    --iterations "$ITERATIONS" \
    --eval-interval 10 \
    --plot-every 5 \
    --checkpoint-freq 5 \
    --no-early-stop

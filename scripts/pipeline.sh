#!/usr/bin/env bash
# Train → print next steps for export and web server (assignment2 root).
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
ITERATIONS="${ITERATIONS:-800}"
export PYTHONPATH="${ROOT}/src${PYTHONPATH:+:$PYTHONPATH}"
python "${ROOT}/src/train.py" --iterations "${ITERATIONS}" --checkpoint-dir checkpoints "$@"
echo ""
echo "Next: export main module for the Flask UI (replace CHECKPOINT with a path from logs above):"
echo "  python src/export_model.py CHECKPOINT -o checkpoints/model_weights.pt"
echo "Then play:"
echo "  python src/server.py --model checkpoints/model_weights.pt --port 5001"

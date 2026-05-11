#!/usr/bin/env bash
# Throughput ablation: edit src/config.py PPO_CONFIG (train_batch_size, num_env_runners)
# only between runs — this script only prints suggested commands.

set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

echo "=== Throughput ablation (manual) ==="
echo "1. Edit PPO_CONFIG in src/config.py: train_batch_size, num_env_runners."
echo "2. Short runs (example, 20 iters, fixed seed):"
echo "   ITERATIONS=20 python src/train.py --iterations \"\${ITERATIONS:-20}\" --checkpoint-dir checkpoints --seed 42"
echo "3. Fill report.html §16 table: sec/iter from training.log or iter_seconds in metrics.csv;"
echo "   main return from logs (module_episode_returns_mean[main]) or mean_reward column."
echo "4. Restore defaults before long production runs."

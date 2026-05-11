# Assignment 2 — Super Tic-Tac-Toe RL

> **Student Name:** 孙海崴
> **Student ID:** 21230713

🚨 **IMPORTANT: MUST READ AND WATCH** 🚨
**Primary grading artifact:** Please open [`report.html`](report.html) in a desktop browser. **You MUST read this report and watch the full live gameplay demo video included in Section 18!** The report contains the definitive quantitative metrics, tactical style analysis, and defensive highlights that prove the success of this project.

**Start at §15 — Experiments & grading guide** in the report for a full checklist of implemented features, suggested comparison commands, figure/CSV legends, and a pipeline diagram (no need to read the repo line-by-line). Section 8.5 documents symmetry augmentation; Section 17 lists delivered items.

This `README.md` is a **short command index**; narrative detail lives in `report.html`.

---

## For graders / instructors (30-second path)

1. Open **`report.html` → §15 Experiments & grading guide** — what was built, what to run, how to read `metrics.csv` and the four PNGs.
2. Run **`python -m pytest tests/ -v`** — **441** tests (some in `test_server.py` may be skipped if Flask is missing).
3. Optional: run **one training job** and refresh the report — figures under `results/latest/plots/` embed automatically.

---

## Before submission (checklist)

1. **`pytest`** — `python -m pytest tests/ -q` → expect **441 passed** (or ~410 passed + skipped if Flask is missing).
2. **Training once** — e.g. `python src/train.py --iterations 50 --seed 42` (or your full budget) so `results/latest/plots/*.png` exists and **`report.html` §15 figures render** when opened locally or in a browser.
3. **Git / zip** — `.gitignore` ignores `results/run_*` and `results/LATEST.txt` (large logs); you **may commit** `results/latest/plots/*.png` so graders see curves without retraining. Copy `run_meta.json` reasoning: exact torch/torchrl versions live there after a run.
4. **Versions** — note `torch` / `torchrl` from `run_meta.json` in the report or cover sheet if the course asks for environment.

---

## Documentation sync

**After changing training scripts, default hyperparameters, checkpoints, evaluation, server API, or test counts, update both `README.md` and `report.html`.**

---

## Setup

```bash
pip install -r requirements.txt
```

### Optional: dev tools (lint / types)

```bash
pip install -r requirements-dev.txt
ruff check src tests    # style & obvious bugs (see pyproject.toml)
mypy                    # type-checks core game modules only
./scripts/lint.sh       # ruff + mypy
./scripts/verify_all.sh --lint   # pytest then lint
```

Exact replication of a machine: after installing, run `pip freeze > requirements-lock.txt` (see `requirements-lock.example.txt`).

## Tests (run before submission)

```bash
python -m pytest tests/ -v
```

**One-shot (recommended for a first full check):** from the `assignment2new/` root run `./scripts/verify_all.sh` — this runs the **entire** pytest suite (same as above). To also smoke-test the **real training stack** (TorchRL + `train.py`, uses MPS GPU on Apple Silicon), run `./scripts/verify_all.sh --train` or set `VERIFY_TRAIN_ITERATIONS=3 ./scripts/verify_all.sh --train`. The training step passes **`--fast`** (smaller CNN + lighter PPO) so it finishes sooner than a full-quality run; for final experiments omit `--fast` and use `config.py` defaults. Training smoke writes under `results/run_*` and `checkpoints/` like a normal job. For a manual quick run: `PYTHONPATH=src python src/train.py --iterations 5 --fast --no-early-stop`.

Currently **441** tests collected (some in `test_server.py` skipped if Flask is missing). Key coverage: `test_rules.py` includes partial threat heatmap (2-of-4 developing threats); `test_env.py` covers all 7 scripted opponents (line_rusher, row_rusher, col_rusher, center_biased, edge_explorer strategy preferences + full-board edge cases), `TestScriptedTypesInPool`, and env edge cases (multiple resets, different-seed divergence, multi-step observation correctness); `test_self_play.py` covers opponent pool, snapshotting, draw tracking, wrap-around, 7 scripted types, PFSP sampling, threshold-exact snapshot trigger, unrecognized outcome handling, and deep-copy isolation; `test_augment_learner.py` checks PPO batch flip symmetry; `test_build_config.py` / `test_config_sync.py` assert PPO/model wiring vs `config.py` including clip_param, gamma, lambda, vf_loss_coeff, grad_clip, curriculum config, and all TRAINING_CONFIG fields; `test_ppo_update.py` covers the core PPO update (all 7 metric keys, finite losses, clip_fraction range, entropy positivity, grad_norm, augment path, minibatch edge cases, weight change verification), rollout collection (tensor shapes, valid action range, episode endings, scripted opponents), and env slot machinery; `test_eval_rollout.py` covers evaluation with greedy action + mask + outcome resolution, `_build_obs_for_board` (shape, channels, last move plane), torch tensor inputs, net-vs-net play, and conflicting info handling; `test_compare_models.py` covers architecture detection, obs builders, action masks, and game play; `test_analyze_training.py` covers CSV parsing, moving average, and chart generation; `test_export_model.py` covers checkpoint export and error handling; `test_server.py` covers Flask routes + `build_obs`, `get_action_mask`, `make_move` unit tests; `test_training_viz.py` covers CSV I/O, plotting with eval overlays and rich PPO data, run directory creation; `test_integration.py` includes a full training smoke test (`train(num_iterations=1, fast=True)`) and network determinism verification.

---

## Implementation summary (maps to code & report)

| Theme | What to mention in the report |
|-------|-------------------------------|
| Rules & env | PDF-consistent board, stochastic placement, PettingZoo AEC, **strict zero-sum** step rewards (`env.py`); observation **(7, 12, 12)** 7-channel ego-centric; action **Discrete(96)** compact; **reward shaping annealing** (`SHAPING_SCHEDULE`: piecewise decay of shaping bonuses, forfeit penalty constant) |
| Learning | **TorchRL** PPO (MPS-accelerated on Apple Silicon), league self-play (`main` vs frozen snapshots; history branch uses **PFSP-weighted sampling**), masked CNN (**8×ResBlock + SE**), piecewise LR & entropy on **env-step lifetime** (`config.py`); curriculum opening (`random_opening_prob=0.35`, `random_opening_steps=6`); **9 scripted opponents** with a stronger tactical mix at **40%** probability (`random_legal`, `heuristic`, `line_rusher`, `row_rusher`, `col_rusher`, `center_biased`, `edge_explorer`, `greedy_tactical`, `lookahead_scripted`); **vectorised multi-env rollout** with batched GPU inference (12 collectors); **symmetry augmentation** (`--augment` flag: 50% minibatch flip) |
| Evaluation | Periodic **greedy** eval: `main` vs `main_v0`, vs **uniform random legal**, **and** vs **heuristic** opponent (`eval_rollout.py`) with 50/50 seat alternation (half as first player, half as second) — relative strength vs absolute baseline; per-rusher eval metrics (`eval_vs_line_rusher`, `eval_vs_row_rusher`, `eval_vs_col_rusher`), tactical-opponent evals (`eval_vs_greedy_tactical`, `eval_vs_lookahead_scripted`), and `eval_block_rate` tracked; **early stop** triggers on vs-heuristic win rate ≥ 0.95; server hard difficulty uses **one-step expected-value lookahead** (`lookahead_action()`) |
| Augmentation (optional) | `--augment`: learner-side L–R flip with joint `obs` / mask / action / logits + `ACTION_LOGP` recompute (`augment.py`, `augment_learner.py`) |
| PPO loss | Clip-only PPO (no KL penalty); manual clip-PPO in `train.py` with TorchRL GAE |
| Reproducibility | `results/run_*/run_meta.json` (seed, versions, argv, git) |

---

## Code map (`src/`)

| File | Role |
|------|------|
| `config.py` | PPO numeric defaults, model size, opponent slots, self-play and training defaults, `SHAPING_SCHEDULE` |
| `board.py`, `rules.py`, `stochastic.py` | Game geometry, win detection, random placement |
| `env.py` | PettingZoo AEC environment |
| `network.py` | Standalone `SuperTTTNet` for export / Flask |
| `self_play.py` | OpponentPool: league snapshots, weighted sampling, win-rate tracking, **PFSP** (`_pfsp_sample_history_slot()`) |
| `train.py` | TorchRL PPO training loop (MPS), manual rollout collection, CSV, plots, checkpoints |
| `training_viz.py` | `metrics.csv` (29 columns), 3×3 combined PNG dashboard, `results/LATEST.txt` |
| `eval_rollout.py` | Greedy eval `main` vs `main_v0` + vs random-legal bot + vs heuristic bot; **`lookahead_action()`** for server hard difficulty (one-step expected-value lookahead) |
| `augment.py` | Vertical-axis symmetry: `flip_action`, `flip_obs`, `flip_mask` |
| `augment_learner.py` | Batch-level 50% minibatch flip (joint obs/mask/action/logits/logp) |
| `rllib_fs_path.py` | Absolute local paths for checkpoints |
| `export_model.py` | Checkpoint → flat `.pt` for the server |
| `compare_models.py` | Cross-architecture model comparison (old vs new, vs random/heuristic baselines) |
| `analyze_training.py` | Post-training CSV analysis and publication-quality chart generation |
| `server.py` | Flask UI and move API; model version management (`/api/versions`, `/api/load_version`) |

---

## Training

Checkpoints default every **5** iterations. Each run creates a **new** directory `results/run_YYYYMMDD_HHMMSS/` (older runs are not updated in place). See `results/LATEST.txt` for the latest run path.

```bash
# Recommended full training (with symmetry augmentation):
PYTHONPATH=src python src/train.py --seed 42 --checkpoint-dir checkpoints --augment

# Override iteration count for smoke tests:
PYTHONPATH=src python src/train.py --iterations 50 --seed 42 --checkpoint-dir checkpoints --augment
```

Common flags:

- `--seed N`: overrides `TRAINING_CONFIG["default_seed"]` (default 42); seeds Python, NumPy, torch.
- `--checkpoint-freq`, `--plot-every`, `--results-dir`.
- `--no-early-stop`: do not stop when vs-heuristic eval win rate reaches `stop_reward` (0.95).
- `--augment`: **recommended** — enable vertical-axis symmetry augmentation (50% minibatch flip; zero extra env sampling). Joint flip of obs, `action_mask`, logits, remapped actions; **`log_prob` recomputed** from permuted logits (see `report.html` §8.5).
- `--fast`: quick test mode — smaller CNN (64 filters, 2 ResBlocks), batch=2048, for fast iteration verification.

**Offline eval:** on each `eval_interval` iteration, greedy `main` vs `main_v0`, `main` vs uniform-random-legal bot, **and** `main` vs heuristic bot for **`eval_num_episodes` games (default 50 in `config.py`)**. Eval plays half games as first player and half as second player (50/50 seat alternation). Tactical scripted evals (`greedy_tactical`, `lookahead_scripted`) use a smaller episode cap inside `train.py` because they are substantially more expensive per move. **`metrics.csv` columns (29 total):** training (`iteration`, `global_iteration`, `mean_reward`, `win_rate`, `num_episodes_lifetime`, `iter_seconds`), PPO diagnostics (`policy_loss`, `value_loss`, `entropy`, `clip_fraction`, `approx_kl`, `explained_variance`, `grad_norm`, `learning_rate`, `entropy_coeff`, `shaping_mult`, `steps_per_sec`, `total_env_steps`, `snapshot_count`), eval (`eval_win_rate`, `eval_draw_rate`, `eval_vs_random_win`, `eval_vs_heuristic_win`, `eval_vs_line_rusher`, `eval_vs_row_rusher`, `eval_vs_col_rusher`, `eval_vs_greedy_tactical`, `eval_vs_lookahead_scripted`, `eval_block_rate`).

**Console output** includes per-iteration: reward, win_rate, learning rate, entropy coeff, policy loss, value loss, clip fraction, approx KL, gradient norm, explained variance, shaping multiplier, snapshot count, throughput (steps/sec), wall time, and ETA.

Outputs:

- `results/run_<timestamp>/training.log`, `metrics.csv`, `plots/*.png` (3×3 combined dashboard + individual panels), `run_meta.json`
- `results/latest/plots/` (copy of the latest plots for stable `report.html` image links), `results/LATEST.txt`

### One-shot pipeline

```bash
chmod +x scripts/pipeline.sh   # once
./scripts/pipeline.sh
# or: ITERATIONS=100 ./scripts/pipeline.sh
```

### Resume training

```bash
python src/train.py --restore "checkpoints/.../checkpoint_000050" --iterations 1800 --checkpoint-dir checkpoints
```

### Move distribution (optional — does not touch training)

`metrics.csv` has no per-move data. To see **how often greedy play visits each of the 96 valid cells** (e.g. centre vs edge), run this **separate** script while training continues elsewhere:

```bash
PYTHONPATH=src python src/analyze_move_distribution.py checkpoints/checkpoint_000700.pt \
  --games 800 --opponent heuristic --out-dir results/move_dist_700
```

Writes `move_counts.csv` and `move_frequency_heatmap.png`. Uses `random_opening_prob=0` so counts reflect the policy. See `src/analyze_move_distribution.py` for `--opponent` and `--no-plot`.

---

## Default hyperparameters

**`config.py`** holds PPO and CNN defaults. `train.py` uses these directly with clip-only PPO and TorchRL GAE. Full table is in **`report.html` §9**.

| Key | Value (current) |
|-----|-----------------|
| `lr` | piecewise vs **env steps**: 3e-4 → 1e-4 at `300×train_batch_size` steps → 3e-5 at `600×train_batch_size` → 1e-5 at `900×train_batch_size`; see `config.py` |
| `train_batch_size` | 12288 |
| `sgd_minibatch_size` | 512 |
| `num_epochs` | 8 |
| `num_collectors` | 12 (concurrent envs with batched GPU inference) |
| `entropy_coeff` | piecewise vs env steps: 0.05 → 0.03 at `150×train_batch_size` → 0.015 at `400×train_batch_size` → 0.005 at `700×train_batch_size`; see `config.py` |
| `grad_clip` | 0.5 |
| CNN `num_filters` / `num_res_blocks` / value MLP | 192 / 8 / 768 (with SE attention) |
| Opponent slots (`main_v0` …) | 8 |
| Scripted opponents | 9 types at 40% with stronger tactical weighting: random_legal, heuristic (with partial threat detection), line_rusher, row_rusher, col_rusher, center_biased, edge_explorer, greedy_tactical, lookahead_scripted |
| Reward shaping | symmetric zero-sum: placement ±0.01, block threat +0.10, create threat +0.06, forfeit ±0.015, threat-aware blocking/creation |
| `shaping_schedule` | piecewise on env steps: `[(0, 1.0), (900*12288, 1.0), (1050*12288, 0.3), (1150*12288, 0.20), (1500*12288, 0.20), (1700*12288, 0.12)]` -- multiplies PLACEMENT/BLOCK/THREAT bonuses (NOT forfeit); avoids late-training reward hacking while keeping a small late-stage tactical signal |

**Memory / swap:** Prefer running from a **local disk** copy (not iCloud). If the machine still struggles, reduce **`num_collectors`** first (e.g. 4), then optionally `train_batch_size` / `sgd_minibatch_size` / `num_epochs` and CNN width in `config.py` together.

---

## Export weights

```bash
python src/export_model.py checkpoints/<checkpoint_folder> -o checkpoints/model_weights.pt
```

## Local play + report

```bash
python src/server.py --model checkpoints/model_weights.pt --port 5001
```

- UI: <http://localhost:5001> (round-trip latency per move; `GET /api/status` shows load state); selectable first/second player, difficulty levels (easy/medium/hard). Hard difficulty uses **one-step expected-value lookahead** (`lookahead_action()` in `eval_rollout.py`) for stronger play
- Report mirror: <http://localhost:5001/report>
- Server accepts `ai_first` and `difficulty` in `POST /reset`
- Model version switching: `GET /api/versions` lists available versions; `POST /api/load_version` with `{"version": "1.0"}` hot-swaps weights
- Omit `--model`: AI plays uniform random **legal** moves (not an untrained network forward pass)
- Closer to training sampling: `--stochastic`

### Production (Waitress example)

```bash
pip install waitress
cd src
waitress-serve --listen=127.0.0.1:5001 server:app
```

---

## Model comparison

```bash
python src/compare_models.py --a models/v1.0 --b models/v2.0 --games 100
```

Compares two model versions head-to-head with seat alternation. Supports old (3-ch/144-action) and new (7-ch/96-action) architectures. Use `--a random` or `--a heuristic` for baseline agents.

## Training analysis

```bash
python src/analyze_training.py results/run_*/metrics.csv -o results/run_*/plots
```

Generates publication-quality charts: reward curves, win rate progression, training phases, throughput analysis, learning efficiency, and summary dashboard.

---

## Throughput / ablation

See **`report.html` §16** for a printable table template and suggested short-run commands. `scripts/ablation_notes.sh` only prints hints; it does not change `config.py`.

**Fair comparison:** use the same `--seed` when comparing batch size / `num_collectors` / `--augment` (all runs use clip-only PPO, no KL penalty).

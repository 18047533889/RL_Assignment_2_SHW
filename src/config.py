"""
config.py — Single source of truth for PPO, CNN, league, and training-loop knobs.

TorchRL version: no Ray dependency. MPS-accelerated on Apple Silicon.
"""

import torch

DEVICE = (
    "mps" if torch.backends.mps.is_available()
    else "cuda" if torch.cuda.is_available()
    else "cpu"
)

_TRAIN_BATCH = 12288

PPO_CONFIG = {
    "lr_schedule": [
        (0, 1e-4),
        (300 * _TRAIN_BATCH, 5e-5),
        (600 * _TRAIN_BATCH, 2e-5),
        (900 * _TRAIN_BATCH, 8e-6),
        (1200 * _TRAIN_BATCH, 1.2e-5),
        (1500 * _TRAIN_BATCH, 6e-6),
        (1855 * _TRAIN_BATCH, 3e-5),
        (2400 * _TRAIN_BATCH, 1e-5),
        (2800 * _TRAIN_BATCH, 5e-6),
    ],
    "entropy_schedule": [
        (0, 0.05),
        (150 * _TRAIN_BATCH, 0.03),
        (400 * _TRAIN_BATCH, 0.015),
        (700 * _TRAIN_BATCH, 0.005),
        (1200 * _TRAIN_BATCH, 0.008),
        (1500 * _TRAIN_BATCH, 0.005),
        (1855 * _TRAIN_BATCH, 0.02),
        (2400 * _TRAIN_BATCH, 0.01),
        (2800 * _TRAIN_BATCH, 0.005),
    ],
    "clip_param": 0.2,
    "gamma": 0.99,
    "lambda_": 0.95,
    "vf_loss_coeff": 1.8,
    "train_batch_size": _TRAIN_BATCH,
    "sgd_minibatch_size": 512,
    "num_epochs": 4,
    "num_collectors": 12,
    "grad_clip": 0.5,
}

MODEL_CONFIG = {
    "num_filters": 192,
    "num_res_blocks": 8,
    "value_fc_hidden": 768,
}

NUM_OPPONENT_SLOTS = 8
OPPONENT_MODULE_IDS: list[str] = [f"main_v{i}" for i in range(NUM_OPPONENT_SLOTS)]

SELF_PLAY_CONFIG = {
    "win_rate_threshold": 0.62,
    "initial_snapshot_prob": 0.10,
    "history_snapshot_prob": 0.28,
    "current_self_prob": 0.22,
    "scripted_prob": 0.40,
}

TRAINING_CONFIG = {
    "num_iterations": 1800,
    "checkpoint_freq": 5,
    "default_seed": 42,
    "eval_interval": 10,
    "eval_num_episodes": 50,
    "stop_reward": 0.95,
}

CURRICULUM_CONFIG = {
    "random_opening_prob": 0.35,
    "random_opening_steps": 6,
    "forfeit_injection_prob": 0.02,
}

SHAPING_SCHEDULE = [
    (0, 1.0),
    (900 * _TRAIN_BATCH, 1.0),
    (1050 * _TRAIN_BATCH, 0.3),
    (1150 * _TRAIN_BATCH, 0.20),
    (1500 * _TRAIN_BATCH, 0.20),
    (1700 * _TRAIN_BATCH, 0.12),
    # Resume point: lift shaping back up so the new persist/partial-block/waste signals
    # actually register, then decay back down over ~1000 iters so terminal wins dominate.
    (1855 * _TRAIN_BATCH, 0.50),
    (2300 * _TRAIN_BATCH, 0.35),
    (2700 * _TRAIN_BATCH, 0.20),
    (3000 * _TRAIN_BATCH, 0.15),
]

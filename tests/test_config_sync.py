"""Internal consistency of config.py: _TRAIN_BATCH, LR/entropy breakpoints, PPO defaults, league slots."""
import pytest

import config
from network import SuperTTTNet


def test_train_batch_matches_ppo_and_constant():
    assert config._TRAIN_BATCH == 12288
    assert config.PPO_CONFIG["train_batch_size"] == config._TRAIN_BATCH


def test_lr_schedule_env_step_breakpoints():
    lr = config.PPO_CONFIG["lr_schedule"]
    tb = config._TRAIN_BATCH
    assert lr[0][0] == 0
    assert lr[0][1] == pytest.approx(1e-4)
    assert lr[1][0] == 300 * tb
    assert lr[1][1] == pytest.approx(5e-5)
    assert lr[2][0] == 600 * tb
    assert lr[2][1] == pytest.approx(2e-5)
    assert lr[3][0] == 900 * tb
    assert lr[3][1] == pytest.approx(8e-6)
    assert lr[4][0] == 1200 * tb
    assert lr[4][1] == pytest.approx(1.2e-5)
    assert lr[5][0] == 1500 * tb
    assert lr[5][1] == pytest.approx(6e-6)
    assert lr[6][0] == 1855 * tb
    assert lr[6][1] == pytest.approx(3e-5)
    assert lr[7][0] == 2400 * tb
    assert lr[7][1] == pytest.approx(1e-5)
    assert lr[8][0] == 2800 * tb
    assert lr[8][1] == pytest.approx(5e-6)


def test_entropy_schedule_env_step_breakpoints():
    ec = config.PPO_CONFIG["entropy_schedule"]
    tb = config._TRAIN_BATCH
    assert ec[0][0] == 0
    assert ec[0][1] == pytest.approx(0.05)
    assert ec[1][0] == 150 * tb
    assert ec[1][1] == pytest.approx(0.03)
    assert ec[2][0] == 400 * tb
    assert ec[2][1] == pytest.approx(0.015)
    assert ec[3][0] == 700 * tb
    assert ec[3][1] == pytest.approx(0.005)
    assert ec[4][0] == 1200 * tb
    assert ec[4][1] == pytest.approx(0.008)
    assert ec[5][0] == 1500 * tb
    assert ec[5][1] == pytest.approx(0.005)
    assert ec[6][0] == 1855 * tb
    assert ec[6][1] == pytest.approx(0.02)
    assert ec[7][0] == 2400 * tb
    assert ec[7][1] == pytest.approx(0.01)
    assert ec[8][0] == 2800 * tb
    assert ec[8][1] == pytest.approx(0.005)


def test_ppo_sgd_and_epochs():
    assert config.PPO_CONFIG["sgd_minibatch_size"] == 512
    assert config.PPO_CONFIG["num_epochs"] == 4
    assert config.PPO_CONFIG["num_collectors"] == 12


def test_model_config_matches_supertttnet_defaults():
    mc = config.MODEL_CONFIG
    net = SuperTTTNet()
    assert len(net.res_blocks) == mc["num_res_blocks"]
    assert net.initial_conv.out_channels == mc["num_filters"]
    assert net.value_fc1.out_features == mc["value_fc_hidden"]
    assert net.value_fc2.in_features == mc["value_fc_hidden"]


def test_opponent_ids_match_slot_count():
    assert len(config.OPPONENT_MODULE_IDS) == config.NUM_OPPONENT_SLOTS
    assert config.OPPONENT_MODULE_IDS == [f"main_v{i}" for i in range(config.NUM_OPPONENT_SLOTS)]


def test_training_default_iterations():
    assert config.TRAINING_CONFIG["num_iterations"] == 1800


def test_self_play_branch_probabilities_sum_to_one():
    sp = config.SELF_PLAY_CONFIG
    s = (
        float(sp["initial_snapshot_prob"])
        + float(sp["history_snapshot_prob"])
        + float(sp["current_self_prob"])
        + float(sp.get("scripted_prob", 0.0))
    )
    assert s == pytest.approx(1.0, abs=1e-9)


def test_shaping_schedule_starts_at_one():
    assert config.SHAPING_SCHEDULE[0] == (0, 1.0)


def test_shaping_schedule_decays():
    final_val = config.SHAPING_SCHEDULE[-1][1]
    assert final_val < 1.0
    assert final_val >= 0.0


def test_shaping_schedule_tail_monotonic_non_increasing():
    # Post-1855 resume bumps shaping back up (new signals need to register),
    # so the schedule is no longer globally monotonic. Require the tail from
    # the resume bump onward to decay monotonically back down.
    tb = config._TRAIN_BATCH
    resume_step = 1855 * tb
    tail = [(s, v) for s, v in config.SHAPING_SCHEDULE if s >= resume_step]
    assert len(tail) >= 2
    for i in range(1, len(tail)):
        assert tail[i][1] <= tail[i - 1][1]


def test_ppo_clip_param():
    assert config.PPO_CONFIG["clip_param"] == pytest.approx(0.2)


def test_ppo_gamma():
    assert config.PPO_CONFIG["gamma"] == pytest.approx(0.99)


def test_ppo_lambda():
    assert config.PPO_CONFIG["lambda_"] == pytest.approx(0.95)


def test_ppo_vf_loss_coeff():
    assert config.PPO_CONFIG["vf_loss_coeff"] == pytest.approx(1.8)


def test_ppo_grad_clip():
    assert config.PPO_CONFIG["grad_clip"] == pytest.approx(0.5)


def test_training_config_checkpoint_freq():
    assert config.TRAINING_CONFIG["checkpoint_freq"] == 5


def test_training_config_default_seed():
    assert config.TRAINING_CONFIG["default_seed"] == 42


def test_training_config_eval_interval():
    assert config.TRAINING_CONFIG["eval_interval"] == 10


def test_training_config_eval_num_episodes():
    assert config.TRAINING_CONFIG["eval_num_episodes"] == 50


def test_training_config_stop_reward():
    assert config.TRAINING_CONFIG["stop_reward"] == pytest.approx(0.95)


def test_curriculum_config_values():
    assert config.CURRICULUM_CONFIG["random_opening_prob"] == pytest.approx(0.35)
    assert config.CURRICULUM_CONFIG["random_opening_steps"] == 6


def test_device_is_valid_string():
    assert config.DEVICE in ("mps", "cuda", "cpu")

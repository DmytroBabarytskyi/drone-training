"""Тести винагороди RL (ml/rl/rewards.py) — чисті функції, без Panda3D/Gymnasium."""

import numpy as np

from dronesim.core.contracts import VehicleState
from dronesim.ml.rl.rewards import RewardConfig, compute_reward


def _state(pos, quat=(0.0, 0.0, 0.0, 1.0)) -> VehicleState:
    return VehicleState(pos=np.array(pos, dtype=np.float64), quat=np.array(quat))


def test_moving_closer_gives_positive_progress_reward():
    cfg = RewardConfig()
    target = np.array([10.0, 0.0, 3.0])
    state = _state([5.0, 0.0, 3.0])  # ближче, ніж prev_distance
    result, distance = compute_reward(state, target, prev_distance=10.0, cfg=cfg)
    assert result.reward > 0.0
    assert distance == 5.0


def test_moving_away_gives_negative_progress_reward():
    cfg = RewardConfig()
    target = np.array([10.0, 0.0, 3.0])
    state = _state([0.0, 0.0, 3.0])
    result, _ = compute_reward(state, target, prev_distance=5.0, cfg=cfg)
    assert result.reward < 0.0


def test_hit_when_close_and_facing_target():
    cfg = RewardConfig(hit_radius=2.0, max_aim_angle_rad=0.6)
    target = np.array([1.0, 0.0, 3.0])
    state = _state([0.0, 0.0, 3.0])  # 1м від цілі, дивиться вздовж +X (ідентичний quat)
    result, _ = compute_reward(state, target, prev_distance=1.0, cfg=cfg)
    assert result.hit
    assert result.terminated
    assert result.reward > cfg.hit_bonus - 1.0  # бонус домінує


def test_no_hit_when_close_but_facing_away():
    cfg = RewardConfig(hit_radius=2.0, max_aim_angle_rad=0.3)
    target = np.array([1.0, 0.0, 3.0])
    quat_180 = (0.0, 0.0, 1.0, 0.0)  # 180° навколо Z -> дивиться в -X
    state = _state([0.0, 0.0, 3.0], quat=quat_180)
    result, _ = compute_reward(state, target, prev_distance=1.0, cfg=cfg)
    assert not result.hit


def test_no_hit_when_facing_but_far():
    cfg = RewardConfig(hit_radius=2.0, max_aim_angle_rad=0.6)
    target = np.array([50.0, 0.0, 3.0])
    state = _state([0.0, 0.0, 3.0])
    result, _ = compute_reward(state, target, prev_distance=50.0, cfg=cfg)
    assert not result.hit


def test_crash_when_below_min_altitude():
    cfg = RewardConfig(min_altitude=0.3)
    target = np.array([10.0, 0.0, 3.0])
    state = _state([0.0, 0.0, 0.1])
    result, _ = compute_reward(state, target, prev_distance=10.0, cfg=cfg)
    assert result.crashed
    assert result.terminated
    assert result.reward < 0.0


def test_out_of_bounds_when_too_far():
    cfg = RewardConfig(out_of_bounds_radius=20.0)
    target = np.array([0.0, 0.0, 3.0])
    state = _state([100.0, 0.0, 3.0])
    result, _ = compute_reward(state, target, prev_distance=90.0, cfg=cfg)
    assert result.out_of_bounds
    assert result.terminated


def test_facing_target_gives_los_bonus_even_without_hit():
    cfg = RewardConfig(hit_radius=1.0, los_bonus=0.5, max_aim_angle_rad=0.6, progress_scale=0.0, time_penalty=0.0)
    target = np.array([10.0, 0.0, 3.0])
    state = _state([0.0, 0.0, 3.0])  # дивиться на ціль (+X), але далеко (>hit_radius)
    result, _ = compute_reward(state, target, prev_distance=10.0, cfg=cfg)
    assert not result.hit
    assert abs(result.reward - cfg.los_bonus) < 1e-9


def test_no_progress_reward_when_stationary():
    cfg = RewardConfig(time_penalty=0.0, los_bonus=0.0)
    target = np.array([20.0, 0.0, 3.0])
    state = _state([0.0, 0.0, 3.0])
    result, distance = compute_reward(state, target, prev_distance=20.0, cfg=cfg)
    assert abs(result.reward) < 1e-9
    assert distance == 20.0

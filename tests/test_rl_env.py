"""Тести DroneEnv (ml/rl/env.py, фаза 6) — headless, без Panda3D-рендеру/torch
(Multirotor/PhysicsWorld використовують лише panda3d.bullet, не ShowBase)."""

import numpy as np

from dronesim.ml.rl.env import OBS_DIM, DroneEnv


def test_reset_returns_correct_obs_shape_and_finite():
    env = DroneEnv()
    obs, info = env.reset(seed=0)
    assert obs.shape == (OBS_DIM,)
    assert obs.dtype == np.float32
    assert np.all(np.isfinite(obs))
    assert info == {}


def test_reset_same_seed_is_reproducible():
    env = DroneEnv()
    obs_a, _ = env.reset(seed=42)
    obs_b, _ = env.reset(seed=42)
    assert np.allclose(obs_a, obs_b)


def test_reset_different_seed_gives_different_start():
    env = DroneEnv()
    obs_a, _ = env.reset(seed=1)
    obs_b, _ = env.reset(seed=2)
    assert not np.allclose(obs_a, obs_b)


def test_action_and_observation_spaces_shapes():
    env = DroneEnv()
    assert env.action_space.shape == (4,)
    assert env.observation_space.shape == (OBS_DIM,)


def test_step_returns_expected_types():
    env = DroneEnv()
    env.reset(seed=0)
    obs, reward, terminated, truncated, info = env.step(np.zeros(4, dtype=np.float32))
    assert obs.shape == (OBS_DIM,)
    assert isinstance(reward, float)
    assert isinstance(terminated, bool)
    assert isinstance(truncated, bool)
    assert "distance" in info


def test_zero_action_hovers_without_immediate_crash_or_hit():
    """Нейтральний стік (Pos-hold, throttle=0.5) має просто триматись на місці —
    ні краш, ні випадкове миттєве влучання (старт далі за hit_radius)."""
    env = DroneEnv()
    env.reset(seed=7)
    for _ in range(20):
        obs, reward, terminated, truncated, info = env.step(np.array([0.0, 0.0, 0.0, 0.0]))
        assert np.all(np.isfinite(obs))
        if terminated:
            break
    assert not info["crashed"]


def test_episode_truncates_at_max_steps_when_hovering_far_from_target():
    env = DroneEnv()
    env.reset(seed=3)
    env.max_episode_steps = 15  # прискорити тест
    terminated = truncated = False
    steps = 0
    while not (terminated or truncated):
        _, _, terminated, truncated, info = env.step(np.zeros(4, dtype=np.float32))
        steps += 1
        assert steps <= 15
    assert truncated or terminated


def test_reward_and_distance_are_finite_over_several_steps():
    env = DroneEnv()
    env.reset(seed=5)
    for _ in range(10):
        _, reward, terminated, _, info = env.step(np.array([0.1, 0.0, 0.0, 0.0]))
        assert np.isfinite(reward)
        assert np.isfinite(info["distance"])
        if terminated:
            break

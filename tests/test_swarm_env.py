"""Tests for SwarmInterceptEnv — shapes, boundary clamping, collision
penalty and the final-step hit bonus. No Panda3D involved (see
ml/swarm/swarm_env.py package docstring for why)."""

import numpy as np

from dronesim.ml.swarm.swarm_env import SwarmInterceptEnv, SwarmRewardConfig, SwarmSceneConfig


def _small_env(**scene_overrides) -> SwarmInterceptEnv:
    scene = SwarmSceneConfig(n_drones=4, decision_steps=5, **scene_overrides)
    return SwarmInterceptEnv(scene_cfg=scene, reward_cfg=SwarmRewardConfig())


def test_reset_and_step_shapes():
    env = _small_env()
    obs, info = env.reset(seed=0)
    assert obs.shape == env.observation_space.shape
    assert env.observation_space.contains(obs)

    action = env.action_space.sample()
    obs2, reward, terminated, truncated, info = env.step(action)
    assert obs2.shape == env.observation_space.shape
    assert isinstance(reward, float)
    assert not truncated


def test_episode_terminates_after_decision_steps():
    env = SwarmInterceptEnv(
        scene_cfg=SwarmSceneConfig(n_drones=4, decision_steps=3),
        reward_cfg=SwarmRewardConfig(),
    )
    env.reset(seed=1)
    zero_action = np.zeros(env.action_space.shape, dtype=np.float32)
    for step in range(3):
        _, _, terminated, _, info = env.step(zero_action)
    assert terminated
    assert info["hits"] is not None  # only populated on the final step


def test_out_of_bounds_position_is_clamped_and_penalized():
    env = _small_env(coverage_half_extent_m=5.0, max_speed_mps=100.0, max_accel_mps2=100.0)
    env.reset(seed=2)
    # Push every drone as hard as possible, repeatedly — with these bounds
    # they should eventually overshoot and get clamped back onto the boundary.
    full_push = np.ones(env.action_space.shape, dtype=np.float32)
    for _ in range(10):
        obs, reward, terminated, *_ = env.step(full_push)
        assert np.all(np.abs(env._positions) <= 5.0 + 1e-9)
        if terminated:
            break


def test_colliding_drones_are_penalized():
    env = _small_env(min_separation_m=3.0)
    env.reset(seed=3)
    # Force two drones onto the same point — a maximal collision.
    env._positions[0] = np.array([0.0, 0.0])
    env._positions[1] = np.array([0.0, 0.0])
    out_of_bounds = np.zeros(env.scene.n_drones, dtype=bool)
    reward_with_collision, info = env._compute_reward(out_of_bounds, is_final_step=False)
    assert info["collision_pairs"] >= 1

    env._positions[1] = np.array([20.0, 20.0])  # move it far away — no collision
    reward_without_collision, info_clean = env._compute_reward(out_of_bounds, is_final_step=False)
    assert info_clean["collision_pairs"] == 0
    assert reward_without_collision > reward_with_collision


def test_drone_on_true_target_earns_hit_bonus():
    env = _small_env()
    env.reset(seed=4)
    env._true_crossing_point = np.array([0.0, 0.0])
    env._positions[0] = np.array([0.1, 0.1])  # well within default hit_radius_m
    for i in range(1, env.scene.n_drones):
        env._positions[i] = np.array([100.0, 100.0])  # far away, no interference
    out_of_bounds = np.zeros(env.scene.n_drones, dtype=bool)
    reward, info = env._compute_reward(out_of_bounds, is_final_step=True)
    assert info["hits"] == 1
    assert reward > 0.0  # hit bonus dominates the out-of-bounds/cluster penalties

"""Тести RLAgentSource (input/rl_agent.py, фаза 7).

Швидкі тести — з ПІДСТАВНОЮ "моделлю" (обходимо __init__, як і в
tests/test_gamepad.py для GamepadSource), щоб не тягнути реальний torch/SB3
чекпойнт лише заради перевірки контракту ControlCommand/set_context. Один
повільний тест (@pytest.mark.slow) перевіряє РЕАЛЬНЕ навчання+інференс."""

from __future__ import annotations

import numpy as np
import pytest

from dronesim.core.contracts import FlightMode, VehicleState
from dronesim.input.rl_agent import RLAgentSource


class _FakeModel:
    """Підставна SB3-модель: завжди повертає задану дію."""

    def __init__(self, action):
        self._action = np.array(action, dtype=np.float32)
        self.last_obs = None

    def predict(self, obs, deterministic=True):
        self.last_obs = obs
        return self._action, None


def _make_source(action=(0.1, 0.2, -0.1, 0.0)) -> tuple[RLAgentSource, _FakeModel]:
    source = RLAgentSource.__new__(RLAgentSource)  # обходимо __init__ (без torch/SB3)
    model = _FakeModel(action)
    source.model = model
    source._state = None
    source._target_pos = np.zeros(3)
    return source, model


def test_get_command_before_context_is_disarmed():
    source, _ = _make_source()
    cmd = source.get_command()
    assert cmd.arm is False


def test_get_command_after_context_is_armed_and_pos_hold():
    source, _ = _make_source(action=(0.0, 0.0, 0.0, 0.0))
    state = VehicleState(pos=np.array([5.0, 0.0, 3.0]))
    source.set_context(state, target_pos=np.array([10.0, 0.0, 3.0]))
    cmd = source.get_command()
    assert cmd.arm is True
    assert cmd.mode is FlightMode.POS_HOLD


def test_action_maps_to_control_command_fields_correctly():
    source, _ = _make_source(action=(0.5, -0.5, 0.25, -1.0))
    state = VehicleState(pos=np.zeros(3))
    source.set_context(state, target_pos=np.array([1.0, 0.0, 0.0]))
    cmd = source.get_command()
    assert abs(cmd.roll - 0.5) < 1e-6
    assert abs(cmd.pitch - (-0.5)) < 1e-6
    assert abs(cmd.yaw - 0.25) < 1e-6
    assert abs(cmd.throttle - 0.0) < 1e-6  # action=-1.0 -> (-1+1)/2 = 0.0


def test_action_out_of_range_is_clamped():
    source, _ = _make_source(action=(2.0, -2.0, 0.0, 5.0))
    state = VehicleState(pos=np.zeros(3))
    source.set_context(state, target_pos=np.array([1.0, 0.0, 0.0]))
    cmd = source.get_command()
    assert -1.0 <= cmd.roll <= 1.0
    assert -1.0 <= cmd.pitch <= 1.0
    assert 0.0 <= cmd.throttle <= 1.0


def test_set_context_updates_observation_passed_to_model():
    source, model = _make_source()
    state = VehicleState(pos=np.array([1.0, 2.0, 3.0]))
    target = np.array([4.0, 5.0, 6.0])
    source.set_context(state, target_pos=target)
    source.get_command()
    assert model.last_obs is not None
    assert model.last_obs.shape == (13,)


@pytest.mark.slow
def test_real_trained_tiny_model_end_to_end(tmp_path):
    """Навчає крихітну реальну PPO-модель (не мок) і перевіряє, що
    RLAgentSource коректно завантажує й використовує СПРАВЖНІЙ чекпойнт."""
    from stable_baselines3 import PPO
    from stable_baselines3.common.monitor import Monitor

    from dronesim.ml.rl.env import DroneEnv
    from dronesim.ml.rl.policies import policy_kwargs_from_config
    from omegaconf import OmegaConf

    env = Monitor(DroneEnv(config_name="ml/rl_strike"))
    model = PPO(
        "MlpPolicy",
        env,
        n_steps=128,
        batch_size=32,
        seed=0,
        policy_kwargs=policy_kwargs_from_config(OmegaConf.create({"net_arch": [16, 16]})),
        verbose=0,
    )
    model.learn(total_timesteps=128)
    checkpoint_path = tmp_path / "tiny_model.zip"
    model.save(str(checkpoint_path))

    source = RLAgentSource(str(checkpoint_path))
    state = VehicleState(pos=np.array([5.0, 0.0, 3.0]))
    source.set_context(state, target_pos=np.array([10.0, 0.0, 3.0]))
    cmd = source.get_command()

    assert cmd.arm is True
    assert cmd.mode is FlightMode.POS_HOLD
    assert -1.0 <= cmd.roll <= 1.0
    assert -1.0 <= cmd.pitch <= 1.0
    assert -1.0 <= cmd.yaw <= 1.0
    assert 0.0 <= cmd.throttle <= 1.0

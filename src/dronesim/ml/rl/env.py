"""Gymnasium-середовище наведення на ціль (фаза 6).

Агент керує апаратом ЧЕРЕЗ ІСНУЮЧИЙ ``PosHoldController`` (vehicles/
controllers/position.py, фаза 3) — дія агента це ТОЙ САМИЙ ``ControlCommand``,
що й людина/геймпад подавали б у Pos-hold режимі (SKILL.md правило 2: джерело
керування взаємозамінне). RL вчиться НАВІГАЦІЇ (куди летіти), не стабілізації
польоту — стабілізація вже розв'язана й перевірена в фазі 3.

СТРАТЕГІЯ (ROADMAP, фаза 6): "телепорт-obs" — спостереження містить ІДЕАЛЬНУ
відносну позу цілі напряму з симуляції (не зашумлені детекції YOLO). Заміна на
детекції — окрема, пізніша ітерація поверх того самого контракту дій/розмірності.

Фізика (headless, без рендеру): кожен епізод створює власний ``PhysicsWorld``
+ ``Multirotor`` — простіше й безпечніше за повторне використання (без ризику
витоку стану між епізодами), а створення Bullet-світу дешеве (SKILL.md правило 6).
"""

from __future__ import annotations

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from dronesim.core.config import load_config
from dronesim.core.contracts import ControlCommand, FlightMode
from dronesim.ml.rl.observations import OBS_DIM, make_relative_obs
from dronesim.ml.rl.rewards import RewardConfig, compute_reward
from dronesim.physics.world import PhysicsWorld
from dronesim.vehicles.controllers.position import PosHoldController
from dronesim.vehicles.multirotor import Multirotor

__all__ = ["OBS_DIM", "DroneEnv", "make_relative_obs"]


class DroneEnv(gym.Env):
    """Навігація до випадково розставленої цілі, керування через Pos-hold."""

    metadata = {"render_modes": []}

    def __init__(self, config_name: str = "ml/rl_strike"):
        super().__init__()
        self.cfg = load_config(config_name)
        self.reward_cfg = RewardConfig(**self.cfg.reward)

        self.vehicle_config = str(self.cfg.vehicle_config)
        self.max_episode_steps = int(self.cfg.max_episode_steps)
        self.decision_substeps = int(self.cfg.decision_substeps)
        self._dt = 1.0 / float(self.cfg.physics_hz)

        self.target_radius_range = tuple(self.cfg.target_radius_range)
        self.target_altitude_range = tuple(self.cfg.target_altitude_range)
        self.start_radius_range = tuple(self.cfg.start_radius_range)
        self.start_altitude_range = tuple(self.cfg.start_altitude_range)

        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(4,), dtype=np.float32)
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(OBS_DIM,), dtype=np.float32
        )

        self._world: PhysicsWorld | None = None
        self._vehicle: Multirotor | None = None
        self._controller: PosHoldController | None = None
        self._target_pos = np.zeros(3)
        self._prev_distance = 0.0
        self._step_count = 0

    @property
    def step_duration_s(self) -> float:
        """Реальний час (с) одного виклику ``step()`` (``decision_substeps`` кроків фізики)."""
        return self.decision_substeps * self._dt

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)

        self._world = PhysicsWorld()
        self._vehicle = Multirotor(self._world, config_name=self.vehicle_config)
        self._controller = PosHoldController(self._vehicle)

        azimuth = self.np_random.uniform(0.0, 2.0 * np.pi)
        radius = self.np_random.uniform(*self.target_radius_range)
        altitude = self.np_random.uniform(*self.target_altitude_range)
        self._target_pos = np.array(
            [radius * np.cos(azimuth), radius * np.sin(azimuth), altitude]
        )

        start_azimuth = self.np_random.uniform(0.0, 2.0 * np.pi)
        start_radius = self.np_random.uniform(*self.start_radius_range)
        start_alt_offset = self.np_random.uniform(*self.start_altitude_range)
        start_pos = self._target_pos + np.array(
            [start_radius * np.cos(start_azimuth), start_radius * np.sin(start_azimuth), start_alt_offset]
        )
        start_pos[2] = max(float(start_pos[2]), 1.0)  # не стартувати під землею

        self._vehicle.reset(pos=start_pos)
        self._controller.reset(throttle_neutral=0.5)  # фіксована нейтраль — стаціонарна семантика дії
        self._prev_distance = float(np.linalg.norm(self._target_pos - start_pos))
        self._step_count = 0

        return self._make_obs(), {}

    def _make_obs(self) -> np.ndarray:
        return make_relative_obs(self._vehicle.state, self._target_pos)

    def step(self, action: np.ndarray):
        action = np.clip(np.asarray(action, dtype=np.float64), -1.0, 1.0)
        cmd = ControlCommand(
            roll=float(action[0]),
            pitch=float(action[1]),
            yaw=float(action[2]),
            throttle=float((action[3] + 1.0) / 2.0),
            arm=True,
            mode=FlightMode.POS_HOLD,
        )

        for _ in range(self.decision_substeps):
            thrusts = self._controller.update(cmd, self._vehicle.state, self._dt)
            self._vehicle.apply_motor_commands(thrusts, self._dt)
            self._world.step(self._dt)
            self._vehicle.advance_time(self._dt)

        state = self._vehicle.state
        result, distance = compute_reward(state, self._target_pos, self._prev_distance, self.reward_cfg)
        self._prev_distance = distance
        self._step_count += 1

        obs = self._make_obs()
        terminated = result.terminated
        truncated = self._step_count >= self.max_episode_steps
        info = {
            "hit": result.hit,
            "crashed": result.crashed,
            "out_of_bounds": result.out_of_bounds,
            "distance": distance,
        }
        return obs, result.reward, terminated, truncated, info

    def close(self) -> None:
        self._world = None
        self._vehicle = None
        self._controller = None

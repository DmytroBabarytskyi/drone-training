"""Винагорода для DroneEnv (фаза 6): зближення з ціллю, утримання ЛОС, штрафи.

Чиста функція — не залежить від Panda3D/Gymnasium, тестується напряму на
синтетичних ``VehicleState`` (як і scenarios/strike_range.py у фазі 4).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from dronesim.core.contracts import VehicleState
from dronesim.utils.math3d import vehicle_forward_world


@dataclass
class RewardConfig:
    """Усі числові параметри винагороди — з ``configs/ml/*.yaml`` (без магічних чисел)."""

    progress_scale: float = 10.0        # винагорода за зменшення дистанції (за метр)
    los_bonus: float = 0.05             # бонус/крок за утримання лінії прицілювання
    max_aim_angle_rad: float = 0.6      # ~34°, поріг "дивиться на ціль"
    hit_radius: float = 2.0             # м, поріг "влучання"
    hit_bonus: float = 100.0
    crash_penalty: float = -50.0
    out_of_bounds_penalty: float = -50.0
    out_of_bounds_radius: float = 60.0  # м від цілі
    time_penalty: float = -0.01         # за крок, заохочує ефективність
    min_altitude: float = 0.3           # м, нижче -> "краш"


@dataclass
class RewardResult:
    """Підсумок одного кроку винагороди."""

    reward: float
    hit: bool
    crashed: bool
    out_of_bounds: bool

    @property
    def terminated(self) -> bool:
        return self.hit or self.crashed or self.out_of_bounds


def compute_reward(
    state: VehicleState,
    target_pos: np.ndarray,
    prev_distance: float,
    cfg: RewardConfig,
) -> tuple[RewardResult, float]:
    """Один крок винагороди. Повертає ``(результат, нова_дистанція)`` —
    ``нова_дистанція`` передавай як ``prev_distance`` на наступному кроці."""
    distance = float(np.linalg.norm(target_pos - state.pos))
    progress = (prev_distance - distance) * cfg.progress_scale

    forward = vehicle_forward_world(state.quat)
    to_target = target_pos - state.pos
    safe_distance = max(distance, 1e-6)
    cos_angle = float(np.dot(forward, to_target)) / (float(np.linalg.norm(forward)) * safe_distance)
    cos_angle = min(1.0, max(-1.0, cos_angle))
    angle = np.arccos(cos_angle)
    facing_target = bool(angle <= cfg.max_aim_angle_rad)

    hit = bool(distance <= cfg.hit_radius and facing_target)
    crashed = bool(state.pos[2] <= cfg.min_altitude)
    out_of_bounds = bool(distance > cfg.out_of_bounds_radius)

    reward = progress + cfg.time_penalty
    if facing_target:
        reward += cfg.los_bonus
    if hit:
        reward += cfg.hit_bonus
    if crashed:
        reward += cfg.crash_penalty
    if out_of_bounds:
        reward += cfg.out_of_bounds_penalty

    result = RewardResult(reward=reward, hit=hit, crashed=crashed, out_of_bounds=out_of_bounds)
    return result, distance

"""Оцінка безпеки інтерактивного польоту (доп. фаза поліш): межі висоти/поля,
переворот. Чиста функція без Panda3D-залежностей — тестується headless.

НЕ форкає ``ml/rl/rewards.py::RewardConfig`` — та логіка визначає
crashed/out_of_bounds ВІДНОСНО ЦІЛІ (для навчання RL, ``DroneEnv``), тут
потрібні пороги ВІДНОСНО spawn-точки/вертикалі апарата (для людини-пілота) —
різна семантика, тому окремий конфіг (``configs/safety.yaml``) і дублювання
свідоме, не недогляд (docs/DECISIONS.md).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np

from dronesim.core.config import load_config
from dronesim.core.contracts import VehicleState
from dronesim.utils.math3d import quat_to_euler_rad

_RAD2DEG = 180.0 / np.pi

SafetyLevel = Literal["ok", "warning", "critical"]


@dataclass(frozen=True)
class SafetyStatus:
    level: SafetyLevel
    reason: str  # ASCII (HUD-сумісно), порожній рядок якщо level="ok"


@dataclass(frozen=True)
class SafetyConfig:
    soft_ceiling_m: float
    hard_ceiling_m: float
    bounds_radius_soft_m: float
    bounds_radius_hard_m: float
    crash_tilt_deg: float
    crash_grace_period_s: float
    restart_key: str


def load_safety_config(config_name: str = "safety") -> SafetyConfig:
    return SafetyConfig(**load_config(config_name))


def evaluate_flight_safety(
    state: VehicleState, spawn_xy: np.ndarray, cfg: SafetyConfig
) -> SafetyStatus:
    """Оцінити поточний стан апарата проти меж ``cfg``. Порядок перевірок:
    спершу ВСІ critical-умови (найважливіші для безпеки), потім warning —
    інакше "критично високо, але ще й трохи занадто далеко" показало б лише
    попередження про межі поля, ховаючи важливішу проблему висоти."""
    altitude = float(state.pos[2])
    horizontal_dist = float(np.linalg.norm(state.pos[:2] - spawn_xy))
    roll_rad, pitch_rad, _yaw_rad = quat_to_euler_rad(state.quat)
    tilt_deg = max(abs(roll_rad), abs(pitch_rad)) * _RAD2DEG

    if tilt_deg >= cfg.crash_tilt_deg:
        return SafetyStatus("critical", f"FLIPPED ({tilt_deg:.0f} deg tilt) - restarting")
    if altitude >= cfg.hard_ceiling_m:
        return SafetyStatus(
            "critical", f"ALTITUDE LIMIT {altitude:.0f}m >= {cfg.hard_ceiling_m:.0f}m - restarting"
        )
    if horizontal_dist >= cfg.bounds_radius_hard_m:
        return SafetyStatus(
            "critical",
            f"OUT OF BOUNDS {horizontal_dist:.0f}m >= {cfg.bounds_radius_hard_m:.0f}m - restarting",
        )

    if altitude >= cfg.soft_ceiling_m:
        return SafetyStatus(
            "warning", f"Approaching altitude limit: {altitude:.0f}/{cfg.hard_ceiling_m:.0f}m"
        )
    if horizontal_dist >= cfg.bounds_radius_soft_m:
        return SafetyStatus(
            "warning",
            f"Approaching field boundary: {horizontal_dist:.0f}/{cfg.bounds_radius_hard_m:.0f}m",
        )

    return SafetyStatus("ok", "")

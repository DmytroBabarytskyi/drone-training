"""Тести core/safety.py (доп. фаза поліш) — чиста функція, без Panda3D."""

from __future__ import annotations

import numpy as np

from dronesim.core.contracts import VehicleState
from dronesim.core.safety import SafetyConfig, evaluate_flight_safety, load_safety_config

_CFG = SafetyConfig(
    soft_ceiling_m=40.0,
    hard_ceiling_m=80.0,
    bounds_radius_soft_m=60.0,
    bounds_radius_hard_m=120.0,
    crash_tilt_deg=70.0,
    crash_grace_period_s=3.0,
    restart_key="r",
)
_SPAWN = np.zeros(2)


def _roll_quat(deg: float) -> np.ndarray:
    theta = np.radians(deg)
    return np.array([np.sin(theta / 2), 0.0, 0.0, np.cos(theta / 2)])


def test_ok_when_within_all_limits():
    state = VehicleState(pos=np.array([5.0, 0.0, 10.0]))
    status = evaluate_flight_safety(state, _SPAWN, _CFG)
    assert status.level == "ok"
    assert status.reason == ""


def test_warning_near_soft_ceiling():
    state = VehicleState(pos=np.array([0.0, 0.0, 45.0]))
    status = evaluate_flight_safety(state, _SPAWN, _CFG)
    assert status.level == "warning"
    assert "altitude" in status.reason.lower()


def test_critical_past_hard_ceiling():
    state = VehicleState(pos=np.array([0.0, 0.0, 90.0]))
    status = evaluate_flight_safety(state, _SPAWN, _CFG)
    assert status.level == "critical"
    assert "ALTITUDE" in status.reason


def test_warning_near_soft_bounds_radius():
    state = VehicleState(pos=np.array([65.0, 0.0, 5.0]))
    status = evaluate_flight_safety(state, _SPAWN, _CFG)
    assert status.level == "warning"
    assert "boundary" in status.reason.lower()


def test_critical_past_hard_bounds_radius():
    state = VehicleState(pos=np.array([130.0, 0.0, 5.0]))
    status = evaluate_flight_safety(state, _SPAWN, _CFG)
    assert status.level == "critical"
    assert "OUT OF BOUNDS" in status.reason


def test_critical_when_flipped_over():
    state = VehicleState(pos=np.array([0.0, 0.0, 5.0]), quat=_roll_quat(90.0))
    status = evaluate_flight_safety(state, _SPAWN, _CFG)
    assert status.level == "critical"
    assert "FLIPPED" in status.reason


def test_ok_with_small_tilt_below_threshold():
    state = VehicleState(pos=np.array([0.0, 0.0, 5.0]), quat=_roll_quat(20.0))
    status = evaluate_flight_safety(state, _SPAWN, _CFG)
    assert status.level == "ok"


def test_critical_takes_priority_over_warning():
    # Достатньо далеко за М'ЯКУ межу поля (спрацював би warning), АЛЕ й
    # перевернутий (critical) — critical має перемогти, а не сховатись.
    state = VehicleState(pos=np.array([65.0, 0.0, 5.0]), quat=_roll_quat(90.0))
    status = evaluate_flight_safety(state, _SPAWN, _CFG)
    assert status.level == "critical"
    assert "FLIPPED" in status.reason


def test_load_safety_config_reads_yaml():
    cfg = load_safety_config()
    assert cfg.hard_ceiling_m > cfg.soft_ceiling_m > 0
    assert cfg.bounds_radius_hard_m > cfg.bounds_radius_soft_m > 0
    assert cfg.restart_key

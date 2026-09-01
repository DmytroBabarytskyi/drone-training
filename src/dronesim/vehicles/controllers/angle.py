"""Angle/Self-level польотний режим: стік → цільовий КУТ нахилу (не швидкість).

На відміну від Acro (фаза 2): відпустив стік — апарат сам вирівнюється до
нульового крену/тангажу. Рискання (yaw) лишається rate-контрольованим, як в
Acro, — "утримання курсу" (heading-hold) це окрема функція, тут не реалізована.

Каскадна схема: зовнішній PID (кут → бажана кутова швидкість) → внутрішній
``RateLoopMixer`` (той самий, що й в Acro) → мотор-мікшер. ``update_from_angles``
—低-рівневий вхід, яким користуються ``AltHoldController``/``PosHoldController``
(vehicles/controllers/position.py), щоб подати вже ОБЧИСЛЕНИЙ цільовий кут
(не стік) у той самий контур (docs/DECISIONS.md, «Фаза 3: каскадні контролери»).
"""

from __future__ import annotations

import numpy as np

from dronesim.core.contracts import ControlCommand, VehicleState
from dronesim.utils.math3d import quat_to_euler_rad
from dronesim.utils.pid import PID
from dronesim.vehicles.controllers.rate_loop import RateLoopMixer
from dronesim.vehicles.multirotor import Multirotor

_DEG2RAD = np.pi / 180.0
_ANGLE_AXES = ("roll", "pitch")


class AngleController:
    """Стік → цільовий кут → PID → бажана кутова швидкість → RateLoopMixer."""

    def __init__(self, vehicle: Multirotor):
        self.vehicle = vehicle
        cfg = vehicle.cfg

        self._rate_loop = RateLoopMixer(vehicle)
        self._angle_pid = {axis: PID(**cfg.pid_angle[axis]) for axis in _ANGLE_AXES}
        # Публічні: PosHold/AltHold (position.py) читають їх напряму для власних розрахунків.
        self.max_angle_rad = float(cfg.max_angle_deg) * _DEG2RAD
        # Yaw і в Angle-режимі лишається rate-контрольованим (докстрінг модуля) —
        # максимальна швидкість рискання з `rate_profile` (доп. фаза "справжній
        # Liftoff", напр. fpv_5inch) або старого `max_rates` (quad_large).
        if "rate_profile" in cfg:
            self.max_yaw_rate_rad = float(cfg.rate_profile.yaw.max_rate_deg_s) * _DEG2RAD
        else:
            self.max_yaw_rate_rad = float(cfg.max_rates.yaw) * _DEG2RAD

    def reset(self) -> None:
        for pid in self._angle_pid.values():
            pid.reset()
        self._rate_loop.reset()

    def update_from_angles(
        self,
        desired_roll_rad: float,
        desired_pitch_rad: float,
        desired_yaw_rate_rad: float,
        throttle_cmd: float,
        state: VehicleState,
        dt: float,
    ) -> np.ndarray:
        """Низькорівневий крок: цільові КУТИ (рад) + бажана швидкість рискання (рад/с).

        Використовується і ``update()`` (стік), і зовнішніми контурами AltHold/
        PosHold, яким потрібно подати вже обчислений кут напряму, не через стік.
        """
        current_roll_rad, current_pitch_rad, _ = quat_to_euler_rad(state.quat)

        desired_rates = {
            "roll": self._angle_pid["roll"].update(desired_roll_rad - current_roll_rad, dt),
            "pitch": self._angle_pid["pitch"].update(desired_pitch_rad - current_pitch_rad, dt),
            "yaw": desired_yaw_rate_rad,
        }
        return self._rate_loop.update_from_rates(desired_rates, state, throttle_cmd, dt)

    def update(self, cmd: ControlCommand, state: VehicleState, dt: float) -> np.ndarray:
        """Один крок Angle: стік → цільовий кут (як частка ``max_angle_deg``)."""
        desired_roll = cmd.roll * self.max_angle_rad
        desired_pitch = cmd.pitch * self.max_angle_rad
        desired_yaw_rate = cmd.yaw * self.max_yaw_rate_rad
        throttle_cmd = float(np.clip(cmd.throttle, 0.0, 1.0)) if cmd.arm else 0.0
        return self.update_from_angles(
            desired_roll, desired_pitch, desired_yaw_rate, throttle_cmd, state, dt
        )

"""Alt-hold і Pos-hold: автоматичне утримання висоти/позиції (фаза 3).

Каскадна архітектура (як у реальних політних контролерах, PX4/ArduPilot):

    PosHold:  похибка XY-позиції -> бажана горизонтальна швидкість (PID)
                  -> похибка швидкості -> цільовий кут нахилу (PID)
    AltHold:  похибка висоти -> корекція нормованої команди мотора (один PID)
    Angle:    цільовий кут -> бажана кутова швидкість (PID) -> RateLoopMixer

``AltHoldController`` обгортає ``AngleController`` (замінює лише throttle на
автоматичний, roll/pitch лишає керованими стіком чи PosHold). ``PosHoldController``
обгортає ``AltHoldController`` (замінює roll/pitch стік на обчислений кут).

СПРОЩЕННЯ (свідоме, для фази 3 — див. docs/DECISIONS.md): контур позиції НЕ
компенсує рискання (yaw) апарата — тілт вважається у світових XY напряму, без
повороту в систему координат апарата. Коректно для тестів, де yaw лишається
близько 0 (вітер — чиста лінійна сила без моменту, physics/wind.py). Якщо
знадобиться довільний yaw під час Pos-hold — тут точка розширення.

Throttle-стік у AltHold/PosHold означає ІНШЕ, ніж в Acro/Angle: не "потужність
мотора", а "бажана швидкість зміни цільової висоти" відносно нейтралі. За
замовчуванням контролер ЛІНИВО фіксує нейтраль при першому виклику (яким би не
був стік джерела введення на той момент) — так само лінива фіксація цільової
висоти/позиції. Це уникає залежності від конвенції конкретного
``ControlSource`` (клавіатура тримає холостий хід на 0.0, геймпад — на 0.5 при
центрованому стіку).

RL-ВИНЯТОК (фаза 6, docs/DECISIONS.md): лінива фіксація ОК для людини (стік
завжди має якесь визначене положення при вході в режим), але для RL-агента
перша дія випадкова (політика ще не навчена) — це зробило б "нейтраль"
недетермінованою від епізоду до епізоду, ускладнюючи навчання. Тому
``reset(throttle_neutral=...)`` дозволяє ЗАДАТИ фіксоване значення одразу,
замість очікування першого виклику ``update()``.
"""

from __future__ import annotations

import numpy as np

from dronesim.core.contracts import ControlCommand, VehicleState
from dronesim.utils.pid import PID
from dronesim.vehicles.controllers.angle import AngleController
from dronesim.vehicles.multirotor import Multirotor

_HORIZONTAL_AXES = ("x", "y")


def _hover_throttle_cmd(vehicle: Multirotor) -> float:
    """Нормована команда мотора (0..1), за якої сумарна тяга 4 моторів = вазі."""
    weight = vehicle.cfg.mass * 9.81
    thrust_per_motor = weight / 4.0
    return float(np.clip(np.sqrt(thrust_per_motor / vehicle.cfg.max_thrust_per_motor), 0.0, 1.0))


class AltHoldController:
    """Angle + автоматичне утримання висоти. Стік газу -> швидкість зміни цільової висоти."""

    def __init__(self, vehicle: Multirotor):
        self.vehicle = vehicle
        self.angle = AngleController(vehicle)  # публічно: PosHold читає напряму
        self._altitude_pid = PID(**vehicle.cfg.pid_altitude)
        self._max_climb_rate = float(vehicle.cfg.max_climb_rate_mps)
        self._hover_throttle = _hover_throttle_cmd(vehicle)

        self._target_altitude: float | None = None
        self._throttle_neutral: float | None = None

    def reset(self, throttle_neutral: float | None = None) -> None:
        self.angle.reset()
        self._altitude_pid.reset()
        self._target_altitude = None
        self._throttle_neutral = throttle_neutral

    def compute_throttle_cmd(self, cmd: ControlCommand, state: VehicleState, dt: float) -> float:
        """Публічно: PosHoldController викликає це напряму, обходячи roll/pitch update()."""
        if self._target_altitude is None:
            self._target_altitude = state.pos[2]
        if self._throttle_neutral is None:
            self._throttle_neutral = cmd.throttle

        climb_rate_cmd = (cmd.throttle - self._throttle_neutral) * self._max_climb_rate
        self._target_altitude += climb_rate_cmd * dt

        altitude_error = self._target_altitude - state.pos[2]
        throttle_correction = self._altitude_pid.update(altitude_error, dt)
        return float(np.clip(self._hover_throttle + throttle_correction, 0.0, 1.0))

    def update(self, cmd: ControlCommand, state: VehicleState, dt: float) -> np.ndarray:
        """Один крок Alt-hold: roll/pitch/yaw від стіка (Angle), throttle — автоматично."""
        if not cmd.arm:
            self._target_altitude = None
            self._throttle_neutral = None
            return self.angle.update_from_angles(0.0, 0.0, 0.0, 0.0, state, dt)

        throttle_cmd = self.compute_throttle_cmd(cmd, state, dt)
        desired_roll = cmd.roll * self.angle.max_angle_rad
        desired_pitch = cmd.pitch * self.angle.max_angle_rad
        desired_yaw_rate = cmd.yaw * self.angle.max_yaw_rate_rad
        return self.angle.update_from_angles(
            desired_roll, desired_pitch, desired_yaw_rate, throttle_cmd, state, dt
        )


class PosHoldController:
    """Alt-hold + автоматичне утримання горизонтальної позиції (GPS-hold/Loiter).

    Стік центровано -> тримає точку (протидіє вітру). Стік відхилено -> летить
    у цьому напрямку зі швидкістю до ``max_horizontal_speed_mps``, і ціль
    позиції неперервно "переносить" в поточну точку (відпустив — тримає нову)."""

    def __init__(self, vehicle: Multirotor):
        self.vehicle = vehicle
        self._alt_hold = AltHoldController(vehicle)

        self._pos_pid = {axis: PID(**vehicle.cfg.pid_position) for axis in _HORIZONTAL_AXES}
        self._vel_pid = {axis: PID(**vehicle.cfg.pid_velocity) for axis in _HORIZONTAL_AXES}
        self._max_speed = float(vehicle.cfg.max_horizontal_speed_mps)
        self._max_tilt_rad = self._alt_hold.angle.max_angle_rad
        self._stick_deadband = 0.05

        self._target_xy: np.ndarray | None = None

    def reset(self, throttle_neutral: float | None = None) -> None:
        self._alt_hold.reset(throttle_neutral=throttle_neutral)
        for pid in self._pos_pid.values():
            pid.reset()
        for pid in self._vel_pid.values():
            pid.reset()
        self._target_xy = None

    def update(self, cmd: ControlCommand, state: VehicleState, dt: float) -> np.ndarray:
        if not cmd.arm:
            self._target_xy = None
            return self._alt_hold.update(cmd, state, dt)

        if self._target_xy is None:
            self._target_xy = state.pos[:2].copy()

        # pitch стік -> world-X (вперед/назад), roll стік -> world-Y (право/ліво);
        # коректно лише за yaw≈0 (див. спрощення у докстрінгу модуля).
        stick_xy = np.array([cmd.pitch, cmd.roll])
        manual = np.linalg.norm(stick_xy) > self._stick_deadband

        if manual:
            desired_vel_xy = stick_xy * self._max_speed
            self._target_xy = state.pos[:2].copy()  # тримати "тут" одразу після відпускання
        else:
            pos_error = self._target_xy - state.pos[:2]
            desired_vel_xy = np.array(
                [self._pos_pid[axis].update(pos_error[i], dt) for i, axis in enumerate(_HORIZONTAL_AXES)]
            )

        vel_error_xy = desired_vel_xy - state.vel[:2]
        tilt = np.array(
            [self._vel_pid[axis].update(vel_error_xy[i], dt) for i, axis in enumerate(_HORIZONTAL_AXES)]
        )
        tilt = np.clip(tilt, -self._max_tilt_rad, self._max_tilt_rad)

        # Емпірично перевірено (docs/DECISIONS.md, «Фаза 3: знак тілту Y/roll»):
        # позитивний КУТ КРЕНУ дає ВІД'ЄМНЕ прискорення по world-Y (на відміну від
        # pitch/X, де знаки збігаються) — тому крен інвертуємо тут, в ОДНОМУ місці.
        desired_pitch, desired_roll = tilt[0], -tilt[1]
        throttle_cmd = self._alt_hold.compute_throttle_cmd(cmd, state, dt)
        desired_yaw_rate = cmd.yaw * self._alt_hold.angle.max_yaw_rate_rad
        return self._alt_hold.angle.update_from_angles(
            desired_roll, desired_pitch, desired_yaw_rate, throttle_cmd, state, dt
        )

"""Спільний внутрішній контур: PID rate-loop + мотор-мікшер X-квада.

Виділено з ``acro.py`` у фазі 3, коли з'явився Angle-режим: обидва (Acro і
Angle) закінчуються ОДНИМ і тим самим внутрішнім контуром "бажана кутова
швидкість (рад/с) → PID → мікшер → тяги моторів". Acro подає сюди
`стік*max_rate` напряму; Angle подає вихід ЗОВНІШНЬОГО кутового PID
(docs/DECISIONS.md, запис «Фаза 3: каскадні контролери»).

Знаки мотор-мікшера виводяться аналітично з геометрії ``Multirotor.MOTOR_LAYOUT``
(момент r×F = (y·f, -x·f, spin_dir·k·f) відносно (roll, pitch, yaw)) і
підтверджені емпірично в тестах напрямку обертання (tests/test_acro_controller.py).
Мікшер працює в нормованому просторі команди мотора (0..1, як PWM); крива тяги
(``aerodynamics.motor_thrust``) застосовується ПІСЛЯ мікшування — так само, як у
реальних політних контролерах (нелінійність тяги живе нижче рівня мікшера).

ДОП. ФАЗА "РЕАЛІЗМ ФІЗИКИ" — усі нижче опційні (``cfg.get``, відсутність поля
= стара поведінка):
- ``rate_feedforward`` (дефолт 0 на вісь): доданок, пропорційний ШВИДКОСТІ
  ЗМІНИ бажаної кутової швидкості (не самій похибці) — реальні FC додають
  feed-forward, щоб стік відчувався "зчепленим", не чекаючи, доки PID
  накопичить похибку від різкого руху стіка.
- ``airmode`` (дефолт False): мікшер РОЗТЯГУЄ (не обрізає) вихід при насиченні
  — на нульовому газі корекції нахилу інакше просто обрізаються до 0, і
  апарат втрачає керованість (реальний BF-airmode: якщо мінімальний мотор
  вийшов би за 0 — зсунути ВСІ на дефіцит, зберігши повний розкид корекцій).
- ``gyro_noise_std_rad_s``/``gyro_noise_seed`` (дефолт 0=вимкнено): адитивний
  гаусів шум на ЗЧИТАНУ кутову швидкість (симуляція шуму гіроскопа) — впливає
  лише на те, що БАЧИТЬ контролер, не на істинний ``VehicleState`` (контракт
  лишається чистою фізичною правдою, SKILL.md правило "не змішуй шари").
- ``control_latency_steps`` (дефолт 0=вимкнено): затримка N кроків фізики між
  обчисленням команди й тим, яку тягу РЕАЛЬНО повертає мікшер (реальний FC
  теж має латентність сенсор->обчислення->привід).
"""

from __future__ import annotations

from collections import deque

import numpy as np

from dronesim.core.contracts import VehicleState
from dronesim.physics import aerodynamics
from dronesim.utils.pid import PID
from dronesim.vehicles.multirotor import Multirotor

_RATE_AXES = ("roll", "pitch", "yaw")


def world_to_body_ang_vel(quat_xyzw: np.ndarray, ang_vel_world: np.ndarray) -> np.ndarray:
    """Перевести кутову швидкість зі світових координат у зв'язані (тіла).

    Гіроскоп реального польотного контролера вимірює кутову швидкість у
    зв'язаних осях, тож rate-loop має порівнювати команду саме з нею, а не зі
    світовою ``VehicleState.ang_vel``. Обернений поворот для одиничного
    кватерніона — це транспонована матриця повороту: ``body = R(q)^T @ world``.
    """
    x, y, z, w = quat_xyzw
    rot = np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ]
    )
    return rot.T @ ang_vel_world


class RateLoopMixer:
    """PID rate-loop (roll/pitch/yaw) + мотор-мікшер. Спільний для Acro й Angle."""

    def __init__(self, vehicle: Multirotor):
        self.vehicle = vehicle
        cfg = vehicle.cfg

        self._pid = {axis: PID(**cfg.pid_rate[axis]) for axis in _RATE_AXES}

        layout = np.asarray(vehicle.MOTOR_LAYOUT, dtype=np.float64)  # (4,3): x, y, spin_dir
        self._roll_sign = np.sign(layout[:, 1])
        self._pitch_sign = np.sign(-layout[:, 0])
        self._yaw_sign = layout[:, 2]

        self._airmode = bool(cfg.get("airmode", False))

        ff_cfg = cfg.get("rate_feedforward", None) or {}
        self._ff_gain = {axis: float(ff_cfg.get(axis, 0.0)) for axis in _RATE_AXES}
        self._prev_desired_rate = {axis: 0.0 for axis in _RATE_AXES}

        gyro_noise_std = float(cfg.get("gyro_noise_std_rad_s", 0.0))
        self._gyro_noise_std = gyro_noise_std
        self._gyro_rng = (
            np.random.default_rng(int(cfg.get("gyro_noise_seed", 0))) if gyro_noise_std > 0.0 else None
        )

        latency_steps = int(cfg.get("control_latency_steps", 0))
        self._latency_queue: deque | None = deque(maxlen=latency_steps + 1) if latency_steps > 0 else None

    def reset(self) -> None:
        for pid in self._pid.values():
            pid.reset()
        self._prev_desired_rate = {axis: 0.0 for axis in _RATE_AXES}
        if self._latency_queue is not None:
            self._latency_queue.clear()

    def update_from_rates(
        self,
        desired_rates_rad: dict[str, float],
        state: VehicleState,
        throttle_cmd: float,
        dt: float,
    ) -> np.ndarray:
        """Крок rate-loop із ГОТОВИМИ бажаними кутовими швидкостями (рад/с).

        ``throttle_cmd`` — вже нормована команда мотора (0..1), 0 якщо дизармовано.
        Повертає 4 тяги моторів (Н) для ``Vehicle.apply_motor_commands``.
        """
        body_rate = world_to_body_ang_vel(state.quat, state.ang_vel)
        if self._gyro_rng is not None:
            body_rate = body_rate + self._gyro_rng.normal(0.0, self._gyro_noise_std, size=3)

        pid_out = {}
        for i, axis in enumerate(_RATE_AXES):
            desired = desired_rates_rad[axis]
            error = desired - body_rate[i]
            out = self._pid[axis].update(error, dt, measurement=body_rate[i])

            ff_gain = self._ff_gain[axis]
            if ff_gain != 0.0 and dt > 0.0:
                out += ff_gain * (desired - self._prev_desired_rate[axis]) / dt
            self._prev_desired_rate[axis] = desired

            pid_out[axis] = out

        motor_norm = (
            throttle_cmd
            + pid_out["roll"] * self._roll_sign
            + pid_out["pitch"] * self._pitch_sign
            + pid_out["yaw"] * self._yaw_sign
        )

        if self._airmode:
            # Розтягнути (не обрізати) при насиченні — зберігає ПОВНИЙ розкид
            # корекцій нахилу навіть на малому/нульовому газі (реальний BF-airmode).
            min_val = float(motor_norm.min())
            if min_val < 0.0:
                motor_norm = motor_norm - min_val
            max_val = float(motor_norm.max())
            if max_val > 1.0:
                motor_norm = motor_norm - (max_val - 1.0)

        motor_norm = np.clip(motor_norm, 0.0, 1.0)  # остання страховка

        max_thrust = self.vehicle.cfg.max_thrust_per_motor
        thrusts = np.array([aerodynamics.motor_thrust(c, max_thrust) for c in motor_norm])

        if self._latency_queue is not None:
            self._latency_queue.append(thrusts)
            return self._latency_queue[0]  # найстаріший у черзі -> N-кроків затримки
        return thrusts

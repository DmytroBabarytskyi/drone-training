"""Acro/Rate польотний режим: стіки → цільові кутові швидкості → PID → тяги моторів.

Це основний режим Liftoff: стік задає КУТОВУ ШВИДКІСТЬ (не кут) — відпустив
стік, обертання триває, доки PID не загасить помилку до нуля; апарат сам не
вирівнюється (на відміну від Angle-режиму, фаза 3, vehicles/controllers/angle.py).

Контракт: ``AcroController.update(cmd, state, dt) -> np.ndarray(4,)`` — тяги в
Ньютонах, готові для ``Vehicle.apply_motor_commands`` (vehicles/base.py).

Сам rate-loop + мотор-мікшер — у ``vehicles/controllers/rate_loop.py``
(``RateLoopMixer``), спільний з Angle-режимом. Тут лише переклад стіка в
бажану кутову швидкість — двома можливими способами (доп. фаза "справжній
Liftoff"): якщо конфіг апарата має ``rate_profile`` (напр. ``fpv_5inch`` —
реалістичний Acro-апарат) — реальна формула BetaFlight ``rates.py::
actual_rate_rad_s``; якщо лише старий ``max_rates`` (напр. ``quad_large`` —
літає в Angle/Alt-hold/Pos-hold, Acro тут лише запасний режим) — простий
лінійний ``stick * max_rate``, без форку класу під два конфіги.
"""

from __future__ import annotations

import numpy as np

from dronesim.core.contracts import ControlCommand, VehicleState
from dronesim.vehicles.controllers.rate_loop import RateLoopMixer
from dronesim.vehicles.controllers.rates import actual_rate_rad_s
from dronesim.vehicles.multirotor import Multirotor

_DEG2RAD = np.pi / 180.0
_RATE_AXES = ("roll", "pitch", "yaw")


class AcroController:
    """Стік → кутова швидкість → RateLoopMixer, для X-квадрокоптера в Acro-режимі."""

    def __init__(self, vehicle: Multirotor):
        self.vehicle = vehicle
        self._rate_loop = RateLoopMixer(vehicle)
        self._rate_profile = vehicle.cfg.rate_profile if "rate_profile" in vehicle.cfg else None
        self._max_rates_rad = (
            None
            if self._rate_profile is not None
            else {axis: float(vehicle.cfg.max_rates[axis]) * _DEG2RAD for axis in _RATE_AXES}
        )

    def reset(self) -> None:
        self._rate_loop.reset()

    def _desired_rate_rad_s(self, axis: str, stick: float) -> float:
        if self._rate_profile is not None:
            p = self._rate_profile[axis]
            return actual_rate_rad_s(stick, p.center_sensitivity_deg_s, p.max_rate_deg_s, p.expo)
        return stick * self._max_rates_rad[axis]

    def update(self, cmd: ControlCommand, state: VehicleState, dt: float) -> np.ndarray:
        """Один крок Acro: повертає 4 тяги моторів (Н) для цього кроку фізики."""
        desired_rates = {
            "roll": self._desired_rate_rad_s("roll", cmd.roll),
            "pitch": self._desired_rate_rad_s("pitch", cmd.pitch),
            "yaw": self._desired_rate_rad_s("yaw", cmd.yaw),
        }
        throttle_cmd = float(np.clip(cmd.throttle, 0.0, 1.0)) if cmd.arm else 0.0
        return self._rate_loop.update_from_rates(desired_rates, state, throttle_cmd, dt)

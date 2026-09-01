"""Джерело керування — клавіатура через події Panda3D (``base.accept``).

Клавіатура цифрова (натиснуто/ні), а стік — аналоговий, тож емулюємо
"віртуальний стік": утримання клавіші плавно розганяє вісь до ±1 (``ramp``),
відпускання повертає крен/тангаж/рискання до нейтралі (як пружинний стік
Acro-режиму). Throttle — виняток: не самоцентрується (як реальний важіль газу
з тертям), тримає останнє значення, поки не натиснуто up/down.

Приймає будь-який об'єкт із методом ``accept(event, callback, extraArgs)``
(типово — ``ShowBase``/``DirectObject``), щоб тестуватися без реального вікна.
"""

from __future__ import annotations

from typing import Protocol

from dronesim.core.config import load_config
from dronesim.core.contracts import FLIGHT_MODE_CYCLE, ControlCommand, FlightMode
from dronesim.input.base import ControlSource
from dronesim.input.calibration import apply_expo

_STICK_AXES = ("roll", "pitch", "yaw")  # самоцентрувальні (пружинний стік)


class _Acceptor(Protocol):
    def accept(self, event: str, callback, extraArgs: list) -> None: ...  # noqa: N803


class KeyboardSource(ControlSource):
    """Стан клавіш → нормовані осі з плавним розгоном (ramp) і expo-кривою."""

    def __init__(self, base: _Acceptor, config_name: str = "input/keyboard"):
        self.cfg = load_config(config_name)
        self._keys_down: dict[str, bool] = {}
        self._axis_values = {"roll": 0.0, "pitch": 0.0, "yaw": 0.0, "throttle": -1.0}
        self._armed = False
        self._prev_arm_key = False
        self._mode_index = 0
        self._prev_mode_key = False

        for action, key in self.cfg.bindings.items():
            base.accept(str(key), self._set_key, [action, True])
            base.accept(f"{key}-up", self._set_key, [action, False])

    def set_start_mode(self, mode: FlightMode) -> None:
        """Встановити режим, з якого стартує цикл ``Tab`` (доп. фаза "справжній
        Liftoff" — стабільніший стартовий режим за замовчуванням, якщо апарат
        його підтримує; викликати ДО першого ``get_command()``)."""
        self._mode_index = FLIGHT_MODE_CYCLE.index(mode)

    def _set_key(self, action: str, pressed: bool) -> None:
        self._keys_down[action] = pressed

    def _is_down(self, action: str) -> bool:
        return self._keys_down.get(action, False)

    def _update_axis(
        self, name: str, positive_action: str, negative_action: str, dt: float, self_center: bool, rate: float
    ) -> None:
        direction = 0.0
        if self._is_down(positive_action):
            direction += 1.0
        if self._is_down(negative_action):
            direction -= 1.0

        current = self._axis_values[name]

        if direction != 0.0:
            current = max(-1.0, min(1.0, current + direction * rate * dt))
        elif self_center:
            if current > 0.0:
                current = max(0.0, current - rate * dt)
            elif current < 0.0:
                current = min(0.0, current + rate * dt)
        # інакше (throttle): тримає поточне значення без змін

        self._axis_values[name] = current

    def update(self, dt: float) -> None:
        """Оновити внутрішній стан осей. Викликати щокадру з реальним dt рендеру."""
        tilt_rate = self.cfg.ramp_per_second
        # Газ має ОКРЕМИЙ, повільніший ramp (реальний важіль газу не стрибає
        # 0->100% за пів секунди); fallback на tilt-rate для старих конфігів
        # без поля, щоб не зламати зворотну сумісність.
        throttle_rate = float(self.cfg.get("throttle_ramp_per_second", tilt_rate))
        self._update_axis("roll", "roll_right", "roll_left", dt, self_center=True, rate=tilt_rate)
        self._update_axis("pitch", "pitch_forward", "pitch_back", dt, self_center=True, rate=tilt_rate)
        self._update_axis("yaw", "yaw_right", "yaw_left", dt, self_center=True, rate=tilt_rate)
        self._update_axis("throttle", "throttle_up", "throttle_down", dt, self_center=False, rate=throttle_rate)

        arm_key = self._is_down("arm_toggle")
        if arm_key and not self._prev_arm_key:
            self._armed = not self._armed
        self._prev_arm_key = arm_key

        mode_key = self._is_down("mode_cycle")
        if mode_key and not self._prev_mode_key:
            self._mode_index = (self._mode_index + 1) % len(FLIGHT_MODE_CYCLE)
        self._prev_mode_key = mode_key

    def get_command(self) -> ControlCommand:
        expo = self.cfg.expo
        return ControlCommand(
            roll=apply_expo(self._axis_values["roll"], expo),
            pitch=apply_expo(self._axis_values["pitch"], expo),
            yaw=apply_expo(self._axis_values["yaw"], expo),
            throttle=(self._axis_values["throttle"] + 1.0) / 2.0,
            arm=self._armed,
            mode=FLIGHT_MODE_CYCLE[self._mode_index],
        )

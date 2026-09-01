"""Джерело керування — геймпад PlayStation (DualShock4/DualSense) через pygame/SDL.

Мапінг осей/кнопок і криві (deadzone/expo) — з ``configs/input/ps_gamepad.yaml``
(``core/config.py``). Індекси осей НЕ хардкодяться в коді (SKILL.md, правило 3) —
вони різняться між ОС/драйверами і калібруються в YAML.
"""

from __future__ import annotations

import pygame

from dronesim.core.config import load_config
from dronesim.core.contracts import FLIGHT_MODE_CYCLE, ControlCommand, FlightMode
from dronesim.input.base import ControlSource
from dronesim.input.calibration import apply_axis_curve


class GamepadSource(ControlSource):
    """Читає стан першого підключеного джойстика щокроку керування."""

    def __init__(self, config_name: str = "input/ps_gamepad", joystick_index: int = 0):
        self.cfg = load_config(config_name)
        self._armed = False
        self._prev_arm_button = False
        self._mode_index = 0
        self._prev_mode_button = False

        if not pygame.get_init():
            pygame.init()
        pygame.joystick.init()
        if pygame.joystick.get_count() == 0:
            raise RuntimeError(
                "Геймпад не знайдено. Підключи DualShock4/DualSense або запусти "
                "`dronesim fly --input keyboard`."
            )
        self._joystick = pygame.joystick.Joystick(joystick_index)
        self._joystick.init()

    def set_start_mode(self, mode: FlightMode) -> None:
        """Встановити режим, з якого стартує цикл перемикача (доп. фаза
        "справжній Liftoff" — стабільніший стартовий режим за замовчуванням,
        якщо апарат його підтримує; викликати ДО першого ``get_command()``)."""
        self._mode_index = FLIGHT_MODE_CYCLE.index(mode)

    def _read_axis(self, name: str) -> float:
        axis_cfg = self.cfg.axes[name]
        raw = self._joystick.get_axis(axis_cfg.index)
        return apply_axis_curve(raw, self.cfg.deadzone, self.cfg.expo, bool(axis_cfg.invert))

    def get_command(self) -> ControlCommand:
        pygame.event.pump()  # оновити внутрішній стан SDL (потрібно щокадру)

        roll = self._read_axis("roll")
        pitch = self._read_axis("pitch")
        yaw = self._read_axis("yaw")
        throttle_stick = self._read_axis("throttle")  # [-1, 1], -1 = холостий хід
        throttle = (throttle_stick + 1.0) / 2.0

        arm_button = bool(self._joystick.get_button(self.cfg.buttons.arm))
        if arm_button and not self._prev_arm_button:  # тумблер лише на фронт натискання
            self._armed = not self._armed
        self._prev_arm_button = arm_button

        mode_button = bool(self._joystick.get_button(self.cfg.buttons.mode_switch))
        if mode_button and not self._prev_mode_button:
            self._mode_index = (self._mode_index + 1) % len(FLIGHT_MODE_CYCLE)
        self._prev_mode_button = mode_button

        return ControlCommand(
            roll=roll,
            pitch=pitch,
            yaw=yaw,
            throttle=throttle,
            arm=self._armed,
            mode=FLIGHT_MODE_CYCLE[self._mode_index],
        )

    def close(self) -> None:
        self._joystick.quit()

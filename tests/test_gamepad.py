"""Тести мапінгу геймпада (input/gamepad.py) через ФЕЙКОВИЙ джойстик.

Реального геймпада на цій машині немає (SKILL.md, «Типові пастки») — тому
перевіряємо саме мапінг осей/кнопок config'у на підставленому об'єкті, без
pygame.joystick.init() і без справжнього SDL-пристрою. SDL-відеодрайвер
форсуємо в "dummy", щоб pygame.event.pump() мав ініціалізовану чергу подій
навіть на машині без дисплея (портативно для CI)."""

import os

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")

import pygame  # noqa: E402

from dronesim.core.config import load_config  # noqa: E402
from dronesim.core.contracts import FlightMode  # noqa: E402
from dronesim.input.gamepad import GamepadSource  # noqa: E402

pygame.init()


class _FakeJoystick:
    def __init__(self, axes: dict[int, float], buttons: dict[int, bool] | None = None):
        self._axes = axes
        self._buttons = buttons or {}

    def get_axis(self, index: int) -> float:
        return self._axes.get(index, 0.0)

    def get_button(self, index: int) -> bool:
        return self._buttons.get(index, False)


def _make_source(axes: dict[int, float], buttons: dict[int, bool] | None = None) -> GamepadSource:
    source = GamepadSource.__new__(GamepadSource)  # обходимо __init__ (не чіпаємо pygame/SDL)
    source.cfg = load_config("input/ps_gamepad")
    source._armed = False
    source._prev_arm_button = False
    source._mode_index = 0
    source._prev_mode_button = False
    source._joystick = _FakeJoystick(axes, buttons)
    return source


def test_neutral_sticks_give_zero_roll_pitch_yaw_and_idle_throttle():
    source = _make_source({0: 0.0, 1: 0.0, 2: 0.0, 3: 0.0})
    cmd = source.get_command()
    assert cmd.roll == 0.0
    assert cmd.pitch == 0.0
    assert cmd.yaw == 0.0
    assert abs(cmd.throttle - 0.5) < 1e-6  # стік throttle центр -> 0.5 (не 0)
    assert cmd.mode is FlightMode.ACRO


def test_throttle_stick_full_up_maps_to_full_throttle():
    # axes[1] інвертовано в конфізі (invert: true) -> "вгору" = -1.0 на стіку SDL
    source = _make_source({0: 0.0, 1: -1.0, 2: 0.0, 3: 0.0})
    cmd = source.get_command()
    assert cmd.throttle > 0.95


def test_throttle_stick_full_down_maps_to_zero_throttle():
    source = _make_source({0: 0.0, 1: 1.0, 2: 0.0, 3: 0.0})
    cmd = source.get_command()
    assert cmd.throttle < 0.05


def test_roll_axis_not_inverted_passes_through():
    source = _make_source({0: 0.0, 1: 0.0, 2: 0.8, 3: 0.0})
    cmd = source.get_command()
    assert cmd.roll > 0.0  # invert: false у конфізі -> знак зберігається


def test_arm_button_toggles_on_press_edge_not_while_held():
    source = _make_source({0: 0.0, 1: 0.0, 2: 0.0, 3: 0.0}, buttons={9: True})
    cmd1 = source.get_command()
    assert cmd1.arm is True  # перший виклик з натиснутою кнопкою -> увімкнено

    cmd2 = source.get_command()  # кнопка й далі утримується -> НЕ має перемкнутись назад
    assert cmd2.arm is True

    source._joystick._buttons[9] = False
    source.get_command()  # відпустили
    source._joystick._buttons[9] = True
    cmd3 = source.get_command()  # новий фронт натискання -> перемикається назад
    assert cmd3.arm is False


def test_mode_switch_button_cycles_on_press_edge_not_while_held():
    source = _make_source({0: 0.0, 1: 0.0, 2: 0.0, 3: 0.0}, buttons={8: True})
    cmd1 = source.get_command()
    assert cmd1.mode is FlightMode.ANGLE  # перший фронт -> ACRO -> ANGLE

    cmd2 = source.get_command()  # кнопка й далі утримується -> без змін
    assert cmd2.mode is FlightMode.ANGLE

    source._joystick._buttons[8] = False
    source.get_command()
    source._joystick._buttons[8] = True
    cmd3 = source.get_command()  # новий фронт -> ANGLE -> ALT_HOLD
    assert cmd3.mode is FlightMode.ALT_HOLD


def test_set_start_mode_changes_initial_mode_before_first_command():
    source = _make_source({0: 0.0, 1: 0.0, 2: 0.0, 3: 0.0})
    assert source.get_command().mode is FlightMode.ACRO  # дефолт без виклику

    source.set_start_mode(FlightMode.ANGLE)
    assert source.get_command().mode is FlightMode.ANGLE

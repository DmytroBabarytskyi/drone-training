"""Тести HUD-оверлею (render/hud.py). Конвертація кватерніона тепер у
utils/math3d.py (test_math3d.py) — тут лише сам HUD."""

import numpy as np

from dronesim.core.contracts import ControlCommand, VehicleState
from dronesim.render.hud import HUD, build_keyboard_hint, build_takeoff_hint

_BINDINGS = {
    "pitch_forward": "w",
    "pitch_back": "s",
    "roll_left": "a",
    "roll_right": "d",
    "yaw_left": "q",
    "yaw_right": "e",
    "throttle_up": "shift",
    "throttle_down": "control",
    "arm_toggle": "space",
    "mode_cycle": "tab",
}


def test_hud_update_does_not_raise(engine):
    hud = HUD(engine)
    try:
        state = VehicleState(pos=np.array([1.0, 2.0, 3.0]), t=5.0)
        cmd = ControlCommand(throttle=0.5, arm=True)
        hud.update(state, cmd)
        # Показники розподілені по краях: arm — ліворуч угорі, час — праворуч угорі.
        assert "ARMED" in hud._mode_text.getText()
        assert "5.00" in hud._timer_text.getText()
    finally:
        hud.close()


def test_hud_shows_disarmed_status(engine):
    hud = HUD(engine)
    try:
        state = VehicleState()
        cmd = ControlCommand(arm=False)
        hud.update(state, cmd)
        assert "disarmed" in hud._mode_text.getText()
    finally:
        hud.close()


def test_hud_shows_active_mode_when_given(engine):
    hud = HUD(engine)
    try:
        state = VehicleState()
        cmd = ControlCommand(arm=True)
        hud.update(state, cmd, active_mode="pos_hold")
        assert "pos_hold" in hud._mode_text.getText()
    finally:
        hud.close()


def test_hud_shows_speed_from_velocity(engine):
    hud = HUD(engine)
    try:
        state = VehicleState(vel=np.array([3.0, 4.0, 2.0]))  # горизонтальна=5.0 (3-4-5)
        cmd = ControlCommand(arm=True)
        hud.update(state, cmd)
        assert "5.0 m/s (h)" in hud._speed_text.getText()
        assert "+2.0 m/s (v)" in hud._speed_text.getText()
    finally:
        hud.close()


def test_hud_shows_controls_hint_only_while_disarmed(engine):
    hud = HUD(engine, controls_hint="ПІДКАЗКА")
    try:
        hud.update(VehicleState(), ControlCommand(arm=False))
        assert "ПІДКАЗКА" in hud._hint_text.getText()

        hud.update(VehicleState(), ControlCommand(arm=True))
        assert hud._hint_text.getText() == ""
    finally:
        hud.close()


def test_hud_shows_warning_text_and_hides_when_ok(engine):
    hud = HUD(engine)
    try:
        hud.update(VehicleState(), ControlCommand(arm=True), warning_text="МЕЖА ВИСОТИ", warning_level="critical")
        assert "МЕЖА ВИСОТИ" in hud._warning_text.getText()

        hud.update(VehicleState(), ControlCommand(arm=True))
        assert hud._warning_text.getText() == ""
    finally:
        hud.close()


def test_build_keyboard_hint_uses_bindings_not_hardcoded_keys():
    hint = build_keyboard_hint(_BINDINGS)
    assert "W/S" in hint
    assert "SHIFT/CONTROL" in hint
    assert "SPACE=arm" in hint


def test_build_takeoff_hint_mentions_arm_and_throttle_keys():
    hint = build_takeoff_hint(_BINDINGS)
    assert "SPACE" in hint
    assert "SHIFT" in hint

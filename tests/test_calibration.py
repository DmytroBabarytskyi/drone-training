"""Тести чистих функцій калібрування осей (input/calibration.py)."""

from dronesim.input.calibration import apply_axis_curve, apply_deadzone, apply_expo


def test_deadzone_zeroes_small_values():
    assert apply_deadzone(0.02, 0.05) == 0.0
    assert apply_deadzone(-0.02, 0.05) == 0.0


def test_deadzone_rescales_remaining_range():
    # На межі deadzone -> 0; на краю ходу (1.0) -> 1.0
    assert abs(apply_deadzone(0.05, 0.05)) < 1e-9
    assert abs(apply_deadzone(1.0, 0.05) - 1.0) < 1e-9


def test_expo_zero_is_linear():
    assert apply_expo(0.5, 0.0) == 0.5
    assert apply_expo(-0.3, 0.0) == -0.3


def test_expo_softens_center_for_positive_expo():
    # При expo>0 значення біля центру менші за лінійні (плавніше керування)
    assert apply_expo(0.3, 0.5) < 0.3
    assert apply_expo(1.0, 0.5) == 1.0  # на максимумі expo не змінює краю ходу


def test_axis_curve_invert():
    assert apply_axis_curve(0.5, deadzone=0.0, expo=0.0, invert=True) == -0.5
    assert apply_axis_curve(0.5, deadzone=0.0, expo=0.0, invert=False) == 0.5


def test_axis_curve_clamped_to_unit_range():
    result = apply_axis_curve(2.0, deadzone=0.0, expo=0.0, invert=False)
    assert -1.0 <= result <= 1.0

"""Тести BetaFlight Actual Rates (vehicles/controllers/rates.py, доп. фаза
"справжній Liftoff") — формула звірена з офіційним вихідним кодом BetaFlight
(``src/main/fc/rc.c::applyActualRates``)."""

from __future__ import annotations

import numpy as np

from dronesim.vehicles.controllers.rates import actual_rate_rad_s

_CENTER = 190.0
_MAX = 670.0
_EXPO = 0.15
_DEG2RAD = np.pi / 180.0


def test_zero_stick_gives_zero_rate():
    assert actual_rate_rad_s(0.0, _CENTER, _MAX, _EXPO) == 0.0


def test_full_stick_gives_max_rate():
    rate = actual_rate_rad_s(1.0, _CENTER, _MAX, _EXPO)
    assert abs(rate - _MAX * _DEG2RAD) < 1e-6


def test_full_negative_stick_gives_negative_max_rate():
    rate = actual_rate_rad_s(-1.0, _CENTER, _MAX, _EXPO)
    assert abs(rate + _MAX * _DEG2RAD) < 1e-6


def test_rate_is_symmetric_for_opposite_stick():
    for stick in (0.1, 0.3, 0.6, 0.9):
        pos = actual_rate_rad_s(stick, _CENTER, _MAX, _EXPO)
        neg = actual_rate_rad_s(-stick, _CENTER, _MAX, _EXPO)
        assert abs(pos + neg) < 1e-9


def test_rate_is_monotonically_increasing_with_stick():
    sticks = np.linspace(-1.0, 1.0, 41)
    rates = [actual_rate_rad_s(float(s), _CENTER, _MAX, _EXPO) for s in sticks]
    assert all(b > a for a, b in zip(rates, rates[1:]))


def test_small_stick_rate_dominated_by_center_sensitivity():
    """Біля центру внесок expo-кривої мізерний (stick^5 -> 0 швидше за stick) —
    результат має бути близьким до чисто лінійного center_sensitivity*stick."""
    stick = 0.02
    rate = actual_rate_rad_s(stick, _CENTER, _MAX, _EXPO)
    linear_only = stick * _CENTER * _DEG2RAD
    assert abs(rate - linear_only) < 0.1 * abs(linear_only)


def test_zero_expo_is_pure_linear_interpolation():
    """expo=0 -> крива вироджується в лінійну інтерполяцію center->max."""
    for stick in (0.2, 0.5, 0.8, 1.0):
        rate = actual_rate_rad_s(stick, _CENTER, _MAX, expo=0.0)
        expected_deg_s = stick * _CENTER + max(0.0, _MAX - _CENTER) * stick * abs(stick)
        assert abs(rate - expected_deg_s * _DEG2RAD) < 1e-9


def test_equal_center_and_max_rate_is_pure_linear_regardless_of_expo():
    """Якщо center_sensitivity == max_rate, stick_movement=0 -> завжди лінійно,
    незалежно від expo (немає простору для кривої)."""
    for expo in (0.0, 0.3, 0.8, 1.0):
        rate = actual_rate_rad_s(0.5, 300.0, 300.0, expo)
        assert abs(rate - 0.5 * 300.0 * _DEG2RAD) < 1e-9

"""Тести детермінованого генератора вітру (physics/wind.py)."""

import numpy as np

from dronesim.physics.wind import Wind


def test_zero_wind_gives_zero_force():
    wind = Wind()
    assert np.allclose(wind.force_at(0.0), 0.0)
    assert np.allclose(wind.force_at(100.0), 0.0)


def test_base_force_only_is_constant_over_time():
    wind = Wind(base_force=(2.0, -1.0, 0.0))
    f0 = wind.force_at(0.0)
    f5 = wind.force_at(5.0)
    assert np.allclose(f0, [2.0, -1.0, 0.0])
    assert np.allclose(f5, [2.0, -1.0, 0.0])


def test_gust_is_horizontal_only():
    wind = Wind(gust_amplitude=3.0, seed=1)
    for t in (0.1, 1.0, 2.5, 10.0):
        force = wind.force_at(t)
        assert force[2] == 0.0


def test_gust_bounded_by_amplitude():
    wind = Wind(gust_amplitude=3.0, gust_frequency_hz=0.5, seed=2)
    for t in np.linspace(0, 10, 200):
        force = wind.force_at(t)
        assert np.linalg.norm(force) <= 3.0 + 1e-9


def test_same_seed_is_reproducible():
    wind_a = Wind(gust_amplitude=1.0, seed=42)
    wind_b = Wind(gust_amplitude=1.0, seed=42)
    assert np.allclose(wind_a.force_at(3.0), wind_b.force_at(3.0))


def test_different_seed_gives_different_direction():
    wind_a = Wind(gust_amplitude=1.0, seed=1)
    wind_b = Wind(gust_amplitude=1.0, seed=999)
    assert not np.allclose(wind_a.force_at(3.0), wind_b.force_at(3.0))


def test_turbulence_disabled_by_default_matches_old_behavior():
    """``turbulence_intensity=0`` (дефолт) -> точнісінько стара модель (лише
    стала + гармонічний порив), без жодної додаткової випадковості."""
    wind = Wind(base_force=(2.0, -1.0, 0.0), gust_amplitude=1.0, seed=7)
    for t in (0.0, 1.0, 5.0, 12.3):
        force = wind.force_at(t)
        expected_gust = 1.0 * np.sin(2.0 * np.pi * 0.2 * t + wind._phase)  # noqa: SLF001
        expected = wind.base_force + expected_gust * wind._gust_direction  # noqa: SLF001
        assert np.allclose(force, expected)


def test_turbulence_adds_variability_beyond_base_and_gust():
    wind = Wind(base_force=(1.0, 0.0, 0.0), turbulence_intensity=0.5, seed=3)
    forces = [wind.force_at(t) for t in np.linspace(0.0, 20.0, 50)]
    # Без турбулентності всі значення були б ідентичними (лише стала складова).
    assert not all(np.allclose(f, forces[0]) for f in forces)


def test_turbulence_is_reproducible_by_seed():
    wind_a = Wind(turbulence_intensity=0.5, seed=11)
    wind_b = Wind(turbulence_intensity=0.5, seed=11)
    assert np.allclose(wind_a.force_at(4.2), wind_b.force_at(4.2))


def test_turbulence_different_seed_gives_different_force():
    wind_a = Wind(turbulence_intensity=0.5, seed=11)
    wind_b = Wind(turbulence_intensity=0.5, seed=12)
    assert not np.allclose(wind_a.force_at(4.2), wind_b.force_at(4.2))


def test_torque_is_zero_when_turbulence_disabled():
    wind = Wind(gust_amplitude=3.0, seed=1)  # без turbulence_intensity
    for t in (0.0, 1.0, 5.0, 10.0):
        assert np.allclose(wind.torque_at(t), 0.0)


def test_torque_is_nonzero_when_turbulence_enabled():
    wind = Wind(turbulence_intensity=0.5, seed=5)
    torques = [wind.torque_at(t) for t in np.linspace(0.1, 20.0, 50)]
    assert any(np.linalg.norm(tq) > 1e-9 for tq in torques)


def test_torque_reproducible_by_seed():
    wind_a = Wind(turbulence_intensity=0.5, seed=21)
    wind_b = Wind(turbulence_intensity=0.5, seed=21)
    assert np.allclose(wind_a.torque_at(6.0), wind_b.torque_at(6.0))

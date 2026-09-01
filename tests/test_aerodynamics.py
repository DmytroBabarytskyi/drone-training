"""Тести аеродинаміки мотора (physics/aerodynamics.py) — чисті функції, без Panda3D."""

import numpy as np

from dronesim.physics import aerodynamics


def test_motor_thrust_scales_quadratically():
    max_thrust = 10.0
    assert aerodynamics.motor_thrust(0.0, max_thrust) == 0.0
    assert aerodynamics.motor_thrust(1.0, max_thrust) == max_thrust
    assert abs(aerodynamics.motor_thrust(0.5, max_thrust) - max_thrust * 0.25) < 1e-9


def test_motor_thrust_clamps_out_of_range_cmd():
    assert aerodynamics.motor_thrust(-1.0, 10.0) == 0.0
    assert aerodynamics.motor_thrust(2.0, 10.0) == 10.0


def test_reactive_torque_sign_follows_spin_dir():
    t_cw = aerodynamics.reactive_torque(5.0, 0.02, +1)
    t_ccw = aerodynamics.reactive_torque(5.0, 0.02, -1)
    assert t_cw > 0
    assert t_ccw < 0
    assert abs(t_cw) == abs(t_ccw)


def test_reactive_torque_zero_thrust_gives_zero_torque():
    assert aerodynamics.reactive_torque(0.0, 0.02, +1) == 0.0


def test_linear_drag_opposes_velocity():
    v = np.array([2.0, -1.0, 0.0])
    drag = aerodynamics.linear_drag(v, 0.1)
    assert drag[0] < 0
    assert drag[1] > 0
    assert drag[2] == 0


def test_linear_drag_zero_at_rest():
    drag = aerodynamics.linear_drag(np.zeros(3), 0.1)
    assert (drag == 0).all()


def test_quadratic_drag_opposes_velocity_and_scales_with_speed_squared():
    v_slow = np.array([1.0, 0.0, 0.0])
    v_fast = np.array([2.0, 0.0, 0.0])
    drag_slow = aerodynamics.quadratic_drag(v_slow, 0.5)
    drag_fast = aerodynamics.quadratic_drag(v_fast, 0.5)
    assert drag_slow[0] < 0.0  # опір проти руху
    # Подвоєння швидкості -> учетверо більша сила (∝v²), не удвічі (лінійна модель).
    assert abs(drag_fast[0] / drag_slow[0] - 4.0) < 1e-9


def test_quadratic_drag_anisotropic_with_vector_coeff():
    v = np.array([2.0, 0.0, 2.0])
    drag = aerodynamics.quadratic_drag(v, np.array([0.1, 0.1, 0.5]))
    assert abs(drag[2]) > abs(drag[0])  # більший коефіцієнт по Z -> більша сила


def test_quadratic_drag_zero_at_rest():
    assert (aerodynamics.quadratic_drag(np.zeros(3), 0.3) == 0.0).all()


def test_motor_lag_disabled_when_tau_is_zero():
    current = np.zeros(4)
    commanded = np.full(4, 5.0)
    result = aerodynamics.motor_lag(current, commanded, tau_s=0.0, dt=1 / 240)
    assert np.allclose(result, commanded)  # tau<=0 -> миттєвий відгук (стара поведінка)


def test_motor_lag_approaches_target_gradually_not_instantly():
    current = np.zeros(4)
    commanded = np.full(4, 10.0)
    after_one_step = aerodynamics.motor_lag(current, commanded, tau_s=0.05, dt=1 / 240)
    assert 0.0 < after_one_step[0] < 10.0  # не миттєво на цілі, і не нуль


def test_motor_lag_converges_to_target_over_many_steps():
    current = np.zeros(4)
    commanded = np.full(4, 10.0)
    for _ in range(int(240 * 1.0)):  # 1с реального часу, tau=0.05с -> має практично досягти цілі
        current = aerodynamics.motor_lag(current, commanded, tau_s=0.05, dt=1 / 240)
    assert np.allclose(current, commanded, atol=0.01)


def test_motor_lag_stable_even_when_dt_larger_than_tau():
    # dt >> tau: дискретний фільтр alpha=dt/(tau+dt) має лишатись у [0,1] -> без
    # перерегулювання/розбіжності (на відміну від наївного exp(-dt/tau) за великих dt).
    current = np.array([0.0])
    commanded = np.array([10.0])
    result = aerodynamics.motor_lag(current, commanded, tau_s=0.01, dt=1.0)
    assert 0.0 <= result[0] <= 10.0


def test_battery_thrust_factor_full_charge_no_load_is_near_one():
    assert abs(aerodynamics.battery_thrust_factor(soc=1.0, load_frac=0.0, sag_coeff=0.1) - 1.0) < 1e-9


def test_battery_thrust_factor_drops_with_low_soc():
    high = aerodynamics.battery_thrust_factor(soc=1.0, load_frac=0.0, sag_coeff=0.0)
    low = aerodynamics.battery_thrust_factor(soc=0.0, load_frac=0.0, sag_coeff=0.0)
    assert low < high
    assert low > 0.0  # ніколи не миттєвий обрив до нуля (спрощення, докстрінг функції)


def test_battery_thrust_factor_sags_under_load():
    no_load = aerodynamics.battery_thrust_factor(soc=1.0, load_frac=0.0, sag_coeff=0.2)
    full_load = aerodynamics.battery_thrust_factor(soc=1.0, load_frac=1.0, sag_coeff=0.2)
    assert full_load < no_load


def test_ground_effect_factor_disabled_when_radius_zero():
    assert aerodynamics.ground_effect_factor(height_m=0.5, rotor_radius_m=0.0) == 1.0


def test_ground_effect_factor_boosts_near_ground():
    near = aerodynamics.ground_effect_factor(height_m=0.05, rotor_radius_m=0.0635)
    far = aerodynamics.ground_effect_factor(height_m=5.0, rotor_radius_m=0.0635)
    assert near > far
    assert abs(far - 1.0) < 0.01  # далеко від землі -> практично без ефекту


def test_ground_effect_factor_capped_at_max_boost():
    factor = aerodynamics.ground_effect_factor(height_m=0.001, rotor_radius_m=0.0635, max_boost=0.25)
    assert factor <= 1.25 + 1e-9

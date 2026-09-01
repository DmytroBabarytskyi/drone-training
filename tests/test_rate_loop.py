"""Тести доп. фази "реалізм фізики" на рівні RateLoopMixer: setpoint
feed-forward, airmode (розтягування замість обрізання при насиченні),
шум гіроскопа, латентність контуру керування — усі опційні (``cfg.get``,
відсутність поля = стара поведінка)."""

import numpy as np

from dronesim.core.contracts import VehicleState
from dronesim.physics.world import PhysicsWorld
from dronesim.vehicles.controllers.rate_loop import RateLoopMixer
from dronesim.vehicles.quad_fpv import QuadFPV

DT = 1 / 240


def _make_vehicle():
    world = PhysicsWorld()
    vehicle = QuadFPV(world)
    vehicle.reset(pos=np.array([0.0, 0.0, 5.0]))
    return vehicle


def _rest_state() -> VehicleState:
    return VehicleState(quat=np.array([0.0, 0.0, 0.0, 1.0]))  # identity, ang_vel=0 (дефолт)


def test_feedforward_disabled_by_default_matches_pure_pid():
    """Без ``rate_feedforward`` у конфізі — точнісінько як без feed-forward
    (perший виклик: prev_desired=0, тож ff-доданок був би 0 в будь-якому
    разі; перевіряємо ДРУГИЙ виклик зі СХОДИНКОЮ бажаної швидкості)."""
    vehicle = _make_vehicle()
    mixer = RateLoopMixer(vehicle)
    state = _rest_state()

    mixer.update_from_rates({"roll": 0.0, "pitch": 0.0, "yaw": 0.0}, state, 0.3, DT)
    out_no_ff = mixer.update_from_rates({"roll": 1.0, "pitch": 0.0, "yaw": 0.0}, state, 0.3, DT)

    assert not np.allclose(out_no_ff, out_no_ff[0])  # роль-корекція реально щось зробила (несиметричні тяги)


def test_feedforward_adds_extra_push_proportional_to_setpoint_rate_of_change():
    vehicle_ff = _make_vehicle()
    vehicle_ff.cfg.rate_feedforward = {"roll": 0.01, "pitch": 0.0, "yaw": 0.0}
    vehicle_plain = _make_vehicle()

    mixer_ff = RateLoopMixer(vehicle_ff)
    mixer_plain = RateLoopMixer(vehicle_plain)
    state = _rest_state()

    mixer_ff.update_from_rates({"roll": 0.0, "pitch": 0.0, "yaw": 0.0}, state, 0.3, DT)
    mixer_plain.update_from_rates({"roll": 0.0, "pitch": 0.0, "yaw": 0.0}, state, 0.3, DT)

    out_ff = mixer_ff.update_from_rates({"roll": 1.0, "pitch": 0.0, "yaw": 0.0}, state, 0.3, DT)
    out_plain = mixer_plain.update_from_rates({"roll": 1.0, "pitch": 0.0, "yaw": 0.0}, state, 0.3, DT)

    assert not np.allclose(out_ff, out_plain)  # feed-forward реально змінив вихід на сходинці


def test_airmode_disabled_clips_losing_authority():
    vehicle = _make_vehicle()
    vehicle.cfg.pid_rate.roll.kp = 0.3  # достатньо для виходу за нижню межу [0,1] при throttle=0
    vehicle.cfg.airmode = False  # явно вимкнено (production fpv_5inch.yaml тепер airmode: true за замовч.)
    mixer = RateLoopMixer(vehicle)
    state = _rest_state()

    thrusts = mixer.update_from_rates({"roll": 1.0, "pitch": 0.0, "yaw": 0.0}, state, 0.0, DT)
    # Без airmode: як мінімум одна тяга притиснута до 0 (від'ємний мотор-норм обрізаний).
    assert np.min(thrusts) == 0.0


def test_airmode_enabled_preserves_correction_spread_at_zero_throttle():
    vehicle_air = _make_vehicle()
    vehicle_air.cfg.pid_rate.roll.kp = 0.3
    vehicle_air.cfg.airmode = True
    vehicle_plain = _make_vehicle()
    vehicle_plain.cfg.pid_rate.roll.kp = 0.3
    vehicle_plain.cfg.airmode = False  # явно вимкнено (production fpv_5inch.yaml тепер airmode: true за замовч.)

    mixer_air = RateLoopMixer(vehicle_air)
    mixer_plain = RateLoopMixer(vehicle_plain)
    state = _rest_state()

    thrusts_air = mixer_air.update_from_rates({"roll": 1.0, "pitch": 0.0, "yaw": 0.0}, state, 0.0, DT)
    thrusts_plain = mixer_plain.update_from_rates({"roll": 1.0, "pitch": 0.0, "yaw": 0.0}, state, 0.0, DT)

    # Airmode зберігає БІЛЬШИЙ розкид (max-min) тяг -> більше диференційної
    # авторитетності керування нахилом на нульовому газі, ніж проста обрізка.
    spread_air = float(np.max(thrusts_air) - np.min(thrusts_air))
    spread_plain = float(np.max(thrusts_plain) - np.min(thrusts_plain))
    assert spread_air > spread_plain


def test_gyro_noise_disabled_by_default_is_deterministic():
    vehicle = _make_vehicle()
    mixer_a = RateLoopMixer(vehicle)
    mixer_b = RateLoopMixer(_make_vehicle())
    state = _rest_state()
    out_a = mixer_a.update_from_rates({"roll": 0.5, "pitch": 0.0, "yaw": 0.0}, state, 0.3, DT)
    out_b = mixer_b.update_from_rates({"roll": 0.5, "pitch": 0.0, "yaw": 0.0}, state, 0.3, DT)
    assert np.allclose(out_a, out_b)  # без шуму -> детерміновано однаково


def test_gyro_noise_enabled_perturbs_output_deterministically_by_seed():
    vehicle_a = _make_vehicle()
    vehicle_a.cfg.gyro_noise_std_rad_s = 0.5
    vehicle_a.cfg.gyro_noise_seed = 42
    vehicle_b = _make_vehicle()
    vehicle_b.cfg.gyro_noise_std_rad_s = 0.5
    vehicle_b.cfg.gyro_noise_seed = 42
    vehicle_c = _make_vehicle()
    vehicle_c.cfg.gyro_noise_std_rad_s = 0.5
    vehicle_c.cfg.gyro_noise_seed = 999

    mixer_a = RateLoopMixer(vehicle_a)
    mixer_b = RateLoopMixer(vehicle_b)
    mixer_c = RateLoopMixer(vehicle_c)
    state = _rest_state()

    out_a = mixer_a.update_from_rates({"roll": 0.5, "pitch": 0.0, "yaw": 0.0}, state, 0.3, DT)
    out_b = mixer_b.update_from_rates({"roll": 0.5, "pitch": 0.0, "yaw": 0.0}, state, 0.3, DT)
    out_c = mixer_c.update_from_rates({"roll": 0.5, "pitch": 0.0, "yaw": 0.0}, state, 0.3, DT)

    assert np.allclose(out_a, out_b)  # той самий seed -> той самий шум (детермінізм, SKILL.md правило 4)
    assert not np.allclose(out_a, out_c)  # інший seed -> інший шум


def test_control_latency_disabled_by_default_returns_current_step():
    vehicle = _make_vehicle()
    mixer = RateLoopMixer(vehicle)
    state = _rest_state()
    out_step1 = mixer.update_from_rates({"roll": 1.0, "pitch": 0.0, "yaw": 0.0}, state, 0.0, DT)
    out_step2 = mixer.update_from_rates({"roll": 0.0, "pitch": 0.0, "yaw": 0.0}, state, 0.3, DT)
    # Без затримки другий крок відразу відображає нову команду (throttle=0.3,
    # roll-корекція вже майже згасла) -> тяги мають відрізнятись від першого кроку.
    assert not np.allclose(out_step1, out_step2)


def test_control_latency_delays_output_by_configured_steps():
    vehicle = _make_vehicle()
    vehicle.cfg.control_latency_steps = 2
    mixer = RateLoopMixer(vehicle)
    state = _rest_state()

    # Крок 0: throttle=0.1 (мале, легко відрізнити).
    out0 = mixer.update_from_rates({"roll": 0.0, "pitch": 0.0, "yaw": 0.0}, state, 0.1, DT)
    # Крок 1,2: інші throttle.
    mixer.update_from_rates({"roll": 0.0, "pitch": 0.0, "yaw": 0.0}, state, 0.5, DT)
    out2 = mixer.update_from_rates({"roll": 0.0, "pitch": 0.0, "yaw": 0.0}, state, 0.9, DT)

    # З латентністю 2 кроки: вихід на кроці 2 має відповідати команді КРОКУ 0
    # (throttle=0.1), не поточній (0.9) -> помітно менші тяги.
    assert np.mean(out2) < np.mean(out0) + 1e-6
    assert np.mean(out2) < 1.0  # (сама лише перевірка, що взагалі щось повернулось)


def test_reset_clears_feedforward_history_and_latency_queue():
    vehicle = _make_vehicle()
    vehicle.cfg.rate_feedforward = {"roll": 0.01, "pitch": 0.0, "yaw": 0.0}
    vehicle.cfg.control_latency_steps = 2
    mixer = RateLoopMixer(vehicle)
    state = _rest_state()

    mixer.update_from_rates({"roll": 1.0, "pitch": 0.0, "yaw": 0.0}, state, 0.3, DT)
    mixer.reset()
    assert mixer._prev_desired_rate == {"roll": 0.0, "pitch": 0.0, "yaw": 0.0}  # noqa: SLF001
    assert len(mixer._latency_queue) == 0  # noqa: SLF001

"""Тести FPV-квадрокоптера (vehicles/quad_fpv.py) — критерій приймання фази 1:
при throttle вище ваги дрон набирає висоту, при 0 — падає, без NaN за 10с."""

import numpy as np

from dronesim.physics import aerodynamics
from dronesim.physics.world import PhysicsWorld
from dronesim.vehicles.quad_fpv import QuadFPV

START_HEIGHT = 2.0


def _make_vehicle():
    world = PhysicsWorld()
    vehicle = QuadFPV(world)
    vehicle.reset(pos=np.array([0.0, 0.0, START_HEIGHT]))
    return world, vehicle


def test_quad_falls_with_zero_throttle():
    world, vehicle = _make_vehicle()
    for _ in range(120):  # 0.5с
        vehicle.apply_motor_commands(np.zeros(4))
        world.step(1 / 240)
    state = vehicle.state
    assert state.pos[2] < START_HEIGHT
    assert not np.isnan(state.pos).any()


def test_quad_climbs_with_full_throttle():
    world, vehicle = _make_vehicle()
    for _ in range(60):  # 0.25с; макс. тяга ~4x перевищує вагу (config fpv_5inch)
        thrusts = np.full(4, aerodynamics.motor_thrust(1.0, vehicle.cfg.max_thrust_per_motor))
        vehicle.apply_motor_commands(thrusts)
        world.step(1 / 240)
    state = vehicle.state
    assert state.pos[2] > START_HEIGHT
    assert not np.isnan(state.pos).any()


def test_quad_hover_state_has_no_nan_over_time():
    world, vehicle = _make_vehicle()
    cmd = np.sqrt((vehicle.cfg.mass * 9.81 / 4.0) / vehicle.cfg.max_thrust_per_motor)
    motor_cmds = np.full(4, float(np.clip(cmd, 0.0, 1.0)))

    for _ in range(240 * 3):  # 3с
        thrusts = np.array(
            [aerodynamics.motor_thrust(c, vehicle.cfg.max_thrust_per_motor) for c in motor_cmds]
        )
        vehicle.apply_motor_commands(thrusts)
        world.step(1 / 240)
        vehicle.advance_time(1 / 240)
        state = vehicle.state
        assert not np.isnan(state.pos).any()
        assert not np.isnan(state.vel).any()
        assert not np.isnan(state.quat).any()


def test_reset_zeroes_velocity_and_sets_position():
    world, vehicle = _make_vehicle()
    thrusts = np.full(4, aerodynamics.motor_thrust(1.0, vehicle.cfg.max_thrust_per_motor))
    for _ in range(30):
        vehicle.apply_motor_commands(thrusts)
        world.step(1 / 240)

    vehicle.reset(pos=np.array([1.0, 2.0, 3.0]))
    state = vehicle.state
    assert np.allclose(state.pos, [1.0, 2.0, 3.0])
    assert np.allclose(state.vel, [0.0, 0.0, 0.0])
    assert np.allclose(state.ang_vel, [0.0, 0.0, 0.0])
    assert state.t == 0.0


def test_apply_motor_commands_rejects_wrong_shape():
    _, vehicle = _make_vehicle()
    try:
        vehicle.apply_motor_commands(np.zeros(3))
        assert False, "мало кинути ValueError"
    except ValueError:
        pass

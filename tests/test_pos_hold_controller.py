"""Тести Pos-hold контролера (vehicles/controllers/position.py) — критерій
приймання фази 3: повертається в точку після поштовху вітром (physics/wind.py)."""

import numpy as np

from dronesim.core.contracts import ControlCommand
from dronesim.physics.wind import Wind
from dronesim.physics.world import PhysicsWorld
from dronesim.vehicles.controllers.position import PosHoldController
from dronesim.vehicles.quad_large import QuadLarge

START_HEIGHT = 5.0


def _make():
    world = PhysicsWorld()
    vehicle = QuadLarge(world)
    vehicle.reset(pos=np.array([0.0, 0.0, START_HEIGHT]))
    controller = PosHoldController(vehicle)
    return world, vehicle, controller


def _neutral_cmd() -> ControlCommand:
    return ControlCommand(roll=0.0, pitch=0.0, yaw=0.0, throttle=0.5, arm=True)


def _fly(world, vehicle, controller, cmd, steps, dt=1 / 240, wind=None, t0=0.0):
    t = t0
    for _ in range(steps):
        thrusts = controller.update(cmd, vehicle.state, dt)
        if wind is not None:
            vehicle.apply_external_force(wind.force_at(t))
        vehicle.apply_motor_commands(thrusts)
        world.step(dt)
        t += dt
    return t


def test_holds_position_with_neutral_stick_no_wind():
    world, vehicle, controller = _make()
    cmd = _neutral_cmd()
    _fly(world, vehicle, controller, cmd, 240 * 5)  # 5с розгону/стабілізації
    state = vehicle.state
    assert np.linalg.norm(state.pos[:2]) < 0.5
    assert not np.isnan(state.pos).any()


def test_manual_pitch_stick_moves_in_positive_x_direction():
    world, vehicle, controller = _make()
    cmd = ControlCommand(roll=0.0, pitch=0.3, yaw=0.0, throttle=0.5, arm=True)
    _fly(world, vehicle, controller, cmd, 240 * 2)  # 2с
    state = vehicle.state
    assert state.pos[0] > 0.2  # летить вперед (world +X), не назад
    assert not np.isnan(state.pos).any()


def test_manual_roll_stick_moves_in_positive_y_direction():
    world, vehicle, controller = _make()
    cmd = ControlCommand(roll=0.3, pitch=0.0, yaw=0.0, throttle=0.5, arm=True)
    _fly(world, vehicle, controller, cmd, 240 * 2)
    state = vehicle.state
    assert state.pos[1] > 0.2
    assert not np.isnan(state.pos).any()


def test_returns_to_point_after_temporary_wind_push():
    world, vehicle, controller = _make()
    cmd = _neutral_cmd()
    dt = 1 / 240

    # 1) стабілізуватись у вихідній точці
    t = _fly(world, vehicle, controller, cmd, 240 * 3, dt=dt)
    target_before = vehicle.state.pos[:2].copy()

    # 2) поштовх вітром на кілька секунд (стала горизонтальна сила)
    wind = Wind(base_force=(15.0, 0.0, 0.0))
    t = _fly(world, vehicle, controller, cmd, 240 * 3, dt=dt, wind=wind, t0=t)
    pushed_pos = vehicle.state.pos[:2].copy()
    assert np.linalg.norm(pushed_pos - target_before) > 0.5  # дійсно зсунуло

    # 3) вітер зникає -> має повернутися близько до вихідної точки
    _fly(world, vehicle, controller, cmd, 240 * 6, dt=dt, t0=t)  # 6с на повернення
    final_pos = vehicle.state.pos[:2].copy()

    assert np.linalg.norm(final_pos - target_before) < 0.5
    assert not np.isnan(vehicle.state.pos).any()


def test_disarmed_falls_regardless_of_position_hold():
    world, vehicle, controller = _make()
    cmd = ControlCommand(roll=0.0, pitch=0.0, yaw=0.0, throttle=0.5, arm=False)
    _fly(world, vehicle, controller, cmd, 120)
    assert vehicle.state.pos[2] < START_HEIGHT

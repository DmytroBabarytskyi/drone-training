"""Тести Acro rate-loop контролера (vehicles/controllers/acro.py).

Перевіряють головний ігровий контур фази 2 headless: дизармований апарат падає
незалежно від стіків; на завису утримується висота; стік крену/рискання
викликає обертання в очікуваному напрямку без розбіжності (NaN/розгону)."""

import numpy as np

from dronesim.core.contracts import ControlCommand
from dronesim.physics.world import PhysicsWorld
from dronesim.vehicles.controllers.acro import AcroController
from dronesim.vehicles.quad_fpv import QuadFPV

START_HEIGHT = 2.0


def _make():
    world = PhysicsWorld()
    vehicle = QuadFPV(world)
    vehicle.reset(pos=np.array([0.0, 0.0, START_HEIGHT]))
    controller = AcroController(vehicle)
    return world, vehicle, controller


def _hover_throttle(vehicle: QuadFPV) -> float:
    weight = vehicle.cfg.mass * 9.81
    thrust_per_motor = weight / 4.0
    return float(np.clip(np.sqrt(thrust_per_motor / vehicle.cfg.max_thrust_per_motor), 0.0, 1.0))


def _run(world, vehicle, controller, cmd, steps, dt=1 / 240):
    for _ in range(steps):
        thrusts = controller.update(cmd, vehicle.state, dt)
        vehicle.apply_motor_commands(thrusts)
        world.step(dt)
    return vehicle.state


def test_disarmed_falls_regardless_of_stick():
    world, vehicle, controller = _make()
    cmd = ControlCommand(roll=0.5, pitch=0.3, yaw=-0.2, throttle=1.0, arm=False)
    state = _run(world, vehicle, controller, cmd, 120)  # 0.5с
    assert state.pos[2] < START_HEIGHT


def test_zero_stick_hover_holds_altitude_roughly():
    world, vehicle, controller = _make()
    cmd = ControlCommand(roll=0.0, pitch=0.0, yaw=0.0, throttle=_hover_throttle(vehicle), arm=True)

    max_drift = 0.0
    for _ in range(240 * 5):  # 5с
        thrusts = controller.update(cmd, vehicle.state, 1 / 240)
        vehicle.apply_motor_commands(thrusts)
        world.step(1 / 240)
        max_drift = max(max_drift, abs(vehicle.state.pos[2] - START_HEIGHT))

    state = vehicle.state
    assert not np.isnan(state.pos).any()
    assert not np.isnan(state.ang_vel).any()
    assert max_drift < 1.0


def test_positive_roll_stick_rolls_in_commanded_direction():
    world, vehicle, controller = _make()
    cmd = ControlCommand(roll=0.6, pitch=0.0, yaw=0.0, throttle=_hover_throttle(vehicle), arm=True)
    state = _run(world, vehicle, controller, cmd, 120)  # 0.5с
    assert not np.isnan(state.pos).any()
    assert state.ang_vel[0] > 0.3  # обертання навколо X у бік команди, не навпаки


def test_negative_roll_stick_rolls_opposite_direction():
    world, vehicle, controller = _make()
    cmd = ControlCommand(roll=-0.6, pitch=0.0, yaw=0.0, throttle=_hover_throttle(vehicle), arm=True)
    state = _run(world, vehicle, controller, cmd, 120)
    assert not np.isnan(state.pos).any()
    assert state.ang_vel[0] < -0.3


def test_positive_yaw_stick_yaws_in_commanded_direction():
    world, vehicle, controller = _make()
    cmd = ControlCommand(roll=0.0, pitch=0.0, yaw=0.6, throttle=_hover_throttle(vehicle), arm=True)
    state = _run(world, vehicle, controller, cmd, 120)
    assert not np.isnan(state.pos).any()
    assert state.ang_vel[2] > 0.2


def test_positive_pitch_stick_pitches_in_commanded_direction():
    world, vehicle, controller = _make()
    cmd = ControlCommand(roll=0.0, pitch=0.6, yaw=0.0, throttle=_hover_throttle(vehicle), arm=True)
    state = _run(world, vehicle, controller, cmd, 120)
    assert not np.isnan(state.pos).any()
    assert state.ang_vel[1] > 0.3

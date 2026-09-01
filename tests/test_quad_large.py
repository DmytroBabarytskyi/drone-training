"""Тести базової фізики великого мультиротора (vehicles/quad_large.py) —
той самий контракт, що й QuadFPV (фаза 1), інша фізична модель через конфіг."""

import numpy as np

from dronesim.physics import aerodynamics
from dronesim.physics.world import PhysicsWorld
from dronesim.vehicles.quad_large import QuadLarge

START_HEIGHT = 2.0


def _make_vehicle():
    world = PhysicsWorld()
    vehicle = QuadLarge(world)
    vehicle.reset(pos=np.array([0.0, 0.0, START_HEIGHT]))
    return world, vehicle


def test_quad_large_falls_with_zero_throttle():
    world, vehicle = _make_vehicle()
    for _ in range(120):
        vehicle.apply_motor_commands(np.zeros(4))
        world.step(1 / 240)
    assert vehicle.state.pos[2] < START_HEIGHT
    assert not np.isnan(vehicle.state.pos).any()


def test_quad_large_climbs_with_full_throttle():
    world, vehicle = _make_vehicle()
    for _ in range(60):
        thrusts = np.full(4, aerodynamics.motor_thrust(1.0, vehicle.cfg.max_thrust_per_motor))
        vehicle.apply_motor_commands(thrusts)
        world.step(1 / 240)
    assert vehicle.state.pos[2] > START_HEIGHT
    assert not np.isnan(vehicle.state.pos).any()


def test_quad_large_is_much_heavier_than_fpv():
    _, vehicle = _make_vehicle()
    assert vehicle.cfg.mass > 4.0  # набагато важчий за fpv_5inch (0.55 кг)


def test_quad_large_hover_command_reasonable():
    _, vehicle = _make_vehicle()
    weight = vehicle.cfg.mass * 9.81
    hover_cmd = np.sqrt((weight / 4.0) / vehicle.cfg.max_thrust_per_motor)
    assert 0.0 < hover_cmd < 1.0

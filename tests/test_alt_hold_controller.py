"""Тести Alt-hold контролера (vehicles/controllers/position.py) — критерій
приймання фази 3: тримає висоту ±0.3 м без входу по throttle."""

import numpy as np

from dronesim.core.contracts import ControlCommand
from dronesim.physics.world import PhysicsWorld
from dronesim.vehicles.controllers.position import AltHoldController
from dronesim.vehicles.quad_large import QuadLarge

START_HEIGHT = 5.0


def _make():
    world = PhysicsWorld()
    vehicle = QuadLarge(world)
    vehicle.reset(pos=np.array([0.0, 0.0, START_HEIGHT]))
    controller = AltHoldController(vehicle)
    return world, vehicle, controller


def _neutral_cmd(throttle: float = 0.5) -> ControlCommand:
    return ControlCommand(roll=0.0, pitch=0.0, yaw=0.0, throttle=throttle, arm=True)


def test_holds_altitude_within_tolerance_with_neutral_throttle():
    world, vehicle, controller = _make()
    cmd = _neutral_cmd()

    max_drift = 0.0
    dt = 1 / 240
    for _ in range(240 * 8):  # 8с
        thrusts = controller.update(cmd, vehicle.state, dt)
        vehicle.apply_motor_commands(thrusts)
        world.step(dt)
        max_drift = max(max_drift, abs(vehicle.state.pos[2] - START_HEIGHT))

    assert max_drift < 0.3
    assert not np.isnan(vehicle.state.pos).any()


def test_disarmed_falls():
    world, vehicle, controller = _make()
    cmd = ControlCommand(roll=0.0, pitch=0.0, yaw=0.0, throttle=0.5, arm=False)
    dt = 1 / 240
    for _ in range(120):
        thrusts = controller.update(cmd, vehicle.state, dt)
        vehicle.apply_motor_commands(thrusts)
        world.step(dt)
    assert vehicle.state.pos[2] < START_HEIGHT


def test_throttle_neutral_captured_lazily_works_for_keyboard_idle_zero():
    # Клавіатура ідле-тримає throttle=0.0 (фаза 2), не 0.5 — перевіряємо, що
    # AltHold все одно тримає висоту, бо нейтраль ловиться лениво з ПЕРШОЇ команди.
    world, vehicle, controller = _make()
    cmd = _neutral_cmd(throttle=0.0)

    max_drift = 0.0
    dt = 1 / 240
    for _ in range(240 * 5):
        thrusts = controller.update(cmd, vehicle.state, dt)
        vehicle.apply_motor_commands(thrusts)
        world.step(dt)
        max_drift = max(max_drift, abs(vehicle.state.pos[2] - START_HEIGHT))

    assert max_drift < 0.3


def test_climb_stick_increases_altitude():
    world, vehicle, controller = _make()
    dt = 1 / 240
    neutral = _neutral_cmd(throttle=0.5)
    for _ in range(60):  # захопити нейтраль на 0.5
        thrusts = controller.update(neutral, vehicle.state, dt)
        vehicle.apply_motor_commands(thrusts)
        world.step(dt)

    climb_cmd = _neutral_cmd(throttle=1.0)
    for _ in range(240 * 3):
        thrusts = controller.update(climb_cmd, vehicle.state, dt)
        vehicle.apply_motor_commands(thrusts)
        world.step(dt)

    assert vehicle.state.pos[2] > START_HEIGHT + 0.5


def test_explicit_throttle_neutral_holds_altitude_from_first_call():
    """Фаза 6: reset(throttle_neutral=X) не чекає першого виклику update() —
    важливо для RL, де перша дія випадкова (docs/DECISIONS.md, фаза 6)."""
    world, vehicle, controller = _make()
    controller.reset(throttle_neutral=0.5)

    max_drift = 0.0
    dt = 1 / 240
    # Перший виклик одразу з РІЗКО ІНШИМ throttle, ніж 0.5 (симулює випадкову
    # першу дію RL) — з лінивою фіксацією це стало б новою (хибною) нейтраллю.
    cmd = _neutral_cmd(throttle=0.9)
    for i in range(240 * 3):
        if i > 30:
            cmd = _neutral_cmd(throttle=0.5)  # далі тримаємо стік по центру
        thrusts = controller.update(cmd, vehicle.state, dt)
        vehicle.apply_motor_commands(thrusts)
        world.step(dt)
        if i > 30:
            max_drift = max(max_drift, abs(vehicle.state.pos[2] - START_HEIGHT))

    assert max_drift < 0.3  # тримає ВИХІДНУ висоту, а не ту, куди занесло за перші 30 кроків

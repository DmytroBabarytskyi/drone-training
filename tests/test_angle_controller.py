"""Тести Angle/Self-level контролера (vehicles/controllers/angle.py).

Перевіряють каскад стік->кут->rate->мотори на великому мультироторі: апарат
вирівнюється при нейтральному стіку, тримає командований кут (не розбігається),
і напрямок нахилу відповідає знаку команди."""

import numpy as np

from dronesim.core.contracts import ControlCommand
from dronesim.physics.world import PhysicsWorld
from dronesim.utils.math3d import quat_to_euler_rad
from dronesim.vehicles.controllers.angle import AngleController
from dronesim.vehicles.quad_fpv import QuadFPV
from dronesim.vehicles.quad_large import QuadLarge

START_HEIGHT = 5.0


def _make():
    world = PhysicsWorld()
    vehicle = QuadLarge(world)
    vehicle.reset(pos=np.array([0.0, 0.0, START_HEIGHT]))
    controller = AngleController(vehicle)
    return world, vehicle, controller


def _hover_throttle(vehicle) -> float:
    weight = vehicle.cfg.mass * 9.81
    return float(np.clip(np.sqrt((weight / 4.0) / vehicle.cfg.max_thrust_per_motor), 0.0, 1.0))


def _run(world, vehicle, controller, cmd, steps, dt=1 / 240):
    for _ in range(steps):
        thrusts = controller.update(cmd, vehicle.state, dt)
        vehicle.apply_motor_commands(thrusts)
        world.step(dt)
    return vehicle.state


def test_zero_stick_converges_to_level_and_holds_altitude_ish():
    world, vehicle, controller = _make()
    cmd = ControlCommand(roll=0.0, pitch=0.0, yaw=0.0, throttle=_hover_throttle(vehicle), arm=True)
    state = _run(world, vehicle, controller, cmd, 240 * 4)  # 4с
    roll, pitch, _ = quat_to_euler_rad(state.quat)
    assert abs(roll) < 0.05
    assert abs(pitch) < 0.05
    assert not np.isnan(state.pos).any()


def test_positive_roll_stick_converges_to_positive_roll_angle():
    world, vehicle, controller = _make()
    cmd = ControlCommand(roll=0.5, pitch=0.0, yaw=0.0, throttle=_hover_throttle(vehicle), arm=True)
    state = _run(world, vehicle, controller, cmd, 240 * 3)  # 3с — час встановитись
    roll, pitch, _ = quat_to_euler_rad(state.quat)
    expected = 0.5 * vehicle.cfg.max_angle_deg * np.pi / 180.0
    assert roll > 0.0
    assert abs(roll - expected) < 0.15  # встановився близько до цільового кута, не розбігся
    assert not np.isnan(state.pos).any()


def test_negative_pitch_stick_converges_to_negative_pitch_angle():
    world, vehicle, controller = _make()
    cmd = ControlCommand(roll=0.0, pitch=-0.5, yaw=0.0, throttle=_hover_throttle(vehicle), arm=True)
    state = _run(world, vehicle, controller, cmd, 240 * 3)
    _, pitch, _ = quat_to_euler_rad(state.quat)
    assert pitch < 0.0
    assert not np.isnan(state.pos).any()


def test_releasing_stick_returns_to_level():
    world, vehicle, controller = _make()
    cmd_tilted = ControlCommand(roll=0.5, pitch=0.0, yaw=0.0, throttle=_hover_throttle(vehicle), arm=True)
    _run(world, vehicle, controller, cmd_tilted, 240 * 2)

    cmd_level = ControlCommand(roll=0.0, pitch=0.0, yaw=0.0, throttle=_hover_throttle(vehicle), arm=True)
    state = _run(world, vehicle, controller, cmd_level, 240 * 3)
    roll, pitch, _ = quat_to_euler_rad(state.quat)
    assert abs(roll) < 0.05
    assert abs(pitch) < 0.05


def _make_fpv():
    world = PhysicsWorld()
    vehicle = QuadFPV(world)
    vehicle.reset(pos=np.array([0.0, 0.0, START_HEIGHT]))
    controller = AngleController(vehicle)
    return world, vehicle, controller


def test_fpv_angle_controller_works_with_rate_profile_not_max_rates():
    """Регресійний тест: fpv_5inch (доп. фаза "справжній Liftoff") має
    ``rate_profile``, НЕ ``max_rates`` — AngleController.__init__ читав
    ``cfg.max_rates.yaw`` напряму й падав з ConfigAttributeError, доки не
    додано fallback на ``rate_profile.yaw.max_rate_deg_s``."""
    world, vehicle, controller = _make_fpv()
    assert controller.max_yaw_rate_rad > 0.0


def test_fpv_angle_mode_converges_without_large_overshoot():
    """fpv_5inch — набагато легший/маневреніший апарат за quad_large; гейни
    підібрані ОКРЕМО (не перенесені) — критерій: перерегулювання відносно
    невелике (не десятки градусів понад ціль, як було з першою спробою
    гейнів kp=6.0/kd=0.05 — до +19deg над max_angle_deg=40)."""
    world, vehicle, controller = _make_fpv()
    cmd = ControlCommand(roll=1.0, pitch=0.0, yaw=0.0, throttle=_hover_throttle(vehicle), arm=True)
    max_roll_deg = 0.0
    dt = 1 / 240
    for _ in range(240):
        thrusts = controller.update(cmd, vehicle.state, dt)
        vehicle.apply_motor_commands(thrusts)
        world.step(dt)
        roll_deg = quat_to_euler_rad(vehicle.state.quat)[0] * 180.0 / np.pi
        max_roll_deg = max(max_roll_deg, roll_deg)
    assert max_roll_deg < vehicle.cfg.max_angle_deg + 5.0


def test_fpv_angle_mode_releasing_stick_settles_without_large_oscillation():
    world, vehicle, controller = _make_fpv()
    cmd_tilted = ControlCommand(roll=1.0, pitch=0.0, yaw=0.0, throttle=_hover_throttle(vehicle), arm=True)
    _run(world, vehicle, controller, cmd_tilted, 240)  # 1с тілту

    cmd_level = ControlCommand(roll=0.0, pitch=0.0, yaw=0.0, throttle=_hover_throttle(vehicle), arm=True)
    dt = 1 / 240
    min_roll_deg = 0.0
    for _ in range(360):  # 1.5с на затухання
        thrusts = controller.update(cmd_level, vehicle.state, dt)
        vehicle.apply_motor_commands(thrusts)
        world.step(dt)
        min_roll_deg = min(min_roll_deg, quat_to_euler_rad(vehicle.state.quat)[0] * 180.0 / np.pi)
    final_roll_deg = quat_to_euler_rad(vehicle.state.quat)[0] * 180.0 / np.pi
    assert abs(final_roll_deg) < 3.0
    assert min_roll_deg > -5.0  # не перегойдується сильно у зворотний бік

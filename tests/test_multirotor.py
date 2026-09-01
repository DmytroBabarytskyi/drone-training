"""Тести доп. фази "реалізм фізики" на рівні Multirotor: інерція моторів,
просідання батареї, екранний ефект, анізотропний опір — усі опційні,
вмикаються лише коли передано ``dt`` у ``apply_motor_commands`` (``dt=None``
лишає стару поведінку — сумісно з test_quad_fpv.py/test_quad_large.py)."""

import numpy as np
from panda3d.core import Vec3

from dronesim.physics import aerodynamics
from dronesim.physics.world import PhysicsWorld
from dronesim.vehicles.quad_fpv import QuadFPV

DT = 1 / 240


def _make_vehicle(pos=(0.0, 0.0, 5.0)):
    world = PhysicsWorld()
    vehicle = QuadFPV(world)
    vehicle.reset(pos=np.array(pos, dtype=np.float64))
    return world, vehicle


def test_apply_motor_commands_without_dt_is_instant_backward_compatible():
    """``dt=None`` (дефолт) -> без інерції моторів, без просідання батареї —
    точнісінько стара поведінка (як у test_quad_fpv.py)."""
    _, vehicle = _make_vehicle()
    thrusts = np.full(4, aerodynamics.motor_thrust(1.0, vehicle.cfg.max_thrust_per_motor))
    vehicle.apply_motor_commands(thrusts)  # без dt
    assert vehicle.state.battery == 1.0  # батарея не витрачається без dt


def test_motor_thrust_ramps_up_gradually_when_dt_given():
    """З dt (``motor_tau_s`` fpv_5inch = 0.03с) тяга наростає поступово, тож
    апарат набирає вертикальну швидкість ПОВІЛЬНІШЕ за перші кадри різкого
    стрибка газу з 0 до макс., ніж миттєвий (``dt=None``) відгук."""
    world_instant, vehicle_instant = _make_vehicle()
    world_lag, vehicle_lag = _make_vehicle()
    thrusts = np.full(4, aerodynamics.motor_thrust(1.0, vehicle_instant.cfg.max_thrust_per_motor))

    for _ in range(5):
        vehicle_instant.apply_motor_commands(thrusts)  # миттєво
        world_instant.step(DT)
        vehicle_lag.apply_motor_commands(thrusts, DT)  # з інерцією
        world_lag.step(DT)

    assert vehicle_lag.state.vel[2] < vehicle_instant.state.vel[2]  # повільніший розгін


def test_battery_soc_drains_under_sustained_load():
    world, vehicle = _make_vehicle()
    thrusts = np.full(4, aerodynamics.motor_thrust(1.0, vehicle.cfg.max_thrust_per_motor))
    for _ in range(240 * 2):  # 2с на повному газі
        vehicle.apply_motor_commands(thrusts, DT)
        world.step(DT)
    assert vehicle.state.battery < 1.0


def test_battery_soc_does_not_drain_at_zero_throttle():
    world, vehicle = _make_vehicle()
    for _ in range(240):
        vehicle.apply_motor_commands(np.zeros(4), DT)
        world.step(DT)
    assert vehicle.state.battery == 1.0


def test_battery_soc_persists_across_reset():
    """Заряд НЕ скидається при ``reset()`` (рестарт після краху не "заряджає"
    батарею — той самий акумулятор триває крізь сесію, docs/DECISIONS.md)."""
    world, vehicle = _make_vehicle()
    thrusts = np.full(4, aerodynamics.motor_thrust(1.0, vehicle.cfg.max_thrust_per_motor))
    for _ in range(240 * 3):
        vehicle.apply_motor_commands(thrusts, DT)
        world.step(DT)
    soc_before_reset = vehicle.state.battery
    assert soc_before_reset < 1.0

    vehicle.reset(pos=np.array([0.0, 0.0, 1.0]))
    assert vehicle.state.battery == soc_before_reset


def test_ground_effect_reduces_altitude_loss_near_ground():
    """Однакова команда газу (трохи нижче реального завису) біля землі має
    просідати ПОВІЛЬНІШЕ, ніж високо в повітрі — екранний ефект (IGE)
    компенсує частину дефіциту тяги (aerodynamics.ground_effect_factor)."""
    world_low, vehicle_low = _make_vehicle(pos=(0.0, 0.0, 0.05))
    world_high, vehicle_high = _make_vehicle(pos=(0.0, 0.0, 5.0))
    below_hover_frac = 0.85
    thrusts = np.full(4, below_hover_frac * (vehicle_low.cfg.mass * 9.81 / 4.0))

    for _ in range(30):
        vehicle_low.apply_motor_commands(thrusts, DT)
        world_low.step(DT)
        vehicle_high.apply_motor_commands(thrusts, DT)
        world_high.step(DT)

    # Обидва падають (нижче реального завису), але низький апарат ПОВІЛЬНІШЕ
    # (менш від'ємна вертикальна швидкість) завдяки екранному ефекту.
    assert vehicle_low.state.vel[2] > vehicle_high.state.vel[2]


def test_quadratic_drag_is_anisotropic_between_forward_and_vertical_motion():
    """``drag_coeff`` fpv_5inch — 3-вектор [0.06, 0.06, 0.11] (більший опір по
    Z, тілесна вісь). Однакова початкова швидкість уздовж локальної X і
    уздовж локальної Z (тіло в identity-орієнтації -> тілесні = світові осі)
    має дати РІЗНЕ гальмування (менше падіння швидкості по X, більше по Z)."""
    world_x, vehicle_x = _make_vehicle()
    world_z, vehicle_z = _make_vehicle()
    vehicle_x._node.setLinearVelocity(Vec3(5.0, 0.0, 0.0))  # noqa: SLF001
    vehicle_z._node.setLinearVelocity(Vec3(0.0, 0.0, 5.0))  # noqa: SLF001

    hover_thrust = np.full(4, vehicle_x.cfg.mass * 9.81 / 4.0)  # компенсувати гравітацію по Z
    for _ in range(10):
        vehicle_x.apply_motor_commands(hover_thrust, DT)
        world_x.step(DT)
        vehicle_z.apply_motor_commands(hover_thrust, DT)
        world_z.step(DT)

    # Порівнюємо, наскільки кожен загальмувався відносно старту (5.0 м/с):
    braking_x = 5.0 - float(vehicle_x._node.getLinearVelocity().x)  # noqa: SLF001
    braking_z = 5.0 - float(vehicle_z._node.getLinearVelocity().z)  # noqa: SLF001
    assert braking_z > braking_x

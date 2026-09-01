"""Тести навчальних цілей і воріт (world/targets.py) — чиста геометрія, без Panda3D."""

import numpy as np

from dronesim.world.targets import Gate, Target, check_hit


def test_static_target_position_is_constant():
    target = Target(target_id=0, base_pos=np.array([1.0, 2.0, 3.0]))
    assert np.allclose(target.position_at(0.0), [1.0, 2.0, 3.0])
    assert np.allclose(target.position_at(100.0), [1.0, 2.0, 3.0])


def test_moving_target_oscillates_between_endpoints():
    target = Target(
        target_id=0,
        base_pos=np.array([0.0, 0.0, 0.0]),
        moves_to=np.array([10.0, 0.0, 0.0]),
        period_s=4.0,
    )
    assert np.allclose(target.position_at(0.0), [0.0, 0.0, 0.0])  # старт у base_pos
    assert np.allclose(target.position_at(2.0), [10.0, 0.0, 0.0])  # половина періоду -> moves_to
    assert np.allclose(target.position_at(4.0), [0.0, 0.0, 0.0], atol=1e-9)  # повний період -> назад


def test_moving_target_is_deterministic_given_same_t():
    target = Target(
        target_id=0, base_pos=np.array([0.0, 0.0, 0.0]), moves_to=np.array([5.0, 5.0, 0.0]), period_s=3.0
    )
    assert np.allclose(target.position_at(1.7), target.position_at(1.7))


def test_check_hit_true_within_radius_and_angle():
    target = Target(target_id=0, base_pos=np.array([10.0, 0.0, 0.0]), radius=2.0)
    vehicle_pos = np.array([9.0, 0.0, 0.0])  # 1м від цілі, у межах radius=2.0
    forward = np.array([1.0, 0.0, 0.0])  # дивиться прямо на ціль
    assert check_hit(target, vehicle_pos, forward, t=0.0, max_angle_rad=0.3)


def test_check_hit_false_outside_radius():
    target = Target(target_id=0, base_pos=np.array([10.0, 0.0, 0.0]), radius=2.0)
    vehicle_pos = np.array([0.0, 0.0, 0.0])  # 10м від цілі
    forward = np.array([1.0, 0.0, 0.0])
    assert not check_hit(target, vehicle_pos, forward, t=0.0, max_angle_rad=0.3)


def test_check_hit_false_when_looking_away():
    target = Target(target_id=0, base_pos=np.array([10.0, 0.0, 0.0]), radius=2.0)
    vehicle_pos = np.array([9.0, 0.0, 0.0])
    forward = np.array([-1.0, 0.0, 0.0])  # дивиться в протилежний бік
    assert not check_hit(target, vehicle_pos, forward, t=0.0, max_angle_rad=0.3)


def test_check_hit_uses_moving_target_position_at_time_t():
    target = Target(
        target_id=0,
        base_pos=np.array([0.0, 0.0, 0.0]),
        moves_to=np.array([10.0, 0.0, 0.0]),
        period_s=4.0,
        radius=1.5,
    )
    forward = np.array([1.0, 0.0, 0.0])
    # у t=2.0 ціль в moves_to=[10,0,0]; апарат поруч з [10,0,0] -> влучання
    assert check_hit(target, np.array([9.0, 0.0, 0.0]), forward, t=2.0, max_angle_rad=0.3)
    # той самий апарат при t=0.0 ціль ще в [0,0,0] -> занадто далеко
    assert not check_hit(target, np.array([9.0, 0.0, 0.0]), forward, t=0.0, max_angle_rad=0.3)


def test_gate_passes_through_forward_crossing():
    gate = Gate(gate_id=0, center=np.array([5.0, 0.0, 0.0]), normal=np.array([1.0, 0.0, 0.0]), radius=2.0)
    prev_pos = np.array([4.0, 0.0, 0.0])  # позаду площини (d<0)
    curr_pos = np.array([6.0, 0.0, 0.0])  # попереду площини (d>=0), в межах radius
    assert gate.passes_through(prev_pos, curr_pos)


def test_gate_does_not_pass_through_backward_crossing():
    gate = Gate(gate_id=0, center=np.array([5.0, 0.0, 0.0]), normal=np.array([1.0, 0.0, 0.0]), radius=2.0)
    prev_pos = np.array([6.0, 0.0, 0.0])  # попереду
    curr_pos = np.array([4.0, 0.0, 0.0])  # позаду -> назад, не зараховується
    assert not gate.passes_through(prev_pos, curr_pos)


def test_gate_does_not_pass_through_outside_radius():
    gate = Gate(gate_id=0, center=np.array([5.0, 0.0, 0.0]), normal=np.array([1.0, 0.0, 0.0]), radius=1.0)
    prev_pos = np.array([4.0, 5.0, 0.0])  # перетинає площину, але далеко вбік (y=5)
    curr_pos = np.array([6.0, 5.0, 0.0])
    assert not gate.passes_through(prev_pos, curr_pos)


def test_gate_no_crossing_stays_on_same_side():
    gate = Gate(gate_id=0, center=np.array([5.0, 0.0, 0.0]), normal=np.array([1.0, 0.0, 0.0]), radius=2.0)
    prev_pos = np.array([1.0, 0.0, 0.0])
    curr_pos = np.array([2.0, 0.0, 0.0])
    assert not gate.passes_through(prev_pos, curr_pos)

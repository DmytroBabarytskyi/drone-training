"""Тести конвертації кватерніона в кути Ейлера (utils/math3d.py)."""

import numpy as np

from dronesim.utils.math3d import (
    quat_conjugate,
    quat_rotate_vector,
    quat_to_euler_rad,
    vehicle_forward_world,
)


def test_identity_quat_is_zero_angles():
    roll, pitch, yaw = quat_to_euler_rad(np.array([0.0, 0.0, 0.0, 1.0]))
    assert abs(roll) < 1e-9
    assert abs(pitch) < 1e-9
    assert abs(yaw) < 1e-9


def test_quarter_turn_yaw():
    # Поворот на 90° навколо Z: quat = (0, 0, sin(45°), cos(45°))
    half = np.pi / 4
    quat = np.array([0.0, 0.0, np.sin(half), np.cos(half)])
    roll, pitch, yaw = quat_to_euler_rad(quat)
    assert abs(roll) < 1e-9
    assert abs(pitch) < 1e-9
    assert abs(yaw - np.pi / 2) < 1e-9


def test_quarter_turn_roll():
    half = np.pi / 4
    quat = np.array([np.sin(half), 0.0, 0.0, np.cos(half)])
    roll, pitch, yaw = quat_to_euler_rad(quat)
    assert abs(roll - np.pi / 2) < 1e-9
    assert abs(pitch) < 1e-9
    assert abs(yaw) < 1e-9


def test_quat_rotate_vector_identity_is_noop():
    v = np.array([1.0, 2.0, 3.0])
    rotated = quat_rotate_vector(np.array([0.0, 0.0, 0.0, 1.0]), v)
    assert np.allclose(rotated, v)


def test_quat_rotate_vector_90deg_yaw_matches_euler_convention():
    # Той самий кватерніон, що й test_quarter_turn_yaw: yaw=+90° навколо Z.
    half = np.pi / 4
    quat = np.array([0.0, 0.0, np.sin(half), np.cos(half)])
    forward = np.array([1.0, 0.0, 0.0])
    rotated = quat_rotate_vector(quat, forward)
    # +90° навколо Z переводить +X -> +Y (узгоджено з правою системою координат)
    assert np.allclose(rotated, [0.0, 1.0, 0.0], atol=1e-9)


def test_vehicle_forward_world_identity_is_local_x():
    forward = vehicle_forward_world(np.array([0.0, 0.0, 0.0, 1.0]))
    assert np.allclose(forward, [1.0, 0.0, 0.0])
    assert abs(np.linalg.norm(forward) - 1.0) < 1e-9


def test_quat_conjugate_undoes_rotation():
    half = np.pi / 4
    quat = np.array([0.0, 0.0, np.sin(half), np.cos(half)])  # +90° yaw
    v_world = np.array([3.0, -2.0, 1.0])
    v_body = quat_rotate_vector(quat_conjugate(quat), v_world)
    v_back_to_world = quat_rotate_vector(quat, v_body)
    assert np.allclose(v_back_to_world, v_world, atol=1e-9)


def test_quat_conjugate_identity_is_noop():
    identity = np.array([0.0, 0.0, 0.0, 1.0])
    assert np.allclose(quat_conjugate(identity), identity)

"""Тести генератора ракурсів (ml/data/scripted_flight.py) — чиста математика."""

import numpy as np

from dronesim.ml.data.scripted_flight import generate_viewpoints


def test_generates_requested_count():
    viewpoints = generate_viewpoints(
        np.array([0.0, 0.0, 0.0]), n=20, radius_range=(5.0, 10.0), altitude_range=(-1.0, 1.0), seed=0
    )
    assert len(viewpoints) == 20


def test_look_at_is_always_target_pos():
    target = np.array([3.0, -2.0, 5.0])
    viewpoints = generate_viewpoints(target, n=10, radius_range=(5.0, 5.0), altitude_range=(0.0, 0.0), seed=1)
    for vp in viewpoints:
        assert np.allclose(vp.look_at, target)


def test_radius_within_requested_range():
    target = np.array([0.0, 0.0, 3.0])
    viewpoints = generate_viewpoints(
        target, n=50, radius_range=(4.0, 8.0), altitude_range=(0.0, 0.0), seed=2
    )
    for vp in viewpoints:
        horizontal_dist = np.linalg.norm((vp.pos - target)[:2])
        assert 4.0 - 1e-6 <= horizontal_dist <= 8.0 + 1e-6


def test_altitude_offset_within_requested_range():
    target = np.array([0.0, 0.0, 3.0])
    viewpoints = generate_viewpoints(
        target, n=50, radius_range=(5.0, 5.0), altitude_range=(-2.0, 4.0), seed=3
    )
    for vp in viewpoints:
        z_offset = vp.pos[2] - target[2]
        assert -2.0 - 1e-6 <= z_offset <= 4.0 + 1e-6


def test_same_seed_is_reproducible():
    target = np.array([1.0, 1.0, 1.0])
    vps_a = generate_viewpoints(target, n=5, radius_range=(3.0, 6.0), altitude_range=(-1.0, 1.0), seed=42)
    vps_b = generate_viewpoints(target, n=5, radius_range=(3.0, 6.0), altitude_range=(-1.0, 1.0), seed=42)
    for a, b in zip(vps_a, vps_b):
        assert np.allclose(a.pos, b.pos)


def test_different_seed_gives_different_viewpoints():
    target = np.array([0.0, 0.0, 0.0])
    vps_a = generate_viewpoints(target, n=5, radius_range=(3.0, 6.0), altitude_range=(-1.0, 1.0), seed=1)
    vps_b = generate_viewpoints(target, n=5, radius_range=(3.0, 6.0), altitude_range=(-1.0, 1.0), seed=2)
    assert not all(np.allclose(a.pos, b.pos) for a, b in zip(vps_a, vps_b))

"""Tests for the analytic hex-packing placement baseline."""

import numpy as np

from dronesim.ml.swarm.analytic_policy import AnalyticPlacementPolicy, hex_lattice_slots


def test_lattice_has_requested_count_and_respects_spacing():
    centre = np.array([5.0, -3.0])
    slots = hex_lattice_slots(25, spacing=6.0, centre=centre)

    assert slots.shape == (25, 2)
    pairwise = np.linalg.norm(slots[:, None, :] - slots[None, :, :], axis=-1)
    off_diagonal = pairwise[~np.eye(len(slots), dtype=bool)]
    # Hex lattice nearest-neighbour distance is exactly `spacing`.
    assert off_diagonal.min() > 6.0 - 1e-6


def test_lattice_is_centred_on_the_prediction():
    centre = np.array([12.0, 7.0])
    slots = hex_lattice_slots(19, spacing=4.0, centre=centre)
    np.testing.assert_allclose(slots.mean(axis=0), centre, atol=1e-9)


def test_spacing_never_violates_min_separation():
    # Kill radius so small that 2*r would put slots closer than the swarm's
    # own separation limit — separation must win.
    policy = AnalyticPlacementPolicy(n_drones=9, hit_radius_m=0.5, min_separation_m=4.0,
                                     max_speed_mps=25.0, max_accel_mps2=15.0)
    assert policy.spacing == 4.0


def test_action_points_toward_assigned_slots_and_is_bounded():
    policy = AnalyticPlacementPolicy(n_drones=9, hit_radius_m=3.0, min_separation_m=2.5,
                                     max_speed_mps=25.0, max_accel_mps2=15.0)
    predicted = np.array([20.0, 0.0])
    policy.reset(predicted)

    positions = np.zeros((9, 2))  # all far to the left of the prediction
    action = policy.act(positions).reshape(9, 2)

    assert np.all(np.abs(action) <= 1.0 + 1e-9)
    # Every drone should be commanded rightward, toward the formation.
    assert np.all(action[:, 0] > 0.0)


def test_assignment_is_one_to_one():
    policy = AnalyticPlacementPolicy(n_drones=6, hit_radius_m=3.0, min_separation_m=2.5,
                                     max_speed_mps=25.0, max_accel_mps2=15.0)
    policy.reset(np.array([0.0, 0.0]))
    positions = np.array([[10.0, 0.0], [-10.0, 0.0], [0.0, 10.0],
                          [0.0, -10.0], [7.0, 7.0], [-7.0, -7.0]])
    action = policy.act(positions).reshape(6, 2)
    # A one-to-one assignment means no two drones are driven to an identical
    # target, so no two commands should be exactly equal here.
    assert len({tuple(np.round(a, 6)) for a in action}) == 6

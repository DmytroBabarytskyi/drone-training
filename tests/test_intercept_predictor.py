"""Tests for the deterministic plane-crossing predictor (a pure function,
no ML, no Panda3D — see ml/swarm/intercept_predictor.py)."""

import numpy as np

from dronesim.ml.swarm.intercept_predictor import CueingSample, predict_plane_crossing


def _noiseless_samples(start, velocity, times):
    start = np.array(start, dtype=np.float64)
    velocity = np.array(velocity, dtype=np.float64)
    return [CueingSample(t=t, pos=start + velocity * t) for t in times]


def test_exact_linear_trajectory_predicts_exact_crossing():
    # Target starts at x=100, flies straight at (-20, 1, -0.5) m/s.
    start = [100.0, 5.0, 2.0]
    velocity = [-20.0, 1.0, -0.5]
    samples = _noiseless_samples(start, velocity, times=[0.0, 1.0, 2.0])

    prediction = predict_plane_crossing(samples, plane_x=0.0, sensor_noise_std=1.0)

    assert prediction is not None
    expected_t_cross = 100.0 / 20.0  # time to travel from x=100 to x=0 at 20 m/s
    expected_point = np.array(
        [5.0 + 1.0 * expected_t_cross, 2.0 - 0.5 * expected_t_cross]
    )
    assert prediction.time_to_cross == expected_t_cross - 2.0  # 2.0s already elapsed
    np.testing.assert_allclose(prediction.point, expected_point, atol=1e-9)


def test_too_few_samples_returns_none():
    samples = _noiseless_samples([10.0, 0.0, 0.0], [-5.0, 0.0, 0.0], times=[0.0])
    assert predict_plane_crossing(samples, plane_x=0.0, sensor_noise_std=1.0) is None


def test_target_moving_away_returns_none():
    # Moving in +x, away from a plane at x=0 that's already behind it.
    samples = _noiseless_samples([10.0, 0.0, 0.0], [5.0, 0.0, 0.0], times=[0.0, 1.0])
    assert predict_plane_crossing(samples, plane_x=0.0, sensor_noise_std=1.0) is None


def test_more_samples_and_less_extrapolation_tighten_confidence():
    fast_close = predict_plane_crossing(
        _noiseless_samples([50.0, 0.0, 0.0], [-40.0, 0.0, 0.0], times=[0.0, 0.5, 1.0]),
        plane_x=0.0,
        sensor_noise_std=3.0,
    )
    far_sparse = predict_plane_crossing(
        _noiseless_samples([400.0, 0.0, 0.0], [-40.0, 0.0, 0.0], times=[0.0, 1.0]),
        plane_x=0.0,
        sensor_noise_std=3.0,
    )
    assert fast_close is not None and far_sparse is not None
    assert fast_close.confidence_radius < far_sparse.confidence_radius

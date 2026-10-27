"""Tests for the EW-degraded sizing laws (ml/swarm/ew_laws.py)."""

import numpy as np
import pytest

from dronesim.ml.swarm.coverage_laws import ETA2, planar_coverage
from dronesim.ml.swarm.ew_laws import (
    PAIR_OVERLAP_C0,
    degraded_ceiling,
    degraded_coverage,
    degraded_mesh_size,
    drift_sigma,
    hex_pairs_per_agent,
    lens_area,
    pair_overlap_fraction,
    reformed_coverage,
)


def test_no_degradation_reduces_to_planar_law():
    for n in (1, 9, 25, 60):
        assert degraded_coverage(n, 9.25, 2.0) == pytest.approx(planar_coverage(n, 9.25, 2.0))


def test_loss_scales_coverage_linearly():
    intact = degraded_coverage(25, 9.25, 2.0)
    assert degraded_coverage(25, 9.25, 2.0, loss=0.3) == pytest.approx(0.7 * intact)


def test_common_sigma_is_a_wider_target_error():
    assert degraded_coverage(25, 4.0, 2.0, common_sigma=3.0) == pytest.approx(
        planar_coverage(25, 5.0, 2.0)
    )


def test_bias_only_hurts():
    assert degraded_coverage(25, 9.25, 2.0, common_bias=5.0) < degraded_coverage(25, 9.25, 2.0)


def test_lens_area_limits():
    r = 1.5
    assert lens_area(0.0, r) == pytest.approx(np.pi * r * r)
    assert lens_area(2.0 * r, r) == pytest.approx(0.0)
    assert lens_area(5.0 * r, r) == 0.0


def test_pair_overlap_matches_leading_order():
    r = 1.0
    x = 0.01
    assert pair_overlap_fraction(x, r) / x**1.5 == pytest.approx(PAIR_OVERLAP_C0, rel=0.02)
    assert pair_overlap_fraction(0.0, r) == 0.0


def test_pair_overlap_is_scale_free():
    assert pair_overlap_fraction(0.6, 2.0) == pytest.approx(pair_overlap_fraction(0.3, 1.0))


def test_pairs_per_agent_approaches_three():
    assert hex_pairs_per_agent(1) == 0.0
    assert hex_pairs_per_agent(7) == pytest.approx(12 / 7)
    assert hex_pairs_per_agent(10_000) == pytest.approx(3.0, abs=0.05)


def test_reforming_recovers_ceiling_and_beats_holes():
    n, sigma, r, q = 400, 2.0, 2.0, 0.4
    assert reformed_coverage(n, sigma, r, q) == pytest.approx(ETA2, abs=1e-3)
    assert reformed_coverage(n, sigma, r, q) > degraded_coverage(n, sigma, r, loss=q)
    assert reformed_coverage(n, sigma, r, 0.0) == pytest.approx(planar_coverage(n, sigma, r))


def test_ceiling_and_mesh_size():
    r = 2.0
    assert degraded_ceiling(r) == pytest.approx(ETA2)
    assert degraded_ceiling(r, loss=0.25) == pytest.approx(0.75 * ETA2)
    assert degraded_mesh_size(0.7, 9.25, r, loss=0.25) == float("inf")

    n = degraded_mesh_size(0.5, 9.25, r, loss=0.1, jitter_sigma=0.5)
    # The inverse uses the interior pair count, so it is conservative.
    assert degraded_coverage(int(np.ceil(n)), 9.25, r, loss=0.1, jitter_sigma=0.5) >= 0.5


def test_drift_sigma_is_per_axis():
    assert drift_sigma(100.0, 0.02) == pytest.approx(2.0 / np.sqrt(2.0))

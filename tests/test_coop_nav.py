"""Tests for the cooperative-navigation bound (ml/swarm/coop_nav.py)."""

import numpy as np
import pytest

from dronesim.ml.swarm.analytic_policy import hex_lattice_slots
from dronesim.ml.swarm.coop_nav import (
    ErrorModes,
    coop_coverage,
    error_modes,
    floppy_fraction,
    information_matrix,
    thin_edges,
    touching_pairs,
)
from dronesim.ml.swarm.coverage_laws import planar_coverage

R = 3.0
SPACING = 2.0 * R


def _mesh(n):
    slots = hex_lattice_slots(n, SPACING, np.zeros(2))
    return slots, touching_pairs(slots, SPACING)


def test_touching_pairs_of_a_seven_cluster():
    _, edges = _mesh(7)
    assert len(edges) == 12


def test_prior_only_modes():
    n, sn = 25, 1.5
    slots, _ = _mesh(n)
    info = information_matrix(slots, np.empty((0, 2), dtype=int), sigma_n=sn, sigma_r=0.1)
    m = error_modes(slots, info, SPACING)
    assert m.translation == pytest.approx(sn / np.sqrt(n))
    assert m.shape == pytest.approx(sn)
    assert m.pair == pytest.approx(sn)
    assert m.marginal == pytest.approx(sn * np.sqrt((2 * n - 1) / (2 * n)))


def test_ranging_leaves_translation_untouched():
    n, sn = 25, 1.5
    slots, edges = _mesh(n)
    m = error_modes(slots, information_matrix(slots, edges, sigma_n=sn, sigma_r=0.1), SPACING)
    assert m.translation == pytest.approx(sn / np.sqrt(n))
    assert m.pair < 0.1
    assert m.shape < 0.2


def test_centroid_sensor_fixes_translation_exactly():
    n, sn, sg, fixes = 25, 1.5, 4.0, 10
    slots, edges = _mesh(n)
    info = information_matrix(slots, edges, sigma_n=sn, sigma_r=0.1,
                              centroid_sigma=sg, n_fixes=fixes)
    m = error_modes(slots, info, SPACING)
    assert m.translation == pytest.approx(1.0 / np.sqrt(n / sn**2 + fixes / sg**2))


def test_anchors_need_sigma():
    slots, edges = _mesh(7)
    with pytest.raises(ValueError):
        information_matrix(slots, edges, sigma_n=1.0, sigma_r=0.1, anchors=(0,))


def test_zero_error_coverage_is_the_planar_law():
    zero = ErrorModes(0.0, 0.0, 0.0, 0.0, 0.0)
    assert coop_coverage(25, 4.57, R, zero) == pytest.approx(planar_coverage(25, 4.57, R))


def test_ranging_recovers_coverage():
    n, sn = 25, 1.5
    slots, edges = _mesh(n)
    alone = error_modes(slots, information_matrix(slots, edges[:0], sigma_n=sn, sigma_r=0.1),
                        SPACING)
    ranged = error_modes(slots, information_matrix(slots, edges, sigma_n=sn, sigma_r=0.1),
                         SPACING)
    assert coop_coverage(n, 4.57, R, ranged) > coop_coverage(n, 4.57, R, alone) + 0.1


def test_floppy_fraction():
    slots, edges = _mesh(25)
    assert floppy_fraction(slots, edges) == 0.0
    assert floppy_fraction(slots, edges[:0]) == pytest.approx((50 - 3) / 50)


def test_thin_edges_is_seeded():
    _, edges = _mesh(61)
    a = thin_edges(edges, 0.7, np.random.default_rng(1))
    b = thin_edges(edges, 0.7, np.random.default_rng(1))
    assert np.array_equal(a, b)
    assert thin_edges(edges, 1.0, np.random.default_rng(1)) is edges

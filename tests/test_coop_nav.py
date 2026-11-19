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


# --- B2: the estimator and the attacks on it ---------------------------------

from dronesim.ml.swarm.coop_nav import (  # noqa: E402
    NavConfig,
    empirical_modes,
    gauss_newton,
    observe,
    place_mesh,
    placement_coverage,
    robust_estimate,
    spoofed_coverage,
)

SENSOR = {"centroid_sigma": 4.0, "n_fixes": 10}


def test_nuisance_spoof_leaves_translation_to_the_sensor():
    n, sn, sg, fixes = 25, 1.5, 4.0, 10
    slots, edges = _mesh(n)
    info = information_matrix(slots, edges, sigma_n=sn, sigma_r=0.1, centroid_sigma=sg,
                              n_fixes=fixes, spoof_nuisance=True)
    m = error_modes(slots, info, SPACING)
    assert m.translation == pytest.approx(sg / np.sqrt(fixes))


def test_gauss_newton_recovers_truth_from_clean_measurements():
    slots, edges = _mesh(19)
    cfg = NavConfig(1.0, 1e-4, **SENSOR)
    rng = np.random.default_rng(3)
    truth = slots + rng.normal(0.0, 0.5, size=slots.shape)
    obs = observe(truth, edges, cfg, rng)
    est = gauss_newton(obs, cfg)
    # Near-exact ranges fix every edge; translation and rotation are left to
    # the noisy prior and sensor.
    def lengths(x):
        return np.linalg.norm(x[edges[:, 1]] - x[edges[:, 0]], axis=1)

    assert np.abs(lengths(est.x) - lengths(truth)).max() < 1e-3
    assert np.abs(est.x - truth).max() < 1.5
    assert est.dof == 2 * 19 + len(edges) + 2 - 2 * 19


def test_spoof_model_needs_a_reference():
    slots, edges = _mesh(7)
    cfg = NavConfig(1.0, 0.1)
    obs = observe(slots, edges, cfg, np.random.default_rng(0))
    with pytest.raises(ValueError):
        gauss_newton(obs, cfg, model_spoof=True)


def test_estimator_errors_match_the_bound():
    slots, edges = _mesh(19)
    cfg = NavConfig(1.0, 0.1, **SENSOR)
    rng = np.random.default_rng(4)
    errs = np.array([slots - place_mesh(slots, edges, cfg, rng, estimator="naive")[0]
                     for _ in range(300)])
    emp = empirical_modes(slots, errs, SPACING)
    bound = error_modes(slots, cfg.info(slots, edges), SPACING)
    assert emp.translation == pytest.approx(bound.translation, rel=0.15)
    assert emp.pair == pytest.approx(bound.pair, rel=0.15)


def test_robust_estimate_removes_a_large_spoof():
    slots, edges = _mesh(19)
    cfg = NavConfig(1.0, 0.1, **SENSOR)
    rng = np.random.default_rng(5)
    spoof = np.array([15.0, 0.0])
    naive = place_mesh(slots, edges, cfg, rng, estimator="naive", spoof=spoof)[0]
    robust, est = place_mesh(slots, edges, cfg, rng, estimator="robust", spoof=spoof)
    assert "spoof" in est.excluded
    assert np.linalg.norm((slots - naive).mean(axis=0)) > 10.0
    assert np.linalg.norm((slots - robust).mean(axis=0)) < 4.0


def test_robust_estimate_drops_a_lying_anchor():
    slots, edges = _mesh(19)
    cfg = NavConfig(1.0, 0.1, anchors=(0, 9, 14), anchor_sigma=0.5)
    rng = np.random.default_rng(6)
    lie = np.zeros((3, 2))
    lie[1] = [12.0, 0.0]
    obs = observe(slots, edges, cfg, rng, anchor_bias=lie)
    est = robust_estimate(obs, cfg)
    assert est.excluded == (9,)


def test_placement_coverage_of_the_exact_lattice():
    n, sigma = 25, 4.57
    slots, _ = _mesh(n)
    cov = placement_coverage(np.repeat(slots[None], 50, axis=0), sigma, R,
                             np.random.default_rng(7), n_targets=2000)
    assert cov == pytest.approx(planar_coverage(n, sigma, R), abs=0.01)


def test_spoofed_coverage_limits():
    n, sigma = 25, 10.0
    slots, edges = _mesh(n)
    cfg = NavConfig(1.5, 0.1, **SENSOR)
    kw = {"spacing": SPACING}
    clean = coop_coverage(n, sigma, R, error_modes(slots, cfg.info(slots, edges), SPACING))
    assert spoofed_coverage(slots, edges, cfg, sigma, R, estimator="naive", **kw) \
        == pytest.approx(clean)
    big = 30.0
    naive = spoofed_coverage(slots, edges, cfg, sigma, R, spoof=big, estimator="naive", **kw)
    robust = spoofed_coverage(slots, edges, cfg, sigma, R, spoof=big, **kw)
    lost = spoofed_coverage(slots, edges, cfg, sigma, R, spoof=big, uplink_loss=1.0, **kw)
    assert naive < 0.2 < robust
    assert lost == pytest.approx(
        spoofed_coverage(slots, edges, NavConfig(1.5, 0.1), sigma, R, spoof=big, **kw))

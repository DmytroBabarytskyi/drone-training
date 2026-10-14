"""Monte Carlo coverage measurement, shared by the coverage experiments.

The mesh geometry comes from ``analytic_policy.hex_lattice_slots`` — the same
generator the analytic placement policy flies to — so a measured coverage is
a measurement of the production lattice, not of an idealised one.
"""

from __future__ import annotations

import numpy as np

from dronesim.ml.swarm.analytic_policy import hex_lattice_slots

CHUNK = 20_000  # points per block, keeps the distance matrix well under 1 GB


def covered_fraction(slots: np.ndarray, points: np.ndarray, r: float) -> float:
    """Fraction of ``points`` within ``r`` of at least one slot."""
    hits = 0
    for start in range(0, len(points), CHUNK):
        block = points[start : start + CHUNK]
        d2 = np.sum((block[:, None, :] - slots[None, :, :]) ** 2, axis=-1)
        hits += int(np.count_nonzero(np.min(d2, axis=1) <= r * r))
    return hits / len(points)


def gaussian_points(rng: np.random.Generator, sigma: float, trials: int, dims: int = 2):
    return rng.normal(0.0, sigma, size=(trials, dims))


def hex_coverage_mc(
    n_agents: int,
    sigma: float,
    r: float,
    trials: int,
    rng: np.random.Generator,
    spacing: float | None = None,
) -> float:
    """Measured coverage of ``n_agents`` hex-packed discs over an isotropic
    2-D Gaussian of width ``sigma``, centred on the mode.

    ``spacing`` defaults to ``2*r`` — touching, non-overlapping discs, the
    packing regime the closed form assumes.
    """
    if n_agents <= 0:
        return 0.0
    slots = hex_lattice_slots(n_agents, spacing or 2.0 * r, np.zeros(2))
    points = gaussian_points(rng, sigma, trials)
    return covered_fraction(slots, points, r)


def fcc_lattice_slots(n_slots: int, spacing: float, centre: np.ndarray) -> np.ndarray:
    """The ``n_slots`` face-centred-cubic lattice points closest to ``centre``.

    The volumetric counterpart of ``hex_lattice_slots``: FCC is a densest
    packing of equal spheres (Kepler), so this is the 3-D arrangement the
    volumetric law assumes. Same construction — build comfortably more points
    than needed, keep the closest, which yields a roughly spherical cluster
    matching the radial symmetry of the error distribution.
    """
    # FCC primitive vectors, scaled so nearest-neighbour distance is `spacing`.
    a = spacing / np.sqrt(2.0)
    b1 = np.array([a, a, 0.0])
    b2 = np.array([a, 0.0, a])
    b3 = np.array([0.0, a, a])

    rings = int(np.ceil((max(n_slots, 1) / 4.0) ** (1.0 / 3.0))) + 2
    idx = np.arange(-rings, rings + 1)
    i, j, k = np.meshgrid(idx, idx, idx, indexing="ij")
    points = (
        i.ravel()[:, None] * b1 + j.ravel()[:, None] * b2 + k.ravel()[:, None] * b3
    )
    order = np.argsort(np.linalg.norm(points, axis=1))
    return points[order[:n_slots]] + centre


def fcc_coverage_mc(
    n_agents: int,
    sigma: float,
    r: float,
    trials: int,
    rng: np.random.Generator,
) -> float:
    """Measured coverage of ``n_agents`` FCC-packed spheres over an isotropic
    3-D Gaussian of width ``sigma``."""
    if n_agents <= 0:
        return 0.0
    slots = fcc_lattice_slots(n_agents, 2.0 * r, np.zeros(3))
    points = gaussian_points(rng, sigma, trials, dims=3)
    return covered_fraction(slots, points, r)


def smallest_n_reaching(
    target: float,
    sigma: float,
    r: float,
    trials: int,
    rng: np.random.Generator,
    n_max: int = 4000,
) -> int | None:
    """Smallest mesh size whose *measured* coverage reaches ``target``.

    Used to invert the law empirically instead of fitting the exponent back
    from the closed form, which would be circular. Doubling search followed
    by a bisection, so the cost is logarithmic in ``n_max``.
    """
    lo, hi = 1, 1
    while hi <= n_max:
        if hex_coverage_mc(hi, sigma, r, trials, rng) >= target:
            break
        lo = hi + 1
        hi *= 2
    else:
        return None

    while lo < hi:
        mid = (lo + hi) // 2
        if hex_coverage_mc(mid, sigma, r, trials, rng) >= target:
            hi = mid
        else:
            lo = mid + 1
    return lo

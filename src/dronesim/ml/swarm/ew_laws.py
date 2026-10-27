"""Sizing laws for a mesh degraded by electronic warfare — the extension of
``coverage_laws.py`` behind the follow-up preprint.

The planar law assumes every agent is alive, knows where it is, and sits on
its lattice slot. Jamming and spoofing break each of those separately, and the
four failures below enter the law in structurally different ways:

``loss``          per-agent probability ``q`` that an agent drops out of the
                  mesh (navigation or link lost, agent leaves its slot). With
                  disjoint discs the expected coverage is exactly ``(1-q)``
                  times the intact one, so the *ceiling itself* falls to
                  ``eta2*(1-q)`` and no mesh size recovers it.
``common_sigma``  per-axis std of a navigation error *shared* by the whole
                  mesh (a common reference, a spoofed fix). The mesh moves
                  rigidly, so this is indistinguishable from a wider target
                  error: ``sigma_eff^2 = sigma^2 + sigma_c^2``.
``common_bias``   a deterministic shared offset (a coherent spoof). Enters as
                  the non-centrality of a chi-square with two degrees of
                  freedom — the bias case of the robustness experiment.
``jitter_sigma``  per-axis std of an *independent* per-agent navigation error
                  (dead reckoning / visual-inertial drift). Each disc keeps its
                  own mass to first order, but neighbouring discs that were
                  touching now overlap, and overlap is lost coverage. The loss
                  per touching pair is the expected lens area, which scales as
                  ``(sigma_n/r)^(3/2)`` — see :data:`PAIR_OVERLAP_C0`.

Combined, with ``F`` the chi-square CDF and ``f`` the pair overlap fraction::

    p = (1-q) * eta2 * F(rho^2/sigma_eff^2; 2, b^2/sigma_eff^2)
        - (1-q)^2 * f(sigma_n/r) * sum_pairs pi*r^2 * phi(m_ij)

with ``rho^2 = N*r^2/eta2``, ``sigma_eff^2 = sigma^2 + sigma_c^2 + sigma_n^2``
and ``phi(m_ij)`` the target density at the midpoint of touching pair ij.
When the error is wide next to the mesh the sum is about three times the
covered mass, giving the compact form ``... * (1 - 3*(1-q)*f)`` used by the
ceiling and the inverse.

The jitter term is a pair expansion: it is accurate to about 0.01 absolute
for ``sigma_n <= r/2`` and turns optimistic beyond, where triple overlaps
appear and agents start leaving their neighbourhood altogether. Past
``sigma_n ~ r`` it is not even monotone, so do not use it there.

Nothing here simulates a jammer: the degradations are parameters, so the laws
say what a given level of degradation costs, not how likely it is.
"""

from __future__ import annotations

from functools import lru_cache

import numpy as np
from scipy.integrate import quad
from scipy.spatial import cKDTree
from scipy.special import gamma
from scipy.stats import binom, ncx2, rice

from dronesim.ml.swarm.analytic_policy import hex_lattice_slots
from dronesim.ml.swarm.coverage_laws import ETA2, planar_coverage

__all__ = [
    "PAIR_OVERLAP_C0",
    "INTERIOR_PAIRS_PER_AGENT",
    "lens_area",
    "pair_overlap_fraction",
    "hex_pairs_per_agent",
    "degraded_coverage",
    "reformed_coverage",
    "degraded_ceiling",
    "degraded_mesh_size",
    "drift_sigma",
]

# Leading-order constant of pair_overlap_fraction: f ~ C0*(sigma_n/r)^(3/2).
# A lens of depth eps = 2r - D has area (4/3)*sqrt(r)*eps^(3/2); the pair
# separation fluctuates along the joining line with std sqrt(2)*sigma_n, and
# E[Z^(3/2); Z>0] = 2^(3/4)*Gamma(5/4)/(2*sqrt(pi)) for a standard normal Z.
PAIR_OVERLAP_C0 = float(
    (4.0 / 3.0) * 2.0**0.75 * (2.0**0.75 * gamma(1.25) / (2.0 * np.sqrt(np.pi))) / np.pi
)  # 0.3069...

# Each interior agent of a hex pack touches six neighbours: three pairs.
INTERIOR_PAIRS_PER_AGENT = 3.0


def lens_area(distance, r: float) -> np.ndarray | float:
    """Area of the intersection of two discs of radius ``r`` whose centres are
    ``distance`` apart; zero once they no longer touch."""
    d = np.asarray(distance, dtype=np.float64)
    inside = np.clip(d / (2.0 * r), 0.0, 1.0)
    area = 2.0 * r * r * np.arccos(inside) - 0.5 * d * np.sqrt(np.maximum(4.0 * r * r - d * d, 0.0))
    return np.where(d < 2.0 * r, area, 0.0)


def pair_overlap_fraction(jitter_sigma: float, r: float) -> float:
    """Expected overlap of two originally touching discs, as a fraction of one
    disc's area, when each centre is displaced by an independent isotropic
    Gaussian of per-axis std ``jitter_sigma``.

    The centre separation is then Rice-distributed with location ``2r`` and
    scale ``sqrt(2)*jitter_sigma``; the expectation of the lens area over it is
    a one-dimensional integral, evaluated by quadrature so the result is
    deterministic. Tends to ``PAIR_OVERLAP_C0 * (jitter_sigma/r)^1.5``.
    """
    if jitter_sigma <= 0.0:
        return 0.0
    scale = np.sqrt(2.0) * jitter_sigma
    dist = rice(b=2.0 * r / scale, scale=scale)
    expected, _ = quad(lambda d: lens_area(d, r) * dist.pdf(d), 0.0, 2.0 * r, limit=200)
    return float(expected / (np.pi * r * r))


@lru_cache(maxsize=512)
def hex_pairs_per_agent(n_agents: int) -> float:
    """Touching pairs per agent in the ``n_agents``-slot hex cluster that
    :func:`hex_lattice_slots` produces. Approaches three from below as the
    boundary's share shrinks."""
    if n_agents < 2:
        return 0.0
    slots = hex_lattice_slots(n_agents, spacing=1.0, centre=np.zeros(2))
    # Unit spacing: touching pairs are exactly the pairs closer than ~1.
    touching = len(cKDTree(slots).query_pairs(1.0 + 1e-9))
    return float(touching / n_agents)


def degraded_coverage(
    n_agents: int,
    sigma: float,
    r: float,
    *,
    loss: float = 0.0,
    common_sigma: float = 0.0,
    common_bias: float = 0.0,
    jitter_sigma: float = 0.0,
) -> float:
    """Expected coverage of a hex-packed mesh of ``n_agents`` under the four
    degradations in the module docstring. Reduces to
    :func:`~dronesim.ml.swarm.coverage_laws.planar_coverage` when all are zero.

    Lost agents are assumed to leave the mesh *without* the survivors
    re-forming: that is the case when the loss is not observed, e.g. a jammed
    link. :func:`reformed_coverage` is the alternative.
    """
    if not 0.0 <= loss <= 1.0:
        raise ValueError("loss must lie in [0, 1]")
    sigma_eff2 = sigma * sigma + common_sigma**2 + jitter_sigma**2
    rho2 = n_agents * r * r / ETA2
    enclosed = ncx2.cdf(rho2 / sigma_eff2, 2, common_bias**2 / sigma_eff2)
    covered = (1.0 - loss) * ETA2 * enclosed
    if jitter_sigma <= 0.0:
        return float(covered)
    # Each surviving touching pair loses its expected lens, weighted by the
    # density where the lens forms. Weighting by mass rather than counting
    # pairs matters when the error is narrow next to the mesh: the mass then
    # sits on interior discs, which have six neighbours, not the average.
    pair_mass = _pair_mass(int(n_agents), float(np.sqrt(sigma_eff2)), r, common_bias)
    lost = (1.0 - loss) ** 2 * pair_overlap_fraction(jitter_sigma, r) * pair_mass
    return float(covered - lost)


def _pair_mass(n_agents: int, sigma_eff: float, r: float, bias: float) -> float:
    """Sum over touching pairs of ``pi*r^2`` times the target density at the
    pair's midpoint — the mass a full lens-sized overlap there would cost."""
    if n_agents < 2:
        return 0.0
    slots = hex_lattice_slots(n_agents, spacing=2.0 * r, centre=np.zeros(2))
    pairs = np.array(sorted(cKDTree(slots).query_pairs(2.0 * r * (1.0 + 1e-9))))
    if len(pairs) == 0:
        return 0.0
    mid = 0.5 * (slots[pairs[:, 0]] + slots[pairs[:, 1]]) - np.array([bias, 0.0])
    density = np.exp(-np.sum(mid * mid, axis=1) / (2.0 * sigma_eff**2)) / (2.0 * np.pi * sigma_eff**2)
    return float(np.pi * r * r * np.sum(density))


def reformed_coverage(n_agents: int, sigma: float, r: float, loss: float) -> float:
    """Expected coverage when survivors *do* learn of the losses and re-pack
    into a smaller hex mesh centred on the mode.

    The surviving count is binomial, so this is the planar law averaged over
    it. The ceiling returns to ``eta2``: the gap to :func:`degraded_coverage`
    is what knowing about the losses — i.e. a working link — is worth.
    """
    k = np.arange(n_agents + 1)
    weights = binom.pmf(k, n_agents, 1.0 - loss)
    return float(np.sum(weights * planar_coverage(k, sigma, r)))


def degraded_ceiling(r: float, *, loss: float = 0.0, jitter_sigma: float = 0.0) -> float:
    """Coverage no mesh size can exceed under ``loss`` and ``jitter_sigma``.

    Shared errors (``common_sigma``, ``common_bias``) do not appear: they cost
    agents, not ceiling, since a larger mesh still reaches the mode.
    """
    f = pair_overlap_fraction(jitter_sigma, r)
    return float(ETA2 * (1.0 - loss) * (1.0 - (1.0 - loss) * INTERIOR_PAIRS_PER_AGENT * f))


def degraded_mesh_size(
    p: float,
    sigma: float,
    r: float,
    *,
    loss: float = 0.0,
    common_sigma: float = 0.0,
    jitter_sigma: float = 0.0,
) -> float:
    """Agents needed for coverage ``p`` under unbiased degradation, inverting
    :func:`degraded_coverage` in closed form.

    Uses the interior pair count for every agent, which overstates overlap at
    small ``N`` and so errs towards more agents. Returns ``inf`` at or above
    :func:`degraded_ceiling`.
    """
    ceiling = degraded_ceiling(r, loss=loss, jitter_sigma=jitter_sigma)
    if p >= ceiling:
        return float("inf")
    sigma_eff2 = sigma * sigma + common_sigma**2 + jitter_sigma**2
    return float(-2.0 * ETA2 * sigma_eff2 / (r * r) * np.log(1.0 - p / ceiling))


def drift_sigma(distance, drift_fraction: float) -> np.ndarray | float:
    """Per-axis navigation std after flying ``distance`` without an absolute
    fix, for a dead-reckoning drift quoted as a fraction of distance travelled.

    Drift figures are quoted as a radial error, so the per-axis std is that
    divided by ``sqrt(2)``. Linear growth is the usual spec-sheet convention
    for visual-inertial odometry, not a law of nature.
    """
    d = np.asarray(distance, dtype=np.float64)
    return drift_fraction * d / np.sqrt(2.0)

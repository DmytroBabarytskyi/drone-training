"""Closed-form sizing and timing laws for a mesh covering an uncertain
arrival point — the analysis behind the preprint in ``paper/``.

These are the bounds that hold *before* any placement policy is chosen, so
they belong next to the prototype rather than inside it: nothing in the live
engagement loop calls them, but every sizing decision the prototype embodies
(how many drones, what spacing, which commit time) is an instance of them.

Keeping them here rather than in the experiment scripts means a validation
run exercises the same implementation a future sizing tool would, instead of
a parallel copy of the algebra that can drift away from it.

Symbols follow the paper:

``eta2``   densest packing of equal circles in the plane, pi/sqrt(12).
``eta3``   densest packing of equal spheres, pi/sqrt(18) (Kepler).
``theta2`` thinnest *covering* of the plane by equal circles, 2*pi/sqrt(27).
``sigma``  per-axis standard deviation of the predicted arrival point.
``r``      coverage radius of one agent.
``s``      minimum separation between two agents.

The packing laws below assume ``s >= 2r``, so coverage discs are disjoint.
When the platform tolerates ``s < 2r`` the arrangement is a covering instead
and :func:`covering_coverage` applies, which has no ceiling; the two regimes
cross at :func:`packing_covering_crossover`.
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import brentq
from scipy.special import erf

__all__ = [
    "ETA2",
    "ETA3",
    "THETA2",
    "planar_coverage",
    "planar_mesh_size",
    "volumetric_coverage",
    "volumetric_mesh_size",
    "covering_coverage",
    "covering_mesh_size",
    "packing_covering_crossover",
    "area_ratio_mesh_size",
    "reach_distance",
    "manoeuvre_sigma",
    "formation_half_extent",
    "blanket_mesh_size",
    "blanket_capacity",
]

ETA2 = float(np.pi / np.sqrt(12.0))      # 0.9068996...
ETA3 = float(np.pi / np.sqrt(18.0))      # 0.7404804...
THETA2 = float(2.0 * np.pi / np.sqrt(27.0))  # 1.2091995...


# ---------------------------------------------------------------- planar law


def planar_coverage(n_agents, sigma: float, r: float) -> np.ndarray | float:
    """Probability that an isotropic-Gaussian draw lands within ``r`` of one of
    ``n_agents`` hexagonally packed agents centred on the mode.

    ``N`` touching discs of radius ``r`` occupy a roughly circular region of
    area ``N*pi*r^2/eta2``, i.e. out to ``rho = r*sqrt(N/eta2)``. A fraction
    ``eta2`` of that region is covered and the enclosed Gaussian mass is
    ``1 - exp(-rho^2/(2*sigma^2))``, giving equation (6) of the paper::

        p(N) = eta2 * (1 - exp(-N*r^2 / (2*eta2*sigma^2)))

    The limit is ``eta2``: no arrangement of disjoint discs covers more than
    90.69% of the density, at any ``N``.
    """
    n = np.asarray(n_agents, dtype=np.float64)
    return ETA2 * (1.0 - np.exp(-n * r * r / (2.0 * ETA2 * sigma * sigma)))


def planar_mesh_size(p: float, sigma: float, r: float) -> float:
    """Agents needed for coverage ``p`` — equation (7), the inverse of
    :func:`planar_coverage`.

    Returns ``inf`` for any target at or above the packing ceiling ``eta2``,
    which is the behaviour an area-ratio rule famously lacks: it returns a
    finite count for targets that are unreachable at every mesh size.
    """
    if p >= ETA2:
        return float("inf")
    return float(-2.0 * ETA2 * sigma * sigma / (r * r) * np.log(1.0 - p / ETA2))


# ------------------------------------------------------------ volumetric law


def volumetric_coverage(n_agents, sigma: float, r: float) -> np.ndarray | float:
    """Volumetric analogue of :func:`planar_coverage` — equation (8).

    Two substitutions carry the derivation over: ``eta3`` for the densest
    sphere packing, and the chi distribution with three degrees of freedom
    for the enclosed mass of an isotropic 3-D Gaussian::

        F3(rho; sigma) = erf(rho/(sigma*sqrt(2)))
                         - sqrt(2/pi) * (rho/sigma) * exp(-rho^2/(2*sigma^2))

    with ``rho = r*(N/eta3)^(1/3)`` and ``p3(N) = eta3*F3``. The ceiling is
    ``eta3`` = 74.05%.
    """
    n = np.asarray(n_agents, dtype=np.float64)
    rho = r * np.cbrt(n / ETA3)
    u = rho / sigma
    f3 = erf(u / np.sqrt(2.0)) - np.sqrt(2.0 / np.pi) * u * np.exp(-0.5 * u * u)
    return ETA3 * f3


def volumetric_mesh_size(p: float, sigma: float, r: float, n_max: float = 1e9) -> float:
    """Agents needed for volumetric coverage ``p``, inverting
    :func:`volumetric_coverage` numerically.

    Returns ``inf`` for any target at or above the sphere-packing ceiling
    ``eta3`` = 74.05%.
    """
    if p >= ETA3:
        return float("inf")
    return float(brentq(lambda n: volumetric_coverage(n, sigma, r) - p, 1e-9, n_max))


# ---------------------------------------------------------- covering regime


def covering_coverage(n_agents, sigma: float, r: float) -> np.ndarray | float:
    """Coverage when overlap is permitted (``s < 2r``) — equation (9).

    The arrangement is then a *covering*, not a packing. Kershner's thinnest
    covering of the plane by equal circles has density ``theta2 > 1``, so
    overlap wastes area and ``N`` discs blanket a region of area only
    ``N*pi*r^2/theta2`` — but the blanket has no gaps, so there is no leading
    constant and no ceiling::

        p_cov(N) = 1 - exp(-N*r^2 / (2*theta2*sigma^2))
    """
    n = np.asarray(n_agents, dtype=np.float64)
    return 1.0 - np.exp(-n * r * r / (2.0 * THETA2 * sigma * sigma))


def covering_mesh_size(p: float, sigma: float, r: float) -> float:
    """Agents needed for coverage ``p`` in the covering regime. No ceiling."""
    if not 0.0 <= p < 1.0:
        raise ValueError("p must lie in [0, 1)")
    return float(-2.0 * THETA2 * sigma * sigma / (r * r) * np.log(1.0 - p))


def packing_covering_crossover() -> float:
    """Coverage ``p*`` at which covering overtakes packing — equation (10).

    Solves ``planar_coverage(u) == covering_coverage(u)`` in the scale-free
    variable ``u = N*r^2/(2*sigma^2)``. Below ``p*`` the packing law is the
    right tool and ``eta2`` is a genuine bound; above it, insisting on
    disjoint discs is simply a worse arrangement.
    """
    def gap(u: float) -> float:
        packing = ETA2 * (1.0 - np.exp(-u / ETA2))
        covering = 1.0 - np.exp(-u / THETA2)
        return packing - covering

    u_star = brentq(gap, 1e-6, 50.0, xtol=1e-12, rtol=1e-14)
    return float(ETA2 * (1.0 - np.exp(-u_star / ETA2)))


# ------------------------------------------------------- the rule it replaces


def area_ratio_mesh_size(m: float, sigma: float, r: float) -> float:
    """The naive area ratio ``N = (m*sigma)^2 / r^2``, kept here so the audit
    in the paper compares against a real implementation rather than a
    paraphrase. It tacitly assumes the discs tile the plane perfectly, which
    they cannot.
    """
    return float((m * sigma) ** 2 / (r * r))


# ------------------------------------------------------------ timing / transit


def reach_distance(T, v_max: float, a: float) -> np.ndarray | float:
    """Distance an agent covers in ``T`` seconds starting *and* ending at rest
    — equation (11).

    An agent that must arrive and hold station brakes as well as accelerates,
    so it spends ``T/2`` accelerating and ``T/2`` decelerating::

        d(T) = a*T^2/4                     for T <= 2*v_max/a
             = v_max*T - v_max^2/a         otherwise

    The familiar ``v_max*T - v_max^2/(2*a)`` overstates the reach by one
    ramp's worth.
    """
    t = np.asarray(T, dtype=np.float64)
    t_switch = 2.0 * v_max / a
    return np.where(t <= t_switch, a * t * t / 4.0, v_max * t - v_max * v_max / a)


def manoeuvre_sigma(T, g_sigma: float, g: float = 9.81) -> np.ndarray | float:
    """Displacement of the crossing point from unmodelled lateral acceleration.

    The track fit assumes a straight line; an acceleration ``alpha`` displaces
    the crossing point by ``alpha*T^2/2``. Parameterised by a per-axis standard
    deviation in units of ``g`` — a deliberately crude stand-in for the
    correlated-acceleration models of the tracking literature, adequate
    because the conclusion turns on which term dominates rather than on the
    exact constant.
    """
    t = np.asarray(T, dtype=np.float64)
    return 0.5 * g_sigma * g * t * t


def formation_half_extent(n_agents, r: float) -> np.ndarray | float:
    """Half-extent ``L = r*sqrt(N/eta2)`` of a hex-packed formation of ``N``
    agents — the radius out to which the packed region reaches.
    """
    n = np.asarray(n_agents, dtype=np.float64)
    return r * np.sqrt(n / ETA2)


def blanket_mesh_size(p: float, corridor_half_width: float, r: float) -> float:
    """Agents needed to blanket a fraction ``p`` of a corridor of radius ``C``
    with disjoint discs: ``N = p*C^2/r^2``.

    This is the static alternative to repositioning — it needs no prediction,
    no commit decision and no transit.
    """
    return float(p * corridor_half_width ** 2 / (r * r))


def blanket_capacity(corridor_half_width: float, r: float) -> float:
    """Most disjoint discs a corridor of radius ``C`` can hold, asymptotically:
    ``eta2*C^2/r^2``. Fewer once the boundary is accounted for, so a target
    above this capacity is unreachable statically however many agents exist.
    """
    return float(ETA2 * corridor_half_width ** 2 / (r * r))

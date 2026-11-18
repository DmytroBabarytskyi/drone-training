"""Cooperative navigation for an interceptor mesh — the information bound and
how its error maps onto the degradation laws of ``ew_laws.py``.

Each agent holds its lattice slot in its *estimated* frame, so its coverage
disc sits wherever the estimate is wrong. Four sources feed the estimate:

``M1``  own navigation: a prior on every agent with independent per-axis
        error ``sigma_n`` (dead-reckoning drift).
``M2``  ranging between neighbours, noise ``sigma_r`` per range. Ranges are
        unchanged by moving or turning the whole mesh, so they say nothing
        about the two translations and the rotation: they fix *shape* only.
``M3``  the ground sensor that already tracks the target. Either it sees only
        the mesh *centroid* (``centroid_sigma``) or it resolves every agent
        (``agent_sigma``); both per fix, with ``n_fixes`` fixes.
``M4``  absolute anchors: a few agents with an absolute fix (``anchor_sigma``).

The bound is the linearised information matrix ``J``; ``P = J^-1`` is the
Cramer-Rao covariance of the 2N position errors. It is split into the three
modes that cost a mesh differently:

``translation``  per-axis std of the centroid error. The whole mesh moves:
                 the shared-error channel of ``ew_laws`` — costs agents only.
``rotation``     std of the rotation about the centroid, in radians. For an
                 isotropic arrival error centred on the mesh it costs nothing.
``shape``        per-axis RMS of what is left. Neighbours that were touching
                 now overlap: the independent-drift channel — costs ceiling.

What the overlap term of ``ew_laws`` needs is not the marginal shape error but
the error of each touching pair *along the line joining it*, which ranging
measures directly and therefore keeps small even when the marginal is not.
:class:`ErrorModes` carries both, and :func:`coop_coverage` feeds each to the
term that uses it.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.linalg import cho_factor, cho_solve
from scipy.spatial import cKDTree

from dronesim.ml.swarm.ew_laws import degraded_coverage

__all__ = [
    "ErrorModes",
    "touching_pairs",
    "thin_edges",
    "information_matrix",
    "error_modes",
    "coop_coverage",
    "floppy_fraction",
]


@dataclass(frozen=True)
class ErrorModes:
    """Cramer-Rao error of a mesh, split by what each part costs."""

    translation: float  # per-axis std of the centroid, m
    rotation: float     # std of the rotation about the centroid, rad
    shape: float        # per-axis RMS of the non-rigid remainder, m
    pair: float         # per-axis-equivalent std of touching pairs along their axis, m
    marginal: float     # per-axis RMS of one agent's error, rotation excluded, m


def touching_pairs(slots: np.ndarray, spacing: float) -> np.ndarray:
    """Index pairs of lattice slots exactly ``spacing`` apart, shape (E, 2)."""
    pairs = cKDTree(slots).query_pairs(spacing * (1.0 + 1e-9))
    return np.array(sorted(pairs), dtype=np.int64).reshape(-1, 2)


def thin_edges(edges: np.ndarray, keep: float, rng: np.random.Generator) -> np.ndarray:
    """Keep each ranging edge independently with probability ``keep`` — a
    jammer that silences links at random."""
    if keep >= 1.0:
        return edges
    return edges[rng.random(len(edges)) < keep]


def _edge_rows(slots: np.ndarray, edges: np.ndarray) -> np.ndarray:
    """Rigidity matrix: one row per edge, ``-u`` on agent i and ``+u`` on j."""
    n = len(slots)
    rows = np.zeros((len(edges), 2 * n))
    if len(edges) == 0:
        return rows
    u = slots[edges[:, 1]] - slots[edges[:, 0]]
    u /= np.linalg.norm(u, axis=1, keepdims=True)
    idx = np.arange(len(edges))
    for k in range(2):
        rows[idx, 2 * edges[:, 0] + k] = -u[:, k]
        rows[idx, 2 * edges[:, 1] + k] = u[:, k]
    return rows


def information_matrix(
    slots: np.ndarray,
    edges: np.ndarray,
    *,
    sigma_n: float,
    sigma_r: float,
    centroid_sigma: float | None = None,
    agent_sigma: float | None = None,
    n_fixes: int = 1,
    anchors: tuple[int, ...] = (),
    anchor_sigma: float | None = None,
) -> np.ndarray:
    """Fisher information of the 2N agent positions under M1-M4.

    Linearised about the lattice: a range only informs the displacement
    along its own edge. The M1 prior keeps ``J`` invertible however many
    edges are lost, so every mode stays bounded by ``sigma_n``.
    """
    n = len(slots)
    info = np.eye(2 * n) / sigma_n**2
    if len(edges):
        rows = _edge_rows(slots, edges)
        info += rows.T @ rows / sigma_r**2
    if centroid_sigma is not None:
        # z = mean(x) + noise: information n_fixes/sigma^2 on the centroid.
        t = np.tile(np.eye(2), (n, 1)) / n
        info += n_fixes * (t @ t.T) / centroid_sigma**2
    if agent_sigma is not None:
        info += n_fixes * np.eye(2 * n) / agent_sigma**2
    if anchors:
        if anchor_sigma is None:
            raise ValueError("anchors need anchor_sigma")
        for k in anchors:
            info[2 * k:2 * k + 2, 2 * k:2 * k + 2] += np.eye(2) / anchor_sigma**2
    return info


def _rigid_basis(slots: np.ndarray) -> np.ndarray:
    """Orthonormal basis of the two translations and the rotation about the
    centroid, shape (2N, 3)."""
    n = len(slots)
    c = slots - slots.mean(axis=0)
    tx = np.tile([1.0, 0.0], n)
    ty = np.tile([0.0, 1.0], n)
    rot = np.column_stack([-c[:, 1], c[:, 0]]).ravel()
    q, _ = np.linalg.qr(np.column_stack([tx, ty, rot]))
    return q


def error_modes(slots: np.ndarray, info: np.ndarray, spacing: float) -> ErrorModes:
    """Split the Cramer-Rao covariance ``info^-1`` into the modes above."""
    n = len(slots)
    cov = cho_solve(cho_factor(info), np.eye(2 * n))

    centroid = np.tile(np.eye(2), (n, 1)) / n
    translation = float(np.sqrt(np.trace(centroid.T @ cov @ centroid) / 2.0))

    c = slots - slots.mean(axis=0)
    rot = np.column_stack([-c[:, 1], c[:, 0]]).ravel()
    rot_op = rot / (rot @ rot)  # least-squares rotation angle from displacements
    rotation = float(np.sqrt(rot_op @ cov @ rot_op))

    basis = _rigid_basis(slots)
    rigid_part = basis.T @ cov @ basis
    shape = float(np.sqrt(max(np.trace(cov) - np.trace(rigid_part), 0.0) / (2.0 * (n - 1.5))))

    pairs = touching_pairs(slots, spacing)
    if len(pairs):
        rows = _edge_rows(slots, pairs)
        pair_var = np.einsum("ij,jk,ik->i", rows, cov, rows)
        pair = float(np.sqrt(pair_var.mean() / 2.0))
    else:
        pair = 0.0
    # Rotation excluded: tr((I-Q) P (I-Q)) = tr(P) - rhat' P rhat for Q = rhat rhat'.
    rhat = rot / np.linalg.norm(rot)
    marginal = float(np.sqrt((np.trace(cov) - rhat @ cov @ rhat) / (2.0 * n)))
    return ErrorModes(translation, rotation, shape, pair, marginal)


def coop_coverage(
    n_agents: int,
    sigma: float,
    r: float,
    modes: ErrorModes,
    *,
    loss: float = 0.0,
    frame_bias_sigma: float = 0.0,
    common_bias: float = 0.0,
) -> float:
    """Coverage of a mesh navigating with error ``modes``, through ``ew_laws``.

    Rotation is dropped: turning the mesh about its centroid leaves its
    coverage of an isotropic, centred arrival error unchanged. The overlap
    term uses the pair error; the widening of each disc's mass uses the
    marginal error, which the remainder hands over as shared spread so that
    ``sigma_eff`` comes out right.

    ``frame_bias_sigma`` is the error between the frame the mesh navigates in
    and the frame the target is predicted in — zero when both come from the
    same sensor, the sensor's registration error when the mesh uses absolute
    anchors instead.
    """
    shared2 = modes.marginal**2 - modes.pair**2 + frame_bias_sigma**2
    return degraded_coverage(
        n_agents, sigma, r,
        loss=loss,
        common_sigma=float(np.sqrt(max(shared2, 0.0))),
        common_bias=common_bias,
        jitter_sigma=modes.pair,
    )


def floppy_fraction(slots: np.ndarray, edges: np.ndarray) -> float:
    """Share of the 2N degrees of freedom that the ranges leave unconstrained,
    beyond the three rigid motions — zero for a rigid mesh."""
    n = len(slots)
    if len(edges) == 0:
        return float((2 * n - 3) / (2 * n))
    rank = np.linalg.matrix_rank(_edge_rows(slots, edges))
    return float(max(2 * n - 3 - rank, 0) / (2 * n))

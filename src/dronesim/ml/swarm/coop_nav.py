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

The second half is the mechanism itself, to check the bound against:
:func:`observe` draws one engagement's measurements (with a coherent spoof of
M1, uplink loss of M3 fixes and lying anchors as options),
:func:`gauss_newton` fits them, :func:`robust_estimate` adds chi-square fault
exclusion, :func:`place_mesh` runs range-fit-move rounds and returns where the
agents really end up, and :func:`spoofed_coverage` is the matching
prediction under spoofing.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.linalg import cho_factor, cho_solve
from scipy.spatial import cKDTree
from scipy.stats import binom, ncx2
from scipy.stats import chi2 as chi2_dist

from dronesim.ml.swarm.ew_laws import degraded_coverage

__all__ = [
    "ESTIMATORS",
    "ErrorModes",
    "Estimate",
    "NavConfig",
    "Observation",
    "gauss_newton",
    "observe",
    "place_mesh",
    "placement_coverage",
    "robust_estimate",
    "spoofed_coverage",
    "touching_pairs",
    "thin_edges",
    "information_matrix",
    "error_modes",
    "empirical_modes",
    "modes_from_cov",
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
    spoof_nuisance: bool = False,
) -> np.ndarray:
    """Fisher information of the 2N agent positions under M1-M4.

    Linearised about the lattice: a range only informs the displacement
    along its own edge. The M1 prior keeps ``J`` invertible however many
    edges are lost, so every mode stays bounded by ``sigma_n``.

    ``spoof_nuisance`` treats a shared offset of all M1 fixes as unknown and
    marginalises it out: own navigation then says nothing about translation,
    which must come from M3 or M4, or ``J`` is singular.
    """
    n = len(slots)
    info = np.eye(2 * n) / sigma_n**2
    if spoof_nuisance:
        # Schur complement of the offset b: J_xx - J_xb J_bb^-1 J_bx with
        # J_xb = T/sigma_n^2 and J_bb = N I/sigma_n^2.
        t = np.tile(np.eye(2), (n, 1))
        info -= (t @ t.T) / (n * sigma_n**2)
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
    return modes_from_cov(slots, cho_solve(cho_factor(info), np.eye(2 * len(slots))), spacing)


def empirical_modes(slots: np.ndarray, errors: np.ndarray, spacing: float) -> ErrorModes:
    """The same split for a sample of estimation errors, shape (T, N, 2).
    Uses the second moment, not the covariance, so a bias counts as error."""
    e = errors.reshape(len(errors), -1)
    return modes_from_cov(slots, e.T @ e / len(e), spacing)


def modes_from_cov(slots: np.ndarray, cov: np.ndarray, spacing: float) -> ErrorModes:
    """Split a (2N, 2N) error covariance into the modes above."""
    n = len(slots)
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


# ---------------------------------------------------------------------------
# B2: the estimator that is supposed to reach the bound, and the attacks on it
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class NavConfig:
    """What a mesh measures and how well: the sources M1-M4 of the module
    docstring. ``n_fixes`` is the number of M3 fixes sent up; each one is lost
    on the uplink with the probability passed to :func:`observe`."""

    sigma_n: float
    sigma_r: float
    centroid_sigma: float | None = None
    agent_sigma: float | None = None
    n_fixes: int = 1
    anchors: tuple[int, ...] = ()
    anchor_sigma: float | None = None

    @property
    def has_sensor(self) -> bool:
        return self.centroid_sigma is not None or self.agent_sigma is not None

    def info(
        self,
        slots: np.ndarray,
        edges: np.ndarray,
        *,
        n_fixes: int | None = None,
        spoof_nuisance: bool = False,
    ) -> np.ndarray:
        """:func:`information_matrix` for this configuration, with ``n_fixes``
        M3 fixes received (default: all of them)."""
        k = self.n_fixes if n_fixes is None else n_fixes
        return information_matrix(
            slots, edges,
            sigma_n=self.sigma_n, sigma_r=self.sigma_r,
            centroid_sigma=self.centroid_sigma if k > 0 else None,
            agent_sigma=self.agent_sigma if k > 0 else None,
            n_fixes=max(k, 1),
            anchors=self.anchors, anchor_sigma=self.anchor_sigma,
            spoof_nuisance=spoof_nuisance,
        )


@dataclass(frozen=True)
class Observation:
    """One engagement's measurements."""

    own: np.ndarray                 # (N, 2) M1 fixes
    edges: np.ndarray               # (E, 2) ranging edges that worked
    ranges: np.ndarray              # (E,) measured ranges
    fixes: int                      # M3 fixes that got through the uplink
    centroid: np.ndarray | None     # (2,) mean of the centroid fixes received
    agent_fix: np.ndarray | None    # (N, 2) mean of the per-agent fixes received
    anchor_fix: np.ndarray          # (K, 2), one per ``NavConfig.anchors``


@dataclass(frozen=True)
class Estimate:
    """Output of :func:`gauss_newton` / :func:`robust_estimate`."""

    x: np.ndarray                   # (N, 2) estimated positions
    spoof: np.ndarray | None        # estimated shared M1 offset, if modelled
    chi2: float                     # weighted residual sum of squares
    dof: int
    iterations: int
    excluded: tuple = ()            # faults removed: "spoof" and/or anchor slots


def observe(
    truth: np.ndarray,
    edges: np.ndarray,
    cfg: NavConfig,
    rng: np.random.Generator,
    *,
    spoof=None,
    nav_error: np.ndarray | None = None,
    uplink_loss: float = 0.0,
    anchor_bias=None,
) -> Observation:
    """Draw the measurements of a mesh whose agents are truly at ``truth``.

    ``spoof`` is a coherent offset added to every M1 fix (a spoofed absolute
    reference the own navigation leans on); ranges and the ground sensor do
    not see it. ``nav_error`` replaces the whole M1 error, spoof included.
    Each of the ``n_fixes`` M3 fixes is lost with probability ``uplink_loss``.
    ``anchor_bias`` — shape (2,) or (K, 2) — is added to the anchor fixes: a
    map-to-sensor registration error, or anchors that lie.
    """
    n = len(truth)
    if nav_error is None:
        b = np.zeros(2) if spoof is None else np.asarray(spoof, dtype=np.float64)
        nav_error = b + rng.normal(0.0, cfg.sigma_n, size=(n, 2))
    own = truth + nav_error
    if len(edges):
        d = np.linalg.norm(truth[edges[:, 1]] - truth[edges[:, 0]], axis=1)
        ranges = d + rng.normal(0.0, cfg.sigma_r, size=len(d))
    else:
        ranges = np.empty(0)
    fixes = int(rng.binomial(cfg.n_fixes, 1.0 - uplink_loss)) if cfg.has_sensor else 0
    centroid = agent_fix = None
    if fixes and cfg.centroid_sigma is not None:
        centroid = truth.mean(axis=0) + rng.normal(0.0, cfg.centroid_sigma / np.sqrt(fixes), 2)
    if fixes and cfg.agent_sigma is not None:
        agent_fix = truth + rng.normal(0.0, cfg.agent_sigma / np.sqrt(fixes), size=(n, 2))
    k = len(cfg.anchors)
    anchor_fix = truth[list(cfg.anchors)].reshape(k, 2).copy()
    if k:
        if cfg.anchor_sigma is None:
            raise ValueError("anchors need anchor_sigma")
        anchor_fix += rng.normal(0.0, cfg.anchor_sigma, size=(k, 2))
        if anchor_bias is not None:
            anchor_fix += np.asarray(anchor_bias, dtype=np.float64)
    return Observation(own, edges, ranges, fixes, centroid, agent_fix, anchor_fix)


def _has_reference(obs: Observation, anchor_mask: np.ndarray) -> bool:
    """Whether anything besides M1 pins translation."""
    return obs.centroid is not None or obs.agent_fix is not None or bool(anchor_mask.any())


def _system(z, obs, cfg, model_spoof, anchor_mask, with_jac=True):
    """Whitened residuals ``meas - model`` and (unless ``with_jac`` is off)
    the model Jacobian at ``z``."""
    n = len(obs.own)
    p = len(z)
    x = z[:2 * n].reshape(n, 2)
    eye2 = np.eye(2)
    res, jac = [], []

    def block(rows):
        j = np.zeros((rows, p))
        jac.append(j)
        return j

    pred = x + z[2 * n:] if model_spoof else x
    res.append(((obs.own - pred) / cfg.sigma_n).ravel())
    if with_jac:
        j = block(2 * n)
        j[:, :2 * n] = np.eye(2 * n) / cfg.sigma_n
        if model_spoof:
            j[:, 2 * n:] = np.tile(eye2, (n, 1)) / cfg.sigma_n

    e = obs.edges
    if len(e):
        diff = x[e[:, 1]] - x[e[:, 0]]
        d = np.maximum(np.linalg.norm(diff, axis=1), 1e-9)
        res.append((obs.ranges - d) / cfg.sigma_r)
        if with_jac:
            u = diff / d[:, None]
            j = block(len(e))
            idx = np.arange(len(e))
            for k in range(2):
                j[idx, 2 * e[:, 0] + k] = -u[:, k] / cfg.sigma_r
                j[idx, 2 * e[:, 1] + k] = u[:, k] / cfg.sigma_r

    if obs.centroid is not None:
        s = cfg.centroid_sigma / np.sqrt(obs.fixes)
        res.append((obs.centroid - x.mean(axis=0)) / s)
        if with_jac:
            block(2)[:, :2 * n] = np.tile(eye2, (1, n)) / (n * s)

    if obs.agent_fix is not None:
        s = cfg.agent_sigma / np.sqrt(obs.fixes)
        res.append(((obs.agent_fix - x) / s).ravel())
        if with_jac:
            block(2 * n)[:, :2 * n] = np.eye(2 * n) / s

    for k, a in enumerate(cfg.anchors):
        if not anchor_mask[k]:
            continue
        res.append((obs.anchor_fix[k] - x[a]) / cfg.anchor_sigma)
        if with_jac:
            block(2)[:, 2 * a:2 * a + 2] = eye2 / cfg.anchor_sigma

    return np.concatenate(res), (np.vstack(jac) if with_jac else None)


def _solve(z, obs, cfg, model_spoof, mask, max_iter, tol):
    """Plain Gauss-Newton with backtracking from ``z``."""
    r, jac = _system(z, obs, cfg, model_spoof, mask)
    cost = float(r @ r)
    it = 0
    for it in range(1, max_iter + 1):
        dz = np.linalg.solve(jac.T @ jac, jac.T @ r)
        step = 1.0
        while True:
            z_new = z + step * dz
            r_new, jac_new = _system(z_new, obs, cfg, model_spoof, mask)
            cost_new = float(r_new @ r_new)
            if cost_new <= cost or step < 1e-3:
                break
            step *= 0.5
        z, r, jac, cost = z_new, r_new, jac_new, cost_new
        if np.max(np.abs(step * dz)) < tol:
            break
    return z, r, it


def _triangles(edges: np.ndarray) -> list[tuple[int, int, int]]:
    """(i, j, k) with i ranging to j and k, and j to k: agent i can be
    mirrored across the line j-k without changing either of its ranges."""
    nbrs: dict[int, set[int]] = {}
    for a, b in edges:
        nbrs.setdefault(int(a), set()).add(int(b))
        nbrs.setdefault(int(b), set()).add(int(a))
    out = []
    for i, ni in nbrs.items():
        for j in ni:
            for k in ni & nbrs[j]:
                if j < k:
                    out.append((i, j, k))
    return out


def _best_flip(z, cost, triangles, obs, cfg, model_spoof, mask):
    """The single mirror move that lowers the cost most, or ``None``."""
    best_cost, best_z = cost - 1e-6, None
    for i, j, k in triangles:
        xi, xj, xk = z[2 * i:2 * i + 2], z[2 * j:2 * j + 2], z[2 * k:2 * k + 2]
        u = xk - xj
        norm2 = float(u @ u)
        if norm2 < 1e-12:
            continue
        foot = xj + u * float((xi - xj) @ u) / norm2
        z_try = z.copy()
        z_try[2 * i:2 * i + 2] = 2.0 * foot - xi
        r_try, _ = _system(z_try, obs, cfg, model_spoof, mask, with_jac=False)
        c = float(r_try @ r_try)
        if c < best_cost:
            best_cost, best_z = c, z_try
    return best_z


def gauss_newton(
    obs: Observation,
    cfg: NavConfig,
    *,
    model_spoof: bool = False,
    anchor_mask: np.ndarray | None = None,
    x0: np.ndarray | None = None,
    max_iter: int = 50,
    tol: float = 1e-4,
    repair_flips: bool = True,
    max_repairs: int = 10,
) -> Estimate:
    """Maximum-likelihood positions from M1-M4 by Gauss-Newton, started at
    the own-navigation fixes and backtracking whenever a step raises the cost.

    ``model_spoof`` adds a shared offset of all M1 fixes as two extra
    unknowns: own navigation then no longer informs translation, so something
    else (M3 or a live anchor) must, or the call raises. ``anchor_mask``
    switches individual anchors off.

    Ranges are blind to an agent mirrored across the line through two
    neighbours that range to it and to each other, and Gauss-Newton started
    on the wrong side stays there. With ``repair_flips`` every such mirror
    image is tried after convergence, the one that lowers the cost most is
    taken, and the fit restarts from it, up to ``max_repairs`` times.
    """
    n = len(obs.own)
    mask = np.ones(len(cfg.anchors), dtype=bool) if anchor_mask is None else np.asarray(anchor_mask)
    if model_spoof and not _has_reference(obs, mask):
        raise ValueError("a spoofed own navigation leaves translation unobservable "
                         "without M3 or an anchor")
    start = obs.own if x0 is None else x0
    z = np.concatenate([start.ravel(), np.zeros(2)]) if model_spoof else start.ravel().copy()
    z, r, it = _solve(z, obs, cfg, model_spoof, mask, max_iter, tol)
    cost = float(r @ r)
    triangles = _triangles(obs.edges) if repair_flips else []
    for _ in range(max_repairs if repair_flips else 0):
        flipped = _best_flip(z, cost, triangles, obs, cfg, model_spoof, mask)
        if flipped is None:
            break
        z, r, extra = _solve(flipped, obs, cfg, model_spoof, mask, max_iter, tol)
        it += extra
        cost = float(r @ r)
    return Estimate(
        x=z[:2 * n].reshape(n, 2),
        spoof=z[2 * n:].copy() if model_spoof else None,
        chi2=cost,
        dof=len(r) - len(z),
        iterations=it,
    )


def robust_estimate(
    obs: Observation,
    cfg: NavConfig,
    *,
    alpha: float = 1e-3,
    max_faults: int | None = None,
    x0: np.ndarray | None = None,
) -> Estimate:
    """Gauss-Newton with greedy fault exclusion.

    Candidate faults are a coherent spoof of M1 (modelled as an unknown
    offset — needs a reference) and each anchor (dropped). Each round fits
    every candidate added to the faults found so far and keeps the one that
    lowers chi-square most, if it lowers it by more than the ``1 - alpha``
    quantile of chi-square with two degrees of freedom — the drop expected
    from fitting two more parameters to honest data. Without a fault the
    result equals :func:`gauss_newton` except with probability ``~alpha``.
    """
    thr = float(chi2_dist.isf(alpha, 2))
    mask = np.ones(len(cfg.anchors), dtype=bool)
    spoof = False
    best = gauss_newton(obs, cfg, x0=x0)
    excluded: list = []
    limit = len(cfg.anchors) + 1 if max_faults is None else max_faults
    while len(excluded) < limit:
        trials = []
        if not spoof and _has_reference(obs, mask):
            trials.append(("spoof", gauss_newton(obs, cfg, model_spoof=True,
                                                    anchor_mask=mask, x0=x0)))
        for k in np.flatnonzero(mask):
            m = mask.copy()
            m[k] = False
            if spoof and not _has_reference(obs, m):
                continue
            trials.append((int(cfg.anchors[k]),
                           gauss_newton(obs, cfg, model_spoof=spoof, anchor_mask=m, x0=x0)))
        if not trials:
            break
        fault, est = min(trials, key=lambda t: t[1].chi2)
        if best.chi2 - est.chi2 <= thr:
            break
        excluded.append(fault)
        if fault == "spoof":
            spoof = True
        else:
            mask[list(cfg.anchors).index(fault)] = False
        best = est
    return Estimate(best.x, best.spoof, best.chi2, best.dof, best.iterations, tuple(excluded))


ESTIMATORS = ("naive", "robust", "nuisance")


def place_mesh(
    slots: np.ndarray,
    edges: np.ndarray,
    cfg: NavConfig,
    rng: np.random.Generator,
    *,
    estimator: str = "robust",
    spoof=None,
    uplink_loss: float = 0.0,
    anchor_bias=None,
    alpha: float = 1e-3,
    rounds: int = 1,
) -> tuple[np.ndarray, Estimate]:
    """One engagement end to end. Every agent flies to its slot on its own
    navigation, so it truly sits at ``slot - nav_error``; the mesh ranges,
    estimates, and each agent then moves by ``slot - estimate``. Its own
    navigation error is unchanged over that short move, so it ends at
    ``slot - (estimate - truth)``. Returns those final true positions.

    ``estimator``: ``naive`` trusts every source; ``robust`` is
    :func:`robust_estimate`; ``nuisance`` always models a spoof of M1 when a
    reference exists (and is naive otherwise).

    With ``rounds > 1`` the mesh ranges again from where it now stands and
    repeats, every round with fresh ranges, M3 fixes and anchor fixes but the
    same own-navigation error, starting the fit at the slots (where each
    agent now believes it is). The first fit starts at the own fixes, which
    for ``sigma_n`` comparable with the spacing are far enough off the true
    shape to land in a folded local minimum; later rounds range a mesh that
    is nearly the lattice and can climb out of it.
    """
    if estimator not in ESTIMATORS:
        raise ValueError(f"estimator must be one of {ESTIMATORS}")
    n = len(slots)
    b = np.zeros(2) if spoof is None else np.asarray(spoof, dtype=np.float64)
    nav_error = b + rng.normal(0.0, cfg.sigma_n, size=(n, 2))
    truth = slots - nav_error
    est = None
    for k in range(rounds):
        obs = observe(truth, edges, cfg, rng, nav_error=nav_error,
                      uplink_loss=uplink_loss, anchor_bias=anchor_bias)
        x0 = None if k == 0 else slots
        if estimator == "naive":
            est = gauss_newton(obs, cfg, x0=x0)
        elif estimator == "robust":
            est = robust_estimate(obs, cfg, alpha=alpha, x0=x0)
        else:
            ref = _has_reference(obs, np.ones(len(cfg.anchors), dtype=bool))
            est = gauss_newton(obs, cfg, model_spoof=ref, x0=x0)
        truth = slots - (est.x - truth)
    return truth, est


def placement_coverage(
    positions: np.ndarray,
    sigma: float,
    r: float,
    rng: np.random.Generator,
    *,
    n_targets: int = 200,
    chunk: int = 50,
) -> float:
    """Monte Carlo coverage of placed meshes, ``positions`` of shape
    (T, N, 2): ``n_targets`` arrivals ~ N(0, sigma^2 I) per mesh, and the
    share that land within ``r`` of some agent."""
    hits = 0
    for start in range(0, len(positions), chunk):
        pos = positions[start:start + chunk]
        t = rng.normal(0.0, sigma, size=(len(pos), n_targets, 2))
        d2 = np.sum((t[:, :, None, :] - pos[:, None, :, :]) ** 2, axis=-1)
        hits += int(np.sum(d2.min(axis=-1) <= r * r))
    return hits / (len(positions) * n_targets)


def spoofed_coverage(
    slots: np.ndarray,
    edges: np.ndarray,
    cfg: NavConfig,
    sigma: float,
    r: float,
    *,
    spacing: float,
    spoof: float = 0.0,
    uplink_loss: float = 0.0,
    estimator: str = "robust",
    alpha: float = 1e-3,
    frame_bias_sigma: float = 0.0,
) -> float:
    """The prediction "law A + bound" for a coherent spoof of magnitude
    ``spoof`` on M1, averaged over how many of the M3 fixes get through.

    Given ``k`` fixes, a linear-Gaussian fit that trusts M1 keeps the part of
    the spoof that M1 outweighs: ``mean(P T b / sigma_n^2)``, a shared bias.
    The robust test statistic is non-central chi-square with two degrees of
    freedom and non-centrality ``b' C_b^-1 b``, ``C_b`` the covariance of the
    offset estimate; when it fires the mesh navigates with the spoof
    marginalised out (:func:`information_matrix` with ``spoof_nuisance``) and
    no bias. The result mixes the two branches by the detection probability.
    Conditioning on a miss leaves a somewhat smaller bias than assumed, so
    the prediction errs low near the detection threshold.
    """
    n = len(slots)
    t = np.tile(np.eye(2), (n, 1))
    b = np.array([spoof, 0.0])
    thr = float(chi2_dist.isf(alpha, 2))
    ks = np.arange(cfg.n_fixes + 1) if cfg.has_sensor else np.array([0])
    weights = binom.pmf(ks, cfg.n_fixes, 1.0 - uplink_loss) if cfg.has_sensor else np.ones(1)
    total = 0.0
    for k, w in zip(ks, weights):
        if w < 1e-12:
            continue
        info = cfg.info(slots, edges, n_fixes=int(k))
        cov = cho_solve(cho_factor(info), np.eye(2 * n))
        residual_bias = float(np.linalg.norm(t.T @ cov @ (t @ b) / cfg.sigma_n**2 / n))
        trusting = coop_coverage(n, sigma, r, error_modes(slots, info, spacing),
                                 common_bias=residual_bias, frame_bias_sigma=frame_bias_sigma)
        has_ref = (k > 0) or bool(cfg.anchors)
        if estimator == "naive" or not has_ref:
            total += w * trusting
            continue
        if estimator == "nuisance":
            p_detect = 1.0
        else:
            aug = np.block([[info, t / cfg.sigma_n**2],
                            [t.T / cfg.sigma_n**2, n * np.eye(2) / cfg.sigma_n**2]])
            cov_b = np.linalg.inv(aug)[-2:, -2:]
            lam = float(b @ np.linalg.solve(cov_b, b))
            p_detect = float(ncx2.sf(thr, 2, lam)) if lam > 0 else alpha
        nuis = cfg.info(slots, edges, n_fixes=int(k), spoof_nuisance=True)
        detected = coop_coverage(n, sigma, r, error_modes(slots, nuis, spacing),
                                 frame_bias_sigma=frame_bias_sigma)
        total += w * (p_detect * detected + (1.0 - p_detect) * trusting)
    return float(total)

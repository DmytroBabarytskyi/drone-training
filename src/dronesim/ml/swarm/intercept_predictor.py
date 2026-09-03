"""Deterministic prediction of where a fast target will cross the swarm's
plane, from a handful of noisy recon position samples.

This is plain kinematics, not machine learning, and on purpose: predicting
where a ballistic target will be from a few tracking samples is a solved,
well-understood problem (real fire-control systems have done linear and
quadratic extrapolation for decades). There is nothing here for a neural
network to learn that a least-squares fit doesn't already give for free, and
using one would only add noise and training cost to the one part of the
pipeline that does not need it. The learned part of this prototype is
strictly downstream of this module: given this module's prediction, how the
swarm should distribute itself in the plane (see ``swarm_env.py``).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class CueingSample:
    """One recon report: target's approximate 3D position at a point in time.

    Recon does not track the target continuously — it reports a noisy fix
    every so often (``configs/ml/swarm_intercept.yaml: cueing_interval_s``).
    The swarm has to work with these sparse, noisy fixes, not ground truth.
    """

    t: float
    pos: np.ndarray  # shape (3,): (x, y, z)


@dataclass(frozen=True)
class CrossingPrediction:
    """Where and when the target is predicted to cross the swarm's plane."""

    point: np.ndarray  # shape (2,): (y, z) in-plane coordinates
    time_to_cross: float  # seconds from "now" (the last sample's time)
    confidence_radius: float  # 1-sigma-ish radius of uncertainty around `point`


def predict_plane_crossing(
    samples: list[CueingSample],
    plane_x: float,
    sensor_noise_std: float,
) -> CrossingPrediction | None:
    """Fit a straight-line trajectory through `samples` and find where it
    crosses `x = plane_x`.

    Returns ``None`` if there are too few samples to fit a line, or if the
    fitted trajectory is not actually heading toward the plane (e.g. moving
    parallel to it, or away from it) — in both cases there is nothing
    meaningful to predict yet, and the caller should wait for more cueing
    or treat the target as not currently a threat.
    """
    if len(samples) < 2:
        return None

    t = np.array([s.t for s in samples], dtype=np.float64)
    pos = np.stack([s.pos for s in samples]).astype(np.float64)

    # Degree-1 least-squares fit per axis: pos_axis(t) = slope * t + intercept.
    # `np.polyfit` needs the samples spread out in time, not stacked at one
    # instant, or the fit is degenerate — guard that explicitly rather than
    # letting numpy silently return garbage.
    if float(np.ptp(t)) < 1e-6:
        return None

    fits = [np.polyfit(t, pos[:, axis], deg=1) for axis in range(3)]
    slope = np.array([f[0] for f in fits])
    intercept = np.array([f[1] for f in fits])

    if abs(slope[0]) < 1e-6:
        return None  # not moving along x at all — no well-defined crossing time

    t_now = float(t[-1])
    t_cross = (plane_x - intercept[0]) / slope[0]
    time_to_cross = t_cross - t_now
    if time_to_cross <= 0.0:
        return None  # the fitted line crosses the plane in the past, not ahead

    crossing_xyz = intercept + slope * t_cross
    point = crossing_xyz[1:3]

    confidence_radius = _propagate_crossing_uncertainty(
        t, sensor_noise_std, slope, t_cross
    )

    return CrossingPrediction(
        point=point, time_to_cross=float(time_to_cross), confidence_radius=confidence_radius
    )


def _propagate_crossing_uncertainty(
    t: np.ndarray, sensor_noise_std: float, slope: np.ndarray, t_cross: float
) -> float:
    """Standard-error of the predicted crossing point, propagated from the
    known per-sample sensor noise through the OLS line fit and through the
    fact that ``t_cross`` itself is an estimate (delta method), instead of
    the ad hoc "more samples, less time = tighter" heuristic this replaced.

    That heuristic (``sensor_noise_std / sqrt(n) + 0.15 * sensor_noise_std *
    time_to_cross``) got the *direction* of every effect right but not the
    *size*: an evaluation run showed the true crossing point landing inside
    the reported radius only 10% of the time, with the actual RMS error
    running about 3x the heuristic's number. The gap is this formula's
    missing term: error in the fitted x-velocity does not just blur the
    crossing time a little, it gets multiplied by ``t_cross`` (often several
    seconds) before it ever reaches the y/z prediction, so a small slope
    error compounds into a large position error at long range — exactly the
    kind of effect a hand-picked coefficient cannot capture but a variance
    propagation does automatically.

    For a line ``value = a + b*t`` fit by OLS to samples with i.i.d. noise
    ``sensor_noise_std``, the standard results are::

        Sxx = sum((t_i - mean(t))^2)
        Var(b)      = sigma^2 / Sxx
        Var(a)      = sigma^2 * (1/n + mean(t)^2 / Sxx)
        Cov(a, b)   = -sigma^2 * mean(t) / Sxx

    Applied twice: once to the x-axis fit, to get ``Var(t_cross)`` via the
    delta method on ``t_cross = (plane_x - a_x) / b_x``; once to each of the
    y/z-axis fits, to get their direct variance at ``t = t_cross``, plus the
    extra ``slope_axis^2 * Var(t_cross)`` term from not knowing ``t_cross``
    exactly. The x-axis noise and the y/z-axis noise are independent samples
    in this simulation, so there is no cross-term between them to add.
    """
    n = len(t)
    t_mean = float(np.mean(t))
    sxx = float(np.sum((t - t_mean) ** 2))
    sigma2 = sensor_noise_std**2

    # OLS variance/covariance of the fitted intercept and slope — the same
    # for every axis, since all three use the same sample times `t` and the
    # same noise level (homoscedastic by construction: identical Gaussian
    # noise is added to x, y and z in `_simulate_target_and_cueing`).
    var_a = sigma2 * (1.0 / n + t_mean**2 / sxx)
    var_b = sigma2 / sxx
    cov_ab = -sigma2 * t_mean / sxx

    def axis_variance_at(t_query: float) -> float:
        """Var(a + b*t_query) for the fitted line, at a possibly-uncertain t_query."""
        return var_a + (t_query**2) * var_b + 2.0 * t_query * cov_ab

    # Var(t_cross) via the delta method on t_cross = (plane_x - a_x) / slope_x:
    # d(t_cross)/d(a_x) = -1/slope_x, d(t_cross)/d(b_x) = -t_cross/slope_x.
    d_dt_a, d_dt_b = -1.0 / slope[0], -t_cross / slope[0]
    var_t_cross = d_dt_a**2 * var_a + d_dt_b**2 * var_b + 2.0 * d_dt_a * d_dt_b * cov_ab

    # Each in-plane axis (y, z) has its own direct fit variance at t=t_cross,
    # plus an extra term from t_cross itself being uncertain: to first order,
    # value(t_cross) shifts by slope_axis * delta(t_cross).
    total_variance = sum(
        axis_variance_at(t_cross) + (slope[axis] ** 2) * var_t_cross for axis in (1, 2)
    )
    return float(np.sqrt(max(total_variance, 0.0)))

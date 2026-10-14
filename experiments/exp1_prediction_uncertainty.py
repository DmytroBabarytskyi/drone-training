"""Experiment 1 — Table I and Figure 1: propagated prediction uncertainty.

Validates the closed form of Section IV against Monte Carlo over noisy
redraws of a fixed true trajectory, on a 4x4 grid of fix counts and lead
times, and measures its calibration.

The radial uncertainty ``R`` is NOT recomputed here: it comes from
``intercept_predictor._propagate_crossing_uncertainty``, the same function
the live engagement loop calls. The only thing this script adds is the
Monte Carlo the closed form is checked against.

Scenario. A point travels on a straight line at a fixed velocity and crosses
the plane ``x = 0``. The grid is parameterised by ``(n, t_c)`` rather than by
``(n, standoff)``: holding the standoff fixed while adding fixes consumes the
whole approach and leaves nothing to extrapolate, which confounds the two
effects under test. The standoff is solved for instead, so every cell is a
valid geometry.

The velocity carries a lateral component on purpose. The ``b_k^2 Var(t_c)``
term of equation (5) vanishes identically for a point travelling exactly
along the x axis, so a validation scenario built without lateral velocity
would not exercise the very term the heuristic omits.
"""

from __future__ import annotations

import numpy as np

from _common import Experiment, check, table

from dronesim.ml.swarm.intercept_predictor import (
    CueingSample,
    _propagate_crossing_uncertainty,
    predict_plane_crossing,
)

# --- scenario ---------------------------------------------------------------
SIGMA_S = 4.0            # per-axis sensor noise, m (configs/ml/swarm_intercept.yaml)
FIX_INTERVAL_S = 0.4     # cueing_interval_s from the same config
PLANE_X = 0.0
VELOCITY = np.array([-130.0, 10.0, 10.0])  # m/s; |v| = 130.8, lateral 10.9% of closing
TRIALS_PER_CELL = 4_000

FIX_COUNTS = (3, 5, 8, 12)
LEAD_TIMES = (0.5, 1.0, 2.0, 4.0)

# Published Table I, keyed by (n, t_c): closed-form R, Monte Carlo R,
# hand-tuned heuristic, calibration.
PUBLISHED = {
    (3, 0.5): (9.60, 9.58, 2.61, 63.2),
    (3, 1.0): (14.42, 14.32, 2.91, 64.1),
    (3, 2.0): (24.30, 24.53, 3.51, 63.0),
    (3, 4.0): (44.26, 44.63, 4.71, 63.1),
    (5, 0.5): (6.36, 6.44, 2.09, 62.4),
    (5, 1.0): (8.46, 8.49, 2.39, 62.4),
    (5, 2.0): (12.81, 12.78, 2.99, 62.8),
    (5, 4.0): (21.68, 21.65, 4.19, 63.1),
    (8, 0.5): (4.62, 4.62, 1.71, 63.3),
    (8, 1.0): (5.62, 5.59, 2.01, 64.1),
    (8, 2.0): (7.71, 7.72, 2.61, 63.3),
    (8, 4.0): (11.99, 12.10, 3.81, 62.7),
    (12, 0.5): (3.60, 3.59, 1.45, 63.3),
    (12, 1.0): (4.13, 4.13, 1.75, 62.1),
    (12, 2.0): (5.24, 5.29, 2.35, 62.2),
    (12, 4.0): (7.53, 7.44, 3.55, 64.7),
}

NOMINAL_CALIBRATION = 63.21  # 1 - exp(-1), the mass inside a radius sigma*sqrt(2)


def heuristic(sigma_s: float, n: int, t_c: float) -> float:
    """The hand-tuned rule the closed form replaced: ``sigma_s/sqrt(n) +
    0.15*sigma_s*t_c``. It captures the direction of every effect and none of
    their sizes."""
    return sigma_s / np.sqrt(n) + 0.15 * sigma_s * t_c


def true_trajectory(n_fixes: int, lead_time: float):
    """Fix times and the true positions at those times, with the standoff
    solved so the crossing falls ``lead_time`` after the last fix."""
    t = np.arange(n_fixes, dtype=np.float64) * FIX_INTERVAL_S
    t_cross = float(t[-1]) + lead_time
    # Position at t: p(t) = p0 + v*t, and p_x(t_cross) = PLANE_X.
    p0 = np.array([PLANE_X - VELOCITY[0] * t_cross, 0.0, 0.0])
    true_pos = p0[None, :] + VELOCITY[None, :] * t[:, None]
    true_crossing = (p0 + VELOCITY * t_cross)[1:3]
    return t, true_pos, t_cross, true_crossing


def run_cell(exp: Experiment, n_fixes: int, lead_time: float, offset: int) -> dict:
    t, true_pos, t_cross, true_crossing = true_trajectory(n_fixes, lead_time)

    # Closed form, straight from the prototype.
    r_closed = _propagate_crossing_uncertainty(t, SIGMA_S, VELOCITY, t_cross)

    rng = exp.rng(offset)
    errors = np.empty(TRIALS_PER_CELL)
    inside = 0
    valid = 0
    for i in range(TRIALS_PER_CELL):
        noisy = true_pos + rng.normal(0.0, SIGMA_S, size=true_pos.shape)
        samples = [CueingSample(t=float(tt), pos=noisy[k]) for k, tt in enumerate(t)]
        pred = predict_plane_crossing(samples, PLANE_X, SIGMA_S)
        if pred is None:
            continue
        err = float(np.linalg.norm(pred.point - true_crossing))
        errors[valid] = err
        # Calibration: does the truth fall inside the radius the predictor
        # itself reported on this trial?
        if err <= pred.confidence_radius:
            inside += 1
        valid += 1

    errors = errors[:valid]
    r_mc = float(np.sqrt(np.mean(errors**2)))
    calib = 100.0 * inside / valid

    pub_r, pub_mc, pub_heur, pub_calib = PUBLISHED[(n_fixes, lead_time)]
    return {
        "n_fixes": n_fixes,
        "lead_time_s": lead_time,
        "t_cross_s": round(t_cross, 3),
        "standoff_m": round(abs(VELOCITY[0]) * t_cross, 1),
        "R_closed_form_m": round(r_closed, 2),
        "R_monte_carlo_m": round(r_mc, 2),
        "err_pct": round(abs(r_closed - r_mc) / r_mc * 100.0, 1),
        "heuristic_m": round(heuristic(SIGMA_S, n_fixes, lead_time), 2),
        "understatement_x": round(r_closed / heuristic(SIGMA_S, n_fixes, lead_time), 1),
        "calibration_pct": round(calib, 1),
        "trials": valid,
        "published": {
            "R_closed_form_m": pub_r,
            "R_monte_carlo_m": pub_mc,
            "heuristic_m": pub_heur,
            "calibration_pct": pub_calib,
        },
    }


def main() -> None:
    exp = Experiment(
        "exp1_prediction_uncertainty",
        "Table I / Figure 1 - propagated prediction uncertainty vs Monte Carlo",
    )

    rows = []
    for i, n in enumerate(FIX_COUNTS):
        for j, t_c in enumerate(LEAD_TIMES):
            rows.append(run_cell(exp, n, t_c, offset=100 * i + j))

    print()
    print(
        table(
            rows,
            [
                ("n_fixes", "n", "{:d}"),
                ("lead_time_s", "t_c (s)", "{:.1f}"),
                ("R_closed_form_m", "R (m)", "{:.2f}"),
                ("R_monte_carlo_m", "MC (m)", "{:.2f}"),
                ("err_pct", "err. (%)", "{:.1f}"),
                ("heuristic_m", "heur. (m)", "{:.2f}"),
                ("calibration_pct", "calib. (%)", "{:.1f}"),
            ],
        )
    )

    worst_err = max(r["err_pct"] for r in rows)
    calib_lo = min(r["calibration_pct"] for r in rows)
    calib_hi = max(r["calibration_pct"] for r in rows)
    under_lo = min(r["understatement_x"] for r in rows)
    under_hi = max(r["understatement_x"] for r in rows)

    print()
    print("  checks against the published table")
    checks = [
        check("worst-cell closed form vs MC", worst_err, 1.3, 0.7, "%"),
        check("calibration floor", calib_lo, 62.1, 1.2, "%"),
        check("calibration ceiling", calib_hi, 64.7, 1.2, "%"),
        check("heuristic understatement, min", under_lo, 2.1, 0.4, "x"),
        check("heuristic understatement, max", under_hi, 9.5, 0.6, "x"),
    ]
    # Every cell's closed form must match the printed value to the printed
    # precision; the Monte Carlo is checked in aggregate above because it is
    # the stochastic half of the comparison.
    cell_deltas = [
        abs(r["R_closed_form_m"] - r["published"]["R_closed_form_m"]) for r in rows
    ]
    checks.append(check("worst closed-form cell vs paper", max(cell_deltas), 0.0, 0.02, " m"))

    exp.write(
        {
            "scenario": {
                "sensor_noise_std_m": SIGMA_S,
                "fix_interval_s": FIX_INTERVAL_S,
                "velocity_mps": VELOCITY.tolist(),
                "speed_mps": round(float(np.linalg.norm(VELOCITY)), 1),
                "lateral_fraction_of_closing": round(
                    float(np.hypot(*VELOCITY[1:]) / abs(VELOCITY[0])), 4
                ),
                "plane_x_m": PLANE_X,
                "trials_per_cell": TRIALS_PER_CELL,
                "nominal_calibration_pct": NOMINAL_CALIBRATION,
            },
            "rows": rows,
            "summary": {
                "worst_cell_err_pct": worst_err,
                "calibration_range_pct": [calib_lo, calib_hi],
                "heuristic_understatement_range_x": [under_lo, under_hi],
            },
            "checks": checks,
            "all_checks_pass": all(c["pass"] for c in checks),
        }
    )


if __name__ == "__main__":
    main()

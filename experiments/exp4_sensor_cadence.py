"""Experiment 4 — Table III: mesh cost against sensor cadence.

Which lever matters. Because ``N`` scales as ``sigma^2/r^2``, halving sigma and
doubling r are equally valuable, and no property of the placement algorithm
changes the exponent. This script prices the sensing lever concretely: the
repositioning window is held fixed so ``t_c`` is constant and only the fix
count varies, which isolates cadence from lead time.

The competing lever — coverage radius — is swept at the end at fixed cadence,
to show it behaving exactly as the exponent dictates.
"""

from __future__ import annotations

import numpy as np

from _common import Experiment, check, table

from dronesim.ml.swarm.coverage_laws import planar_mesh_size
from dronesim.ml.swarm.intercept_predictor import (
    CueingSample,
    _propagate_crossing_uncertainty,
    predict_plane_crossing,
)

# --- scenario (same trajectory family as experiment 1) ----------------------
SIGMA_S = 4.0
R = 3.0
LEAD_TIME_S = 2.0        # repositioning window, held fixed so t_c is constant
TRACK_DURATION_S = 2.230  # fixes are spread over this span, so rate = (n-1)/track
PLANE_X = 0.0
VELOCITY = np.array([-130.0, 10.0, 10.0])
TARGET_P = 0.8
MC_TRIALS = 2_000

FIX_COUNTS = (3, 4, 5, 6, 8, 12, 16, 24)

# Published Table III: n -> (rate Hz, R, MC, sigma, N at p=0.8)
PUBLISHED = {
    3: (0.90, 11.68, 11.66, 8.26, 29.4),
    4: (1.34, 11.00, 10.96, 7.78, 26.1),
    5: (1.79, 10.34, 10.47, 7.31, 23.0),
    6: (2.24, 9.75, 9.76, 6.89, 20.5),
    8: (3.14, 8.79, 8.71, 6.22, 16.7),
    12: (4.93, 7.47, 7.43, 5.28, 12.0),
    16: (6.72, 6.60, 6.57, 4.67, 9.4),
    24: (10.31, 5.50, 5.50, 3.89, 6.5),
}

# Section V-F also quotes the radius lever at fixed cadence.
PUBLISHED_R_LEVER = {1.0: 184, 8.0: 3}


def row_for(exp: Experiment, n_fixes: int, offset: int) -> dict:
    t = np.linspace(0.0, TRACK_DURATION_S, n_fixes)
    t_cross = float(t[-1]) + LEAD_TIME_S
    rate_hz = (n_fixes - 1) / TRACK_DURATION_S

    r_closed = _propagate_crossing_uncertainty(t, SIGMA_S, VELOCITY, t_cross)
    sigma = r_closed / np.sqrt(2.0)

    # Monte Carlo on the same geometry, through the prototype's predictor.
    p0 = np.array([PLANE_X - VELOCITY[0] * t_cross, 0.0, 0.0])
    true_pos = p0[None, :] + VELOCITY[None, :] * t[:, None]
    true_crossing = (p0 + VELOCITY * t_cross)[1:3]
    rng = exp.rng(offset)
    errs = []
    for _ in range(MC_TRIALS):
        noisy = true_pos + rng.normal(0.0, SIGMA_S, size=true_pos.shape)
        pred = predict_plane_crossing(
            [CueingSample(t=float(tt), pos=noisy[k]) for k, tt in enumerate(t)],
            PLANE_X,
            SIGMA_S,
        )
        if pred is not None:
            errs.append(float(np.linalg.norm(pred.point - true_crossing)))
    r_mc = float(np.sqrt(np.mean(np.square(errs))))

    n_needed = planar_mesh_size(TARGET_P, sigma, R)
    pub_rate, pub_r, pub_mc, pub_sigma, pub_n = PUBLISHED[n_fixes]
    return {
        "n_fixes": n_fixes,
        "rate_hz": round(rate_hz, 2),
        "R_closed_form_m": round(r_closed, 2),
        "R_monte_carlo_m": round(r_mc, 2),
        "sigma_m": round(sigma, 2),
        "n_at_target": round(n_needed, 1),
        "published": {
            "rate_hz": pub_rate,
            "R_m": pub_r,
            "MC_m": pub_mc,
            "sigma_m": pub_sigma,
            "n_at_target": pub_n,
        },
    }


def main() -> None:
    exp = Experiment(
        "exp4_sensor_cadence",
        "Table III - mesh cost vs sensor cadence at fixed repositioning window",
    )

    rows = [row_for(exp, n, offset=10 * i) for i, n in enumerate(FIX_COUNTS)]

    print()
    print(
        table(
            rows,
            [
                ("n_fixes", "fixes", "{:d}"),
                ("rate_hz", "rate (Hz)", "{:.2f}"),
                ("R_closed_form_m", "R (m)", "{:.2f}"),
                ("R_monte_carlo_m", "MC (m)", "{:.2f}"),
                ("sigma_m", "sigma (m)", "{:.2f}"),
                ("n_at_target", f"N at p={TARGET_P}", "{:.1f}"),
            ],
        )
    )

    sigma_tighten = rows[0]["sigma_m"] / rows[-1]["sigma_m"]
    print()
    print(
        f"  raising the cadence from {FIX_COUNTS[0]} to {FIX_COUNTS[-1]} fixes tightens "
        f"sigma by {sigma_tighten:.1f}x and takes the mesh from "
        f"{rows[0]['n_at_target']:.0f} agents to {rows[-1]['n_at_target']:.0f}"
    )

    # The competing lever. Quoted in Section V-F at the 6-fix cadence
    # (sigma = 6.89 m), which is what reproduces both endpoints of the
    # published range; the row is named here rather than left implicit.
    R_LEVER_FIXES = 6
    sigma_ref = next(r["sigma_m"] for r in rows if r["n_fixes"] == R_LEVER_FIXES)
    r_lever = {
        r: round(planar_mesh_size(TARGET_P, sigma_ref, r), 1) for r in (1.0, 2.0, 4.0, 8.0)
    }
    print(f"  at fixed cadence ({R_LEVER_FIXES} fixes, sigma = {sigma_ref:.2f} m), moving r:")
    for r, n in r_lever.items():
        print(f"    r = {r:4.1f} m -> {n:7.1f} agents")

    print()
    print("  checks against the published table")
    checks = []
    for row in rows:
        pub = row["published"]
        checks.append(
            check(f"n={row['n_fixes']}: rate", row["rate_hz"], pub["rate_hz"], 0.02, " Hz")
        )
        checks.append(
            check(f"n={row['n_fixes']}: R", row["R_closed_form_m"], pub["R_m"], 0.02, " m")
        )
        checks.append(
            check(f"n={row['n_fixes']}: sigma", row["sigma_m"], pub["sigma_m"], 0.02, " m")
        )
        checks.append(
            check(
                f"n={row['n_fixes']}: N at p={TARGET_P}",
                row["n_at_target"],
                pub["n_at_target"],
                0.3,
            )
        )
    checks.append(check("sigma tightening, 3 -> 24 fixes", sigma_tighten, 2.1, 0.1, "x"))
    for r, pub_n in PUBLISHED_R_LEVER.items():
        checks.append(check(f"radius lever r={r}", r_lever[r], pub_n, 3.0))

    exp.write(
        {
            "parameters": {
                "sensor_noise_std_m": SIGMA_S,
                "r_m": R,
                "lead_time_s": LEAD_TIME_S,
                "track_duration_s": TRACK_DURATION_S,
                "velocity_mps": VELOCITY.tolist(),
                "target_p": TARGET_P,
                "mc_trials_per_row": MC_TRIALS,
            },
            "rows": rows,
            "sigma_tightening_x": round(sigma_tighten, 2),
            "radius_lever": {"at_n_fixes": R_LEVER_FIXES, "sigma_m": sigma_ref, "n_by_r": r_lever},
            "checks": checks,
            "all_checks_pass": all(c["pass"] for c in checks),
        }
    )


if __name__ == "__main__":
    main()

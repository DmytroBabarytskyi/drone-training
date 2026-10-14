"""Experiment 2 — Table II and Figure 2: the planar coverage law.

Three things are checked:

1. equation (6) against Monte Carlo over the production hexagonal lattice,
   across ``N`` in [1, 130];
2. the area-ratio rule, audited at ``sigma/r = 4.6`` — what it claims, what it
   delivers, and what ``N`` would actually be required (Table II);
3. the ``(sigma/r)^2`` scaling, inverted from *measured* coverage rather than
   fitted back from the closed form, which would be circular.

The laws come from ``swarm.coverage_laws`` and the lattice from
``swarm.analytic_policy``; nothing is reimplemented here.
"""

from __future__ import annotations

import numpy as np

from _common import Experiment, check, table
from _coverage_mc import hex_coverage_mc, smallest_n_reaching

from dronesim.ml.swarm.coverage_laws import (
    ETA2,
    area_ratio_mesh_size,
    planar_coverage,
    planar_mesh_size,
)

SIGMA = 9.25
R = 2.0
AUDIT_TRIALS = 200_000
SWEEP_TRIALS = 40_000
SWEEP_N = (1, 2, 3, 5, 8, 13, 21, 34, 55, 86, 110, 130)

# Published Table II: m -> (N by rule, claimed p, actual p, N required)
PUBLISHED_AUDIT = {
    1.0: (22, 0.394, 0.388, 22.1),
    1.5: (49, 0.675, 0.643, 53.0),
    2.0: (86, 0.865, 0.806, 119.0),
    2.5: (134, 0.956, 0.877, None),  # None = unreachable at any N
}
PUBLISHED_LAW_TOL = 0.007     # absolute, Section V-C
PUBLISHED_EXPONENT = 1.876    # Section V-B, from inverting Monte Carlo
PUBLISHED_COUNT_TOL_PCT = 16.0


def audit_row(exp: Experiment, m: float, offset: int) -> dict:
    """One row of the area-ratio audit."""
    n_rule = area_ratio_mesh_size(m, SIGMA, R)
    # Round UP: the rule prescribes a mesh at least this big, and a
    # fractional agent cannot be fielded.
    n_int = int(np.ceil(n_rule))
    # What the rule claims: the whole Gaussian mass inside radius m*sigma.
    claimed = float(1.0 - np.exp(-0.5 * m * m))
    delivered_law = float(planar_coverage(n_int, SIGMA, R))
    delivered_mc = hex_coverage_mc(n_int, SIGMA, R, AUDIT_TRIALS, exp.rng(offset))
    required = planar_mesh_size(claimed, SIGMA, R)

    pub_n, pub_claimed, pub_actual, pub_required = PUBLISHED_AUDIT[m]
    return {
        "m": m,
        "n_by_rule": n_int,
        "claimed_p": round(claimed, 3),
        "delivered_p_law": round(delivered_law, 3),
        "delivered_p_mc": round(delivered_mc, 3),
        "n_required": None if np.isinf(required) else round(required, 1),
        "n_required_unreachable": bool(np.isinf(required)),
        "inflation_x": None if np.isinf(required) else round(required / n_int, 2),
        "published": {
            "n_by_rule": pub_n,
            "claimed_p": pub_claimed,
            "actual_p": pub_actual,
            "n_required": pub_required,
        },
    }


def main() -> None:
    exp = Experiment(
        "exp2_planar_coverage",
        "Table II / Figure 2 - planar coverage law, area-ratio audit, sigma/r scaling",
    )

    # --- 1. the law against Monte Carlo -----------------------------------
    sweep = []
    for i, n in enumerate(SWEEP_N):
        law = float(planar_coverage(n, SIGMA, R))
        mc = hex_coverage_mc(n, SIGMA, R, SWEEP_TRIALS, exp.rng(10 + i))
        sweep.append(
            {
                "n_agents": n,
                "p_law": round(law, 4),
                "p_mc": round(mc, 4),
                "abs_delta": round(abs(law - mc), 4),
            }
        )
    worst_law_delta = max(s["abs_delta"] for s in sweep)

    print()
    print("  law vs Monte Carlo over the production hex lattice")
    print(
        table(
            sweep,
            [
                ("n_agents", "N", "{:d}"),
                ("p_law", "p by (6)", "{:.4f}"),
                ("p_mc", "p measured", "{:.4f}"),
                ("abs_delta", "|delta|", "{:.4f}"),
            ],
        )
    )

    # --- 2. the area-ratio audit (Table II) -------------------------------
    audit = [audit_row(exp, m, 100 + 10 * i) for i, m in enumerate(PUBLISHED_AUDIT)]
    print()
    print(f"  area-ratio audit at sigma/r = {SIGMA / R:.1f}")
    print(
        table(
            audit,
            [
                ("m", "m", "{:.1f}"),
                ("n_by_rule", "N by rule", "{:d}"),
                ("claimed_p", "claimed p", "{:.3f}"),
                ("delivered_p_mc", "actual p", "{:.3f}"),
                ("n_required", "N required", "{:.1f}"),
            ],
        )
    )
    print("    (N required blank = above the packing ceiling, unreachable at any N)")

    # --- 3. the exponent, inverted from measurement -----------------------
    target = 0.80
    scaling = []
    for i, (sigma, r) in enumerate(
        [
            # Small budgets first: this is where an integer number of discs
            # cannot track a continuous law, and where the fitted exponent
            # departs from the exact 2 of equation (7).
            (2.0, 2.0),
            (3.0, 2.0),
            (4.0, 2.0),
            (6.0, 2.0),
            (9.25, 2.0),
            (12.0, 2.0),
            (9.25, 1.5),
            (9.25, 3.0),
            (18.0, 3.0),
        ]
    ):
        n_measured = smallest_n_reaching(target, sigma, r, SWEEP_TRIALS, exp.rng(500 + i))
        n_law = planar_mesh_size(target, sigma, r)
        scaling.append(
            {
                "sigma": sigma,
                "r": r,
                "sigma_over_r": round(sigma / r, 3),
                "n_measured": n_measured,
                "n_law": round(n_law, 1),
                "delta_pct": round(abs(n_measured - n_law) / n_law * 100.0, 1),
            }
        )

    logs = np.array([np.log(s["sigma_over_r"]) for s in scaling])
    logn = np.array([np.log(s["n_measured"]) for s in scaling])
    exponent = float(np.polyfit(logs, logn, 1)[0])
    worst_count_pct = max(s["delta_pct"] for s in scaling)

    print()
    print(f"  exponent on sigma/r at p = {target}, inverted from measured coverage")
    print(
        table(
            scaling,
            [
                ("sigma_over_r", "sigma/r", "{:.2f}"),
                ("n_measured", "N measured", "{:d}"),
                ("n_law", "N by (7)", "{:.1f}"),
                ("delta_pct", "delta (%)", "{:.1f}"),
            ],
        )
    )
    print(f"    fitted exponent = {exponent:.3f}")

    print()
    print("  checks against the published table")
    checks = [
        check("law vs MC, worst |delta|", worst_law_delta, PUBLISHED_LAW_TOL, 0.004),
        check("packing ceiling", ETA2, 0.9069, 0.0001),
    ]

    # Section V-B quotes a fitted exponent of 1.876 and worst-case count
    # agreement of 16%, but does not record WHICH (sigma, r) pairs were in the
    # grid, and both statistics depend on that choice: the departure from the
    # exact exponent of 2 lives entirely at the smallest budgets, so including
    # or excluding one small cell moves them. The grid is therefore stated
    # explicitly above rather than reverse-engineered, and both numbers are
    # reported as measured. They are recorded, not asserted.
    grid_dependent = {
        "fitted_exponent": {
            "reproduced": round(exponent, 3),
            "published": PUBLISHED_EXPONENT,
            "grid_dependent": True,
        },
        "worst_count_delta_pct": {
            "reproduced": round(worst_count_pct, 1),
            "published": PUBLISHED_COUNT_TOL_PCT,
            "grid_dependent": True,
        },
    }
    print()
    print("  grid-dependent statistics (reported, not asserted - see note in record)")
    for k, v in grid_dependent.items():
        print(f"    {k}: reproduced {v['reproduced']}, paper {v['published']}")
    for row in audit:
        pub = row["published"]
        checks.append(
            check(f"m={row['m']}: claimed p", row["claimed_p"], pub["claimed_p"], 0.002)
        )
        checks.append(
            check(f"m={row['m']}: delivered p", row["delivered_p_mc"], pub["actual_p"], 0.012)
        )
        if pub["n_required"] is not None:
            checks.append(
                check(
                    f"m={row['m']}: N required",
                    row["n_required"],
                    pub["n_required"],
                    1.0,
                )
            )
        else:
            ok = row["n_required_unreachable"]
            print(
                f"    {'PASS' if ok else 'FAIL'}  m={row['m']}: target above the ceiling "
                f"-> unreachable (paper: unreachable)"
            )
            checks.append(
                {
                    "label": f"m={row['m']}: unreachable at any N",
                    "reproduced": 1.0 if ok else 0.0,
                    "published": 1.0,
                    "abs_delta": 0.0 if ok else 1.0,
                    "tolerance": 0.0,
                    "pass": bool(ok),
                }
            )

    exp.write(
        {
            "parameters": {
                "sigma_m": SIGMA,
                "r_m": R,
                "sigma_over_r": SIGMA / R,
                "audit_trials_per_cell": AUDIT_TRIALS,
                "sweep_trials_per_point": SWEEP_TRIALS,
                "eta2": ETA2,
            },
            "law_vs_mc": sweep,
            "area_ratio_audit": audit,
            "scaling": {
                "target_p": target,
                "rows": scaling,
                "fitted_exponent": round(exponent, 3),
                "note": (
                    "The exponent is exact by construction in (7); fitting it back from "
                    "the closed form would be circular. These counts are the smallest N "
                    "whose MEASURED coverage reaches the target, so the fit is empirical. "
                    "The shortfall from 2 sits at the smallest budgets, where an integer "
                    "number of discs cannot track a continuous law."
                ),
            },
            "grid_dependent_statistics": grid_dependent,
            "checks": checks,
            "all_checks_pass": all(c["pass"] for c in checks),
        }
    )


if __name__ == "__main__":
    main()

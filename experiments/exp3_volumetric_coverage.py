"""Experiment 3 — Figure 3: the volumetric coverage law.

If the arrival altitude is also uncertain, the mesh must cover a volume. The
derivation carries over with two substitutions: ``eta3 = pi/sqrt(18)`` for the
densest sphere packing, and the chi distribution with three degrees of freedom
for the enclosed mass.

Checked against Monte Carlo over a face-centred-cubic lattice, and the
practical consequence — the ratio of volumetric to planar mesh size — is
recomputed from the laws rather than quoted.

The exponent on ``sigma/r`` is exactly 3 by construction, not by measurement:
``rho`` scales as ``(N/eta3)^(1/3)``, so inverting at fixed ``p`` gives a cubic
dependence identically. The empirical content here is the agreement of
``p3(N)`` with Monte Carlo, not the exponent.
"""

from __future__ import annotations

import numpy as np

from _common import Experiment, check, table
from _coverage_mc import fcc_coverage_mc

from dronesim.ml.swarm.coverage_laws import (
    ETA3,
    planar_mesh_size,
    volumetric_coverage,
    volumetric_mesh_size,
)

SIGMA = 9.25
R = 2.0
TRIALS = 40_000
SWEEP_N = (1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 900)

PUBLISHED_LAW_TOL = 0.005      # absolute, Section V-D
PUBLISHED_CEILING = 0.7405
PUBLISHED_PLANAR_AT_HALF = 31
PUBLISHED_VOLUMETRIC_AT_HALF = 473
PUBLISHED_RATIO = 15


def main() -> None:
    exp = Experiment(
        "exp3_volumetric_coverage",
        "Figure 3 - volumetric coverage law vs Monte Carlo over an FCC lattice",
    )

    sweep = []
    for i, n in enumerate(SWEEP_N):
        law = float(volumetric_coverage(n, SIGMA, R))
        mc = fcc_coverage_mc(n, SIGMA, R, TRIALS, exp.rng(i))
        sweep.append(
            {
                "n_agents": n,
                "p_law": round(law, 4),
                "p_mc": round(mc, 4),
                "abs_delta": round(abs(law - mc), 4),
            }
        )
    worst = max(s["abs_delta"] for s in sweep)

    print()
    print("  volumetric law vs Monte Carlo over an FCC lattice")
    print(
        table(
            sweep,
            [
                ("n_agents", "N", "{:d}"),
                ("p_law", "p3 by (8)", "{:.4f}"),
                ("p_mc", "p measured", "{:.4f}"),
                ("abs_delta", "|delta|", "{:.4f}"),
            ],
        )
    )

    # The practical consequence: constraining the arrival to a known plane.
    sigma_over_r = 4.6
    r_ref = 2.0
    sigma_ref = sigma_over_r * r_ref
    n_planar = planar_mesh_size(0.5, sigma_ref, r_ref)
    n_volumetric = volumetric_mesh_size(0.5, sigma_ref, r_ref)
    ratio = n_volumetric / n_planar

    print()
    print(f"  at sigma/r = {sigma_over_r} and p = 0.5")
    print(f"    planar mesh      {n_planar:7.1f} agents")
    print(f"    volumetric mesh  {n_volumetric:7.1f} agents")
    print(f"    ratio            {ratio:7.1f}x")

    print()
    print("  checks against the published figures")
    checks = [
        check("law vs MC, worst |delta|", worst, PUBLISHED_LAW_TOL, 0.004),
        check("volumetric ceiling", ETA3, PUBLISHED_CEILING, 0.0001),
        check("planar mesh at p=0.5", n_planar, PUBLISHED_PLANAR_AT_HALF, 1.0),
        check("volumetric mesh at p=0.5", n_volumetric, PUBLISHED_VOLUMETRIC_AT_HALF, 12.0),
        check("volumetric/planar ratio", ratio, PUBLISHED_RATIO, 1.0, "x"),
    ]

    exp.write(
        {
            "parameters": {
                "sigma_m": SIGMA,
                "r_m": R,
                "trials_per_point": TRIALS,
                "eta3": ETA3,
            },
            "law_vs_mc": sweep,
            "plane_vs_volume": {
                "sigma_over_r": sigma_over_r,
                "target_p": 0.5,
                "n_planar": round(n_planar, 1),
                "n_volumetric": round(n_volumetric, 1),
                "ratio": round(ratio, 1),
            },
            "exponent_note": (
                "Exactly 3 by construction: rho scales as (N/eta3)^(1/3), so inverting "
                "at fixed p gives a cubic dependence identically. Not measured here, "
                "because fitting it back from the closed form would be circular."
            ),
            "checks": checks,
            "all_checks_pass": all(c["pass"] for c in checks),
        }
    )


if __name__ == "__main__":
    main()

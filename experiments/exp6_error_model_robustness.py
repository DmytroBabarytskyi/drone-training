"""Experiment 6 — Table V and Figure 6: robustness of the error model.

Equation (6) assumes an isotropic Gaussian error. Three departures are likely
in practice: heavy tails from occasional bad fixes, anisotropy because the
crossing geometry rarely leaves the two in-plane axes equally determined, and
bias when the predictor is systematically off-centre rather than merely
imprecise.

A mesh is sized *as if* the error were Gaussian -- which is what a
practitioner holding only sigma would do -- and then measured against each
true distribution.

Normalisation matters and is stated explicitly, because the sign of the
anisotropy result depends on it:

* the Student-t cases hold the **marginal variance** at ``sigma^2``;
* the elliptical cases hold the **trace of the covariance** at ``2*sigma^2``,
  so the RMS radial error matches the isotropic case and only the shape
  differs. The marginals cannot themselves be held fixed under anisotropy,
  since one axis must widen as the other narrows.
"""

from __future__ import annotations

import numpy as np

from _common import Experiment, check, table
from _coverage_mc import covered_fraction

from dronesim.ml.swarm.analytic_policy import hex_lattice_slots
from dronesim.ml.swarm.coverage_laws import planar_mesh_size

SIGMA = 9.25
R = 2.0
TARGET_P = 0.60
TRIALS = 200_000        # for the delivered-coverage column
BISECT_TRIALS = 50_000  # for the "N for target" search: enough to pick an integer

# Published Table V: model -> (delivered coverage, N needed to restore target)
PUBLISHED = {
    "gaussian": (0.602, 43),
    "student_t_nu10": (0.626, 39),
    "student_t_nu5": (0.663, 35),
    "student_t_nu3": (0.734, 24),
    "elliptical_2to1": (0.625, 40),
    "elliptical_4to1": (0.629, 37),
    "elliptical_8to1": (0.635, 37),
    "biased_0.5sigma": (0.562, 47),
    "biased_1.0sigma": (0.456, 67),
    "biased_2.0sigma": (0.185, 136),
}


def draw_gaussian(rng, n):
    return rng.normal(0.0, SIGMA, size=(n, 2))


def draw_student_t(rng, n, nu):
    """Student-t with ``nu`` degrees of freedom, scaled so the marginal
    variance equals ``SIGMA^2`` -- a statement about shape at fixed spread."""
    scale = SIGMA / np.sqrt(nu / (nu - 2.0))
    return rng.standard_t(nu, size=(n, 2)) * scale


def draw_elliptical(rng, n, ratio):
    """Axis-aligned elliptical Gaussian with axis ratio ``ratio``, holding the
    trace of the covariance at ``2*SIGMA^2``."""
    sy = SIGMA * np.sqrt(2.0 / (ratio**2 + 1.0))
    sx = ratio * sy
    return np.stack([rng.normal(0.0, sx, n), rng.normal(0.0, sy, n)], axis=1)


def draw_biased(rng, n, k):
    """Gaussian of the assumed width, displaced by ``k*sigma`` -- what an
    unmodelled manoeuvre produces: the mesh is centred in the wrong place."""
    return rng.normal(0.0, SIGMA, size=(n, 2)) + np.array([k * SIGMA, 0.0])


MODELS = {
    "gaussian": lambda rng, n: draw_gaussian(rng, n),
    "student_t_nu10": lambda rng, n: draw_student_t(rng, n, 10),
    "student_t_nu5": lambda rng, n: draw_student_t(rng, n, 5),
    "student_t_nu3": lambda rng, n: draw_student_t(rng, n, 3),
    "elliptical_2to1": lambda rng, n: draw_elliptical(rng, n, 2.0),
    "elliptical_4to1": lambda rng, n: draw_elliptical(rng, n, 4.0),
    "elliptical_8to1": lambda rng, n: draw_elliptical(rng, n, 8.0),
    "biased_0.5sigma": lambda rng, n: draw_biased(rng, n, 0.5),
    "biased_1.0sigma": lambda rng, n: draw_biased(rng, n, 1.0),
    "biased_2.0sigma": lambda rng, n: draw_biased(rng, n, 2.0),
}


def coverage_under(model, n_agents: int, points: np.ndarray) -> float:
    slots = hex_lattice_slots(n_agents, 2.0 * R, np.zeros(2))
    return covered_fraction(slots, points, R)


def smallest_n_for(model_key: str, exp: Experiment, offset: int, n_cap: int = 400) -> int:
    """Smallest mesh that restores ``TARGET_P`` under this error model."""
    rng = exp.rng(offset)
    points = MODELS[model_key](rng, BISECT_TRIALS)
    lo, hi = 1, n_cap
    if coverage_under(model_key, hi, points) < TARGET_P:
        return -1
    while lo < hi:
        mid = (lo + hi) // 2
        if coverage_under(model_key, mid, points) >= TARGET_P:
            hi = mid
        else:
            lo = mid + 1
    return lo


def main() -> None:
    exp = Experiment(
        "exp6_error_model_robustness",
        "Table V / Figure 6 - coverage delivered by a mesh sized for a Gaussian",
    )

    # Round UP, as the paper does: the law prescribes a mesh at least this
    # big and a fractional agent cannot be fielded.
    n_sized = int(np.ceil(planar_mesh_size(TARGET_P, SIGMA, R)))
    print(f"[{exp.name}] mesh sized for a Gaussian at p={TARGET_P}: N = {n_sized}")

    rows = []
    for i, key in enumerate(MODELS):
        rng = exp.rng(10 * i)
        points = MODELS[key](rng, TRIALS)
        delivered = coverage_under(key, n_sized, points)
        n_needed = smallest_n_for(key, exp, 500 + 10 * i)
        pub_delivered, pub_n = PUBLISHED[key]
        rows.append(
            {
                "error_model": key,
                "delivered": round(delivered, 3),
                "surplus": round(delivered - TARGET_P, 3),
                "n_for_target": n_needed,
                "inflation_x": round(n_needed / n_sized, 2),
                "published": {"delivered": pub_delivered, "n_for_target": pub_n},
            }
        )

    print()
    print(f"  mesh sized for a Gaussian (N = {n_sized}, target {TARGET_P})")
    print(
        table(
            rows,
            [
                ("error_model", "error model", "{:s}"),
                ("delivered", "delivered", "{:.3f}"),
                ("surplus", "surplus", "{:+.3f}"),
                ("n_for_target", "N for target", "{:d}"),
            ],
        )
    )

    by_key = {r["error_model"]: r for r in rows}
    bias_1 = TARGET_P - by_key["biased_1.0sigma"]["delivered"]
    bias_2 = TARGET_P - by_key["biased_2.0sigma"]["delivered"]
    shape_conservative = all(
        by_key[k]["surplus"] > 0
        for k in MODELS
        if k.startswith(("student_t", "elliptical"))
    )
    print()
    print(
        f"  a one-sigma aim-point offset costs {100 * bias_1:.0f} points of coverage, "
        f"a two-sigma offset {100 * bias_2:.0f}"
    )
    print(
        f"  shape departures leave the law conservative: "
        f"{'yes' if shape_conservative else 'NO'}"
    )

    print()
    print("  checks against the published table")
    checks = [check("mesh sized for a Gaussian", n_sized, 43, 1.0)]
    for row in rows:
        pub = row["published"]
        checks.append(
            check(f"{row['error_model']}: delivered", row["delivered"], pub["delivered"], 0.030)
        )
        checks.append(
            check(
                f"{row['error_model']}: N for target",
                row["n_for_target"],
                pub["n_for_target"],
                max(3.0, 0.12 * pub["n_for_target"]),
            )
        )
    checks.append(check("one-sigma bias cost (points)", 100 * bias_1, 15, 3.0))
    checks.append(check("two-sigma bias cost (points)", 100 * bias_2, 42, 4.0))
    checks.append(
        check(
            "two-sigma bias inflation",
            by_key["biased_2.0sigma"]["inflation_x"],
            3.2,
            0.4,
            "x",
        )
    )
    print(
        f"    {'PASS' if shape_conservative else 'FAIL'}  every shape departure "
        f"over-delivers (the split the paper reports)"
    )
    checks.append(
        {
            "label": "shape departures conservative",
            "reproduced": 1.0 if shape_conservative else 0.0,
            "published": 1.0,
            "abs_delta": 0.0 if shape_conservative else 1.0,
            "tolerance": 0.0,
            "pass": bool(shape_conservative),
        }
    )

    exp.write(
        {
            "parameters": {
                "sigma_m": SIGMA,
                "r_m": R,
                "target_p": TARGET_P,
                "n_sized_for_gaussian": n_sized,
                "trials_per_model": TRIALS,
                "normalisation": {
                    "student_t": "marginal variance held at sigma^2",
                    "elliptical": "trace of covariance held at 2*sigma^2",
                    "bias": "Gaussian of the assumed width, centre displaced by k*sigma",
                },
            },
            "rows": rows,
            "summary": {
                "one_sigma_bias_cost_points": round(100 * bias_1, 1),
                "two_sigma_bias_cost_points": round(100 * bias_2, 1),
                "shape_departures_conservative": bool(shape_conservative),
            },
            "checks": checks,
            "all_checks_pass": all(c["pass"] for c in checks),
        }
    )


if __name__ == "__main__":
    main()

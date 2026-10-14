"""Figure generator — builds every figure in the paper from the recorded
JSON results, not by re-running the physics.

Each experiment writes its record under ``experiments/results/``; this script
reads those and renders the six PNGs. Run the experiments first (or use
``run_all.py``, which does both in order).

The filenames are the ones ``main.tex`` includes. Note that the file numbers
and the figure numbers do not line up, because the figures appear in the
manuscript in a different order from the one they were produced in:

    fig1_uncertainty.png     -> Figure 1
    fig2_coverage.png        -> Figure 2
    fig3_volume.png          -> Figure 3
    fig5_strategies.png      -> Figure 4
    fig6_robustness.png      -> Figure 5
    fig4_commit_window.png   -> Figure 6

Usage:
    python experiments/make_figures.py [--out DIR]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from _common import RESULTS  # noqa: E402

plt.rcParams.update(
    {
        "figure.dpi": 150,
        "font.size": 9,
        "axes.grid": True,
        "grid.alpha": 0.3,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "legend.frameon": False,
    }
)

ETA2 = float(np.pi / np.sqrt(12.0))
ETA3 = float(np.pi / np.sqrt(18.0))


def load(name: str) -> dict:
    path = RESULTS / f"{name}.json"
    if not path.exists():
        raise SystemExit(f"missing {path}; run the experiments first (see run_all.py)")
    return json.loads(path.read_text(encoding="utf-8"))


# ------------------------------------------------------------------ figure 1


def fig1_uncertainty(out: Path) -> None:
    d = load("exp1_prediction_uncertainty")
    rows = d["rows"]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.2, 3.1))

    closed = [r["R_closed_form_m"] for r in rows]
    mc = [r["R_monte_carlo_m"] for r in rows]
    ax1.plot([0, max(mc) * 1.05], [0, max(mc) * 1.05], "k--", lw=0.8, label="$y=x$")
    ax1.scatter(mc, closed, s=22, zorder=3)
    ax1.set_xlabel("Monte Carlo RMS radial error (m)")
    ax1.set_ylabel("closed-form $R$ (m)")
    ax1.set_title("(a) closed form against Monte Carlo", loc="left", fontsize=9)
    ax1.legend(loc="lower right")

    for n in sorted({r["n_fixes"] for r in rows}):
        sel = sorted([r for r in rows if r["n_fixes"] == n], key=lambda r: r["lead_time_s"])
        t = [r["lead_time_s"] for r in sel]
        ax2.plot(t, [r["R_closed_form_m"] for r in sel], "o-", ms=3, label=f"$n={n}$")
        ax2.plot(t, [r["heuristic_m"] for r in sel], ":", lw=1.0, color=ax2.lines[-1].get_color())
    ax2.set_xlabel("lead time $t_c$ (s)")
    ax2.set_ylabel("$R$ (m)")
    ax2.set_yscale("log")
    ax2.set_title("(b) growth with lead time; dotted = heuristic", loc="left", fontsize=9)
    ax2.legend(ncol=2, fontsize=8)

    fig.tight_layout()
    fig.savefig(out / "fig1_uncertainty.png")
    plt.close(fig)


# ------------------------------------------------------------------ figure 2


def fig2_coverage(out: Path) -> None:
    d = load("exp2_planar_coverage")
    sweep = d["law_vs_mc"]
    audit = d["area_ratio_audit"]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.2, 3.1))

    n = [s["n_agents"] for s in sweep]
    ax1.plot(n, [s["p_law"] for s in sweep], "-", label="equation (6)")
    ax1.plot(n, [s["p_mc"] for s in sweep], "o", ms=4, label="Monte Carlo")
    ax1.axhline(ETA2, ls=":", color="k", lw=1.0)
    ax1.annotate(
        rf"packing ceiling $\eta_2={ETA2:.4f}$",
        xy=(n[-1], ETA2),
        xytext=(-4, -12),
        textcoords="offset points",
        ha="right",
        fontsize=8,
    )
    ax1.set_xlabel("mesh size $N$")
    ax1.set_ylabel("coverage probability")
    ax1.set_ylim(0, 1.0)
    ax1.set_title("(a) coverage against mesh size", loc="left", fontsize=9)
    ax1.legend(loc="lower right")

    m = [a["m"] for a in audit]
    width = 0.35
    x = np.arange(len(m))
    ax2.bar(x - width / 2, [a["claimed_p"] for a in audit], width, label="claimed by the rule")
    ax2.bar(x + width / 2, [a["delivered_p_mc"] for a in audit], width, label="delivered")
    ax2.axhline(ETA2, ls=":", color="k", lw=1.0)
    ax2.set_xticks(x, [f"$m={v}$" for v in m])
    ax2.set_ylabel("coverage probability")
    ax2.set_ylim(0, 1.05)
    ax2.set_title("(b) the area-ratio rule audited", loc="left", fontsize=9)
    ax2.legend(loc="upper left", fontsize=8)

    fig.tight_layout()
    fig.savefig(out / "fig2_coverage.png")
    plt.close(fig)


# ------------------------------------------------------------------ figure 3


def fig3_volume(out: Path) -> None:
    dv = load("exp3_volumetric_coverage")
    dp = load("exp2_planar_coverage")
    fig, ax = plt.subplots(figsize=(4.0, 3.3))

    np_ = [s["n_agents"] for s in dp["law_vs_mc"]]
    ax.plot(np_, [s["p_law"] for s in dp["law_vs_mc"]], "-", label="planar, eq. (6)")
    ax.plot(np_, [s["p_mc"] for s in dp["law_vs_mc"]], "o", ms=3.5)

    nv = [s["n_agents"] for s in dv["law_vs_mc"]]
    ax.plot(nv, [s["p_law"] for s in dv["law_vs_mc"]], "-", label="volumetric, eq. (8)")
    ax.plot(nv, [s["p_mc"] for s in dv["law_vs_mc"]], "s", ms=3.5)

    ax.axhline(ETA2, ls=":", color="k", lw=1.0)
    ax.axhline(ETA3, ls=":", color="k", lw=1.0)
    ax.annotate(rf"$\eta_2={ETA2:.4f}$", xy=(1.2, ETA2 + 0.012), fontsize=8)
    ax.annotate(rf"$\eta_3={ETA3:.4f}$", xy=(1.2, ETA3 + 0.012), fontsize=8)
    ax.set_xscale("log")
    ax.set_xlabel("mesh size $N$")
    ax.set_ylabel("coverage probability")
    ax.set_ylim(0, 1.0)
    ax.legend(loc="upper left", fontsize=8)
    fig.tight_layout()
    fig.savefig(out / "fig3_volume.png")
    plt.close(fig)


# ------------------------------------------------ figure 4 (file: strategies)


def fig5_strategies(out: Path) -> None:
    d = load("exp5_placement_strategies")
    cov = d["coverage"]
    rel = d["relative_to_hexagonal"]
    budgets = [int(b) for b in d["parameters"]["budgets"]]
    labels = {
        "hexagonal": "hexagonal lattice",
        "fermat_spiral": "Fermat spiral",
        "square_lattice": "square lattice",
        "lloyd_cvt": "Lloyd / CVT",
        "random_nonoverlapping": "random, non-overlapping",
    }
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.2, 3.1))
    for key, label in labels.items():
        ax1.plot(budgets, [cov[key][str(b)] for b in budgets], "o-", ms=4, label=label)
        ax2.plot(budgets, [rel[key][str(b)] for b in budgets], "o-", ms=4, label=label)
    ax1.set_xlabel("mesh size $N$")
    ax1.set_ylabel("coverage probability")
    ax1.set_title("(a) absolute coverage", loc="left", fontsize=9)
    ax1.legend(fontsize=7.5, loc="upper left")
    ax2.axhline(1.0, ls=":", color="k", lw=1.0)
    ax2.set_xlabel("mesh size $N$")
    ax2.set_ylabel("coverage relative to hexagonal")
    ax2.set_title("(b) relative to hexagonal packing", loc="left", fontsize=9)
    fig.tight_layout()
    fig.savefig(out / "fig5_strategies.png")
    plt.close(fig)


# ------------------------------------------------ figure 5 (file: robustness)


def fig6_robustness(out: Path) -> None:
    d = load("exp6_error_model_robustness")
    rows = d["rows"]
    target = d["parameters"]["target_p"]
    n_sized = d["parameters"]["n_sized_for_gaussian"]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.2, 3.3))
    names = [r["error_model"] for r in rows]
    x = np.arange(len(names))
    colours = ["0.55" if not n.startswith("biased") else "tab:red" for n in names]
    ax1.bar(x, [r["delivered"] for r in rows], color=colours)
    ax1.axhline(target, ls="--", color="k", lw=1.0, label=f"target {target}")
    ax1.set_xticks(x, names, rotation=55, ha="right", fontsize=7)
    ax1.set_ylabel("coverage delivered")
    ax1.set_title(f"(a) delivered at $N={n_sized}$", loc="left", fontsize=9)
    ax1.legend(fontsize=8)

    ax2.bar(x, [r["n_for_target"] for r in rows], color=colours)
    ax2.axhline(n_sized, ls="--", color="k", lw=1.0, label=f"$N={n_sized}$ as sized")
    ax2.set_xticks(x, names, rotation=55, ha="right", fontsize=7)
    ax2.set_ylabel("agents needed to restore the target")
    ax2.set_title("(b) budget needed", loc="left", fontsize=9)
    ax2.legend(fontsize=8)
    fig.text(
        0.5,
        0.005,
        "grey: shape departures (law conservative)    red: bias (law fragile)",
        ha="center",
        fontsize=8,
    )
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    fig.savefig(out / "fig6_robustness.png")
    plt.close(fig)


# ---------------------------------------------- figure 6 (file: commit window)


def fig4_commit_window(out: Path) -> None:
    d = load("exp7_commit_window")
    sweep = d["budget_sweep"]
    p = d["parameters"]
    t = [r["T_s"] for r in sweep]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.2, 3.1))

    ax1.plot(t, [r["sigma_sensor_m"] for r in sweep], "o-", ms=3, label="sensor term")
    ax1.plot(t, [r["sigma_manoeuvre_m"] for r in sweep], "s-", ms=3, label="manoeuvre term")
    ax1.plot(t, [r["sigma_total_m"] for r in sweep], "-", lw=1.4, label="in quadrature")
    ax1.set_yscale("log")
    ax1.set_xlabel("commit time $T$ (s)")
    ax1.set_ylabel(r"$\sigma$ (m)")
    ax1.set_title(
        f"(a) uncertainty budget at {p['manoeuvre_g']}g", loc="left", fontsize=9
    )
    ax1.legend(fontsize=8, loc="lower right")

    ax2.plot(
        t, [r["required_transit_worst_m"] for r in sweep], "-", label=r"required, $\sqrt{2}(L+C)$"
    )
    ax2.plot(
        t,
        [r["required_transit_optimistic_m"] for r in sweep],
        "--",
        label="required, optimistic ($C$)",
    )
    ax2.plot(t, [r["reach_m"] for r in sweep], "-", lw=1.6, color="k", label="reach $d(T)$")
    ax2.set_xlabel("commit time $T$ (s)")
    ax2.set_ylabel("distance (m)")
    ax2.set_yscale("log")
    ax2.set_title("(b) transit required against reach", loc="left", fontsize=9)
    ax2.legend(fontsize=8, loc="lower right")

    fig.tight_layout()
    fig.savefig(out / "fig4_commit_window.png")
    plt.close(fig)


FIGURES = (
    fig1_uncertainty,
    fig2_coverage,
    fig3_volume,
    fig5_strategies,
    fig6_robustness,
    fig4_commit_window,
)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--out",
        default=str(Path(__file__).resolve().parent / "figures"),
        help="directory to write the PNGs into",
    )
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    for fn in FIGURES:
        fn(out)
        print(f"  {fn.__name__} -> {out.name}/")
    print(f"{len(FIGURES)} figures written to {out}")


if __name__ == "__main__":
    main()

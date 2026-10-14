"""Figure generator — builds every figure in the paper from the recorded
JSON results, not by re-running the physics.

Each experiment writes its record under ``experiments/results/``; this script
reads those and renders the six PNGs. Run the experiments first (or use
``run_all.py``, which does both in order).

Style lives in ``_figstyle.py``: a validated, colour-vision-safe palette with
a marker and dash pattern per series, so the figures survive greyscale
printing in a two-column layout.

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

import numpy as np

from _common import RESULTS
from _figstyle import INK_MUTED, PALETTE, RULE, apply, panel_title, series

apply()
import matplotlib.pyplot as plt  # noqa: E402

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
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.0, 2.9))

    closed = [r["R_closed_form_m"] for r in rows]
    mc = [r["R_monte_carlo_m"] for r in rows]
    top = max(max(mc), max(closed)) * 1.08
    ax1.plot([0, top], [0, top], ls=(0, (4, 2)), color=RULE, lw=0.9, label="$y=x$")
    ax1.scatter(mc, closed, s=20, color=PALETTE[0], zorder=3, edgecolors="white",
                linewidths=0.4)
    ax1.set_xlim(0, top)
    ax1.set_ylim(0, top)
    ax1.set_xlabel("Monte Carlo RMS radial error (m)")
    ax1.set_ylabel("closed-form $R$ (m)")
    panel_title(ax1, "(a) closed form against Monte Carlo")
    ax1.legend(loc="lower right")

    counts = sorted({r["n_fixes"] for r in rows})
    for i, n in enumerate(counts):
        sel = sorted([r for r in rows if r["n_fixes"] == n], key=lambda r: r["lead_time_s"])
        t = [r["lead_time_s"] for r in sel]
        kw = series(i)
        ax2.plot(t, [r["R_closed_form_m"] for r in sel], label=f"$n={n}$", **kw)
        ax2.plot(t, [r["heuristic_m"] for r in sel], ls=":", lw=1.0,
                 color=kw["color"], marker="", zorder=1)
    ax2.set_yscale("log")
    ax2.set_xlabel("lead time $t_c$ (s)")
    ax2.set_ylabel("$R$ (m)")
    ax2.set_ylim(top=ax2.get_ylim()[1] * 2.6)  # headroom so the legend clears the data
    panel_title(ax2, "(b) growth with lead time; dotted = heuristic")
    ax2.legend(loc="upper left", ncol=4, columnspacing=1.0, handletextpad=0.4)

    fig.tight_layout()
    fig.savefig(out / "fig1_uncertainty.png")
    plt.close(fig)


# ------------------------------------------------------------------ figure 2


def fig2_coverage(out: Path) -> None:
    d = load("exp2_planar_coverage")
    sweep = d["law_vs_mc"]
    audit = d["area_ratio_audit"]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.0, 2.9))

    n = [s["n_agents"] for s in sweep]
    ax1.plot(n, [s["p_law"] for s in sweep], color=PALETTE[0], lw=1.6,
             label="equation (6)", zorder=2)
    ax1.plot(n, [s["p_mc"] for s in sweep], ls="none", marker="o", ms=4.2,
             color=PALETTE[1], label="Monte Carlo", zorder=3)
    ax1.set_ylim(0, 1.0)
    ax1.set_xlim(0, max(n) * 1.03)
    ax1.axhline(ETA2, ls=(0, (1.5, 1.8)), color=RULE, lw=0.9, zorder=1)
    # Parked top-left, where the curve is still near zero.
    ax1.text(0.03, 0.955, rf"packing ceiling $\eta_2={ETA2:.4f}$",
             transform=ax1.transAxes, ha="left", va="top", fontsize=7.2,
             color=INK_MUTED)
    ax1.set_xlabel("mesh size $N$")
    ax1.set_ylabel("coverage probability")
    panel_title(ax1, "(a) coverage against mesh size")
    ax1.legend(loc="lower right")

    m = [a["m"] for a in audit]
    x = np.arange(len(m))
    w = 0.36
    ax2.bar(x - w / 2, [a["claimed_p"] for a in audit], w, color=PALETTE[0],
            label="claimed by the rule", zorder=2)
    ax2.bar(x + w / 2, [a["delivered_p_mc"] for a in audit], w, color=PALETTE[1],
            label="delivered", zorder=2)
    ax2.axhline(ETA2, ls=(0, (1.5, 1.8)), color=RULE, lw=0.9, zorder=3)
    ax2.set_ylim(0, 1.22)
    ax2.text(0.02, ETA2 + 0.015, rf"$\eta_2={ETA2:.4f}$", fontsize=7.2,
             color=INK_MUTED, va="bottom", ha="left")
    ax2.set_xticks(x, [f"$m={v}$" for v in m])
    ax2.set_ylabel("coverage probability")
    panel_title(ax2, "(b) the area-ratio rule audited")
    # Above the bars, outside the data: nothing to collide with.
    ax2.legend(loc="upper center", ncol=2, bbox_to_anchor=(0.5, 1.0),
               columnspacing=1.0, handletextpad=0.5)

    fig.tight_layout()
    fig.savefig(out / "fig2_coverage.png")
    plt.close(fig)


# ------------------------------------------------------------------ figure 3


def fig3_volume(out: Path) -> None:
    dv = load("exp3_volumetric_coverage")
    dp = load("exp2_planar_coverage")
    fig, ax = plt.subplots(figsize=(3.5, 2.9))

    npl = [s["n_agents"] for s in dp["law_vs_mc"]]
    ax.plot(npl, [s["p_law"] for s in dp["law_vs_mc"]], color=PALETTE[0], lw=1.6,
            label="planar, eq. (6)", zorder=2)
    ax.plot(npl, [s["p_mc"] for s in dp["law_vs_mc"]], ls="none", marker="o",
            ms=3.6, color=PALETTE[0], zorder=3)

    nv = [s["n_agents"] for s in dv["law_vs_mc"]]
    ax.plot(nv, [s["p_law"] for s in dv["law_vs_mc"]], color=PALETTE[2], lw=1.6,
            dashes=(5, 1.6), label="volumetric, eq. (8)", zorder=2)
    ax.plot(nv, [s["p_mc"] for s in dv["law_vs_mc"]], ls="none", marker="^",
            ms=3.6, color=PALETTE[2], zorder=3)

    ax.set_xscale("log")
    ax.set_ylim(0, 1.04)
    ax.set_xlabel("mesh size $N$")
    ax.set_ylabel("coverage probability")

    for y, lab in ((ETA2, rf"$\eta_2={ETA2:.4f}$"), (ETA3, rf"$\eta_3={ETA3:.4f}$")):
        ax.axhline(y, ls=(0, (1.5, 1.8)), color=RULE, lw=0.9, zorder=1)
        # Right edge, above each line: the curves are below it there.
        ax.text(0.985, y + 0.018, lab, transform=ax.get_yaxis_transform(),
                ha="right", va="bottom", fontsize=7.2, color=INK_MUTED,
                clip_on=False)
    # Mid-left, between the two rising curves and below both ceilings.
    ax.legend(loc="center left", bbox_to_anchor=(0.02, 0.56))

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
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.0, 3.1))
    handles = []
    for i, (key, label) in enumerate(labels.items()):
        kw = series(i)
        (h,) = ax1.plot(budgets, [cov[key][str(b)] for b in budgets], label=label, **kw)
        ax2.plot(budgets, [rel[key][str(b)] for b in budgets], **kw)
        handles.append(h)
    ax1.set_xlabel("mesh size $N$")
    ax1.set_ylabel("coverage probability")
    panel_title(ax1, "(a) absolute coverage")
    ax2.axhline(1.0, ls=(0, (1.5, 1.8)), color=RULE, lw=0.9, zorder=1)
    ax2.set_xlabel("mesh size $N$")
    ax2.set_ylabel("coverage relative to hexagonal")
    panel_title(ax2, "(b) relative to hexagonal packing")

    # One shared legend below both panels: five series will not fit inside
    # either without landing on a curve.
    fig.legend(handles, list(labels.values()), loc="lower center", ncol=3,
               frameon=False, bbox_to_anchor=(0.5, -0.02), columnspacing=1.4,
               handletextpad=0.5)
    fig.tight_layout(rect=(0, 0.11, 1, 1))
    fig.savefig(out / "fig5_strategies.png")
    plt.close(fig)


# ------------------------------------------------ figure 5 (file: robustness)


def fig6_robustness(out: Path) -> None:
    d = load("exp6_error_model_robustness")
    rows = d["rows"]
    target = d["parameters"]["target_p"]
    n_sized = d["parameters"]["n_sized_for_gaussian"]

    pretty = {
        "gaussian": "Gaussian",
        "student_t_nu10": r"$t,\ \nu{=}10$",
        "student_t_nu5": r"$t,\ \nu{=}5$",
        "student_t_nu3": r"$t,\ \nu{=}3$",
        "elliptical_2to1": "ell. 2:1",
        "elliptical_4to1": "ell. 4:1",
        "elliptical_8to1": "ell. 8:1",
        "biased_0.5sigma": r"bias $0.5\sigma$",
        "biased_1.0sigma": r"bias $1.0\sigma$",
        "biased_2.0sigma": r"bias $2.0\sigma$",
    }
    names = [pretty[r["error_model"]] for r in rows]
    is_bias = [r["error_model"].startswith("biased") for r in rows]
    colours = [PALETTE[4] if b else PALETTE[0] for b in is_bias]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.0, 3.2))
    x = np.arange(len(names))

    ax1.bar(x, [r["delivered"] for r in rows], color=colours, zorder=2)
    ax1.axhline(target, ls=(0, (4, 2)), color=RULE, lw=0.9, zorder=3)
    ax1.set_ylim(0, 0.92)
    ax1.text(0.98, target + 0.012, f"target {target}", transform=ax1.get_yaxis_transform(),
             ha="right", va="bottom", fontsize=7.2, color=INK_MUTED)
    ax1.set_xticks(x, names, rotation=52, ha="right")
    ax1.set_ylabel("coverage delivered")
    panel_title(ax1, f"(a) delivered at $N={n_sized}$")

    ax2.bar(x, [r["n_for_target"] for r in rows], color=colours, zorder=2)
    ax2.axhline(n_sized, ls=(0, (4, 2)), color=RULE, lw=0.9, zorder=3)
    ax2.set_ylim(0, max(r["n_for_target"] for r in rows) * 1.22)
    ax2.text(0.98, n_sized + 3, f"$N={n_sized}$ as sized",
             transform=ax2.get_yaxis_transform(), ha="right", va="bottom",
             fontsize=7.2, color=INK_MUTED)
    ax2.set_xticks(x, names, rotation=52, ha="right")
    ax2.set_ylabel("agents needed to restore the target")
    panel_title(ax2, "(b) budget needed")

    from matplotlib.patches import Patch

    fig.legend(
        [Patch(color=PALETTE[0]), Patch(color=PALETTE[4])],
        ["shape departure (law conservative)", "bias (law fragile)"],
        loc="lower center", ncol=2, frameon=False, bbox_to_anchor=(0.5, -0.02),
        columnspacing=1.6,
    )
    fig.tight_layout(rect=(0, 0.09, 1, 1))
    fig.savefig(out / "fig6_robustness.png")
    plt.close(fig)


# ---------------------------------------------- figure 6 (file: commit window)


def fig4_commit_window(out: Path) -> None:
    d = load("exp7_commit_window")
    sweep = d["budget_sweep"]
    p = d["parameters"]
    t = [r["T_s"] for r in sweep]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.0, 2.9))

    for i, (key, label) in enumerate(
        (
            ("sigma_sensor_m", "sensor term"),
            ("sigma_manoeuvre_m", "manoeuvre term"),
            ("sigma_total_m", "in quadrature"),
        )
    ):
        kw = series(i)
        kw["markevery"] = 3
        ax1.plot(t, [r[key] for r in sweep], label=label, **kw)
    ax1.set_yscale("log")
    ax1.set_ylim(top=ax1.get_ylim()[1] * 9)  # headroom for the legend
    ax1.set_xlabel("commit time $T$ (s)")
    ax1.set_ylabel(r"$\sigma$ (m)")
    panel_title(ax1, f"(a) uncertainty budget at {p['manoeuvre_g']}g")
    ax1.legend(loc="upper left", ncol=1)

    for i, (key, label) in enumerate(
        (
            ("required_transit_worst_m", r"required, $\sqrt{2}(L+C)$"),
            ("required_transit_optimistic_m", "required, optimistic ($C$)"),
            ("reach_m", "reach $d(T)$"),
        )
    ):
        kw = series(i)
        kw["markevery"] = 3
        ax2.plot(t, [r[key] for r in sweep], label=label, **kw)
    ax2.set_yscale("log")
    ax2.set_ylim(top=ax2.get_ylim()[1] * 7)
    ax2.set_xlabel("commit time $T$ (s)")
    ax2.set_ylabel("distance (m)")
    panel_title(ax2, "(b) transit required against reach")
    ax2.legend(loc="upper left", ncol=1)

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

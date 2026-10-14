"""Single entry point: regenerate every number and every figure in the paper.

    python experiments/run_all.py

Runs the eight experiment scripts in order, then the figure generator, then
prints a summary of every check against the values printed in the manuscript.
Exit status is non-zero if any assertable check failed, so this doubles as a
regression test on the analysis.

Experiment 8 needs the three PPO checkpoints, which live under ``runs/`` and
are therefore gitignored; fetch them from the release attached to the commit
the paper cites. Pass ``--skip-policy`` to run everything else without them.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"

SCRIPTS = (
    "exp1_prediction_uncertainty.py",
    "exp2_planar_coverage.py",
    "exp3_volumetric_coverage.py",
    "exp4_sensor_cadence.py",
    "exp5_placement_strategies.py",
    "exp6_error_model_robustness.py",
    "exp7_commit_window.py",
    "exp8_policy_comparison.py",
)

TABLES = {
    "exp1_prediction_uncertainty": "Table I, Figure 1",
    "exp2_planar_coverage": "Table II, Figure 2",
    "exp3_volumetric_coverage": "Figure 3",
    "exp4_sensor_cadence": "Table III",
    "exp5_placement_strategies": "Table IV, Figure 4",
    "exp6_error_model_robustness": "Table V, Figure 5",
    "exp7_commit_window": "Table VI, Figure 6",
    "exp8_policy_comparison": "Tables VII and VIII",
}


def run(script: str, quiet: bool) -> int:
    print(f"\n{'=' * 72}\n{script}\n{'=' * 72}")
    proc = subprocess.run(
        [sys.executable, str(HERE / script)],
        cwd=HERE,
        capture_output=quiet,
        text=True,
    )
    if quiet and proc.returncode != 0:
        print(proc.stdout or "")
        print(proc.stderr or "")
    return proc.returncode


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--skip-policy", action="store_true", help="skip experiment 8")
    ap.add_argument("--quiet", action="store_true", help="only show the summary")
    ap.add_argument("--figures-out", default=str(HERE / "figures"))
    args = ap.parse_args()

    t0 = time.perf_counter()
    scripts = [s for s in SCRIPTS if not (args.skip_policy and s.startswith("exp8"))]

    failed_scripts = []
    for s in scripts:
        if run(s, args.quiet) != 0:
            failed_scripts.append(s)

    print(f"\n{'=' * 72}\nmake_figures.py\n{'=' * 72}")
    fig = subprocess.run(
        [sys.executable, str(HERE / "make_figures.py"), "--out", args.figures_out],
        cwd=HERE,
        capture_output=args.quiet,
        text=True,
    )
    if fig.returncode != 0:
        failed_scripts.append("make_figures.py")
        print(fig.stdout or "")
        print(fig.stderr or "")

    # ----------------------------------------------------------- summary
    print(f"\n{'=' * 72}\nSUMMARY: reproduced values against the manuscript\n{'=' * 72}")
    total = passed = 0
    reported = 0
    for name, where in TABLES.items():
        path = RESULTS / f"{name}.json"
        if not path.exists():
            print(f"  {where:24s} {name:32s} NOT RUN")
            continue
        rec = json.loads(path.read_text(encoding="utf-8"))
        checks = rec.get("checks", [])
        ok = sum(1 for c in checks if c["pass"])
        total += len(checks)
        passed += ok
        extra = sum(
            len(rec.get(k, {}))
            for k in (
                "grid_dependent_statistics",
                "construction_dependent_rows",
                "grid_sensitive_table6_rows",
            )
        )
        reported += extra
        flag = "ok " if ok == len(checks) else "FAIL"
        note = f"  (+{extra} reported, not asserted)" if extra else ""
        print(
            f"  {flag} {where:24s} {ok:3d}/{len(checks):<3d} checks  "
            f"{rec['elapsed_s']:6.1f} s{note}"
        )

    elapsed = time.perf_counter() - t0
    print(f"\n  {passed}/{total} assertable checks passed")
    if reported:
        print(
            f"  {reported} further values are reported but not asserted, because the "
            f"manuscript\n  does not record the grid or construction they depend on "
            f"(see the JSON records)."
        )
    print(f"  total wall time {elapsed:.0f} s")

    if failed_scripts:
        print(f"\n  scripts that errored: {', '.join(failed_scripts)}")
        sys.exit(1)
    if passed != total:
        sys.exit(1)


if __name__ == "__main__":
    main()

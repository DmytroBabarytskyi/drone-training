"""Shared plumbing for the preprint's experiment scripts.

Every script here follows the same contract:

* it imports the prediction, lattice, coverage-law and environment code from
  ``src/dronesim`` rather than reimplementing any formula, so a validation run
  tests the implementation that actually runs;
* it fixes and *states* its Monte Carlo seed;
* it writes one JSON record under ``experiments/results/`` containing the
  table it produced, the parameters that produced it, and the published values
  it is checked against, so the figure generator and any later audit read data
  rather than re-running the physics.

Run an individual script directly, or all of them through ``run_all.py``.
"""

from __future__ import annotations

import json
import platform
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
RESULTS = Path(__file__).resolve().parent / "results"
sys.path.insert(0, str(ROOT / "src"))

# Monte Carlo seeds, fixed once here so every run is reproducible and the
# seed used by each experiment is a stated number rather than folklore.
SEEDS = {
    "exp1_prediction_uncertainty": 20260301,
    "exp2_planar_coverage": 20260302,
    "exp3_volumetric_coverage": 20260303,
    "exp4_sensor_cadence": 20260304,
    "exp5_placement_strategies": 20260305,
    "exp6_error_model_robustness": 20260306,
    "exp7_commit_window": 20260307,
    "exp8_policy_comparison": 20260308,
}


def git_commit() -> str:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True, timeout=10
        )
        return out.stdout.strip() or "unknown"
    except Exception:
        return "unknown"


def git_dirty() -> bool:
    try:
        out = subprocess.run(
            ["git", "status", "--porcelain"], cwd=ROOT, capture_output=True, text=True, timeout=10
        )
        return bool(out.stdout.strip())
    except Exception:
        return False


@dataclass
class Experiment:
    """One experiment run: times itself and writes its JSON record."""

    name: str
    description: str
    _t0: float = 0.0

    def __post_init__(self) -> None:
        self._t0 = time.perf_counter()
        self.seed = SEEDS[self.name]
        print(f"[{self.name}] {self.description}")
        print(f"[{self.name}] seed = {self.seed}")

    def rng(self, offset: int = 0) -> np.random.Generator:
        """Seeded generator. ``offset`` lets one script use independent
        streams per cell without reseeding from entropy."""
        return np.random.default_rng(self.seed + offset)

    def write(self, payload: dict[str, Any]) -> Path:
        RESULTS.mkdir(parents=True, exist_ok=True)
        elapsed = time.perf_counter() - self._t0
        record = {
            "experiment": self.name,
            "description": self.description,
            "seed": self.seed,
            "elapsed_s": round(elapsed, 2),
            "environment": {
                "python": platform.python_version(),
                "numpy": np.__version__,
                "platform": platform.platform(),
            },
            "git_commit": git_commit(),
            "git_dirty": git_dirty(),
            **payload,
        }
        path = RESULTS / f"{self.name}.json"
        path.write_text(json.dumps(record, indent=2, default=_jsonable), encoding="utf-8")
        print(f"[{self.name}] done in {elapsed:.1f} s -> {path.relative_to(ROOT)}")
        return path


def _jsonable(o: Any) -> Any:
    if isinstance(o, (bool, np.bool_)):
        return bool(o)
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    raise TypeError(f"not JSON serialisable: {type(o)}")


def table(rows: list[dict[str, Any]], columns: list[tuple[str, str, str]]) -> str:
    """Render ``rows`` as a fixed-width table.

    ``columns`` is a list of ``(key, header, format)`` triples, e.g.
    ``("err_pct", "err. (%)", "{:.1f}")``.
    """
    widths = []
    for key, header, fmt in columns:
        cells = [fmt.format(r[key]) if r.get(key) is not None else "-" for r in rows]
        widths.append(max(len(header), *(len(c) for c in cells)) if cells else len(header))
    out = ["  ".join(h.rjust(w) for (_, h, _), w in zip(columns, widths))]
    out.append("  ".join("-" * w for w in widths))
    for r in rows:
        cells = []
        for (key, _, fmt), w in zip(columns, widths):
            v = r.get(key)
            cells.append((fmt.format(v) if v is not None else "-").rjust(w))
        out.append("  ".join(cells))
    return "\n".join(out)


def check(label: str, got: float, published: float, tol: float, unit: str = "") -> dict[str, Any]:
    """Compare a reproduced number against the value printed in the paper.

    Returns a dict for the JSON record and prints a PASS/FAIL line. ``tol`` is
    absolute, in the same unit as the values.
    """
    delta = abs(got - published)
    ok = delta <= tol
    print(
        f"    {'PASS' if ok else 'FAIL'}  {label}: reproduced {got:.4g}{unit}, "
        f"paper {published:.4g}{unit} (|d| = {delta:.3g}, tol {tol:g})"
    )
    return {
        "label": label,
        "reproduced": float(got),
        "published": float(published),
        "abs_delta": float(delta),
        "tolerance": float(tol),
        "pass": bool(ok),
    }

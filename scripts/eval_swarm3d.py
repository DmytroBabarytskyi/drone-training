"""Headless batch evaluation of the 3-D interception engagement.

Same engagement the windowed runner flies, without a renderer, over many
seeds. Reports detonation rate (the mesh got a drone inside lethal radius),
kill rate (the fragmentation actually worked), and the miss-distance
distribution that drives both.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from dronesim.core.config import load_config  # noqa: E402
from dronesim.swarm3d.engagement import Engagement  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="swarm3d/engagement")
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--out", default="docs/swarm3d_eval.json")
    args = parser.parse_args(argv)

    cfg = load_config(args.config)
    started = time.time()
    results = []

    for i in range(args.episodes):
        seed = args.seed + i
        r = Engagement(cfg, seed=seed).run()
        results.append(
            {
                "seed": seed,
                "detonated": r.detonated,
                "intercepted": r.intercepted,
                "drones_within_lethal": r.drones_within_lethal,
                "miss_distance_m": r.miss_distance_m,
                "formation_error_m": r.formation_error_m,
                "time_to_resolution_s": r.time_to_resolution_s,
                "phase_timeline": r.phase_timeline,
            }
        )
        print(
            f"seed {seed:3d}  det={int(r.detonated)} kill={int(r.intercepted)} "
            f"n={r.drones_within_lethal:2d}  miss={r.miss_distance_m:6.1f} m  "
            f"form_err={r.formation_error_m:5.1f} m",
            flush=True,
        )

    misses = np.array([r["miss_distance_m"] for r in results])
    form = np.array([r["formation_error_m"] for r in results])
    report = {
        "episodes": args.episodes,
        "detonation_rate": float(np.mean([r["detonated"] for r in results])),
        "kill_rate": float(np.mean([r["intercepted"] for r in results])),
        "median_miss_m": float(np.median(misses)),
        "p25_miss_m": float(np.percentile(misses, 25)),
        "median_formation_error_m": float(np.median(form)),
        "lethal_radius_m": float(cfg.warhead.lethal_radius_m),
        "wall_time_s": round(time.time() - started, 1),
        "results": results,
    }

    print()
    print(f"detonation rate          : {report['detonation_rate']:.2f}")
    print(f"kill rate                : {report['kill_rate']:.2f}")
    print(f"median miss distance     : {report['median_miss_m']:.1f} m "
          f"(lethal radius {report['lethal_radius_m']:.1f} m)")
    print(f"median formation error   : {report['median_formation_error_m']:.1f} m")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\nReport written to {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

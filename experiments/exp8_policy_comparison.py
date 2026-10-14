"""Experiment 8 — Tables VII and VIII: placement policies against the ceiling.

Five policies over the same 100 episode seeds through the identical
environment code path:

* a null policy that never moves (the resting lattice);
* centralised PPO at each of three trainer seeds;
* analytic hexagonal packing;
* the same analytic policy with short-range transit repulsion.

``run_episode`` and ``build_scene_and_reward`` are imported from the project's
own evaluation harness rather than reimplemented, so this is not a comparison
of evaluation harnesses. Intervals are percentile bootstrap over episodes, and
the comparison against the null policy is *paired*, because every policy sees
the same episodes and the episode is therefore a matched unit.

Both denominators are reported. The environment's separation limit is 2.5 m
against ``2r`` = 6 m, so coverage discs may overlap and the applicable law is
the covering one; the packing ceiling is carried alongside to show what
quoting it would have flattered.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

from _common import ROOT, Experiment, check, table

sys.path.insert(0, str(ROOT / "scripts"))

from dronesim.ml.swarm.analytic_policy import AnalyticPlacementPolicy
from dronesim.ml.swarm.coverage_laws import covering_coverage, planar_coverage
from dronesim.ml.swarm.swarm_env import SwarmInterceptEnv
from eval_swarm_intercept import build_scene_and_reward, run_episode  # noqa: E402

EPISODES = 100
SEED0 = 3000              # episode seeds 3000..3099, the harness default
BOOTSTRAP_RESAMPLES = 10_000

# The archived checkpoints live beside the experiments so a fresh clone can
# run this script; ``runs/`` is a working directory and is gitignored. The
# training run writes there, and the archived copies are what the paper's
# numbers were produced from.
_ARCHIVE = Path(__file__).resolve().parent / "checkpoints"
_WORKING = {
    "ppo_seed0": ROOT / "runs/rl/ppo_swarm_intercept/model.zip",
    "ppo_seed1": ROOT / "runs/rl/ppo_swarm_intercept/model_seed1.zip",
    "ppo_seed2": ROOT / "runs/rl/ppo_swarm_intercept/model_seed2.zip",
}
CHECKPOINTS = {
    key: (
        _ARCHIVE / f"ppo_swarm_intercept_{key.removeprefix('ppo_')}.zip"
        if (_ARCHIVE / f"ppo_swarm_intercept_{key.removeprefix('ppo_')}.zip").exists()
        else working
    )
    for key, working in _WORKING.items()
}

# Published Tables VII and VIII.
PUBLISHED = {
    "null": {"hit": 0.14, "ci": (0.08, 0.21), "coll": 0.00, "packing": 15.5, "covering": 14.2},
    "ppo_seed0": {"hit": 0.12, "ci": (0.06, 0.19), "coll": 0.78, "packing": 13.3, "covering": 12.1},
    "ppo_seed1": {"hit": 0.19, "ci": (0.12, 0.27), "coll": 0.74, "packing": 21.0, "covering": 19.2},
    "ppo_seed2": {"hit": 0.14, "ci": (0.08, 0.21), "coll": 0.61, "packing": 15.5, "covering": 14.2},
    "analytic_hex": {"hit": 0.84, "ci": (0.77, 0.91), "coll": 0.62, "packing": 92.9, "covering": 85.0},
    "analytic_hex_repulsion": {
        "hit": 0.79, "ci": (0.71, 0.87), "coll": 0.49, "packing": 87.3, "covering": 79.9,
    },
}
PUBLISHED_PAIRED = {
    "ppo_seed0": (-0.020, (-0.110, 0.070)),
    "ppo_seed1": (0.050, (-0.060, 0.160)),
    "ppo_seed2": (0.000, (-0.100, 0.100)),
    "analytic_hex": (0.700, (0.600, 0.790)),
    "analytic_hex_repulsion": (0.650, (0.550, 0.750)),
}
PUBLISHED_SIGMA = 4.57
PUBLISHED_PACKING_CEILING = 0.904
PUBLISHED_COVERING_CEILING = 0.988


class NullPolicy:
    """Never moves: a zero desired-velocity command every step."""

    def __init__(self, env: SwarmInterceptEnv):
        self._n = env.scene.n_drones

    def predict(self, obs, deterministic=True):
        return np.zeros(2 * self._n, dtype=np.float32), None


class AnalyticAdapter:
    """The analytic policy behind the harness's predict() interface.

    ``repulsion_gain`` 0 is plain hexagonal packing; the project's default of
    1.2 is the "+ transit repulsion" variant.
    """

    def __init__(self, env: SwarmInterceptEnv, repulsion_gain: float):
        self._env = env
        self._policy = AnalyticPlacementPolicy(
            env.scene.n_drones,
            env.scene.hit_radius_m,
            env.scene.min_separation_m,
            env.scene.max_speed_mps,
            env.scene.max_accel_mps2,
            repulsion_gain=repulsion_gain,
        )
        self._armed_for = None

    def predict(self, obs, deterministic=True):
        key = tuple(self._env._predicted_point)
        if key != self._armed_for:
            self._policy.reset(self._env._predicted_point)
            self._armed_for = key
        return self._policy.act(self._env._positions), None


def bootstrap_ci(values: np.ndarray, rng: np.random.Generator) -> tuple[float, float]:
    idx = rng.integers(0, len(values), size=(BOOTSTRAP_RESAMPLES, len(values)))
    means = values[idx].mean(axis=1)
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def paired_bootstrap(diff: np.ndarray, rng: np.random.Generator):
    idx = rng.integers(0, len(diff), size=(BOOTSTRAP_RESAMPLES, len(diff)))
    means = diff[idx].mean(axis=1)
    return (
        float(diff.mean()),
        float(np.percentile(means, 2.5)),
        float(np.percentile(means, 97.5)),
    )


def evaluate_policy(name: str, make_policy, scene_cfg, reward_cfg) -> dict:
    env = SwarmInterceptEnv(scene_cfg=scene_cfg, reward_cfg=reward_cfg)
    policy = make_policy(env)
    hits, colls, sigmas, reachable = [], [], [], []
    for i in range(EPISODES):
        result, _ = run_episode(policy, env, SEED0 + i, record=False)
        hits.append(1.0 if (result["hits"] or 0) > 0 else 0.0)
        colls.append(1.0 if result["collision_pairs_final_step"] > 0 else 0.0)
        sigmas.append(result["confidence_radius"] / np.sqrt(2.0))
        reachable.append(
            float(np.max(np.abs(result["true_crossing_point"])) <= scene_cfg.coverage_half_extent_m)
        )
    print(f"    {name:24s} hit {np.mean(hits):.2f}  coll {np.mean(colls):.2f}")
    return {
        "hit": np.array(hits),
        "coll": np.array(colls),
        "sigma_mean": float(np.mean(sigmas)),
        "reachable_fraction": float(np.mean(reachable)),
    }


def main() -> None:
    exp = Experiment(
        "exp8_policy_comparison",
        "Tables VII and VIII - placement policies against the applicable ceiling",
    )

    scene_cfg, reward_cfg = build_scene_and_reward()
    print(
        f"[{exp.name}] N={scene_cfg.n_drones} r={scene_cfg.hit_radius_m} m "
        f"s={scene_cfg.min_separation_m} m sector=+/-{scene_cfg.coverage_half_extent_m} m "
        f"window={scene_cfg.dt_s * scene_cfg.decision_steps:.1f} s"
    )

    missing = [k for k, p in CHECKPOINTS.items() if not p.exists()]
    if missing:
        raise SystemExit(
            f"missing checkpoints for {missing}. Expected them under "
            f"experiments/checkpoints/ (archived, in version control) or "
            f"runs/rl/ppo_swarm_intercept/ (working directory, gitignored)."
        )

    builders = {"null": NullPolicy}
    for key, path in CHECKPOINTS.items():
        def make(env, _p=path):
            from stable_baselines3 import PPO

            return PPO.load(str(_p), device="cpu")

        builders[key] = make
    builders["analytic_hex"] = lambda env: AnalyticAdapter(env, repulsion_gain=0.0)
    builders["analytic_hex_repulsion"] = lambda env: AnalyticAdapter(env, repulsion_gain=1.2)

    print()
    print(f"  {EPISODES} episodes, seeds {SEED0}..{SEED0 + EPISODES - 1}")
    raw = {
        name: evaluate_policy(name, build, scene_cfg, reward_cfg)
        for name, build in builders.items()
    }

    # --- the denominator ---------------------------------------------------
    sigma_mean = float(np.mean([v["sigma_mean"] for v in raw.values()]))
    reach_factor = float(np.min([v["reachable_fraction"] for v in raw.values()]))
    packing_ceiling = float(
        planar_coverage(scene_cfg.n_drones, sigma_mean, scene_cfg.hit_radius_m)
    )
    covering_ceiling = float(
        covering_coverage(scene_cfg.n_drones, sigma_mean, scene_cfg.hit_radius_m)
    )
    applicable = covering_ceiling if scene_cfg.min_separation_m < 2 * scene_cfg.hit_radius_m else packing_ceiling
    print()
    print(f"  mean sigma {sigma_mean:.2f} m, reachability factor {reach_factor:.3f}")
    print(
        f"  packing ceiling {reach_factor * packing_ceiling:.3f}, "
        f"covering ceiling {reach_factor * covering_ceiling:.3f} "
        f"(applicable, since s = {scene_cfg.min_separation_m} < 2r = "
        f"{2 * scene_cfg.hit_radius_m})"
    )

    # --- Table VII ---------------------------------------------------------
    rng = exp.rng()
    rows = []
    for name, v in raw.items():
        lo, hi = bootstrap_ci(v["hit"], rng)
        hit = float(v["hit"].mean())
        rows.append(
            {
                "policy": name,
                "hit_rate": round(hit, 3),
                "ci_lo": round(lo, 3),
                "ci_hi": round(hi, 3),
                "collisions": round(float(v["coll"].mean()), 3),
                "of_packing_pct": round(100.0 * hit / (reach_factor * packing_ceiling), 1),
                "of_covering_pct": round(100.0 * hit / (reach_factor * covering_ceiling), 1),
            }
        )
    ppo_mean = float(np.mean([r["hit_rate"] for r in rows if r["policy"].startswith("ppo")]))

    print()
    print(
        table(
            rows,
            [
                ("policy", "policy", "{:s}"),
                ("hit_rate", "hit rate", "{:.2f}"),
                ("ci_lo", "lo", "{:.2f}"),
                ("ci_hi", "hi", "{:.2f}"),
                ("collisions", "coll.", "{:.2f}"),
                ("of_packing_pct", "packing %", "{:.1f}"),
                ("of_covering_pct", "covering %", "{:.1f}"),
            ],
        )
    )
    print(f"    PPO mean over three trainer seeds: {ppo_mean:.2f}")

    # --- Table VIII: paired bootstrap against the null policy --------------
    paired = []
    for name, v in raw.items():
        if name == "null":
            continue
        diff = v["hit"] - raw["null"]["hit"]
        mean, lo, hi = paired_bootstrap(diff, rng)
        corr = (
            float(np.corrcoef(v["hit"], raw["null"]["hit"])[0, 1])
            if v["hit"].std() > 0 and raw["null"]["hit"].std() > 0
            else 0.0
        )
        paired.append(
            {
                "policy": name,
                "mean_difference": round(mean, 3),
                "ci_lo": round(lo, 3),
                "ci_hi": round(hi, 3),
                "contains_zero": bool(lo <= 0.0 <= hi),
                "episode_correlation_with_null": round(corr, 3),
            }
        )

    print()
    print("  paired bootstrap of the per-episode difference against the null policy")
    print(
        table(
            paired,
            [
                ("policy", "policy", "{:s}"),
                ("mean_difference", "mean diff", "{:+.3f}"),
                ("ci_lo", "lo", "{:+.3f}"),
                ("ci_hi", "hi", "{:+.3f}"),
                ("contains_zero", "contains 0", "{!s:s}"),
            ],
        )
    )

    print()
    print("  checks against the published tables")
    checks = [
        check("mean sigma", sigma_mean, PUBLISHED_SIGMA, 0.15, " m"),
        check("reachability factor", reach_factor, 1.000, 0.001),
        check("packing ceiling", packing_ceiling, PUBLISHED_PACKING_CEILING, 0.006),
        check("covering ceiling", covering_ceiling, PUBLISHED_COVERING_CEILING, 0.006),
        check("PPO mean over seeds", ppo_mean, 0.15, 0.03),
    ]
    by_name = {r["policy"]: r for r in rows}
    for name, pub in PUBLISHED.items():
        r = by_name[name]
        checks.append(check(f"{name}: hit rate", r["hit_rate"], pub["hit"], 0.05))
        checks.append(check(f"{name}: collisions", r["collisions"], pub["coll"], 0.08))
        checks.append(
            check(f"{name}: % of covering ceiling", r["of_covering_pct"], pub["covering"], 5.0, "%")
        )
    for p in paired:
        pub_mean, pub_ci = PUBLISHED_PAIRED[p["policy"]]
        checks.append(check(f"{p['policy']}: paired mean", p["mean_difference"], pub_mean, 0.05))
        expect_zero = pub_ci[0] <= 0.0 <= pub_ci[1]
        ok = p["contains_zero"] == expect_zero
        print(
            f"    {'PASS' if ok else 'FAIL'}  {p['policy']}: interval "
            f"{'contains' if p['contains_zero'] else 'excludes'} zero "
            f"(paper: {'contains' if expect_zero else 'excludes'})"
        )
        checks.append(
            {
                "label": f"{p['policy']}: zero inside interval matches paper",
                "reproduced": 1.0 if p["contains_zero"] else 0.0,
                "published": 1.0 if expect_zero else 0.0,
                "abs_delta": 0.0 if ok else 1.0,
                "tolerance": 0.0,
                "pass": bool(ok),
            }
        )

    exp.write(
        {
            "parameters": {
                "episodes": EPISODES,
                "episode_seeds": [SEED0, SEED0 + EPISODES - 1],
                "bootstrap_resamples": BOOTSTRAP_RESAMPLES,
                "n_drones": scene_cfg.n_drones,
                "hit_radius_m": scene_cfg.hit_radius_m,
                "min_separation_m": scene_cfg.min_separation_m,
                "coverage_half_extent_m": scene_cfg.coverage_half_extent_m,
                "window_s": scene_cfg.dt_s * scene_cfg.decision_steps,
                "checkpoints": {k: str(v.relative_to(ROOT)) for k, v in CHECKPOINTS.items()},
            },
            "denominator": {
                "mean_sigma_m": round(sigma_mean, 3),
                "reachability_factor": reach_factor,
                "packing_ceiling": round(packing_ceiling, 4),
                "covering_ceiling": round(covering_ceiling, 4),
                "applicable_ceiling": round(applicable, 4),
                "applicable_regime": "covering",
            },
            "table7": rows,
            "ppo_mean_hit_rate": round(ppo_mean, 3),
            "table8_paired": paired,
            "checks": checks,
            "all_checks_pass": all(c["pass"] for c in checks),
        }
    )


if __name__ == "__main__":
    main()

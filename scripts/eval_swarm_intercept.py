"""End-to-end evaluation of the trained swarm-intercept policy, mirroring
``eval_autonomous.py``'s approach (phase 7): run many seeded episodes
headless, report a hit-rate summary, and don't hide a bad outcome — the
report format is identical whether the policy is good or not.

Optionally renders a handful of episodes to PNG frame sequences (one
directory per episode) for turning into a video afterward with ffmpeg —
see ``render_frames_to_video`` for the exact command.

Usage:
    python scripts/eval_swarm_intercept.py --checkpoint runs/rl/ppo_swarm_intercept/model.zip
    python scripts/eval_swarm_intercept.py --render-episodes 3 --render-dir runs/swarm_render
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

# Same ordering constraint as eval_autonomous.py: torch/stable_baselines3
# first, before anything that could transitively pull in Panda3D. This
# script never imports Panda3D at all, but keeping the convention avoids a
# footgun if it grows to.
import torch  # noqa: F401,E402

from omegaconf import OmegaConf  # noqa: E402

from dronesim.core.config import load_config  # noqa: E402
from dronesim.ml.swarm.swarm_env import (  # noqa: E402
    SwarmInterceptEnv,
    SwarmRewardConfig,
    SwarmSceneConfig,
)


def build_scene_and_reward(config_name: str = "ml/swarm_intercept"):
    """Load scene/reward from YAML — NOT the dataclass defaults.

    Evaluating against the dataclass defaults while training reads the YAML
    silently compares two different scenarios (it surfaced here as an
    observation-shape mismatch only because the drone count differed; had only
    the kill radius or sector size been changed, the numbers would have been
    quietly wrong instead of erroring).
    """
    cfg = load_config(config_name)
    scene_dict = OmegaConf.to_container(cfg.scene, resolve=True)
    scene_dict["target_speed_range_mps"] = tuple(scene_dict["target_speed_range_mps"])
    scene_dict["target_start_distance_range_m"] = tuple(
        scene_dict["target_start_distance_range_m"]
    )
    reward_dict = OmegaConf.to_container(cfg.reward, resolve=True)
    return SwarmSceneConfig(**scene_dict), SwarmRewardConfig(**reward_dict)


class _AnalyticAdapter:
    """Wraps the analytic policy in the same predict() interface PPO exposes,
    so both run through the identical evaluation path."""

    def __init__(self, env: SwarmInterceptEnv):
        from dronesim.ml.swarm.analytic_policy import AnalyticPlacementPolicy

        self._env = env
        self._policy = AnalyticPlacementPolicy(
            env.scene.n_drones,
            env.scene.hit_radius_m,
            env.scene.min_separation_m,
            env.scene.max_speed_mps,
            env.scene.max_accel_mps2,
        )
        self._armed_for = None

    def predict(self, obs, deterministic=True):
        # The formation depends only on the prediction, which is fixed for the
        # episode — rebuild it whenever a new episode's prediction appears.
        key = tuple(self._env._predicted_point)
        if key != self._armed_for:
            self._policy.reset(self._env._predicted_point)
            self._armed_for = key
        return self._policy.act(self._env._positions), None


def run_episode(model, env: SwarmInterceptEnv, seed: int, record: bool):
    obs, _ = env.reset(seed=seed)
    trajectory = []
    if record:
        trajectory.append(
            {
                "positions": env._positions.copy(),
                "predicted_point": env._predicted_point.copy(),
                "confidence_radius": env._confidence_radius,
            }
        )

    terminated = False
    info = {}
    while not terminated:
        action, _ = model.predict(obs, deterministic=True)
        obs, _reward, terminated, _truncated, info = env.step(action)
        if record:
            trajectory.append({"positions": env._positions.copy()})

    return {
        "seed": seed,
        "hits": info["hits"],
        "collision_pairs_final_step": info["collision_pairs"],
        "true_crossing_point": env._true_crossing_point.tolist(),
        "predicted_point": env._predicted_point.tolist(),
        "confidence_radius": env._confidence_radius,
        "final_positions": env._positions.tolist(),
    }, trajectory


def measure_density_vs_distance(results: list[dict], n_bands: int = 5) -> dict:
    """Empirically measure how drone density actually varies with distance
    from the predicted point, in units of sigma (the Gaussian model's
    per-axis std, sigma = confidence_radius / sqrt(2) — see
    ``swarm_env.py::_make_coverage_checkpoints``).

    This answers the question directly rather than assuming a shape: bin
    every drone's final distance from its episode's predicted point (as a
    fraction of that episode's sigma, so episodes with different prediction
    uncertainty are comparable), then convert each band's drone count into
    an actual areal density using the band's true ring area — same units the
    reward was shaped in, not a guessed ratio.
    """
    band_edges = np.linspace(0.0, 2.5, n_bands + 1)  # in units of sigma
    counts = np.zeros(n_bands)
    total_episodes = len(results)

    for r in results:
        sigma = r["confidence_radius"] / np.sqrt(2.0)
        if sigma < 1e-6:
            continue
        predicted = np.array(r["predicted_point"])
        positions = np.array(r["final_positions"])
        dist_sigma = np.linalg.norm(positions - predicted, axis=1) / sigma
        for i in range(n_bands):
            counts[i] += np.sum((dist_sigma >= band_edges[i]) & (dist_sigma < band_edges[i + 1]))

    mean_sigma = float(
        np.mean([r["confidence_radius"] / np.sqrt(2.0) for r in results if r["confidence_radius"] > 1e-6])
    )
    bands = []
    for i in range(n_bands):
        inner_m = band_edges[i] * mean_sigma
        outer_m = band_edges[i + 1] * mean_sigma
        ring_area_m2 = np.pi * (outer_m**2 - inner_m**2)
        density_per_m2 = counts[i] / total_episodes / ring_area_m2 if ring_area_m2 > 1e-9 else 0.0
        bands.append(
            {
                "sigma_range": [float(band_edges[i]), float(band_edges[i + 1])],
                "approx_m_range": [round(float(inner_m), 1), round(float(outer_m), 1)],
                "mean_drones_in_band": round(float(counts[i] / total_episodes), 2),
                "density_drones_per_m2": round(float(density_per_m2), 3),
            }
        )
    return {"mean_sigma_m": round(mean_sigma, 2), "bands": bands}


def evaluate(checkpoint: str, episodes: int, seed0: int, render_episodes: int, render_dir: str,
             policy_kind: str = "learned"):
    scene_cfg, reward_cfg = build_scene_and_reward()
    env = SwarmInterceptEnv(scene_cfg=scene_cfg, reward_cfg=reward_cfg)

    if policy_kind == "analytic":
        model = _AnalyticAdapter(env)
    else:
        from stable_baselines3 import PPO

        model = PPO.load(checkpoint)

    results = []
    for i in range(episodes):
        seed = seed0 + i
        record = i < render_episodes
        result, trajectory = run_episode(model, env, seed, record)
        results.append(result)
        if record:
            out_dir = Path(render_dir) / f"episode_{seed}"
            save_trajectory(out_dir, env.scene, trajectory, result)
        print(f"seed={seed:5d}  hits={result['hits']}  collisions={result['collision_pairs_final_step']}")

    hit_rate = sum(1 for r in results if r["hits"] and r["hits"] > 0) / episodes
    mean_hits = sum(r["hits"] for r in results) / episodes
    any_collision_rate = sum(1 for r in results if r["collision_pairs_final_step"] > 0) / episodes
    density_profile = measure_density_vs_distance(results)

    report = {
        "checkpoint": checkpoint if policy_kind == "learned" else "analytic-hex-packing",
        "policy": policy_kind,
        "episodes": episodes,
        "hit_rate": hit_rate,
        "mean_hits_per_episode": mean_hits,
        "any_collision_rate": any_collision_rate,
        "density_vs_distance": density_profile,
        "results": results,
    }
    print(f"\nmean sigma (per-axis prediction std): {density_profile['mean_sigma_m']} m")
    print("measured density vs. distance from predicted point:")
    for band in density_profile["bands"]:
        lo_m, hi_m = band["approx_m_range"]
        print(
            f"  {band['sigma_range'][0]:.1f}-{band['sigma_range'][1]:.1f} sigma "
            f"(~{lo_m}-{hi_m} m):  {band['mean_drones_in_band']:.2f} drones/episode  "
            f"->  {band['density_drones_per_m2']:.3f} drones/m^2"
        )
    print(
        f"\nhit_rate={hit_rate:.2f}  mean_hits={mean_hits:.2f}  "
        f"any_collision_rate={any_collision_rate:.2f}"
    )
    return report


def save_trajectory(out_dir: Path, scene: SwarmSceneConfig, trajectory: list[dict], result: dict):
    """Save one episode's per-step state as JSON — kept separate from
    rendering so the render step can be redone/restyled without re-running
    the (slow, stochastic) policy rollout."""
    out_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "scene": {
            "coverage_half_extent_m": scene.coverage_half_extent_m,
            "hit_radius_m": scene.hit_radius_m,
            "min_separation_m": scene.min_separation_m,
        },
        "result": result,
        "steps": [
            {"positions": step["positions"].tolist()} for step in trajectory
        ],
        "predicted_point": trajectory[0]["predicted_point"].tolist(),
        "confidence_radius": trajectory[0]["confidence_radius"],
    }
    (out_dir / "trajectory.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")


def render_frames_to_video(episode_dir: Path, fps: int = 20) -> Path:
    """Render one saved trajectory to an mp4. Requires matplotlib and ffmpeg
    on PATH. Kept as a separate pass from evaluation so a rendering bug
    never means re-running the policy."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Circle

    payload = json.loads((episode_dir / "trajectory.json").read_text(encoding="utf-8"))
    extent = payload["scene"]["coverage_half_extent_m"]
    hit_radius = payload["scene"]["hit_radius_m"]
    predicted_point = np.array(payload["predicted_point"])
    confidence_radius = payload["confidence_radius"]
    true_point = np.array(payload["result"]["true_crossing_point"])
    hits = payload["result"]["hits"]
    steps = payload["steps"]
    n_steps = len(steps)

    frames_dir = episode_dir / "frames"
    frames_dir.mkdir(exist_ok=True)

    for i, step in enumerate(steps):
        positions = np.array(step["positions"])
        fig, ax = plt.subplots(figsize=(8, 8), dpi=120)
        ax.set_facecolor("#0b0f14")
        fig.patch.set_facecolor("#0b0f14")
        ax.set_xlim(-extent * 1.15, extent * 1.15)
        ax.set_ylim(-extent * 1.15, extent * 1.15)
        ax.set_aspect("equal")
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)

        ax.add_patch(
            plt.Rectangle(
                (-extent, -extent), 2 * extent, 2 * extent,
                fill=False, edgecolor="#2a3542", linewidth=1.5,
            )
        )
        ax.add_patch(
            Circle(predicted_point, confidence_radius, fill=False, edgecolor="#3a6fb0",
                   linestyle="--", linewidth=1.5, alpha=0.8)
        )
        ax.scatter(*predicted_point, marker="+", s=200, color="#3a6fb0", linewidths=2)

        is_last = i == n_steps - 1
        if is_last:
            ax.add_patch(Circle(true_point, hit_radius, fill=False, edgecolor="#e0483e", linewidth=2))
            ax.scatter(*true_point, marker="x", s=220, color="#e0483e", linewidths=3, zorder=5)

        dist_to_truth = np.linalg.norm(positions - true_point, axis=1) if is_last else None
        colors = []
        for j in range(len(positions)):
            if is_last and dist_to_truth[j] <= hit_radius:
                colors.append("#59d97a")
            else:
                colors.append("#e8e8e8")
        ax.scatter(positions[:, 0], positions[:, 1], s=90, color=colors, zorder=4,
                   edgecolors="#0b0f14", linewidths=1.0)

        title = f"t = {i} / {n_steps - 1}"
        if is_last:
            title += f"   —   {hits} drone(s) on target" if hits else "   —   MISS"
        ax.set_title(title, color="#c8d2dc", fontsize=14, pad=12)

        fig.savefig(frames_dir / f"frame_{i:04d}.png", facecolor=fig.get_facecolor())
        plt.close(fig)

    # Hold on the final frame for a beat so the outcome is readable.
    last_frame = frames_dir / f"frame_{n_steps - 1:04d}.png"
    for extra in range(1, int(fps * 1.5)):
        (frames_dir / f"frame_{n_steps - 1 + extra:04d}.png").write_bytes(last_frame.read_bytes())

    out_path = episode_dir / "episode.mp4"
    subprocess.run(
        [
            "ffmpeg", "-y", "-framerate", str(fps),
            "-i", str(frames_dir / "frame_%04d.png"),
            "-c:v", "libx264", "-pix_fmt", "yuv420p",
            str(out_path),
        ],
        check=True,
        capture_output=True,
    )
    return out_path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", default="runs/rl/ppo_swarm_intercept/model.zip")
    parser.add_argument("--episodes", type=int, default=50)
    parser.add_argument("--seed", type=int, default=3000)
    parser.add_argument("--render-episodes", type=int, default=0)
    parser.add_argument("--render-dir", default="runs/swarm_render")
    parser.add_argument("--out", default="docs/swarm_intercept_eval.json")
    parser.add_argument("--render-only", action="store_true", help="Skip eval, just render already-saved trajectories in --render-dir")
    parser.add_argument("--policy", choices=["learned", "analytic"], default="learned",
                        help="Which placement policy to evaluate")
    args = parser.parse_args(argv)

    if args.render_only:
        for episode_dir in sorted(Path(args.render_dir).glob("episode_*")):
            print(f"Rendering {episode_dir} ...")
            video_path = render_frames_to_video(episode_dir)
            print(f"  -> {video_path}")
        return 0

    t0 = time.time()
    report = evaluate(args.checkpoint, args.episodes, args.seed, args.render_episodes,
                      args.render_dir, args.policy)
    report["wall_time_s"] = time.time() - t0

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Report written to {out_path}")

    if args.render_episodes > 0:
        for episode_dir in sorted(Path(args.render_dir).glob("episode_*")):
            print(f"Rendering {episode_dir} ...")
            video_path = render_frames_to_video(episode_dir)
            print(f"  -> {video_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

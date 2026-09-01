"""Оцінка навченого RL-агента: success rate, час до цілі, к-сть крашів (фаза 6,
критерій приймання: узгоджений success rate на НОВИХ розкладках цілей)."""

from __future__ import annotations

import argparse

import numpy as np


def evaluate(
    checkpoint_path: str,
    env_config: str = "ml/rl_strike",
    n_episodes: int = 20,
    seed: int = 1000,
) -> dict:
    """Прогнати навченого агента ``n_episodes`` разів (нові seed -> нові розкладки
    цілей) і повернути ``{success_rate, crash_rate, mean_time_to_hit_s, episodes}``."""
    from stable_baselines3 import PPO

    from dronesim.ml.rl.env import DroneEnv

    model = PPO.load(checkpoint_path)
    env = DroneEnv(config_name=env_config)

    hits = 0
    crashes = 0
    times_to_hit = []

    for i in range(n_episodes):
        obs, _ = env.reset(seed=seed + i)
        terminated = truncated = False
        steps = 0
        info: dict = {}
        while not (terminated or truncated):
            action, _ = model.predict(obs, deterministic=True)
            obs, _reward, terminated, truncated, info = env.step(action)
            steps += 1
        if info.get("hit"):
            hits += 1
            times_to_hit.append(steps * env.step_duration_s)
        if info.get("crashed"):
            crashes += 1

    return {
        "success_rate": hits / n_episodes,
        "crash_rate": crashes / n_episodes,
        "mean_time_to_hit_s": float(np.mean(times_to_hit)) if times_to_hit else None,
        "episodes": n_episodes,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Оцінити навченого RL-агента (фаза 6)")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--config", default="ml/rl_strike")
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--seed", type=int, default=1000)
    args = parser.parse_args(argv)

    metrics = evaluate(args.checkpoint, args.config, args.episodes, args.seed)
    print(
        f"success_rate={metrics['success_rate']:.2f}  crash_rate={metrics['crash_rate']:.2f}  "
        f"mean_time_to_hit_s={metrics['mean_time_to_hit_s']}  episodes={metrics['episodes']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

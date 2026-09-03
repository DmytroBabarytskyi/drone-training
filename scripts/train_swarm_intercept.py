"""Train PPO (Stable-Baselines3) on ``SwarmInterceptEnv``.

Mirrors ``dronesim.ml.rl.train`` (single-quad training) in structure and
conventions, but targets the swarm formation-control environment instead —
kept as a separate script rather than folded into ``train.py`` because the
two environments take different config shapes (scene/reward sections here
vs. flat fields there) and there is no shared training logic worth
abstracting for a first prototype.
"""

from __future__ import annotations

import argparse

from omegaconf import OmegaConf

from dronesim.core.config import REPO_ROOT, load_config


def _scene_and_reward_from_cfg(cfg):
    from dronesim.ml.swarm.swarm_env import SwarmRewardConfig, SwarmSceneConfig

    scene_dict = OmegaConf.to_container(cfg.scene, resolve=True)
    scene_dict["target_speed_range_mps"] = tuple(scene_dict["target_speed_range_mps"])
    scene_dict["target_start_distance_range_m"] = tuple(
        scene_dict["target_start_distance_range_m"]
    )
    reward_dict = OmegaConf.to_container(cfg.reward, resolve=True)
    return SwarmSceneConfig(**scene_dict), SwarmRewardConfig(**reward_dict)


def train(config_name: str = "ml/ppo_swarm_intercept") -> str:
    """Train PPO on the swarm-intercept env; returns the saved checkpoint path."""
    from stable_baselines3 import PPO
    from stable_baselines3.common.monitor import Monitor

    from dronesim.ml.rl.policies import policy_kwargs_from_config
    from dronesim.ml.swarm.swarm_env import SwarmInterceptEnv

    cfg = load_config(config_name)
    env_cfg = load_config(str(cfg.env_config))
    scene_cfg, reward_cfg = _scene_and_reward_from_cfg(env_cfg)

    log_dir = REPO_ROOT / str(cfg.log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)

    env = Monitor(
        SwarmInterceptEnv(scene_cfg=scene_cfg, reward_cfg=reward_cfg),
        filename=str(log_dir / "monitor.csv"),
    )

    model = PPO(
        "MlpPolicy",
        env,
        learning_rate=float(cfg.learning_rate),
        n_steps=int(cfg.n_steps),
        batch_size=int(cfg.batch_size),
        gamma=float(cfg.gamma),
        seed=int(cfg.seed),
        policy_kwargs=policy_kwargs_from_config(cfg.policy),
        verbose=1,
    )
    model.learn(total_timesteps=int(cfg.total_timesteps))

    checkpoint_path = REPO_ROOT / str(cfg.checkpoint_path)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    model.save(str(checkpoint_path))
    return str(checkpoint_path)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Train PPO on SwarmInterceptEnv")
    parser.add_argument("--config", default="ml/ppo_swarm_intercept")
    args = parser.parse_args(argv)

    checkpoint_path = train(args.config)
    print(f"Saved checkpoint: {checkpoint_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

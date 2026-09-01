"""Навчання PPO (Stable-Baselines3) на ``DroneEnv`` (фаза 6).

Крива навчання (ROADMAP): спершу "телепорт-obs" (``configs/ml/rl_strike.yaml``,
дефолтний ``DroneEnv``) — агент бачить ІДЕАЛЬНУ відносну позу цілі. Заміна на
детекції YOLO — окрема, пізніша ітерація поверх того самого контракту дій.
"""

from __future__ import annotations

import argparse

from dronesim.core.config import REPO_ROOT, load_config


def train(config_name: str = "ml/ppo_strike") -> str:
    """Навчити PPO за конфігом; повертає шлях до збереженого чекпойнта."""
    from stable_baselines3 import PPO
    from stable_baselines3.common.monitor import Monitor

    from dronesim.ml.rl.env import DroneEnv
    from dronesim.ml.rl.policies import policy_kwargs_from_config

    cfg = load_config(config_name)
    log_dir = REPO_ROOT / str(cfg.log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)

    # Monitor пише monitor.csv (епізод: reward, довжина, час) — це і є "крива
    # навчання" критерію приймання; tensorboard навмисно НЕ використовується
    # (зайва важка залежність, яку не додали в extras "ml", docs/DECISIONS.md).
    env = Monitor(DroneEnv(config_name=str(cfg.env_config)), filename=str(log_dir / "monitor.csv"))

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
    parser = argparse.ArgumentParser(description="Навчити PPO на DroneEnv (фаза 6)")
    parser.add_argument("--config", default="ml/ppo_strike")
    args = parser.parse_args(argv)

    checkpoint_path = train(args.config)
    print(f"Збережено чекпойнт: {checkpoint_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

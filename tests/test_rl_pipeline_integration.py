"""Інтеграційний тест RL-конвеєра фази 6: навчання (мало кроків) -> оцінка.

НАВМИСНО зменшений масштаб (кілька тисяч timesteps) — перевірка КОРЕКТНОСТІ
З'ЄДНАННЯ (PPO навчається на DroneEnv, чекпойнт зберігається й завантажується,
eval() рахує метрики), не повноцінна збіжність до success rate ≥70% (це —
окремий, довший прогін, задокументований у docs/DECISIONS.md)."""

from __future__ import annotations

import pytest

from dronesim.ml.rl.eval import evaluate

pytestmark = pytest.mark.slow


def test_train_tiny_then_eval_runs_end_to_end(tmp_path):
    from stable_baselines3 import PPO
    from stable_baselines3.common.monitor import Monitor

    from dronesim.ml.rl.env import DroneEnv
    from dronesim.ml.rl.policies import policy_kwargs_from_config
    from omegaconf import OmegaConf

    env = Monitor(DroneEnv(config_name="ml/rl_strike"))
    model = PPO(
        "MlpPolicy",
        env,
        n_steps=128,
        batch_size=32,
        seed=0,
        policy_kwargs=policy_kwargs_from_config(OmegaConf.create({"net_arch": [16, 16]})),
        verbose=0,
    )
    model.learn(total_timesteps=256)

    checkpoint_path = tmp_path / "model.zip"
    model.save(str(checkpoint_path))

    metrics = evaluate(str(checkpoint_path), env_config="ml/rl_strike", n_episodes=2, seed=999)
    assert 0.0 <= metrics["success_rate"] <= 1.0
    assert 0.0 <= metrics["crash_rate"] <= 1.0
    assert metrics["episodes"] == 2

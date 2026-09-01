"""Політика для PPO (фаза 6): MLP для векторних спостережень ("телепорт-obs").

CNN-політика — точка розширення на потім, коли спостереження почне містити
кадр (детекції YOLO замість ідеальної пози цілі, ROADMAP "Крива навчання").
Тут явно фіксуємо вибір і гіперпараметри мережі з конфігу, а не хардкодимо їх
(SKILL.md правило 3).
"""

from __future__ import annotations


def policy_kwargs_from_config(cfg) -> dict:
    """Гіперпараметри архітектури політики (розміри прихованих шарів MLP)."""
    net_arch = list(cfg.net_arch)
    return {"net_arch": net_arch}


def select_policy_name(observation_is_image: bool = False) -> str:
    """"MlpPolicy" для векторних спостережень (фаза 6, teleport-obs);
    "CnnPolicy" — коли спостереження міститиме кадр (пізніша ітерація)."""
    return "CnnPolicy" if observation_is_image else "MlpPolicy"

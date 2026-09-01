"""Інтерфейс сценарію (місії). Реалізації: gate_race, patrol, strike_range.

Сценарій — це ігровий контур поверх симуляції: він знає про цілі/ворота, рахує
очки й вирішує, коли епізод завершено. Використовується і людиною (геймплей), і
RL-середовищем (``ml/rl/env.py`` обгортає сценарій у Gymnasium.Env).

ЕТИКА (SKILL.md): «влучання» — суто подія симуляції (зближення з віртуальним
маркером у межах порогу) + очки. Ніякої реальної зброї.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

from dronesim.core.contracts import VehicleState


@dataclass
class ScenarioResult:
    """Підсумок кроку/епізоду сценарію."""

    reward: float = 0.0     # для RL
    score: float = 0.0      # ігрові очки для людини
    done: bool = False
    info: dict | None = None


class Scenario(ABC):
    """Абстрактна місія."""

    @abstractmethod
    def reset(self, seed: int | None = None) -> None:
        """Розставити цілі/ворота (детерміновано за seed), обнулити рахунок."""

    @abstractmethod
    def step(self, state: VehicleState) -> ScenarioResult:
        """Оцінити поточний стан апарата: очки, винагорода, чи завершено."""

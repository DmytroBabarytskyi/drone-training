"""Скриптовані ракурси навколо цілей — домен-рандомізація пози камери (фаза 5).

Детерміновано за ``seed`` (SKILL.md правило 4): рандомізує азимут/радіус/висоту
навколо кожної цілі, щоб датасет покривав різні кути огляду й дистанції без
потреби в реальному польоті/фізиці (``ml/data/recorder.py`` розміщує апарат
кінематично в кожному ``Viewpoint``, не рахуючи PhysicsWorld).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class Viewpoint:
    """Одна поза для запису кадру: позиція апарата + точка, куди дивитись."""

    pos: np.ndarray
    look_at: np.ndarray


def generate_viewpoints(
    target_pos: np.ndarray,
    n: int,
    radius_range: tuple[float, float],
    altitude_range: tuple[float, float],
    seed: int,
) -> list[Viewpoint]:
    """Згенерувати ``n`` детермінованих (за ``seed``) ракурсів навколо ``target_pos``.

    Азимут — рівномірно по колу; радіус і висота (відносно цілі) — рівномірно
    в заданих діапазонах. ``look_at`` завжди сама ``target_pos`` (апарат
    гарантовано дивиться на ціль з кожного ракурсу).
    """
    rng = np.random.default_rng(seed)
    viewpoints = []
    for _ in range(n):
        azimuth = rng.uniform(0.0, 2.0 * np.pi)
        radius = rng.uniform(*radius_range)
        altitude_offset = rng.uniform(*altitude_range)
        offset = np.array(
            [radius * np.cos(azimuth), radius * np.sin(azimuth), altitude_offset]
        )
        viewpoints.append(Viewpoint(pos=target_pos + offset, look_at=target_pos.copy()))
    return viewpoints

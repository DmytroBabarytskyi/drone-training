"""Навчальні цілі й ворота — маркери з відомими координатами (фаза 4).

ЕТИКА (SKILL.md розділ 0): усі "цілі"/"ворота" тут — суто віртуальні маркери
в симуляції. "Влучання"/"проходження воріт" — геометричні події (дистанція,
кут, перетин площини), не фізика зброї чи зіткнень.

Цілі й ворота НЕ є фізичними тілами Bullet: позиція обчислюється тут, чисто й
детерміновано за часом (для рухомих цілей) — рендер (render/visuals.py) лише
малює маркер за вже обчисленою позицією; фізика в цьому не бере участі
(правило "рендер/дані не рахують фізику й навпаки", SKILL.md розділ 3.6).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class Target:
    """Навчальна ціль. Якщо ``moves_to`` задано — гойдається між ним і ``base_pos``
    за гладкою (косинусоїдальною) траєкторією з періодом ``period_s``.

    ``kind`` (доп. фаза "реальні моделі") — яка ВІЗУАЛЬНА модель представляє
    ціль (``app.py::_sync_target_markers`` обирає будівник за цим полем):
    ``"marker"`` (типово, дефолт ЦЬОГО dataclass — bullseye-картка,
    ``patrol.py`` не перевизначає) або ``"vehicle"``/``"artillery"``
    (``strike_range.py`` явно ставить дефолт ``"vehicle"`` для СВОЇХ цілей —
    техніка/артилерія з ефектом знищення при ураженні, НЕ bullseye-recolor).
    """

    target_id: int
    base_pos: np.ndarray
    radius: float = 1.5
    moves_to: np.ndarray | None = None
    period_s: float = 10.0
    hit: bool = False
    kind: str = "marker"

    def position_at(self, t: float) -> np.ndarray:
        """Позиція цілі в момент часу ``t`` (чиста функція — детерміновано)."""
        if self.moves_to is None:
            return self.base_pos
        frac = 0.5 * (1.0 - np.cos(2.0 * np.pi * t / self.period_s))
        return self.base_pos + frac * (self.moves_to - self.base_pos)


@dataclass
class Gate:
    """Ворота для гоночного сценарію: центр, нормаль напрямку проходження, радіус кільця.

    ``normal`` — напрямок ПРАВИЛЬНОГО прольоту (з боку "до" в бік "після"),
    нормований. Проходження зараховується лише в цьому напрямку (не назад).
    """

    gate_id: int
    center: np.ndarray
    normal: np.ndarray
    radius: float = 2.5

    def _signed_distance(self, pos: np.ndarray) -> float:
        return float(np.dot(pos - self.center, self.normal))

    def passes_through(self, prev_pos: np.ndarray, curr_pos: np.ndarray) -> bool:
        """True, якщо відрізок ``prev_pos -> curr_pos`` перетнув площину воріт
        у правильному напрямку (``normal``) в межах кільця радіусом ``radius``."""
        d_prev = self._signed_distance(prev_pos)
        d_curr = self._signed_distance(curr_pos)
        if d_prev >= 0.0 or d_curr < 0.0:
            return False  # не перетнули площину саме в напрямку normal (ззаду -> спереду)

        frac = d_prev / (d_prev - d_curr)  # d_prev<0, d_curr>=0 -> frac у [0,1]
        crossing = prev_pos + frac * (curr_pos - prev_pos)
        offset = crossing - self.center
        lateral = offset - np.dot(offset, self.normal) * self.normal
        return float(np.linalg.norm(lateral)) <= self.radius


def check_hit(
    target: Target,
    vehicle_pos: np.ndarray,
    vehicle_forward: np.ndarray,
    t: float,
    max_angle_rad: float,
) -> bool:
    """Геометрична перевірка "влучання": дистанція ≤ ``target.radius`` І кут між
    носом апарата й напрямком на ціль ≤ ``max_angle_rad`` (імітація прицілювання).

    Жодної фізики зброї — суто геометрія (дистанція+кут), як зазначено в ROADMAP.md.
    """
    target_pos = target.position_at(t)
    to_target = target_pos - vehicle_pos
    dist = float(np.linalg.norm(to_target))
    if dist > target.radius:
        return False
    if dist < 1e-6:
        return True  # апарат практично в центрі цілі — кут не визначений, зараховуємо

    forward_norm = float(np.linalg.norm(vehicle_forward))
    cos_angle = float(np.dot(vehicle_forward, to_target)) / (forward_norm * dist)
    cos_angle = min(1.0, max(-1.0, cos_angle))
    angle = np.arccos(cos_angle)
    return bool(angle <= max_angle_rad)

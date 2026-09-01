"""Полігон: навести апарат на віртуальну ціль і "влучити" — критерій приймання
фази 4: детермінує влучання за геометрією (дистанція+кут), лог подій у файл.

ЕТИКА (SKILL.md розділ 0): "влучання" тут — геометрична подія (зближення в межах
``radius`` + кут прицілювання в межах ``max_aim_angle_deg``), не фізика зброї.
Кожна ціль зараховується не більше одного разу. Сценарій завершується, коли всі
цілі уражені, або спливає ``time_limit_s``.
"""

from __future__ import annotations

import numpy as np

from dronesim.core.config import load_config
from dronesim.core.contracts import VehicleState
from dronesim.scenarios.base import Scenario, ScenarioResult
from dronesim.utils.math3d import vehicle_forward_world
from dronesim.world.targets import Target, check_hit

_DEG2RAD = np.pi / 180.0


class StrikeRangeScenario(Scenario):
    """Полігон з нерухомими й рухомими цілями."""

    def __init__(self, config_name: str = "scenarios/strike_range"):
        self.cfg = load_config(config_name)
        self.max_aim_angle_rad = float(self.cfg.max_aim_angle_deg) * _DEG2RAD
        self.time_limit_s = float(self.cfg.time_limit_s)
        self.targets: list[Target] = []
        self.events: list[dict] = []
        self._finished = False

    def reset(self, seed: int | None = None) -> None:
        self._finished = False
        self.targets = [
            Target(
                target_id=i,
                base_pos=np.array(t.pos, dtype=np.float64),
                radius=float(t.radius),
                moves_to=(np.array(t.moves_to, dtype=np.float64) if "moves_to" in t else None),
                period_s=float(t.get("period_s", 10.0)),
                # Дефолт "vehicle" (НЕ dataclass-дефолт "marker") — цілі
                # strike_range показуються як техніка/артилерія з ефектом
                # знищення (доп. фаза "реальні моделі"), на відміну від
                # patrol (розвідка/спостереження, не "ураження").
                kind=str(t.get("kind", "vehicle")),
            )
            for i, t in enumerate(self.cfg.targets)
        ]
        self.events = []

    def step(self, state: VehicleState) -> ScenarioResult:
        forward = vehicle_forward_world(state.quat)
        reward = 0.0

        for target in self.targets:
            if target.hit:
                continue
            if check_hit(target, state.pos, forward, state.t, self.max_aim_angle_rad):
                target.hit = True
                reward += 1.0
                self.events.append({"t": state.t, "event": "hit", "target_id": target.target_id})

        hits = sum(1 for t in self.targets if t.hit)
        all_hit = hits == len(self.targets)
        timed_out = state.t >= self.time_limit_s

        done = all_hit or timed_out
        if done and not self._finished:
            self._finished = True
            self.events.append(
                {"t": state.t, "event": "finish", "hits": hits, "total": len(self.targets)}
            )

        return ScenarioResult(
            reward=reward,
            score=float(hits),
            done=done,
            info={"hits": hits, "total": len(self.targets), "timed_out": timed_out},
        )

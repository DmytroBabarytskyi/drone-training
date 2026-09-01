"""Патруль зони: цілі з'являються за розкладом, апарат "помічає" їх прольотом поруч.

На відміну від strike_range (дистанція+кут прицілювання), тут лише дистанція —
патруль про ВИЯВЛЕННЯ цілей (розвідка), не про наведення. Цілі з'являються
детерміновано за часом (``t_spawn`` з конфігу), не всі одразу — імітує "появу"
цілей під час облітання зони. Сценарій завершується, коли всі заспавнені цілі
поміченено (і розклад вичерпано), або спливає ``duration_s``.
"""

from __future__ import annotations

import numpy as np

from dronesim.core.config import load_config
from dronesim.core.contracts import VehicleState
from dronesim.scenarios.base import Scenario, ScenarioResult
from dronesim.world.targets import Target


class PatrolScenario(Scenario):
    """Облітання зони з появою цілей за розкладом."""

    def __init__(self, config_name: str = "scenarios/patrol"):
        self.cfg = load_config(config_name)
        self.duration_s = float(self.cfg.duration_s)
        self._spawn_schedule = list(self.cfg.spawns)
        self.targets: list[Target] = []
        self.events: list[dict] = []
        self._next_spawn_index = 0

    def reset(self, seed: int | None = None) -> None:
        self.targets = []
        self.events = []
        self._next_spawn_index = 0

    def _spawn_due_targets(self, t: float) -> None:
        while (
            self._next_spawn_index < len(self._spawn_schedule)
            and t >= float(self._spawn_schedule[self._next_spawn_index].t_spawn)
        ):
            spec = self._spawn_schedule[self._next_spawn_index]
            target = Target(
                target_id=self._next_spawn_index,
                base_pos=np.array(spec.pos, dtype=np.float64),
                radius=float(spec.radius),
            )
            self.targets.append(target)
            self.events.append({"t": t, "event": "spawn", "target_id": target.target_id})
            self._next_spawn_index += 1

    def step(self, state: VehicleState) -> ScenarioResult:
        self._spawn_due_targets(state.t)

        reward = 0.0
        for target in self.targets:
            if target.hit:
                continue
            dist = float(np.linalg.norm(target.position_at(state.t) - state.pos))
            if dist <= target.radius:
                target.hit = True
                reward = 1.0
                self.events.append({"t": state.t, "event": "observed", "target_id": target.target_id})

        observed = sum(1 for t in self.targets if t.hit)
        schedule_exhausted = self._next_spawn_index >= len(self._spawn_schedule)
        all_observed = schedule_exhausted and observed == len(self.targets)
        timed_out = state.t >= self.duration_s
        done = all_observed or timed_out

        return ScenarioResult(
            reward=reward,
            score=float(observed),
            done=done,
            info={
                "observed": observed,
                "spawned": len(self.targets),
                "total": len(self._spawn_schedule),
                "timed_out": timed_out,
            },
        )

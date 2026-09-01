"""Рейс крізь ворота — тренувальний режим у дусі Liftoff. Рахує кола й час.

Ворота розставлені по колу (``configs/scenarios/gate_race.yaml``); проліт крізь
них СТРОГО за порядком (0, 1, 2, ..., N-1) завершує коло. Сценарій завершується
після ``laps_to_win`` кіл. Усі події (кожні ворота, кожне коло, фініш) пишуться
в ``self.events`` — ``app.py``/``scripts`` можуть скинути їх у файл через
``scenarios/event_log.py`` (критерій приймання фази 4).
"""

from __future__ import annotations

import numpy as np

from dronesim.core.config import load_config
from dronesim.core.contracts import VehicleState
from dronesim.scenarios.base import Scenario, ScenarioResult
from dronesim.world.targets import Gate


class GateRaceScenario(Scenario):
    """Гонка крізь послідовність воріт по колу."""

    def __init__(self, config_name: str = "scenarios/gate_race"):
        self.cfg = load_config(config_name)
        self.laps_to_win = int(self.cfg.laps_to_win)
        self.gates: list[Gate] = []
        self.events: list[dict] = []

        self._next_gate_index = 0
        self._laps_completed = 0
        self._prev_pos: np.ndarray | None = None
        self._lap_start_t = 0.0
        self._lap_times: list[float] = []

    @property
    def next_gate_index(self) -> int:
        """Індекс воріт, крізь які треба пролетіти НАСТУПНИМИ (для автопілота, фаза 7)."""
        return self._next_gate_index

    def reset(self, seed: int | None = None) -> None:
        self.gates = [
            Gate(
                gate_id=i,
                center=np.array(g.center, dtype=np.float64),
                normal=np.asarray(g.normal, dtype=np.float64)
                / np.linalg.norm(np.asarray(g.normal, dtype=np.float64)),
                radius=float(g.radius),
            )
            for i, g in enumerate(self.cfg.gates)
        ]
        self._next_gate_index = 0
        self._laps_completed = 0
        self._prev_pos = None
        self._lap_start_t = 0.0
        self._lap_times = []
        self.events = []

    def step(self, state: VehicleState) -> ScenarioResult:
        pos = state.pos
        reward = 0.0
        done = False

        if self._prev_pos is not None:
            gate = self.gates[self._next_gate_index]
            if gate.passes_through(self._prev_pos, pos):
                reward = 1.0
                self.events.append({"t": state.t, "event": "gate", "gate_id": gate.gate_id})
                self._next_gate_index += 1

                if self._next_gate_index >= len(self.gates):
                    self._next_gate_index = 0
                    self._laps_completed += 1
                    lap_time = state.t - self._lap_start_t
                    self._lap_times.append(lap_time)
                    self._lap_start_t = state.t
                    self.events.append(
                        {
                            "t": state.t,
                            "event": "lap",
                            "lap": self._laps_completed,
                            "lap_time": lap_time,
                        }
                    )
                    if self._laps_completed >= self.laps_to_win:
                        done = True
                        self.events.append({"t": state.t, "event": "finish"})

        self._prev_pos = pos.copy()
        return ScenarioResult(
            reward=reward,
            score=float(self._laps_completed),
            done=done,
            info={
                "laps": self._laps_completed,
                "next_gate": self._next_gate_index,
                "lap_times": list(self._lap_times),
            },
        )

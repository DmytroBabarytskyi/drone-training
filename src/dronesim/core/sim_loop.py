"""Цикл симуляції з фіксованим кроком і годинник.

Це серце проєкту (див. ARCHITECTURE.md). Фізика крутиться з ``physics_hz``,
керування/рендер — рідше (``control_hz``). Годинник детермінований: час = кроки * dt.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable


@dataclass
class SimClock:
    """Детермінований годинник симуляції."""

    physics_hz: float = 240.0
    step_count: int = 0

    @property
    def dt(self) -> float:
        return 1.0 / self.physics_hz

    @property
    def t(self) -> float:
        return self.step_count * self.dt

    def tick(self) -> None:
        self.step_count += 1


class SimLoop:
    """Мінімальний цикл: викликає ``on_step(clock)`` кожен крок фізики.

    Реальні компоненти (фізика, контролер, рендер) підключаються як колбеки
    або в підкласі. На фазі 0 достатньо порожнього циклу для перевірки годинника.
    """

    def __init__(self, physics_hz: float = 240.0):
        self.clock = SimClock(physics_hz=physics_hz)
        self._running = False

    def run(self, on_step: Callable[[SimClock], None], steps: int | None = None) -> None:
        self._running = True
        try:
            while self._running:
                on_step(self.clock)
                self.clock.tick()
                if steps is not None and self.clock.step_count >= steps:
                    break
        finally:
            self._running = False

    def stop(self) -> None:
        self._running = False

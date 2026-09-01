"""Інтерфейс джерела керування. Реалізації: клавіатура, геймпад, RL-агент.

КЛЮЧОВИЙ КОНТРАКТ (ARCHITECTURE.md): усі джерела повертають ``ControlCommand``.
Фізика й контролери не знають, людина це чи ML. Не додавай тут залежностей від
Panda3D чи pygame — вони належать конкретним реалізаціям.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from dronesim.core.contracts import ControlCommand


class ControlSource(ABC):
    """Абстрактне джерело команд керування."""

    @abstractmethod
    def get_command(self) -> ControlCommand:
        """Повернути поточну команду керування (викликається щокроку керування)."""

    def close(self) -> None:  # noqa: B027 - опційний хук
        """Звільнити ресурси (джойстик, вікно). За замовчуванням нічого."""

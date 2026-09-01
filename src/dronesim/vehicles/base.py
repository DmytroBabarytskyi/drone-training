"""Інтерфейс апарата. Реалізації: FPV-квад, великий мультиротор, fixed-wing.

КОНТРАКТ: апарат приймає вихід польотного контролера (тяги моторів / сили) і
віддає ``VehicleState``. Він НЕ читає введення напряму — між введенням і апаратом
стоїть ``FlightController`` (vehicles/controllers/).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

import numpy as np

from dronesim.core.contracts import VehicleState

if TYPE_CHECKING:
    from panda3d.core import NodePath


class Vehicle(ABC):
    """Абстрактний літальний апарат у фізичному світі."""

    @abstractmethod
    def reset(self, pos: np.ndarray | None = None) -> None:
        """Поставити апарат у стартову позу, обнулити швидкості."""

    @abstractmethod
    def apply_motor_commands(self, motor_thrusts: np.ndarray, dt: float | None = None) -> None:
        """Прикласти тяги моторів (Н) до фізичного тіла на поточному кроці.

        ``dt`` — опційний крок фізики (доп. фаза "реалізм фізики"): вмикає
        інерцію моторів/просідання батареї, якщо апарат їх підтримує;
        ``None`` (дефолт) — миттєвий відгук без батареї (стара поведінка,
        зворотно сумісно з викликами без ``dt``)."""

    @property
    @abstractmethod
    def state(self) -> VehicleState:
        """Поточний стан апарата зі світу фізики."""

    @property
    @abstractmethod
    def node_path(self) -> "NodePath":
        """``NodePath`` фізичного тіла — ЛИШЕ для прив'язки візуалу/камери (render/).

        Дані для ML/сценаріїв/фізики йдуть через ``state`` (VehicleState), не
        через це — це не частина контракту даних, а зручність для рендеру, щоб
        візуальні вузли й камера успадковували трансформацію через граф сцени.
        """

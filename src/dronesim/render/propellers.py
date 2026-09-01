"""Візуальні гвинти апарата (доп. фаза поліш) — суто рендер, БЕЗ впливу на
фізику: ``VehicleState.motor_rpm`` лишається завжди 0 (обертання моторів не
моделюється, ``vehicles/multirotor.py``), тож обертання лопатей тут керується
НАПРЯМУ нормованою тягою з ``physics_step`` (``app.py``), а не читанням
``motor_rpm`` — інакше лопаті взагалі ніколи б не оберталися (правило "рендер
не рахує фізику", SKILL.md, діє й у зворотний бік: рендер не читає дані,
яких физика не рахує).
"""

from __future__ import annotations

import numpy as np
from panda3d.core import CardMaker, NodePath, Vec4

_BLADE_COLOR = (0.08, 0.08, 0.09, 1.0)


def add_propeller_visual(
    parent: NodePath,
    position: tuple[float, float, float],
    blade_length: float = 0.15,
) -> NodePath:
    """Проста плоска лопать (дві перехресні картки) в позиції одного мотора."""
    prop_np = parent.attachNewNode("propeller_visual")
    prop_np.setPos(*position)

    for heading in (0, 90):
        cm = CardMaker("blade")
        cm.setFrame(-blade_length, blade_length, -blade_length * 0.12, blade_length * 0.12)
        blade = prop_np.attachNewNode(cm.generate())
        blade.setP(-90)  # горизонтально (у площині обертання гвинта)
        blade.setH(heading)

    prop_np.setColor(Vec4(*_BLADE_COLOR))
    prop_np.setTwoSided(True)
    return prop_np


class PropellerRig:
    """4 гвинти апарата, що обертаються ВІЗУАЛЬНО пропорційно нормованій тязі
    останньої команди мотора — не читає ``motor_rpm`` (завжди 0, физика його
    не рахує), а приймає тяги напряму від викликача (``app.py::physics_step``,
    одразу після ``apply_motor_commands``)."""

    def __init__(
        self,
        parent: NodePath,
        motor_positions: np.ndarray,
        spin_dirs: np.ndarray,
        blade_length: float = 0.15,
        max_spin_deg_per_s: float = 2400.0,
    ):
        self._props = [add_propeller_visual(parent, tuple(pos), blade_length) for pos in motor_positions]
        self._spin_dirs = np.asarray(spin_dirs, dtype=np.float64)
        self._max_spin_deg_per_s = max_spin_deg_per_s
        self._headings = np.zeros(len(self._props))

    def update(self, motor_thrusts: np.ndarray, max_thrust_per_motor: float, dt: float) -> None:
        normalized = np.clip(np.asarray(motor_thrusts, dtype=np.float64) / max_thrust_per_motor, 0.0, 1.0)
        self._headings += self._spin_dirs * self._max_spin_deg_per_s * normalized * dt
        for prop, heading in zip(self._props, self._headings, strict=True):
            prop.setH(float(heading))

    def close(self) -> None:
        for prop in self._props:
            prop.removeNode()

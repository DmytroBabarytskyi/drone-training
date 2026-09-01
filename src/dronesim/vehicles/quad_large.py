"""Великий мультиротор — тонкий підклас ``Multirotor``.

Важчий, повільніший апарат для "розумних" польотних режимів (Angle/Alt-hold/
Pos-hold, фаза 3), на відміну від акробатичного FPV (Acro, фаза 2). Фізична
модель ідентична ``QuadFPV`` (див. vehicles/multirotor.py) — відрізняється
лише параметрами з ``configs/vehicles/quad_large.yaml`` (маса, інерція,
повільніші rate, додаткові контури PID для утримання кута/висоти/позиції).
"""

from __future__ import annotations

from panda3d.core import NodePath

from dronesim.physics.world import PhysicsWorld
from dronesim.vehicles.multirotor import Multirotor


class QuadLarge(Multirotor):
    """Великий мультиротор — важкий, стабільний, для Angle/Alt-hold/Pos-hold."""

    def __init__(
        self,
        physics_world: PhysicsWorld,
        config_name: str = "vehicles/quad_large",
        parent: NodePath | None = None,
    ):
        super().__init__(physics_world, config_name, parent=parent)

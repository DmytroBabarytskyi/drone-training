"""5-дюймовий FPV-фрістайл квадрокоптер — тонкий підклас ``Multirotor``.

Уся фізична модель (тверде тіло, мотори, мікшер тяги) — спільна для всіх
X-квадів і живе в ``vehicles/multirotor.py``. Цей клас лише фіксує дефолтний
конфіг (``configs/vehicles/fpv_5inch.yaml``) — легкий/швидкий фрістайл-апарат
з акробатичними rate (Acro-режим, фаза 2).
"""

from __future__ import annotations

from panda3d.core import NodePath

from dronesim.physics.world import PhysicsWorld
from dronesim.vehicles.multirotor import Multirotor


class QuadFPV(Multirotor):
    """FPV-квадрокоптер 5" — легкий, акробатичний (Acro-режим)."""

    def __init__(
        self,
        physics_world: PhysicsWorld,
        config_name: str = "vehicles/fpv_5inch",
        parent: NodePath | None = None,
    ):
        super().__init__(physics_world, config_name, parent=parent)

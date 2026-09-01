"""Ефекти знищення (доп. фаза "реальні моделі"): спалах + дим (візуальні
білборд-картки) + уламки (СПРАВЖНЯ Bullet-фізика) — тригериться РІВНО РАЗ на
перехід ``target.hit`` False→True (``app.py::_sync_target_markers``), той
самий патерн "рендер-only компонент з update(dt)", що й
``render/propellers.py::PropellerRig``/``render/audio.py::AudioController``.

Уламки — реальні динамічні тіла через уже готовий ``PhysicsWorld`` (той
самий світ, що й апарат), НЕ анімація-підробка: випадкова початкова
швидкість, далі Bullet сам рахує їхній політ/падіння.
"""

from __future__ import annotations

import numpy as np
from panda3d.core import CardMaker, NodePath, TransparencyAttrib, Vec4

from dronesim.physics.world import PhysicsWorld
from dronesim.render.visuals import add_box_visual

_FLASH_COLOR = Vec4(1.0, 0.9, 0.5, 1.0)
_SMOKE_COLOR = Vec4(0.28, 0.28, 0.28, 0.6)
_DEBRIS_COLOR = (0.15, 0.13, 0.1, 1.0)


class ExplosionEffect:
    """Один вибух у точці ``pos``: спалах+дим (згасають самі через
    ``update(dt)``) + уламки (реальні Bullet-тіла, видаляються з
    ``PhysicsWorld`` після ``debris_lifetime_s``, доступно через
    ``self.finished``, щоб викликач (``app.py``) прибрав завершені ефекти
    зі свого списку)."""

    def __init__(self, parent: NodePath, world: PhysicsWorld, pos: np.ndarray, cfg, seed: int | None = None):
        self._world = world
        self._t = 0.0
        self._flash_duration = float(cfg.flash_duration_s)
        self._smoke_duration = float(cfg.smoke_duration_s)
        self._debris_lifetime = float(cfg.debris_lifetime_s)
        self.finished = False
        rng = np.random.default_rng(seed)

        self._root = parent.attachNewNode("explosion")
        self._root.setPos(*pos)

        flash_size = float(cfg.flash_size_m)
        cm = CardMaker("flash")
        cm.setFrame(-flash_size, flash_size, -flash_size, flash_size)
        self._flash = self._root.attachNewNode(cm.generate())
        self._flash.setColor(_FLASH_COLOR)
        self._flash.setTransparency(TransparencyAttrib.MAlpha)
        self._flash.setBillboardPointEye()

        smoke_size = float(cfg.smoke_size_m)
        self._smoke_puffs: list[NodePath] = []
        for _ in range(int(cfg.smoke_puff_count)):
            cm_smoke = CardMaker("smoke")
            cm_smoke.setFrame(-smoke_size, smoke_size, -smoke_size, smoke_size)
            puff = self._root.attachNewNode(cm_smoke.generate())
            puff.setColor(_SMOKE_COLOR)
            puff.setTransparency(TransparencyAttrib.MAlpha)
            puff.setBillboardPointEye()
            puff.setPos(*rng.uniform(-0.4, 0.4, size=3))
            self._smoke_puffs.append(puff)

        debris_half = float(cfg.debris_half_extent_m)
        debris_mass = float(cfg.debris_mass_kg)
        speed_min, speed_max = float(cfg.debris_speed_min_mps), float(cfg.debris_speed_max_mps)
        self._debris: list[NodePath] = []
        for _ in range(int(cfg.debris_count)):
            frag_np = world.add_box_obstacle(
                half_extents=(debris_half, debris_half, debris_half),
                pos=tuple(float(p) for p in pos),
                mass=debris_mass,
                parent=parent,
            )
            add_box_visual(frag_np, (debris_half, debris_half, debris_half), color=_DEBRIS_COLOR)

            direction = rng.normal(size=3)
            direction[2] = abs(direction[2]) + 0.4  # переважно вгору, не крізь землю
            direction = direction / np.linalg.norm(direction)
            speed = rng.uniform(speed_min, speed_max)
            frag_np.node().setLinearVelocity(tuple(float(v) for v in direction * speed))
            frag_np.node().setAngularVelocity(tuple(float(v) for v in rng.uniform(-8.0, 8.0, size=3)))
            self._debris.append(frag_np)

    def update(self, dt: float) -> None:
        if self.finished:
            return
        self._t += dt

        flash_alpha = max(0.0, 1.0 - self._t / self._flash_duration)
        self._flash.setAlphaScale(flash_alpha)
        self._flash.setScale(1.0 + self._t * 2.0)

        smoke_alpha = max(0.0, 1.0 - self._t / self._smoke_duration)
        for puff in self._smoke_puffs:
            puff.setAlphaScale(smoke_alpha)
            puff.setZ(puff.getZ() + dt * 0.5)  # дим повільно піднімається

        if self._t >= self._debris_lifetime:
            self._cleanup()

    def _cleanup(self) -> None:
        for frag_np in self._debris:
            self._world.remove(frag_np)
        self._debris.clear()
        self._root.removeNode()
        self.finished = True

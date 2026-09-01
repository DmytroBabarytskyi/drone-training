"""Обгортка над ``panda3d.bullet`` — фізичний світ.

Працює повністю headless: ``BulletWorld`` не потребує вікна/``ShowBase`` і жодного
графічного контексту, тож придатний і для інтерактивного польоту, і для тестів,
і для збору ML-даних без рендеру (SKILL.md, розділ «Типові пастки»).

Позиції/оренти тіл читаються й пишуться через ``NodePath`` (стандартний спосіб
Panda3D для ``BulletRigidBodyNode`` — синхронізація з Bullet відбувається
автоматично при зміні трансформи ``NodePath`` і при кожному ``doPhysics``).

``attach(..., parent=None)`` за замовчуванням лишає тіло НЕприв'язаним до жодного
графа сцени (як у фазі 1 — чисто фізичний headless-тест без рендеру). Якщо
передати ``parent`` (типово ``engine.render``, фаза 2), тіло стає частиною
видимої сцени, і рендер може чіплювати до нього видиму геометрію/камеру
дочірніми вузлами (render/visuals.py, render/camera.py) — без жодного ручного
синхронізування трансформів.
"""

from __future__ import annotations

from panda3d.bullet import BulletBoxShape, BulletPlaneShape, BulletRigidBodyNode, BulletWorld
from panda3d.core import NodePath, Point3, Vec3

DEFAULT_GRAVITY = -9.81  # м/с²


class PhysicsWorld:
    """Тонка обгортка ``BulletWorld``: гравітація, детермінований крок, приєднання тіл."""

    def __init__(self, gravity: float = DEFAULT_GRAVITY):
        self.world = BulletWorld()
        self.world.setGravity(Vec3(0, 0, gravity))

    def step(self, dt: float) -> None:
        """Один фізичний крок довжиною ``dt`` без внутрішньої інтерполяції підкроків.

        ``dt`` очікується рівним кроку ``SimClock`` (фіксований крок, ARCHITECTURE.md).
        """
        self.world.doPhysics(dt, 1, dt)

    def attach(self, node: BulletRigidBodyNode, parent: NodePath | None = None) -> NodePath:
        """Прикріпити тіло до світу; повертає ``NodePath`` для читання/запису пози.

        ``parent=None`` (типово) — тіло поза графом сцени, лише фізика (фаза 1).
        """
        self.world.attachRigidBody(node)
        node_path = NodePath(node)
        if parent is not None:
            node_path.reparentTo(parent)
        return node_path

    def remove(self, node_path: NodePath) -> None:
        """Прибрати тіло зі світу Bullet І зі сцени (доп. фаза "реальні моделі" —
        уламки ефектів знищення мають скінченне життя, на відміну від
        постійних перешкод/апарата)."""
        self.world.removeRigidBody(node_path.node())
        node_path.removeNode()

    def add_ground_plane(self, z: float = 0.0, parent: NodePath | None = None) -> NodePath:
        """Статична нескінченна земельна площина на висоті ``z``."""
        shape = BulletPlaneShape(Vec3(0, 0, 1), z)
        body = BulletRigidBodyNode("ground")
        body.addShape(shape)
        body.setMass(0.0)  # маса 0 у Bullet = статичне тіло
        return self.attach(body, parent=parent)

    def add_box_obstacle(
        self,
        half_extents: tuple[float, float, float],
        pos: tuple[float, float, float],
        mass: float = 0.0,
        parent: NodePath | None = None,
    ) -> NodePath:
        """Куб-перешкода (статичний за замовчуванням; ``mass > 0`` — динамічний)."""
        shape = BulletBoxShape(Vec3(*half_extents))
        body = BulletRigidBodyNode("obstacle")
        body.addShape(shape)
        body.setMass(mass)
        node_path = self.attach(body, parent=parent)
        node_path.setPos(Point3(*pos))
        return node_path

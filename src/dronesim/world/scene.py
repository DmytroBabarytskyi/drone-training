"""Сцена: земля + статичні перешкоди — з ``configs/world/`` (фаза 1) або
``assets/scenes/`` (фаза 4, іменовані сцени під конкретні сценарії).

Схема YAML однакова для обох джерел: ``ground_z`` + список ``obstacles`` з
``pos``/``half_extents`` + опційний ``kind`` (доп. фаза "реальні моделі":
``tree``/``fence``/``trench``/``road``/... — обирає, ЯКИЙ візуальний
будівник з ``render/models.py`` викликач (``app.py``) прикріпить поверх
фізичного тіла; за замовчуванням ``box``, зворотно сумісно з наявними
сценами, що не мають цього поля). Різниця лише в тому, ЗВІДКИ читається
файл — тюнинг (``configs/``) чи ігровий контент рівня (``assets/``), як
визначено в ``core/config.py``.

ФІЗИКА ЛИШАЄТЬСЯ ПРОСТОЮ КОРОБКОЮ ЗАВЖДИ (``add_box_obstacle``), незалежно
від ``kind`` — міняється лише візуальна геометрія поверх неї (``app.py``).
Зміна форми колізії під конкретну візуальну модель — поза обсягом (ризик
для фізики апарата заради суто декоративної точності).
"""

from __future__ import annotations

from panda3d.core import NodePath

from dronesim.core.config import load_asset, load_config
from dronesim.physics.world import PhysicsWorld

_ObstacleList = list[tuple[NodePath, tuple[float, float, float], str]]


def _build_scene_from_cfg(
    cfg, world: PhysicsWorld, parent: NodePath | None
) -> tuple[NodePath, _ObstacleList]:
    ground_np = world.add_ground_plane(z=float(cfg.ground_z), parent=parent)

    obstacles: _ObstacleList = []
    for obstacle in cfg.obstacles:
        half_extents = tuple(obstacle.half_extents)
        obstacle_np = world.add_box_obstacle(
            half_extents=half_extents,
            pos=tuple(obstacle.pos),
            parent=parent,
        )
        kind = str(obstacle.get("kind", "box"))
        obstacles.append((obstacle_np, half_extents, kind))

    return ground_np, obstacles


def build_basic_scene(
    world: PhysicsWorld, config_name: str = "world/basic_scene", parent: NodePath | None = None
) -> tuple[NodePath, _ObstacleList]:
    """Додати землю та статичні перешкоди за конфігом з ``configs/`` (фаза 1, дефолтна сцена).

    ``parent`` — куди прикріпити тіла в графі сцени для рендеру (``engine.render``,
    фаза 2); ``None`` лишає сцену поза графом (headless фаза 1).

    Повертає ``(ground_node_path, [(obstacle_node_path, half_extents, kind), ...])``,
    щоб викликач (напр. app.py) міг додати видиму геометрію (render/visuals.py,
    render/models.py — за ``kind``) без повторного читання YAML.
    """
    return _build_scene_from_cfg(load_config(config_name), world, parent)


def build_scene_from_assets(
    world: PhysicsWorld, scene_name: str, parent: NodePath | None = None
) -> tuple[NodePath, _ObstacleList]:
    """Додати землю та статичні перешкоди за іменованою сценою з ``assets/scenes/``.

    Використовується для сценаріїв (фаза 4): ``assets/scenes/gate_race.yaml``,
    ``strike_range.yaml``, ``patrol.yaml`` — декоративні перешкоди навколо
    воріт/цілей конкретного сценарію (самі ворота/цілі — окремо, world/targets.py).
    """
    return _build_scene_from_cfg(load_asset(f"scenes/{scene_name}"), world, parent)


def load_scene_decor(scene_name: str) -> dict:
    """Прочитати ВІЗУАЛЬНІ (без фізики) декорації сцени — опційні top-level
    секції YAML поза ``obstacles`` (доп. фаза "графіка 3.0"):

    - ``wheat_patches``: список ``{center: [x, y], size: [sx, sy], count}``
      — пшеничні ділянки (``render/models.py::scatter_wheat_patch``).
      Пшениця НЕ obstacle навмисно: стебла м'які, апарат має пролітати
      КРІЗЬ них, а Bullet-коробка (єдина фізика obstacle) блокувала б політ.

    Повертає dict з відсутніми секціями як порожніми списками — викликач
    (``app.py``) не мусить перевіряти наявність ключів. Сцени без цих
    секцій (усі наявні) отримують порожні списки — зворотна сумісність."""
    cfg = load_asset(f"scenes/{scene_name}")
    return {
        "wheat_patches": [
            {"center": tuple(p.center), "size": tuple(p.size), "count": int(p.get("count", 800))}
            for p in cfg.get("wheat_patches", [])
        ],
    }

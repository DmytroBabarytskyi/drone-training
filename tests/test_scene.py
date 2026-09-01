"""Тести сцени (world/scene.py): дефолтна (configs/) і іменовані (assets/, фаза 4)."""

from dronesim.physics.world import PhysicsWorld
from dronesim.world.scene import build_basic_scene, build_scene_from_assets


def test_build_basic_scene_adds_ground_and_obstacles():
    world = PhysicsWorld()
    build_basic_scene(world)

    result = world.world.getRigidBodies()
    names = [body.getName() for body in result]

    assert "ground" in names
    assert names.count("obstacle") == 3  # три перешкоди з configs/world/basic_scene.yaml


def test_build_scene_from_assets_gate_race():
    world = PhysicsWorld()
    build_scene_from_assets(world, "gate_race")

    names = [body.getName() for body in world.world.getRigidBodies()]
    assert "ground" in names
    # 3 первісні коробки + декоративна сценерія (дерева/паркан/дорога, доп.
    # фаза "реальні моделі") — усі так само реальні Bullet-тіла (фізика
    # завжди проста коробка, лише візуал відрізняється за kind).
    assert names.count("obstacle") == 13  # assets/scenes/gate_race.yaml


def test_build_scene_from_assets_patrol_has_decorative_scenery_only():
    """patrol НЕ має ігрових перешкод (цілі з'являються за розкладом, не
    статичні коробки) — але МАЄ декоративну сценерію (доп. фаза "реальні
    моделі": дерева/дорога, усі ``kind`` != "box")."""
    world = PhysicsWorld()
    ground_np, obstacles = build_scene_from_assets(world, "patrol")

    assert ground_np is not None
    assert len(obstacles) > 0
    assert all(kind != "box" for _np, _half_extents, kind in obstacles)


def test_obstacle_kind_defaults_to_box_when_field_omitted():
    """Доп. фаза "реальні моделі": obstacle-записи БЕЗ ``kind`` у YAML (як усі
    записи в assets/scenes/patrol.yaml) не мають зламатись — регресійний
    тест зворотної сумісності зі старими сценами."""
    world = PhysicsWorld()
    _ground_np, obstacles = build_scene_from_assets(world, "gate_race")
    # Перші 3 записи gate_race.yaml — без kind (стара розкладка) -> "box".
    assert [kind for _np, _half_extents, kind in obstacles[:3]] == ["box", "box", "box"]


def test_obstacle_kind_read_from_yaml(tmp_path, monkeypatch):
    from dronesim.core import config as config_module

    monkeypatch.setattr(config_module, "CONFIGS_DIR", tmp_path)
    (tmp_path / "world").mkdir()
    (tmp_path / "world" / "test_kinds.yaml").write_text(
        "ground_z: 0.0\n"
        "obstacles:\n"
        "  - { pos: [1.0, 0.0, 1.0], half_extents: [1.0, 1.0, 1.0] }\n"
        "  - { pos: [2.0, 0.0, 1.0], half_extents: [1.0, 1.0, 1.0], kind: tree }\n"
        "  - { pos: [3.0, 0.0, 1.0], half_extents: [1.0, 1.0, 1.0], kind: fence }\n",
        encoding="utf-8",
    )
    world = PhysicsWorld()
    _ground_np, obstacles = build_basic_scene(world, config_name="world/test_kinds")
    kinds = [kind for _np, _half_extents, kind in obstacles]
    assert kinds == ["box", "tree", "fence"]

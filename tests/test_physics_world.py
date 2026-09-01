"""Тести обгортки фізичного світу (physics/world.py) — headless Bullet, без вікна."""

from dronesim.physics.world import PhysicsWorld


def test_box_falls_and_rests_on_ground_plane():
    world = PhysicsWorld()
    world.add_ground_plane(z=0.0)
    box = world.add_box_obstacle(half_extents=(0.5, 0.5, 0.5), pos=(0.0, 0.0, 5.0), mass=1.0)

    for _ in range(int(3.0 * 240)):  # 3с при 240Hz — достатньо, щоб впасти й осісти
        world.step(1 / 240)

    z = box.getZ()
    assert 0.4 < z < 0.6  # осів на землі, центр коробки ~ half-extent над z=0


def test_static_obstacle_does_not_move():
    world = PhysicsWorld()
    world.add_ground_plane(z=0.0)
    obstacle = world.add_box_obstacle(half_extents=(0.5, 0.5, 0.5), pos=(2.0, 1.0, 0.5), mass=0.0)

    for _ in range(240):
        world.step(1 / 240)

    assert abs(obstacle.getX() - 2.0) < 1e-6
    assert abs(obstacle.getY() - 1.0) < 1e-6
    assert abs(obstacle.getZ() - 0.5) < 1e-6


def test_step_runs_without_bodies():
    world = PhysicsWorld()
    for _ in range(10):
        world.step(1 / 240)  # не мусить кидати винятків навіть у порожньому світі


def test_remove_detaches_body_from_world_and_scene():
    """Доп. фаза "реальні моделі": уламки ефектів знищення мають скінченне
    життя — після ``remove`` тіло більше не бере участі у фізиці (не падає/
    не існує) і зникає зі сцени."""
    world = PhysicsWorld()
    world.add_ground_plane(z=0.0)
    box = world.add_box_obstacle(half_extents=(0.2, 0.2, 0.2), pos=(0.0, 0.0, 5.0), mass=1.0)

    world.remove(box)

    assert box.isEmpty()  # NodePath видалено зі сцени
    # прибраний зі світу Bullet -> подальші кроки фізики не кидають винятків
    for _ in range(10):
        world.step(1 / 240)

"""Тести ефектів знищення (render/effects.py, доп. фаза "реальні моделі") —
headless: PhysicsWorld + engine fixture (рендер потрібен для карток
спалаху/диму, фізика — для уламків)."""

import numpy as np

from dronesim.core.config import load_config
from dronesim.physics.world import PhysicsWorld
from dronesim.render.effects import ExplosionEffect

_CFG = load_config("effects")


def test_explosion_creates_configured_number_of_debris_fragments(engine):
    world = PhysicsWorld()
    world.add_ground_plane(z=0.0)
    parent = engine.render.attachNewNode("test_parent_explosion")
    try:
        effect = ExplosionEffect(parent, world, np.array([0.0, 0.0, 3.0]), _CFG, seed=1)
        assert len(effect._debris) == int(_CFG.debris_count)
    finally:
        parent.removeNode()


def test_debris_are_real_bullet_bodies_affected_by_gravity(engine):
    world = PhysicsWorld()
    world.add_ground_plane(z=0.0)
    parent = engine.render.attachNewNode("test_parent_gravity")
    try:
        effect = ExplosionEffect(parent, world, np.array([0.0, 0.0, 3.0]), _CFG, seed=1)
        initial_positions = [tuple(frag.getPos()) for frag in effect._debris]

        for _ in range(60):  # 0.25с при 240Hz
            world.step(1 / 240)
            effect.update(1 / 240)

        for frag, initial_pos in zip(effect._debris, initial_positions, strict=True):
            pos = frag.getPos()
            assert not np.isnan([pos.x, pos.y, pos.z]).any()
            moved = np.linalg.norm(np.array(tuple(pos)) - np.array(initial_pos))
            assert moved > 0.01  # реально зрушили з місця (початкова швидкість + гравітація)
    finally:
        parent.removeNode()


def test_flash_and_smoke_fade_out_over_time(engine):
    world = PhysicsWorld()
    parent = engine.render.attachNewNode("test_parent_fade")
    try:
        effect = ExplosionEffect(parent, world, np.array([0.0, 0.0, 3.0]), _CFG, seed=1)
        assert effect._flash.getColorScale()[3] == 1.0  # повна яскравість на старті

        effect.update(_CFG.flash_duration_s * 1.5)
        assert effect._flash.getColorScale()[3] == 0.0  # згас повністю
    finally:
        if not effect.finished:
            parent.removeNode()


def test_explosion_cleans_up_debris_after_lifetime(engine):
    world = PhysicsWorld()
    world.add_ground_plane(z=0.0)
    parent = engine.render.attachNewNode("test_parent_cleanup")
    try:
        effect = ExplosionEffect(parent, world, np.array([0.0, 0.0, 3.0]), _CFG, seed=1)
        assert effect.finished is False

        dt = 1 / 30
        steps = int(_CFG.debris_lifetime_s / dt) + 2
        for _ in range(steps):
            world.step(dt)
            effect.update(dt)

        assert effect.finished is True
        assert effect._debris == []
    finally:
        parent.removeNode()


def test_explosion_debris_positioned_at_explosion_point_initially(engine):
    world = PhysicsWorld()
    parent = engine.render.attachNewNode("test_parent_pos")
    try:
        pos = np.array([2.0, -1.0, 4.0])
        effect = ExplosionEffect(parent, world, pos, _CFG, seed=1)
        for frag in effect._debris:
            frag_pos = frag.getPos()
            assert abs(frag_pos.x - pos[0]) < 1e-6
            assert abs(frag_pos.y - pos[1]) < 1e-6
            assert abs(frag_pos.z - pos[2]) < 1e-6
    finally:
        parent.removeNode()

"""Тести видимої геометрії-плейсхолдера (render/visuals.py)."""

import numpy as np
from panda3d.core import Vec3

from dronesim.render import visuals
from dronesim.render.visuals import (
    add_box_visual,
    add_gate_marker_visual,
    add_ground_visual,
    add_marker_visual,
    add_target_marker_visual,
    set_target_marker_hit,
)


def test_box_faces_all_point_outward(engine):
    """Регрес-тест на реальний баг (доп. фаза "графіка 3.0", реальний фідбек
    "не відрізниш, куди падає сонце"): Y-грані ``_BOX_FACES`` мали ОДНАКОВУ
    пару headings, що давало нормаль ВСЕРЕДИНУ коробки — освітлення
    рахувалось неправильно на половині граней КОЖНОЇ коробки в грі.
    Перевіряє ВСІ 6 граней empірично через ``getRelativeVector`` (не
    теоретично — SKILL.md пастка "не виводь HPR формулами")."""
    for hpr, axis, sign in visuals._BOX_FACES:  # noqa: SLF001
        node = engine.render.attachNewNode("face_test")
        try:
            node.setHpr(*hpr)
            world_normal = engine.render.getRelativeVector(node, Vec3(0, -1, 0))
            expected = [0.0, 0.0, 0.0]
            expected[axis] = float(sign)
            for i in range(3):
                assert abs(world_normal[i] - expected[i]) < 1e-3, (hpr, axis, sign, world_normal)
        finally:
            node.removeNode()


def test_add_box_visual_bounds_match_half_extents(engine):
    parent = engine.render.attachNewNode("test_parent_box")
    try:
        half_extents = (0.5, 0.3, 0.2)
        box = add_box_visual(parent, half_extents)
        mn, mx = box.getTightBounds()
        assert abs(mx.x - half_extents[0]) < 1e-3
        assert abs(mn.x + half_extents[0]) < 1e-3
        assert abs(mx.y - half_extents[1]) < 1e-3
        assert abs(mx.z - half_extents[2]) < 1e-3
    finally:
        parent.removeNode()


def test_add_ground_visual_is_flat_horizontal(engine):
    parent = engine.render.attachNewNode("test_parent_ground")
    try:
        ground = add_ground_visual(parent, size=10.0)
        mn, mx = ground.getTightBounds()
        assert abs(mx.z - mn.z) < 1e-3  # плоска по Z (горизонтальна)
        assert abs(mx.x - 10.0) < 1e-3
        assert abs(mx.y - 10.0) < 1e-3
    finally:
        parent.removeNode()


def test_add_marker_visual_is_positioned_correctly(engine):
    parent = engine.render.attachNewNode("test_parent_marker")
    try:
        marker = add_marker_visual(parent, (3.0, -2.0, 5.0), size=0.5)
        pos = marker.getPos(engine.render)
        assert abs(pos.x - 3.0) < 1e-6
        assert abs(pos.y - (-2.0)) < 1e-6
        assert abs(pos.z - 5.0) < 1e-6
    finally:
        parent.removeNode()


def test_add_ground_visual_has_grass_texture(engine):
    parent = engine.render.attachNewNode("test_parent_ground_tex")
    try:
        ground = add_ground_visual(parent, size=10.0)
        assert ground.hasTexture()
    finally:
        parent.removeNode()


def test_add_target_marker_visual_is_positioned_and_textured(engine):
    parent = engine.render.attachNewNode("test_parent_target")
    try:
        marker = add_target_marker_visual(parent, (1.0, 2.0, 3.0), size=0.8)
        pos = marker.getPos(engine.render)
        assert abs(pos.x - 1.0) < 1e-6
        assert abs(pos.y - 2.0) < 1e-6
        assert abs(pos.z - 3.0) < 1e-6
        # Дві перехресні картки -> 2 дочірні вузли з текстурою.
        cards = marker.getChildren()
        assert len(cards) == 2
        assert all(card.hasTexture() for card in cards)
    finally:
        parent.removeNode()


def test_set_target_marker_hit_changes_color_scale(engine):
    parent = engine.render.attachNewNode("test_parent_hit")
    try:
        marker = add_target_marker_visual(parent, (0.0, 0.0, 0.0))
        set_target_marker_hit(marker, True)
        scale_hit = marker.getColorScale()
        assert scale_hit[1] > scale_hit[0]  # зелений домінує над червоним

        set_target_marker_hit(marker, False)
        scale_ok = marker.getColorScale()
        assert abs(scale_ok[0] - 1.0) < 1e-6
        assert abs(scale_ok[1] - 1.0) < 1e-6
    finally:
        parent.removeNode()


def test_add_gate_marker_visual_has_four_bars_around_hole(engine):
    parent = engine.render.attachNewNode("test_parent_gate")
    try:
        gate = add_gate_marker_visual(
            parent, (5.0, 0.0, 3.0), radius=2.5, normal=(1.0, 0.0, 0.0)
        )
        assert abs(gate.getPos(engine.render).x - 5.0) < 1e-6
        assert len(gate.getChildren()) == 4  # 4 бруси рамки

        mn, mx = gate.getTightBounds()
        span = mx - mn
        # Рамка з отвором радіусом 2.5 має розмах ~5м у двох "поперечних" осях.
        assert max(span.x, span.y, span.z) >= 4.5
    finally:
        parent.removeNode()


def test_add_gate_marker_visual_orients_toward_arbitrary_normal(engine):
    parent = engine.render.attachNewNode("test_parent_gate_normal")
    try:
        gate = add_gate_marker_visual(
            parent, (0.0, 0.0, 3.0), radius=1.0, normal=(0.0, 1.0, 0.0)
        )
        assert not np.any(np.isnan([gate.getHpr().x, gate.getHpr().y, gate.getHpr().z]))
    finally:
        parent.removeNode()

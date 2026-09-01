"""Тести FPV-камери (render/camera.py) — offscreen-захоплення кадру з вікна."""

import numpy as np

from dronesim.physics.world import PhysicsWorld
from dronesim.render.camera import (
    FPVCamera,
    attach_chase_camera,
    attach_main_camera,
    update_chase_camera,
)
from dronesim.render.visuals import add_box_visual
from dronesim.vehicles.quad_fpv import QuadFPV


def test_get_frame_returns_expected_shape_and_dtype(engine):
    world = PhysicsWorld()
    vehicle = QuadFPV(world, parent=engine.render)
    vehicle.reset(pos=np.array([0.0, 0.0, 2.0]))

    # яскравий куб просто перед носом апарата (+X, forward — див. render/camera.py)
    target = engine.render.attachNewNode("target")
    target.setPos(5.0, 0.0, 2.0)
    add_box_visual(target, (2.0, 2.0, 2.0), color=(1.0, 0.1, 0.1, 1.0))

    camera = FPVCamera(engine, vehicle.node_path, width=64, height=48)
    try:
        # taskMgr.step() (не graphicsEngine.renderFrame() напряму!) — щоб встиг
        # відпрацювати внутрішній task simplepbr, який виставляє шейдерний вхід
        # camera_world_position перед фактичним рендером кадру.
        engine.taskMgr.step()
        engine.taskMgr.step()

        frame = camera.get_frame(vehicle.state, t=0.0)
        assert frame.rgb.shape == (48, 64, 3)
        assert frame.rgb.dtype == np.uint8
        assert frame.t == 0.0
    finally:
        camera.close()
        target.removeNode()


def test_camera_follows_vehicle_reparent(engine):
    world = PhysicsWorld()
    vehicle = QuadFPV(world, parent=engine.render)
    vehicle.reset(pos=np.array([1.0, 2.0, 3.0]))

    camera = FPVCamera(engine, vehicle.node_path, width=32, height=24)
    try:
        cam_world_pos = camera._np.getPos(engine.render)
        # камера прикріплена до апарата зі зсувом ~0.05 по X -> має бути поруч з апаратом
        vehicle_pos = vehicle.state.pos
        assert np.linalg.norm(
            np.array([cam_world_pos.x, cam_world_pos.y, cam_world_pos.z]) - vehicle_pos
        ) < 0.5
    finally:
        camera.close()


def test_fpv_camera_near_plane_is_small_not_default_1m(engine):
    """Реальний фідбек після польоту: "камера бачить через землю" — Panda3D
    ДЕФОЛТНИЙ near-plane (1.0м) обрізав землю/об'єкти ближче за 1м, а апарат
    може стояти на землі/літати низько (mount_offset лише 0.02-0.2м)."""
    world = PhysicsWorld()
    vehicle = QuadFPV(world, parent=engine.render)
    vehicle.reset(pos=np.array([0.0, 0.0, 1.0]))

    camera = FPVCamera(engine, vehicle.node_path, width=32, height=24)
    try:
        near = camera._np.node().getLens().getNear()  # noqa: SLF001
        assert near < 0.1
    finally:
        camera.close()


def test_attach_main_camera_sets_small_near_plane(engine):
    world = PhysicsWorld()
    vehicle = QuadFPV(world, parent=engine.render)
    vehicle.reset(pos=np.array([0.0, 0.0, 1.0]))
    try:
        attach_main_camera(engine, vehicle.node_path)
        assert engine.camLens.getNear() < 0.1
    finally:
        engine.camera.reparentTo(engine.render)
        engine.camera.setPos(0, 0, 0)
        engine.camera.setHpr(0, 0, 0)
        engine.camLens.setNear(1.0)  # повернути дефолт для наступних тестів фікстури


def test_attach_main_camera_follows_vehicle(engine):
    """Фаза 5: головна (видима) камера вікна має слідувати за апаратом, не
    лишатись статичною в початку координат (docs/DECISIONS.md, фаза 5)."""
    world = PhysicsWorld()
    vehicle = QuadFPV(world, parent=engine.render)
    vehicle.reset(pos=np.array([4.0, -3.0, 2.0]))

    try:
        attach_main_camera(engine, vehicle.node_path)
        cam_world_pos = engine.camera.getPos(engine.render)
        assert np.linalg.norm(
            np.array([cam_world_pos.x, cam_world_pos.y, cam_world_pos.z]) - vehicle.state.pos
        ) < 0.5
    finally:
        engine.camera.reparentTo(engine.render)  # прибрати за собою (спільний фікстур)
        engine.camera.setPos(0, 0, 0)
        engine.camera.setHpr(0, 0, 0)


def test_attach_chase_camera_detaches_from_vehicle_parent(engine):
    world = PhysicsWorld()
    vehicle = QuadFPV(world, parent=engine.render)
    vehicle.reset(pos=np.array([1.0, 2.0, 3.0]))
    try:
        attach_main_camera(engine, vehicle.node_path)
        assert engine.camera.getParent() == vehicle.node_path

        attach_chase_camera(engine)
        assert engine.camera.getParent() == engine.render
    finally:
        engine.camera.reparentTo(engine.render)
        engine.camera.setPos(0, 0, 0)
        engine.camera.setHpr(0, 0, 0)


def test_update_chase_camera_positions_behind_and_looks_at_vehicle(engine):
    try:
        attach_chase_camera(engine)
        vehicle_pos = np.array([10.0, 5.0, 3.0])
        update_chase_camera(engine, vehicle_pos, yaw_rad=0.0)

        cam_pos = engine.camera.getPos(engine.render)
        # yaw=0, offset за замовчуванням (0,-6,2.5) -> камера позаду вздовж -Y
        assert abs(cam_pos.x - 10.0) < 1e-6
        assert abs(cam_pos.y - (5.0 - 6.0)) < 1e-6
        assert abs(cam_pos.z - (3.0 + 2.5)) < 1e-6
    finally:
        engine.camera.reparentTo(engine.render)
        engine.camera.setPos(0, 0, 0)
        engine.camera.setHpr(0, 0, 0)


def test_update_chase_camera_rotates_offset_with_yaw(engine):
    try:
        attach_chase_camera(engine)
        vehicle_pos = np.array([0.0, 0.0, 0.0])
        # yaw=90deg -> offset (0,-6,2.5) повертається на 90 градусів навколо Z
        update_chase_camera(engine, vehicle_pos, yaw_rad=np.pi / 2)

        cam_pos = engine.camera.getPos(engine.render)
        assert abs(cam_pos.x - 6.0) < 1e-6
        assert abs(cam_pos.y - 0.0) < 1e-6
        assert abs(cam_pos.z - 2.5) < 1e-6
    finally:
        engine.camera.reparentTo(engine.render)
        engine.camera.setPos(0, 0, 0)
        engine.camera.setHpr(0, 0, 0)

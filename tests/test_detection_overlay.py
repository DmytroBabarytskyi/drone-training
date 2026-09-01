"""Тести PIP-оверлею детекцій (render/detection_overlay.py, фаза 5)."""

import numpy as np

from dronesim.core.contracts import Detection
from dronesim.physics.world import PhysicsWorld
from dronesim.render.camera import FPVCamera
from dronesim.render.detection_overlay import DetectionOverlay
from dronesim.vehicles.quad_fpv import QuadFPV


def test_overlay_creates_and_clears_boxes(engine):
    world = PhysicsWorld()
    vehicle = QuadFPV(world, parent=engine.render)
    vehicle.reset(pos=np.array([0.0, 0.0, 2.0]))
    camera = FPVCamera(engine, vehicle.node_path, width=64, height=48)

    try:
        overlay = DetectionOverlay(engine, camera.texture, pip_size=0.4)
        try:
            assert overlay._box_node is None

            dets = [Detection(bbox_xywh=(32.0, 24.0, 10.0, 10.0), class_id=0, conf=0.9)]
            overlay.update(dets, frame_width=64, frame_height=48)
            assert overlay._box_node is not None

            overlay.update([], frame_width=64, frame_height=48)
            assert overlay._box_node is None
        finally:
            overlay.close()
    finally:
        camera.close()


def test_overlay_replaces_previous_boxes_not_accumulates(engine):
    world = PhysicsWorld()
    vehicle = QuadFPV(world, parent=engine.render)
    vehicle.reset(pos=np.array([0.0, 0.0, 2.0]))
    camera = FPVCamera(engine, vehicle.node_path, width=64, height=48)

    try:
        overlay = DetectionOverlay(engine, camera.texture)
        try:
            det = Detection(bbox_xywh=(10.0, 10.0, 5.0, 5.0), class_id=0, conf=0.5)
            overlay.update([det], frame_width=64, frame_height=48)
            first_node = overlay._box_node
            overlay.update([det], frame_width=64, frame_height=48)
            second_node = overlay._box_node
            assert first_node is not second_node  # старий вузол видалено, не додано поверх
        finally:
            overlay.close()
    finally:
        camera.close()

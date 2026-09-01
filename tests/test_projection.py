"""Тести проєкції 3D->2D для авто-розмітки (ml/data/projection.py).

``PerspectiveLens`` — легкий об'єкт panda3d.core, не потребує ShowBase/вікна."""

from panda3d.core import PerspectiveLens, Point3

from dronesim.ml.data.projection import project_box_to_bbox, project_point


def _make_lens(fov_deg: float = 90.0) -> PerspectiveLens:
    lens = PerspectiveLens()
    lens.setFov(fov_deg)
    return lens


def test_point_straight_ahead_projects_near_center():
    lens = _make_lens()
    uv = project_point(lens, Point3(0.0, 10.0, 0.0))
    assert uv is not None
    u, v = uv
    assert abs(u - 0.5) < 1e-6
    assert abs(v - 0.5) < 1e-6


def test_point_behind_camera_is_none():
    lens = _make_lens()
    uv = project_point(lens, Point3(0.0, -5.0, 0.0))
    assert uv is None


def test_point_off_to_the_right_has_larger_u():
    lens = _make_lens()
    uv_center = project_point(lens, Point3(0.0, 10.0, 0.0))
    uv_right = project_point(lens, Point3(3.0, 10.0, 0.0))
    assert uv_right is not None
    assert uv_right[0] > uv_center[0]


def test_point_above_has_smaller_v_top_left_origin():
    lens = _make_lens()
    uv_center = project_point(lens, Point3(0.0, 10.0, 0.0))
    uv_up = project_point(lens, Point3(0.0, 10.0, 3.0))
    assert uv_up is not None
    assert uv_up[1] < uv_center[1]  # вище в світі -> менше v (верхній лівий початок)


def test_point_far_outside_fov_is_none():
    lens = _make_lens(fov_deg=30.0)
    uv = project_point(lens, Point3(100.0, 1.0, 0.0))  # майже вбік від дуже вузького обʼєктива
    assert uv is None


def test_box_bbox_center_matches_point_projection():
    lens = _make_lens()
    bbox = project_box_to_bbox(lens, Point3(0.0, 10.0, 0.0), half_extent=0.5)
    assert bbox is not None
    cx, cy, w, h = bbox
    assert abs(cx - 0.5) < 0.05
    assert abs(cy - 0.5) < 0.05
    assert w > 0.0
    assert h > 0.0


def test_box_bbox_larger_when_closer():
    lens = _make_lens()
    bbox_far = project_box_to_bbox(lens, Point3(0.0, 20.0, 0.0), half_extent=1.0)
    bbox_near = project_box_to_bbox(lens, Point3(0.0, 5.0, 0.0), half_extent=1.0)
    assert bbox_far is not None and bbox_near is not None
    assert bbox_near[2] > bbox_far[2]  # ближча ціль -> ширший bbox
    assert bbox_near[3] > bbox_far[3]


def test_box_bbox_none_when_center_behind_camera():
    lens = _make_lens()
    bbox = project_box_to_bbox(lens, Point3(0.0, -10.0, 0.0), half_extent=1.0)
    assert bbox is None


def test_box_bbox_clamped_within_frame():
    lens = _make_lens()
    # Дуже великий маркер, майже впритул -> кути виходять за межі кадру, bbox має клемпитись у [0,1].
    bbox = project_box_to_bbox(lens, Point3(0.0, 1.0, 0.0), half_extent=5.0)
    assert bbox is not None
    cx, cy, w, h = bbox
    assert 0.0 <= cx - w / 2
    assert cx + w / 2 <= 1.0
    assert 0.0 <= cy - h / 2
    assert cy + h / 2 <= 1.0

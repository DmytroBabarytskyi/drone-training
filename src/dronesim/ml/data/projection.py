"""Проєкція 3D-точок цілей у 2D bbox кадру камери — авто-розмітка датасету (фаза 5).

Використовує РЕАЛЬНИЙ ``Lens.project()`` Panda3D (а не власну реімплементацію
FOV-математики), щоб розмітка гарантовано збігалася з тим, що камера дійсно
рендерить — той самий підхід, що й для FOV камери в render/camera.py.
``PerspectiveLens`` — легкий об'єкт ``panda3d.core``, не потребує ShowBase/вікна,
тож ці функції тестуються напряму, без фікстури ``engine``.

Результат — нормований bbox у форматі YOLO: ``(x_center, y_center, width, height)``,
усі значення в [0, 1] відносно розміру кадру.
"""

from __future__ import annotations

from panda3d.core import Lens, Point2, Point3


def project_point(lens: Lens, point_in_camera_space: Point3) -> tuple[float, float] | None:
    """Спроєктувати точку (вже в ЛОКАЛЬНИХ координатах камери) у нормовані [0,1]
    пікселі (початок координат — верхній лівий кут, як у зображеннях).

    ``None``, якщо точка позаду камери (forward камери — локальна +Y, як у
    render/camera.py) або поза полем зору об'єктива.
    """
    if point_in_camera_space[1] <= 0.0:
        return None
    ndc = Point2()
    in_view = lens.project(point_in_camera_space, ndc)
    if not in_view:
        return None
    u = (ndc[0] + 1.0) / 2.0
    v = 1.0 - (ndc[1] + 1.0) / 2.0
    return float(u), float(v)


def project_box_to_bbox(
    lens: Lens, center_camera_space: Point3, half_extent: float
) -> tuple[float, float, float, float] | None:
    """Спроєктувати кубічний маркер (центр + half_extent, локальні координати
    камери) у нормований YOLO-bbox ``(x_center, y_center, width, height)``.

    ``None``, якщо центр цілі не потрапляє в кадр (ціль вважається "не видно"
    навіть якщо якийсь із кутів кроп-бокса випадково опинився б у полі зору —
    авто-розмітка не повинна створювати рамки для цілей, на які апарат
    фактично не дивиться).
    """
    center_uv = project_point(lens, center_camera_space)
    if center_uv is None:
        return None

    cx, cy, cz = center_camera_space
    corners_uv = []
    for dx in (-half_extent, half_extent):
        for dy in (-half_extent, half_extent):
            for dz in (-half_extent, half_extent):
                uv = project_point(lens, Point3(cx + dx, cy + dy, cz + dz))
                if uv is not None:
                    corners_uv.append(uv)

    xs = [c[0] for c in corners_uv] + [center_uv[0]]
    ys = [c[1] for c in corners_uv] + [center_uv[1]]
    x_min, x_max = max(0.0, min(xs)), min(1.0, max(xs))
    y_min, y_max = max(0.0, min(ys)), min(1.0, max(ys))

    width = x_max - x_min
    height = y_max - y_min
    if width <= 0.0 or height <= 0.0:
        return None
    return (x_min + width / 2.0, y_min + height / 2.0, width, height)

"""FPV-камера: офскрін-буфер Panda3D, прив'язаний до носа апарата.

Контракт: ``get_frame()`` повертає ``CameraFrame`` (core/contracts.py) з RGB-
кадром (H, W, 3) uint8. Реалізовано вже тут, у фазі 2 (не відкладено до фази 5) —
Panda3D дає захоплення кадру "з коробки" через texture buffer; ML (фаза 5)
перевикористає цей самий клас без змін, лише споживаючи ``CameraFrame``.

Камера кріпиться як дочірній вузол до ``NodePath`` апарата (``Vehicle.node_path``),
тож рухається разом з ним автоматично через граф сцени — без ручної синхронізації.

УВАГА (виявлено емпірично у фазі 4, docs/DECISIONS.md): "вперед" для тіла
апарата — це ЛОКАЛЬНА вісь +X (так визначено в ``vehicles/multirotor.py``,
де ``x`` у розкладці моторів відповідає pitch/forward; підтверджено тестом
``test_manual_pitch_stick_moves_in_positive_x_direction``). Panda3D-камера за
замовчуванням (identity HPR) дивиться вздовж СВОЄЇ локальної +Y — тому камеру
довертаємо на ``H=-90``, щоб вона дивилась саме вздовж +X носа апарата.
"""

from __future__ import annotations

import numpy as np
from direct.showbase.ShowBase import ShowBase
from panda3d.core import NodePath, Point3

from dronesim.core.contracts import CameraFrame, VehicleState

# Дефолтне монтування камери на носі апарата — спільне для FPVCamera (офскрін-
# буфер для ML) і attach_main_camera (видима камера вікна), щоб обидві
# показували ОДИН І ТОЙ САМИЙ вид (інакше запис даних і те, що бачить пілот,
# розійшлися б).
DEFAULT_MOUNT_OFFSET = (0.05, 0.0, 0.02)
DEFAULT_MOUNT_HPR = (-90, 0, 0)  # локальна +Y камери -> +X носа апарата (forward)

# БАГ, ВИЯВЛЕНИЙ РЕАЛЬНИМ ПОЛЬОТОМ: "камера бачить через землю" — Panda3D
# ДЕФОЛТНИЙ near-plane лінзи (1.0м) набагато більший за реальну відстань
# апарата до землі (апарат стоїть на землі/літає низько, mount_offset лише
# 0.02-0.2м від корпусу) — усе БЛИЖЧЕ за 1м (включно із самою землею під
# апаратом) обрізається near-площиною й мовчки НЕ рендериться, тож крізь
# "діру" видно небо/фон позаду. Той самий клас багу, що й DirectionalLight
# без setPos (фаза поліш, docs/DECISIONS.md) — API не кидає винятку, просто
# геометрія зникає. 0.03м — менше за найближчу дистанцію камера->корпус
# апарата (0.02м offset), тож власне тіло апарата теж не зникає з кадру.
DEFAULT_NEAR_M = 0.03


class FPVCamera:
    """Камера від першої особи, прикріплена до носа апарата.

    ``mount_hpr`` за замовчуванням компенсує конвенцію "вперед=+X" апарата
    (``DEFAULT_MOUNT_HPR``). Передай ``(0,0,0)``, якщо ``mount_to`` — вузол,
    зорієнтований через ``NodePath.lookAt()`` (Panda3D: forward=+Y напряму,
    без потреби в компенсації) — так робить ``ml/data/recorder.py``, де немає
    "апарата" з власною фізичною конвенцією осей, лише кінематична камера.
    """

    def __init__(
        self,
        base: ShowBase,
        mount_to: NodePath,
        width: int = 320,
        height: int = 240,
        fov_deg: float = 100.0,
        mount_offset: tuple[float, float, float] = DEFAULT_MOUNT_OFFSET,
        mount_hpr: tuple[float, float, float] = DEFAULT_MOUNT_HPR,
        near_m: float = DEFAULT_NEAR_M,
    ):
        self._base = base
        self._width = width
        self._height = height

        self._buffer = base.win.makeTextureBuffer("fpv_camera", width, height, None, True)
        self._texture = self._buffer.getTexture()

        self._np = base.makeCamera(self._buffer)
        self._np.reparentTo(mount_to)
        self._np.setPos(Point3(*mount_offset))
        self._np.setHpr(*mount_hpr)
        lens = self._np.node().getLens()
        lens.setFov(fov_deg)
        lens.setNear(near_m)

    @property
    def texture(self):
        """Panda3D-текстура офскрін-буфера (для ``render/detection_overlay.py``, фаза 5)."""
        return self._texture

    def get_frame(self, pose: VehicleState, t: float) -> CameraFrame:
        """Прочитати поточний уже відрендерений кадр офскрін-буфера як ``CameraFrame``."""
        return CameraFrame(rgb=self._read_rgb(), pose=pose, t=t)

    def _read_rgb(self) -> np.ndarray:
        if not self._texture.hasRamImage():
            return np.zeros((self._height, self._width, 3), dtype=np.uint8)
        data = self._texture.getRamImage()
        arr = np.frombuffer(data, dtype=np.uint8).reshape((self._height, self._width, 4))
        rgb = arr[:, :, [2, 1, 0]]  # BGRA -> RGB
        rgb = rgb[::-1]  # текстура зберігається знизу-вгору
        return np.ascontiguousarray(rgb)

    def close(self) -> None:
        self._base.graphicsEngine.removeWindow(self._buffer)


def attach_main_camera(
    base: ShowBase,
    mount_to: NodePath,
    fov_deg: float = 100.0,
    mount_offset: tuple[float, float, float] = DEFAULT_MOUNT_OFFSET,
    near_m: float = DEFAULT_NEAR_M,
) -> None:
    """Прикріпити ГОЛОВНУ (видиму у вікні ``base.win``) камеру ShowBase до носа апарата.

    БАГ, ЯКИЙ ЦЕ ВИПРАВЛЯЄ (виявлено у фазі 5, docs/DECISIONS.md): ``FPVCamera``
    рендерить лише в offscreen-буфер для ML — головна камера вікна (``base.camera``)
    ніколи не рухалась, тож інтерактивний ``fly`` показував статичний вид від
    початку координат, а не FPV. Після цього виклику видиме вікно показує ТОЙ
    САМИЙ вид, що й ``FPVCamera`` (однакове монтування/FOV).
    """
    base.camera.reparentTo(mount_to)
    base.camera.setPos(Point3(*mount_offset))
    base.camera.setHpr(*DEFAULT_MOUNT_HPR)
    base.camLens.setFov(fov_deg)
    base.camLens.setNear(near_m)  # див. DEFAULT_NEAR_M: "камера бачить через землю"


# Chase-cam (доп. фаза поліш): offset ЗАДАНО У СВІТОВИХ координатах (не
# дочірній вузол апарата) — інакше камера перевертається разом із дроном при
# крені/тангажі, що дезорієнтує для допоміжного виду третьої особи. Слідує
# лише за YAW апарата (не roll/pitch), щоб лишатись "позаду" під час поворотів.
DEFAULT_CHASE_OFFSET = (0.0, -6.0, 2.5)
DEFAULT_CHASE_FOV_DEG = 70.0


def attach_chase_camera(
    base: ShowBase, fov_deg: float = DEFAULT_CHASE_FOV_DEG
) -> None:
    """Відв'язати головну камеру від апарата (chase-режим) — позицію/орієнтацію
    після цього оновлює ``update_chase_camera`` щокадру."""
    base.camera.reparentTo(base.render)
    base.camLens.setFov(fov_deg)


def update_chase_camera(
    base: ShowBase,
    vehicle_pos: np.ndarray,
    yaw_rad: float,
    offset: tuple[float, float, float] = DEFAULT_CHASE_OFFSET,
) -> None:
    """Оновити позицію chase-камери: offset позаду апарата, повернутий лише за
    рисканням (yaw) — камера НЕ нахиляється разом із креном/тангажем, інакше
    вигляд третьої особи дезорієнтує саме там, де він мав би допомагати
    (реальний фідбек користувача: важко зрозуміти орієнтацію апарата)."""
    cos_y, sin_y = np.cos(yaw_rad), np.sin(yaw_rad)
    ox, oy, oz = offset
    world_offset = np.array([ox * cos_y - oy * sin_y, ox * sin_y + oy * cos_y, oz])
    cam_pos = np.asarray(vehicle_pos, dtype=float) + world_offset
    base.camera.setPos(*cam_pos)
    base.camera.lookAt(Point3(*np.asarray(vehicle_pos, dtype=float)))

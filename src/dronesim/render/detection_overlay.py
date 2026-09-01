"""Оверлей детекцій (фаза 5): picture-in-picture з кадром FPV-камери + bbox.

Показує В КУТКУ ЕКРАНА точно той кадр, який бачить ``Detector`` (пікселі 1:1),
і малює bbox прямо на ньому. Свідомий вибір замість накладання bbox напряму на
основний 3D-вид: головна камера (після фіксу ``attach_main_camera``, фаза 5) і
офскрін-буфер ``FPVCamera`` мають однаковий mount/FOV, але можуть мати РІЗНИЙ
пропорції кадру (aspect ratio) від вікна — пряме накладання вимагало б це
узгоджувати. PIP на власній текстурі — завжди точне вирівнювання без цієї
залежності (docs/DECISIONS.md, фаза 5).
"""

from __future__ import annotations

from direct.gui.OnscreenImage import OnscreenImage
from panda3d.core import LineSegs, NodePath, Texture, Vec4

from dronesim.core.contracts import Detection


class DetectionOverlay:
    """PIP (picture-in-picture) з FPV-кадром і рамками детекцій, у кутку екрана."""

    def __init__(self, base, texture: Texture, pip_size: float = 0.5):
        self._base = base
        aspect_ratio = base.getAspectRatio()
        self._pip_size = pip_size

        self._root = base.aspect2d.attachNewNode("detection_pip")
        self._root.setPos(aspect_ratio - pip_size / 2 - 0.02, 0, 1.0 - pip_size / 2 - 0.02)

        self._image = OnscreenImage(image=texture, parent=self._root, scale=pip_size / 2)
        self._box_node: NodePath | None = None

    def update(self, detections: list[Detection], frame_width: int, frame_height: int) -> None:
        """Перемалювати рамки детекцій поверх PIP. Викликати щокроку, де є новий кадр."""
        if self._box_node is not None:
            self._box_node.removeNode()
            self._box_node = None
        if not detections:
            return

        half = self._pip_size / 2
        lines = LineSegs()
        lines.setThickness(2.0)
        lines.setColor(Vec4(0.1, 1.0, 0.2, 1.0))

        for det in detections:
            cx, cy, w, h = det.bbox_xywh
            u0, v0 = (cx - w / 2) / frame_width, (cy - h / 2) / frame_height
            u1, v1 = (cx + w / 2) / frame_width, (cy + h / 2) / frame_height
            x0, x1 = (u0 * 2 - 1) * half, (u1 * 2 - 1) * half
            z0, z1 = (1 - v0 * 2) * half, (1 - v1 * 2) * half  # v росте вниз -> z вгору
            lines.moveTo(x0, 0, z0)
            lines.drawTo(x1, 0, z0)
            lines.drawTo(x1, 0, z1)
            lines.drawTo(x0, 0, z1)
            lines.drawTo(x0, 0, z0)

        self._box_node = self._root.attachNewNode(lines.create())

    def close(self) -> None:
        self._image.destroy()
        self._root.removeNode()

"""Детектор цілей: інференс навченої YOLO-моделі (фаза 5).

Контракт: ``Detector.predict(frame) -> list[Detection]`` (core/contracts.py) —
приймає RGB (H,W,3) uint8, як ``CameraFrame.rgb``. Споживачі: HUD-оверлей
(render/hud.py), RL-спостереження (ml/rl/env.py, фаза 6).
"""

from __future__ import annotations

import numpy as np

from dronesim.core.contracts import Detection


class Detector:
    """Обгортка над ``ultralytics.YOLO`` для інференсу в межах симулятора."""

    def __init__(self, weights_path: str, conf_threshold: float = 0.25):
        from ultralytics import YOLO

        self.model = YOLO(weights_path)
        self.conf_threshold = conf_threshold

    def predict(self, frame: np.ndarray) -> list[Detection]:
        """Виявити цілі на кадрі (RGB, H×W×3, uint8) -> список ``Detection``."""
        results = self.model.predict(frame, conf=self.conf_threshold, verbose=False)
        detections = []
        for result in results:
            for box in result.boxes:
                x_center, y_center, w, h = box.xywh[0].tolist()
                detections.append(
                    Detection(
                        bbox_xywh=(x_center, y_center, w, h),
                        class_id=int(box.cls[0].item()),
                        conf=float(box.conf[0].item()),
                    )
                )
        return detections

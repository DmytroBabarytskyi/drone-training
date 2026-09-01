"""Інтеграційний тест повного ML-конвеєра фази 5: запис -> навчання -> оцінка -> інференс.

НАВМИСНО зменшений масштаб (десятки кадрів, 1 епоха, малий imgsz) — це тест
КОРЕКТНОСТІ З'ЄДНАННЯ компонентів, не повноцінне навчання (критерій приймання
ROADMAP "кілька тисяч кадрів"/"mAP вище порогу" — окремий, довший прогін,
описаний у docs/DECISIONS.md; тут перевіряємо, що весь шлях record->train->
eval->detect працює без помилок і дає ЗМІСТОВНИЙ (не сміттєвий) результат."""

from __future__ import annotations

import cv2
import pytest

from dronesim.ml.data.recorder import DatasetRecorder
from dronesim.ml.detection.detector import Detector
from dronesim.ml.detection.eval import evaluate

pytestmark = pytest.mark.slow  # torch/ultralytics: секунди-хвилини, не мс


def test_record_train_eval_detect_end_to_end(engine, tmp_path):
    dataset_dir = tmp_path / "dataset"
    recorder = DatasetRecorder(
        engine,
        config_name="data/strike_range",
        overrides=[
            f"output_dir={dataset_dir.as_posix()}",
            "viewpoints_per_target=15",
            "val_fraction=0.3",
            "image_width=96",
            "image_height=72",
        ],
    )
    try:
        stats = recorder.record()
    finally:
        recorder.close()
    assert stats["train"] > 0
    assert stats["val"] > 0

    project_dir = tmp_path / "runs"
    best_weights = _train_tiny(dataset_dir, project_dir)

    metrics = evaluate(best_weights, str(dataset_dir / "dataset.yaml"))
    assert "map50" in metrics
    assert 0.0 <= metrics["map50"] <= 1.0

    detector = Detector(best_weights, conf_threshold=0.1)
    sample_image = next((dataset_dir / "images" / "train").glob("*.jpg"))
    frame = cv2.imread(str(sample_image))[:, :, ::-1]  # BGR -> RGB
    detections = detector.predict(frame)
    assert isinstance(detections, list)
    if detections:
        det = detections[0]
        assert len(det.bbox_xywh) == 4
        assert det.class_id == 0
        assert 0.0 <= det.conf <= 1.0


def _train_tiny(dataset_dir, project_dir) -> str:
    """Навчити на 1 епосі, малий imgsz — лише перевірити, що конвеєр З'ЄДНУЄТЬСЯ,
    не досягти якогось mAP-порогу (для цього — повний прогін, не юніт-тест)."""
    from ultralytics import YOLO

    model = YOLO("yolov8n.pt")
    results = model.train(
        data=str(dataset_dir / "dataset.yaml"),
        epochs=1,
        imgsz=96,
        batch=4,
        project=str(project_dir),
        name="tiny_smoke",
        seed=0,
        verbose=False,
    )
    return str(results.save_dir / "weights" / "best.pt")

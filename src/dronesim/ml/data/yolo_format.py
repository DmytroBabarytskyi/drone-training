"""Формат датасету Ultralytics YOLO: images/, labels/, dataset.yaml (фаза 5).

Чисті функції запису — не залежать від Panda3D, тестуються напряму на
тимчасових директоріях (``tmp_path``). ``OmegaConf`` — той самий інструмент
YAML, що й ``core/config.py``, для узгодженості в проєкті.
"""

from __future__ import annotations

from pathlib import Path

from omegaconf import OmegaConf


def write_label(path: str | Path, class_id: int, bbox_xywh_norm: tuple[float, float, float, float]) -> None:
    """Записати один YOLO-label файл: ``"class_id x_center y_center width height"``
    (усі значення нормовані в [0, 1] відносно розміру зображення)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    cx, cy, w, h = bbox_xywh_norm
    path.write_text(f"{class_id} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}\n", encoding="utf-8")


def write_empty_label(path: str | Path) -> None:
    """Порожній label-файл (кадр без цілі в кадрі) — YOLO очікує файл навіть без об'єктів."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("", encoding="utf-8")


def write_dataset_yaml(path: str | Path, dataset_root: str | Path, class_names: list[str]) -> None:
    """Записати ``dataset.yaml`` для ``ultralytics.YOLO(...).train(data=...)``."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    content = {
        "path": str(Path(dataset_root).resolve()),
        "train": "images/train",
        "val": "images/val",
        "names": {i: name for i, name in enumerate(class_names)},
    }
    OmegaConf.save(config=OmegaConf.create(content), f=str(path))

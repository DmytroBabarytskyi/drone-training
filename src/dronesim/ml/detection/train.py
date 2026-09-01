"""Навчання Ultralytics YOLO на зібраному датасеті (фаза 5).

Тонка обгортка над ``ultralytics.YOLO`` — усі параметри з ``configs/ml/*.yaml``
(SKILL.md правило 3: жодних магічних чисел у коді). ``base_model`` (типово
``yolov8n.pt``) — попередньо навчена модель для transfer learning; перший
запуск завантажує її з інтернету (кешується ultralytics локально надалі).
"""

from __future__ import annotations

import argparse

from dronesim.core.config import REPO_ROOT, load_config


def train(config_name: str = "ml/yolo") -> str:
    """Навчити YOLO за конфігом; повертає шлях до найкращих ваг (``best.pt``)."""
    from ultralytics import YOLO

    cfg = load_config(config_name)
    model = YOLO(cfg.base_model)
    results = model.train(
        data=str(REPO_ROOT / str(cfg.dataset_yaml)),
        epochs=int(cfg.epochs),
        imgsz=int(cfg.imgsz),
        batch=int(cfg.batch),
        project=str(REPO_ROOT / str(cfg.project_dir)),
        name=str(cfg.run_name),
        seed=int(cfg.seed),
        verbose=False,
    )
    return str(results.save_dir / "weights" / "best.pt")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Навчити YOLO на датасеті цілей")
    parser.add_argument("--config", default="ml/yolo")
    args = parser.parse_args(argv)

    best_weights = train(args.config)
    print(f"Найкраща модель: {best_weights}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

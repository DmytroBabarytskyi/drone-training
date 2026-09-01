"""Оцінка навченої YOLO: mAP@0.5 на валідаційній вибірці (критерій приймання фази 5)."""

from __future__ import annotations

import argparse

from dronesim.core.config import REPO_ROOT, load_config


def evaluate(weights_path: str, dataset_yaml: str) -> dict:
    """Повернути ``{"map50": ..., "map50_95": ...}`` для навченої моделі."""
    from ultralytics import YOLO

    model = YOLO(weights_path)
    # ЯВНО задаємо project/name — інакше ultralytics падає назад на ГЛОБАЛЬНІ
    # налаштування (`~/.config/Ultralytics/settings.json`), які на цій машині
    # вказують на ЧУЖУ директорію ІНШОГО проєкту (виявлено емпірично, фаза 5,
    # docs/DECISIONS.md) — без цього evaluate() писав би файли ПОЗА нашим репо.
    metrics = model.val(
        data=dataset_yaml,
        project=str(REPO_ROOT / "runs" / "detect"),
        name="val",
        verbose=False,
    )
    return {
        "map50": float(metrics.box.map50),
        "map50_95": float(metrics.box.map),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Оцінити навчену YOLO (mAP)")
    parser.add_argument("--weights", required=True)
    parser.add_argument("--config", default="ml/yolo", help="Для дефолтного dataset_yaml")
    args = parser.parse_args(argv)

    cfg = load_config(args.config)
    metrics = evaluate(args.weights, str(REPO_ROOT / str(cfg.dataset_yaml)))
    print(f"mAP@0.5: {metrics['map50']:.3f}  mAP@0.5:0.95: {metrics['map50_95']:.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

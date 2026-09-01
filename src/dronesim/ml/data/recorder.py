"""``DatasetRecorder`` — скриптовано записує авто-розмічений датасет YOLO
навколо цілей сценарію (фаза 5, критерій приймання: тисячі авто-розмічених кадрів).

Кінематичне розміщення (без фізики/Bullet): для збору даних потрібна лише
коректна поза камери відносно цілі, не реальна динаміка польоту — SKILL.md
правило 6 ("не змішуй шари": рендер+геометрія тут, ніякої фізики).

Маркери цілей записуються БЕЗ PBR-освітлення (``setLightOff()``) — стабільний,
контрастний колір з будь-якого ракурсу спрощує задачу детекції (порівняно з
художньо затіненими маркерами інтерактивного польоту, render/visuals.py).
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
from panda3d.core import Point3

from dronesim.core.config import REPO_ROOT, load_config
from dronesim.ml.data.projection import project_box_to_bbox
from dronesim.ml.data.scripted_flight import generate_viewpoints
from dronesim.ml.data.yolo_format import write_dataset_yaml, write_label
from dronesim.render.camera import FPVCamera
from dronesim.render.visuals import add_marker_visual
from dronesim.scenarios.registry import build_scenario


class DatasetRecorder:
    """Записує датасет YOLO навколо цілей сценарію, вказаного в конфізі."""

    def __init__(
        self,
        engine,
        config_name: str = "data/strike_range",
        overrides: list[str] | None = None,
    ):
        self.engine = engine
        self.cfg = load_config(config_name, overrides=overrides)

        self.rig = engine.render.attachNewNode("recording_rig")
        self.camera = FPVCamera(
            engine,
            self.rig,
            width=int(self.cfg.image_width),
            height=int(self.cfg.image_height),
            fov_deg=float(self.cfg.fov_deg),
            mount_offset=(0.0, 0.0, 0.0),
            mount_hpr=(0.0, 0.0, 0.0),  # rig орієнтується через lookAt() (forward=+Y нативно)
        )

    def _place_rig(self, pos: np.ndarray, look_at: np.ndarray) -> None:
        self.rig.setPos(*pos)
        self.rig.lookAt(*look_at)

    def _render_frame(self) -> np.ndarray:
        self.engine.taskMgr.step()
        self.engine.taskMgr.step()  # simplepbr shader input (tests/test_camera.py, той самий трюк)
        return self.camera.get_frame(pose=None, t=0.0).rgb

    def _project_bbox(self, target_pos: np.ndarray) -> tuple[float, float, float, float] | None:
        cam_node = self.camera._np
        lens = cam_node.node().getLens()
        local_target = cam_node.getRelativePoint(self.engine.render, Point3(*target_pos))
        return project_box_to_bbox(lens, local_target, float(self.cfg.target_half_extent))

    def record(self) -> dict:
        """Записати повний датасет: для кожної цілі сценарію — N ракурсів, лише
        ті кадри, де ціль справді потрапляє в поле зору. Повертає статистику."""
        scenario = build_scenario(str(self.cfg.scenario), seed=int(self.cfg.seed))
        targets = list(getattr(scenario, "targets", ()))
        if not targets:
            raise ValueError(
                f"Сценарій '{self.cfg.scenario}' не має цілей (targets) — нічого записувати."
            )

        for target in targets:
            add_marker_visual(self.engine.render, tuple(target.base_pos), size=0.5).setLightOff()

        output_dir = REPO_ROOT / str(self.cfg.output_dir)
        rng = np.random.default_rng(int(self.cfg.seed))
        saved = {"train": 0, "val": 0, "skipped": 0}

        for target in targets:
            viewpoints = generate_viewpoints(
                target.base_pos,
                n=int(self.cfg.viewpoints_per_target),
                radius_range=tuple(self.cfg.radius_range),
                altitude_range=tuple(self.cfg.altitude_range),
                seed=int(self.cfg.seed) + target.target_id,
            )
            for i, vp in enumerate(viewpoints):
                split = "val" if rng.random() < float(self.cfg.val_fraction) else "train"
                self._place_rig(vp.pos, vp.look_at)
                rgb = self._render_frame()
                bbox = self._project_bbox(target.base_pos)
                if bbox is None:
                    saved["skipped"] += 1
                    continue

                stem = f"target{target.target_id}_{i:05d}"
                self._save(rgb, bbox, output_dir, split, stem)
                saved[split] += 1

        write_dataset_yaml(
            output_dir / "dataset.yaml", output_dir, class_names=list(self.cfg.class_names)
        )
        return saved

    def _save(
        self,
        rgb: np.ndarray,
        bbox: tuple[float, float, float, float],
        output_dir: Path,
        split: str,
        stem: str,
    ) -> None:
        image_path = output_dir / "images" / split / f"{stem}.jpg"
        image_path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(image_path), rgb[:, :, ::-1])  # RGB -> BGR для cv2

        label_path = output_dir / "labels" / split / f"{stem}.txt"
        write_label(label_path, class_id=0, bbox_xywh_norm=bbox)

    def close(self) -> None:
        self.camera.close()
        self.rig.removeNode()


def main(argv: list[str] | None = None) -> int:
    import argparse

    from dronesim.render.engine import Engine

    parser = argparse.ArgumentParser(description="Записати датасет YOLO навколо цілей сценарію")
    parser.add_argument("--config", default="data/strike_range")
    parser.add_argument(
        "--override", nargs="*", default=None, help="OmegaConf overrides, напр. seed=1 viewpoints_per_target=10"
    )
    args = parser.parse_args(argv)

    engine = Engine(offscreen=True)  # headless — збір даних не потребує вікна (SKILL.md)
    recorder = DatasetRecorder(engine, config_name=args.config, overrides=args.override)
    try:
        stats = recorder.record()
        print(f"Записано: {stats}")
    finally:
        recorder.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

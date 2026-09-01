"""Тести DatasetRecorder (ml/data/recorder.py) — невеликий обсяг для швидкості
тестів; повний прогін (тисячі кадрів) — окремий скрипт/CLI, не юніт-тест."""

from dronesim.ml.data.recorder import DatasetRecorder


def test_record_creates_images_labels_and_dataset_yaml(engine, tmp_path):
    output_dir = tmp_path / "tiny_strike_range"
    recorder = DatasetRecorder(
        engine,
        config_name="data/strike_range",
        overrides=[
            f"output_dir={output_dir.as_posix()}",
            "viewpoints_per_target=3",
            "val_fraction=0.5",
        ],
    )
    try:
        stats = recorder.record()
        total_saved = stats["train"] + stats["val"]
        assert total_saved > 0
        assert stats["train"] + stats["val"] + stats["skipped"] == 3 * 4  # 4 цілі strike_range

        images = list(output_dir.glob("images/*/*.jpg"))
        labels = list(output_dir.glob("labels/*/*.txt"))
        assert len(images) == total_saved
        assert len(labels) == total_saved
        assert (output_dir / "dataset.yaml").exists()

        # кожен label має рівно 5 чисел: class cx cy w h, усі в [0,1]
        for label_path in labels:
            parts = label_path.read_text(encoding="utf-8").strip().split()
            assert len(parts) == 5
            cx, cy, w, h = (float(x) for x in parts[1:])
            assert 0.0 <= cx <= 1.0
            assert 0.0 <= cy <= 1.0
            assert 0.0 < w <= 1.0
            assert 0.0 < h <= 1.0
    finally:
        recorder.close()

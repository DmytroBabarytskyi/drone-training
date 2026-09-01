"""Тести запису датасету у форматі YOLO (ml/data/yolo_format.py)."""

from omegaconf import OmegaConf

from dronesim.ml.data.yolo_format import write_dataset_yaml, write_empty_label, write_label


def test_write_label_format(tmp_path):
    path = tmp_path / "labels" / "train" / "img_0001.txt"
    write_label(path, class_id=0, bbox_xywh_norm=(0.5, 0.4, 0.2, 0.3))
    content = path.read_text(encoding="utf-8").strip()
    parts = content.split()
    assert parts[0] == "0"
    assert abs(float(parts[1]) - 0.5) < 1e-6
    assert abs(float(parts[2]) - 0.4) < 1e-6
    assert abs(float(parts[3]) - 0.2) < 1e-6
    assert abs(float(parts[4]) - 0.3) < 1e-6


def test_write_label_creates_parent_dirs(tmp_path):
    path = tmp_path / "a" / "b" / "label.txt"
    write_label(path, class_id=1, bbox_xywh_norm=(0.1, 0.1, 0.1, 0.1))
    assert path.exists()


def test_write_empty_label_creates_empty_file(tmp_path):
    path = tmp_path / "empty.txt"
    write_empty_label(path)
    assert path.exists()
    assert path.read_text(encoding="utf-8") == ""


def test_write_dataset_yaml_has_expected_keys(tmp_path):
    dataset_root = tmp_path / "strike_range"
    yaml_path = dataset_root / "dataset.yaml"
    write_dataset_yaml(yaml_path, dataset_root, class_names=["target"])

    cfg = OmegaConf.load(yaml_path)
    assert cfg.train == "images/train"
    assert cfg.val == "images/val"
    assert cfg.names[0] == "target"
    assert str(dataset_root.resolve()) in cfg.path

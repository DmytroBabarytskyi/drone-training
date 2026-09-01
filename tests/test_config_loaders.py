"""Тести завантажувачів конфігів (core/config.py): configs/ і assets/ (фаза 4)."""

import pytest

from dronesim.core.config import load_asset, load_config


def test_load_config_reads_existing_vehicle_config():
    cfg = load_config("vehicles/fpv_5inch")
    assert cfg.mass > 0


def test_load_config_missing_raises_file_not_found():
    with pytest.raises(FileNotFoundError):
        load_config("no_such_config")


def test_load_asset_reads_scene_yaml():
    cfg = load_asset("scenes/patrol")
    assert "ground_z" in cfg


def test_load_asset_missing_raises_file_not_found():
    with pytest.raises(FileNotFoundError):
        load_asset("scenes/no_such_scene")

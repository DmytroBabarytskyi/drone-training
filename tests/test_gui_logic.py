"""Тести чистої (без Qt) логіки GUI-лаунчера (доп. фаза "GUI-лаунчер"):
gui/scan.py, gui/process.py, gui/calibration_store.py, gui/rate_profile_store.py."""

from __future__ import annotations

from dronesim.gui import calibration_store, process, rate_profile_store, scan


def test_list_vehicles_finds_known_configs():
    vehicles = scan.list_vehicles()
    assert "fpv_5inch" in vehicles
    assert "quad_large" in vehicles


def test_list_scenarios_matches_registry():
    assert set(scan.list_scenarios()) == {"gate_race", "patrol", "strike_range"}


def test_list_scenario_levels_includes_blank_default_and_known_levels():
    levels = scan.list_scenario_levels("gate_race")
    assert levels[0] == ""  # база (level=None) — перший пункт
    assert "sprint" in levels
    assert "circuit8" in levels


def test_list_rl_checkpoints_finds_trained_model():
    checkpoints = scan.list_rl_checkpoints()
    names = [c.display_name for c in checkpoints]
    assert any("ppo_strike" in n and n.endswith("model.zip") for n in names)


def test_list_yolo_weights_only_matches_weights_dir_not_eval_artifacts():
    weights = scan.list_yolo_weights()
    for w in weights:
        assert w.path.parent.name == "weights"
        assert w.path.suffix == ".pt"


def test_build_fly_argv_minimal():
    argv = process.build_fly_argv(vehicle="fpv_5inch", input_source="keyboard")
    assert "dronesim.app" in argv
    assert "fly" in argv
    assert "--vehicle" in argv and "fpv_5inch" in argv
    assert "--scenario" not in argv  # не задано -> не додається


def test_build_fly_argv_with_scenario_and_level():
    argv = process.build_fly_argv(
        vehicle="quad_large", input_source="keyboard", scenario="gate_race", level="sprint"
    )
    assert "--scenario" in argv
    assert "gate_race" in argv
    assert "--level" in argv
    assert "sprint" in argv


def test_build_fly_argv_level_without_scenario_is_ignored():
    """Рівень без сценарію не має сенсу -> не повинен потрапити в argv."""
    argv = process.build_fly_argv(vehicle="fpv_5inch", input_source="keyboard", level="sprint")
    assert "--level" not in argv


def test_build_autonomous_argv_includes_checkpoint():
    argv = process.build_autonomous_argv(checkpoint="runs/rl/ppo_strike/model.zip")
    assert "autonomous" in argv
    assert "--checkpoint" in argv
    assert "runs/rl/ppo_strike/model.zip" in argv


def test_load_calibration_reads_keyboard_config():
    cfg = calibration_store.load_calibration("keyboard")
    assert "expo" in cfg
    assert "ramp_per_second" in cfg


def test_save_calibration_updates_only_given_key(tmp_path, monkeypatch):
    src = calibration_store.INPUT_CONFIG_PATHS["keyboard"]
    tmp_config = tmp_path / "keyboard.yaml"
    tmp_config.write_text(src.read_text(encoding="utf-8"), encoding="utf-8")
    monkeypatch.setitem(calibration_store.INPUT_CONFIG_PATHS, "keyboard", tmp_config)

    original = calibration_store.load_calibration("keyboard")
    calibration_store.save_calibration("keyboard", {"expo": 0.42})
    updated = calibration_store.load_calibration("keyboard")

    assert updated["expo"] == 0.42
    assert updated["ramp_per_second"] == original["ramp_per_second"]
    assert updated["bindings"] == original["bindings"]


def test_has_rate_profile_true_for_fpv_5inch_false_for_quad_large():
    assert rate_profile_store.has_rate_profile("fpv_5inch") is True
    assert rate_profile_store.has_rate_profile("quad_large") is False


def test_load_rate_profile_returns_all_axes_for_fpv_5inch():
    profile = rate_profile_store.load_rate_profile("fpv_5inch")
    assert set(profile) == {"roll", "pitch", "yaw"}
    assert set(profile["roll"]) == set(rate_profile_store.PARAMS)


def test_load_rate_profile_empty_for_vehicle_without_it():
    assert rate_profile_store.load_rate_profile("quad_large") == {}


def test_save_rate_profile_updates_only_given_axis(tmp_path, monkeypatch):
    src = rate_profile_store.vehicle_config_path("fpv_5inch")
    tmp_config = tmp_path / "fpv_5inch.yaml"
    tmp_config.write_text(src.read_text(encoding="utf-8"), encoding="utf-8")
    monkeypatch.setattr(
        rate_profile_store, "vehicle_config_path", lambda vehicle: tmp_config
    )

    original = rate_profile_store.load_rate_profile("fpv_5inch")
    rate_profile_store.save_rate_profile(
        "fpv_5inch", {"roll": {"max_rate_deg_s": 500.0}}
    )
    updated = rate_profile_store.load_rate_profile("fpv_5inch")

    assert updated["roll"]["max_rate_deg_s"] == 500.0
    assert updated["roll"]["expo"] == original["roll"]["expo"]
    assert updated["pitch"] == original["pitch"]
    assert updated["yaw"] == original["yaw"]

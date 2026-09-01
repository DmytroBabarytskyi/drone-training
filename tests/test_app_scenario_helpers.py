"""Тести допоміжних функцій app.py для сценаріїв (фаза 4) — без Panda3D/вікна.

``_build_scenario``/``_finish_scenario`` не торкаються рушія, тож тестуються
напряму (на відміну від повного ``build_fly_session``, який вимагає subprocess —
tests/test_app_fly.py, через конфлікт кількох ``ShowBase`` в одному процесі)."""

import pytest

from dronesim.app import _build_scenario, _finish_scenario
from dronesim.core import config as config_module
from dronesim.scenarios.base import ScenarioResult
from dronesim.scenarios.event_log import read_event_log
from dronesim.scenarios.gate_race import GateRaceScenario
from dronesim.scenarios.registry import SCENARIO_REGISTRY


def test_build_scenario_returns_reset_gate_race_instance():
    scenario = _build_scenario("gate_race", seed=42)
    assert isinstance(scenario, GateRaceScenario)
    assert len(scenario.gates) == 4  # configs/scenarios/gate_race.yaml


def test_build_scenario_unknown_name_raises_clear_error():
    with pytest.raises(ValueError, match="gate_race"):  # повідомляє доступні варіанти
        _build_scenario("no_such_scenario", seed=None)


def test_scenario_registry_covers_all_three_roadmap_scenarios():
    assert set(SCENARIO_REGISTRY) == {"gate_race", "patrol", "strike_range"}


def test_finish_scenario_prints_results_and_writes_log(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(config_module, "REPO_ROOT", tmp_path)

    scenario = _build_scenario("strike_range", seed=1)
    scenario.events = [{"t": 1.0, "event": "hit", "target_id": 0}]
    result = ScenarioResult(score=1.0, done=True, info={"hits": 1, "total": 4})

    _finish_scenario("strike_range", scenario, result, elapsed_s=12.5)

    captured = capsys.readouterr()
    assert "strike_range" in captured.out
    assert "1/4" in captured.out

    log_files = list((tmp_path / "logs").glob("strike_range_*.jsonl"))
    assert len(log_files) == 1
    assert read_event_log(log_files[0]) == scenario.events

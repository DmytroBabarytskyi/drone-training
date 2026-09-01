"""Тести рівнів сценаріїв (доп. фаза "розмаїття контенту"):
``scenarios/registry.py::list_levels``/``build_scenario(..., level=...)``."""

from __future__ import annotations

import pytest

from dronesim.scenarios.registry import build_scenario, level_file_stem, list_levels


def test_list_levels_finds_new_gate_race_levels():
    levels = list_levels("gate_race")
    assert {"sprint", "circuit6", "circuit8", "marathon", "technical", "vertical"} <= set(levels)


def test_list_levels_finds_new_strike_range_levels():
    levels = list_levels("strike_range")
    assert {"easy", "hard", "moving", "sniper", "swarm", "arena"} <= set(levels)


def test_list_levels_finds_new_patrol_levels():
    levels = list_levels("patrol")
    assert {"short", "long", "dense", "wide", "relay", "endurance"} <= set(levels)


def test_list_levels_unknown_scenario_raises():
    with pytest.raises(ValueError):
        list_levels("no_such_scenario")


def test_level_file_stem_default_none_is_bare_name():
    assert level_file_stem("gate_race", None) == "gate_race"


def test_level_file_stem_with_level_appends_suffix():
    assert level_file_stem("gate_race", "sprint") == "gate_race_sprint"


def test_build_scenario_without_level_matches_old_default_behavior():
    """Критично: жоден наявний виклик build_scenario(name, seed) без level НЕ
    повинен змінити поведінку (RL-навчання/оцінка покладаються на це)."""
    scenario = build_scenario("gate_race", seed=1)
    assert len(scenario.gates) == 4  # базова розкладка (діамант), незмінна


def test_build_scenario_with_level_loads_different_content():
    sprint = build_scenario("gate_race", seed=1, level="sprint")
    circuit8 = build_scenario("gate_race", seed=1, level="circuit8")
    assert len(sprint.gates) == 4
    assert len(circuit8.gates) == 8
    assert len(sprint.gates) != len(circuit8.gates)


def test_build_scenario_strike_range_levels_have_expected_target_counts():
    easy = build_scenario("strike_range", seed=1, level="easy")
    swarm = build_scenario("strike_range", seed=1, level="swarm")
    assert len(easy.targets) == 3
    assert len(swarm.targets) == 8


def test_build_scenario_patrol_levels_have_expected_spawn_counts():
    short = build_scenario("patrol", seed=1, level="short")
    endurance = build_scenario("patrol", seed=1, level="endurance")
    assert len(short._spawn_schedule) == 2
    assert len(endurance._spawn_schedule) == 6


def test_build_scenario_unknown_level_raises_file_not_found():
    with pytest.raises(FileNotFoundError):
        build_scenario("gate_race", seed=1, level="no_such_level")

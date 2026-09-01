"""Тести спільного реєстру сценаріїв (scenarios/registry.py, фаза 5)."""

import pytest

from dronesim.scenarios.gate_race import GateRaceScenario
from dronesim.scenarios.registry import SCENARIO_REGISTRY, build_scenario


def test_registry_covers_all_three_scenarios():
    assert set(SCENARIO_REGISTRY) == {"gate_race", "patrol", "strike_range"}


def test_build_scenario_returns_reset_instance():
    scenario = build_scenario("gate_race", seed=1)
    assert isinstance(scenario, GateRaceScenario)
    assert len(scenario.gates) == 4


def test_build_scenario_unknown_raises_value_error():
    with pytest.raises(ValueError, match="strike_range"):
        build_scenario("no_such_scenario")

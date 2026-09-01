"""Тести форматування екрану результатів (ui/results.py)."""

from dronesim.scenarios.base import ScenarioResult
from dronesim.ui.results import format_results


def test_gate_race_result_shows_laps_and_best_time():
    result = ScenarioResult(
        score=2.0, done=False, info={"laps": 2, "next_gate": 1, "lap_times": [12.5, 11.8]}
    )
    text = format_results("gate_race", result, elapsed_s=30.0)
    assert "Кіл пройдено: 2" in text
    assert "11.80" in text  # найкраще (менше) коло
    assert "gate_race" in text


def test_strike_range_result_shows_hits_and_misses():
    result = ScenarioResult(score=3.0, done=True, info={"hits": 3, "total": 4})
    text = format_results("strike_range", result, elapsed_s=45.2)
    assert "3/4" in text
    assert "промахів: 1" in text
    assert "Завершено: так" in text


def test_patrol_result_shows_observed():
    result = ScenarioResult(score=2.0, done=False, info={"observed": 2, "spawned": 2, "total": 3})
    text = format_results("patrol", result, elapsed_s=60.0)
    assert "Помічено цілей: 2/3" in text


def test_result_without_scenario_specific_info_still_formats():
    result = ScenarioResult(score=0.0, done=False, info={})
    text = format_results("unknown", result, elapsed_s=5.0)
    assert "Очки: 0" in text

"""Тести сценарію gate_race (scenarios/gate_race.py) — критерій приймання
фази 4: рахує кола й час. Синтетичні VehicleState (без фізики/апарата) —
сценарій споживає лише позицію/час, як і задумано контрактом Scenario."""

import numpy as np

from dronesim.core.contracts import VehicleState
from dronesim.scenarios.gate_race import GateRaceScenario


def _state(pos, t) -> VehicleState:
    return VehicleState(pos=np.array(pos, dtype=np.float64), t=t)


def test_no_progress_without_movement():
    scenario = GateRaceScenario()
    scenario.reset()
    result = scenario.step(_state([0.0, 0.0, 3.0], t=0.0))
    assert result.score == 0.0
    assert not result.done


def test_passing_first_gate_scores_reward_and_advances():
    scenario = GateRaceScenario()
    scenario.reset()
    scenario.step(_state([14.0, 0.0, 3.0], t=0.0))  # позаду площини воріт 0 (center x=15)
    result = scenario.step(_state([16.0, 0.0, 3.0], t=1.0))  # перетнули -> gate 0
    assert result.reward == 1.0
    assert result.info["next_gate"] == 1


def test_full_lap_increments_lap_count_and_records_lap_time():
    scenario = GateRaceScenario()
    scenario.reset()
    waypoints = [
        (14.0, 0.0, 3.0),
        (16.0, 0.0, 3.0),  # gate 0
        (0.0, 14.0, 3.0),
        (0.0, 16.0, 3.0),  # gate 1
        (-14.0, 0.0, 3.0),
        (-16.0, 0.0, 3.0),  # gate 2
        (0.0, -14.0, 3.0),
        (0.0, -16.0, 3.0),  # gate 3 -> завершує коло
    ]
    result = None
    for i, wp in enumerate(waypoints):
        result = scenario.step(_state(wp, t=float(i)))

    assert result.info["laps"] == 1
    assert len(result.info["lap_times"]) == 1
    assert result.info["lap_times"][0] > 0.0
    assert not result.done  # laps_to_win за замовчуванням 3


def test_wrong_order_gate_does_not_count():
    scenario = GateRaceScenario()
    scenario.reset()
    # Одразу летимо до gate 1 (0,15,3), пропускаючи gate 0 -> не має зарахуватись
    scenario.step(_state([0.0, 14.0, 3.0], t=0.0))
    result = scenario.step(_state([0.0, 16.0, 3.0], t=1.0))
    assert result.reward == 0.0
    assert result.info["next_gate"] == 0  # усе ще чекає на gate 0


def test_finishes_after_laps_to_win():
    scenario = GateRaceScenario()
    scenario.reset()
    scenario.laps_to_win = 1  # спростити тест на реальному об'єкті
    waypoints = [
        (14.0, 0.0, 3.0),
        (16.0, 0.0, 3.0),
        (0.0, 14.0, 3.0),
        (0.0, 16.0, 3.0),
        (-14.0, 0.0, 3.0),
        (-16.0, 0.0, 3.0),
        (0.0, -14.0, 3.0),
        (0.0, -16.0, 3.0),
    ]
    result = None
    for i, wp in enumerate(waypoints):
        result = scenario.step(_state(wp, t=float(i)))

    assert result.done
    assert any(e["event"] == "finish" for e in scenario.events)


def test_events_recorded_for_gates_and_laps():
    scenario = GateRaceScenario()
    scenario.reset()
    scenario.step(_state([14.0, 0.0, 3.0], t=0.0))
    scenario.step(_state([16.0, 0.0, 3.0], t=1.0))
    gate_events = [e for e in scenario.events if e["event"] == "gate"]
    assert len(gate_events) == 1
    assert gate_events[0]["gate_id"] == 0

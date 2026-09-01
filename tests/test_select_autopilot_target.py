"""Тести вибору цілі для автопілота (input/rl_agent.py::select_autopilot_target,
фаза 7) — узагальнено для strike_range/patrol (.targets) і gate_race (.gates)."""

import numpy as np

from dronesim.input.rl_agent import select_autopilot_target
from dronesim.scenarios.gate_race import GateRaceScenario
from dronesim.scenarios.strike_range import StrikeRangeScenario


def test_no_scenario_holds_current_position():
    target = select_autopilot_target(None, vehicle_pos=np.array([1.0, 2.0, 3.0]), t=0.0)
    assert np.allclose(target, [1.0, 2.0, 3.0])


def test_strike_range_targets_nearest_unhit():
    scenario = StrikeRangeScenario()
    scenario.reset(seed=1)
    vehicle_pos = np.array([0.0, 0.0, 0.0])
    target = select_autopilot_target(scenario, vehicle_pos, t=0.0)

    expected = min(
        scenario.targets, key=lambda tg: np.linalg.norm(tg.position_at(0.0) - vehicle_pos)
    ).position_at(0.0)
    assert np.allclose(target, expected)


def test_strike_range_skips_hit_targets():
    scenario = StrikeRangeScenario()
    scenario.reset(seed=1)
    nearest = min(
        scenario.targets, key=lambda tg: np.linalg.norm(tg.position_at(0.0) - np.zeros(3))
    )
    nearest.hit = True

    target = select_autopilot_target(scenario, vehicle_pos=np.zeros(3), t=0.0)
    assert not np.allclose(target, nearest.position_at(0.0))


def test_strike_range_falls_back_to_any_target_when_all_hit():
    scenario = StrikeRangeScenario()
    scenario.reset(seed=1)
    for tg in scenario.targets:
        tg.hit = True

    target = select_autopilot_target(scenario, vehicle_pos=np.zeros(3), t=0.0)
    assert any(np.allclose(target, tg.position_at(0.0)) for tg in scenario.targets)


def test_gate_race_targets_next_gate_center():
    scenario = GateRaceScenario()
    scenario.reset(seed=1)
    target = select_autopilot_target(scenario, vehicle_pos=np.zeros(3), t=0.0)
    assert np.allclose(target, scenario.gates[0].center)


def test_gate_race_advances_target_after_passing_gate():
    scenario = GateRaceScenario()
    scenario.reset(seed=1)
    gate0 = scenario.gates[0]
    # Пролітаємо крізь ворота 0: позаду -> попереду площини (той самий трюк, що й test_gate_race.py)
    prev_pos = gate0.center - gate0.normal * 1.0
    curr_pos = gate0.center + gate0.normal * 1.0
    from dronesim.core.contracts import VehicleState

    scenario.step(VehicleState(pos=prev_pos, t=0.0))
    scenario.step(VehicleState(pos=curr_pos, t=1.0))

    target = select_autopilot_target(scenario, vehicle_pos=curr_pos, t=1.0)
    assert np.allclose(target, scenario.gates[1].center)

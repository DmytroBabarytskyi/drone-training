"""Тести сценарію patrol (scenarios/patrol.py) — цілі з'являються за розкладом."""

import numpy as np

from dronesim.core.contracts import VehicleState
from dronesim.scenarios.patrol import PatrolScenario


def _state(pos, t) -> VehicleState:
    return VehicleState(pos=np.array(pos, dtype=np.float64), t=t)


def test_no_targets_spawned_before_first_t_spawn():
    scenario = PatrolScenario()
    scenario.reset()
    result = scenario.step(_state([0.0, 0.0, 3.0], t=0.0))  # перший спавн t_spawn=2.0
    assert result.info["spawned"] == 0
    assert result.score == 0.0


def test_target_spawns_at_scheduled_time():
    scenario = PatrolScenario()
    scenario.reset()
    result = scenario.step(_state([0.0, 0.0, 3.0], t=2.5))
    assert result.info["spawned"] == 1
    assert any(e["event"] == "spawn" for e in scenario.events)


def test_flying_close_to_spawned_target_observes_it():
    scenario = PatrolScenario()
    scenario.reset()
    scenario.step(_state([0.0, 0.0, 3.0], t=2.5))  # заспавнити ціль 0 на (10,5,3)
    result = scenario.step(_state([10.0, 5.0, 3.0], t=3.0))  # прямо на цілі
    assert result.reward == 1.0
    assert result.info["observed"] == 1


def test_far_from_target_does_not_observe():
    scenario = PatrolScenario()
    scenario.reset()
    scenario.step(_state([0.0, 0.0, 3.0], t=2.5))
    result = scenario.step(_state([0.0, 0.0, 3.0], t=3.0))
    assert result.reward == 0.0
    assert result.info["observed"] == 0


def test_observed_only_counted_once():
    scenario = PatrolScenario()
    scenario.reset()
    # Апарат уже на місці цілі в момент її спавну -> влучання одразу в цьому кроці.
    result0 = scenario.step(_state([10.0, 5.0, 3.0], t=2.5))
    result1 = scenario.step(_state([10.0, 5.0, 3.0], t=3.0))
    assert result0.reward == 1.0
    assert result1.reward == 0.0  # уже помічена, вдруге не рахується
    assert result1.info["observed"] == 1


def test_done_when_time_limit_exceeded():
    scenario = PatrolScenario()
    scenario.reset()
    result = scenario.step(_state([0.0, 0.0, 3.0], t=scenario.duration_s + 1.0))
    assert result.done
    assert result.info["timed_out"] is True


def test_not_done_while_targets_remain_unspawned_or_unobserved():
    scenario = PatrolScenario()
    scenario.reset()
    result = scenario.step(_state([0.0, 0.0, 3.0], t=2.5))  # лише перша ціль заспавнена
    assert not result.done

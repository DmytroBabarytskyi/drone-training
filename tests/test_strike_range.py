"""Тести сценарію strike_range (scenarios/strike_range.py) — критерій приймання
фази 4: детермінує влучання за геометрією (дистанція+кут)."""

import numpy as np

from dronesim.core.contracts import VehicleState
from dronesim.scenarios.strike_range import StrikeRangeScenario


def _state(pos, quat=(0.0, 0.0, 0.0, 1.0), t=0.0) -> VehicleState:
    return VehicleState(pos=np.array(pos, dtype=np.float64), quat=np.array(quat), t=t)


def test_targets_default_to_vehicle_kind_not_marker():
    """Доп. фаза "реальні моделі": цілі strike_range показуються як технічні
    моделі (техніка/артилерія), а не bullseye-маркер (дефолт dataclass
    ``Target``/patrol)."""
    scenario = StrikeRangeScenario()
    scenario.reset()
    assert all(t.kind in {"vehicle", "artillery"} for t in scenario.targets)
    assert any(t.kind == "artillery" for t in scenario.targets)  # є розмаїття типів


def test_no_hit_far_from_all_targets():
    scenario = StrikeRangeScenario()
    scenario.reset()
    result = scenario.step(_state([0.0, 0.0, 3.0], t=0.0))
    assert result.score == 0.0
    assert not result.done


def test_hit_first_static_target_close_and_aimed():
    scenario = StrikeRangeScenario()
    scenario.reset()
    # target 0 at (20,0,3): підлетіти близько на тій самій висоті, дивитись на нього (+X)
    result = scenario.step(_state([19.0, 0.0, 3.0], quat=(0.0, 0.0, 0.0, 1.0), t=0.0))
    assert result.reward == 1.0
    assert result.info["hits"] == 1
    assert any(e["event"] == "hit" and e["target_id"] == 0 for e in scenario.events)


def test_target_hit_only_counted_once():
    scenario = StrikeRangeScenario()
    scenario.reset()
    state = _state([19.0, 0.0, 3.0], t=0.0)
    result1 = scenario.step(state)
    result2 = scenario.step(_state([19.0, 0.0, 3.0], t=1.0))
    assert result1.reward == 1.0
    assert result2.reward == 0.0  # уже уражена, вдруге не рахується
    assert result2.info["hits"] == 1


def test_no_hit_when_out_of_aim_angle():
    scenario = StrikeRangeScenario()
    scenario.reset()
    # апарат близько до цілі 0 (20,0,3), але дивиться в протилежний бік (-X)
    quat_180 = (0.0, 0.0, 1.0, 0.0)  # 180° навколо Z
    result = scenario.step(_state([19.0, 0.0, 3.0], quat=quat_180, t=0.0))
    assert result.reward == 0.0


def test_done_when_all_targets_hit():
    scenario = StrikeRangeScenario()
    scenario.reset()
    for target in scenario.targets:
        target.hit = True
    result = scenario.step(_state([0.0, 0.0, 3.0], t=1.0))
    assert result.done
    assert any(e["event"] == "finish" for e in scenario.events)


def test_done_when_time_limit_exceeded():
    scenario = StrikeRangeScenario()
    scenario.reset()
    result = scenario.step(_state([0.0, 0.0, 3.0], t=scenario.time_limit_s + 1.0))
    assert result.done
    assert result.info["timed_out"] is True


def test_finish_event_recorded_only_once():
    scenario = StrikeRangeScenario()
    scenario.reset()
    for target in scenario.targets:
        target.hit = True
    scenario.step(_state([0.0, 0.0, 3.0], t=1.0))
    scenario.step(_state([0.0, 0.0, 3.0], t=2.0))
    finish_events = [e for e in scenario.events if e["event"] == "finish"]
    assert len(finish_events) == 1


def test_moving_target_hit_depends_on_time():
    # Апарат стоїть біля base_pos рухомої цілі (дивиться на неї вздовж +X).
    scenario = StrikeRangeScenario()
    scenario.reset()
    moving = next(t for t in scenario.targets if t.moves_to is not None)
    near_base = moving.base_pos + np.array([-1.0, 0.0, 0.0])

    # У t=0 ціль дійсно там (frac=0 -> base_pos) -> влучання.
    scenario.step(_state(near_base, t=0.0))
    assert moving.hit

    # Той самий апарат у тій самій точці, але в момент half_period ціль уже
    # перелетіла до moves_to (frac=1) -> тут НЕ влучання (ціль фізично не там).
    scenario2 = StrikeRangeScenario()
    scenario2.reset()
    moving2 = next(t for t in scenario2.targets if t.moves_to is not None)
    half_period = moving2.period_s / 2.0
    scenario2.step(_state(near_base, t=half_period))
    assert not moving2.hit

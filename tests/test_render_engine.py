"""Тести рушія Panda3D (render/engine.py) — offscreen, без вікна ОС.

Годинник Panda3D переводимо в нереальний час (``MNonRealTime``) із фіксованим
``dt``, щоб крок фізики в тесті був детермінованим, а не залежав від реального
часу виконання (SKILL.md, «Типові пастки»)."""

from panda3d.core import ClockObject


def _step_with_fixed_dt(engine, dt: float, n: int) -> None:
    # engine — спільний на всю сесію (conftest.py); попередні тести (реальний час)
    # могли залишити залишок у акумуляторі кадрового часу. Скидаємо його, інакше
    # цей контрольований вимір змішається з чужим "боргом" реального часу.
    engine._accumulator = 0.0
    clock = ClockObject.getGlobalClock()
    clock.setMode(ClockObject.MNonRealTime)
    clock.setDt(dt)
    for _ in range(n):
        engine.taskMgr.step()
    clock.setMode(ClockObject.MNormal)


def test_engine_creates_offscreen_without_window(engine):
    assert engine.win is not None
    assert engine.render is not None


def test_physics_callback_fires_once_per_matching_frame_dt(engine):
    calls = []
    engine.set_physics_callback(lambda dt: calls.append(dt))
    try:
        physics_dt = engine.clock.dt
        _step_with_fixed_dt(engine, physics_dt, 5)
        assert len(calls) == 5
        assert all(abs(c - physics_dt) < 1e-9 for c in calls)
    finally:
        engine.set_physics_callback(None)


def test_physics_task_caps_steps_per_frame_to_avoid_spiral_of_death(engine):
    calls = []
    engine.set_physics_callback(lambda dt: calls.append(dt))
    try:
        physics_dt = engine.clock.dt
        _step_with_fixed_dt(engine, physics_dt * 100, 1)  # величезний "лаг" за один кадр
        assert len(calls) <= 8
    finally:
        engine.set_physics_callback(None)

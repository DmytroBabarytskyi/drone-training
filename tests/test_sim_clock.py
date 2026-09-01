"""Тести годинника й циклу симуляції (фаза 0)."""

from dronesim.core.contracts import ControlCommand, FlightMode
from dronesim.core.sim_loop import SimClock, SimLoop


def test_clock_dt_and_time():
    clock = SimClock(physics_hz=240.0)
    assert clock.dt == 1 / 240
    for _ in range(240):
        clock.tick()
    assert clock.step_count == 240
    assert abs(clock.t - 1.0) < 1e-9


def test_sim_loop_runs_fixed_steps():
    loop = SimLoop(physics_hz=100.0)
    seen = []
    loop.run(lambda c: seen.append(c.step_count), steps=10)
    assert loop.clock.step_count == 10
    assert seen == list(range(10))


def test_control_command_defaults_and_array():
    cmd = ControlCommand()
    assert cmd.mode is FlightMode.ACRO
    assert cmd.arm is False
    arr = cmd.as_array()
    assert arr.shape == (4,)
    assert (arr == 0).all()

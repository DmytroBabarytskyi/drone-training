"""Тест PID-контролера (utils/pid.py)."""

from dronesim.utils.pid import PID


def test_pid_proportional_sign():
    pid = PID(kp=1.0, ki=0.0, kd=0.0)
    assert pid.update(2.0, dt=0.01) == 2.0
    assert pid.update(-3.0, dt=0.01) == -3.0


def test_pid_integral_antiwindup():
    pid = PID(kp=0.0, ki=1.0, kd=0.0, i_limit=0.5)
    for _ in range(1000):
        out = pid.update(1.0, dt=0.01)
    assert out <= 0.5 + 1e-9  # інтеграл обмежений


def test_pid_reset():
    pid = PID(kp=0.0, ki=1.0, kd=0.0)
    pid.update(1.0, dt=0.1)
    pid.reset()
    assert pid._integral == 0.0

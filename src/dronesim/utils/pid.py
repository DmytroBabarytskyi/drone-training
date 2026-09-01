"""PID-контролер — базовий будівельний блок стабілізації (rate/angle/position loops).

Реалізований повністю (фаза 2 його використовує). Гейни приходять з конфігу
апарата (``configs/vehicles/*.yaml``), а не хардкодяться.

ДОП. ФАЗА "РЕАЛІЗМ ФІЗИКИ" (обидва — опційні, дефолт відтворює СТАРУ поведінку
1:1):
- ``d_filter_alpha`` (0..1, дефолт 1.0 = без фільтра): експоненційний
  низькочастотний фільтр на D-складову (реальні польотні контролери завжди
  фільтрують похідну — сирий "d(error)/dt" підсилює будь-який шум/дрібні
  стрибки помилки в неприємний "кік"). ``alpha=1.0`` означає
  ``filtered = filtered + 1.0*(raw-filtered) = raw`` — точнісінько стара
  формула без фільтра.
- ``measurement`` (опційний аргумент ``update()``): якщо передано — похідна
  рахується як "похідна ВИМІРУ" (``-(measurement-prev_measurement)/dt``)
  замість "похідна ПОХИБКИ". Математично тотожні, ЯКЩО setpoint не
  змінюється між кроками — АЛЕ якщо setpoint РІЗКО стрибає (стрімкий рух
  стіка), похідна-від-похибки дає миттєвий "derivative kick" (D бачить
  стрибок setpoint як миттєву "швидкість помилки"), а похідна-від-виміру —
  ні (вимір апарата фізично не може стрибнути миттєво). Це те, як реально
  влаштовані PID у Betaflight/PX4.
"""

from __future__ import annotations


class PID:
    """Скалярний PID з анти-віндапом (обмеження інтегральної складової)."""

    def __init__(self, kp: float, ki: float, kd: float, i_limit: float = 1.0, d_filter_alpha: float = 1.0):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.i_limit = i_limit
        self.d_filter_alpha = d_filter_alpha
        self._integral = 0.0
        self._prev_error = 0.0
        self._prev_measurement: float | None = None
        self._filtered_derivative = 0.0

    def reset(self) -> None:
        self._integral = 0.0
        self._prev_error = 0.0
        self._prev_measurement = None
        self._filtered_derivative = 0.0

    def update(self, error: float, dt: float, measurement: float | None = None) -> float:
        self._integral += error * dt
        self._integral = max(-self.i_limit, min(self.i_limit, self._integral))

        if measurement is not None:
            if self._prev_measurement is None or dt <= 0.0:
                raw_derivative = 0.0
            else:
                raw_derivative = -(measurement - self._prev_measurement) / dt
            self._prev_measurement = measurement
        else:
            raw_derivative = (error - self._prev_error) / dt if dt > 0 else 0.0
        self._prev_error = error

        self._filtered_derivative += self.d_filter_alpha * (raw_derivative - self._filtered_derivative)

        return self.kp * error + self.ki * self._integral + self.kd * self._filtered_derivative

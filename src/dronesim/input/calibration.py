"""Чисті функції калібрування осей керування: deadzone, expo, інверсія.

Спільні для ``input/keyboard.py`` та ``input/gamepad.py``, щоб криві поводилися
однаково незалежно від фізичного джерела. Параметри (``deadzone``, ``expo``)
приходять з ``configs/input/*.yaml`` — тут лише формули, без жодного джерела вводу.
"""

from __future__ import annotations


def apply_deadzone(value: float, deadzone: float) -> float:
    """Обнулити малі значення біля центру і перемасштабувати решту в [-1, 1]."""
    if abs(value) <= deadzone:
        return 0.0
    sign = 1.0 if value > 0 else -1.0
    return sign * (abs(value) - deadzone) / (1.0 - deadzone)


def apply_expo(value: float, expo: float) -> float:
    """Нелінійна крива: плавно біля центру, різкіше на краях (як "expo" на пультах).

    ``expo`` ∈ [0, 1]: 0 — лінійно, 1 — максимально нелінійно (кубічна крива).
    """
    return (1.0 - expo) * value + expo * (value**3)


def apply_axis_curve(raw: float, deadzone: float, expo: float, invert: bool) -> float:
    """Повний конвеєр обробки осі: інверсія → deadzone → expo, результат у [-1, 1]."""
    value = -raw if invert else raw
    value = apply_deadzone(value, deadzone)
    value = apply_expo(value, expo)
    return max(-1.0, min(1.0, value))

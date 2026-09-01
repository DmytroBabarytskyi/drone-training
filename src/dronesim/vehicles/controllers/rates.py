"""BetaFlight "Actual Rates" — реальна формула перетворення стіка на цільову
кутову швидкість (доп. фаза "справжній Liftoff"), замість спрощеного
``stick * max_rate`` + окремого кубічного expo, що було раніше.

Формула звірена з офіційним вихідним кодом BetaFlight (дефолтний режим
rates з версії 4.3+): ``src/main/fc/rc.c::applyActualRates``. На відміну від
попередньої реалізації, expo й максимальна швидкість НЕ розділяються чисто
на "форма стіка" (input-шар) і "множник" (тут) — вони пов'язані в ОДНІЙ
формулі, тому цей контролер працює з СИРИМ (лише deadzone/invert, без expo)
стіком; expo реалізується ТУТ, на рівні rate-профілю апарата (як і в
реальному BetaFlight — expo/rates належать конфігу польотного контролера,
не передавачу). ``configs/input/*.yaml::expo`` тому виставлено в 0.0.
"""

from __future__ import annotations

_DEG2RAD = 3.141592653589793 / 180.0


def actual_rate_rad_s(
    stick: float, center_sensitivity_deg_s: float, max_rate_deg_s: float, expo: float
) -> float:
    """Цільова кутова швидкість (рад/с) для нормованого стіка ``stick`` (-1..1).

    ``center_sensitivity_deg_s`` — швидкість біля центру стіка (чутливість
    для дрібних корекцій), ``max_rate_deg_s`` — швидкість при повному
    відхиленні, ``expo`` (0..1) — зсуває криву БЛИЖЧЕ до центру/країв, не
    змінюючи ні центр, ні максимум (незалежні параметри, на відміну від
    старого кубічного блендінгу, де "expo" й "максимум" були зчеплені).
    """
    stick_abs = abs(stick)
    expo_curve = stick_abs * (stick**5 * expo + stick * (1.0 - expo))
    stick_movement = max(0.0, max_rate_deg_s - center_sensitivity_deg_s)
    angle_rate_deg_s = stick * center_sensitivity_deg_s + stick_movement * expo_curve
    return angle_rate_deg_s * _DEG2RAD

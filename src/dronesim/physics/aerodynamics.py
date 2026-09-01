"""Аеродинаміка мотора: крива тяги, реактивний момент, лінійний опір.

Чисті функції без залежності від Panda3D/Bullet — тестуються ізольовано, без
фізичного світу. Мотор моделюється командою ``cmd`` ∈ [0, 1], пропорційною
частоті обертання (rpm-подібний сигнал): тяга ∝ cmd² (тобто ∝ rpm², як у
типових моделях БПЛА: thrust = kf * rpm²). Параметри (``max_thrust``,
``thrust_to_torque``, ``drag_coeff``) приходять з ``configs/vehicles/*.yaml``
— тут немає жодного захардкодженого коефіцієнта.
"""

from __future__ import annotations

import numpy as np


def motor_thrust(cmd: float, max_thrust: float) -> float:
    """Тяга одного мотора (Н) від нормованої команди ``cmd`` ∈ [0, 1].

    Тяга ∝ cmd² — узгоджено з тим, що cmd пропорційний оборотам (rpm), а
    реальна тяга гвинта пропорційна rpm².
    """
    cmd_clamped = min(max(cmd, 0.0), 1.0)
    return max_thrust * cmd_clamped**2


def reactive_torque(thrust: float, thrust_to_torque: float, spin_dir: float) -> float:
    """Реактивний момент навколо осі тяги мотора (Н·м).

    Знак визначається напрямком обертання гвинта (``spin_dir``: +1 за годинниковою,
    -1 проти) — у X-конфігурації квадрокоптера сусідні мотори обертаються назустріч,
    щоб взаємно компенсувати реактивний момент при рівній тязі (yaw-нейтральність завису).
    """
    return spin_dir * thrust_to_torque * thrust


def linear_drag(velocity: np.ndarray, drag_coeff: float) -> np.ndarray:
    """Проста лінійна сила опору (Н), напрямлена проти вектора швидкості.

    Лишено для зворотної сумісності/тестів; фактичний польот (``Multirotor``)
    використовує ``quadratic_drag`` (доп. фаза "реалізм фізики") — реальний
    аеродинамічний опір квадратичний за швидкістю, лінійний був спрощенням фази 3.
    """
    return -drag_coeff * np.asarray(velocity, dtype=np.float64)


def quadratic_drag(velocity_body: np.ndarray, drag_coeff: np.ndarray | float) -> np.ndarray:
    """Квадратичний опір (Н) У ЗВ'ЯЗАНИХ ОСЯХ ТІЛА: ``F_i = -k_i * |v_i| * v_i``
    (реальний аеродинамічний опір ∝ v², не ∝ v — лінійна модель фази 3 була
    спрощенням, доп. фаза "реалізм фізики"). ``drag_coeff`` — скаляр
    (ізотропний опір) або 3-вектор (АНІЗОТРОПНИЙ: різна ефективна площа
    перерізу вздовж різних тілесних осей — типово більша вздовж Z через
    диски гвинтів/рами, ніж вздовж X/Y). Виклик очікує ``velocity_body`` ВЖЕ
    у зв'язаних осях (переклад зі світових — на боці виклику, той самий
    патерн, що й ``world_to_body_ang_vel`` у controllers/rate_loop.py)."""
    v = np.asarray(velocity_body, dtype=np.float64)
    k = np.atleast_1d(np.asarray(drag_coeff, dtype=np.float64))
    return -k * np.abs(v) * v


def motor_lag(
    current_thrust: np.ndarray, commanded_thrust: np.ndarray, tau_s: float, dt: float
) -> np.ndarray:
    """Фільтр першого порядку (спуск/розкрутка мотора): реальний мотор+гвинт не
    досягає цільової тяги миттєво (доп. фаза "реалізм фізики" — до цього тяга
    змінювалась стрибком у той самий кадр фізики). ``tau_s`` — стала часу
    (~63% шляху до цілі за ``tau_s`` секунд); дискретна форма
    ``alpha = dt/(tau_s+dt)`` — безумовно стійка (на відміну від
    ``exp(-dt/tau)``, коректна навіть коли ``dt`` порівнянний з ``tau_s`` чи
    більший, без перерегулювання)."""
    if tau_s <= 0.0:
        return np.asarray(commanded_thrust, dtype=np.float64)
    alpha = dt / (tau_s + dt)
    return current_thrust + alpha * (np.asarray(commanded_thrust, dtype=np.float64) - current_thrust)


def battery_thrust_factor(soc: float, load_frac: float, sag_coeff: float) -> float:
    """Миттєвий коефіцієнт доступної тяги (0..1) від заряду батареї (доп. фаза
    "реалізм фізики"): (1) спадає з розрядом ``soc`` (0..1), але навіть при
    порожній батареї лишає 60% (не миттєвий обрив тяги — спрощення, реальний
    LiPo під навантаженням "провисає" різкіше під кінець розряду, тут
    лінійно); (2) МИТТЄВО просідає під навантаженням ``load_frac`` (0..1,
    поточна нормована команда) і відновлюється, щойно газ відпущено —
    імітація внутрішнього опору акумулятора (просідання напруги під струмом)."""
    soc_factor = 0.6 + 0.4 * max(0.0, min(1.0, soc))
    sag = 1.0 - sag_coeff * max(0.0, min(1.0, load_frac))
    return max(0.0, soc_factor * sag)


def ground_effect_factor(height_m: float, rotor_radius_m: float, max_boost: float = 0.25) -> float:
    """Множник тяги через екранний ефект (IGE — in-ground-effect) поблизу землі
    (доп. фаза "реалізм фізики"): спрощена формула Cheeseman-Bennett
    ``T_IGE/T_OGE = 1 / (1 - (R/(4h))^2)`` (стандартна оцінка для
    гелікоптерів/мультироторів; ``R`` — радіус гвинта, ``h`` — висота гвинта
    над землею). Формула йде в нескінченність при ``h -> R/4`` — обрізано
    стелею ``max_boost`` (не з експериментальних даних цього апарата, лише
    запобіжник від вибуху формули на малій висоті). ``rotor_radius_m <= 0``
    означає "ефект вимкнено" (апарат не налаштований) -> нейтральний 1.0."""
    if rotor_radius_m <= 0.0:
        return 1.0
    if height_m <= 0.0:
        return 1.0 + max_boost
    ratio = rotor_radius_m / (4.0 * height_m)
    if ratio >= 1.0:
        return 1.0 + max_boost
    factor = 1.0 / (1.0 - ratio * ratio)
    return min(factor, 1.0 + max_boost)

"""Спільна 3D-математика: перетворення кватерніонів тощо.

Чисті функції, без залежності від Panda3D/Bullet — використовуються і
рендером (render/hud.py, для відображення), і контролерами (vehicles/
controllers/angle.py, для контуру утримання кута) — тож живуть тут, а не в
одному з цих модулів (щоб контролер не тягнув приватну функцію з render/,
порушуючи межі шарів, SKILL.md правило 6).
"""

from __future__ import annotations

import numpy as np


def quat_to_euler_rad(quat_xyzw: np.ndarray) -> tuple[float, float, float]:
    """Кути Ейлера (roll, pitch, yaw) у радіанах з кватерніона (xyzw, як VehicleState.quat)."""
    x, y, z, w = quat_xyzw
    roll = np.arctan2(2 * (w * x + y * z), 1 - 2 * (x * x + y * y))
    sin_pitch = np.clip(2 * (w * y - z * x), -1.0, 1.0)
    pitch = np.arcsin(sin_pitch)
    yaw = np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
    return float(roll), float(pitch), float(yaw)


def quat_rotate_vector(quat_xyzw: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Повернути вектор ``v`` кватерніоном (xyzw): v' = q * v * q⁻¹ (векторна форма Родріга)."""
    x, y, z, w = quat_xyzw
    q_vec = np.array([x, y, z])
    t = 2.0 * np.cross(q_vec, v)
    return v + w * t + np.cross(q_vec, t)


def quat_conjugate(quat_xyzw: np.ndarray) -> np.ndarray:
    """Спряжений кватерніон (xyzw) — для одиничного кватерніона це й обернений
    (протилежний поворот). Використовується для перекладу вектора зі світових
    координат у зв'язані (тіла): ``quat_rotate_vector(quat_conjugate(q), world_v)``."""
    x, y, z, w = quat_xyzw
    return np.array([-x, -y, -z, w])


def vehicle_forward_world(quat_xyzw: np.ndarray) -> np.ndarray:
    """Напрямок носа апарата (нормований) у світових координатах.

    "Вперед" для тіла апарата — це ЛОКАЛЬНА +X (vehicles/multirotor.py: вісь
    ``x`` розкладки моторів відповідає pitch/forward; підтверджено емпірично
    у фазі 3, docs/DECISIONS.md). Використовується сценаріями (фаза 4) для
    геометричної перевірки "прицілювання" на ціль (world/targets.py).
    """
    return quat_rotate_vector(quat_xyzw, np.array([1.0, 0.0, 0.0]))

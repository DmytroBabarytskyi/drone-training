"""Контракти даних — СПІЛЬНА МОВА всіх модулів проєкту.

Ці структури — священні (див. SKILL.md, правило 1). Будь-яка зміна поля тут
зачіпає весь проєкт: спершу онови контракт, потім усіх споживачів (`grep` по
`src/`), потім тести. Ніколи не форкай ці типи локально в модулях.

Ідея: людина (геймпад/клавіатура) і ML-агент видають ОДНАКОВИЙ ``ControlCommand``.
Фізика, контролери й рендер не знають, звідки прийшла команда.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

import numpy as np


class FlightMode(str, Enum):
    """Польотні режими (реалізації — у ``vehicles/controllers/``)."""

    ACRO = "acro"          # Rate-режим, як у Liftoff: стік = кутова швидкість
    ANGLE = "angle"        # Self-level: стік = кут нахилу
    ALT_HOLD = "alt_hold"  # утримання висоти
    POS_HOLD = "pos_hold"  # утримання позиції (GPS-hold)


# Порядок перемикання режимів клавішею/кнопкою (input/keyboard.py, input/gamepad.py).
# Якщо апарат не підтримує режим (немає потрібних секцій у конфізі) — app.py
# відкатується до ACRO; єдине джерело істини для порядку циклу — тут.
FLIGHT_MODE_CYCLE = (FlightMode.ACRO, FlightMode.ANGLE, FlightMode.ALT_HOLD, FlightMode.POS_HOLD)


@dataclass
class ControlCommand:
    """Нормована команда керування. Спільна для людини і ML.

    roll/pitch/yaw/throttle у діапазоні [-1, 1] (throttle зазвичай [0, 1]
    для мультиротора, але тримаємо float — контролер сам інтерпретує режим).
    """

    roll: float = 0.0
    pitch: float = 0.0
    yaw: float = 0.0
    throttle: float = 0.0
    arm: bool = False
    mode: FlightMode = FlightMode.ACRO

    def as_array(self) -> np.ndarray:
        return np.array([self.roll, self.pitch, self.yaw, self.throttle], dtype=np.float32)


@dataclass
class VehicleState:
    """Повний стан апарата у світових координатах (SI-одиниці)."""

    pos: np.ndarray = field(default_factory=lambda: np.zeros(3))       # м
    quat: np.ndarray = field(default_factory=lambda: np.array([0, 0, 0, 1.0]))  # xyzw
    vel: np.ndarray = field(default_factory=lambda: np.zeros(3))       # м/с
    ang_vel: np.ndarray = field(default_factory=lambda: np.zeros(3))   # рад/с
    motor_rpm: np.ndarray = field(default_factory=lambda: np.zeros(4))
    battery: float = 1.0  # [0, 1]
    t: float = 0.0        # час симуляції, с


@dataclass
class CameraFrame:
    """Кадр FPV-камери, який споживає рендер і ML."""

    rgb: np.ndarray                 # (H, W, 3) uint8
    pose: VehicleState              # поза камери на момент кадру
    t: float                        # час симуляції
    depth: np.ndarray | None = None  # (H, W) float, опційно


@dataclass
class Detection:
    """Одна детекція від ``Detector`` (формат YOLO-подібний)."""

    bbox_xywh: tuple[float, float, float, float]  # піксельні координати (центр+розмір)
    class_id: int
    conf: float

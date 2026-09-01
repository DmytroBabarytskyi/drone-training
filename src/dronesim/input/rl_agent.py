"""RL-агент як ``ControlSource`` (фаза 7) — навчена політика (SB3 PPO) керує апаратом.

Контракт спостереження — ТОЧНО той самий, що й під час навчання
(``ml/rl/observations.py::make_relative_obs``), інакше політика бачить чужий формат і
поводиться довільно. ``ControlCommand`` виходить у ``mode=POS_HOLD`` — той
самий режим, що й `DroneEnv` використовував для навчання (``vehicles/
controllers/position.py::PosHoldController``); викликач (``app.py``) має
переконатись, що апарат підтримує цей режим (SKILL.md правило 2: джерело
керування взаємозамінне — тут це діє в обидва боки: контролер не знає,
людина це чи AI, але AI видає команди, розраховані на КОНКРЕТНИЙ контролер).

``ControlSource.get_command()`` не приймає аргументів (щоб інтерфейс лишався
однаковим для людини й AI) — тому світовий контекст (стан апарата, позиція
цілі) подається ОКРЕМИМ методом ``set_context(...)`` перед кожним викликом
``get_command()``, той самий патерн, що й ``KeyboardSource.update(dt)`` (фаза 2).

КРИТИЧНО (docs/DECISIONS.md, фаза 5/7): якщо цей клас створюється в тому
самому процесі, що й Panda3D — ``RLAgentSource(...)`` (яка тягне
``stable_baselines3`` -> ``torch``) МАЄ бути створена ДО імпорту
``dronesim.render.engine``. НЕ ДОСТАТНЬО просто впорядкувати ІМПОРТИ в
``app.py`` — ``make_relative_obs`` імпортується САМЕ з
``ml/rl/observations.py`` (не з ``ml/rl/env.py``!), бо ``env.py`` транзитивно
тягне ``physics.world`` (Panda3D) для ``DroneEnv`` — навіть просто
``from ml.rl.env import make_relative_obs`` уже завантажив би Panda3D
ДО того, як ``RLAgentSource.__init__`` встигне імпортувати torch. Уроком з
цього: перевіряй ТРАНЗИТИВНІ імпорти, не лише свої власні top-level.
"""

from __future__ import annotations

import numpy as np

from dronesim.core.contracts import ControlCommand, FlightMode, VehicleState
from dronesim.input.base import ControlSource
from dronesim.ml.rl.observations import make_relative_obs


class RLAgentSource(ControlSource):
    """Навчена SB3-політика (PPO) як джерело керування."""

    def __init__(self, checkpoint_path: str):
        from stable_baselines3 import PPO

        self.model = PPO.load(checkpoint_path)
        self._state: VehicleState | None = None
        self._target_pos = np.zeros(3)

    def set_context(self, state: VehicleState, target_pos: np.ndarray) -> None:
        """Подати поточний стан апарата й позицію цілі. Викликати ПЕРЕД ``get_command()``."""
        self._state = state
        self._target_pos = target_pos

    def get_command(self) -> ControlCommand:
        if self._state is None:
            # Контекст ще не подано (set_context() жодного разу не викликали) —
            # безпечний дефолт: не armed, нічого не робити, а не падати.
            return ControlCommand(arm=False)

        obs = make_relative_obs(self._state, self._target_pos)
        action, _ = self.model.predict(obs, deterministic=True)
        action = np.clip(action, -1.0, 1.0)

        return ControlCommand(
            roll=float(action[0]),
            pitch=float(action[1]),
            yaw=float(action[2]),
            throttle=float((action[3] + 1.0) / 2.0),
            arm=True,
            mode=FlightMode.POS_HOLD,
        )

    def close(self) -> None:
        pass


def select_autopilot_target(scenario, vehicle_pos: np.ndarray, t: float) -> np.ndarray:
    """Обрати 3D-точку, куди автопілот має летіти зараз, з активного сценарію.

    Узагальнено для будь-якого сценарію (фаза 4): `strike_range`/`patrol` мають
    ``.targets`` (список ``Target``), `gate_race` — ``.gates`` (список ``Gate``,
    ціль — центр НАСТУПНИХ воріт). Евристика: найближча ще НЕ уражена ціль
    (або найближчі ворота); якщо всі уражені — найближча взагалі (не втрачає
    ціль після завершення сценарію). Без сценарію чи цілей — тримати поточну
    позицію (безпечний дефолт: зависнути, не летіти в невідомість).
    """
    gates = getattr(scenario, "gates", None)
    if gates:
        next_gate_index = getattr(scenario, "next_gate_index", 0)
        gate = gates[next_gate_index % len(gates)]
        return np.asarray(gate.center, dtype=np.float64)

    targets = getattr(scenario, "targets", None)
    if targets:
        candidates = [tg for tg in targets if not tg.hit] or list(targets)
        closest = min(
            candidates, key=lambda tg: float(np.linalg.norm(tg.position_at(t) - vehicle_pos))
        )
        return np.asarray(closest.position_at(t), dtype=np.float64)

    return np.asarray(vehicle_pos, dtype=np.float64).copy()

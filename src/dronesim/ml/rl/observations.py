"""Контракт спостереження ``DroneEnv`` — ВІДОКРЕМЛЕНО від ``env.py`` навмисно (фаза 7).

Чиста numpy-функція, БЕЗ жодної залежності від Panda3D/Bullet — на відміну від
``env.py``, який імпортує ``PhysicsWorld``/``Multirotor`` (Panda3D) для
самого `DroneEnv`. ``input/rl_agent.py`` (інференс у грі) імпортує САМЕ
звідси, а не з ``env.py``, щоб НЕ тягнути Panda3D транзитивно.

КРИТИЧНО (docs/DECISIONS.md, фаза 5/7): якщо ``input/rl_agent.py`` імпортує
``make_relative_obs`` з ``env.py``, це транзитивно завантажує
``dronesim.physics.world`` (панда-модулі) ще ДО того, як ``RLAgentSource``
встигає імпортувати ``stable_baselines3``/``torch`` — на Windows це ламає
торч (DLL-конфлікт), навіть якщо порядок імпортів у самому ``app.py``
здавався правильним. Тримай цей модуль СПРАВДІ вільним від Panda3D-імпортів.
"""

from __future__ import annotations

import numpy as np

from dronesim.core.contracts import VehicleState
from dronesim.utils.math3d import quat_conjugate, quat_rotate_vector

OBS_DIM = 13  # rel_pos_body(3) + vel_body(3) + quat(4) + ang_vel_body(3)


def make_relative_obs(state: VehicleState, target_pos: np.ndarray) -> np.ndarray:
    """Спостереження: rel_pos_body(3) + vel_body(3) + quat(4) + ang_vel_body(3),
    усе в тілі-фреймі апарата (крім quat). СПІЛЬНА для навчання (``ml/rl/env.py``)
    і інференсу (``input/rl_agent.py``) — навчена політика має бачити ТОЧНО
    той самий формат, що й під час навчання (SKILL.md, «RL не вчиться»/дрейф
    контрактів). Не дублюй цю логіку — імпортуй звідси."""
    q_conj = quat_conjugate(state.quat)
    rel_body = quat_rotate_vector(q_conj, target_pos - state.pos)
    vel_body = quat_rotate_vector(q_conj, state.vel)
    return np.concatenate([rel_body, vel_body, state.quat, state.ang_vel]).astype(np.float32)

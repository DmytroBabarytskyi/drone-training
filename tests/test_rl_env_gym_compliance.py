"""Перевірка відповідності DroneEnv офіційному Gymnasium API (фаза 6)."""

import warnings

from gymnasium.utils.env_checker import check_env

from dronesim.ml.rl.env import DroneEnv


def test_env_passes_gymnasium_check_env():
    env = DroneEnv()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # незашкідливі попередження про необмежений Box
        check_env(env, skip_render_check=True)

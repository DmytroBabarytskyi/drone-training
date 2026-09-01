"""Побудова argv і запуск ``dronesim.app`` ОКРЕМИМ процесом (доп. фаза
"GUI-лаунчер") — GUI ніколи не імпортує Panda3D/torch напряму, тож усі
DLL-порядкові пастки (docs/DECISIONS.md) просто не стосуються цього процесу.

``argv``-функції — чисті (без побічних ефектів), тестовані без Qt і без
реального запуску процесу; ``launch()`` — єдина функція з побічним ефектом.
"""

from __future__ import annotations

import os
import subprocess
import sys

from dronesim.core.config import REPO_ROOT


def build_fly_argv(
    vehicle: str,
    input_source: str,
    scenario: str | None = None,
    level: str | None = None,
    seed: int | None = None,
    cam_width: int = 320,
    cam_height: int = 240,
    wind: float = 0.0,
    detector: str | None = None,
    autopilot: str | None = None,
) -> list[str]:
    argv = [
        sys.executable, "-m", "dronesim.app", "fly",
        "--vehicle", vehicle, "--input", input_source,
        "--cam-width", str(cam_width), "--cam-height", str(cam_height),
    ]
    if scenario:
        argv += ["--scenario", scenario]
        if level:
            argv += ["--level", level]
    if seed is not None:
        argv += ["--seed", str(seed)]
    if wind:
        argv += ["--wind", str(wind)]
    if detector:
        argv += ["--detector", detector]
    if autopilot:
        argv += ["--autopilot", autopilot]
    return argv


def build_autonomous_argv(
    checkpoint: str,
    scenario: str = "strike_range",
    level: str | None = None,
    vehicle: str = "quad_large",
    seed: int | None = None,
) -> list[str]:
    argv = [
        sys.executable, "-m", "dronesim.app", "autonomous",
        "--checkpoint", checkpoint, "--scenario", scenario, "--vehicle", vehicle,
    ]
    if level:
        argv += ["--level", level]
    if seed is not None:
        argv += ["--seed", str(seed)]
    return argv


def launch(argv: list[str]) -> subprocess.Popen:
    """Запустити симуляцію окремим процесом (``Popen``, не ``.run()`` — GUI
    лишається відповідним, можна запускати кілька разів поспіль).

    ``PYTHONPATH`` явно містить ``src/`` — щоб дочірній ``python -m
    dronesim.app`` ГАРАНТОВАНО запускав ВИХІДНИЙ код проєкту, а не якусь
    стару копію в site-packages (інакше зміни в грі не відобразились би у
    польоті, запущеному з лаунчера). ``{**os.environ, ...}`` — повне
    середовище + доповнення (НЕ частковий словник: на Windows частковий env
    ламає нативні DLL torch, docs/DECISIONS.md)."""
    src = str(REPO_ROOT / "src")
    existing = os.environ.get("PYTHONPATH", "")
    pythonpath = src + (os.pathsep + existing if existing else "")
    env = {**os.environ, "PYTHONPATH": pythonpath}
    return subprocess.Popen(argv, cwd=REPO_ROOT, env=env)

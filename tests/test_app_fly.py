"""Смоук-тест CLI-команди ``fly`` (фаза 2, критерій приймання ROADMAP).

Запускається в ОКРЕМОМУ процесі (subprocess), а не в межах основного pytest-
процесу: ``fly`` створює власний ``Engine``/``ShowBase``, а Panda3D погано
переносить кілька екземплярів ``ShowBase`` в одному процесі (конфлікт із
session-фікстурою ``engine`` в conftest.py, яку використовують інші тести
рендеру). ``--offscreen --max-steps`` дають скінченний headless-прогін замість
блокуючого ``engine.run()``."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]

# {**os.environ, ...}, НЕ {"PYTHONPATH": ...} окремо — інакше дочірній процес
# лишається БЕЗ PATH/SYSTEMROOT/... (на Windows це ламає нативні DLL, напр.
# torch/asyncio -> WinError 10106; docs/DECISIONS.md, фаза 7).
_SUBPROCESS_ENV = {**os.environ, "PYTHONPATH": str(REPO_ROOT / "src")}


def _run_fly(extra_args: list[str], timeout: int = 60) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, "-m", "dronesim.app", "fly", *extra_args],
        cwd=REPO_ROOT,
        env=_SUBPROCESS_ENV,
        capture_output=True,
        text=True,
        timeout=timeout,
    )


def test_fly_keyboard_headless_smoke():
    result = _run_fly(
        ["--input", "keyboard", "--offscreen", "--max-steps", "120", "--cam-width", "64", "--cam-height", "48"]
    )
    assert result.returncode == 0, f"stdout={result.stdout}\nstderr={result.stderr}"


def test_fly_quad_large_with_wind_headless_smoke():
    """Фаза 3: великий БПЛА + режими Angle/Alt-hold/Pos-hold + вітер, повний CLI."""
    result = _run_fly(
        [
            "--vehicle", "quad_large", "--input", "keyboard", "--offscreen",
            "--max-steps", "300", "--cam-width", "64", "--cam-height", "48", "--wind", "5.0",
        ]
    )
    assert result.returncode == 0, f"stdout={result.stdout}\nstderr={result.stderr}"


def test_fly_gate_race_scenario_headless_smoke():
    """Фаза 4: --scenario gate_race підключається без падінь у повному CLI."""
    result = _run_fly(
        [
            "--vehicle", "quad_large", "--input", "keyboard", "--scenario", "gate_race",
            "--seed", "42", "--offscreen", "--max-steps", "120", "--cam-width", "64", "--cam-height", "48",
        ]
    )
    assert result.returncode == 0, f"stdout={result.stdout}\nstderr={result.stderr}"


def test_fly_strike_range_scenario_headless_smoke():
    result = _run_fly(
        [
            "--vehicle", "quad_large", "--input", "keyboard", "--scenario", "strike_range",
            "--seed", "1", "--offscreen", "--max-steps", "120", "--cam-width", "64", "--cam-height", "48",
        ]
    )
    assert result.returncode == 0, f"stdout={result.stdout}\nstderr={result.stderr}"


def test_fly_patrol_scenario_headless_smoke():
    result = _run_fly(
        [
            "--vehicle", "quad_large", "--input", "keyboard", "--scenario", "patrol",
            "--offscreen", "--max-steps", "120", "--cam-width", "64", "--cam-height", "48",
        ]
    )
    assert result.returncode == 0, f"stdout={result.stdout}\nstderr={result.stderr}"


def test_fly_unknown_scenario_rejected_by_argparse():
    result = _run_fly(["--scenario", "no_such_scenario", "--offscreen", "--max-steps", "10"])
    assert result.returncode != 0


def test_fly_unknown_vehicle_config_fails_clearly():
    result = _run_fly(
        ["--vehicle", "no_such_vehicle", "--input", "keyboard", "--offscreen", "--max-steps", "10"]
    )
    assert result.returncode != 0
    assert "no_such_vehicle" in result.stderr or "FileNotFoundError" in result.stderr

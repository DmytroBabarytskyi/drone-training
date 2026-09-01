"""Тести автономного режиму (фаза 7): ``dronesim autonomous``, ``fly --autopilot``,
перемикач Manual↔Autonomous. Усе через subprocess — той самий привід, що й
tests/test_app_fly.py (``build_fly_session`` створює власний ``Engine``,
Panda3D погано переносить кілька ``ShowBase`` в одному процесі)."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path
from textwrap import dedent

REPO_ROOT = Path(__file__).resolve().parents[1]
CHECKPOINT = REPO_ROOT / "runs" / "rl" / "ppo_strike" / "model.zip"

# ВАЖЛИВО: {**os.environ, ...}, НЕ {"PYTHONPATH": ...} окремо — інакше дочірній
# процес лишається БЕЗ PATH/SYSTEMROOT/... і torch падає (asyncio.windows_events
# -> _overlapped -> WinError 10106), геть інша помилка за DLL-порядок Panda3D
# (docs/DECISIONS.md, фаза 7) — виявлено емпірично саме на цих тестах.
_SUBPROCESS_ENV = {**os.environ, "PYTHONPATH": str(REPO_ROOT / "src")}


def _run(argv: list[str], timeout: int = 90) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, *argv],
        cwd=REPO_ROOT,
        env=_SUBPROCESS_ENV,
        capture_output=True,
        text=True,
        timeout=timeout,
    )


def _skip_if_no_checkpoint():
    import pytest

    if not CHECKPOINT.exists():
        pytest.skip(f"Немає навченого чекпойнта {CHECKPOINT} (запусти ml/rl/train.py, фаза 6)")


def test_autonomous_scenario_headless_smoke():
    """Критерій приймання ROADMAP: dronesim autonomous --scenario strike_range
    проходить сценарій без ручного втручання (тут — не падає за N кроків)."""
    _skip_if_no_checkpoint()
    result = _run(
        [
            "-m", "dronesim.app", "autonomous",
            "--scenario", "strike_range", "--checkpoint", str(CHECKPOINT),
            "--seed", "1", "--offscreen", "--max-steps", "200",
            "--cam-width", "64", "--cam-height", "48",
        ]
    )
    assert result.returncode == 0, f"stdout={result.stdout}\nstderr={result.stderr}"


def test_fly_with_autopilot_flag_headless_smoke():
    """--autopilot лише УВІМКНЮЄ можливість (старт у MANUAL) — не повинно падати."""
    _skip_if_no_checkpoint()
    result = _run(
        [
            "-m", "dronesim.app", "fly",
            "--vehicle", "quad_large", "--scenario", "strike_range",
            "--autopilot", str(CHECKPOINT), "--seed", "1",
            "--offscreen", "--max-steps", "100", "--cam-width", "64", "--cam-height", "48",
        ]
    )
    assert result.returncode == 0, f"stdout={result.stdout}\nstderr={result.stderr}"


def test_autopilot_requires_pos_hold_capable_vehicle():
    """fpv_5inch не має Pos-hold -> --autopilot має впасти з чіткою помилкою,
    не мовчки поводитись довільно (SKILL.md: явна помилка краще тихого багу)."""
    _skip_if_no_checkpoint()
    result = _run(
        [
            "-m", "dronesim.app", "fly",
            "--vehicle", "fpv_5inch", "--autopilot", str(CHECKPOINT),
            "--offscreen", "--max-steps", "10",
        ]
    )
    assert result.returncode != 0
    assert "Pos-hold" in result.stderr


def test_autonomous_vehicle_actually_gains_and_holds_altitude():
    """Регресійний тест на реальний баг (доп. фаза поліш, не лише "не падає"):
    апарат стартував на висоті ручного зльоту (1м) — надто низько для
    RL-політики, навченої НАВІГАЦІЇ вже в повітрі (ml/rl_strike.yaml,
    ``start_altitude_range`` ніколи не ставить старт біля землі) — і осідав
    на землю ще до стабілізації Pos-hold, "застрягаючи" там (реальний фідбек
    користувача: "заструє в землі і не летить"). Виправлено: автономний
    старт відбувається на 3м (``app.py::build_fly_session``,
    ``starts_airborne``), як і в ``scripts/eval_autonomous.py``.

    Годинник Panda3D переводимо в ``MNonRealTime`` із фіксованим dt (той самий
    прийом, що й tests/test_render_engine.py / test_app_restart.py) — інакше
    кількість реально пройдених симульованих секунд за N викликів
    ``taskMgr.step()`` залежить від навантаження машини (кожен виклик містить
    інференс RL-політики через torch, тобто реальний час на ітерацію
    непостійний), і тест був би недетермінованим (виявлено емпірично — падав
    саме через це, не через реальну поломку польоту)."""
    _skip_if_no_checkpoint()
    script = dedent(f"""
        import argparse
        import sys
        sys.path.insert(0, r"{REPO_ROOT / "src"}")

        from dronesim.app import build_fly_session

        args = argparse.Namespace(
            vehicle="quad_large", input="keyboard", scenario="strike_range", seed=1,
            cam_width=64, cam_height=48, wind=0.0, detector=None, detector_conf=0.25,
            offscreen=True, max_steps=None, autopilot=r"{CHECKPOINT}", start_autonomous=True,
        )
        session = build_fly_session(args)
        engine = session["engine"]
        vehicle = session["vehicle"]

        from panda3d.core import ClockObject  # ПІСЛЯ build_fly_session (torch вже імпортовано)
        physics_dt = engine.clock.dt
        engine._accumulator = 0.0
        clock = ClockObject.getGlobalClock()
        clock.setMode(ClockObject.MNonRealTime)
        clock.setDt(physics_dt)
        for _ in range(int(3.0 / physics_dt)):  # 3с симульованого часу, детерміновано
            engine.taskMgr.step()
        clock.setMode(ClockObject.MNormal)

        altitude = vehicle.state.pos[2]
        assert altitude > 1.5, f"апарат застряг біля землі: altitude={{altitude:.3f}}m"
        print("ALTITUDE_OK", altitude)
    """)
    result = subprocess.run(
        [sys.executable, "-c", script],
        cwd=REPO_ROOT,
        env=_SUBPROCESS_ENV,
        capture_output=True,
        text=True,
        timeout=90,
    )
    assert result.returncode == 0, f"stdout={result.stdout}\nstderr={result.stderr}"
    assert "ALTITUDE_OK" in result.stdout


def test_toggle_switches_between_manual_and_autonomous():
    """Перевіряє САМ перемикач (клавіша 'p'): будуємо сесію напряму (не через
    dronesim.app CLI), симулюємо подію 'p' через Panda3D messenger, дивимось
    на internal state_box, повернутий build_fly_session (критерій приймання:
    клавіша перехоплення миттєво повертає керування людині)."""
    _skip_if_no_checkpoint()
    script = dedent(f"""
        import argparse
        import sys
        sys.path.insert(0, r"{REPO_ROOT / "src"}")

        from dronesim.app import build_fly_session

        args = argparse.Namespace(
            vehicle="quad_large", input="keyboard", scenario="strike_range", seed=1,
            cam_width=64, cam_height=48, wind=0.0, detector=None, detector_conf=0.25,
            offscreen=True, max_steps=None, autopilot=r"{CHECKPOINT}", start_autonomous=False,
        )
        session = build_fly_session(args)
        engine = session["engine"]
        state = session["state"]

        assert state["autonomous"] is False, "має стартувати в MANUAL"

        engine.messenger.send("p")
        engine.taskMgr.step()
        assert state["autonomous"] is True, "після 'p' має бути AUTONOMOUS"

        engine.messenger.send("p")
        engine.taskMgr.step()
        assert state["autonomous"] is False, "після другого 'p' має повернутись MANUAL"

        print("TOGGLE_OK")
    """)
    result = subprocess.run(
        [sys.executable, "-c", script],
        cwd=REPO_ROOT,
        env=_SUBPROCESS_ENV,
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 0, f"stdout={result.stdout}\nstderr={result.stderr}"
    assert "TOGGLE_OK" in result.stdout

"""Тести рестарту (доп. фаза поліш): ручна клавіша + авто-рестарт після
переходу критичних меж безпеки (configs/safety.yaml). Через subprocess — той
самий привід, що й tests/test_app_autonomous.py (``build_fly_session`` створює
власний ``Engine``, Panda3D погано переносить кілька ``ShowBase`` в процесі)."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path
from textwrap import dedent

REPO_ROOT = Path(__file__).resolve().parents[1]
_SUBPROCESS_ENV = {**os.environ, "PYTHONPATH": str(REPO_ROOT / "src")}


def _run_script(script: str, timeout: int = 60) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, "-c", script],
        cwd=REPO_ROOT,
        env=_SUBPROCESS_ENV,
        capture_output=True,
        text=True,
        timeout=timeout,
    )


def test_manual_restart_key_resets_vehicle_position():
    script = dedent(f"""
        import argparse
        import sys
        import numpy as np
        sys.path.insert(0, r"{REPO_ROOT / "src"}")

        from dronesim.app import build_fly_session

        args = argparse.Namespace(
            vehicle="quad_large", input="keyboard", scenario=None, seed=1,
            cam_width=64, cam_height=48, wind=0.0, detector=None, detector_conf=0.25,
            offscreen=True, max_steps=None, autopilot=None, start_autonomous=False,
        )
        session = build_fly_session(args)
        engine = session["engine"]
        vehicle = session["vehicle"]

        vehicle.reset(pos=np.array([40.0, 0.0, 20.0]))
        assert abs(vehicle.state.pos[0] - 40.0) < 0.5

        engine.messenger.send("r")
        engine.taskMgr.step()

        # Мануальний старт тепер НА ЗЕМЛІ (як у Liftoff, не 1м у повітрі):
        # рестарт повертає апарат у центр (XY~0) біля землі (не на 20м).
        assert np.linalg.norm(vehicle.state.pos[:2]) < 0.5, vehicle.state.pos
        assert vehicle.state.pos[2] < 0.5, vehicle.state.pos
        print("RESTART_OK")
    """)
    result = _run_script(script)
    assert result.returncode == 0, f"stdout={result.stdout}\nstderr={result.stderr}"
    assert "RESTART_OK" in result.stdout


def test_auto_restart_after_grace_period_when_out_of_bounds():
    """Критерій приймання (доп. фаза поліш): переліт твердої межі поля веде до
    авто-рестарту після ``crash_grace_period_s`` (configs/safety.yaml), без
    ручного втручання.

    Годинник Panda3D переводимо в ``MNonRealTime`` із фіксованим dt (той самий
    прийом, що й tests/test_render_engine.py) — інакше крок фізики залежить
    від РЕАЛЬНОГО часу виконання python-циклу, а не від кількості викликів
    ``taskMgr.step()``, і тест був би недетермінованим/повільним.
    """
    script = dedent(f"""
        import argparse
        import sys
        import numpy as np
        sys.path.insert(0, r"{REPO_ROOT / "src"}")

        from panda3d.core import ClockObject
        from dronesim.app import build_fly_session
        from dronesim.core.safety import load_safety_config

        args = argparse.Namespace(
            vehicle="quad_large", input="keyboard", scenario=None, seed=1,
            cam_width=64, cam_height=48, wind=0.0, detector=None, detector_conf=0.25,
            offscreen=True, max_steps=None, autopilot=None, start_autonomous=False,
        )
        session = build_fly_session(args)
        engine = session["engine"]
        vehicle = session["vehicle"]
        cfg = load_safety_config()

        vehicle.reset(pos=np.array([cfg.bounds_radius_hard_m + 5.0, 0.0, 5.0]))

        physics_dt = engine.clock.dt
        steps_needed = int((cfg.crash_grace_period_s + 1.0) / physics_dt)

        engine._accumulator = 0.0
        clock = ClockObject.getGlobalClock()
        clock.setMode(ClockObject.MNonRealTime)
        clock.setDt(physics_dt)
        for _ in range(steps_needed):
            engine.taskMgr.step()
        clock.setMode(ClockObject.MNormal)

        assert np.linalg.norm(vehicle.state.pos[:2]) < cfg.bounds_radius_soft_m, vehicle.state.pos
        print("AUTO_RESTART_OK")
    """)
    result = _run_script(script, timeout=120)
    assert result.returncode == 0, f"stdout={result.stdout}\nstderr={result.stderr}"
    assert "AUTO_RESTART_OK" in result.stdout

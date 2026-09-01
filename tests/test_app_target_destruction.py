"""Тест знищення цілей strike_range (доп. фаза "реальні моделі"): ураження
цілі видаляє її 3D-модель (техніка/артилерія) і тригерить ``ExplosionEffect``
рівно раз, замість перефарбовки bullseye-маркера (patrol). Через subprocess —
той самий привід, що й tests/test_app_autonomous.py."""

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


def test_hitting_strike_range_target_destroys_model_and_triggers_explosion():
    script = dedent(f"""
        import argparse
        import sys
        import numpy as np
        sys.path.insert(0, r"{REPO_ROOT / "src"}")

        from dronesim.app import build_fly_session

        args = argparse.Namespace(
            vehicle="quad_large", input="keyboard", scenario="strike_range", seed=1,
            cam_width=64, cam_height=48, wind=0.0, detector=None, detector_conf=0.25,
            offscreen=True, max_steps=None, autopilot=None, start_autonomous=False,
        )
        session = build_fly_session(args)
        engine = session["engine"]
        vehicle = session["vehicle"]
        scenario = session["scenario"]
        marker_by_id = session["marker_by_id"]
        active_effects = session["active_effects"]

        # Ціль 0 (strike_range.yaml): хіт-центр pos=[20,0,3] -> телепортуємо
        # апарат впритул на тій самій висоті, дивлячись на неї (+X, дефолтний
        # quat) -> геометричне влучання.
        vehicle.reset(pos=np.array([19.0, 0.0, 3.0]))
        engine.taskMgr.step()

        assert scenario.targets[0].hit is True, "ціль мала бути уражена"
        marker = marker_by_id[0]
        assert marker.isEmpty(), "модель цілі мала бути видалена (знищена)"
        assert len(active_effects) == 1, f"мав спрацювати рівно 1 ExplosionEffect, отримано {{len(active_effects)}}"

        # Ще один кадр -> той самий hit НЕ повинен тригернути ДРУГИЙ вибух.
        engine.taskMgr.step()
        assert len(active_effects) == 1, "повторний хіт по вже знищеній цілі не повинен додавати ще один ефект"

        print("DESTRUCTION_OK")
    """)
    result = _run_script(script)
    assert result.returncode == 0, f"stdout={result.stdout}\nstderr={result.stderr}"
    assert "DESTRUCTION_OK" in result.stdout


def test_patrol_targets_still_use_marker_recolor_not_explosion():
    """patrol — розвідка/спостереження, НЕ "ураження" — цілі мають лишатись
    bullseye-маркерами, що перефарбовуються, без ефектів знищення."""
    script = dedent(f"""
        import argparse
        import sys
        from panda3d.core import ClockObject
        sys.path.insert(0, r"{REPO_ROOT / "src"}")

        from dronesim.app import build_fly_session

        args = argparse.Namespace(
            vehicle="quad_large", input="keyboard", scenario="patrol", seed=1,
            cam_width=64, cam_height=48, wind=0.0, detector=None, detector_conf=0.25,
            offscreen=True, max_steps=None, autopilot=None, start_autonomous=False,
        )
        session = build_fly_session(args)
        engine = session["engine"]
        scenario = session["scenario"]

        # Детерміновано прогнати до першого t_spawn (configs/scenarios/patrol.yaml:
        # 2.0с) -> перша ціль з'являється в scenario.targets.
        physics_dt = engine.clock.dt
        engine._accumulator = 0.0
        clock = ClockObject.getGlobalClock()
        clock.setMode(ClockObject.MNonRealTime)
        clock.setDt(physics_dt)
        for _ in range(int(3.0 / physics_dt)):
            engine.taskMgr.step()
        clock.setMode(ClockObject.MNormal)

        assert len(scenario.targets) > 0, "ціль мала вже з'явитися за розкладом"
        assert all(t.kind == "marker" for t in scenario.targets), (
            "patrol-цілі мають лишатись дефолтним kind='marker' (не техніка/артилерія)"
        )
        print("PATROL_KIND_OK")
    """)
    result = _run_script(script)
    assert result.returncode == 0, f"stdout={result.stdout}\nstderr={result.stderr}"
    assert "PATROL_KIND_OK" in result.stdout

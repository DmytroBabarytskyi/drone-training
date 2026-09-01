"""Тест перемикача камери FPV<->chase (доп. фаза поліш): клавіша 'c'.
Через subprocess — той самий привід, що й tests/test_app_autonomous.py
(``build_fly_session`` створює власний ``Engine``)."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path
from textwrap import dedent

REPO_ROOT = Path(__file__).resolve().parents[1]
_SUBPROCESS_ENV = {**os.environ, "PYTHONPATH": str(REPO_ROOT / "src")}


def test_camera_toggle_key_switches_fpv_and_chase():
    script = dedent(f"""
        import argparse
        import sys
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
        state = session["state"]

        assert state["camera_mode"] == "fpv"
        assert engine.camera.getParent() == vehicle.node_path

        engine.messenger.send("c")
        engine.taskMgr.step()
        assert state["camera_mode"] == "chase"
        assert engine.camera.getParent() == engine.render

        engine.messenger.send("c")
        engine.taskMgr.step()
        assert state["camera_mode"] == "fpv"
        assert engine.camera.getParent() == vehicle.node_path

        print("CAMERA_TOGGLE_OK")
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
    assert "CAMERA_TOGGLE_OK" in result.stdout

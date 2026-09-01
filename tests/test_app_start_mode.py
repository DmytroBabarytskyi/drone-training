"""Тест стартового режиму польоту (доп. фаза "справжній Liftoff"):
``build_fly_session`` вибирає Angle (самовирівнювання), якщо апарат його
підтримує, замість Acro — стабільніше перше знайомство з керуванням
(реальний фідбек користувача: "вилітаю як ракета"). Через subprocess — той
самий привід, що й tests/test_app_autonomous.py (``build_fly_session``
створює власний ``Engine``)."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path
from textwrap import dedent

REPO_ROOT = Path(__file__).resolve().parents[1]
_SUBPROCESS_ENV = {**os.environ, "PYTHONPATH": str(REPO_ROOT / "src")}


def _check_start_mode(vehicle: str, expected_mode: str) -> None:
    script = dedent(f"""
        import argparse
        import sys
        sys.path.insert(0, r"{REPO_ROOT / "src"}")

        from dronesim.app import build_fly_session

        args = argparse.Namespace(
            vehicle="{vehicle}", input="keyboard", scenario=None, seed=1,
            cam_width=64, cam_height=48, wind=0.0, detector=None, detector_conf=0.25,
            offscreen=True, max_steps=None, autopilot=None, start_autonomous=False,
        )
        session = build_fly_session(args)
        active_mode = session["state"]["active_mode"].value
        assert active_mode == "{expected_mode}", f"очікувано {expected_mode}, отримано {{active_mode}}"
        # cmd.mode з control_source має ЗБІГАТИСЬ (не лише internal state_box)
        cmd_mode = session["control_source"].get_command().mode.value
        assert cmd_mode == "{expected_mode}", f"control_source: очікувано {expected_mode}, отримано {{cmd_mode}}"
        print("START_MODE_OK")
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
    assert "START_MODE_OK" in result.stdout


def test_fpv_5inch_starts_in_angle_mode_now_that_it_supports_it():
    _check_start_mode("fpv_5inch", "angle")


def test_quad_large_starts_in_angle_mode():
    _check_start_mode("quad_large", "angle")

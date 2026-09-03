"""Watch a 3-D interception engagement in a window.

Opens the normal Panda3D renderer, scrambles a mesh of interceptors against an
inbound cruise-missile-class threat, and flies the whole engagement in real
time: climb, form up across the approach corridor, seeker handover, detonation
or leak-through.

Run headless batches with ``scripts/eval_swarm3d.py`` instead.

Usage:
    python scripts/run_swarm3d.py
    python scripts/run_swarm3d.py --seed 7 --speed 0.5
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from dronesim.core.config import load_config  # noqa: E402
from dronesim.swarm3d.engagement import Engagement, Phase  # noqa: E402

THREAT_COLOR = (0.85, 0.25, 0.20, 1.0)
DRONE_COLOR = (0.80, 0.85, 0.90, 1.0)
AIM_COLOR = (0.25, 0.55, 0.85, 1.0)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="swarm3d/engagement")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument(
        "--speed",
        type=float,
        default=1.0,
        help="Playback rate. 0.25 slows the terminal phase down enough to watch.",
    )
    parser.add_argument("--offscreen", action="store_true", help="No window (smoke test)")
    parser.add_argument("--max-steps", type=int, default=0, help="Stop after N physics steps")
    args = parser.parse_args(argv)

    cfg = load_config(args.config)

    from dronesim.render.engine import Engine
    from dronesim.render.visuals import add_box_visual, add_ground_visual

    engine = Engine(
        physics_hz=float(cfg.sim.physics_hz),
        window_title="dronesim — swarm interception",
        offscreen=args.offscreen,
        shadow_extent_m=400.0,
    )

    engagement = Engagement(cfg, seed=args.seed, parent=engine.render)

    add_ground_visual(engine.render, size=3000.0, tile_size_m=25.0)

    # Interceptors get a visible body; the threat and the aim marker are plain
    # nodes we reposition by hand each frame, since neither is a Bullet body.
    for drone in engagement.drones:
        add_box_visual(drone.node_path, half_extents=(0.25, 0.25, 0.10), color=DRONE_COLOR)

    threat_node = engine.render.attachNewNode("threat")
    add_box_visual(threat_node, half_extents=(3.0, 0.6, 0.6), color=THREAT_COLOR)

    aim_node = engine.render.attachNewNode("aim")
    add_box_visual(aim_node, half_extents=(0.4, 6.0, 6.0), color=(*AIM_COLOR[:3], 0.25))
    aim_node.setTransparency(True)

    # Camera: off to one side of the intercept plane, looking back up the
    # threat's approach path, so the mesh forming up and the threat closing are
    # both in frame.
    plane_x = engagement.plane_x
    engine.camera.setPos(plane_x - 260.0, -300.0, 170.0)
    engine.camera.lookAt(plane_x + 60.0, 0.0, 90.0)

    state = {"steps": 0, "done": False, "announced": set(), "carry": 0.0}

    def physics_step(dt: float) -> None:
        if state["done"]:
            return

        # `speed` scales how much engagement time advances per rendered frame;
        # the engagement itself always steps at its own fixed dt.
        state["carry"] += args.speed
        while state["carry"] >= 1.0 and not state["done"]:
            state["carry"] -= 1.0
            finished = engagement.step()
            state["steps"] += 1

            if engagement.phase.value not in state["announced"]:
                state["announced"].add(engagement.phase.value)
                print(f"[{engagement.t:6.2f}s] phase -> {engagement.phase.value}")

            if finished or (args.max_steps and state["steps"] >= args.max_steps):
                state["done"] = True
                result = engagement._result
                print()
                print(f"  detonated        : {result.detonated}")
                print(f"  intercepted      : {result.intercepted}")
                print(f"  drones in lethal : {result.drones_within_lethal}")
                print(f"  closest approach : {result.miss_distance_m:.1f} m")
                print(f"  formation error  : {result.formation_error_m:.1f} m")
                print(f"  resolved at      : {result.time_to_resolution_s} s")
                if not args.offscreen:
                    print("\n  Close the window to exit.")

        threat_node.setPos(*engagement.threat_pos)
        aim_node.setPos(plane_x, float(engagement._aim_point[0]), float(engagement._aim_point[1]))

    engine.set_physics_callback(physics_step)

    if args.offscreen:
        for _ in range(args.max_steps or 2000):
            engine.taskMgr.step()
            if state["done"]:
                break
        return 0

    engine.run()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

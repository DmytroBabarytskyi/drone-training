"""Headless-перевірка фізики (фаза 1, критерій приймання).

Підбирає нормовану команду мотора (однакову для всіх 4), за якої сумарна тяга
дорівнює вазі апарата, і крутить 10 с симуляції без вікна/рендеру — лише Bullet
(SKILL.md: фізика працює headless). Перевіряє відсутність NaN і що дрон
утримує приблизний завис (не падає й не йде в нескінченний розгін).

Запуск: python scripts/hover_test.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from dronesim.core.sim_loop import SimClock  # noqa: E402
from dronesim.physics import aerodynamics  # noqa: E402
from dronesim.physics.world import PhysicsWorld  # noqa: E402
from dronesim.vehicles.quad_fpv import QuadFPV  # noqa: E402

START_HEIGHT = 2.0
DURATION_S = 10.0
MAX_ALLOWED_DRIFT_M = 0.5  # допустиме відхилення висоти за весь прогін


def hover_cmd(vehicle: QuadFPV) -> float:
    """Нормована команда мотора (0..1), за якої сумарна тяга 4 моторів = вазі."""
    weight = vehicle.cfg.mass * 9.81
    thrust_per_motor_needed = weight / 4.0
    # thrust = max_thrust * cmd^2  =>  cmd = sqrt(thrust / max_thrust)
    cmd = np.sqrt(thrust_per_motor_needed / vehicle.cfg.max_thrust_per_motor)
    return float(np.clip(cmd, 0.0, 1.0))


def main() -> int:
    world = PhysicsWorld()
    vehicle = QuadFPV(world)
    vehicle.reset(pos=np.array([0.0, 0.0, START_HEIGHT]))

    motor_cmds = np.full(4, hover_cmd(vehicle))
    print(f"Підібрана hover-команда мотора: {motor_cmds[0]:.4f} (0..1)")

    clock = SimClock(physics_hz=240.0)
    log_every = int(clock.physics_hz)  # раз на секунду
    max_drift = 0.0

    for _ in range(int(DURATION_S * clock.physics_hz)):
        thrusts = np.array(
            [aerodynamics.motor_thrust(c, vehicle.cfg.max_thrust_per_motor) for c in motor_cmds]
        )
        vehicle.apply_motor_commands(thrusts, clock.dt)
        world.step(clock.dt)
        vehicle.advance_time(clock.dt)
        clock.tick()

        state = vehicle.state
        if np.isnan(state.pos).any() or np.isnan(state.vel).any():
            print(f"[ПОМИЛКА] NaN у стані на t={clock.t:.2f}s")
            return 1

        drift = abs(state.pos[2] - START_HEIGHT)
        max_drift = max(max_drift, drift)

        if clock.step_count % log_every == 0:
            print(
                f"t={clock.t:5.2f}s  z={state.pos[2]:6.3f}  vz={state.vel[2]:6.3f}  "
                f"drift={drift:.3f}"
            )

    print(f"Готово. Макс. відхилення висоти за {DURATION_S:.0f}с: {max_drift:.3f} м")
    if max_drift >= MAX_ALLOWED_DRIFT_M:
        print(f"[ПРОВАЛ] Відхилення перевищує поріг {MAX_ALLOWED_DRIFT_M} м")
        return 1
    print("[OK] Завис утримується в межах допуску.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

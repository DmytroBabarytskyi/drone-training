"""Наскрізна оцінка автономного режиму на кількох сценаріях (фаза 7, ROADMAP:
"Оцінка end-to-end на кількох сценаріях; звіт метрик у docs/").

На відміну від ``ml/rl/eval.py`` (фаза 6, оцінює на СИНТЕТИЧНОМУ ``DroneEnv``
з випадковою ціллю), цей скрипт ганяє РЕАЛЬНІ ігрові сценарії
(``scenarios/registry.py``: strike_range/patrol/gate_race) з РЕАЛЬНИМ
``RLAgentSource`` + ``select_autopilot_target`` (``input/rl_agent.py``) — той
самий шлях, що й ``dronesim.app autonomous``, лише headless (без Panda3D/
Engine, лише Bullet-фізика, як ``DroneEnv``) заради швидкості масового прогону.

Політика навчена ЛИШЕ на strike_range-подібній задачі (``ml/rl_strike.yaml``:
випадкова нерухома/повільна ціль поруч) — тому очікувано найкращий результат
на strike_range; patrol (ціль виявляється лише дистанцією) і gate_race (ціль —
центр воріт, потрібен проліт КРІЗЬ них, а не просто наближення) — тест
узагальнення на не бачені під час навчання типи задач, результат може бути
гірший. Це чесно задокументовано у звіті, не приховано.

Запуск: python scripts/eval_autonomous.py --checkpoint runs/rl/ppo_strike/model.zip
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

# КРИТИЧНО (docs/DECISIONS.md, фаза 7): torch (через stable_baselines3) МАЄ
# бути імпортований ДО будь-якого Panda3D-модуля в цьому процесі — нижче
# ``physics.world`` транзитивно тягне panda3d.bullet, тож просто ІМПОРТУВАТИ
# його раніше за torch уже ламає нативні DLL на Windows (той самий привід, що
# й tests/conftest.py). Досить імпортувати тут першим рядком, використовувати
# можна пізніше.
import torch  # noqa: F401,E402

from dronesim.core.sim_loop import SimClock  # noqa: E402
from dronesim.input.rl_agent import RLAgentSource, select_autopilot_target  # noqa: E402
from dronesim.physics.world import PhysicsWorld  # noqa: E402
from dronesim.scenarios.registry import build_scenario  # noqa: E402
from dronesim.vehicles.controllers.position import PosHoldController  # noqa: E402
from dronesim.vehicles.multirotor import Multirotor  # noqa: E402

DECISION_HZ = 20.0  # той самий темп рішень, що й під час навчання (ml/rl_strike.yaml)
PHYSICS_HZ = 240.0
DECISION_SUBSTEPS = int(PHYSICS_HZ / DECISION_HZ)

# Власний ліміт часу для гонки воріт — у самого сценарію немає таймауту
# (він завершується лише після ``laps_to_win`` кіл), тож без цього епізод, де
# агент "застряг", гальмував би оцінку нескінченно.
SCENARIO_TIME_LIMIT_S = {
    "strike_range": 120.0,
    "patrol": 60.0,
    "gate_race": 90.0,
}
START_POS = np.array([0.0, 0.0, 3.0])
START_JITTER_XY_M = 3.0  # розкид старту (фаза 7, доп.): без цього seed нічого не змінює
START_JITTER_Z_M = 1.0


def run_episode(agent: RLAgentSource, scenario_name: str, seed: int, vehicle_config: str) -> dict:
    # ВАЖЛИВО: `StrikeRangeScenario`/`PatrolScenario`/`GateRaceScenario` мають
    # ФІКСОВАНІ позиції цілей/воріт із YAML-конфігу — на відміну від `DroneEnv`
    # (фаза 6), їхній ``reset(seed=...)`` НЕ рандомізує розкладку. Без власного
    # джиттера старту тут різні ``seed`` давали б БАЙТ-В-БАЙТ ІДЕНТИЧНИЙ
    # результат (виявлено емпірично — 20 "різних" епізодів дали один і той
    # самий score/lap_time), тобто "20 епізодів" було б фіктивною статистикою.
    rng = np.random.default_rng(seed)
    start_pos = START_POS + np.array(
        [
            rng.uniform(-START_JITTER_XY_M, START_JITTER_XY_M),
            rng.uniform(-START_JITTER_XY_M, START_JITTER_XY_M),
            rng.uniform(-START_JITTER_Z_M, START_JITTER_Z_M),
        ]
    )

    world = PhysicsWorld()
    vehicle = Multirotor(world, config_name=vehicle_config)
    vehicle.reset(pos=start_pos)
    controller = PosHoldController(vehicle)
    controller.reset(throttle_neutral=0.5)
    scenario = build_scenario(scenario_name, seed=seed)

    clock = SimClock(physics_hz=PHYSICS_HZ)
    dt = clock.dt
    time_limit = SCENARIO_TIME_LIMIT_S[scenario_name]
    done = False
    result = None

    while clock.t < time_limit:
        target = select_autopilot_target(scenario, vehicle.state.pos, clock.t)
        agent.set_context(vehicle.state, target)
        cmd = agent.get_command()

        for _ in range(DECISION_SUBSTEPS):
            thrusts = controller.update(cmd, vehicle.state, dt)
            vehicle.apply_motor_commands(thrusts, dt)
            world.step(dt)
            vehicle.advance_time(dt)
            clock.tick()

        result = scenario.step(vehicle.state)
        if result.done:
            done = True
            break

    return {
        "scenario": scenario_name,
        "seed": seed,
        "done": done,
        "score": result.score if result else 0.0,
        "time_s": clock.t,
        "info": result.info if result else {},
    }


def evaluate(checkpoint: str, vehicle_config: str, episodes: int, seed0: int) -> dict:
    agent = RLAgentSource(checkpoint)
    report: dict = {"checkpoint": checkpoint, "episodes_per_scenario": episodes, "scenarios": {}}

    for scenario_name in ["strike_range", "patrol", "gate_race"]:
        episode_results = [
            run_episode(agent, scenario_name, seed0 + i, vehicle_config) for i in range(episodes)
        ]
        completed = sum(1 for r in episode_results if r["done"])
        mean_score = sum(r["score"] for r in episode_results) / len(episode_results)
        mean_time = sum(r["time_s"] for r in episode_results) / len(episode_results)
        report["scenarios"][scenario_name] = {
            "completion_rate": completed / episodes,
            "mean_score": mean_score,
            "mean_time_s": mean_time,
            "episodes": episode_results,
        }
        print(
            f"{scenario_name:14s}  completion_rate={completed / episodes:.2f}  "
            f"mean_score={mean_score:.2f}  mean_time_s={mean_time:.1f}"
        )

    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", default="runs/rl/ppo_strike/model.zip")
    parser.add_argument("--vehicle-config", default="vehicles/quad_large")
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--seed", type=int, default=2000)
    parser.add_argument("--out", default="docs/phase7_autonomous_eval.json")
    args = parser.parse_args(argv)

    t0 = time.time()
    report = evaluate(args.checkpoint, args.vehicle_config, args.episodes, args.seed)
    report["wall_time_s"] = time.time() - t0

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Звіт записано в {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

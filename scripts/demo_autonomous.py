"""Демо-запуск автономного режиму (фаза 7, критерій приймання ROADMAP).

Тонка обгортка над ``dronesim.app autonomous`` зі зручними дефолтами — щоб
показати/записати демо одним викликом, без запам'ятовування прапорців CLI.
За замовчуванням ВІКОННИЙ (не ``--offscreen``) режим: саме так знімається
gif/mp4 для README. Клавіша ``p`` під час польоту перемикає Manual↔Autonomous
(критерій приймання: перехоплення миттєве).

Запуск: python scripts/demo_autonomous.py
        python scripts/demo_autonomous.py --scenario gate_race
        python scripts/demo_autonomous.py --checkpoint runs/rl/ppo_strike/model.zip
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

DEFAULT_CHECKPOINT = "runs/rl/ppo_strike/model.zip"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scenario", default="strike_range", choices=["strike_range", "patrol", "gate_race"])
    parser.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT)
    parser.add_argument("--vehicle", default="quad_large")
    parser.add_argument("--seed", type=int, default=1)
    args = parser.parse_args(argv)

    checkpoint = Path(args.checkpoint)
    if not checkpoint.exists():
        print(
            f"[ПОМИЛКА] Немає чекпойнта {checkpoint}. Спершу навчи агента:\n"
            f"  python -m dronesim.ml.rl.train --config ml/ppo_strike"
        )
        return 1

    print(
        f"Автономний демо-політ: сценарій={args.scenario}, апарат={args.vehicle}, "
        f"чекпойнт={checkpoint}\n"
        f"Керування: 'p' — перемкнути Manual/Autonomous, Esc — вихід."
    )

    from dronesim.app import main as app_main

    return app_main(
        [
            "autonomous",
            "--scenario", args.scenario,
            "--checkpoint", str(checkpoint),
            "--vehicle", args.vehicle,
            "--seed", str(args.seed),
        ]
    )


if __name__ == "__main__":
    raise SystemExit(main())

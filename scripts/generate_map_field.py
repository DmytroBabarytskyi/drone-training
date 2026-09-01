"""Генератор Карти 1 "Поле" (доп. фаза "графіка 3.0", реальний фідбек: "поле,
посадки, траншеї, окопи, в них танки, артилерія, заборчики").

Пише ДВА YAML (рівень ``field`` наявної системи рівнів, scenarios/registry.py):
- ``assets/scenes/strike_range_field.yaml`` — сцена: посадки (лісосмуги —
  РЯДИ дерев як ФІЗИЧНІ obstacle, апарат реально заплутується між стовбурами),
  траншеї, паркани, пшеничні ділянки (візуальна секція wheat_patches).
- ``configs/scenarios/strike_range_field.yaml`` — правила: цілі (танки/
  артилерія) РОЗМІЩЕНІ БІЛЯ ТРАНШЕЙ (координати узгоджені зі сценою).

Чекнутий у репо генератор замість ручного YAML — 60+ дерев у рядах руками
не підтримати; перегенерація: ``python scripts/generate_map_field.py``.
Запуск: python -m dronesim.app fly --scenario strike_range --level field
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]

# Лісосмуги (посадки): лінія + крок 2.4м між деревами і 2.2м між рядами —
# досить щільно, щоб пролетіти можна було лише обережно (реальний фідбек:
# "дрон в них має заплутуватись"); справжня smuga з 2 рядів, як у полі.
_PLANTATIONS = (
    {"start": (-30.0, 18.0), "end": (30.0, 18.0), "rows": 2, "row_gap": 2.2, "spacing": 2.4},
    {"start": (-35.0, -25.0), "end": (25.0, -25.0), "rows": 2, "row_gap": 2.2, "spacing": 2.4},
)

# Траншейна лінія: 3 сегменти окопів зі "зламом" (як справжня траншея, не
# пряма лінія), у/біля них — цілі (танки/артилерія, конфіг сценарію нижче).
_TRENCHES = (
    {"pos": (18.0, 2.0), "half": (5.0, 1.2, 0.5), "h": 15.0},
    {"pos": (24.0, -6.0), "half": (4.0, 1.2, 0.5), "h": -20.0},
    {"pos": (12.0, 9.0), "half": (3.5, 1.2, 0.5), "h": 40.0},
)

_FENCES = (
    {"pos": (-12.0, 6.0), "half": (5.0, 0.1, 0.55), "h": 0.0},
    {"pos": (-18.0, -8.0), "half": (4.0, 0.1, 0.55), "h": 90.0},
)

_WHEAT_PATCHES = (
    {"center": (-25.0, 30.0), "size": (22.0, 14.0), "count": 1600},
    {"center": (-30.0, -38.0), "size": (18.0, 12.0), "count": 1200},
)

# Цілі: танки/артилерія В/БІЛЯ траншей (узгоджено з _TRENCHES вручну).
_TARGETS = (
    {"pos": (18.0, 3.5, 2.0), "radius": 2.5, "kind": "vehicle"},
    {"pos": (24.0, -7.5, 2.0), "radius": 2.5, "kind": "artillery"},
    {"pos": (11.0, 10.5, 2.0), "radius": 2.5, "kind": "vehicle"},
    {"pos": (30.0, 8.0, 2.0), "radius": 2.5, "kind": "vehicle",
     "moves_to": (36.0, 14.0, 2.0), "period_s": 12.0},
)


def _tree_line(start, end, rows, row_gap, spacing, rng) -> list[dict]:
    """Ряди дерев уздовж лінії: ФІЗИЧНІ obstacle (kind: tree) з невеликим
    джиттером позиції, щоб посадка не виглядала ідеальною сіткою."""
    start_v, end_v = np.array(start), np.array(end)
    direction = end_v - start_v
    length = float(np.linalg.norm(direction))
    direction = direction / length
    normal = np.array([-direction[1], direction[0]])
    n_trees = int(length / spacing) + 1

    entries = []
    for row in range(rows):
        row_offset = (row - (rows - 1) / 2.0) * row_gap
        for i in range(n_trees):
            base = start_v + direction * (i * spacing) + normal * row_offset
            jitter = rng.uniform(-0.5, 0.5, size=2)
            x, y = float(base[0] + jitter[0]), float(base[1] + jitter[1])
            hz = float(rng.uniform(1.3, 1.8))  # висота дерева (half) — різні дерева
            r = float(rng.uniform(0.7, 1.0))
            entries.append({"pos": [round(x, 2), round(y, 2), round(hz, 2)],
                            "half_extents": [round(r, 2), round(r, 2), round(hz, 2)],
                            "kind": "tree"})
    return entries


def main() -> None:
    rng = np.random.default_rng(42)

    obstacles: list[dict] = []
    for p in _PLANTATIONS:
        obstacles += _tree_line(p["start"], p["end"], p["rows"], p["row_gap"], p["spacing"], rng)
    for t in _TRENCHES:
        obstacles.append({"pos": [t["pos"][0], t["pos"][1], t["half"][2]],
                          "half_extents": list(t["half"]), "kind": "trench"})
    for f in _FENCES:
        obstacles.append({"pos": [f["pos"][0], f["pos"][1], f["half"][2]],
                          "half_extents": list(f["half"]), "kind": "fence"})

    scene_lines = [
        "# ЗГЕНЕРОВАНО scripts/generate_map_field.py — НЕ редагуй вручну,",
        "# зміни в генераторі + перегенерація (докладніше в його докстрінгу).",
        "# Карта 1 \"Поле\": посадки (ряди дерев - ФІЗИЧНІ, апарат заплутується),",
        "# траншеї з цілями, паркани, пшеничні ділянки (візуальні).",
        "ground_z: 0.0",
        "obstacles:",
    ]
    for o in obstacles:
        scene_lines.append(
            f"  - {{ pos: {o['pos']}, half_extents: {o['half_extents']}, kind: {o['kind']} }}"
        )
    scene_lines.append("wheat_patches:")
    for w in _WHEAT_PATCHES:
        scene_lines.append(
            f"  - {{ center: [{w['center'][0]}, {w['center'][1]}], "
            f"size: [{w['size'][0]}, {w['size'][1]}], count: {w['count']} }}"
        )
    scene_path = REPO_ROOT / "assets" / "scenes" / "strike_range_field.yaml"
    scene_path.write_text("\n".join(scene_lines) + "\n", encoding="utf-8")

    scenario_lines = [
        "# ЗГЕНЕРОВАНО scripts/generate_map_field.py — НЕ редагуй вручну.",
        "# Рівень \"field\" (Карта 1 \"Поле\"): цілі в/біля траншей (узгоджено зі сценою).",
        "max_aim_angle_deg: 20.0",
        "time_limit_s: 150.0",
        "targets:",
    ]
    for t in _TARGETS:
        extra = ""
        if "moves_to" in t:
            extra = f", moves_to: {list(t['moves_to'])}, period_s: {t['period_s']}"
        scenario_lines.append(
            f"  - {{ pos: {list(t['pos'])}, radius: {t['radius']}, kind: {t['kind']}{extra} }}"
        )
    scenario_path = REPO_ROOT / "configs" / "scenarios" / "strike_range_field.yaml"
    scenario_path.write_text("\n".join(scenario_lines) + "\n", encoding="utf-8")

    print(f"OK: {scene_path.relative_to(REPO_ROOT)} ({len(obstacles)} obstacles)")
    print(f"OK: {scenario_path.relative_to(REPO_ROOT)} ({len(_TARGETS)} targets)")


if __name__ == "__main__":
    main()

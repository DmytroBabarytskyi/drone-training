"""Екран результатів сценарію: текстовий підсумок (час, влучання/кола, промахи).

Чиста функція форматування — не залежить від Panda3D/рендеру. Викликається
``app.py`` наприкінці сценарію (друк у консоль) і може бути показана в HUD
(render/hud.py) тим самим текстом.
"""

from __future__ import annotations

from dronesim.scenarios.base import ScenarioResult


def format_results(scenario_name: str, result: ScenarioResult, elapsed_s: float) -> str:
    """Людський підсумок сценарію за фінальним ``ScenarioResult``."""
    lines = [
        "=== Результати сценарію ===",
        f"Сценарій: {scenario_name}",
        f"Час: {elapsed_s:.1f} с",
        f"Очки: {result.score:.0f}",
        f"Завершено: {'так' if result.done else 'ні'}",
    ]

    info = result.info or {}
    if "laps" in info:
        lines.append(f"Кіл пройдено: {info['laps']}")
        lap_times = info.get("lap_times") or []
        if lap_times:
            best = min(lap_times)
            lines.append(f"Найкраще коло: {best:.2f} с")
    if "hits" in info and "total" in info:
        misses = info["total"] - info["hits"]
        lines.append(f"Влучань: {info['hits']}/{info['total']} (промахів: {misses})")
    if "observed" in info and "total" in info:
        lines.append(f"Помічено цілей: {info['observed']}/{info['total']}")

    return "\n".join(lines)

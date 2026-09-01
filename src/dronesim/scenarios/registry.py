"""Реєстр сценаріїв за іменем — спільний для CLI (``app.py``) і ML
(``ml/data/recorder.py``, фаза 5), щоб не дублювати список у двох місцях.

Ліниві імпорти (``importlib``) — щоб звичайний вільний політ без `--scenario`
не тягнув `scenarios`/`world.targets` без потреби.

РІВНІ (доп. фаза "розмаїття контенту"): один ТИП сценарію (правила гри —
``GateRaceScenario``/``PatrolScenario``/``StrikeRangeScenario``) може мати
кілька РОЗКЛАДОК контенту (``configs/scenarios/{name}_{level}.yaml`` +
``assets/scenes/{name}_{level}.yaml``) — наприклад різні траси для
``gate_race``. ``level=None`` — стара поведінка (єдиний файл
``configs/scenarios/{name}.yaml``, без суфіксу), тому НІЧИЙ наявний виклик
``build_scenario(name, seed)`` без ``level`` не змінює поведінку (важливо:
RL-навчання/оцінка, ``ml/rl/env.py``/``scripts/eval_autonomous.py``,
викликають саме так).
"""

from __future__ import annotations

import importlib

from dronesim.core.config import CONFIGS_DIR

SCENARIO_REGISTRY = {
    "gate_race": ("dronesim.scenarios.gate_race", "GateRaceScenario"),
    "patrol": ("dronesim.scenarios.patrol", "PatrolScenario"),
    "strike_range": ("dronesim.scenarios.strike_range", "StrikeRangeScenario"),
}


def level_file_stem(name: str, level: str | None) -> str:
    """Ім'я файлу (без розширення/каталогу) для сценарію ``name`` + рівень
    ``level``: ``name`` якщо рівень не задано (стара поведінка, один файл),
    інакше ``name_level``. Спільне для ``configs/scenarios/*.yaml`` (тут) і
    ``assets/scenes/*.yaml`` (``app.py`` для декоративної сцени) — назва
    рівня одна на обидва місця, щоб не розходились."""
    return name if level is None else f"{name}_{level}"


def build_scenario(name: str, seed: int | None = None, level: str | None = None):
    """Побудувати й скинути (``reset(seed)``) сценарій за іменем реєстру.

    ``level`` — опційна назва конкретної розкладки контенту (див. докстрінг
    модуля); ``None`` (за замовчуванням) зберігає стару поведінку.
    """
    if name not in SCENARIO_REGISTRY:
        raise ValueError(f"Невідомий сценарій '{name}'. Доступні: {sorted(SCENARIO_REGISTRY)}")
    module_name, class_name = SCENARIO_REGISTRY[name]
    scenario_cls = getattr(importlib.import_module(module_name), class_name)
    scenario = scenario_cls(config_name=f"scenarios/{level_file_stem(name, level)}")
    scenario.reset(seed=seed)
    return scenario


def list_levels(name: str) -> list[str]:
    """Перелічити доступні рівні для типу сценарію ``name`` (``configs/
    scenarios/{name}_*.yaml``, без базового файлу без суфіксу) — для
    GUI-дропдауна (доп. фаза "GUI-лаунчер"). Порожній список — лише
    базова розкладка (``level=None``) доступна."""
    if name not in SCENARIO_REGISTRY:
        raise ValueError(f"Невідомий сценарій '{name}'. Доступні: {sorted(SCENARIO_REGISTRY)}")
    prefix = f"{name}_"
    scenarios_dir = CONFIGS_DIR / "scenarios"
    levels = [
        p.stem[len(prefix) :] for p in scenarios_dir.glob(f"{prefix}*.yaml") if p.stem.startswith(prefix)
    ]
    return sorted(levels)

"""Сканування дискового стану для GUI-лаунчера (доп. фаза "GUI-лаунчер") —
чисті функції без Qt, повністю тестовані headless. GUI-віджети (main_window.py)
лише відображають результат цих функцій, нічого не рахують самі.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from dronesim.core.config import CONFIGS_DIR, REPO_ROOT
from dronesim.scenarios.registry import SCENARIO_REGISTRY, list_levels


def list_vehicles() -> list[str]:
    """Імена конфігів апаратів (``configs/vehicles/*.yaml``, без розширення)."""
    vehicles_dir = CONFIGS_DIR / "vehicles"
    return sorted(p.stem for p in vehicles_dir.glob("*.yaml"))


def list_scenarios() -> list[str]:
    return sorted(SCENARIO_REGISTRY)


def list_scenario_levels(scenario: str) -> list[str]:
    """Рівні сценарію + порожній рядок на позначення базової розкладки
    (``level=None``) — перший елемент, щоб UI показував його за замовчуванням."""
    return ["", *list_levels(scenario)]


@dataclass(frozen=True)
class ModelInfo:
    path: Path
    size_bytes: int
    mtime: float

    @property
    def display_name(self) -> str:
        """Шлях відносно ``runs/`` — коротше й зрозуміліше за абсолютний шлях."""
        try:
            return str(self.path.relative_to(REPO_ROOT))
        except ValueError:
            return str(self.path)


def _scan_models(pattern_root: Path, glob_pattern: str) -> list[ModelInfo]:
    if not pattern_root.exists():
        return []
    return sorted(
        (
            ModelInfo(p, p.stat().st_size, p.stat().st_mtime)
            for p in pattern_root.glob(glob_pattern)
        ),
        key=lambda m: m.mtime,
        reverse=True,
    )


def list_rl_checkpoints() -> list[ModelInfo]:
    """Навчені RL-чекпойнти: ``runs/rl/**/*.zip`` (найновіші першими)."""
    return _scan_models(REPO_ROOT / "runs" / "rl", "**/*.zip")


def list_yolo_weights() -> list[ModelInfo]:
    """Навчені ваги YOLO: ``runs/detect/**/weights/*.pt`` — саме ``weights/``,
    не всю ``runs/detect/**`` (там ще лежать ``val-*`` директорії з артефактами
    оцінки — PNG-графіки/CSV, не ваги моделі)."""
    return _scan_models(REPO_ROOT / "runs" / "detect", "**/weights/*.pt")

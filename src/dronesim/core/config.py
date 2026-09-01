"""Завантаження конфігів через OmegaConf. Єдина точка читання YAML.

Правило проєкту (SKILL.md, правило 3): жодних магічних чисел у коді — усі
параметри живуть у ``configs/*.yaml`` і читаються тут.
"""

from __future__ import annotations

from pathlib import Path

from omegaconf import DictConfig, OmegaConf

# Корінь репозиторію: .../src/dronesim/core/config.py -> вгору 3 рівні
REPO_ROOT = Path(__file__).resolve().parents[3]
CONFIGS_DIR = REPO_ROOT / "configs"
ASSETS_DIR = REPO_ROOT / "assets"


def _load_yaml(base_dir: Path, name: str, overrides: list[str] | None) -> DictConfig:
    path = base_dir / name
    if path.suffix == "":
        path = path.with_suffix(".yaml")
    if not path.exists():
        raise FileNotFoundError(f"Конфіг не знайдено: {path}")

    cfg = OmegaConf.load(path)
    if overrides:
        cfg = OmegaConf.merge(cfg, OmegaConf.from_dotlist(overrides))
    assert isinstance(cfg, DictConfig)
    return cfg


def load_config(name: str, overrides: list[str] | None = None) -> DictConfig:
    """Завантажити конфіг за ім'ям відносно ``configs/`` (з .yaml або без).

    Тюнинговані параметри (маса, PID-гейни, deadzone тощо) — сюди. Ігровий
    контент (сцени/рівні) — ``load_asset(...)`` нижче, ``assets/``.
    overrides — список у стилі OmegaConf, напр. ``["vehicle.mass=0.5"]``.
    """
    return _load_yaml(CONFIGS_DIR, name, overrides)


def load_asset(name: str, overrides: list[str] | None = None) -> DictConfig:
    """Завантажити ігровий контент (сцену/рівень) за ім'ям відносно ``assets/``.

    Розділення з ``load_config``: ``configs/`` — тюнинг движка/апаратів,
    ``assets/`` — вміст сцен/рівнів (ROADMAP.md, фаза 4: ``assets/scenes/*.yaml``).
    """
    return _load_yaml(ASSETS_DIR, name, overrides)

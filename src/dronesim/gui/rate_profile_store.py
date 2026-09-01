"""Читання/запис rate-профілю Acro (``configs/vehicles/*.yaml::rate_profile``)
для GUI-вкладки "Калібрування" (доп. фаза "справжній Liftoff" — користувач
може сам підлаштувати "різкість" керування, не лише редагувати YAML вручну).
Чисті функції над ``OmegaConf`` — без Qt, тестовані headless.
"""

from __future__ import annotations

from omegaconf import OmegaConf

from dronesim.core.config import CONFIGS_DIR

AXES = ("roll", "pitch", "yaw")
PARAMS = ("center_sensitivity_deg_s", "max_rate_deg_s", "expo")


def vehicle_config_path(vehicle: str):
    return CONFIGS_DIR / "vehicles" / f"{vehicle}.yaml"


def has_rate_profile(vehicle: str) -> bool:
    """False для апаратів без Acro rate_profile (напр. ``quad_large`` — літає
    в Angle/Alt-hold/Pos-hold, реальні BF-rates там концептуально не
    застосовуються, докладніше docs/DECISIONS.md)."""
    cfg = OmegaConf.load(vehicle_config_path(vehicle))
    return "rate_profile" in cfg


def load_rate_profile(vehicle: str) -> dict:
    cfg = OmegaConf.load(vehicle_config_path(vehicle))
    if "rate_profile" not in cfg:
        return {}
    return OmegaConf.to_container(cfg.rate_profile)


def save_rate_profile(vehicle: str, profile: dict) -> None:
    """Оновити ЛИШЕ ``rate_profile`` — решта конфігу апарата (маса, PID
    rate-loop тощо) лишається незмінною."""
    path = vehicle_config_path(vehicle)
    cfg = OmegaConf.load(path)
    for axis in AXES:
        if axis not in profile:
            continue
        for param, value in profile[axis].items():
            OmegaConf.update(cfg, f"rate_profile.{axis}.{param}", value)
    OmegaConf.save(cfg, path)

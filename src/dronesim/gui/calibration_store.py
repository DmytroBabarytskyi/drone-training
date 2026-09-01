"""Читання/запис калібрування вводу (``configs/input/*.yaml``) для GUI-вкладки
"Калібрування" (доп. фаза "GUI-лаунчер"). Чисті функції над ``OmegaConf`` —
без Qt, тестовані headless.
"""

from __future__ import annotations

from omegaconf import OmegaConf

from dronesim.core.config import CONFIGS_DIR

INPUT_CONFIG_PATHS = {
    "keyboard": CONFIGS_DIR / "input" / "keyboard.yaml",
    "gamepad": CONFIGS_DIR / "input" / "ps_gamepad.yaml",
}


def load_calibration(input_name: str) -> dict:
    path = INPUT_CONFIG_PATHS[input_name]
    return OmegaConf.to_container(OmegaConf.load(path))


def save_calibration(input_name: str, updates: dict[str, float]) -> None:
    """Оновити ЛИШЕ передані ключі (напр. ``deadzone``/``expo``) — решта
    конфігу (bindings/axes/buttons) лишається незмінною."""
    path = INPUT_CONFIG_PATHS[input_name]
    cfg = OmegaConf.load(path)
    for key, value in updates.items():
        OmegaConf.update(cfg, key, value)
    OmegaConf.save(cfg, path)

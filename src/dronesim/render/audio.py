"""Звук (доп. фаза поліш): гул двигуна прив'язаний до throttle + сигнали
arm/disarm. Файли — процедурно згенеровані (``scripts/generate_audio_assets.py``),
не завантажені ззовні (``assets/audio/``).

Захист від відсутності аудіо-пристрою (межа системи, де перевірка доречна,
SKILL.md правило "перевіряй лише на межах") — headless/тестове середовище
може не мати робочого звукового бекенду; ``AudioController`` тоді просто
нічого не відтворює, а не падає.
"""

from __future__ import annotations

from pathlib import Path

from direct.showbase.ShowBase import ShowBase
from panda3d.core import Filename

from dronesim.core.config import ASSETS_DIR

MIN_VOLUME = 0.15  # гул чути навіть на нульовому газі (двигуни на холостому ході)
MAX_VOLUME = 0.7
MIN_PLAY_RATE = 0.7
MAX_PLAY_RATE = 1.6


class AudioController:
    """Гул двигуна (циклічний, гучність/тон від throttle) + сигнали arm/disarm."""

    def __init__(self, base: ShowBase, audio_dir: Path = ASSETS_DIR / "audio"):
        self._enabled = bool(base.sfxManagerList)
        if not self._enabled:
            return

        # Panda3D очікує Unix-стиль шляхів навіть на Windows — сирий backslash-
        # шлях мовчки НЕ відкривається (є попередження в лозі, але без винятку).
        def _panda_path(name: str) -> str:
            return Filename.fromOsSpecific(str(audio_dir / name)).getFullpath()

        self._engine_hum = base.loader.loadSfx(_panda_path("engine_hum.wav"))
        self._arm_beep = base.loader.loadSfx(_panda_path("arm_beep.wav"))
        self._disarm_beep = base.loader.loadSfx(_panda_path("disarm_beep.wav"))
        self._explosion = base.loader.loadSfx(_panda_path("explosion.wav"))

        if self._engine_hum is not None:
            self._engine_hum.setLoop(True)
            self._engine_hum.setVolume(MIN_VOLUME)
            self._engine_hum.play()

    def update(self, throttle: float, armed: bool) -> None:
        """Викликати щокадру фізики: підлаштувати гул під поточний throttle (0..1)."""
        if not self._enabled or self._engine_hum is None:
            return
        throttle = max(0.0, min(1.0, throttle))
        target_volume = MIN_VOLUME + (MAX_VOLUME - MIN_VOLUME) * throttle if armed else MIN_VOLUME
        self._engine_hum.setVolume(target_volume)
        self._engine_hum.setPlayRate(MIN_PLAY_RATE + (MAX_PLAY_RATE - MIN_PLAY_RATE) * throttle)

    def notify_arm_changed(self, armed: bool) -> None:
        """Викликати РІВНО РАЗ на межі зміни ``cmd.arm`` (не щокадру)."""
        if not self._enabled:
            return
        beep = self._arm_beep if armed else self._disarm_beep
        if beep is not None:
            beep.play()

    def play_explosion(self) -> None:
        """Доп. фаза "реальні моделі" — викликати РІВНО РАЗ на ураження цілі
        (той самий патерн, що й ``notify_arm_changed``)."""
        if not self._enabled or self._explosion is None:
            return
        self._explosion.play()

    def close(self) -> None:
        if not self._enabled:
            return
        self._engine_hum.stop()

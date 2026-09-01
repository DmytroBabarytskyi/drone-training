"""Одноразовий генератор звукових файлів (доп. фаза поліш) — синтезує прості
WAV процедурно (numpy + стдлібний ``wave``), а НЕ завантажує готові аудіо-
ассети ззовні (уникає питань ліцензування, у дусі решти проєкту — усе
процедурне). Результат комітиться в ``assets/audio/`` як звичайні файли;
``render/audio.py`` лише завантажує вже готові WAV, нічого не синтезує в
рантаймі.

Запуск (лише якщо треба перегенерувати): python scripts/generate_audio_assets.py
"""

from __future__ import annotations

import struct
import wave
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = REPO_ROOT / "assets" / "audio"
SAMPLE_RATE = 22050


def _write_wav(path: Path, samples: np.ndarray) -> None:
    samples = np.clip(samples, -1.0, 1.0)
    pcm = (samples * 32767.0).astype(np.int16)
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(SAMPLE_RATE)
        wf.writeframes(struct.pack(f"<{len(pcm)}h", *pcm))


def _engine_hum(duration_s: float = 1.0, fundamental_hz: float = 90.0) -> np.ndarray:
    """Циклічний гул мотора: основна частота + пара гармонік для "гудючої"
    текстури. Ціле число періодів у буфері -> безшовний луп (без клацання на межі)."""
    n_cycles = round(fundamental_hz * duration_s)
    actual_duration = n_cycles / fundamental_hz
    t = np.linspace(0.0, actual_duration, int(SAMPLE_RATE * actual_duration), endpoint=False)
    wave_signal = (
        0.6 * np.sin(2 * np.pi * fundamental_hz * t)
        + 0.25 * np.sin(2 * np.pi * fundamental_hz * 2 * t)
        + 0.15 * np.sin(2 * np.pi * fundamental_hz * 3 * t)
    )
    return wave_signal / np.max(np.abs(wave_signal))


def _beep(freq_start_hz: float, freq_end_hz: float, duration_s: float = 0.15) -> np.ndarray:
    """Короткий тон зі зміною частоти (висхідний=arm, низхідний=disarm) і
    fade-in/out, щоб уникнути клацання на межах."""
    n = int(SAMPLE_RATE * duration_s)
    freq = np.linspace(freq_start_hz, freq_end_hz, n)
    phase = 2 * np.pi * np.cumsum(freq) / SAMPLE_RATE
    signal = np.sin(phase)
    fade = np.ones(n)
    fade_len = max(1, n // 10)
    fade[:fade_len] = np.linspace(0.0, 1.0, fade_len)
    fade[-fade_len:] = np.linspace(1.0, 0.0, fade_len)
    return signal * fade


def _explosion(duration_s: float = 0.8) -> np.ndarray:
    """Вибух (доп. фаза "реальні моделі"): фільтрований шум із швидким
    експоненційним спадом гучності — простіше й дешевше за синтез реального
    вибуху, але впізнавано звучить як "бабах", а не тон/сигнал."""
    n = int(SAMPLE_RATE * duration_s)
    rng = np.random.default_rng(0)
    noise = rng.uniform(-1.0, 1.0, size=n)
    # Проста однополюсна низькочастотна фільтрація (згладжує шум до "гулкого"
    # тембру вибуху, а не білого шуму/статики).
    filtered = np.zeros(n)
    alpha = 0.08
    for i in range(1, n):
        filtered[i] = alpha * noise[i] + (1 - alpha) * filtered[i - 1]
    envelope = np.exp(-np.linspace(0.0, 6.0, n))
    signal = filtered * envelope
    return signal / np.max(np.abs(signal))


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    _write_wav(OUT_DIR / "engine_hum.wav", _engine_hum())
    _write_wav(OUT_DIR / "arm_beep.wav", _beep(600.0, 1200.0))
    _write_wav(OUT_DIR / "disarm_beep.wav", _beep(900.0, 400.0))
    _write_wav(OUT_DIR / "explosion.wav", _explosion())
    print(f"Записано у {OUT_DIR}: engine_hum.wav, arm_beep.wav, disarm_beep.wav, explosion.wav")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

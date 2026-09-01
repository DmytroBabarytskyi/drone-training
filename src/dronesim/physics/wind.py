"""Вітер/пориви як зовнішня сила (фаза 3) + турбулентність/момент (доп. фаза
"реалізм фізики").

Чиста детермінована модель (випадковість лише через переданий ``seed``, як
вимагає SKILL.md правило 4) — без залежності від Panda3D/Bullet: повертає силу
(Н) у світових координатах для заданого моменту часу. Застосування сили до
конкретного апарата — через ``Vehicle.apply_external_force``/
``apply_external_torque`` (vehicles/multirotor.py), яке викликає керуючий код
(сценарій/тест) окремо щокроку фізики, як і мотор-команди — цей модуль лише
РАХУЄ силу/момент, нічого не змінює.

Модель: постійна складова (стала зноса) + синусоїдальний порив уздовж
детермінованого (за seed) горизонтального напрямку (ОБИДВА — стара модель
фази 3, без змін) + ОПЦІЙНА турбулентність (``turbulence_intensity > 0``,
дефолт 0 = вимкнено, стара поведінка): сума кількох гармонік з випадковими
(за seed) частотами/фазами/напрямками — грубе наближення спектра
турбулентності (не справжній Dryden/von Karman, тут спрощено як
багатогармонічна сума, "орієнтовно", докладніше docs/DECISIONS.md), НА
ВІДМІНУ від одиничного синуса — може випадково давати пориви СИЛЬНІШІ за
`turbulence_intensity` (реалістичніше за штучно обмежений одиничний синус,
але саме тому старий ``gust_amplitude`` НЕ замінено, а лишено окремим
детермінованим/обмеженим шаром для сумісності з наявними тестами/сценаріями).
"""

from __future__ import annotations

import numpy as np

_N_TURBULENCE_HARMONICS = 5
_TURBULENCE_FREQ_RANGE_HZ = (0.05, 1.5)  # орієнтовний діапазон "поривчастості" вітру


class Wind:
    """Детермінований генератор сили/моменту вітру: постійна складова +
    гармонічний порив + опційна багатогармонічна турбулентність."""

    def __init__(
        self,
        base_force: tuple[float, float, float] = (0.0, 0.0, 0.0),
        gust_amplitude: float = 0.0,
        gust_frequency_hz: float = 0.2,
        seed: int = 0,
        turbulence_intensity: float = 0.0,
        gust_torque_coeff: float = 0.05,
    ):
        self.base_force = np.array(base_force, dtype=np.float64)
        self.gust_amplitude = gust_amplitude
        self.gust_frequency_hz = gust_frequency_hz
        self.turbulence_intensity = turbulence_intensity

        rng = np.random.default_rng(seed)
        self._phase = float(rng.uniform(0.0, 2.0 * np.pi))
        direction = rng.normal(size=2)  # лише горизонтальна площина (x, y)
        norm = np.linalg.norm(direction)
        direction = direction / norm if norm > 1e-9 else np.array([1.0, 0.0])
        self._gust_direction = np.array([direction[0], direction[1], 0.0])

        # Турбулентність: N гармонік на кожну з 3 осей сили + 3 осей моменту,
        # усі параметри — з ОДНІЄЇ seeded RNG (детермінізм, SKILL.md правило 4).
        low_hz, high_hz = _TURBULENCE_FREQ_RANGE_HZ
        self._force_freqs = rng.uniform(low_hz, high_hz, size=(3, _N_TURBULENCE_HARMONICS))
        self._force_phases = rng.uniform(0.0, 2.0 * np.pi, size=(3, _N_TURBULENCE_HARMONICS))
        self._force_weights = rng.uniform(0.5, 1.0, size=(3, _N_TURBULENCE_HARMONICS))
        self._force_weights /= self._force_weights.sum(axis=1, keepdims=True)  # нормовано -> сума ваг=1 на вісь

        self._torque_freqs = rng.uniform(low_hz, high_hz, size=(3, _N_TURBULENCE_HARMONICS))
        self._torque_phases = rng.uniform(0.0, 2.0 * np.pi, size=(3, _N_TURBULENCE_HARMONICS))
        self._torque_weights = rng.uniform(0.5, 1.0, size=(3, _N_TURBULENCE_HARMONICS))
        self._torque_weights /= self._torque_weights.sum(axis=1, keepdims=True)
        self.gust_torque_coeff = gust_torque_coeff

    def _turbulence_axis(self, t: float, freqs: np.ndarray, phases: np.ndarray, weights: np.ndarray) -> float:
        """Сума ``_N_TURBULENCE_HARMONICS`` синусів (ОДНА вісь), ваги
        нормовані так, щоб амплітуда РІВНЯ turbulence_intensity була типовою
        (не жорсткою межею — сума випадкових фаз рідко дає точний максимум
        усіх гармонік одночасно, тож РЕАЛЬНИЙ пік іноді перевищує
        ``turbulence_intensity`` — це навмисно, реалістичніше за жорстко
        обмежений порив)."""
        return float(np.sum(weights * np.sin(2.0 * np.pi * freqs * t + phases)))

    def force_at(self, t: float) -> np.ndarray:
        """Сила вітру (Н) у світових координатах у момент часу ``t`` симуляції."""
        gust_mag = self.gust_amplitude * np.sin(2.0 * np.pi * self.gust_frequency_hz * t + self._phase)
        force = self.base_force + gust_mag * self._gust_direction

        if self.turbulence_intensity > 0.0:
            turb = np.array(
                [
                    self._turbulence_axis(t, self._force_freqs[i], self._force_phases[i], self._force_weights[i])
                    for i in range(3)
                ]
            )
            force = force + self.turbulence_intensity * turb

        return force

    def torque_at(self, t: float) -> np.ndarray:
        """Стохастичний момент (Н·м) від турбулентності (ЛИШЕ турбулентна
        складова — стала/гармонічна частина вітру НАВМИСНО без моменту,
        docs/DECISIONS.md, фаза 3: сила в центрі мас плеча не створює).
        ``turbulence_intensity == 0`` (дефолт) -> завжди нульовий вектор."""
        if self.turbulence_intensity <= 0.0:
            return np.zeros(3)
        turb = np.array(
            [
                self._turbulence_axis(t, self._torque_freqs[i], self._torque_phases[i], self._torque_weights[i])
                for i in range(3)
            ]
        )
        return self.turbulence_intensity * self.gust_torque_coeff * turb

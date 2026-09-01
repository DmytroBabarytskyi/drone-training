"""Тести звуку (render/audio.py, доп. фаза поліш)."""

from dronesim.render.audio import MAX_VOLUME, MIN_VOLUME, AudioController


def test_audio_controller_enabled_when_device_available(engine):
    audio = AudioController(engine)
    try:
        assert audio._enabled is True
    finally:
        audio.close()


def test_update_scales_volume_with_throttle(engine):
    audio = AudioController(engine)
    try:
        audio.update(throttle=0.0, armed=True)
        vol_low = audio._engine_hum.getVolume()

        audio.update(throttle=1.0, armed=True)
        vol_high = audio._engine_hum.getVolume()

        assert vol_high > vol_low
        assert abs(vol_low - MIN_VOLUME) < 1e-6
        assert abs(vol_high - MAX_VOLUME) < 1e-6
    finally:
        audio.close()


def test_update_uses_min_volume_when_disarmed_regardless_of_throttle(engine):
    audio = AudioController(engine)
    try:
        audio.update(throttle=1.0, armed=False)
        assert abs(audio._engine_hum.getVolume() - MIN_VOLUME) < 1e-6
    finally:
        audio.close()


def test_notify_arm_changed_does_not_raise(engine):
    audio = AudioController(engine)
    try:
        audio.notify_arm_changed(True)
        audio.notify_arm_changed(False)
    finally:
        audio.close()

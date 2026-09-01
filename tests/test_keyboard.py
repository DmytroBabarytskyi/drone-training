"""Тести клавіатурного джерела керування (input/keyboard.py) — фейковий acceptor,
без реального вікна Panda3D. Callback'и, зареєстровані через ``accept``, тут
викликаються вручну для симуляції натискання/відпускання клавіш."""

from dronesim.input.keyboard import KeyboardSource


class _FakeAcceptor:
    """Мінімальна заміна ShowBase: зберігає callback'и, зареєстровані через accept()."""

    def __init__(self):
        self._handlers: dict[str, tuple] = {}

    def accept(self, event: str, callback, extraArgs: list) -> None:
        self._handlers[event] = (callback, extraArgs)

    def press(self, key: str) -> None:
        callback, args = self._handlers[key]
        callback(*args)

    def release(self, key: str) -> None:
        callback, args = self._handlers[f"{key}-up"]
        callback(*args)


def _make_source() -> tuple[KeyboardSource, _FakeAcceptor]:
    acceptor = _FakeAcceptor()
    source = KeyboardSource(acceptor)
    return source, acceptor


def test_neutral_keys_give_zero_sticks_and_idle_throttle():
    source, _ = _make_source()
    source.update(dt=1 / 60)
    cmd = source.get_command()
    assert cmd.roll == 0.0
    assert cmd.pitch == 0.0
    assert cmd.yaw == 0.0
    assert cmd.throttle == 0.0  # старт на холостому ході (безпечний дефолт)


def test_holding_roll_right_ramps_axis_up_over_time():
    source, acceptor = _make_source()
    acceptor.press("d")  # roll_right
    for _ in range(10):
        source.update(dt=1 / 60)
    cmd = source.get_command()
    assert cmd.roll > 0.0


def test_releasing_roll_key_self_centers_back_to_zero():
    source, acceptor = _make_source()
    acceptor.press("d")
    for _ in range(30):
        source.update(dt=1 / 60)
    assert source.get_command().roll > 0.0

    acceptor.release("d")
    for _ in range(120):  # достатньо часу, щоб вісь повернулась до нейтралі
        source.update(dt=1 / 60)
    cmd = source.get_command()
    assert abs(cmd.roll) < 1e-6


def test_throttle_holds_value_after_release_no_self_center():
    source, acceptor = _make_source()
    acceptor.press("shift")  # throttle_up
    for _ in range(30):
        source.update(dt=1 / 60)
    acceptor.release("shift")

    throttle_after_release = source.get_command().throttle
    assert throttle_after_release > 0.0

    for _ in range(60):  # газ НЕ мусить самоцентруватись/спадати без натиснутих клавіш
        source.update(dt=1 / 60)
    assert abs(source.get_command().throttle - throttle_after_release) < 1e-9


def test_throttle_ramps_slower_than_tilt():
    """Газ має ОКРЕМИЙ, повільніший ramp за нахил (реальний фідбек: газ
    "дуже швидко збільшується" при Shift). За однаковий час утримання
    приріст осі газу має бути меншим за приріст осі крену."""
    source_r, acc_r = _make_source()
    source_t, acc_t = _make_source()
    acc_r.press("d")  # roll_right (tilt ramp)
    acc_t.press("shift")  # throttle_up (throttle ramp)
    for _ in range(30):
        source_r.update(dt=1 / 60)
        source_t.update(dt=1 / 60)

    roll_gain = source_r.get_command().roll  # від 0
    throttle_gain = source_t.get_command().throttle - 0.0  # від холостого ходу 0.0
    assert throttle_gain > 0.0
    assert throttle_gain < roll_gain  # газ наростає повільніше


def test_arm_toggle_flips_only_on_press_edge():
    source, acceptor = _make_source()
    acceptor.press("space")
    source.update(dt=1 / 60)
    assert source.get_command().arm is True

    source.update(dt=1 / 60)  # клавіша й далі утримується
    assert source.get_command().arm is True

    acceptor.release("space")
    source.update(dt=1 / 60)
    acceptor.press("space")
    source.update(dt=1 / 60)
    assert source.get_command().arm is False


def test_opposite_keys_held_together_cancel_out():
    source, acceptor = _make_source()
    acceptor.press("a")  # roll_left
    acceptor.press("d")  # roll_right
    for _ in range(30):
        source.update(dt=1 / 60)
    assert source.get_command().roll == 0.0


def test_mode_cycle_key_advances_on_press_edge_not_while_held():
    from dronesim.core.contracts import FlightMode

    source, acceptor = _make_source()
    acceptor.press("tab")
    source.update(dt=1 / 60)
    assert source.get_command().mode is FlightMode.ANGLE

    source.update(dt=1 / 60)  # клавіша й далі утримується -> без змін
    assert source.get_command().mode is FlightMode.ANGLE

    acceptor.release("tab")
    source.update(dt=1 / 60)
    acceptor.press("tab")
    source.update(dt=1 / 60)
    assert source.get_command().mode is FlightMode.ALT_HOLD


def test_set_start_mode_changes_initial_mode_before_first_command():
    """Доп. фаза "справжній Liftoff": app.py вибирає стабільніший стартовий
    режим (Angle), якщо апарат його підтримує — перевіряємо сам механізм."""
    from dronesim.core.contracts import FlightMode

    source, _acceptor = _make_source()
    assert source.get_command().mode is FlightMode.ACRO  # дефолт без виклику

    source.set_start_mode(FlightMode.ANGLE)
    assert source.get_command().mode is FlightMode.ANGLE

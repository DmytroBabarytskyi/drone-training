"""HUD-оверлей: тяга, кути нахилу, швидкість, таймер, статус arm, приціл,
підказки керування та попередження безпеки — поверх FPV-кадру.

Читає лише ``VehicleState``/``ControlCommand`` (контракти core/contracts.py) —
не рахує фізику, лише відображає вже готові дані (правило "рендер не рахує
фізику", SKILL.md). Попередження безпеки так само: рахує їх ``core/safety.py``
(фаза поліш), тут лише мапінг рівня ("warning"/"critical") на колір/текст.

ПОЗИЦІОНУВАННЯ (виправлено після реального фідбеку "немає інтерфейсу"):
телеметрія кріпиться до ``a2dTopLeft`` — вузла, чий локальний origin у
ЛІВОМУ ВЕРХНЬОМУ куті екрана (x росте ВПРАВО від 0, y — ВНИЗ у ВІД'ЄМНІ),
тож координати навмисно малі-додатні по x і малі-від'ємні по y. Попередній
код лишав координати від ``aspect2d`` (центр-origin, (-1.3, 0.9)) під
батьком ``a2dTopLeft`` — текст опинявся ПОЗА екраном (ліворуч-вгорі за
межами), тобто HUD був невидимий. Приціл — на ``aspect2d`` (центр-origin).
"""

from __future__ import annotations

import numpy as np
from direct.gui.OnscreenText import OnscreenText
from direct.showbase.ShowBase import ShowBase
from panda3d.core import LineSegs, TextNode, Vec4

from dronesim.core.contracts import ControlCommand, VehicleState
from dronesim.utils.math3d import quat_to_euler_rad

_RAD2DEG = 180.0 / 3.141592653589793

_WARNING_COLORS = {
    "warning": Vec4(1.0, 0.85, 0.2, 1.0),
    "critical": Vec4(1.0, 0.25, 0.2, 1.0),
}
_SHADOW = Vec4(0.0, 0.0, 0.0, 0.9)  # тінь під текстом — читабельність поверх неба/землі
_CROSSHAIR_COLOR = Vec4(0.35, 1.0, 0.55, 0.9)


def build_keyboard_hint(bindings) -> str:
    """Рядок-легенда клавіш із ``configs/input/keyboard.yaml::bindings`` — ключі
    беруться з конфігу (не хардкодяться), лише семантичні назви дій фіксовані
    контрактом ``KeyboardSource`` (ті самі, що й у yaml). Текст навмисно ASCII
    (див. докстрінг ``HUD`` — дефолтний шрифт Panda3D не має кириличних гліфів)."""
    b = bindings
    return (
        f"{str(b['pitch_forward']).upper()}/{str(b['pitch_back']).upper()}=pitch  "
        f"{str(b['roll_left']).upper()}/{str(b['roll_right']).upper()}=roll  "
        f"{str(b['yaw_left']).upper()}/{str(b['yaw_right']).upper()}=yaw  "
        f"{str(b['throttle_up']).upper()}/{str(b['throttle_down']).upper()}=throttle  "
        f"{str(b['arm_toggle']).upper()}=arm  {str(b['mode_cycle']).upper()}=mode"
    )


def build_takeoff_hint(bindings) -> str:
    """Явна підказка "як злетіти" — навмисно згадує, що газ НЕ повертається сам
    (реалістичний важіль, доп. фаза поліш): без цього утримання throttle_up
    ("рушить як ракета", реальний фідбек користувача) неочевидне для новачка.
    ASCII навмисно (див. докстрінг ``HUD``)."""
    b = bindings
    return (
        f"Press {str(b['arm_toggle']).upper()} to ARM, then gently HOLD "
        f"{str(b['throttle_up']).upper()} to climb - throttle does NOT self-center, "
        f"release the key once you reach the altitude you want."
    )


def _build_crosshair(base: ShowBase) -> object:
    """Приціл по центру екрана (аналог FPV-прицілу): маленький хрест із проміжком
    у центрі + центральна крапка. Малюється через ``LineSegs`` у ``aspect2d``
    (центр-origin), тож завжди строго в центрі кадру незалежно від aspect ratio."""
    segs = LineSegs("crosshair")
    segs.setThickness(2.0)
    segs.setColor(_CROSSHAIR_COLOR)
    gap, arm = 0.018, 0.055
    for sx, sy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
        segs.moveTo(sx * gap, 0, sy * gap)
        segs.drawTo(sx * arm, 0, sy * arm)
    node = base.aspect2d.attachNewNode(segs.create())
    # Центральна крапка
    dot = LineSegs("crosshair_dot")
    dot.setThickness(3.0)
    dot.setColor(_CROSSHAIR_COLOR)
    dot.moveTo(0, 0, 0)
    dot.drawTo(0.001, 0, 0)
    node.attachNewNode(dot.create())
    return node


class HUD:
    """Текстовий+графічний оверлей поверх вікна (не FPV-буфера ML).

    Показники РОЗПОДІЛЕНІ по краях екрана (як у справжньому FPV-HUD, реальний
    фідбек: "всі показники в 1 місці зліва зверху"): режим/arm — ліворуч
    угорі, таймер/рахунок — праворуч угорі, висота й швидкість — праворуч по
    центру, газ — ліворуч унизу, кути — праворуч унизу, попередження — угорі
    по центру, підказка — унизу по центру. Кожен вузол — на своєму 2D-якорі
    (``a2dTopLeft`` тощо), тож тримається краю незалежно від aspect ratio.
    """

    def __init__(self, base: ShowBase, controls_hint: str = ""):
        self._controls_hint = controls_hint

        def _anchor(name: str):
            return getattr(base, name, base.aspect2d)

        def _text(anchor, pos, scale, align, fg=Vec4(1, 1, 1, 1)):
            return OnscreenText(
                text="", pos=pos, scale=scale, fg=fg, shadow=_SHADOW,
                align=align, mayChange=True, parent=anchor,
            )

        # Режим + arm — ліворуч угорі.
        self._mode_text = _text(_anchor("a2dTopLeft"), (0.06, -0.12), 0.06, TextNode.ALeft)
        # Таймер + рахунок сценарію — праворуч угорі.
        self._timer_text = _text(_anchor("a2dTopRight"), (-0.06, -0.12), 0.05, TextNode.ARight)
        # Висота — праворуч, вище центру.
        self._alt_text = _text(_anchor("a2dRightCenter"), (-0.06, 0.14), 0.06, TextNode.ARight)
        # Швидкість — праворуч, нижче центру.
        self._speed_text = _text(_anchor("a2dRightCenter"), (-0.06, -0.06), 0.05, TextNode.ARight)
        # Газ — ліворуч унизу.
        self._throttle_text = _text(_anchor("a2dBottomLeft"), (0.06, 0.10), 0.055, TextNode.ALeft)
        # Кути (roll/pitch/yaw) — праворуч унизу.
        self._attitude_text = _text(_anchor("a2dBottomRight"), (-0.06, 0.10), 0.045, TextNode.ARight)
        # Попередження безпеки — угорі по центру.
        self._warning_text = _text(_anchor("a2dTopCenter"), (0.0, -0.22), 0.07, TextNode.ACenter)
        # Підказка керування (лише роззброєний) — унизу по центру.
        self._hint_text = _text(
            _anchor("a2dBottomCenter"), (0.0, 0.12), 0.045, TextNode.ACenter, fg=Vec4(0.9, 0.9, 0.9, 1.0)
        )
        self._crosshair = _build_crosshair(base)

    def update(
        self,
        state: VehicleState,
        cmd: ControlCommand,
        active_mode: str = "",
        warning_text: str | None = None,
        warning_level: str = "ok",
    ) -> None:
        # Текст навмисно ASCII: дефолтний шрифт Panda3D не має гліфів для
        # деяких символів (без цього кожен кадр валив попередження
        # "No definition ... for character").
        roll_rad, pitch_rad, yaw_rad = quat_to_euler_rad(state.quat)
        roll, pitch, yaw = roll_rad * _RAD2DEG, pitch_rad * _RAD2DEG, yaw_rad * _RAD2DEG
        speed_h = float(np.linalg.norm(state.vel[:2]))
        speed_v = float(state.vel[2])

        armed = "ARMED" if cmd.arm else "disarmed"
        mode_str = f"  {active_mode}" if active_mode else ""
        self._mode_text.setText(f"{armed}{mode_str}")
        self._mode_text.setFg(Vec4(0.4, 1.0, 0.4, 1.0) if cmd.arm else Vec4(1.0, 0.75, 0.3, 1.0))

        self._timer_text.setText(f"t={state.t:6.2f}s")
        self._alt_text.setText(f"ALT {state.pos[2]:5.1f} m")
        self._speed_text.setText(
            f"SPD {speed_h:5.1f} m/s (h)\n{speed_v:+5.1f} m/s (v)"
        )
        self._throttle_text.setText(f"THR {cmd.throttle * 100:4.0f}%")
        self._attitude_text.setText(f"roll {roll:5.0f}  pitch {pitch:5.0f}  yaw {yaw:5.0f}")

        self._hint_text.setText(self._controls_hint if not cmd.arm else "")

        if warning_text and warning_level in _WARNING_COLORS:
            self._warning_text.setText(warning_text)
            self._warning_text.setFg(_WARNING_COLORS[warning_level])
        else:
            self._warning_text.setText("")

    def close(self) -> None:
        for node in (
            self._mode_text, self._timer_text, self._alt_text, self._speed_text,
            self._throttle_text, self._attitude_text, self._warning_text, self._hint_text,
        ):
            node.destroy()
        self._crosshair.removeNode()

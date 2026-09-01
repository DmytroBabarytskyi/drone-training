"""Ігровий рушій на Panda3D: вікно, PBR-освітлення, фіксований крок фізики.

Єдине місце, де живе ``ShowBase`` — решта коду рушія отримує вже готовий
``Engine`` (через ``base``/``render``/``taskMgr`` після ``super().__init__()``),
а не створює власне вікно. Фізика крокує через Panda3D task-менеджер за
фіксованим ``dt`` (акумулятор кадрового часу) — рендер лишається на змінному
FPS вікна, тож фізика й рендер розв'язані (ARCHITECTURE.md).

``offscreen=True`` вмикає безвіконний режим (``window-type offscreen``): є
робочий графічний контекст (GSG) для рендеру в текстуру, але немає видимого
вікна ОС — придатно для тестів і headless-збору ML-даних (фаза 5), на відміну
від ``window-type none``, який взагалі не створює контексту рендеру.
"""

from __future__ import annotations

from collections.abc import Callable

import simplepbr
from direct.showbase.ShowBase import ShowBase
from panda3d.core import AmbientLight, ClockObject, DirectionalLight, Vec4, WindowProperties, loadPrcFileData

from dronesim.core.sim_loop import SimClock

_MAX_PHYSICS_STEPS_PER_FRAME = 8  # запобіжник від "spiral of death" при просіданні FPS


class Engine(ShowBase):
    """Застосунок Panda3D: вікно + сцена + фіксований крок фізики поверх task-менеджера."""

    def __init__(
        self,
        physics_hz: float = 240.0,
        window_title: str = "dronesim",
        offscreen: bool = False,
        enable_shadows: bool = True,
        shadow_buffer_size: int = 1024,
        shadow_extent_m: float = 150.0,
    ):
        # Звук УВІМКНЕНО (доп. фаза поліш) — Panda3D підхоплює OpenAL за
        # замовчуванням, перевірено емпірично (навіть у offscreen), тож немає
        # причин форсувати "audio-library-name null" (попереднє рішення фази 2).
        if offscreen:
            loadPrcFileData("", "window-type offscreen")

        super().__init__()
        # КРИТИЧНО (доп. фаза "графіка 3.0", знайдено через реальний скріншот
        # користувача "весь екран жовтий"): у ВІКОННОМУ режимі ShowBase за
        # замовчуванням вмикає trackball-керування мишею, яке ЩОКАДРУ
        # перезаписує трансформ ``base.camera`` — усі наші
        # ``attach_main_camera``/``attach_chase_camera`` позиціонування
        # МОВЧКИ ігнорувалися, камера сиділа в ЦЕНТРІ корпусу апарата,
        # дивлячись уздовж +Y (не +X носа). В offscreen-режимі миші немає ->
        # trackball не створюється -> усі headless-тести/скріншоти виглядали
        # правильно, а реальна гра — ні (камера всередині жовтого корпусу;
        # раніше це маскувалось near=1.0, який обрізав і корпус, і землю —
        # звідси й скарга "камера бачить крізь землю"). ``disableMouse()``
        # вимикає trackball; керування апаратом — клавіатура/геймпад, миша
        # для камери не потрібна.
        self.disableMouse()
        self.pbr_pipeline = simplepbr.init(enable_shadows=enable_shadows)  # PBR-шейдери/освітлення/тіні

        if not offscreen:
            props = WindowProperties()
            props.setTitle(window_title)
            self.win.requestProperties(props)

        self._setup_lighting(enable_shadows, shadow_buffer_size, shadow_extent_m)

        self.clock = SimClock(physics_hz=physics_hz)
        self._accumulator = 0.0
        self._physics_callback: Callable[[float], None] | None = None
        self._global_clock = ClockObject.getGlobalClock()
        self.taskMgr.add(self._physics_task, "dronesim-physics", sort=10)

    def _setup_lighting(
        self, enable_shadows: bool, shadow_buffer_size: int, shadow_extent_m: float
    ) -> None:
        sun = DirectionalLight("sun")
        # Яскравіше сонце + слабший ambient (доп. фаза "графіка 3.0", реальний
        # фідбек: "не відрізниш, де темніше, куди падає сонце" — попереднє
        # співвідношення сонце:ambient (~1.0:0.27, майже 4:1) давало занадто
        # слабкий контраст після PBR-тонмаппінгу simplepbr; освітлена й
        # затінена грані виглядали майже однаково). Нове ~2.2:0.16 (~14:1) —
        # чіткіше видно, з якого боку сонце, залишаючи затінені грані
        # видимими (не провалюються в чорноту).
        sun.setColor(Vec4(2.2, 2.1, 1.9, 1.0))
        if enable_shadows:
            # Ортографічна лінза сонця має накривати все ігрове поле
            # (configs/safety.yaml::bounds_radius_hard_m ~120м) — інакше об'єкти
            # за межею лінзи рендеряться БЕЗ тіні, що виглядає як баг.
            sun.setShadowCaster(True, shadow_buffer_size, shadow_buffer_size)
            lens = sun.getLens()
            lens.setFilmSize(shadow_extent_m, shadow_extent_m)
            lens.setNearFar(1.0, shadow_extent_m * 2.0)
        sun_np = self.render.attachNewNode(sun)
        sun_np.setHpr(45, -60, 0)
        if enable_shadows:
            # КРИТИЧНО (виявлено емпірично): для ``DirectionalLight`` напрямок
            # світла задає ЛИШЕ ``setHpr`` (позиція не впливає на освітлення
            # як таке), АЛЕ ортографічна камера тіні прив'язана до ФАКТИЧНОЇ
            # позиції вузла світла — без явного ``setPos`` вузол лишається в
            # (0,0,0), тіньова камера опиняється всередині/на рівні сцени, і
            # near/far-діапазон обрізає геометрію — тіні мовчки НЕ рендеряться
            # (жодної помилки, просто відсутні). Зсув "назад" уздовж власної
            # орієнтації розташовує камеру тіні над і позаду сцени коректно.
            sun_np.setPos(sun_np, 0, -shadow_extent_m, 0)
        self.render.setLight(sun_np)

        ambient = AmbientLight("ambient")
        ambient.setColor(Vec4(0.16, 0.17, 0.20, 1.0))
        ambient_np = self.render.attachNewNode(ambient)
        self.render.setLight(ambient_np)

    def set_physics_callback(self, callback: Callable[[float], None]) -> None:
        """Викликається щокроку фізики з фіксованим ``dt`` (=``SimClock.physics_hz``)."""
        self._physics_callback = callback

    def _physics_task(self, task):
        self._accumulator += self._global_clock.getDt()
        dt = self.clock.dt
        steps = 0
        while self._accumulator >= dt and steps < _MAX_PHYSICS_STEPS_PER_FRAME:
            if self._physics_callback is not None:
                self._physics_callback(dt)
            self.clock.tick()
            self._accumulator -= dt
            steps += 1
        return task.cont

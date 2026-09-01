"""CLI-точка входу. Диспетчеризує підкоманди симулятора.

Реалізовано ``sim-empty`` (фаза 0), ``fly`` (фаза 2, опційно з автопілотом —
фаза 7) і ``autonomous`` (фаза 7: проходить сценарій під навченим RL-агентом,
людина може перехопити керування клавішею ``p``).
"""

from __future__ import annotations

import argparse
import math
import sys

import numpy as np

from dronesim.core.sim_loop import SimClock, SimLoop
from dronesim.scenarios.registry import SCENARIO_REGISTRY


def _cmd_sim_empty(args: argparse.Namespace) -> int:
    """Фаза 0: крутить порожній цикл — перевірка годинника і чистого виходу."""
    loop = SimLoop(physics_hz=args.hz)

    def on_step(clock: SimClock) -> None:
        if clock.step_count % args.hz == 0:
            print(f"t={clock.t:6.2f}s  step={clock.step_count}")

    loop.run(on_step, steps=args.steps)
    print(f"Готово: {loop.clock.step_count} кроків, t={loop.clock.t:.2f}s")
    return 0


def _build_controllers(vehicle) -> dict:
    """Зібрати доступні контролери за тим, які секції Є в конфізі апарата.

    fpv_5inch має лише ``pid_rate``/``max_rates`` -> тільки Acro. quad_large
    додатково має ``pid_angle``/``pid_altitude``/``pid_position`` -> Angle/
    Alt-hold/Pos-hold теж доступні. Acro є завжди (запасний режим, fallback
    у physics_step, якщо стік перемкнув на непідтримуваний цим апаратом режим).
    """
    from dronesim.core.contracts import FlightMode
    from dronesim.vehicles.controllers.acro import AcroController
    from dronesim.vehicles.controllers.angle import AngleController
    from dronesim.vehicles.controllers.position import AltHoldController, PosHoldController

    cfg = vehicle.cfg
    controllers = {FlightMode.ACRO: AcroController(vehicle)}
    if "pid_angle" in cfg and "max_angle_deg" in cfg:
        controllers[FlightMode.ANGLE] = AngleController(vehicle)
        if "pid_altitude" in cfg:
            controllers[FlightMode.ALT_HOLD] = AltHoldController(vehicle)
            if "pid_position" in cfg:
                controllers[FlightMode.POS_HOLD] = PosHoldController(vehicle)
    return controllers


def _require_pos_hold(controllers: dict, vehicle_name: str) -> None:
    """Автопілот (фаза 7) видає команди в режимі Pos-hold (той самий, на якому
    навчався ``DroneEnv``, фаза 6) — без Pos-hold його дії безглузді (fallback
    на Acro інтерпретував би їх як rate-команди). Явна помилка краще за
    мовчазну неправильну поведінку (SKILL.md)."""
    from dronesim.core.contracts import FlightMode

    if FlightMode.POS_HOLD not in controllers:
        raise ValueError(
            f"Апарат '{vehicle_name}' не підтримує Pos-hold (немає pid_position у конфізі) — "
            "автопілот (--autopilot/--checkpoint) вимагає апарат з Pos-hold, напр. quad_large."
        )


def _build_scenario(scenario_name: str, seed: int | None, level: str | None = None):
    """Побудувати сценарій за спільним реєстром (scenarios/registry.py)."""
    from dronesim.scenarios.registry import build_scenario

    return build_scenario(scenario_name, seed, level=level)


_TARGET_KIND_BUILDERS = {}  # заповнюється лінивим імпортом у _sync_target_markers


def _sync_target_markers(
    engine, scenario, marker_by_id: dict, t: float, world=None, effects_cfg=None, active_effects=None, audio=None
) -> None:
    """Створити/оновити маркери цілей сценарію (фаза 4; доп. фаза "реальні
    моделі" — техніка/артилерія + ефекти знищення).

    Цілі можуть з'являтися з часом (patrol) або рухатись (moves_to) — тому це
    синхронізація "на кожен кадр", а не одноразова побудова: маркер для нової
    цілі створюється при першій появі в ``scenario.targets``, позиція рухомих
    цілей оновлюється через уже обчислену ``Target.position_at(t)`` (render
    нічого не рахує, лише читає/малює — SKILL.md правило 3.6).

    ``target.kind == "marker"`` (patrol — розвідка, не "ураження"):
    bullseye-картка, уражені цілі перефарбовуються в зелений (стара
    поведінка). ``kind in {"vehicle", "artillery"}`` (strike_range): модель
    ЗНИЩУЄТЬСЯ (видаляється) при ураженні, і РІВНО РАЗ (перевірка
    ``marker.isEmpty()`` — Panda3D-native ознака "вже видалено") тригериться
    ``ExplosionEffect`` + звук, замість перефарбовки.
    """
    if not _TARGET_KIND_BUILDERS:
        from dronesim.render.models import build_artillery, build_vehicle

        _TARGET_KIND_BUILDERS["vehicle"] = build_vehicle
        _TARGET_KIND_BUILDERS["artillery"] = build_artillery

    from dronesim.render.visuals import add_target_marker_visual, set_target_marker_hit

    for target in getattr(scenario, "targets", ()):
        builder = _TARGET_KIND_BUILDERS.get(target.kind)
        marker = marker_by_id.get(target.target_id)

        if marker is None:
            if builder is not None:
                marker = builder(engine.render)
            else:
                marker = add_target_marker_visual(engine.render, tuple(target.base_pos), size=0.8)
            marker_by_id[target.target_id] = marker

        if builder is None:
            marker.setPos(*target.position_at(t))
            set_target_marker_hit(marker, target.hit)
            continue

        # Техніка/артилерія ЗАВЖДИ СТОЇТЬ НА ЗЕМЛІ (реальний фідбек "танк літає
        # у повітрі" — і в рівнях теж!): модель будується з базою в локальному
        # z=0, тож ставимо її на (x, y, 0) незалежно від висоти хіт-центру.
        # Хіт-логіка й далі працює з повним ``position_at`` (хіт-сфера навколо
        # ``center_z``) — цілишся по видимому наземному силуету, влучаєш,
        # пролітаючи крізь сферу над ним. Раніше модель ставили в низ сфери
        # (max(0, center-radius)) — для default (z3/r3) це 0, але рівні з
        # високими цілями й малим радіусом (sniper z6/r1) лишали танк у повітрі.
        pos = target.position_at(t)
        model_pos = (float(pos[0]), float(pos[1]), 0.0)

        if marker.isEmpty():
            continue  # вже знищено цим ходом раніше — нічого робити
        if not target.hit:
            marker.setPos(*model_pos)
            continue

        marker.removeNode()
        if world is not None and effects_cfg is not None and active_effects is not None:
            from dronesim.render.effects import ExplosionEffect

            # Вибух — біля наземного силуету (трохи над землею), не на хіт-центрі.
            burst_pos = np.array([pos[0], pos[1], 0.6])
            active_effects.append(ExplosionEffect(engine.render, world, burst_pos, effects_cfg))
        if audio is not None:
            audio.play_explosion()


def _finish_scenario(scenario_name: str, scenario, result, elapsed_s: float) -> None:
    """Роздрукувати результати (ui/results.py) і скинути лог подій у файл
    (критерій приймання фази 4: "лог подій пишеться у файл").
    """
    import time

    from dronesim.core.config import REPO_ROOT
    from dronesim.scenarios.event_log import write_event_log
    from dronesim.ui.results import format_results

    print(format_results(scenario_name, result, elapsed_s))

    log_path = REPO_ROOT / "logs" / f"{scenario_name}_{int(time.time())}.jsonl"
    write_event_log(scenario.events, log_path)
    print(f"Лог подій: {log_path}")


def build_fly_session(args: argparse.Namespace) -> dict:
    """Зібрати всі компоненти ручного польоту (фаза 2-4), не запускаючи головний цикл.

    Відокремлено від ``_cmd_fly`` навмисно: ``Engine.run()`` блокує до закриття
    вікна, тож для headless-тестів/смоук-перевірок конструюємо сесію тут і самі
    крутимо ``engine.taskMgr.step()`` потрібну кількість разів (tests/test_app_fly.py).
    """
    detector = None
    if getattr(args, "detector", None):
        # КРИТИЧНО (docs/DECISIONS.md, фаза 5): torch/ultralytics МАЄ бути
        # імпортований ДО Panda3D (Engine нижче) в одному процесі — інакше на
        # Windows зіштовхуються нативні DLL. Порядок імпорту, не використання.
        from dronesim.ml.detection.detector import Detector

        detector = Detector(args.detector, conf_threshold=getattr(args, "detector_conf", 0.25))

    autopilot_source = None
    select_autopilot_target = None
    if getattr(args, "autopilot", None):
        # Той самий критичний порядок, що й для Detector вище (torch ДО Panda3D).
        from dronesim.input.rl_agent import RLAgentSource, select_autopilot_target

        autopilot_source = RLAgentSource(args.autopilot)

    from dronesim.core.config import load_config
    from dronesim.core.contracts import FLIGHT_MODE_CYCLE, FlightMode
    from dronesim.core.safety import evaluate_flight_safety, load_safety_config
    from dronesim.physics.wind import Wind
    from dronesim.physics.world import PhysicsWorld
    from dronesim.render.audio import AudioController
    from dronesim.render.camera import FPVCamera, attach_chase_camera, attach_main_camera, update_chase_camera
    from dronesim.render.engine import Engine
    from dronesim.render.hud import HUD, build_keyboard_hint, build_takeoff_hint
    from dronesim.render.models import (
        _terrain_height,
        _voronoi_grouped_regions,
        attach_scenery_visual,
        build_fence_line,
        build_terrain,
        scatter_ground_cover,
        scatter_tree_line,
        scatter_wheat_patch,
    )
    from dronesim.render.propellers import PropellerRig
    from dronesim.render.godrays import GodRays
    from dronesim.render.sky import add_clouds, add_fog, add_sky
    from dronesim.render.visuals import add_box_visual, add_gate_marker_visual
    from dronesim.utils.math3d import quat_to_euler_rad
    from dronesim.vehicles.multirotor import Multirotor
    from dronesim.world.scene import build_basic_scene, build_scene_from_assets

    engine = Engine(
        window_title=f"dronesim — {args.vehicle}", offscreen=getattr(args, "offscreen", False)
    )

    world = PhysicsWorld()
    level = getattr(args, "level", None)
    scenario = (
        _build_scenario(args.scenario, getattr(args, "seed", None), level=level) if args.scenario else None
    )

    scene_decor: dict = {"wheat_patches": []}
    if scenario is not None:
        from dronesim.scenarios.registry import level_file_stem
        from dronesim.world.scene import load_scene_decor

        scene_name = level_file_stem(args.scenario, level)
        ground_np, obstacles = build_scene_from_assets(world, scene_name, parent=engine.render)
        scene_decor = load_scene_decor(scene_name)
    else:
        ground_np, obstacles = build_basic_scene(world, parent=engine.render)

    # РОБОЧА зона (де pos.z == ground_z — обов'язковий інваріант розміщення
    # obstacle/target/gate) рахується ДИНАМІЧНО з фактичної сцени, а не
    # захардкожена — щоб рельєф (нижче) ГАРАНТОВАНО не піднімав землю під
    # жодною перешкодою/ціллю/воротами на жодній карті (і майбутніх теж).
    _extents = [4.0]  # мінімум — точка старту
    for obstacle_np, half_extents, _kind in obstacles:
        _extents.append(math.hypot(obstacle_np.getX(), obstacle_np.getY()) + max(half_extents[0], half_extents[1]))
    for target in getattr(scenario, "targets", ()):
        _extents.append(float(np.hypot(target.base_pos[0], target.base_pos[1])) + target.radius)
        if target.moves_to is not None:
            _extents.append(float(np.hypot(target.moves_to[0], target.moves_to[1])) + target.radius)
    for gate in getattr(scenario, "gates", ()):
        _extents.append(float(np.hypot(gate.center[0], gate.center[1])) + gate.radius)
    flat_radius = min(60.0, max(30.0, max(_extents) + 6.0))
    # Радіус значно більший за попередню смугу (+35) — реальний фідбек:
    # "заповнюється лише коло, за його межами просто пластикова текстура" —
    # рельєф/поля тепер розкидані по значно ширшій площі, не лише вузькому
    # кільці одразу за робочою зоною.
    terrain_outer_radius = flat_radius + 110.0
    terrain_seed = getattr(args, "seed", 0) or 0
    # Радіус трав'яного килима = радіус рельєфу/полів, з розумною стелею
    # (реальний фідбек: "не роби такий великий periметр в 80м — уся площина,
    # що бачить око, має бути в траві й з природним релєфом"). Раніше
    # трав'яний килим (80м) БУВ МЕНШИЙ за зону рельєфу/полів
    # (``terrain_outer_radius``, до ~170м) — усе за 80м мало колір/рельєф
    # землі, але без 3D-стебел трави, що читалось як штучна "межа". Стеля
    # 130м (не безумовно ``terrain_outer_radius``) — знайдено профілюванням:
    # повний радіус для сцен із великим ``flat_radius`` (напр. patrol, до
    # ~170м) дає забагато стебел навіть із density-таpером нижче.
    grass_radius = min(terrain_outer_radius, 130.0)

    # Дорога (kind=road) — без трави на асфальті (реальний фідбек: "на дорозі
    # не росте трава").
    road_rects = [
        (obstacle_np.getX(), obstacle_np.getY(), half_extents[0], half_extents[1])
        for obstacle_np, half_extents, kind in obstacles
        if kind == "road"
    ]

    # Реальний фідбек: "робимо поле і перевіряємо поруч клітинки, якщо теж
    # рівно — додаємо туди поле; де рельєф піднімається/опускається різко —
    # не садять поле" + "поле посаджене рядочками комбайном" + "заборчик МОЖЕ
    # бути вздовж поля" + "посадки — багато дерев лінією вздовж поля".
    # ``_voronoi_grouped_regions(kind="flat")`` дає СУЦІЛЬНІ зв'язні рівні
    # ділянки як обернені прямокутники (центр/довжина/ширина/кут головної осі).
    # Реальний фідбек (наступна ітерація): "в межах кола — все рівно і без
    # полів, за межами кола — немає трави". Причина: дерева/паркан (``kind``
    # tree/fence) — ДРІБНІ декоративні перешкоди, розсіяні по всій робочій
    # зоні (~10-12 штук); при повному блокуванні полів БУДЬ-яким перетином з
    # НИМИ жодне поле (довжиною до 130м) не могло влізти близько до старту —
    # завжди перетинало щось. У реальності поле МОЖЕ мати поодиноке дерево чи
    # межувати з парканом (це й природно — лісосмуга/дерево на краю ниви) —
    # тому в "тверде" блокування йдуть лише ті перешкоди, куди пшениця
    # СПРАВДІ не може рости: точка старту (апарат) і цілі/ворота (ігрова
    # логіка). Дерева/паркан просто трохи "потонуть" у пшениці на межі поля —
    # прийнятно й навіть природно.
    _obstacle_pts = [(0.0, 0.0)]  # точка старту — поле не має її накривати
    _obstacle_pts += [(float(t.base_pos[0]), float(t.base_pos[1])) for t in getattr(scenario, "targets", ())]
    _obstacle_pts += [(float(g.center[0]), float(g.center[1])) for g in getattr(scenario, "gates", ())]

    def _region_is_clear(region: dict) -> bool:
        cx, cy = region["center"]
        ang = math.radians(region["angle_deg"])
        ca, sa = math.cos(-ang), math.sin(-ang)
        half_l, half_w = region["length"] / 2.0 + 3.0, region["width"] / 2.0 + 3.0
        for ox, oy in _obstacle_pts:
            lx = (ox - cx) * ca - (oy - cy) * sa
            ly = (ox - cx) * sa + (oy - cy) * ca
            if abs(lx) < half_l and abs(ly) < half_w:
                return False
        return True

    def _region_is_flat_enough(region: dict) -> bool:
        # ``_voronoi_grouped_regions`` — топологічне наближення (BFS по
        # Delaunay-сусідству): межова клітинка іноді опиняється неподалік
        # хребта, що не був її прямим сусідом. Друга, геометрична перевірка
        # тут — пряме семплювання ``_terrain_height`` по площі регіону —
        # гарантує, що поле/посадка ніколи не лягає на реальний схил,
        # незалежно від точності топологічного наближення вище.
        cx, cy = region["center"]
        rad = math.radians(region["angle_deg"])
        along = np.array([math.cos(rad), math.sin(rad)])
        across = np.array([-math.sin(rad), math.cos(rad)])
        sl, sw = np.meshgrid(np.linspace(-0.4, 0.4, 5), np.linspace(-0.4, 0.4, 5))
        pts = (
            np.array([cx, cy])[None, :]
            + along[None, :] * (sl.ravel() * region["length"])[:, None]
            + across[None, :] * (sw.ravel() * region["width"])[:, None]
        )
        h = _terrain_height(pts[:, 0], pts[:, 1], flat_radius, terrain_outer_radius, 4.5, terrain_seed)
        return bool(np.mean(np.abs(h)) < 0.4)

    # max_cells=5 (було 8) — компактніші поля з'являються БЛИЖЧЕ до старту
    # частіше (велике поле 130м частіше зачіпає ціль/ворота біля центру
    # просто через свій розмір, незалежно від margin).
    field_regions = _voronoi_grouped_regions(flat_radius, terrain_outer_radius, terrain_seed, kind="flat", max_cells=5)
    field_regions = [r for r in field_regions if _region_is_clear(r) and _region_is_flat_enough(r)]
    field_regions.sort(key=lambda r: math.hypot(*r["center"]))

    # ГАРАНТІЯ: якщо природна Вороного-розкладка не дала жодного поля в
    # межах трав'яного килима (``grass_radius``) — випадковість насінин це
    # іноді дає — явно пошукати невелику вільну рівну ділянку близько до
    # старту (той самий підхід, що раніше ``_terrain_flat_spot``, але з
    # тією ж перевіркою чистоти/рівності, що й звичайні поля).
    if not field_regions or math.hypot(*field_regions[0]["center"]) > grass_radius - 10.0:
        near_rng = np.random.default_rng(terrain_seed + 777)
        forced_size = (20.0, 15.0)
        for _ in range(300):
            cr = near_rng.uniform(flat_radius * 0.35, grass_radius - 10.0)
            ctheta = near_rng.uniform(0.0, 2 * math.pi)
            candidate = {
                "center": (cr * math.cos(ctheta), cr * math.sin(ctheta)),
                "angle_deg": float(near_rng.uniform(0.0, 180.0)),
                "length": forced_size[0],
                "width": forced_size[1],
                "cells": [],
            }
            if _region_is_clear(candidate) and _region_is_flat_enough(candidate):
                field_regions.insert(0, candidate)
                break

    field_regions = field_regions[:8]  # обмежити к-сть полів на сцену (продуктивність/читабельність)
    field_rects = [
        (r["center"][0], r["center"][1], r["length"] / 2.0, r["width"] / 2.0, r["angle_deg"])
        for r in field_regions
    ]

    # Посадки Вороного (kind="plantation", доп. фаза "розширення kind") —
    # та сама груп-BFS/PCA логіка, що й поля (``_voronoi_grouped_regions``),
    # лише інший kind клітинок. На відміну від build-time посадок
    # (``scripts/generate_map_field.py`` — ФІЗИЧНІ obstacle, апарат
    # заплутується) це суто ВІЗУАЛЬНИЙ шар, як і решта ``_voronoi_cells``-
    # конвеєра (рельєф/поля) — дрон пролітає крізь неї, не застряє.
    # max_cells=3 (не 6-8, як у полів) — КОЖНЕ дерево тут БАКОВАНА Blender-
    # модель (тисячі трикутників, повний shadow-pass), не дешевий процедурний
    # конус: профілювання (720-крокові физика-тести, patrol seed=1) показало,
    # що навіть 11 викликів scatter_tree_line (усі поля+посадки разом)
    # МАЙЖЕ ПОДВОЮЄ час 720 кроків (44с -> 99с, тест валиться на таймауті
    # 60с) — тому регіонів/рядів/густоти тут СВІДОМО менше, ніж напрошується.
    plantation_regions = _voronoi_grouped_regions(
        flat_radius, terrain_outer_radius, terrain_seed, kind="plantation", min_cells=2, max_cells=2
    )
    plantation_regions = [r for r in plantation_regions if _region_is_clear(r) and _region_is_flat_enough(r)]

    def _overlaps_any(region: dict, others: list[dict]) -> bool:
        # Наближена перевірка перетину повернутих прямокутників: центр
        # ОДНОГО всередині прямокутника ІНШОГО (в обидва боки) — не повна
        # геометрія перетину, але ловить найпомітніший випадок (посадка
        # встромлена в поле чи навпаки), достатньо для декоративного шару.
        for other in others:
            for a, b in ((region, other), (other, region)):
                cx, cy = b["center"]
                ang = math.radians(b["angle_deg"])
                ca, sa = math.cos(-ang), math.sin(-ang)
                ox, oy = a["center"]
                lx = (ox - cx) * ca - (oy - cy) * sa
                ly = (ox - cx) * sa + (oy - cy) * ca
                if abs(lx) < b["length"] / 2.0 and abs(ly) < b["width"] / 2.0:
                    return True
        return False

    plantation_regions = [r for r in plantation_regions if not _overlaps_any(r, field_regions)]
    plantation_regions = plantation_regions[:1]  # кап к-сті посадок на сцену (продуктивність, докладніше вище)

    # Земля значно більша за старі 50м (реальний фідбек: "обрив землі без
    # горизонту") — край тепер за межею туману, тож не видно; небо+сонце+туман
    # роблять "живий" горизонт замість сірої порожнечі. Тепер РЕАЛЬНИЙ рельєф
    # (реальний фідбек: "рельєф, не лише пагорби на горизонті") — плоско лише
    # в робочій зоні (``flat_radius``), далі справжні хребти-горби/яри +
    # зелено-жовтий колірний градієнт плямами (``_terrain_dryness``).
    build_terrain(
        ground_np, radius=250.0, flat_radius=flat_radius, outer_radius=terrain_outer_radius,
        max_height=4.5, seed=terrain_seed,
    )
    # Розсіяна зелень (трава/кущі) навколо центру — щоб поле не було
    # однотонним (реальний фідбек: "карти порожні"). Центр лишається чистим
    # під точку старту. Суто візуальне, зведене в кілька draw-викликів.
    # Густий КИЛИМ трави (density map + jittered grid, ~14 стебел/м² -> десятки
    # тисяч, один векторизований Geom) з галявинами + кущі. Реальний фідбек:
    # "трава — це шар майже усюди, густий, крім десь витоптана". Стебла
    # сідають на реальну висоту рельєфу (``terrain_*``) і тонуються тим самим
    # зелено-жовтим градієнтом, що й земля під ними.
    scatter_ground_cover(
        engine.render, radius=grass_radius, density_per_m2=15.0, seed=terrain_seed,
        exclude_rects=[*road_rects, *field_rects],
        terrain_flat_radius=flat_radius, terrain_outer_radius=terrain_outer_radius,
        terrain_max_height=4.5, terrain_seed=terrain_seed,
        # Повна густота біля гравця, плавне рідшання вдалині (не жорстке
        # "коло") — трава сягає ``grass_radius`` без вибуху к-сті стебел.
        density_taper_start=flat_radius + 15.0, density_taper_end=grass_radius, density_taper_min=0.15,
    )
    # Пшеничні ділянки сцени (опційна секція wheat_patches у YAML сцени,
    # доп. фаза "графіка 3.0") — візуальні, апарат пролітає крізь стебла.
    for i, patch in enumerate(scene_decor["wheat_patches"]):
        scatter_wheat_patch(
            engine.render, center=patch["center"], size=patch["size"], count=patch["count"], seed=17 + i
        )
    # Поля Вороного: пшениця РЯДКАМИ вздовж головної осі + опційний паркан/
    # лісосмуга вздовж довгих країв (реальний фідбек: "заборчик МОЖЕ бути
    # вздовж поля" + "посадки — багато дерев лінією вздовж поля"). Дерева НЕ
    # ставляться на пагорбах/ярах — лише вздовж полів, тож правило "на
    # пагорбі дерев не має бути" виконується самою логікою розміщення.
    field_rng = np.random.default_rng(terrain_seed + 500)
    for i, region in enumerate(field_regions):
        cx, cy = region["center"]
        length, width, ang = region["length"], region["width"], region["angle_deg"]
        count = int(np.clip(length * width * 13.0, 250, 6000))
        scatter_wheat_patch(
            engine.render, center=(cx, cy), size=(length, width), count=count,
            seed=terrain_seed + 700 + i, angle_deg=ang,
        )
        rad = math.radians(ang)
        along = np.array([math.cos(rad), math.sin(rad)])
        across = np.array([-math.sin(rad), math.cos(rad)])
        center_v = np.array([cx, cy])
        edge_roll = field_rng.random()
        for side in (-1.0, 1.0):
            edge_a = center_v + along * (length / 2.0) + across * (side * width / 2.0)
            edge_b = center_v - along * (length / 2.0) + across * (side * width / 2.0)
            if edge_roll < 0.35:
                build_fence_line(engine.render, tuple(edge_a), tuple(edge_b))
            elif edge_roll < 0.70:
                scatter_tree_line(engine.render, tuple(edge_a), tuple(edge_b), seed=terrain_seed + 900 + i * 2 + int(side))
            # >=0.70 (~30%) — без облямівки, гола межа поля (не кожне поле обгороджене).
    # Посадки: заповнити ділянку ПАРАЛЕЛЬНИМИ рядами дерев (не лише
    # облямівка, як межі поля) — справжня лісосмуга/гай. Перевикористовує
    # ГОТОВИЙ ``scatter_tree_line`` на кожен ряд, жодного нового коду
    # побудови дерева.
    for i, region in enumerate(plantation_regions):
        cx, cy = region["center"]
        length, width, ang = region["length"], region["width"], region["angle_deg"]
        rad = math.radians(ang)
        along = np.array([math.cos(rad), math.sin(rad)])
        across = np.array([-math.sin(rad), math.cos(rad)])
        center_v = np.array([cx, cy])
        # n_rows/spacing/довжина навмисно СКРОМНІ (не "заповнити щільно, як
        # поле") — кожне дерево тут БАКОВАНА Blender-модель (тисячі
        # трикутників + shadow-pass), не дешевий процедурний конус;
        # профілювання показало, що навіть 2-3 ряди відчутно дорожчі за
        # процедурну геометрію такого ж масштабу (докладніше при обчисленні
        # ``plantation_regions`` вище).
        n_rows = 2
        row_length = min(length, 28.0)
        for row in range(n_rows):
            row_t = (row + 0.5) / n_rows - 0.5
            row_center = center_v + across * (row_t * width)
            start = row_center - along * (row_length / 2.0)
            end = row_center + along * (row_length / 2.0)
            scatter_tree_line(
                engine.render, tuple(start), tuple(end), spacing=4.5,
                seed=terrain_seed + 1300 + i * 5 + row, species=("pine", "oak", "birch"),
            )
    # Раніше тут стояли окремі декоративні пагорби-купола за межею поля —
    # ВИДАЛЕНО (реальний фідбек: "прибери ці величезні бульбашки на
    # горизонті") — тепер рельєф (``build_terrain`` вище) сам дає геометрію
    # ближче до поля, а за нею земля рівна й тане в димці (``add_fog``
    # нижче), як і має бути на відкритому полі.
    add_sky(engine)
    add_clouds(engine)  # хмари над полем (реальний фідбек: "не вистачає хмар")
    add_fog(engine)
    # Далека площина камери має накривати небо (радіус купола ~1400м); інакше
    # купол/далеку землю обрізало б стандартною далекою площиною (~1000м).
    engine.camLens.setFar(4000.0)
    # Справжні god rays (доп. фаза "графіка 2.0", реальний фідбек: "справжні
    # god rays замість гало+туману") — динамічний пост-процес, ЗАМІНЮЄ
    # видалений статичний "гало" (render/sky.py), не додається поверх нього.
    godrays = GodRays(engine)
    for obstacle_np, half_extents, kind in obstacles:
        attach_scenery_visual(obstacle_np, half_extents, kind)

    for gate in getattr(scenario, "gates", ()):  # ворота статичні -> маркер один раз, без апдейтів
        add_gate_marker_visual(engine.render, tuple(gate.center), radius=gate.radius, normal=tuple(gate.normal))
    marker_by_id: dict[int, object] = {}
    active_effects: list = []  # доп. фаза "реальні моделі": активні ExplosionEffect

    vehicle = Multirotor(world, config_name=f"vehicles/{args.vehicle}", parent=engine.render)
    # RL-агент (фаза 6) навчений НАВІГАЦІЇ, вже перебуваючи в повітрі
    # (``ml/rl_strike.yaml::start_altitude_range`` — старт ЗАВЖДИ поруч із
    # ціллю за висотою, ніколи біля землі) — він НЕ вчився "злітати з землі".
    # Тому АВТОНОМНИЙ старт — на 3м (узгоджено з ``eval_autonomous.py``).
    #
    # МАНУАЛЬНО апарат СТОЇТЬ НА ЗЕМЛІ (як у Liftoff): низ корпусу торкається
    # землі (z = піввисота корпусу + маленький зазор), швидкість нульова
    # (``reset`` обнуляє). Роззброєний, він просто лежить на землі, поки пілот
    # не дасть газ і не злетить плавно. Раніше старт був на 1м у повітрі —
    # роззброєний апарат ВІЛЬНО ПАДАВ, тож "одразу мав певну швидкість, і
    # контролювати було неможливо" (реальний фідбек користувача).
    starts_airborne = autopilot_source is not None and bool(getattr(args, "start_autonomous", False))
    if starts_airborne:
        spawn_pos = np.array([0.0, 0.0, 3.0])
    else:
        ground_rest_z = float(vehicle.cfg.body_half_extent[2]) + 0.02
        spawn_pos = np.array([0.0, 0.0, ground_rest_z])
    vehicle.reset(pos=spawn_pos)
    from dronesim.render.materials import METAL_PAINTED

    add_box_visual(
        vehicle.node_path, vehicle.cfg.body_half_extent, color=(0.9, 0.75, 0.1, 1.0), material=METAL_PAINTED
    )
    propeller_rig = PropellerRig(vehicle.node_path, vehicle.motor_positions, vehicle.motor_spin_dirs)
    safety_cfg = load_safety_config()
    effects_cfg = load_config("effects")  # доп. фаза "реальні моделі" — ефекти знищення

    controllers = _build_controllers(vehicle)
    if autopilot_source is not None:
        _require_pos_hold(controllers, args.vehicle)

    if args.input == "gamepad":
        from dronesim.input.gamepad import GamepadSource

        control_source = GamepadSource()
    else:
        from dronesim.input.keyboard import KeyboardSource

        control_source = KeyboardSource(engine)

    # Стартовий режим (доп. фаза "справжній Liftoff"): Angle (самовирівнювання),
    # якщо апарат його підтримує — стабільніше перше знайомство з керуванням,
    # ніж Acro (де відпущений стік НЕ повертає апарат у горизонт). Хто хоче
    # фрістайл Acro — перемикає `Tab` сам.
    start_mode = FlightMode.ANGLE if FlightMode.ANGLE in controllers else FlightMode.ACRO
    control_source.set_start_mode(start_mode)

    # Монтуємо камеру трохи попереду корпусу (масштабовано під розмір апарата),
    # щоб ніс не потрапляв у кадр (актуально для великих корпусів, quad_large).
    body_half_extent = vehicle.cfg.body_half_extent
    cam_offset = (float(body_half_extent[0]) + 0.05, 0.0, float(body_half_extent[2]) * 0.5)

    camera = FPVCamera(
        engine, vehicle.node_path, width=args.cam_width, height=args.cam_height, mount_offset=cam_offset
    )
    attach_main_camera(engine, vehicle.node_path, mount_offset=cam_offset)

    controls_hint = ""
    if args.input == "keyboard":
        controls_hint = (
            f"{build_keyboard_hint(control_source.cfg.bindings)}\n{build_takeoff_hint(control_source.cfg.bindings)}"
        )
    hud = HUD(engine, controls_hint=controls_hint)
    audio = AudioController(engine)

    detection_overlay = None
    if detector is not None:
        from dronesim.render.detection_overlay import DetectionOverlay

        detection_overlay = DetectionOverlay(engine, camera.texture)

    # Турбулентність (доп. фаза "реалізм фізики") — зв'язана з тим самим
    # --wind: сильніший стабільний вітер природно супроводжується сильнішою
    # поривчастістю. Момент (torque_at) — окремо від сталої/гармонічної сили
    # (та БЕЗ моменту, фаза 3) — лише стохастична складова хитає апарат.
    wind = (
        Wind(
            base_force=(float(args.wind), 0.0, 0.0),
            turbulence_intensity=0.35 * float(args.wind),
            seed=int(getattr(args, "seed", 0) or 0),
        )
        if args.wind
        else None
    )
    state_box = {
        "active_mode": start_mode,
        "scenario_done": False,
        "autonomous": autopilot_source is not None and bool(getattr(args, "start_autonomous", False)),
        "critical_since": None,  # час (state.t), відколи безперервно triable "critical" (None = не triable)
        "camera_mode": "fpv",
        "prev_armed": False,
    }

    def _toggle_autonomous() -> None:
        state_box["autonomous"] = not state_box["autonomous"]
        print(f"[autopilot] {'УВІМКНЕНО' if state_box['autonomous'] else 'вимкнено — керує людина'}")

    if autopilot_source is not None:
        engine.accept("p", _toggle_autonomous)  # перехоплення керування — миттєво, будь-якої миті

    def _toggle_camera() -> None:
        if state_box["camera_mode"] == "fpv":
            state_box["camera_mode"] = "chase"
            attach_chase_camera(engine)
        else:
            state_box["camera_mode"] = "fpv"
            attach_main_camera(engine, vehicle.node_path, mount_offset=cam_offset)
        print(f"[camera] {state_box['camera_mode']}")

    engine.accept("c", _toggle_camera)

    def _restart(reason: str) -> None:
        """Скинути апарат до spawn-точки (доп. фаза поліш: межі/переворот).
        Позиції цілей/воріт фіксовані з конфігу (не рандомізуються за seed —
        емпірично встановлено у фазі 7, docs/DECISIONS.md) — тож маркери НЕ
        потребують перестворення, лише ``scenario.reset()`` скидає прапорці
        "уражено"/лічильники кіл, які й так синхронізуються щокадру."""
        print(f"[restart] {reason}")
        vehicle.reset(pos=spawn_pos)
        controllers[state_box["active_mode"]].reset()
        if scenario is not None:
            scenario.reset(seed=getattr(args, "seed", None))
            state_box["scenario_done"] = False
        state_box["critical_since"] = None

    engine.accept(safety_cfg.restart_key, lambda: _restart("manual restart"))

    def physics_step(dt: float) -> None:
        if autopilot_source is not None and state_box["autonomous"]:
            target_pos = select_autopilot_target(scenario, vehicle.state.pos, vehicle.state.t)
            autopilot_source.set_context(vehicle.state, target_pos)
            cmd = autopilot_source.get_command()
        else:
            cmd = control_source.get_command()

        active_mode = cmd.mode if cmd.mode in controllers else FLIGHT_MODE_CYCLE[0]
        if active_mode != state_box["active_mode"]:
            controllers[active_mode].reset()
            state_box["active_mode"] = active_mode

        controller = controllers[active_mode]
        thrusts = controller.update(cmd, vehicle.state, dt)
        if wind is not None:
            vehicle.apply_external_force(wind.force_at(vehicle.state.t))
            vehicle.apply_external_torque(wind.torque_at(vehicle.state.t))
        vehicle.apply_motor_commands(thrusts, dt)
        propeller_rig.update(thrusts, vehicle.cfg.max_thrust_per_motor, dt)
        godrays.update(dt)
        world.step(dt)
        vehicle.advance_time(dt)

        audio.update(cmd.throttle, cmd.arm)
        if cmd.arm != state_box["prev_armed"]:
            audio.notify_arm_changed(cmd.arm)
            state_box["prev_armed"] = cmd.arm

        if state_box["camera_mode"] == "chase":
            _, _, yaw_rad = quat_to_euler_rad(vehicle.state.quat)
            update_chase_camera(engine, vehicle.state.pos, yaw_rad)

        safety_status = evaluate_flight_safety(vehicle.state, spawn_pos[:2], safety_cfg)
        if safety_status.level == "critical":
            if state_box["critical_since"] is None:
                state_box["critical_since"] = vehicle.state.t
            elif vehicle.state.t - state_box["critical_since"] >= safety_cfg.crash_grace_period_s:
                _restart(safety_status.reason)
        else:
            state_box["critical_since"] = None

        for effect in active_effects:
            effect.update(dt)
        active_effects[:] = [e for e in active_effects if not e.finished]

        pilot_str = "AUTO" if state_box["autonomous"] else "MANUAL"
        mode_str = f"{pilot_str}:{active_mode.value}"
        if scenario is not None:
            result = scenario.step(vehicle.state)
            _sync_target_markers(
                engine, scenario, marker_by_id, vehicle.state.t,
                world=world, effects_cfg=effects_cfg, active_effects=active_effects, audio=audio,
            )
            mode_str = f"{mode_str} | {args.scenario} score={result.score:.0f}"
            if result.done and not state_box["scenario_done"]:
                state_box["scenario_done"] = True
                _finish_scenario(args.scenario, scenario, result, vehicle.state.t)

        hud.update(
            vehicle.state,
            cmd,
            active_mode=mode_str,
            warning_text=safety_status.reason or None,
            warning_level=safety_status.level,
        )

    engine.set_physics_callback(physics_step)

    if hasattr(control_source, "update"):  # KeyboardSource: ramp осей на частоті рендеру
        from panda3d.core import ClockObject

        global_clock = ClockObject.getGlobalClock()

        def _keyboard_update_task(task):
            control_source.update(dt=global_clock.getDt())
            return task.cont

        engine.taskMgr.add(_keyboard_update_task, "dronesim-keyboard-update", sort=5)

    if detector is not None:
        # Інференс YOLO (~20-40мс на CPU) занадто повільний для 240Hz фізики —
        # запускаємо на частоті рендеру, і то не щокадру (детекторний task).
        detect_state = {"frame_counter": 0}
        detect_every_n_frames = 5

        def _detection_task(task):
            detect_state["frame_counter"] += 1
            if detect_state["frame_counter"] % detect_every_n_frames == 0:
                frame = camera.get_frame(vehicle.state, t=vehicle.state.t)
                detections = detector.predict(frame.rgb)
                detection_overlay.update(detections, args.cam_width, args.cam_height)
            return task.cont

        engine.taskMgr.add(_detection_task, "dronesim-detection", sort=7)

    return {
        "engine": engine,
        "world": world,
        "vehicle": vehicle,
        "controllers": controllers,
        "control_source": control_source,
        "camera": camera,
        "hud": hud,
        "scenario": scenario,
        "detection_overlay": detection_overlay,
        "autopilot_source": autopilot_source,
        "propeller_rig": propeller_rig,
        "audio": audio,
        "state": state_box,  # для інтроспекції/тестів (напр. чи спрацював перемикач "p")
        "marker_by_id": marker_by_id,  # для інтроспекції/тестів (доп. фаза "реальні моделі")
        "active_effects": active_effects,
        "godrays": godrays,
    }


def _run_session_loop(session: dict, args: argparse.Namespace) -> int:
    """Спільний головний цикл для ``fly`` і ``autonomous`` (фаза 7): різниця між
    ними лише в тому, ЯК зібрана сесія (build_fly_session), не в тому, як вона
    крутиться/звільняється."""
    engine = session["engine"]
    try:
        if args.max_steps is not None:
            for _ in range(args.max_steps):
                engine.taskMgr.step()
        else:
            engine.run()  # блокує до закриття вікна користувачем
    finally:
        session["camera"].close()
        session["hud"].close()
        session["control_source"].close()
        session["propeller_rig"].close()
        session["audio"].close()
        session["godrays"].remove()
        if session["detection_overlay"] is not None:
            session["detection_overlay"].close()
        if session["autopilot_source"] is not None:
            session["autopilot_source"].close()
    return 0


def _cmd_fly(args: argparse.Namespace) -> int:
    """Фаза 2: ручний політ від першої особи — Panda3D-вікно, Acro-режим.

    Опційно з автопілотом (фаза 7, ``--autopilot``): стартує в MANUAL, клавіша
    ``p`` перемикає на AUTONOMOUS і назад будь-якої миті.
    """
    session = build_fly_session(args)
    return _run_session_loop(session, args)


def _cmd_autonomous(args: argparse.Namespace) -> int:
    """Фаза 7: автономний прохід сценарію під навченим RL-агентом
    (``ml/rl/train.py``, фаза 6). Старт одразу в AUTONOMOUS — клавіша ``p``
    миттєво повертає керування людині (критерій приймання ROADMAP)."""
    args.autopilot = args.checkpoint
    session = build_fly_session(args)
    return _run_session_loop(session, args)


def _not_implemented(phase: str, module: str):
    def _run(args: argparse.Namespace) -> int:
        print(
            f"[TODO] Ця підкоманда реалізується у {phase}.\n"
            f"       Дивись ROADMAP.md і модуль: {module}",
            file=sys.stderr,
        )
        return 2

    return _run


def _add_common_fly_arguments(parser: argparse.ArgumentParser) -> None:
    """Аргументи, спільні для ``fly`` і ``autonomous`` (фаза 7) — обидві
    зрештою збирають сесію через ``build_fly_session``."""
    parser.add_argument("--vehicle", default="fpv_5inch", help="Ім'я конфігу з configs/vehicles/")
    parser.add_argument("--input", choices=["keyboard", "gamepad"], default="keyboard")
    parser.add_argument(
        "--scenario",
        default=None,
        choices=list(SCENARIO_REGISTRY),
        help="Місія (фаза 4): gate_race, patrol, strike_range",
    )
    parser.add_argument(
        "--level",
        default=None,
        help="Конкретна розкладка контенту для --scenario (доп. фаза розмаїття "
        "контенту) — напр. gate_race_forest; без цього прапорця — базова розкладка. "
        "Перелік: dronesim.scenarios.registry.list_levels(scenario)",
    )
    parser.add_argument("--seed", type=int, default=None, help="Seed для детермінованого reset() сценарію")
    parser.add_argument("--cam-width", type=int, default=320)
    parser.add_argument("--cam-height", type=int, default=240)
    parser.add_argument(
        "--wind", type=float, default=0.0, help="Стала сила вітру вздовж world +X, Н (0 = вимкнено)"
    )
    parser.add_argument(
        "--detector",
        default=None,
        help="Шлях до навчених ваг YOLO (best.pt) — увімкнути live-детекцію (фаза 5)",
    )
    parser.add_argument("--detector-conf", type=float, default=0.25, help="Поріг впевненості детектора")
    parser.add_argument(
        "--offscreen", action="store_true", help="Без вікна ОС (headless-перевірка рушія)"
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=None,
        help="Завершити після N кроків замість реального польоту (смоук-тест CLI)",
    )


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="dronesim", description="Симулятор FPV/БПЛА (Liftoff-like)")
    sub = p.add_subparsers(dest="command", required=True)

    e = sub.add_parser("sim-empty", help="Фаза 0: порожній цикл симуляції")
    e.add_argument("--steps", type=int, default=240)
    e.add_argument("--hz", type=int, default=240)
    e.set_defaults(func=_cmd_sim_empty)

    fly = sub.add_parser("fly", help="Ручний політ (фаза 2), опційно з автопілотом (фаза 7)")
    _add_common_fly_arguments(fly)
    fly.add_argument(
        "--autopilot",
        default=None,
        help="Шлях до навченого PPO (фаза 6) — увімкнути можливість автопілота; "
        "старт у MANUAL, клавіша 'p' перемикає на AUTONOMOUS і назад",
    )
    fly.set_defaults(func=_cmd_fly, start_autonomous=False)

    auto = sub.add_parser(
        "autonomous", help="Автономний ML-режим (фаза 7): проходить сценарій під RL-агентом"
    )
    _add_common_fly_arguments(auto)
    auto.add_argument(
        "--checkpoint",
        default="runs/rl/ppo_strike/model.zip",
        help="Шлях до навченого PPO-чекпойнта (ml/rl/train.py, фаза 6)",
    )
    auto.set_defaults(func=_cmd_autonomous, scenario="strike_range", vehicle="quad_large", start_autonomous=True)

    return p


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())

"""Видима геометрія сцени: коробки, площина землі, маркери цілей/воріт.

Рендер лише ДОДАЄ дочірні вузли з геометрією до вже існуючих ``NodePath``
фізичних тіл (``PhysicsWorld``/``Vehicle``) — трансформація успадковується
автоматично через граф сцени (Bullet вже пише в той самий ``NodePath``), тож
цей модуль нічого не рахує й не синхронізує вручну (правило "рендер не
рахує фізику", SKILL.md).

``setTwoSided(True)`` вимикає backface culling для згенерованих граней —
свідомий компроміс: коробка складена з примітивних карток (``CardMaker``), і
без цього неправильно орієнтована грань могла б стати невидимою.

БАГ, ЗНАЙДЕНИЙ РЕАЛЬНИМ ПОЛЬОТОМ ("не відрізниш, куди падає сонце", доп.
фаза "графіка 3.0"): `_BOX_FACES` для Y-осі (перед/зад) мав ОДНАКОВУ пару
(heading=0/180), емпірично перевірену як ТАКУ, ЩО ДАЄ НОРМАЛЬ ВСЕРЕДИНУ
коробки (не назовні) — `setTwoSided` ховає симптом "грань невидима", але
освітлення й далі рахувалось за неправильною нормаллю. Це стосувалось
КОЖНОЇ коробки в грі (корпус апарата, obstacle-перешкоди, дах будівлі,
ворота) — половина Y-граней світилась "як ззаду", тому взагалі не можна
було зрозуміти, з якого боку сонце. Перевірено емпірично через
``getRelativeVector`` (SKILL.md пастка "не виводь HPR теоретично") для ВСІХ
6 граней — X/Z-грані виявились коректними, лише Y-пара була переплутана.

Текстури (трава, мішень) генеруються ПРОЦЕДУРНО через ``PNMImage`` (доп. фаза
поліш) — не завантажуються ззовні, щоб не піднімати питань ліцензування і
лишатись у дусі решти проєкту (усе процедурне, без готових ассетів).
"""

from __future__ import annotations

import numpy as np
from panda3d.core import CardMaker, Material, NodePath, PNMImage, Point2, Point3, Texture, Vec4

# Кожна грань коробки: (heading, pitch, roll), індекс осі зсуву, знак зсуву.
_BOX_FACES = (
    ((180, 0, 0), 1, 1),  # +Y перед, нормаль +Y (виправлено з 0 — була всередину)
    ((0, 0, 0), 1, -1),  # -Y зад, нормаль -Y (виправлено з 180 — була всередину)
    ((90, 0, 0), 0, 1),  # +X право
    ((-90, 0, 0), 0, -1),  # -X ліво
    ((0, -90, 0), 2, 1),  # +Z верх
    ((0, 90, 0), 2, -1),  # -Z низ
)


def add_box_visual(
    parent: NodePath,
    half_extents: tuple[float, float, float],
    color: tuple[float, float, float, float] = (0.6, 0.6, 0.65, 1.0),
    material: Material | None = None,
) -> NodePath:
    """Проста коробка з 6 карток — плейсхолдер-меш для корпусу апарата/перешкод.

    ``material`` — опційний PBR-пресет (``render/materials.py``, доп. фаза
    "графіка 2.0"); без нього Panda3D підставляє глобальний дефолт
    (roughness=1.0, повністю матовий) незалежно від того, що коробка
    представляє."""
    box = parent.attachNewNode("box_visual")
    dims = tuple(half_extents)

    for hpr, axis, sign in _BOX_FACES:
        size = [d for i, d in enumerate(dims) if i != axis]
        cm = CardMaker("face")
        cm.setFrame(-size[0], size[0], -size[1], size[1])
        face = box.attachNewNode(cm.generate())
        face.setHpr(*hpr)
        offset = [0.0, 0.0, 0.0]
        offset[axis] = dims[axis] * sign
        face.setPos(*offset)

    box.setColor(Vec4(*color))
    box.setTwoSided(True)
    if material is not None:
        box.setMaterial(material)
    box.flattenStrong()
    return box


def add_marker_visual(
    parent: NodePath,
    pos: tuple[float, float, float],
    size: float = 0.6,
    color: tuple[float, float, float, float] = (1.0, 0.2, 0.2, 1.0),
) -> NodePath:
    """Маркер-коробка (загальний, невизначеного типу) в заданій позиції.

    На відміну від ``add_box_visual``, НЕ прив'язується до фізичного тіла: цілі
    й ворота (world/targets.py) не є тілами Bullet, лише геометричні маркери.
    Для цілей/воріт ігрових сценаріїв надавай перевагу
    ``add_target_marker_visual``/``add_gate_marker_visual`` нижче — вони
    візуально відрізняються від звичайних перешкод-коробок (доп. фаза поліш);
    ця функція лишається як загальний примітив для іншого використання.
    """
    marker = add_box_visual(parent, (size, size, size), color=color)
    marker.setPos(*pos)
    return marker


def _make_grass_texture(size: int = 128, seed: int = 0) -> Texture:
    """Процедурна текстура трави з БАГАТОМАСШТАБНОЮ варіацією (реальний фідбек:
    "все однотонне, немає текстур"): великі плями світлішої/темнішої трави
    (низька частота) + дрібне зерно (висока частота) + подекуди землисті
    залисини — щоб земля читалась як текстурована, а не як плоска зелена пляма.
    """
    rng = np.random.default_rng(seed)
    # Низька частота: 8x8 блоки, розтягнуті до розміру (великі плями), потім
    # ЗГЛАДЖЕНІ кількома прохід-боксами (інакше видно різкі квадрати-плитку).
    coarse = np.kron(rng.uniform(0.0, 1.0, (8, 8)), np.ones((size // 8, size // 8)))
    for _ in range(3):
        coarse = (
            coarse + np.roll(coarse, 1, 0) + np.roll(coarse, -1, 0)
            + np.roll(coarse, 1, 1) + np.roll(coarse, -1, 1)
        ) / 5.0
    fine = rng.uniform(0.0, 1.0, (size, size))  # дрібне зерно трави
    val = np.clip(0.55 * coarse + 0.45 * fine, 0.0, 1.0)

    green_dark = np.array([0.11, 0.19, 0.07])
    green_light = np.array([0.27, 0.43, 0.19])
    pixels = green_dark[None, None, :] + val[:, :, None] * (green_light - green_dark)[None, None, :]
    # Землисті залисини у найтемніших ~12% плям (перцентиль — стійко до того,
    # що згладжування стискає діапазон коефіцієнта).
    dirt = np.array([0.33, 0.26, 0.16])
    dirt_mask = coarse < np.percentile(coarse, 12)
    pixels[dirt_mask] = dirt[None, :] + fine[dirt_mask][:, None] * 0.05
    pixels = np.clip(pixels, 0.0, 1.0)

    img = PNMImage(size, size)
    for y in range(size):
        for x in range(size):
            img.setXelA(x, y, float(pixels[y, x, 0]), float(pixels[y, x, 1]), float(pixels[y, x, 2]), 1.0)

    tex = Texture("grass")
    tex.load(img)
    tex.setWrapU(Texture.WMRepeat)
    tex.setWrapV(Texture.WMRepeat)
    return tex


def add_ground_visual(
    parent: NodePath,
    size: float = 50.0,
    tile_size_m: float = 4.0,
    seed: int = 0,
) -> NodePath:
    """Квадратна площина землі під ноги, з тайловою процедурною текстурою трави
    (замість суцільного кольору) — для орієнтації в польоті."""
    cm = CardMaker("ground")
    cm.setFrame(-size, size, -size, size)
    repeats = max(1.0, (2.0 * size) / tile_size_m)
    cm.setUvRange(Point2(0.0, 0.0), Point2(repeats, repeats))

    from dronesim.render.materials import FOLIAGE

    ground = parent.attachNewNode(cm.generate())
    ground.setP(-90)  # покласти картку горизонтально (за замовч. вона в площині X-Z)
    ground.setTexture(_make_grass_texture(seed=seed))
    ground.setTwoSided(True)
    ground.setMaterial(FOLIAGE)
    return ground


def _make_bullseye_texture(size: int = 64) -> Texture:
    """Процедурна текстура "мішені" (концентричні червоно-білі кола) — щоб цілі
    виглядали як цілі, а не як ще одна сіра/кольорова коробка-перешкода."""
    img = PNMImage(size, size)
    center = size / 2.0
    n_rings = 4
    for y in range(size):
        for x in range(size):
            dist_frac = min(1.0, np.hypot(x - center, y - center) / center)
            ring = int(dist_frac * n_rings) % 2
            color = (1.0, 0.15, 0.15) if ring == 0 else (1.0, 1.0, 1.0)
            img.setXelA(x, y, *color, 1.0)

    tex = Texture("bullseye")
    tex.load(img)
    return tex


def add_target_marker_visual(
    parent: NodePath, pos: tuple[float, float, float], size: float = 0.8
) -> NodePath:
    """Маркер навчальної цілі: дві перехресні текстуровані картки ("bullseye"),
    видимі під будь-яким кутом підльоту — замість суцільної коробки (доп. фаза
    поліш: цілі мають виглядати як цілі)."""
    marker = parent.attachNewNode("target_visual")
    tex = _make_bullseye_texture()

    for heading in (0, 90):
        cm = CardMaker("target_card")
        cm.setFrame(-size, size, -size, size)
        card = marker.attachNewNode(cm.generate())
        card.setHpr(heading, 0, 0)
        card.setTexture(tex)

    marker.setTwoSided(True)
    marker.setPos(*pos)
    return marker


def set_target_marker_hit(marker: NodePath, hit: bool) -> None:
    """Перефарбувати вже уражену ціль у зелений (замість підміни текстури —
    дешевше й достатньо помітно)."""
    marker.setColorScale(Vec4(0.3, 1.0, 0.3, 1.0) if hit else Vec4(1.0, 1.0, 1.0, 1.0))


def add_gate_marker_visual(
    parent: NodePath,
    pos: tuple[float, float, float],
    radius: float,
    normal: tuple[float, float, float],
    bar_thickness: float = 0.15,
    color: tuple[float, float, float, float] = (0.2, 0.5, 1.0, 1.0),
) -> NodePath:
    """Рамка-кільце (4 бруси навколо отвору) — виглядає як "пролетіти крізь",
    на відміну від суцільної коробки (доп. фаза поліш). Орієнтована
    перпендикулярно ``normal`` (напрямок правильного прольоту, world/targets.py
    ::Gate) через ``lookAt`` — НЕ крутимо Euler вручну після (SKILL.md пастка:
    lookAt + ручна зміна кутів ламається при ненульовому pitch)."""
    gate_np = parent.attachNewNode("gate_visual")
    gate_np.setPos(*pos)
    look_target = np.asarray(pos, dtype=float) + np.asarray(normal, dtype=float)
    gate_np.lookAt(Point3(*look_target))

    half_t = bar_thickness / 2.0
    horizontal_half = (radius + half_t, half_t, half_t)
    vertical_half = (half_t, half_t, radius)
    bars = (
        ((0.0, 0.0, radius), horizontal_half),
        ((0.0, 0.0, -radius), horizontal_half),
        ((radius, 0.0, 0.0), vertical_half),
        ((-radius, 0.0, 0.0), vertical_half),
    )
    for offset, half_extents in bars:
        bar = add_box_visual(gate_np, half_extents, color=color)
        bar.setPos(*offset)

    return gate_np

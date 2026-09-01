"""Небо, сонце й туман (доп. фаза "жива графіка", реальний фідбек користувача:
"немає неба - все сіре, сонця. Є обрив землі без горизонту").

Усе процедурне, без зовнішніх ассетів (як і решта проєкту):
- ``add_sky`` будує градієнтний купол неба (меш ``GeomVertexData`` з
  ПОФАРБОВАНИМИ вершинами: світлий горизонт -> синій зеніт) + диск сонця
  (білборд із процедурною радіальною текстурою світіння). Купол — у бін
  "background" з вимкненою глибиною/шейдером/освітленням/туманом, тож завжди
  малюється позаду сцени й НЕ спотворюється PBR-освітленням чи туманом.
- ``add_fog`` додає лінійний туман кольору горизонту — далекі об'єкти й КРАЙ
  площини землі плавно зливаються з небом (ховає "обрив землі без горизонту").

Купол центрований на початку координат із великим радіусом: ігрове поле
(~±120м, configs/safety.yaml) мізерне проти радіуса купола, тож паралакс при
русі апарата непомітний — не потрібно ні стежити за камерою, ні окремої задачі.
"""

from __future__ import annotations

import math

import numpy as np
from panda3d.core import (
    CardMaker,
    Fog,
    Geom,
    GeomNode,
    GeomTriangles,
    GeomVertexData,
    GeomVertexFormat,
    GeomVertexWriter,
    NodePath,
    PNMImage,
    Texture,
    TransparencyAttrib,
    Vec4,
)

_HORIZON_COLOR = (0.72, 0.80, 0.88)  # блідо-блакитний біля горизонту
_ZENITH_COLOR = (0.28, 0.48, 0.78)  # насиченіший синій у зеніті
_SUN_COLOR = (1.0, 0.95, 0.8)

DEFAULT_SUN_AZIMUTH_DEG = 45.0
DEFAULT_SUN_ELEVATION_DEG = 48.0


def sun_direction(azimuth_deg: float = DEFAULT_SUN_AZIMUTH_DEG, elevation_deg: float = DEFAULT_SUN_ELEVATION_DEG) -> np.ndarray:
    """Нормований напрямок на сонце (світові координати) — спільна формула
    для ``add_sky`` (позиція диска на куполі) і ``render/godrays.py``
    (де на екрані шукати джерело променів), щоб не дублювати азимут/висоту
    у двох місцях і не ризикувати їх розсинхронізацією."""
    az, el = math.radians(azimuth_deg), math.radians(elevation_deg)
    return np.array([math.cos(el) * math.cos(az), math.cos(el) * math.sin(az), math.sin(el)])


def _lerp(a: tuple[float, float, float], b: tuple[float, float, float], t: float) -> tuple[float, float, float]:
    return tuple(a[i] + (b[i] - a[i]) * t for i in range(3))


def _build_sky_dome(radius: float, rings: int, segments: int) -> NodePath:
    """Купол неба: вершини від трохи НИЖЧЕ горизонту (щоб не було прогалини при
    погляді вниз повз край землі) до зеніту, з градієнтом кольору за висотою."""
    fmt = GeomVertexFormat.getV3c4()
    vdata = GeomVertexData("sky", fmt, Geom.UHStatic)
    vdata.setNumRows((rings + 1) * (segments + 1))
    vtx = GeomVertexWriter(vdata, "vertex")
    col = GeomVertexWriter(vdata, "color")

    lat_min = -0.12 * math.pi  # трохи нижче горизонту
    lat_max = 0.5 * math.pi  # зеніт
    for ring in range(rings + 1):
        lat = lat_min + (lat_max - lat_min) * ring / rings
        t = max(0.0, math.sin(lat))  # 0 біля/під горизонтом -> 1 у зеніті
        r, g, b = _lerp(_HORIZON_COLOR, _ZENITH_COLOR, t)
        z = radius * math.sin(lat)
        cos_lat = math.cos(lat)
        for seg in range(segments + 1):
            lon = 2 * math.pi * seg / segments
            vtx.addData3(radius * cos_lat * math.cos(lon), radius * cos_lat * math.sin(lon), z)
            col.addData4(r, g, b, 1.0)

    tris = GeomTriangles(Geom.UHStatic)
    for ring in range(rings):
        for seg in range(segments):
            a = ring * (segments + 1) + seg
            b = a + 1
            c = a + (segments + 1)
            d = c + 1
            tris.addVertices(a, c, b)  # намотка так, щоб грані дивились УСЕРЕДИНУ купола
            tris.addVertices(b, c, d)
    tris.closePrimitive()

    geom = Geom(vdata)
    geom.addPrimitive(tris)
    node = GeomNode("sky_dome")
    node.addGeom(geom)
    return NodePath(node)


def _make_sun_texture(size: int = 64) -> Texture:
    """Радіальне світіння: білий центр -> прозорий край (альфа-спад)."""
    img = PNMImage(size, size, 4)
    center = (size - 1) / 2.0
    for y in range(size):
        for x in range(size):
            d = math.hypot(x - center, y - center) / center
            alpha = max(0.0, 1.0 - d) ** 1.6
            img.setXelA(x, y, _SUN_COLOR[0], _SUN_COLOR[1], _SUN_COLOR[2], float(alpha))
    tex = Texture("sun")
    tex.load(img)
    return tex


def add_sky(
    base,
    radius: float = 1400.0,
    rings: int = 16,
    segments: int = 24,
    sun_azimuth_deg: float = 45.0,
    sun_elevation_deg: float = 48.0,
    sun_size: float = 130.0,
) -> NodePath:
    """Додати купол неба + диск сонця до сцени ``base.render``. Повертає корінь
    неба (``removeNode()`` прибирає все разом)."""
    sky = base.render.attachNewNode("sky")
    # Позаду всієї сцени, без запису глибини/освітлення/туману/шейдера PBR —
    # інакше градієнт спотворився б PBR-освітленням, а туман "з'їв" би все небо.
    sky.setBin("background", 0)
    sky.setDepthWrite(False)
    sky.setLightOff(1)
    sky.setFogOff(1)
    sky.setShaderOff(1)
    sky.setTwoSided(True)  # видимий зсередини незалежно від намотки граней

    dome = _build_sky_dome(radius, rings, segments)
    dome.reparentTo(sky)

    # Сонце: яскравий диск (білборд-картка) у напрямку (азимут/висота) на
    # куполі. РАНІШЕ тут було ЩЕ й широке бліде "гало" (друга картка, scale
    # 3.2, статичне наближення об'єму) — ВИДАЛЕНО (доп. фаза "графіка 2.0",
    # прямий фідбек користувача: "справжні god rays замість гало+туману"),
    # замінено на СПРАВЖНІЙ динамічний пост-процес (render/godrays.py) —
    # реальний радіальний розмив маски сонця, що реагує на позицію камери й
    # потенційну заслону хмарами/рельєфом, а не статична картка.
    direction = sun_direction(sun_azimuth_deg, sun_elevation_deg)
    sun_tex = _make_sun_texture()
    for scale, alpha in ((1.0, 1.0),):  # лише яскравий диск
        cm = CardMaker("sun")
        cm.setFrame(-sun_size * scale, sun_size * scale, -sun_size * scale, sun_size * scale)
        disc = sky.attachNewNode(cm.generate())
        disc.setTexture(sun_tex)
        disc.setTransparency(TransparencyAttrib.MAlpha)
        disc.setBillboardPointEye()
        disc.setPos(*(direction * radius * 0.94))
        disc.setColor(Vec4(*_SUN_COLOR, alpha))
    return sky


def _make_cloud_texture(size: int = 128, seed: int = 0) -> Texture:
    """М'яка хмара: кілька накладених гаусових плям (білий центр -> прозорі краї)."""
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:size, 0:size].astype(float)
    alpha = np.zeros((size, size))
    for _ in range(rng.integers(4, 7)):
        cx, cy = rng.uniform(0.3, 0.7, 2) * size
        r = rng.uniform(0.12, 0.24) * size
        alpha += np.exp(-(((xx - cx) ** 2 + (yy - cy) ** 2) / (2 * r * r)))
    alpha = np.clip(alpha, 0.0, 1.0)
    # М'якший край + напівпрозорість, щоб хмари не були "ватяними" плитами.
    alpha = (alpha ** 1.3) * 0.85
    # Віньєтка: примусово гасимо альфу до 0 біля країв картки, інакше видно
    # прямокутний шов картки в небі (реальний артефакт на скріншоті).
    ax = np.clip(1.0 - (np.abs(np.linspace(-1, 1, size))) ** 3, 0.0, 1.0)
    alpha = alpha * ax[None, :] * ax[:, None]

    img = PNMImage(size, size, 4)
    for y in range(size):
        for x in range(size):
            img.setXelA(x, y, 1.0, 1.0, 1.0, float(alpha[y, x]))
    tex = Texture("cloud")
    tex.load(img)
    return tex


def add_clouds(base, count: int = 16, altitude: float = 180.0, spread: float = 900.0, seed: int = 3) -> NodePath:
    """Розсіяти напівпрозорі хмари-білборди високо над полем (реальний фідбек:
    "не вистачає хмар, об'ємності"). Кожна — м'яка картка, обернена до камери;
    без запису глибини, щоб не сваритися зі сценою. Повертає корінь хмар."""
    rng = np.random.default_rng(seed)
    clouds = base.render.attachNewNode("clouds")
    clouds.setBin("background", 1)  # після купола неба (0), але позаду сцени
    clouds.setDepthWrite(False)
    clouds.setLightOff(1)
    clouds.setFogOff(1)
    clouds.setShaderOff(1)
    clouds.setTransparency(TransparencyAttrib.MAlpha)
    tex = _make_cloud_texture()
    for _ in range(count):
        r = spread * math.sqrt(rng.random())
        theta = rng.uniform(0.0, 2 * math.pi)
        x, y = r * math.cos(theta), r * math.sin(theta)
        z = altitude + rng.uniform(-40.0, 60.0)
        w = rng.uniform(90.0, 190.0)
        cm = CardMaker("cloud")
        cm.setFrame(-w, w, -w * 0.5, w * 0.5)
        cloud = clouds.attachNewNode(cm.generate())
        cloud.setTexture(tex)
        cloud.setBillboardPointEye()
        cloud.setPos(x, y, z)
        cloud.setColor(Vec4(1.0, 1.0, 1.0, float(rng.uniform(0.5, 0.8))))
    return clouds


def add_fog(base, density_near_m: float = 130.0, density_far_m: float = 240.0) -> Fog:
    """Лінійний туман кольору горизонту (реальний фідбек: "як у реальності —
    трохи в даль синіло як димка, щоб зблизька її видно не було, лише до
    горизонту"): жодного ефекту до ``density_near_m`` (робоча зона польоту й
    сам трав'яний килим лишаються чіткими, без імли впритул), потім плавно
    (лінійно) наростає й повністю зливається з кольором горизонту до
    ``density_far_m`` — так само ховає й край площини землі (``build_terrain``,
    радіус 250м), тож жодного видимого "обриву"."""
    fog = Fog("world_fog")
    fog.setColor(Vec4(*_HORIZON_COLOR, 1.0))
    fog.setLinearRange(density_near_m, density_far_m)
    base.render.setFog(fog)
    return fog

"""Процедурна бібліотека СПРАВЖНІХ 3D-форм (доп. фаза "реальні моделі") —
на відміну від `render/visuals.py` (плоскі картки, `CardMaker`), тут —
повноцінні меші через `GeomVertexData`/`GeomTriangles`/`GeomNode` з
нормалями, для об'ємних форм (стовбури дерев, стволи гармат, колеса) із
коректним освітленням по вигнутій поверхні (перевірено емпірично).

Усе процедурне — без завантаження готових 3D-ассетів ззовні (як і решта
проєкту, докладніше docs/DECISIONS.md).

ЕТИКА (SKILL.md розділ 0, доп. фаза "реальні моделі"): тут НЕМАЄ будівників
для людських фігур — свідоме рішення, обговорене з користувачем. Технічна
сценерія (транспорт/укріплення/артилерія-як-обладнання) — стандартний
контент військових ігор/симуляторів, це не стосується.
"""

from __future__ import annotations

import functools
import math

import numpy as np
from panda3d.core import (
    Geom,
    GeomEnums,
    GeomNode,
    GeomTriangles,
    GeomVertexData,
    GeomVertexFormat,
    GeomVertexWriter,
    Material,
    NodePath,
    Vec4,
)

from dronesim.render import materials as mat
from dronesim.render.model_assets import load_model_asset
from dronesim.render.visuals import add_box_visual

_DIRT_COLOR = (0.35, 0.28, 0.2, 1.0)
_TRUNK_COLOR = (0.32, 0.22, 0.12, 1.0)
_FOLIAGE_COLOR = (0.16, 0.32, 0.14, 1.0)


def _mesh_from_triangles(vertices: list[tuple[float, float, float]], normals: list[tuple[float, float, float]], tri_indices: list[tuple[int, int, int]], name: str) -> NodePath:
    fmt = GeomVertexFormat.getV3n3()
    vdata = GeomVertexData(name, fmt, Geom.UHStatic)
    vdata.setNumRows(len(vertices))
    vertex_writer = GeomVertexWriter(vdata, "vertex")
    normal_writer = GeomVertexWriter(vdata, "normal")
    for v, n in zip(vertices, normals, strict=True):
        vertex_writer.addData3(*v)
        normal_writer.addData3(*n)

    tris = GeomTriangles(Geom.UHStatic)
    for a, b, c in tri_indices:
        tris.addVertices(a, b, c)
    tris.closePrimitive()

    geom = Geom(vdata)
    geom.addPrimitive(tris)
    node = GeomNode(name)
    node.addGeom(geom)
    return NodePath(node)


def build_cylinder_mesh(
    radius: float,
    height: float,
    segments: int = 12,
    color: tuple[float, float, float, float] = (0.6, 0.6, 0.6, 1.0),
    material: Material | None = None,
) -> NodePath:
    """Циліндр (вісь Z, база в z=0) з боковою поверхнею + верхньою/нижньою
    кришкою — стовбури дерев, стволи гармат/артилерії, колеса техніки."""
    vertices: list[tuple[float, float, float]] = []
    normals: list[tuple[float, float, float]] = []
    tris: list[tuple[int, int, int]] = []

    # Бічна поверхня: 2 кільця вершин (низ/верх), нормалі — радіальні назовні.
    for i in range(segments):
        angle = 2 * math.pi * i / segments
        x, y = radius * math.cos(angle), radius * math.sin(angle)
        n = (math.cos(angle), math.sin(angle), 0.0)
        vertices.append((x, y, 0.0))
        normals.append(n)
        vertices.append((x, y, height))
        normals.append(n)
    for i in range(segments):
        i2 = (i + 1) % segments
        b0, t0, b1, t1 = i * 2, i * 2 + 1, i2 * 2, i2 * 2 + 1
        tris.append((b0, t0, b1))
        tris.append((b1, t0, t1))

    # Кришки: віяло трикутників від центру (окремі вершини — нормаль ±Z, не
    # ділиться з бічною поверхнею, інакше освітлення кришки було б неправильним).
    base_center_i = len(vertices)
    vertices.append((0.0, 0.0, 0.0))
    normals.append((0.0, 0.0, -1.0))
    top_center_i = len(vertices)
    vertices.append((0.0, 0.0, height))
    normals.append((0.0, 0.0, 1.0))

    base_ring_start = len(vertices)
    for i in range(segments):
        angle = 2 * math.pi * i / segments
        vertices.append((radius * math.cos(angle), radius * math.sin(angle), 0.0))
        normals.append((0.0, 0.0, -1.0))
    top_ring_start = len(vertices)
    for i in range(segments):
        angle = 2 * math.pi * i / segments
        vertices.append((radius * math.cos(angle), radius * math.sin(angle), height))
        normals.append((0.0, 0.0, 1.0))

    for i in range(segments):
        i2 = (i + 1) % segments
        tris.append((base_center_i, base_ring_start + i2, base_ring_start + i))  # низ: за годинниковою (нормаль -Z)
        tris.append((top_center_i, top_ring_start + i, top_ring_start + i2))  # верх: проти годинникової (нормаль +Z)

    mesh = _mesh_from_triangles(vertices, normals, tris, "cylinder")
    mesh.setColor(Vec4(*color))
    mesh.setMaterial(material if material is not None else mat.GENERIC)
    return mesh


def build_cone_mesh(
    radius: float,
    height: float,
    segments: int = 10,
    color: tuple[float, float, float, float] = (0.16, 0.32, 0.14, 1.0),
    material: Material | None = None,
) -> NodePath:
    """Конус (вісь Z, база в z=0, вершина в z=height) — крони дерев (хвойний
    силует), дешевший за повну теселяцію сфери."""
    vertices: list[tuple[float, float, float]] = []
    normals: list[tuple[float, float, float]] = []
    tris: list[tuple[int, int, int]] = []

    # Нормаль бічної поверхні конуса має нахил (не суто радіальна, як у
    # циліндра) — стандартна формула (cos, sin, radius/height), нормована.
    slant_z = radius / height if height > 1e-9 else 0.0
    slant_norm = math.hypot(1.0, slant_z)

    apex_i = len(vertices)
    for i in range(segments):
        angle = 2 * math.pi * i / segments
        vertices.append((0.0, 0.0, height))
        normals.append((math.cos(angle) / slant_norm, math.sin(angle) / slant_norm, slant_z / slant_norm))
    # (кожна грань апекса має ВЛАСНУ вершину-копію з нормаллю ЦІЄЇ грані — інакше
    # спільна вершина апекса усереднила б нормалі сусідніх граней, ламаючи фасетне освітлення)
    base_start = len(vertices)
    for i in range(segments):
        angle = 2 * math.pi * i / segments
        vertices.append((radius * math.cos(angle), radius * math.sin(angle), 0.0))
        normals.append((math.cos(angle) / slant_norm, math.sin(angle) / slant_norm, slant_z / slant_norm))

    for i in range(segments):
        i2 = (i + 1) % segments
        tris.append((apex_i + i, base_start + i, base_start + i2))

    base_center_i = len(vertices)
    vertices.append((0.0, 0.0, 0.0))
    normals.append((0.0, 0.0, -1.0))
    cap_start = len(vertices)
    for i in range(segments):
        angle = 2 * math.pi * i / segments
        vertices.append((radius * math.cos(angle), radius * math.sin(angle), 0.0))
        normals.append((0.0, 0.0, -1.0))
    for i in range(segments):
        i2 = (i + 1) % segments
        tris.append((base_center_i, cap_start + i2, cap_start + i))

    mesh = _mesh_from_triangles(vertices, normals, tris, "cone")
    mesh.setColor(Vec4(*color))
    mesh.setMaterial(material if material is not None else mat.FOLIAGE)
    return mesh


_BIRCH_TRUNK_COLOR = (0.78, 0.76, 0.72, 1.0)  # блідо-біла кора
_BIRCH_FOLIAGE_COLOR = (0.42, 0.56, 0.24, 1.0)  # світліше/жовтіше зелене за хвою
_OAK_FOLIAGE_COLOR = (0.20, 0.38, 0.15, 1.0)  # темніше/насиченіше за хвою, округла крона


#  Висота стовбура, з якою кожен вид був ЗАПЕЧЕНИЙ у Blender
# (``assets/blender/build_trees.py``) — потрібна, щоб масштабувати ЦІЛИЙ
# запечений меш до довільного ``trunk_height`` виклику (одна бакова модель
# на вид, БЕЗ окремих параметрів "висота стовбура"/"радіус крони" нарізно,
# на відміну від процедурного fallback нижче).
_BAKED_TREE_HEIGHT = {"pine": 3.4, "oak": 2.2, "birch": 3.7}

# Референсні (length, width, wall_height) з якими запечено
# assets/models/trench.glb (assets/blender/build_trench.py) — build_trench
# масштабує запитані розміри ВІДНОСНО цих значень (не uniform, як дерево:
# на практиці лише length відрізняється між усіма наявними usage, width/
# wall_height завжди рівні референсу, докладніше docs/DECISIONS.md).
_BAKED_TRENCH_DIMS = (8.0, 2.4, 1.0)


def build_tree(
    parent: NodePath,
    trunk_height: float = 2.5,
    trunk_radius: float = 0.15,
    foliage_radius: float = 1.3,
    species: str = "pine",
    seed: int = 0,
) -> NodePath:
    """Дерево — спершу пробує ДЕТАЛЬНУ Blender-авторовану модель
    (``assets/models/tree_{species}.bam`` — органічний стовбур із
    радіальним джиттером/нахилом, лапата крона з перекритих ікосфер,
    гладке затінення; доп. фаза "реальні моделі",
    ``assets/blender/build_trees.py``), рівномірно масштабовану під
    ``trunk_height``. Якщо файл відсутній — падає на процедурний білдер
    (3 види силуету: ``"pine"`` хвойне, ``"oak"`` широка округла крона,
    ``"birch"`` тонкий блідий стовбур — реальний фідбек: "дерева як
    іграшково намальовані, різні види дерев")."""
    detailed = load_model_asset(parent, f"tree_{species}")
    if detailed is not None:
        base_height = _BAKED_TREE_HEIGHT.get(species, trunk_height)
        scale = trunk_height / base_height if base_height > 0 else 1.0
        detailed.setScale(scale)
        return detailed
    builders = {"pine": _build_pine_tree, "oak": _build_oak_tree, "birch": _build_birch_tree}
    builder = builders.get(species, _build_pine_tree)
    return builder(parent, trunk_height, trunk_radius, foliage_radius, seed)


def _build_pine_tree(parent, trunk_height, trunk_radius, foliage_radius, seed):
    """Хвойне дерево: циліндр-стовбур + 3 конуси-крона зменшуваного розміру
    (пишніший силует) — процедурна заміна коробки-заглушки. Кожен ярус
    трохи іншого зеленого відтінку, щоб крона не була однотонною."""
    tree = parent.attachNewNode("tree_pine")

    trunk = build_cylinder_mesh(trunk_radius, trunk_height, segments=8, color=_TRUNK_COLOR, material=mat.WOOD)
    trunk.reparentTo(tree)

    # 3 яруси крони: більший знизу -> менший угорі, кожен вищий за попередній.
    tiers = ((1.0, 1.9, 0.5, 0.95), (0.78, 1.55, 0.5 + 0.9, 1.05), (0.55, 1.25, 0.5 + 1.7, 1.15))
    for r_scale, h_scale, z_off, shade in tiers:
        cone = build_cone_mesh(
            foliage_radius * r_scale,
            foliage_radius * h_scale,
            segments=10,
            color=(_FOLIAGE_COLOR[0] * shade, _FOLIAGE_COLOR[1] * shade, _FOLIAGE_COLOR[2] * shade, 1.0),
        )
        cone.reparentTo(tree)
        cone.setPos(0, 0, trunk_height * z_off)

    return tree


def _build_oak_tree(parent, trunk_height, trunk_radius, foliage_radius, seed):
    """Листяне дерево: коротший товщий стовбур + ШИРОКА округла крона —
    кластер приплюснутих конусів (низька висота/радіус -> ближче до
    напівсфери, ніж до хвойного шпиля), трохи розкиданих для органічної
    несиметричності."""
    tree = parent.attachNewNode("tree_oak")

    trunk_h = trunk_height * 0.7
    trunk_r = trunk_radius * 1.4
    trunk = build_cylinder_mesh(trunk_r, trunk_h, segments=8, color=_TRUNK_COLOR, material=mat.WOOD)
    trunk.reparentTo(tree)

    rng = np.random.default_rng(seed * 131 + 7)
    canopy_r = foliage_radius * 1.3
    canopy_h = canopy_r * 0.85  # приплюснутий конус -> округліший силует
    clumps = ((0.0, 0.0, 1.0), (0.35, 0.25, 0.72), (-0.3, 0.3, 0.7), (0.1, -0.35, 0.68))
    for i, (dx, dy, scale) in enumerate(clumps):
        shade = float(rng.uniform(0.9, 1.1))
        cone = build_cone_mesh(
            canopy_r * scale,
            canopy_h * scale,
            segments=10,
            color=(_OAK_FOLIAGE_COLOR[0] * shade, _OAK_FOLIAGE_COLOR[1] * shade, _OAK_FOLIAGE_COLOR[2] * shade, 1.0),
        )
        cone.reparentTo(tree)
        cone.setPos(dx * canopy_r * 0.5, dy * canopy_r * 0.5, trunk_h * 0.85 + i * 0.05)

    return tree


def _build_birch_tree(parent, trunk_height, trunk_radius, foliage_radius, seed):
    """Береза: тонкий блідий стовбур, вища й рідша крона (менший радіус за
    хвойне/дубове), світліше-жовтогаряче листя."""
    tree = parent.attachNewNode("tree_birch")

    trunk_h = trunk_height * 1.15
    trunk_r = trunk_radius * 0.6
    trunk = build_cylinder_mesh(trunk_r, trunk_h, segments=6, color=_BIRCH_TRUNK_COLOR, material=mat.WOOD)
    trunk.reparentTo(tree)

    foliage_r = foliage_radius * 0.65
    tiers = ((1.0, 1.6, 0.55), (0.7, 1.3, 0.55 + 0.75))
    for r_scale, h_scale, z_off in tiers:
        cone = build_cone_mesh(foliage_r * r_scale, foliage_r * h_scale, segments=8, color=_BIRCH_FOLIAGE_COLOR)
        cone.reparentTo(tree)
        cone.setPos(0, 0, trunk_h * z_off)

    return tree


def _build_blade_field_mesh(
    parent: NodePath, name: str, x, y, height, width, heading, texture, tint=None, base_z=None
) -> NodePath:
    """Один Geom з УСІМА хрестоподібними стеблами (трава/пшениця), побудований
    ВЕКТОРИЗОВАНО через numpy + прямий блит у буфери Panda3D — на порядки
    швидше за будь-який Python-цикл.

    ЕВОЛЮЦІЯ (для контексту, docs/DECISIONS.md): (1) NodePath-на-пучок +
    ``flattenStrong()`` — O(n²), 4000 пучків = 36с; (2) один Geom через
    ``GeomVertexWriter`` у Python-циклі — лінійно, але ~2M викликів на
    десятки тисяч стебел = кілька секунд; (3) ТУТ — numpy рахує ВСІ вершини
    масивами й блитить одним ``copySubdataFrom`` (той самий прийом, що
    рекомендує документація Panda3D для великих процедурних мешів): 100k+
    стебел за десятки мс, тож густий КИЛИМ трави (реальний фідбек: "трава —
    це шар майже усюди, а не поодинокі об'єкти") став можливим.

    ``x/y/height/width/heading`` — numpy-масиви (N,), по одному на стебло;
    кожне стебло — 3 хрестоподібні картки (0°/60°/120°). ``tint`` — опційний
    масив (N,3) множник кольору на стебло (легка варіація яскравості/відтінку
    для реалізму); ``None`` -> без вершинного кольору. ``base_z`` — опційний
    масив (N,) висоти ЗЕМЛІ під стеблом (реальний рельєф, ``_terrain_height``)
    — щоб трава на схилі не "плавала" над ним; ``None`` -> 0 (рівна земля)."""
    from panda3d.core import TransparencyAttrib

    x = np.asarray(x, dtype=np.float64)
    n = len(x)
    if n == 0:
        return parent.attachNewNode(GeomNode(name))
    y = np.asarray(y, dtype=np.float64)
    height = np.asarray(height, dtype=np.float64)
    width = np.asarray(width, dtype=np.float64)
    heading = np.asarray(heading, dtype=np.float64)
    base_z = np.zeros(n, dtype=np.float64) if base_z is None else np.asarray(base_z, dtype=np.float64)

    ncards = n * 3
    # Кут кожної картки: heading стебла + {0, 120, 240}°.
    ang = (heading[:, None] + np.array([0.0, 2 * math.pi / 3, 4 * math.pi / 3])[None, :]).ravel()
    cx = np.repeat(x, 3)
    cy = np.repeat(y, 3)
    ch = np.repeat(height, 3)
    cw = np.repeat(width, 3)
    cbz = np.repeat(base_z, 3)
    cos_a, sin_a = np.cos(ang), np.sin(ang)

    # 4 кути картки: локальний x-зсув (±w/2) і локальна висота (0 або h),
    # плюс висота землі під стеблом (рельєф).
    corner_lx = np.array([-0.5, 0.5, 0.5, -0.5])
    corner_lz = np.array([0.0, 0.0, 1.0, 1.0])
    lx = cw[:, None] * corner_lx[None, :]  # (ncards, 4)
    lz = ch[:, None] * corner_lz[None, :] + cbz[:, None]
    wx = cx[:, None] + lx * cos_a[:, None]
    wy = cy[:, None] + lx * sin_a[:, None]

    # UV: кути картки низ-ліво(0,0) низ-право(1,0) верх-право(1,1) верх-ліво(0,1).
    uv_u = np.array([0.0, 1.0, 1.0, 0.0])
    uv_v = np.array([0.0, 0.0, 1.0, 1.0])
    has_color = tint is not None

    if has_color:
        # Формат V3n3c4t2 — color як 4 УПАКОВАНІ байти (stride 36, НЕ 4 float32;
        # перевірено емпірично getColumn().getTotalBytes()==4, порядок RGBA).
        # Тому будуємо СТРУКТУРОВАНИЙ numpy-масив із точним розкладом байтів,
        # а не плоский float32 (наївний float-блит зсував би кожну вершину й
        # давав "смуги" через увесь екран — реальний баг зі скріншота).
        dt = np.dtype([("pos", "<f4", 3), ("normal", "<f4", 3), ("color", "u1", 4), ("uv", "<f4", 2)])
        v = np.zeros((ncards, 4), dtype=dt)
        v["pos"][:, :, 0] = wx
        v["pos"][:, :, 1] = wy
        v["pos"][:, :, 2] = lz
        v["normal"][:, :, 2] = 1.0
        card_tint = np.clip(np.repeat(np.asarray(tint, dtype=np.float64), 3, axis=0) * 255.0, 0, 255).astype(np.uint8)
        v["color"][:, :, 0] = card_tint[:, None, 0]
        v["color"][:, :, 1] = card_tint[:, None, 1]
        v["color"][:, :, 2] = card_tint[:, None, 2]
        v["color"][:, :, 3] = 255
        v["uv"][:, :, 0] = uv_u[None, :]
        v["uv"][:, :, 1] = uv_v[None, :]
        raw = v.reshape(ncards * 4).tobytes()
        fmt = GeomVertexFormat.getV3n3c4t2()
    else:
        verts = np.zeros((ncards, 4, 8), dtype=np.float32)  # v3 n3 t2
        verts[:, :, 0] = wx
        verts[:, :, 1] = wy
        verts[:, :, 2] = lz
        verts[:, :, 5] = 1.0
        verts[:, :, 6] = uv_u[None, :]
        verts[:, :, 7] = uv_v[None, :]
        raw = verts.reshape(ncards * 4, 8).tobytes()
        fmt = GeomVertexFormat.getV3n3t2()

    # Нормаль ВГОРУ (0,0,1) для всіх — трюк ігрової рослинності: вертикальні
    # картки з up-нормаллю освітлюються як земля під ними (з геометричними
    # нормалями частина карток дивилась би від сонця й рендерилась чорною —
    # реальний баг, docs/DECISIONS.md).
    base = (np.arange(ncards) * 4)[:, None]
    idx = (base + np.array([0, 1, 2, 0, 2, 3])[None, :]).ravel().astype(np.uint32)

    vdata = GeomVertexData(name, fmt, Geom.UHStatic)
    vdata.uncleanSetNumRows(ncards * 4)
    vdata.modifyArray(0).modifyHandle().copySubdataFrom(0, len(raw), raw)

    prim = GeomTriangles(Geom.UHStatic)
    prim.setIndexType(GeomEnums.NT_uint32)
    iarr = prim.modifyVertices()
    iarr.uncleanSetNumRows(len(idx))
    iarr.modifyHandle().copySubdataFrom(0, idx.nbytes, idx.tobytes())

    geom = Geom(vdata)
    geom.addPrimitive(prim)
    node = GeomNode(name)
    node.addGeom(geom)
    result = parent.attachNewNode(node)
    result.setTexture(texture)
    # MBinary (alpha-TEST), НЕ MAlpha (alpha-BLEND): для КИЛИМА з десятків
    # тисяч стебел alpha-blend вимагав би посортувати за глибиною ЩОКАДРУ
    # (O(n log n) на CPU) + давав би overdraw/артефакти порядку — непридатно.
    # Alpha-test трактує стебло як НЕПРОЗОРЕ з викидом пікселів alpha<поріг:
    # без сортування, z-буфер сам усе впорядковує (стандарт ігрової трави).
    result.setTransparency(TransparencyAttrib.MBinary)
    result.setTwoSided(True)
    result.setMaterial(mat.FOLIAGE)
    return result


_GRASS_BLADE_TEXTURE_CACHE: dict[int, object] = {}


def _make_grass_blade_image(size: int, seed: int, base_color=(0.16, 0.32, 0.11), tip_color=(0.45, 0.62, 0.24), n_strands: int = 8):
    """Текстура-КЛУМП: кілька (``n_strands``) тонких стеблин поруч із
    ПРОЗОРИМИ проміжками, а не одне суцільне "листя" на всю картку (реальний
    фідбек після першого килима: кожна картка виглядала як суцільний
    зелений конус, а не трава). Кожна картка тепер = пучок трави, і
    перекриття карток зливається в СУЦІЛЬНИЙ килим замість "поля шпилів".
    ``base_color``/``tip_color`` параметризовано (та сама форма — для пшениці).

    Векторизовано (numpy): проміжки прозорі -> alpha-test (MBinary) дає
    чіткі тонкі стеблини без сортування."""
    from panda3d.core import PNMImage

    rng = np.random.default_rng(seed)
    img = PNMImage(size, size, 4)
    base_c = np.clip(np.array(base_color), 0.0, 1.0)
    tip_c = np.clip(np.array(tip_color), 0.0, 1.0)

    yy = np.arange(size)
    v = (size - 1 - yy) / (size - 1)  # 0=низ(база), 1=верх(кінчик) — див. коментар нижче
    # PNMImage рядок y=0 — ВЕРХ; CardMaker мапить v=0 на НИЗ картки
    # (перевірено емпірично, docs/DECISIONS.md), тож v обернено відносно y.

    alpha = np.zeros((size, size))
    rgb = np.zeros((size, size, 3))
    xx = np.arange(size)[None, :]
    for _ in range(n_strands):
        cx0 = rng.uniform(0.15, 0.85) * size  # позиція стеблини по ширині
        bend = rng.uniform(-0.25, 0.25)
        top = rng.uniform(0.72, 1.0)  # частина висоти, де стеблина закінчується
        strand_w = rng.uniform(0.06, 0.12) * size  # напівширина стеблини (пікселі) біля бази
        bright = rng.uniform(0.8, 1.18)
        center = cx0 + bend * (v**2) * size * 0.5
        width_px = np.maximum(0.5, strand_w * (1.0 - v / max(top, 1e-6)))  # звужується до кінчика
        width_px = np.where(v <= top, width_px, 0.0)  # обрізати вище кінчика
        dist = np.abs(xx - center[:, None])
        a = np.clip((width_px[:, None] - dist) / 1.2, 0.0, 1.0)
        col = (base_c[None, None, :] * (1.0 - v[:, None, None]) + tip_c[None, None, :] * v[:, None, None]) * bright
        take = a > alpha  # ближча стеблина перекриває
        alpha = np.where(a > alpha, a, alpha)
        rgb = np.where(take[..., None], col, rgb)

    for y in range(size):
        for x in range(size):
            img.setXelA(x, y, float(rgb[y, x, 0]), float(rgb[y, x, 1]), float(rgb[y, x, 2]), float(alpha[y, x]))
    return img


def _make_grass_blade_texture(size: int = 32, seed: int = 0):
    """Текстура ОДНОГО стебла трави: альфа-виріз тейперованого силуету (широка
    темніша база -> вузький світліший кінчик, з легким вигином) — реальна
    форма стебла на прозорому тлі, а не суцільний зелений прямокутник
    (реальний фідбек: "реальну траву а не просто розкраску"). Стандартна
    техніка білборд-трави (той самий принцип, що й процедурні текстури землі/
    дороги/фасаду, render/visuals.py::_make_grass_texture)."""
    from panda3d.core import Texture

    if seed in _GRASS_BLADE_TEXTURE_CACHE:
        return _GRASS_BLADE_TEXTURE_CACHE[seed]

    tex = Texture(f"grass_blade_{seed}")
    tex.load(_make_grass_blade_image(size, seed))
    _GRASS_BLADE_TEXTURE_CACHE[seed] = tex
    return tex


def build_grass_tuft(parent: NodePath, height: float = 0.35, seed: int = 0) -> NodePath:
    """Пучок трави: 3 перехресні картки (як кущ, render/models.py::build_bush)
    з АЛЬФА-ВИРІЗАНОЮ текстурою стебла (не суцільний колір) — той самий
    прийом монтування, що й гвинти апарата (render/propellers.py), стандартна
    ігрова техніка білборд-трави."""
    from panda3d.core import CardMaker, TransparencyAttrib

    tuft = parent.attachNewNode("grass_tuft")
    tex = _make_grass_blade_texture(seed=seed % 7)
    width = height * 0.45
    for heading in (0, 60, 120):
        cm = CardMaker("blade")
        cm.setFrame(-width / 2, width / 2, 0.0, height)
        blade = tuft.attachNewNode(cm.generate())
        blade.setH(heading)
        blade.setTexture(tex)
        # ВАЖЛИВО: transparency/material — на ЛИСТОВОМУ вузлі картки, НЕ на
        # обгортці ``tuft``. Емпірично виявлено: ``NodePath.flattenStrong()``
        # (``scatter_ground_cover`` зводить сотні пучків в один вузол для
        # продуктивності) ВТРАЧАЄ ``TransparencyAttrib``, якщо його
        # встановлено на вузлі-ОБГОРТЦІ, що зникає при флатенізації — на
        # рівні самої картки атрибут коректно зберігається в результуючому
        # Geom. Без цього трава рендерилась суцільними непрозорими
        # прямокутниками замість силуету стебла (реальний баг, знайдений
        # скріншотом after flatten).
        blade.setTransparency(TransparencyAttrib.MAlpha)
        blade.setTwoSided(True)
        blade.setMaterial(mat.FOLIAGE)
    return tuft


_WHEAT_BLADE_TEXTURE_CACHE: dict[int, object] = {}


def _make_wheat_blade_texture(size: int = 32, seed: int = 0):
    """Стебло пшениці — та сама форма/альфа-виріз, що й трава, лише золотисто-
    жовта гама (реальний фідбек: "поле зі пшеницею не вистачає")."""
    from panda3d.core import Texture

    if seed in _WHEAT_BLADE_TEXTURE_CACHE:
        return _WHEAT_BLADE_TEXTURE_CACHE[seed]

    img = _make_grass_blade_image(size, seed, base_color=(0.55, 0.42, 0.12), tip_color=(0.85, 0.72, 0.28))
    tex = Texture(f"wheat_blade_{seed}")
    tex.load(img)
    _WHEAT_BLADE_TEXTURE_CACHE[seed] = tex
    return tex


def build_wheat_tuft(parent: NodePath, height: float = 0.55, seed: int = 0) -> NodePath:
    """Стебло пшениці: вищий і тонший за траву силует (3 перехресні картки),
    золотисто-жовта альфа-вирізана текстура."""
    from panda3d.core import CardMaker, TransparencyAttrib

    tuft = parent.attachNewNode("wheat_tuft")
    tex = _make_wheat_blade_texture(seed=seed % 7)
    width = height * 0.3  # тонше за траву
    for heading in (0, 60, 120):
        cm = CardMaker("wheat_blade")
        cm.setFrame(-width / 2, width / 2, 0.0, height)
        blade = tuft.attachNewNode(cm.generate())
        blade.setH(heading)
        blade.setTexture(tex)
        blade.setTransparency(TransparencyAttrib.MAlpha)  # див. коментар у build_grass_tuft
        blade.setTwoSided(True)
        blade.setMaterial(mat.FOLIAGE)
    return tuft


def scatter_wheat_patch(
    parent: NodePath,
    center: tuple[float, float] = (0.0, 0.0),
    size: tuple[float, float] = (12.0, 12.0),
    count: int = 2500,
    seed: int = 13,
    angle_deg: float = 0.0,
    row_spacing: float = 0.42,
) -> NodePath:
    """Пшеничне поле, посаджене РЯДКАМИ (реальний фідбек: "поле — це не
    просто трава, воно посаджене рядочками спеціальним комбайном") —
    прямі паралельні ряди вздовж довгої осі ``size`` (``length, width``),
    повернені на ``angle_deg`` (``_voronoi_grouped_regions`` дає напрямок
    головної осі поля). Кожен ряд — рівномірний крок стебел уздовж довжини,
    ряди рознесені на ``row_spacing`` УПОПЕРЕК; невеликий джиттер (не
    ідеально робот-рівно, як і в реальному полі через нерівності ґрунту), але
    рядкова структура ЧІТКО читається — на відміну від чистого random
    розкиду. ОДИН векторизований Geom (``_build_blade_field_mesh`` —
    numpy-блит)."""
    rng = np.random.default_rng(seed)
    patch = parent.attachNewNode("wheat_patch")
    cx, cy = center
    length, width = size
    plant_spacing = max(0.10, length * width / max(count, 1) / max(row_spacing, 0.05))
    n_rows = max(1, int(width / row_spacing))
    n_per_row = max(1, int(length / plant_spacing))
    row_i, col_i = np.meshgrid(np.arange(n_rows), np.arange(n_per_row), indexing="ij")
    lx = (col_i.ravel() + 0.5) * plant_spacing - length / 2.0
    ly = (row_i.ravel() + 0.5) * row_spacing - width / 2.0
    n = lx.size
    # Легкий джиттер (ґрунт нерівний), рядки лишаються чітко читаними.
    lx = lx + rng.uniform(-plant_spacing * 0.25, plant_spacing * 0.25, n)
    ly = ly + rng.uniform(-row_spacing * 0.2, row_spacing * 0.2, n)
    ang = math.radians(angle_deg)
    ca, sa = math.cos(ang), math.sin(ang)
    x = cx + lx * ca - ly * sa
    y = cy + lx * sa + ly * ca
    heading = np.full(n, ang) + rng.uniform(-0.08, 0.08, n)  # стебла вздовж рядка, з ледь помітним розкидом
    height = rng.uniform(0.45, 0.65, n)
    width_blade = height * 0.3
    bright = rng.uniform(0.82, 1.12, n)[:, None]
    tint = np.clip(bright * np.array([1.0, 1.0, 1.0])[None, :], 0.0, 1.3)
    _build_blade_field_mesh(patch, "wheat_field", x, y, height, width_blade, heading,
                            _make_wheat_blade_texture(seed=0), tint=tint)
    return patch


_ROCK_COLOR = (0.42, 0.40, 0.38, 1.0)


def build_rock(radius: float = 0.3, seed: int = 0, rings: int = 6, segments: int = 8) -> NodePath:
    """Камінець: неправильна "грудкувата" куля — реальний фідбек: "немає
    камінців". Приплюснутий знизу, щоб природно "стояти" на землі, не як
    ідеальна куля.

    ПЕРШИЙ ВАРІАНТ рендерився ПОВНІСТЮ ЧОРНИМ (виявлено скріншотом, не
    очевидно з коду): нормаль рахувалась НАБЛИЖЕНО (``vertex - довільна
    точка``) для форми, що звужується І знизу, І зверху (повна куля, не
    купол, як ``build_hill``) — той самий сталий "winding" трикутників, що
    коректно дає нормаль НАЗОВНІ для МОНОТОННОГО купола (build_hill: радіус
    лише спадає), для куль (радіус росте, потім спадає) даватиме
    ПРАВИЛЬНУ нормаль лише в одній половині. Виправлено: справжня
    аналітична сферична нормаль (``позиціяـвершини / довжина``, ТОЧНА для
    сфери, а не наближена), і приплюснутість зроблено ЗОВНІ — масштабом
    вузла по Z (``setScale``), не ручним псуванням вершин: Panda3D коректно
    перераховує normal-матрицю (обернено-транспоновану) для НЕРІВНОМІРНОГО
    масштабу, тож нормалі лишаються коректними після сплющення, на відміну
    від ручного зсуву z-координати вершини без відповідного перерахунку
    нормалі."""
    rng = np.random.default_rng(seed)
    vertices: list[tuple[float, float, float]] = []
    normals: list[tuple[float, float, float]] = []
    tris: list[tuple[int, int, int]] = []

    for ring in range(rings + 1):
        v = ring / rings  # 0=низ (полюс), 1=верх (полюс)
        lat = -math.pi / 2 + v * math.pi  # -90°..+90°
        cos_lat, sin_lat = math.cos(lat), math.sin(lat)
        jitter = float(rng.uniform(0.85, 1.15))  # неправильність форми (на радіус, НЕ на нормаль)
        for seg in range(segments):
            lon = 2 * math.pi * seg / segments
            # Точна сферична нормаль (одинична, до джиттеру) — незалежна від
            # неправильності радіуса, тож завжди коректно спрямована назовні.
            nx, ny, nz = cos_lat * math.cos(lon), cos_lat * math.sin(lon), sin_lat
            r = radius * jitter
            vertices.append((r * cos_lat * math.cos(lon), r * cos_lat * math.sin(lon), r * sin_lat + radius))
            normals.append((nx, ny, nz))

    for ring in range(rings):
        for seg in range(segments):
            s2 = (seg + 1) % segments
            a = ring * segments + seg
            b = ring * segments + s2
            c = (ring + 1) * segments + seg
            d = (ring + 1) * segments + s2
            tris.append((a, c, b))
            tris.append((b, c, d))

    mesh = _mesh_from_triangles(vertices, normals, tris, "rock")
    shade = float(rng.uniform(0.85, 1.15))
    mesh.setColor(Vec4(_ROCK_COLOR[0] * shade, _ROCK_COLOR[1] * shade, _ROCK_COLOR[2] * shade, 1.0))
    mesh.setMaterial(mat.DIRT)
    mesh.setTwoSided(True)  # запобіжник: не залежати від сталості winding по всій сфері
    # Приплюснути масштабом ВІД ПОЧАТКУ КООРДИНАТ (база вже в z=0 -> лишається
    # в z=0 після масштабування, бо scale навколо origin не рухає точки НА origin).
    mesh.setScale(1.0, 1.0, 0.55)
    return mesh


def scatter_rocks(parent: NodePath, count: int = 40, radius: float = 70.0, seed: int = 11, exclude_radius: float = 4.0) -> NodePath:
    """Розсіяти камінці по полю (той самий підхід, що й ``scatter_ground_cover``)
    — суто візуальні (без фізики), зведені в один вузол для продуктивності."""
    rng = np.random.default_rng(seed)
    rocks = parent.attachNewNode("rocks")
    span = radius - exclude_radius
    for i in range(count):
        r = exclude_radius + span * math.sqrt(rng.random())
        theta = rng.uniform(0.0, 2 * math.pi)
        x, y = r * math.cos(theta), r * math.sin(theta)
        rock = build_rock(radius=float(rng.uniform(0.08, 0.35)), seed=int(rng.integers(0, 10_000)))
        rock.reparentTo(rocks)
        rock.setPos(float(x), float(y), 0.0)
        rock.setH(float(rng.uniform(0.0, 360.0)))
    rocks.flattenStrong()
    return rocks


_BUSH_TEXTURE_CACHE: dict[int, object] = {}


def _make_bush_texture(size: int = 64, seed: int = 0):
    """Округлий листяний силует куща: накладені гаусові плями зелені з
    альфа-вирізом (та сама техніка, що й хмари, render/sky.py) — без цього
    кущ рендерився ПРЯМОКУТНОЮ плитою картки (реальний артефакт скріншота)."""
    from panda3d.core import PNMImage, Texture

    if seed in _BUSH_TEXTURE_CACHE:
        return _BUSH_TEXTURE_CACHE[seed]

    rng = np.random.default_rng(seed * 61 + 3)
    yy, xx = np.mgrid[0:size, 0:size].astype(float)
    alpha = np.zeros((size, size))
    # Плями зміщені донизу картки (кущ "сидить" на землі, не висить по центру).
    for _ in range(6):
        cx = rng.uniform(0.25, 0.75) * size
        cy = rng.uniform(0.45, 0.8) * size
        r = rng.uniform(0.14, 0.26) * size
        alpha += np.exp(-(((xx - cx) ** 2 + (yy - cy) ** 2) / (2 * r * r)))
    alpha = np.clip(alpha, 0.0, 1.0) ** 1.2
    # Крайова віньєтка (та сама пастка, що й у хмар: без гасіння альфи на
    # краях видно прямокутний шов картки).
    edge = np.clip(1.0 - (np.abs(np.linspace(-1, 1, size))) ** 3, 0.0, 1.0)
    alpha = alpha * edge[None, :] * edge[:, None]

    img = PNMImage(size, size, 4)
    base = np.array([0.16, 0.30, 0.12])
    for y in range(size):
        for x in range(size):
            shade = 0.85 + 0.3 * alpha[y, x]  # центр щільніший -> світліший
            c = np.clip(base * shade, 0.0, 1.0)
            img.setXelA(x, y, float(c[0]), float(c[1]), float(c[2]), float(min(1.0, alpha[y, x] * 1.6)))
    tex = Texture(f"bush_{seed}")
    tex.load(img)
    _BUSH_TEXTURE_CACHE[seed] = tex
    return tex


def build_bush(parent: NodePath, size: float = 0.6, color: tuple[float, float, float, float] = (0.14, 0.27, 0.11, 1.0)) -> NodePath:
    """Кущ: 3 перехресні картки (густіший за пучок трави) — дешева
    об'ємна зелень для наповнення поля."""
    from panda3d.core import CardMaker, GeomVertexRewriter, TransparencyAttrib

    bush = parent.attachNewNode("bush")
    tex = _make_bush_texture(seed=int(size * 100) % 5)
    for heading in (0, 60, 120):
        cm = CardMaker("bush_card")
        cm.setFrame(-size, size, 0.0, size * 1.6)
        card_node = cm.generate()
        # Нормалі ВГОРУ замість геометричних нормалей картки — той самий
        # трюк ігрової рослинності, що й у ``_build_blade_field_mesh`` (див.
        # його коментар): з геометричними нормалями 1-2 з 3 вертикальних
        # карток дивляться ВІД сонця -> лише ambient -> кущ рендерився
        # МАЙЖЕ ЧОРНОЮ плитою (перевірено скріншотом).
        geom = card_node.modifyGeom(0)
        vdata = geom.modifyVertexData()
        rewriter = GeomVertexRewriter(vdata, "normal")
        while not rewriter.isAtEnd():
            rewriter.setData3(0, 0, 1)
        card = bush.attachNewNode(card_node)
        card.setH(heading)
        card.setTexture(tex)
        # На листковому вузлі (не обгортці) — переживає flattenStrong
        # (та сама пастка, що й трава, docs/DECISIONS.md).
        card.setTransparency(TransparencyAttrib.MAlpha)
        card.setTwoSided(True)
        card.setMaterial(mat.FOLIAGE)
    # Текстура ВЖЕ несе колір листя — ``color`` тут перетворюється на ЛЕГКИЙ
    # відтінок (нормований навколо 1.0 за середнім еталонного кольору
    # (0.14,0.27,0.11)), НЕ множиться як повний колір: інакше
    # текстура*колір перемножуються вдвічі і кущ стає майже чорним
    # (реальний артефакт скріншота: подвійне затемнення).
    ref_mean = (0.14 + 0.27 + 0.11) / 3.0
    color_mean = (color[0] + color[1] + color[2]) / 3.0
    shade = color_mean / ref_mean if ref_mean > 0 else 1.0
    bush.setColor(Vec4(shade, shade, shade, 1.0))
    return bush


def _bilinear_sample(x, y, grid, extent):
    """Білінійна вибірка з ``grid`` (G×G, значення 0..1) у точках (x,y),
    де поле [-extent, extent]² мапиться на індекси сітки — векторизовано."""
    g = grid.shape[0]
    u = np.clip((x + extent) / (2 * extent) * (g - 1), 0.0, g - 1 - 1e-6)
    v = np.clip((y + extent) / (2 * extent) * (g - 1), 0.0, g - 1 - 1e-6)
    i0 = u.astype(np.int64)
    j0 = v.astype(np.int64)
    fu, fv = u - i0, v - j0
    g00 = grid[j0, i0]
    g10 = grid[j0, i0 + 1]
    g01 = grid[j0 + 1, i0]
    g11 = grid[j0 + 1, i0 + 1]
    return g00 * (1 - fu) * (1 - fv) + g10 * fu * (1 - fv) + g01 * (1 - fu) * fv + g11 * fu * fv


def _grass_density(x, y, radius, rng, clearings, exclude_rects=None, taper_start=None, taper_end=None, taper_min=0.22):
    """DENSITY MAP (0..1) — "хітмапа" густоти трави в кожній точці (реальний
    фідбек: як зробити рівномірну траву з галявинами). Складники:

    1. Базова висока густина (трава майже усюди).
    2. Низькочастотний згладжений шум -> природні згущення/розрідження
       (не однорідна "газонна" щільність, а живі плями).
    3. Галявини (``clearings``: список ``(cx, cy, cr)``) — витоптані/протерті
       кола, де густина плавно (smoothstep) падає до 0 у ядрі; сюди входить
       і чиста точка старту.
    4. ``exclude_rects`` (``(cx, cy, half_x, half_y)`` або ``(cx, cy, half_x,
       half_y, angle_deg)`` — ОБЕРНЕНИЙ прямокутник, для полів-під-кутом)
       без трави взагалі (дорога, пшеничне поле — воно саме вкриває землю)
       — твердий виріз, без плавного переходу (як межа асфальту/оранки).
    5. ``taper_start``/``taper_end``/``taper_min`` — ПОСТУПОВЕ (smoothstep)
       спадання густини з відстанню від центру: 100% до ``taper_start``,
       лінійно/плавно до ``taper_min`` на ``taper_end``. Реальний фідбек: "не
       роби periметр в 80м — уся видима площина має бути в траві" — трава
       тепер сягає значно далі, АЛЕ без цього поступового спадання рівномірна
       густина аж до горизонту дала б у рази більше стебел (реальна
       регресія продуктивності, знайдена профілюванням: 720 кроків фізики за
       ~60с+ замість <40с). Замість жорсткого "кола, за яким трави нема" —
       трава просто РІДШАЄ вдалині (як і виглядало б природно на відстані),
       не зникає раптово.

    Це стандартний спосіб керувати рослинністю в іграх: розсіяти кандидатів
    рівномірно (jittered grid у ``scatter_ground_cover``), а density map
    вирішує, ЯКІ з них реально проростуть."""
    # Низькочастотний шум: коарс-сітка + кілька згладжень (той самий прийом,
    # що й текстура землі, render/visuals.py::_make_grass_texture).
    coarse = rng.uniform(0.0, 1.0, (10, 10))
    for _ in range(2):
        coarse = (
            coarse + np.roll(coarse, 1, 0) + np.roll(coarse, -1, 0)
            + np.roll(coarse, 1, 1) + np.roll(coarse, -1, 1)
        ) / 5.0
    noise = _bilinear_sample(x, y, coarse, radius)
    noise = (noise - noise.min()) / (np.ptp(noise) + 1e-9)  # нормалізувати 0..1
    density = 0.6 + 0.4 * noise  # базово густо (0.6..1.0)

    if taper_start is not None and taper_end is not None and taper_end > taper_start:
        r = np.hypot(x, y)
        t = np.clip((r - taper_start) / (taper_end - taper_start), 0.0, 1.0)
        t = t * t * (3.0 - 2.0 * t)  # smoothstep — плавно, без видимої межі
        density *= 1.0 - t * (1.0 - taper_min)

    for cx, cy, cr in clearings:
        dist = np.hypot(x - cx, y - cy)
        # smoothstep: 0 у ядрі (dist < 0.35*cr) -> 1 на краю (dist > cr).
        t = np.clip((dist - 0.35 * cr) / (0.65 * cr + 1e-9), 0.0, 1.0)
        density *= t * t * (3.0 - 2.0 * t)
    for rect in exclude_rects or ():
        cx, cy, hx, hy = rect[0], rect[1], rect[2], rect[3]
        angle_deg = rect[4] if len(rect) > 4 else 0.0
        if angle_deg:
            ca, sa = math.cos(math.radians(-angle_deg)), math.sin(math.radians(-angle_deg))
            lx = (x - cx) * ca - (y - cy) * sa
            ly = (x - cx) * sa + (y - cy) * ca
        else:
            lx, ly = x - cx, y - cy
        inside = (np.abs(lx) <= hx) & (np.abs(ly) <= hy)
        density = np.where(inside, 0.0, density)
    return np.clip(density, 0.0, 1.0)


def _segment_distance(px, py, x0, y0, x1, y1):
    """Векторизована відстань від точок ``(px, py)`` до відрізка
    ``(x0,y0)-(x1,y1)`` (проєкція з клемпом ``t``) — для хребтів рельєфу
    (реальний фідбек: "яр/пагорб — це не точки, вони мають тягнутися, як
    хребет, але й закінчуватись")."""
    dx, dy = x1 - x0, y1 - y0
    seg_len2 = dx * dx + dy * dy + 1e-9
    t = np.clip(((px - x0) * dx + (py - y0) * dy) / seg_len2, 0.0, 1.0)
    proj_x, proj_y = x0 + t * dx, y0 + t * dy
    return np.hypot(px - proj_x, py - proj_y)


def _voronoi_cells(flat_radius, outer_radius, seed):
    """Клітинки Вороного — ОСНОВА рельєфу/ландшафту (реальний фідбек:
    "рельєф можна зробити більш нормальним, генерувати heightmap
    реалістичний спочатку з картою Вороного — це буде основа для генерації у
    майбутньому по многокутниках річок/гір/ярів/рівних полів/посадок").
    Розкидані "насінини" (``seeds_x, seeds_y``) — кожна визначає ЦІЛИЙ
    регіон-полігон (клітинку Вороного) характеру: ``"hill"`` (горб),
    ``"ravine"`` (яр/западина), ``"flat"`` (рівне поле — на суцільних
    зв'язних групах таких клітинок пізніше сіється пшеничне поле,
    ``_voronoi_grouped_regions(..., kind="flat")``), ``"plantation"``
    (посадка — суцільні групи дають рядкову лісосмугу,
    ``_voronoi_grouped_regions(..., kind="plantation")``, доп. фаза
    "розширення kind"). Для ``"hill"``/``"ravine"`` — ще й
    ``ridge_angle``/``ridge_half_len``: рельєф росте не круглою "бульбашкою"
    в точці насінини, а ВИТЯГНУТИМ ХРЕБТОМ уздовж випадкового напрямку,
    обмеженої довжини (``_terrain_height_grid`` рахує відстань до цього
    ВІДРІЗКА, не до точки) — "хребет, не точка, але й закінчується".
    ``heights`` — нормована амплітуда (множник ``max_height``);
    ``"plantation"`` (як і ``"flat"``) лишає ``heights``/``ridge_*`` на 0 —
    не впливає на рельєф, лише на розміщення декору (суто візуальне, як і
    решта цього конвеєра). Майбутнє розширення (новий kind — напр. "river")
    додається СЮДИ, без зміни решти конвеєра (grid-семплінг/згладжування
    нижче)."""
    rng = np.random.default_rng(seed)
    extent = outer_radius
    area = (2.0 * extent) ** 2
    n_cells = max(10, int(area / 550.0))
    seeds_x = rng.uniform(-extent, extent, n_cells)
    seeds_y = rng.uniform(-extent, extent, n_cells)
    roll = rng.random(n_cells)
    kinds = np.full(n_cells, "flat", dtype=object)
    heights = np.zeros(n_cells)
    ridge_angle = np.zeros(n_cells)
    ridge_half_len = np.zeros(n_cells)
    hill_mask = roll < 0.27
    ravine_mask = (roll >= 0.27) & (roll < 0.48)
    plantation_mask = (roll >= 0.48) & (roll < 0.60)
    relief_mask = hill_mask | ravine_mask
    kinds[hill_mask] = "hill"
    kinds[ravine_mask] = "ravine"
    kinds[plantation_mask] = "plantation"
    heights[hill_mask] = rng.uniform(0.55, 1.3, int(hill_mask.sum()))
    heights[ravine_mask] = -rng.uniform(0.45, 1.1, int(ravine_mask.sum()))
    n_relief = int(relief_mask.sum())
    ridge_angle[relief_mask] = rng.uniform(0.0, math.pi, n_relief)
    ridge_half_len[relief_mask] = rng.uniform(5.0, 15.0, n_relief)
    return seeds_x, seeds_y, heights, kinds, ridge_angle, ridge_half_len


@functools.lru_cache(maxsize=32)
def _terrain_height_grid(flat_radius, outer_radius, max_height, seed, grid_res=100):
    """Груба сітка висот: КОЖНА клітинка-хребет (``_voronoi_cells``, kind
    hill/ravine) додає bump уздовж свого відрізка (``_segment_distance`` —
    ХРЕБЕТ, не точка), внески сумуються, потім кілька проходів згладжування
    розмивають гострі краї в органічні пологі схили (той самий прийом
    box-blur, що й для шумових сіток вище) — "реалістичний heightmap", а не
    геометричні купола. Радіальне згасання до 0 на краю сітки гарантує, що
    семпли ЗА ``outer_radius`` (clip у ``_bilinear_sample``) не тягнуть
    довільну ненульову висоту в нескінченність.

    ``lru_cache`` — КРИТИЧНО для продуктивності: у межах однієї сесії ця
    сітка (той самий ``flat_radius``/``outer_radius``/``seed``) запитується
    десятки разів (меш землі, трава, кожен кущ, кожен кандидат поля) —
    без кешу кожен виклик перебудовує Вороного+згладжування заново (~30мс),
    що при сотнях викликів (напр. по одному на кущ) давало реальну регресію
    в кілька секунд на побудову сцени (знайдено профілюванням)."""
    seeds_x, seeds_y, heights, kinds, ridge_angle, ridge_half_len = _voronoi_cells(flat_radius, outer_radius, seed)
    extent = outer_radius
    xs = np.linspace(-extent, extent, grid_res)
    gx, gy = np.meshgrid(xs, xs, indexing="ij")
    grid = np.zeros_like(gx)
    ridge_half_width = 11.0  # м — ширина хребта (falloff від осі)
    # Явно лише hill/ravine (не "!= flat") — "plantation" та будь-який
    # МАЙБУТНІЙ не-рельєфний kind мають heights=0 і не мали б впливати на
    # висоту навіть неявно (нуль-множник), але явний фільтр і дешевший, і
    # застраховує від майбутнього kind, що НЕ повинен чіпати рельєф.
    for i in np.where((kinds == "hill") | (kinds == "ravine"))[0]:
        cx, cy, ang, hl = seeds_x[i], seeds_y[i], ridge_angle[i], ridge_half_len[i]
        dirx, diry = math.cos(ang), math.sin(ang)
        x0, y0 = cx - dirx * hl, cy - diry * hl
        x1, y1 = cx + dirx * hl, cy + diry * hl
        d = _segment_distance(gx, gy, x0, y0, x1, y1)
        t = np.clip(1.0 - d / ridge_half_width, 0.0, 1.0)
        bump = t * t * (3.0 - 2.0 * t)
        grid += heights[i] * max_height * bump
    for _ in range(4):
        grid = (
            grid + np.roll(grid, 1, 0) + np.roll(grid, -1, 0)
            + np.roll(grid, 1, 1) + np.roll(grid, -1, 1)
        ) / 5.0
    r = np.hypot(gx, gy)
    fade_t = np.clip((r - 0.78 * extent) / max(0.22 * extent, 1e-6), 0.0, 1.0)
    fade = 1.0 - fade_t * fade_t * (3.0 - 2.0 * fade_t)
    grid *= fade
    return grid, extent


def _terrain_height(x, y, flat_radius, outer_radius, max_height, seed=9):
    """Реальний рельєф на основі heightmap Вороного (``_terrain_height_grid``)
    — плоско (z=0, той самий інваріант ``pos.z == ground_z``, на якому
    тримається розміщення obstacle/target/spawn) у РОБОЧІЙ зоні
    (``flat_radius``, рахується динамічно з фактичних перешкод/цілей/воріт —
    ``app.py``), а далі органічні полігональні ділянки — горби, яри, рівні
    поля (не поодинокі "бульбашки"). Суто ВІЗУАЛЬНЕ — фізика (плоска
    ``BulletPlaneShape``) не міняється, тож у робочій зоні жодного
    розсинхрону немає."""
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    if max_height <= 0.0:
        return np.zeros_like(x)
    r = np.hypot(x, y)
    grid, extent = _terrain_height_grid(flat_radius, outer_radius, max_height, seed)
    h = _bilinear_sample(x, y, grid, extent)
    # Гарантія: точна нуль-площина в робочій зоні, з короткою перехідною
    # смугою на межі ``flat_radius`` (навіть якщо клітинка туди трохи заходить).
    guard = np.clip((r - flat_radius) / 4.0, 0.0, 1.0)
    guard = guard * guard * (3.0 - 2.0 * guard)
    return h * guard


def _voronoi_adjacency(seeds_x, seeds_y, extent, grid_res=80):
    """Наближення сусідства клітинок Вороного БЕЗ важких залежностей
    (``scipy.spatial.Delaunay`` коштує ~2с на перший імпорт у процесі —
    відчутно на тлі бюджету побудови сцени; знайдено профілюванням реальної
    регресії у тесті, що будує сесію в дочірньому процесі з таймаутом).
    Той самий растеризаційний прийом, що й ``_terrain_height_grid`` (стара
    версія): груба сітка -> НАЙБЛИЖЧА насінина на кожен піксель; де сусідні
    (по горизонталі/вертикалі) пікселі належать РІЗНИМ насінинам — ці дві
    насінини сусіди. Це наближення (не точний дуальний граф Делоне), але
    достатнє для BFS "рівна клітинка/рівний сусід" нижче."""
    n = len(seeds_x)
    neighbors: list[set[int]] = [set() for _ in range(n)]
    if n < 2:
        return neighbors
    xs = np.linspace(-extent, extent, grid_res)
    gx, gy = np.meshgrid(xs, xs, indexing="ij")
    d2 = (
        (gx.ravel()[:, None] - seeds_x[None, :]) ** 2
        + (gy.ravel()[:, None] - seeds_y[None, :]) ** 2
    )
    nearest = np.argmin(d2, axis=1).reshape(grid_res, grid_res)
    for a, b in ((nearest[:-1, :], nearest[1:, :]), (nearest[:, :-1], nearest[:, 1:])):
        diff = a != b
        for i, j in zip(*np.where(diff)):
            u, v = int(a[i, j]), int(b[i, j])
            neighbors[u].add(v)
            neighbors[v].add(u)
    return neighbors


def _voronoi_grouped_regions(flat_radius, outer_radius, seed, kind="flat", min_cells=2, max_cells=8):
    """СУЦІЛЬНІ зв'язні групи клітинок Вороного ОДНОГО ``kind`` (реальний
    фідбек про поля: "робимо поле і перевіряємо поруч клітинки, якщо також
    більш-менш рівно — додаємо поле і туди; де рельєф піднімається чи
    опускається різко — не садять поле") — узагальнено з початкової
    "лише flat -> поля пшениці" версії, щоб той самий алгоритм обслуговував
    БУДЬ-ЯКИЙ груповий kind (напр. ``kind="plantation"`` — суцільні посадки,
    доп. фаза "розширення kind"), без дублювання BFS/PCA. Алгоритм: (1)
    наближення сусідства клітинок Вороного через растеризацію
    (``_voronoi_adjacency``); (2) BFS по графу сусідства, обмежений лише
    ``kind == kind`` — це і є "рівно (чи посадка), і сусід теж такий самий" з
    фідбеку, АЛЕ обмежений ``max_cells`` — без цього обмеження суцільна
    ділянка перколює в ОДНУ величезну групу на всю карту (класична
    перколяція на випадковому графі); реальні поля/посадки — окремі ділянки,
    тож завелика зв'язна група ділиться на КІЛЬКА (BFS просто зупиняється на
    капі, недовідвідані клітинки тієї ж групи стають стартом НАСТУПНОЇ в
    зовнішньому циклі); (3) для кожної групи — PCA (власні вектори
    коваріації) дає ГОЛОВНУ ВІСЬ (довгу сторону, вздовж якої тягтиметься
    рядкова оранка чи лісосмуга) і обернений прямокутник, що покриває всі
    клітинки групи з запасом.

    Повертає список dict: ``center (x,y)``, ``angle_deg`` (напрямок рядків),
    ``length``, ``width``, ``cells`` (індекси насінин)."""
    seeds_x, seeds_y, _heights, kinds, _ang, _hl = _voronoi_cells(flat_radius, outer_radius, seed)
    pts = np.column_stack([seeds_x, seeds_y])
    neighbors = _voronoi_adjacency(seeds_x, seeds_y, outer_radius)

    kind_idx = set(int(i) for i in np.where(kinds == kind)[0])
    visited: set[int] = set()
    regions = []
    for i in sorted(kind_idx):
        if i in visited:
            continue
        comp: list[int] = [i]
        visited.add(i)
        frontier = [i]
        while frontier and len(comp) < max_cells:
            cur = frontier.pop()
            for nb in neighbors[cur]:
                if len(comp) >= max_cells:
                    break
                if nb in kind_idx and nb not in visited:
                    visited.add(nb)
                    comp.append(nb)
                    frontier.append(nb)
        if len(comp) < min_cells:
            continue  # замала група — лишається звичайною травою, не полем/посадкою
        comp_pts = pts[comp]
        centroid = comp_pts.mean(axis=0)
        centered = comp_pts - centroid
        if len(comp) >= 3:
            cov = np.cov(centered.T)
            evals, evecs = np.linalg.eigh(cov)
            main_axis = evecs[:, int(np.argmax(evals))]
        else:
            main_axis = np.array([1.0, 0.0])
        angle = math.atan2(main_axis[1], main_axis[0])
        ca, sa = math.cos(-angle), math.sin(-angle)
        rot = np.array([[ca, -sa], [sa, ca]])
        local = centered @ rot.T
        pad = 9.0  # запас навколо крайніх насінин (типовий "радіус" клітинки)
        # Клемп зверху: рідкісний випадок (насінини лягли у витягнутий
        # ланцюжок при випадковому scatter) інакше давав поля довжиною за
        # 250м+ — і сотні стовпців паркану/дерев уздовж них (реальна
        # регресія продуктивності, знайдена профілюванням: 720 кроків
        # фізики за 58с замість <2с через тисячі нефлетнутих NodePath).
        length = min(max(float(local[:, 0].max() - local[:, 0].min()) + 2 * pad, 16.0), 130.0)
        width = min(max(float(local[:, 1].max() - local[:, 1].min()) + 2 * pad, 11.0), 90.0)
        regions.append({
            "center": (float(centroid[0]), float(centroid[1])),
            "angle_deg": math.degrees(angle),
            "length": length,
            "width": width,
            "cells": comp,
        })
    return regions


# Фіксований масштаб поля "вологості/сухості" — НЕ залежить від радіуса
# конкретного виклику (``build_terrain`` рахує на всю сітку ~250м,
# ``scatter_ground_cover`` — лише на своїх ~65м кандидатах). Якби кожен
# виклик нормалізував за ВЛАСНИМ min/max вибраних точок, однакова world-
# координата давала б РІЗНЕ значення сухості для землі й для трави над нею
# (реальний баг, знайдений при звірці: трава й земля не збігались кольором) —
# тому нормалізація рахується ОДИН РАЗ на самій сітці шуму (яка однакова для
# будь-яких x,y при тому самому seed), а не на вибірці точок виклику.
_DRYNESS_EXTENT = 180.0


def _terrain_dryness(x, y, seed=21):
    """Спатіальне поле "сухості/вологості" 0..1 (реальний фідбек: "в одному
    місці трава жовта, в іншому більше вологи і вона зелена — щоб
    відрізнялася") — великі плями (грубіша сітка, ніж у ``_grass_density``,
    щоб плями були помітно БІЛЬШІ за згущення/розрідження густини), спільне
    для кольору землі (``build_terrain``) і тону/висоти трав'яних стебел
    (``scatter_ground_cover``), щоб трава відповідала землі під нею."""
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    rng = np.random.default_rng(seed)
    coarse = rng.uniform(0.0, 1.0, (6, 6))
    for _ in range(2):
        coarse = (
            coarse + np.roll(coarse, 1, 0) + np.roll(coarse, -1, 0)
            + np.roll(coarse, 1, 1) + np.roll(coarse, -1, 1)
        ) / 5.0
    # Нормалізація НА СІТЦІ (однакова для всіх викликів), не на вибірці x,y.
    coarse = (coarse - coarse.min()) / (np.ptp(coarse) + 1e-9)
    # Контраст: розсуває значення від сірої середини до країв -> чіткіші
    # зелені/жовті ПЛЯМИ з вузькою перехідною смугою, а не суцільний
    # оливковий блендінг.
    coarse = np.clip((coarse - 0.5) * 2.4 + 0.5, 0.0, 1.0)
    return _bilinear_sample(x, y, coarse, _DRYNESS_EXTENT)


def build_terrain(
    parent: NodePath,
    radius: float = 250.0,
    flat_radius: float = 42.0,
    outer_radius: float = 120.0,
    max_height: float = 4.5,
    segments: int = 160,
    tile_size_m: float = 4.0,
    seed: int = 9,
    dryness_seed: int = 21,
) -> NodePath:
    """Справжня земля-рельєф (не плоска картка): плоска в РОБОЧІЙ зоні
    (``flat_radius`` — де стоять перешкоди/цілі/старт, ``app.py`` рахує це
    динамічно з фактичної сцени), далі справжні пологі горби до
    ``outer_radius`` (``_terrain_height``), і колірний градієнт зелене->
    жовте плямами по всій поверхні (``_terrain_dryness``) — реальний фідбек:
    "рельєф не лише на горизонті" + "переливи трави, десь жовта".

    Один векторизований Geom (той самий numpy-блит прийом, що й трав'яний
    килим): регулярна сітка ``segments×segments`` вершин, нормалі —
    аналітичний градієнт висоти (``normalize(-dz/dx, -dz/dy, 1)``), колір —
    вершинний (``V3n3c4t2``, ПАКОВАНІ байти — див. ``_build_blade_field_mesh``
    docstring), що МНОЖИТЬ тайловану текстуру трави (стандартний
    modulate-режим Panda3D за замовчуванням)."""
    from dronesim.render.visuals import _make_grass_texture

    n = segments + 1
    xs = np.linspace(-radius, radius, n)
    gx, gy = np.meshgrid(xs, xs, indexing="ij")
    gz = _terrain_height(gx, gy, flat_radius, outer_radius, max_height, seed).astype(np.float64)

    # Нормалі з аналітичного градієнта висоти по сітці.
    dzdx = np.gradient(gz, xs, axis=0)
    dzdy = np.gradient(gz, xs, axis=1)
    nx, ny, nz = -dzdx, -dzdy, np.ones_like(gz)
    nlen = np.sqrt(nx * nx + ny * ny + nz * nz)
    nx, ny, nz = nx / nlen, ny / nlen, nz / nlen

    dryness = _terrain_dryness(gx, gy, dryness_seed) ** 1.15  # трохи більше "чисто зеленого"
    green = np.array([0.16, 0.38, 0.09])
    yellow = np.array([0.62, 0.50, 0.10])
    col = green[None, None, :] * (1.0 - dryness[..., None]) + yellow[None, None, :] * dryness[..., None]

    repeats = max(1.0, (2.0 * radius) / tile_size_m)
    u = (gx + radius) / (2.0 * radius) * repeats
    v = (gy + radius) / (2.0 * radius) * repeats

    dt = np.dtype([("pos", "<f4", 3), ("normal", "<f4", 3), ("color", "u1", 4), ("uv", "<f4", 2)])
    verts = np.zeros((n, n), dtype=dt)
    verts["pos"][..., 0] = gx
    verts["pos"][..., 1] = gy
    verts["pos"][..., 2] = gz
    verts["normal"][..., 0] = nx
    verts["normal"][..., 1] = ny
    verts["normal"][..., 2] = nz
    verts["color"][..., 0] = np.clip(col[..., 0] * 255.0, 0, 255).astype(np.uint8)
    verts["color"][..., 1] = np.clip(col[..., 1] * 255.0, 0, 255).astype(np.uint8)
    verts["color"][..., 2] = np.clip(col[..., 2] * 255.0, 0, 255).astype(np.uint8)
    verts["color"][..., 3] = 255
    verts["uv"][..., 0] = u
    verts["uv"][..., 1] = v
    raw = verts.reshape(n * n).tobytes()

    i0 = np.arange(n - 1)
    ii, jj = np.meshgrid(i0, i0, indexing="ij")
    a = (ii * n + jj).ravel()
    b = (ii * n + jj + 1).ravel()
    c = ((ii + 1) * n + jj).ravel()
    d = ((ii + 1) * n + jj + 1).ravel()
    idx = np.stack([a, c, b, b, c, d], axis=1).ravel().astype(np.uint32)

    vdata = GeomVertexData("terrain", GeomVertexFormat.getV3n3c4t2(), Geom.UHStatic)
    vdata.uncleanSetNumRows(n * n)
    vdata.modifyArray(0).modifyHandle().copySubdataFrom(0, len(raw), raw)

    prim = GeomTriangles(Geom.UHStatic)
    prim.setIndexType(GeomEnums.NT_uint32)
    iarr = prim.modifyVertices()
    iarr.uncleanSetNumRows(len(idx))
    iarr.modifyHandle().copySubdataFrom(0, idx.nbytes, idx.tobytes())

    geom = Geom(vdata)
    geom.addPrimitive(prim)
    node = GeomNode("terrain")
    node.addGeom(geom)
    result = parent.attachNewNode(node)
    result.setTexture(_make_grass_texture(seed=seed))
    result.setTwoSided(True)
    result.setMaterial(mat.FOLIAGE)
    return result


def scatter_ground_cover(
    parent: NodePath,
    radius: float = 70.0,
    density_per_m2: float = 14.0,
    bush_count: int = 90,
    seed: int = 7,
    exclude_radius: float = 3.5,
    clearings: list[tuple[float, float, float]] | None = None,
    exclude_rects: list[tuple[float, float, float, float]] | None = None,
    terrain_flat_radius: float = 1e9,
    terrain_outer_radius: float | None = None,
    terrain_max_height: float = 0.0,
    terrain_seed: int = 9,
    dryness_seed: int = 21,
    density_taper_start: float | None = None,
    density_taper_end: float | None = None,
    density_taper_min: float = 0.22,
) -> NodePath:
    """Густий КИЛИМ трави по всьому полю з природними згущеннями й галявинами
    (реальний фідбек: "трава — це шар майже усюди, густий, крім десь
    витоптана", не поодинокі об'єкти). Суто ВІЗУАЛЬНЕ, без фізики.

    Техніка (стандартна для ігор, докладніше ``_grass_density``): (1)
    JITTERED GRID — поле ділиться на клітинки ``1/sqrt(density_per_m2)``, у
    кожній один кандидат зі зсувом (рівномірне покриття без плям/дір, на
    відміну від чистого random); (2) DENSITY MAP вирішує, які кандидати
    проростуть (згущення/розрідження + галявини + ``exclude_rects`` — дорога
    без трави); (3) увесь килим — ОДИН векторизований Geom
    (``_build_blade_field_mesh``, numpy-блит, десятки мс навіть на 100k+
    стебел).

    ``clearings`` — додаткові витоптані кола ``(cx, cy, cr)`` (напр. навколо
    цілей/техніки); точка старту (``exclude_radius``) додається автоматично.
    ``exclude_rects`` — прямокутні зони ``(cx, cy, half_x, half_y)`` без
    трави взагалі (дорога).

    ``terrain_*`` — якщо land не пласка (``build_terrain``, той самий
    ``_terrain_height``, ті самі параметри й seed), трава/кущі сідають на
    РЕАЛЬНУ висоту рельєфу під ними, а не "плавають" над схилом; за
    замовчуванням (``terrain_max_height=0``) — пласка земля, як і раніше.
    ``dryness_seed`` — те саме поле "сухості" (``_terrain_dryness``), що й
    колір землі, для зелено-жовтого градієнта стебел, що відповідає землі.
    ``density_taper_start``/``density_taper_end`` — плавне спадання густини
    з відстанню (реальний фідбек: "не роби periметр в 80м — уся видима
    площина має бути в траві"): трава сягає значно далі (``radius`` тепер
    можна ставити ~150м+), але без цього таперу рівномірна густина аж до
    горизонту дала б у рази більше стебел, ніж потрібно (реальна регресія
    продуктивності, знайдена профілюванням); замість жорсткого "кола" —
    трава просто рідшає вдалині, без видимої межі."""
    rng = np.random.default_rng(seed)
    cover = parent.attachNewNode("ground_cover")
    all_clearings = [(0.0, 0.0, exclude_radius)] + list(clearings or [])
    # Кілька випадкових "витоптаних" плям для природності.
    for _ in range(4):
        cr = float(rng.uniform(2.5, 5.0))
        ca = rng.uniform(0.0, 2 * math.pi)
        cd = rng.uniform(0.3 * radius, 0.9 * radius)
        all_clearings.append((cd * math.cos(ca), cd * math.sin(ca), cr))

    # 1. Jittered grid кандидатів по диску.
    cell = 1.0 / math.sqrt(density_per_m2)
    n_side = int(2 * radius / cell) + 1
    gi, gj = np.meshgrid(np.arange(n_side), np.arange(n_side), indexing="ij")
    gx = -radius + (gi.ravel() + rng.random(gi.size)) * cell
    gy = -radius + (gj.ravel() + rng.random(gj.size)) * cell
    inside = gx * gx + gy * gy <= radius * radius
    gx, gy = gx[inside], gy[inside]

    # 2. Density map -> монетка: які кандидати проростуть.
    density = _grass_density(
        gx, gy, radius, np.random.default_rng(seed + 1), all_clearings, exclude_rects,
        taper_start=density_taper_start, taper_end=density_taper_end, taper_min=density_taper_min,
    )
    keep = rng.random(len(gx)) < density
    gx, gy = gx[keep], gy[keep]
    n = len(gx)

    # 3. Атрибути стебел: низькі (килим, не шпилі), легка варіація кольору +
    # зелено-жовтий градієнт плямами (``_terrain_dryness``, реальний фідбек:
    # "в одному місці трава жовта, в іншому більше вологи і зелена — щоб
    # відрізнялася"). ТЕ САМЕ поле "вологості" тягне і ВИСОТУ стебла
    # (реальний фідбек: "можна так само варіювати градієнтом... висоту") —
    # вологі/зелені ділянки густіші й вищі, сухі/жовті — коротші, рідші на
    # вигляд (менша висота картки == менше видимої "маси" трави).
    outer_r = terrain_outer_radius if terrain_outer_radius is not None else radius
    gz = _terrain_height(gx, gy, terrain_flat_radius, outer_r, terrain_max_height, terrain_seed)
    dryness = _terrain_dryness(gx, gy, dryness_seed) ** 1.15
    moist = 1.0 - dryness
    height = rng.uniform(0.08, 0.14, n) + moist * rng.uniform(0.06, 0.14, n)
    width = height * 0.7
    heading = rng.uniform(0.0, 2 * math.pi, n)
    bright = rng.uniform(0.8, 1.15, n)[:, None]  # варіація яскравості на стебло
    dryness_col = dryness[:, None]
    green_tint = np.array([1.0, 1.0, 1.0])
    yellow_tint = np.array([1.5, 1.15, 0.35])  # більше R, менше B -> золотисто поверх зеленої текстури
    tint = np.clip(
        bright * (green_tint[None, :] * (1.0 - dryness_col) + yellow_tint[None, :] * dryness_col), 0.0, 1.6
    )
    _build_blade_field_mesh(cover, "grass_field", gx, gy, height, width, heading,
                            _make_grass_blade_texture(seed=0), tint=tint, base_z=gz)

    # Кущі — рідко, лише поза галявинами (не в точці старту). Позиції
    # спершу збираються рекцією-семплінгом (дешево), а ВИСОТА рельєфу під
    # усіма кущами рахується ОДНИМ пакетним викликом ``_terrain_height``
    # ПІСЛЯ циклу — не по одній точці всередині (кожен виклик перебудовує
    # всю сітку Вороного+згладжування, ~30мс; по одній на кущ = сотні мс
    # зайвих мс, реальна регресія продуктивності, знайдена профілюванням
    # після додавання рельєфу до кущів).
    bushes = cover.attachNewNode("bushes")
    bush_x: list[float] = []
    bush_y: list[float] = []
    while len(bush_x) < bush_count:
        r = exclude_radius + (radius - exclude_radius) * math.sqrt(rng.random())
        theta = rng.uniform(0.0, 2 * math.pi)
        bx, by = r * math.cos(theta), r * math.sin(theta)
        if any(math.hypot(bx - cx, by - cy) < cr for cx, cy, cr in all_clearings):
            continue
        bush_x.append(bx)
        bush_y.append(by)
    bush_z = _terrain_height(
        np.array(bush_x), np.array(bush_y), terrain_flat_radius, outer_r, terrain_max_height, terrain_seed
    )
    for bx, by, bz in zip(bush_x, bush_y, bush_z):
        shade = float(rng.uniform(0.9, 1.15))
        item = build_bush(bushes, size=float(rng.uniform(0.35, 0.6)), color=(0.18 * shade, 0.34 * shade, 0.13 * shade, 1.0))
        item.setPos(bx, by, float(bz))
        item.setH(float(rng.uniform(0.0, 360.0)))
    # flattenStrong ЛИШЕ на кущах (не на всьому cover — інакше з'їв би
    # TransparencyAttrib трав'яного Geom, docs/DECISIONS.md).
    bushes.flattenStrong()
    return cover


def build_fence_line(
    parent: NodePath, start: tuple[float, float], end: tuple[float, float], post_spacing: float = 2.0, post_height: float = 1.1
) -> NodePath:
    """Паркан: стовпці + 2 горизонтальні рейки вздовж лінії від ``start`` до ``end``."""
    fence = parent.attachNewNode("fence")
    start_v, end_v = np.array(start, dtype=float), np.array(end, dtype=float)
    length = float(np.linalg.norm(end_v - start_v))
    if length < 1e-6:
        return fence
    direction = (end_v - start_v) / length
    angle_deg = math.degrees(math.atan2(direction[1], direction[0]))
    # Кап на к-сть стовпців — без нього дуже довгий паркан (напр. вздовж
    # великого поля) давав сотні нефлетнутих NodePath і реально гальмував
    # рендер щокадру (знайдено профілюванням, не лише час побудови).
    n_posts = min(max(2, int(length / post_spacing) + 1), 60)

    for i in range(n_posts):
        t = i / (n_posts - 1)
        pos = start_v + direction * length * t
        post = add_box_visual(fence, (0.05, 0.05, post_height / 2), color=_TRUNK_COLOR, material=mat.WOOD)
        post.setPos(float(pos[0]), float(pos[1]), post_height / 2)

    mid = (start_v + end_v) / 2
    for rail_z in (post_height * 0.35, post_height * 0.75):
        rail = add_box_visual(fence, (length / 2, 0.02, 0.04), color=_TRUNK_COLOR, material=mat.WOOD)
        rail.setPos(float(mid[0]), float(mid[1]), rail_z)
        rail.setH(angle_deg)

    # Звести в ОДИН draw-виклик (посилюємо кап вище — без цього кожен стовпець/
    # рейка лишається окремим NodePath, що гальмує cull щокадру при багатьох
    # парканах на сцені).
    fence.flattenStrong()
    return fence


def scatter_tree_line(
    parent: NodePath,
    start: tuple[float, float],
    end: tuple[float, float],
    spacing: float = 3.5,
    seed: int = 0,
    species: tuple[str, ...] = ("pine", "birch"),
) -> NodePath:
    """"Посадка" — лісосмуга/полезахисна лінія дерев уздовж ``start``-``end``
    (реальний фідбек: "посадки — це багато дерев лінією вздовж поля"), як
    справжні захисні смуги вздовж сільськогосподарських полів. Дерева НЕ
    ідеально на лінії (невеликий бічний джиттер) і трохи різняться видом/
    розміром — жива посадка, не паркан. Викликається лише вздовж країв
    ПОЛЯ (``_voronoi_grouped_regions(kind="flat")``), ніколи вздовж пагорба/яру — тому
    "на пагорбі дерев нема" виконується самою логікою розміщення, не
    окремою перевіркою."""
    rng = np.random.default_rng(seed)
    line = parent.attachNewNode("tree_line")
    start_v, end_v = np.array(start, dtype=float), np.array(end, dtype=float)
    length = float(np.linalg.norm(end_v - start_v))
    if length < 1e-6:
        return line
    direction = (end_v - start_v) / length
    normal = np.array([-direction[1], direction[0]])
    # Кап на к-сть дерев — без нього дуже довга лісосмуга (вздовж великого
    # поля) давала сотні нефлетнутих NodePath і гальмувала рендер щокадру
    # (знайдено профілюванням: 720 кроків фізики за 58с замість <2с).
    n_trees = min(max(2, int(length / spacing) + 1), 40)
    for i in range(n_trees):
        t = i / (n_trees - 1)
        jitter_along = rng.uniform(-spacing * 0.15, spacing * 0.15)
        jitter_side = rng.uniform(-0.4, 0.4)
        pos = start_v + direction * (length * t + jitter_along) + normal * jitter_side
        sp = species[int(rng.integers(0, len(species)))]
        tree = build_tree(
            line,
            trunk_height=float(rng.uniform(2.0, 3.2)),
            foliage_radius=float(rng.uniform(1.0, 1.6)),
            species=sp,
            seed=int(rng.integers(0, 1_000_000)),
        )
        tree.setPos(float(pos[0]), float(pos[1]), 0.0)
        tree.setH(float(rng.uniform(0.0, 360.0)))
    # Звести в кілька draw-викликів (за матеріалом), не десятки окремих
    # NodePath на дерево — той самий прийом, що й ``scatter_ground_cover``
    # для кущів (``bushes.flattenStrong()``).
    line.flattenStrong()
    return line


def build_trench(parent: NodePath, length: float = 4.0, width: float = 1.5, wall_height: float = 0.5) -> NodePath:
    """Окоп/траншея — спершу пробує ДЕТАЛЬНУ Blender-авторовану модель
    (``assets/models/trench.bam`` — земляні вали з трапецієподібним
    перетином і радіальним джиттером, мішки з піском на гребені, дерев'яне
    кріплення стінки, дошки-настил на проході; доп. фаза "ландшафт 2.0",
    ``assets/blender/build_trench.py``), масштабовану під запитані
    ``length``/``width``/``wall_height`` відносно ``_BAKED_TRENCH_DIMS``
    (НЕ uniform — 3 незалежні осі, бо width/wall_height на практиці завжди
    рівні референсу, змінюється лише length). Якщо файл відсутній — падає
    на процедурний білдер (``_build_trench_procedural`` — 2 голі земляні
    вали, стара заглушка, реальний фідбек: "2 паралелепіпеди, не траншея")."""
    detailed = load_model_asset(parent, "trench")
    if detailed is not None:
        l0, w0, h0 = _BAKED_TRENCH_DIMS
        detailed.setScale(length / l0, width / w0, wall_height / h0)
        return detailed
    return _build_trench_procedural(parent, length, width, wall_height)


def _build_trench_procedural(parent: NodePath, length: float = 4.0, width: float = 1.5, wall_height: float = 0.5) -> NodePath:
    """Fallback: земляний бруствер (вал) з 2 боків — ВІЗУАЛЬНА ілюзія
    заглиблення, БЕЗ зміни колізії землі (надто ризиковано міняти форму
    землі під фізику апарата; фізичний бокс-obstacle лишається простим,
    докладніше docs/DECISIONS.md)."""
    trench = parent.attachNewNode("trench")
    for side in (-1, 1):
        wall = add_box_visual(trench, (length / 2, 0.25, wall_height / 2), color=_DIRT_COLOR, material=mat.DIRT)
        wall.setPos(0, side * width / 2, wall_height / 2)
    return trench


def _make_road_texture(size: int = 128):
    from panda3d.core import PNMImage, Texture

    img = PNMImage(size, size)
    asphalt = np.array([0.13, 0.13, 0.14])
    rng = np.random.default_rng(0)
    noise = rng.uniform(-0.02, 0.02, size=(size, size, 3))
    lane_y0, lane_y1 = int(size * 0.48), int(size * 0.52)
    for y in range(size):
        for x in range(size):
            if lane_y0 <= y < lane_y1 and (x // 12) % 2 == 0:
                color = (0.85, 0.75, 0.15)  # жовта переривчаста розмітка
            else:
                color = np.clip(asphalt + noise[y, x], 0.0, 1.0)
            img.setXelA(x, y, float(color[0]), float(color[1]), float(color[2]), 1.0)

    tex = Texture("road")
    tex.load(img)
    tex.setWrapU(Texture.WMRepeat)
    tex.setWrapV(Texture.WMRepeat)
    return tex


def build_road_strip(parent: NodePath, length: float = 20.0, width: float = 4.0) -> NodePath:
    """Дорога: текстурована горизонтальна смуга (асфальт + розмітка,
    процедурна текстура — той самий прийом, що й трава/мішені, render/visuals.py)."""
    from panda3d.core import CardMaker, Point2

    cm = CardMaker("road")
    cm.setFrame(-length / 2, length / 2, -width / 2, width / 2)
    repeats_x, repeats_y = max(1.0, length / 4.0), max(1.0, width / 4.0)
    cm.setUvRange(Point2(0.0, 0.0), Point2(repeats_x, repeats_y))
    road = parent.attachNewNode(cm.generate())
    road.setP(-90)
    # Проти z-fighting із землею (реальний фідбек "дорога перетинається із
    # землею і глючить"): polygon depth-offset (зсув у буфері глибини до
    # камери). Вертикальний зазор над землею задає ``attach_scenery_visual``
    # (тут setZ НЕ ставимо — його однаково перезапише грунтування диспетчера).
    road.setDepthOffset(1)
    road.setTexture(_make_road_texture())
    road.setTwoSided(True)
    road.setMaterial(mat.ASPHALT)
    return road


def _make_facade_texture(size: int = 64, seed: int = 0):
    """Процедурна текстура фасаду: бетон + сітка вікон (частина "світиться" —
    тепліший тон, частина темна) — той самий прийом, що й трава/дорога."""
    from panda3d.core import PNMImage, Texture

    rng = np.random.default_rng(seed)
    img = PNMImage(size, size)
    concrete = np.array([0.5, 0.5, 0.52])
    grid = 4  # 4x4 сітка вікон на тайл
    cell = size // grid
    margin = max(1, cell // 5)
    win_lit = np.array([0.95, 0.85, 0.55])
    win_dark = np.array([0.13, 0.15, 0.2])
    # Стан "світиться" вирішується ПО ВІКНУ (не по пікселю — інакше вийшов би
    # шум, а не вікна); детерміновано за клітиною, щоб текстура тайлилась.
    lit_map = rng.random((grid, grid)) < 0.35
    for y in range(size):
        for x in range(size):
            gx, gy = x // cell, y // cell
            cx, cy = x % cell, y % cell
            is_window = margin <= cx < cell - margin and margin <= cy < cell - margin
            if is_window:
                base = win_lit if lit_map[gy % grid, gx % grid] else win_dark
            else:
                base = concrete + rng.uniform(-0.02, 0.02, size=3)  # шум лише в бетоні
            c = np.clip(base, 0.0, 1.0)
            img.setXelA(x, y, float(c[0]), float(c[1]), float(c[2]), 1.0)

    tex = Texture("facade")
    tex.load(img)
    tex.setWrapU(Texture.WMRepeat)
    tex.setWrapV(Texture.WMRepeat)
    return tex


def build_building(
    parent: NodePath, width: float = 6.0, depth: float = 6.0, height: float = 8.0, seed: int = 0
) -> NodePath:
    """Будівля: 4 текстуровані стіни (фасад із вікнами) + плаский дах — база в
    локальному z=0 (стоїть на землі). Процедурна, без готових ассетів."""
    from panda3d.core import CardMaker, Point2

    building = parent.attachNewNode("building")
    tex = _make_facade_texture(seed=seed)
    hw, hd = width / 2, depth / 2
    floors = max(1.0, height / 3.0)

    # 4 стіни (двобічні — не залежимо від намотки/HPR-конвенції граней).
    # БАГ (реальний фідбек: "не відрізниш, на яку сторону падає сонце"):
    # CardMaker дає картку з нормаллю (0,-1,0) за замовчуванням; heading=0 ->
    # нормаль -Y, heading=90 -> +X (перевірено ЕМПІРИЧНО через
    # getRelativeVector, SKILL.md пастка "не виводь HPR теоретично"). Перша
    # версія мала ОДНАКОВИЙ heading для ПРОТИЛЕЖНИХ стін (перед/зад обидві
    # heading=0.0, право/ліво обидві heading=90.0) — тож у половини стін
    # нормаль дивилась ВСЕРЕДИНУ будівлі: `setTwoSided(True)` ховає симптом
    # "стіна невидима", але освітлення й далі рахувалось за НЕПРАВИЛЬНОЮ
    # нормаллю — будівля виглядала однаково затіненою з усіх боків, бо
    # половина стін отримувала світло "як би ззаду".
    walls = (
        (180.0, (0.0, hd, 0.0), width),   # перед (+Y), нормаль +Y
        (0.0, (0.0, -hd, 0.0), width),    # зад (-Y), нормаль -Y
        (90.0, (hw, 0.0, 0.0), depth),    # правий бік (+X), нормаль +X
        (-90.0, (-hw, 0.0, 0.0), depth),  # лівий бік (-X), нормаль -X
    )
    for heading, pos, span in walls:
        cm = CardMaker("wall")
        cm.setFrame(-span / 2, span / 2, 0.0, height)
        cm.setUvRange(Point2(0.0, 0.0), Point2(max(1.0, span / 3.0), floors))
        wall = building.attachNewNode(cm.generate())
        wall.setH(heading)
        wall.setPos(*pos)
        wall.setTexture(tex)
        wall.setMaterial(mat.CONCRETE)

    roof = add_box_visual(building, (hw, hd, 0.15), color=(0.28, 0.28, 0.3, 1.0), material=mat.METAL_PAINTED)
    roof.setPos(0, 0, height + 0.15)
    building.setTwoSided(True)
    return building


def build_vehicle(parent: NodePath, color: tuple[float, float, float, float] = (0.28, 0.32, 0.22, 1.0)) -> NodePath:
    """Техніка: спершу пробує ДЕТАЛЬНУ Blender-авторовану модель
    (``assets/models/vehicle.bam`` — корпус зі скошеним носом/глясісом,
    вежа з люком, ствол з дульним гальмом, 8 коліс з окремими ободами,
    гусениці; доп. фаза "графіка 2.0", ``assets/blender/build_vehicle.py``).
    Якщо файл відсутній (свіжий клон репо, Blender-пайплайн ще не
    запускали) — падає на процедурний білдер нижче (коробки+циліндри,
    доп. фаза "реальні моделі"). ``color`` застосовується ЛИШЕ до
    процедурного fallback (Blender-модель має власні матеріали)."""
    detailed = load_model_asset(parent, "vehicle")
    if detailed is not None:
        return detailed
    return _build_vehicle_procedural(parent, color)


def _build_vehicle_procedural(parent: NodePath, color: tuple[float, float, float, float]) -> NodePath:
    """Fallback: корпус+кабіна (коробки) + колеса (циліндри) + вежа/ствол —
    впізнаваний силует військової техніки (ціль strike_range, доп. фаза
    "реальні моделі"; НЕ людська фігура — свідоме обмеження обсягу)."""
    vehicle = parent.attachNewNode("vehicle")

    hull = add_box_visual(vehicle, (1.2, 0.7, 0.35), color=color, material=mat.METAL_PAINTED)
    hull.setPos(0, 0, 0.55)

    cabin = add_box_visual(vehicle, (0.5, 0.55, 0.3), color=color, material=mat.METAL_PAINTED)
    cabin.setPos(-0.3, 0, 1.15)

    turret = add_box_visual(vehicle, (0.35, 0.35, 0.2), color=color, material=mat.METAL_PAINTED)
    turret.setPos(0.3, 0, 1.05)
    barrel = build_cylinder_mesh(0.05, 0.9, segments=8, color=(0.15, 0.15, 0.15, 1.0), material=mat.METAL_BARE)
    barrel.reparentTo(vehicle)
    # Циліндр за замовчуванням витягнутий вздовж локальної Z (вгору) — HPR
    # (90,90,0) перевірено ЕМПІРИЧНО (top-down скріншот, не виведено з формул
    # HPR — SKILL.md пастка "не переносити конвенцію знаку/осі без перевірки"),
    # щоб напрямок був уздовж +X ("вперед" силуету техніки — довга вісь корпусу).
    barrel.setHpr(90, 90, 0)
    barrel.setPos(0.65, 0, 1.05)

    for x_pos in (-0.75, -0.25, 0.25, 0.75):
        for y_sign in (-1, 1):
            wheel = build_cylinder_mesh(0.3, 0.2, segments=10, color=(0.08, 0.08, 0.08, 1.0), material=mat.RUBBER)
            wheel.reparentTo(vehicle)
            wheel.setP(90)  # лягти циліндр набік (вісь колеса вздовж бортів, не вгору)
            wheel.setPos(x_pos, y_sign * 0.7, 0.3)
    return vehicle


def build_artillery(parent: NodePath, color: tuple[float, float, float, float] = (0.3, 0.28, 0.2, 1.0)) -> NodePath:
    """Артилерія: спершу пробує ДЕТАЛЬНУ Blender-модель
    (``assets/models/artillery.bam`` — основа/лафет/захисний щит/ствол з
    дульним гальмом/колеса, доп. фаза "графіка 2.0",
    ``assets/blender/build_artillery.py``); якщо файл відсутній — падає на
    процедурний білдер нижче (той самий патерн, що й ``build_vehicle``)."""
    detailed = load_model_asset(parent, "artillery")
    if detailed is not None:
        return detailed
    return _build_artillery_procedural(parent, color)


def _build_artillery_procedural(parent: NodePath, color: tuple[float, float, float, float]) -> NodePath:
    """Fallback: основа + ствол під кутом — впізнаваний силует гарматної
    установки (ціль strike_range, доп. фаза "реальні моделі")."""
    artillery = parent.attachNewNode("artillery")

    base = add_box_visual(artillery, (0.8, 0.8, 0.25), color=color, material=mat.METAL_PAINTED)
    base.setPos(0, 0, 0.25)

    mount = add_box_visual(artillery, (0.3, 0.3, 0.25), color=color, material=mat.METAL_PAINTED)
    mount.setPos(0, 0, 0.65)

    barrel = build_cylinder_mesh(0.08, 1.8, segments=8, color=(0.12, 0.12, 0.12, 1.0), material=mat.METAL_BARE)
    barrel.reparentTo(artillery)
    # Нахил у Y-Z площині (перевірено емпірично видом збоку вздовж X — з
    # top-down/деяких кутів камери нахил здається "вертикальним" через
    # ракурсне скорочення, це не помилка орієнтації).
    barrel.setP(-35)
    barrel.setPos(0, 0, 0.7)

    return artillery


def attach_scenery_visual(
    obstacle_np: NodePath, half_extents: tuple[float, float, float], kind: str
) -> NodePath:
    """Прикріпити візуальну модель за ``kind`` (``world/scene.py``, доп. фаза
    "реальні моделі") як дочірній вузол уже розміщеного фізичного тіла
    ``obstacle_np`` — розмір моделі підганяється під ``half_extents`` цього
    тіла, де це має сенс (фізика лишається простою коробкою незалежно від
    ``kind``, докладніше докстрінг модуля ``world/scene.py``).

    ``kind="box"`` (за замовчуванням) — стара поведінка, коробка-заглушка,
    БЕЗ z-зсуву (``add_box_visual`` вже центрований так само, як фізичне тіло).
    Усі ІНШІ будівники очікують свій ЛОКАЛЬНИЙ z=0 як "база на землі", тоді як
    ``obstacle_np`` — ЦЕНТР фізичної коробки (``pos.z == half_extents[2]`` у
    наявних сценах, тобто дно коробки на землі) — тому для них потрібен зсув
    ``-half_extents[2]`` вниз, інакше модель "висіла" б над землею.
    """
    hx, hy, hz = half_extents
    if kind == "box":
        return add_box_visual(obstacle_np, half_extents, color=(0.55, 0.35, 0.25, 1.0), material=mat.GENERIC)

    if kind in ("tree", "tree_pine", "tree_oak", "tree_birch"):
        if kind == "tree":
            # Автоматична різноманітність видів (реальний фідбек: "різні
            # види дерев") — детерміновано за ПОЗИЦІЄЮ обстакла (не за
            # окремим полем у YAML, щоб наявні сцени отримали різноманіття
            # без жодних змін даних), тож той самий обстакл завжди дає той
            # самий вид (відтворюваність не залежить від порядку побудови).
            pos = obstacle_np.getPos()
            species_idx = int(abs(hash((round(pos.x, 2), round(pos.y, 2)))) % 3)
            species = ("pine", "oak", "birch")[species_idx]
        else:
            species = kind.split("_", 1)[1]  # явний вид: tree_oak -> "oak"
        model = build_tree(
            obstacle_np,
            trunk_height=hz * 2.0,
            trunk_radius=min(hx, hy) * 0.3,
            foliage_radius=max(hx, hy),
            species=species,
        )
    elif kind == "fence":
        model = build_fence_line(obstacle_np, (-hx, 0.0), (hx, 0.0), post_height=hz * 2.0)
    elif kind == "trench":
        model = build_trench(obstacle_np, length=hx * 2.0, width=hy * 2.0, wall_height=hz * 2.0)
    elif kind == "road":
        model = build_road_strip(obstacle_np, length=hx * 2.0, width=hy * 2.0)
    elif kind == "building":
        model = build_building(obstacle_np, width=hx * 2.0, depth=hy * 2.0, height=hz * 2.0)
    elif kind == "vehicle":
        model = build_vehicle(obstacle_np)
    elif kind == "artillery":
        model = build_artillery(obstacle_np)
    else:
        return add_box_visual(obstacle_np, half_extents, color=(0.55, 0.35, 0.25, 1.0), material=mat.GENERIC)

    # Дорога — плаский декаль: садимо трохи ВИЩЕ рівня землі (світовий z≈0.06),
    # решта моделей — базою рівно на землю (світовий z=0).
    model.setZ(-hz + 0.06 if kind == "road" else -hz)
    return model

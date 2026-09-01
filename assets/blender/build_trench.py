"""Blender-скрипт (bpy), що будує ДЕТАЛЬНИЙ окіп/траншею — заміна старої
заглушки (2 голі коробки, render/models.py::build_trench, доп. фаза
"ландшафт 2.0", останній відкритий пункт). Реальний фідбек на стару версію:
"2 паралелепіпеди, зробимо якісніше пізніше" (assets/scenes/strike_range.yaml).

ЗАПУСК: виконується ВСЕРЕДИНІ Blender (MCP execute_blender_code або
``blender --background --python assets/blender/build_trench.py``) — скрипт
нічого не знає про Panda3D/проєкт, лише будує ОДИН об'єкт ``trench`` і
експортує ``assets/models/trench.glb``. Конвертація .glb -> .bam — окремий
крок через ``gltf2bam`` (той самий пайплайн, що ``build_vehicle.py``/
``build_trees.py``).

ЕТИКА (SKILL.md розділ 0): НЕМАЄ геометрії людських фігур — лише земля/
дерево/тканина.

**Архітектурне обмеження, що визначає підхід (docs/DECISIONS.md):** рельєф —
суцільний heightmap-меш, і в робочій зоні навколо obstacle висота ПРИМУСОВО
0 (пласко, для геймплейної читабельності) — тому СПРАВЖНЯ яма нижче z=0 під
траншеєю була б прихована під цим пласким мешем землі зверху (він не вирізає
дірку під obstacle). Тому, як і стара заглушка, це ВІЗУАЛЬНА ілюзія
заглиблення через ПІДНЯТІ земляні вали-бруствери з проходом між ними на
рівні землі — лише тепер деталізовано (не голі коробки):
- Земляний вал — bmesh loft вздовж довжини з трапецієподібним перетином
  (широка основа, вузький гребінь, похиліша грань до проходу) і радіальним
  джиттером на кожну "станцію" — та сама техніка, що кільця стовбура дерева
  (``build_trees.py::_build_trunk``), тут прямокутний перетин замість
  круглого.
- Кінці вала звужуються до 0 висоти (рампа) — сегмент "вливається" в землю
  на стику з сусіднім, не обривається вертикальною стіною.
- Мішки з піском на гребені — ряд перекритих джитерених форм (та сама
  техніка перекритих ікосфер, що крона дерева, ``_build_foliage_clump``).
- Дерев'яне кріплення (ревітмент) на похилій до проходу грані — вертикальні
  стовпчики + горизонтальна дошка, тайлинг вздовж лінії (як паркан).
- Дошки-настил на проході між валами, на рівні землі.
"""

import math
import random

import bmesh
import bpy

_OUT_DIR = "D:/PycharmProjects/drone-training/assets/models"

# Референсні розміри (мають збігатись із _BAKED_TRENCH_DIMS у
# render/models.py) — щоб render/models.py масштабував запитані
# length/width/wall_height відносно САМЕ цих значень.
_LENGTH = 8.0
_WIDTH = 2.4  # відстань між зовнішніми основами валів (прохід + вали)
_WALL_HEIGHT = 1.0


def _clear_scene() -> None:
    """Пряме видалення через data-API, НЕ ``bpy.ops.object.select_all``/
    ``delete`` — мовчки no-op без інтерактивного контексту вікна/area при
    виконанні через MCP execute_blender_code (виявлено емпірично в
    build_trees.py)."""
    for obj in list(bpy.data.objects):
        bpy.data.objects.remove(obj, do_unlink=True)
    for mesh in list(bpy.data.meshes):
        if mesh.users == 0:
            bpy.data.meshes.remove(mesh)
    for coll in list(bpy.data.collections):
        if coll.name != "Collection":
            bpy.data.collections.remove(coll)


def _make_material(name: str, base_color, roughness: float, metallic: float = 0.0):
    mat = bpy.data.materials.new(name)
    mat.use_nodes = True
    bsdf = mat.node_tree.nodes.get("Principled BSDF")
    bsdf.inputs["Base Color"].default_value = (*base_color, 1.0)
    bsdf.inputs["Roughness"].default_value = roughness
    bsdf.inputs["Metallic"].default_value = metallic
    return mat


def _build_berm(name, side, length, width, wall_height, material, seed, ramp_len=1.0, stations_per_m=1.4):
    """Один земляний вал: bmesh loft вздовж X, трапецієподібний перетин у
    площині Y-Z, з радіальним джиттером на кожну станцію (жива нерівна
    земля, не токарна поверхня) і рампою на обох кінцях (звужується до 0
    висоти) — ``side`` = -1 (лівий, y від'ємний) чи +1 (правий).

    Перетин параметризовано ЯВНИМИ горизонтальними "пробігами" схилів
    (``inner_run``/``outer_run``), не відносними множниками — перша версія
    рахувала внутрішню (до проходу) грань як різницю двох майже однакових
    відсотків ширини, що на практиці дало прогін лише ~0.12м на 1м висоти
    (~83° від горизонталі, майже прямовисна стіна). Під майже зенітним
    сонцем (``render/engine.py``, pitch -60°) такий крутий схил ловить
    світло лише "на просвіт" і читається майже чорним з будь-якого польотного
    ракурсу (перевірено скріншотом + піксельним семплінгом у Panda3D,
    порівняно з уже прийнятим деревом — там теж є темні ділянки, але яскрава
    крона дає контраст; тут была суцільна темна маса без цього). Явний
    пологіший прогін (типово ~0.5-0.6м на 1м висоти, ~55-60° від
    горизонталі) ловить світло природніше."""
    rng = random.Random(seed)
    ridge_half = 0.15  # половина ширини гребеня (плаский верх)
    inner_run = 0.55  # горизонтальний прогін внутрішнього (до проходу) схилу
    outer_run = 0.65  # горизонтальний прогін зовнішнього схилу

    n_stations = max(4, int(length * stations_per_m))
    bm = bmesh.new()
    rings = []
    for i in range(n_stations + 1):
        t = i / n_stations
        x = -length / 2 + length * t
        # Висота: рампа на кінцях, повна висота в середині.
        dist_from_end = min(x + length / 2, length / 2 - x)
        h_scale = min(1.0, dist_from_end / ramp_len) if ramp_len > 0 else 1.0
        h = wall_height * h_scale
        jitter = 1.0 + rng.uniform(-0.1, 0.1)
        walkway_edge = width / 2
        ridge_in = walkway_edge + inner_run * jitter
        ridge_out = ridge_in + 2 * ridge_half
        base_out = ridge_out + outer_run * jitter
        base_in = side * walkway_edge
        ridge_in = side * ridge_in
        ridge_out = side * ridge_out
        base_out = side * base_out
        z_jit = rng.uniform(-0.02, 0.02) * wall_height
        # 4 вершини перетину: основа-зовні, основа-всередині (прохід),
        # гребінь-всередині, гребінь-зовні (проста трапеція, дно на z=0).
        v_base_out = bm.verts.new((x, base_out, 0.0))
        v_base_in = bm.verts.new((x, base_in, 0.0))
        v_ridge_in = bm.verts.new((x, ridge_in, h + z_jit))
        v_ridge_out = bm.verts.new((x, ridge_out, h + z_jit))
        rings.append((v_base_out, v_base_in, v_ridge_in, v_ridge_out))

    # Торці (перший/останній перетин) — щоб вал не мав дірок з боків.
    bm.faces.new(rings[0] if side > 0 else tuple(reversed(rings[0])))
    bm.faces.new(tuple(reversed(rings[-1])) if side > 0 else rings[-1])

    for i in range(n_stations):
        a, b = rings[i], rings[i + 1]
        # 4 грані периметра перетину: зовнішня похила, верх (гребінь),
        # внутрішня похила (до проходу), низ (на землі, невидимий, але
        # потрібен для замкнутого мешу/нормалей).
        quads = ((0, 1), (1, 2), (2, 3), (3, 0))
        for s0, s1 in quads:
            face = (a[s0], a[s1], b[s1], b[s0]) if side > 0 else (a[s1], a[s0], b[s0], b[s1])
            bm.faces.new(face)

    bm.normal_update()
    mesh = bpy.data.meshes.new(name)
    bm.to_mesh(mesh)
    bm.free()
    obj = bpy.data.objects.new(name, mesh)
    bpy.context.collection.objects.link(obj)
    obj.data.materials.append(material)
    return obj


def _build_sandbag_row(name, side, length, width, wall_height, material, seed, spacing=0.55, ramp_len=1.0):
    """Мішки з піском на гребені вала — ряд перекритих джитерених кубів
    (та сама техніка, що перекриті ікосфери крони дерева) уздовж гребеня,
    лише в межах повної висоти (не на рампах кінців)."""
    rng = random.Random(seed)
    usable = max(0.0, length - 2 * ramp_len)
    n_bags = max(2, int(usable / spacing))
    bags = []
    y = side * (width / 2 + 0.70)  # центр гребеня (_build_berm: inner_run=0.55 + ridge_half=0.15)
    z = wall_height + 0.10
    for i in range(n_bags):
        t = i / max(1, n_bags - 1)
        x = -usable / 2 + usable * t
        bpy.ops.mesh.primitive_cube_add(size=1.0, location=(x, y, z))
        bag = bpy.context.active_object
        bag.name = f"{name}_{i}"
        bag.scale = (
            spacing * 0.42 * (1.0 + rng.uniform(-0.15, 0.15)),
            0.20 * (1.0 + rng.uniform(-0.15, 0.20)),
            0.14 * (1.0 + rng.uniform(-0.15, 0.15)),
        )
        bag.rotation_euler = (0.0, 0.0, math.radians(rng.uniform(-8, 8)))
        bpy.ops.object.transform_apply(location=False, rotation=True, scale=True)
        bag.data.materials.append(material)
        bags.append(bag)
    return bags


def _build_revetment(name, side, length, width, wall_height, material, seed, post_spacing=1.3, ramp_len=1.0):
    """Дерев'яне кріплення внутрішньої (до проходу) грані вала: вертикальні
    стовпчики + одна горизонтальна дошка — тайлинг вздовж лінії, той самий
    патерн, що ``render/models.py::build_fence_line``."""
    rng = random.Random(seed)
    usable = max(0.0, length - 2 * ramp_len)
    n_posts = max(2, int(usable / post_spacing) + 1)
    parts = []
    y = side * (width / 2 - 0.05)
    for i in range(n_posts):
        t = i / max(1, n_posts - 1)
        x = -usable / 2 + usable * t
        h = wall_height * rng.uniform(0.85, 1.0)
        bpy.ops.mesh.primitive_cube_add(size=1.0, location=(x, y, h / 2))
        post = bpy.context.active_object
        post.name = f"{name}_post{i}"
        post.scale = (0.05, 0.05, h / 2)
        bpy.ops.object.transform_apply(location=False, rotation=False, scale=True)
        post.data.materials.append(material)
        parts.append(post)

    bpy.ops.mesh.primitive_cube_add(size=1.0, location=(0.0, y, wall_height * 0.55))
    rail = bpy.context.active_object
    rail.name = f"{name}_rail"
    rail.scale = (usable / 2 * 0.98, 0.04, 0.08)
    bpy.ops.object.transform_apply(location=False, rotation=False, scale=True)
    rail.data.materials.append(material)
    parts.append(rail)
    return parts


def _build_duckboards(name, length, width, material, seed, plank_width=0.22, ramp_len=1.0):
    """Дошки-настил на проході між валами, на рівні землі (z трохи вище 0,
    проти z-fighting із землею — та сама причина, що дорога/паркан)."""
    rng = random.Random(seed)
    usable = max(0.0, length - 2 * ramp_len)
    walk_width = width - 2 * 0.15
    n_planks = max(2, int(walk_width / (plank_width + 0.03)))
    parts = []
    for i in range(n_planks):
        y = -walk_width / 2 + (i + 0.5) * (walk_width / n_planks)
        z = 0.03 + rng.uniform(-0.01, 0.01)  # легка нерівність настилу
        bpy.ops.mesh.primitive_cube_add(size=1.0, location=(0.0, y, z))
        plank = bpy.context.active_object
        plank.name = f"{name}_{i}"
        plank.scale = (usable / 2 * 0.99, plank_width / 2 * 0.92, 0.03)
        bpy.ops.object.transform_apply(location=False, rotation=False, scale=True)
        plank.data.materials.append(material)
        parts.append(plank)
    return parts


def _join_flat(name, parts):
    """Об'єднати перелічені об'єкти в один БЕЗ згладжування — вал має лише
    4 точки на перетин (гострий низькополі трапецієподібний профіль, не
    кругла форма з багатьма сегментами, як стовбур дерева) — плоске
    затінення (кожна грань — власна нормаль) тут коректніше, той самий
    підхід, що корпус техніки в build_vehicle.py (тверда гранчаста форма,
    не органічна крона)."""
    bpy.ops.object.select_all(action="DESELECT")
    for p in parts:
        p.select_set(True)
    bpy.context.view_layer.objects.active = parts[0]
    bpy.ops.object.join()
    result = bpy.context.active_object
    result.name = name
    return result


def _export(obj, out_name):
    bpy.ops.object.select_all(action="DESELECT")
    obj.select_set(True)
    bpy.context.view_layer.objects.active = obj
    out_path = f"{_OUT_DIR}/{out_name}.glb"
    bpy.ops.export_scene.gltf(
        filepath=out_path, export_format="GLB", use_selection=True,
        export_apply=True, export_yup=True,
    )
    return out_path


def build_trench(seed: int = 1) -> str:
    _clear_scene()
    dirt = _make_material("trench_dirt", (0.35, 0.28, 0.20), roughness=0.95)
    sandbag = _make_material("trench_sandbag", (0.55, 0.50, 0.38), roughness=0.90)
    wood = _make_material("trench_wood", (0.32, 0.22, 0.12), roughness=0.85)

    length, width, wall_height = _LENGTH, _WIDTH, _WALL_HEIGHT
    ramp_len = 1.0

    parts = []
    for side in (-1, 1):
        parts.append(
            _build_berm("berm", side, length, width, wall_height, dirt, seed=seed * 7 + side, ramp_len=ramp_len)
        )
        parts += _build_sandbag_row(
            "sandbag", side, length, width, wall_height, sandbag, seed=seed * 11 + side, ramp_len=ramp_len
        )
        parts += _build_revetment(
            "revetment", side, length, width, wall_height, wood, seed=seed * 13 + side, ramp_len=ramp_len
        )
    parts += _build_duckboards("duckboard", length, width, wood, seed=seed * 17, ramp_len=ramp_len)

    trench = _join_flat("trench", parts)
    return _export(trench, "trench")


def build() -> None:
    global result
    result = {"status": "ok", "exported": build_trench()}


if __name__ == "__main__":
    build()

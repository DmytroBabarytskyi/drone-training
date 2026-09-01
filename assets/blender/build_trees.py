"""Blender-скрипт (bpy), що будує 3 РЕАЛІСТИЧНІ види дерев — доп. фаза
"реальні моделі" (детальні органічні меші, замість конусів+циліндра
render/models.py::_build_pine_tree/_build_oak_tree/_build_birch_tree).

ЗАПУСК: виконується ВСЕРЕДИНІ Blender (через MCP execute_blender_code або
``blender --background --python assets/blender/build_trees.py``) — сам
скрипт нічого не знає про Panda3D/проєкт, лише будує 3 окремі об'єкти й
ЕКСПОРТУЄ їх у ``assets/models/tree_pine.glb`` / ``tree_oak.glb`` /
``tree_birch.glb``. Конвертація .glb -> .bam — окремий крок через
``gltf2bam`` (panda3d-gltf), не тут (той самий пайплайн, що й
``build_vehicle.py``/``build_artillery.py``).

ЕТИКА (SKILL.md розділ 0): НЕМАЄ геометрії людських фігур — лише дерева.

Техніка "реалістичності" (на відміну від старих конусів-стеків):
- Стовбур — НЕ ідеальний циліндр: сітка кілець зі звуженням до вершини,
  ЛЕГКИМ нахилом (кумулятивний зсув по висоті) і радіальним джиттером
  на кожне кільце — жива, нерівна кора, не токарний виріб.
- Крона — НЕ суцільний конус/куля: кілька (6-14) перекритих ікосфер
  випадкового розміру/зсуву, ОБ'ЄДНАНИХ (``bpy.ops.object.join``) в один
  меш — силует лишається лапатим/лумпи, як справжнє листя/хвоя, а не
  геометрична примітива.
- Кожен вид має ВЛАСНУ структуру (не один параметризований генератор):
  сосна — вузькі яруси хвої, що звужуються до вершини; дуб — товстий
  стовбур, кілька товстих гілок, одна широка кругла крона; береза —
  тонкий блідий стовбур із темними горизонтальними "мітками" кори (плоскі
  темні смужки), рідша ажурна крона.
"""

import math
import random

import bmesh
import bpy

_OUT_DIR = "D:/PycharmProjects/drone-training/assets/models"


def _clear_scene() -> None:
    """Пряме видалення через data-API, НЕ ``bpy.ops.object.select_all``/
    ``delete`` — ці оператори мовчки НЕ СПРАЦЬОВУЮТЬ (no-op, без помилки),
    коли скрипт виконується через MCP execute_blender_code (немає звичного
    інтерактивного контексту вікна/area, від якого залежить `poll()`
    оператора). Виявлено емпірично: 3 послідовні білди дали ``trunk``,
    ``trunk.001``, ``trunk.002`` замість трьох окремих ``trunk`` — стара
    сцена НЕ очищалась між викликами. Видалення напряму через
    ``bpy.data.objects.remove`` не залежить від контексту вікна."""
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


def _build_trunk(name, height, base_radius, top_radius, segments, rings, lean, material, seed):
    """Органічний стовбур: кільця зі звуженням, кумулятивний нахил (не
    прямовисний стрижень) і радіальний джиттер на кожне кільце (не гладка
    токарна поверхня)."""
    rng = random.Random(seed)
    bm = bmesh.new()
    ring_verts = []
    for ring in range(rings + 1):
        t = ring / rings
        z = height * t
        radius = base_radius + (top_radius - base_radius) * t
        lean_x = lean * (t ** 1.6) * height
        wob_x = rng.uniform(-0.015, 0.015) * height
        wob_y = rng.uniform(-0.015, 0.015) * height
        ring_jitter = 1.0 + rng.uniform(-0.10, 0.10)
        row = []
        for seg in range(segments):
            ang = 2 * math.pi * seg / segments
            r = radius * ring_jitter * (1.0 + 0.06 * math.sin(ang * 3.0 + seed + ring))
            x = math.cos(ang) * r + lean_x + wob_x
            y = math.sin(ang) * r + wob_y
            row.append(bm.verts.new((x, y, z)))
        ring_verts.append(row)

    # Низ (дно стовбура) — віяловий полігон.
    bm.faces.new(reversed(ring_verts[0]))
    for ring in range(rings):
        for seg in range(segments):
            s2 = (seg + 1) % segments
            a, b = ring_verts[ring][seg], ring_verts[ring][s2]
            c, d = ring_verts[ring + 1][seg], ring_verts[ring + 1][s2]
            bm.faces.new((a, b, d, c))
    # Верх — якщо top_radius майже 0, просто з'єднати в полюс.
    if top_radius < 0.02:
        top_center = bm.verts.new((ring_verts[-1][0].co.x, ring_verts[-1][0].co.y, height))
        for seg in range(segments):
            s2 = (seg + 1) % segments
            bm.faces.new((ring_verts[-1][seg], ring_verts[-1][s2], top_center))
    else:
        bm.faces.new(ring_verts[-1])

    bm.normal_update()
    mesh = bpy.data.meshes.new(name)
    bm.to_mesh(mesh)
    bm.free()
    obj = bpy.data.objects.new(name, mesh)
    bpy.context.collection.objects.link(obj)
    obj.data.materials.append(material)
    return obj


def _build_branch(name, origin, length, radius0, radius1, pitch_deg, yaw_deg, material, seed):
    """Тонка гілка-відросток: тейперований циліндр, вирощений з origin під
    кутом (pitch — нахил від стовбура вгору, yaw — навколо стовбура)."""
    bpy.ops.mesh.primitive_cone_add(
        radius1=radius0, radius2=radius1, depth=length, vertices=7,
        location=(0, 0, length / 2.0),
    )
    branch = bpy.context.active_object
    branch.name = name
    bpy.ops.object.transform_apply(location=True, rotation=False, scale=False)
    branch.rotation_euler = (math.radians(90 - pitch_deg), 0.0, math.radians(yaw_deg))
    bpy.ops.object.transform_apply(location=False, rotation=True, scale=False)
    branch.location = origin
    branch.data.materials.append(material)
    return branch


def _build_foliage_clump(name, center, radius, material, seed, n_blobs=8, squash=1.0, spread=0.55, subdiv=2):
    """Крона з перекритих ікосфер (не суцільна куля/конус) — ОБ'ЄДНАНИХ в
    один об'єкт. Перекриття різних розмірів/зсувів дає лапатий, органічний
    силует замість ідеальної примітиви."""
    rng = random.Random(seed)
    blobs = []
    for i in range(n_blobs):
        r = radius * rng.uniform(0.5, 0.95)
        ox = rng.uniform(-radius * spread, radius * spread)
        oy = rng.uniform(-radius * spread, radius * spread)
        oz = rng.uniform(-radius * spread * 0.6, radius * spread * 0.7)
        bpy.ops.mesh.primitive_ico_sphere_add(
            radius=r, subdivisions=subdiv,
            location=(center[0] + ox, center[1] + oy, center[2] + oz),
        )
        blob = bpy.context.active_object
        blob.name = f"{name}_blob{i}"
        blob.scale = (
            1.0 + rng.uniform(-0.1, 0.1),
            1.0 + rng.uniform(-0.1, 0.1),
            squash * (1.0 + rng.uniform(-0.15, 0.1)),
        )
        bpy.ops.object.transform_apply(location=False, rotation=False, scale=True)
        blob.data.materials.append(material)
        blobs.append(blob)

    bpy.ops.object.select_all(action="DESELECT")
    for b in blobs:
        b.select_set(True)
    bpy.context.view_layer.objects.active = blobs[0]
    bpy.ops.object.join()
    foliage = bpy.context.active_object
    foliage.name = name
    return foliage


def _join_all(name, parts):
    bpy.ops.object.select_all(action="DESELECT")
    for p in parts:
        p.select_set(True)
    bpy.context.view_layer.objects.active = parts[0]
    bpy.ops.object.join()
    result = bpy.context.active_object
    result.name = name
    # Гладке затінення (не плоский "лоу-полі" вигляд ікосфер) — реальний
    # фідбек: "красиві, реалістичні дерева". Силует лишається лапатим
    # (перекриті ікосфери), але поверхня читається органічно, не гранчасто.
    bpy.ops.object.shade_smooth()
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


def build_pine(seed: int = 1):
    """Сосна: вузький високий стовбур, 5 ярусів хвої, що звужуються до
    вершини (типовий хвойний силует, але лапатий, не суцільний конус)."""
    _clear_scene()
    bark = _make_material("pine_bark", (0.20, 0.14, 0.10), roughness=0.95)
    needles = _make_material("pine_needles", (0.05, 0.15, 0.06), roughness=0.85)

    height = 3.4
    trunk = _build_trunk("trunk", height, 0.14, 0.035, segments=8, rings=7, lean=0.04, material=bark, seed=seed)

    parts = [trunk]
    n_tiers = 5
    for i in range(n_tiers):
        t = i / (n_tiers - 1)
        z = height * (0.38 + 0.58 * t)
        tier_radius = 0.95 * (1.0 - 0.72 * t)
        clump = _build_foliage_clump(
            f"tier_{i}", (0.0, 0.0, z), tier_radius, needles, seed=seed * 17 + i,
            n_blobs=6, squash=0.55, spread=0.65, subdiv=2,
        )
        parts.append(clump)

    tree = _join_all("tree_pine", parts)
    return _export(tree, "tree_pine")


def build_oak(seed: int = 2):
    """Дуб: товстий короткий стовбур, кілька товстих гілок під кутом, ОДНА
    широка кругла лапата крона (не кілька приплюснутих конусів)."""
    _clear_scene()
    bark = _make_material("oak_bark", (0.24, 0.19, 0.14), roughness=0.95)
    leaves = _make_material("oak_leaves", (0.15, 0.32, 0.09), roughness=0.85)

    height = 2.2
    trunk = _build_trunk("trunk", height, 0.24, 0.15, segments=10, rings=5, lean=0.06, material=bark, seed=seed)

    rng = random.Random(seed + 100)
    parts = [trunk]
    n_branches = 5
    for i in range(n_branches):
        t = rng.uniform(0.5, 0.85)
        z = height * t
        yaw = 360.0 * i / n_branches + rng.uniform(-20, 20)
        pitch = rng.uniform(28.0, 55.0)
        length = rng.uniform(0.55, 0.95)
        branch = _build_branch(
            f"branch_{i}", (0.0, 0.0, z), length, 0.05, 0.018, pitch, yaw, bark, seed=seed * 31 + i,
        )
        parts.append(branch)

    canopy = _build_foliage_clump(
        "canopy", (0.0, 0.0, height * 1.05), 1.55, leaves, seed=seed * 7,
        n_blobs=13, squash=0.78, spread=0.5, subdiv=2,
    )
    parts.append(canopy)

    tree = _join_all("tree_oak", parts)
    return _export(tree, "tree_oak")


def build_birch(seed: int = 3):
    """Береза: тонкий високий блідий стовбур із темними горизонтальними
    "мітками" кори, тонкі висхідні гілки, рідша ажурна світло-зелена крона."""
    _clear_scene()
    bark = _make_material("birch_bark", (0.80, 0.78, 0.74), roughness=0.7)
    bark_mark = _make_material("birch_bark_mark", (0.06, 0.05, 0.05), roughness=0.9)
    leaves = _make_material("birch_leaves", (0.38, 0.46, 0.14), roughness=0.85)

    height = 3.7
    trunk = _build_trunk("trunk", height, 0.09, 0.028, segments=8, rings=7, lean=0.025, material=bark, seed=seed)

    rng = random.Random(seed + 200)
    parts = [trunk]
    # Темні горизонтальні "мітки" кори — плоскі тонкі коробки-смужки,
    # частково обгорнуті навколо стовбура (характерна риса берези).
    for i in range(6):
        z = height * rng.uniform(0.1, 0.75)
        ang = rng.uniform(0.0, 360.0)
        r_at_z = 0.09 + (0.028 - 0.09) * (z / height)
        bpy.ops.mesh.primitive_cube_add(size=1.0, location=(0, 0, z))
        mark = bpy.context.active_object
        mark.name = f"bark_mark_{i}"
        mark.scale = (r_at_z * rng.uniform(0.9, 1.3), r_at_z * 0.25, 0.04)
        bpy.ops.object.transform_apply(location=False, rotation=False, scale=True)
        mark.rotation_euler = (0.0, 0.0, math.radians(ang))
        mark.location = (math.cos(math.radians(ang)) * r_at_z * 0.7, math.sin(math.radians(ang)) * r_at_z * 0.7, z)
        mark.data.materials.append(bark_mark)
        parts.append(mark)

    n_branches = 4
    for i in range(n_branches):
        t = rng.uniform(0.55, 0.85)
        z = height * t
        yaw = 360.0 * i / n_branches + rng.uniform(-25, 25)
        pitch = rng.uniform(45.0, 65.0)
        length = rng.uniform(0.4, 0.65)
        branch = _build_branch(
            f"branch_{i}", (0.0, 0.0, z), length, 0.025, 0.008, pitch, yaw, bark, seed=seed * 41 + i,
        )
        parts.append(branch)

    # Крона берези: РІДША за дуб, але має читатись як ОДНА айриста округла
    # маса вгорі стовбура (реальний фідбек на попередню версію: занадто
    # розкидані грудки виглядали як діагональна "пляма", не крона) — тому
    # ГОЛОВНИЙ великий центральний клубок (задає читаний силует) + кілька
    # МЕНШИХ бічних клубків близько до нього (додають ажурності на краях,
    # не ламають загальний силует).
    canopy_z = height * 0.86
    parts.append(_build_foliage_clump(
        "canopy_core", (0.0, 0.0, canopy_z), 0.62, leaves, seed=seed * 7,
        n_blobs=9, squash=0.8, spread=0.5, subdiv=2,
    ))
    n_side = 5
    for i in range(n_side):
        t_ang = rng.uniform(0.0, 2 * math.pi)
        r = rng.uniform(0.35, 0.55)
        z = canopy_z + rng.uniform(-0.25, 0.35) * height * 0.12
        cx, cy = math.cos(t_ang) * r, math.sin(t_ang) * r
        clump = _build_foliage_clump(
            f"cluster_{i}", (cx, cy, z), rng.uniform(0.28, 0.4), leaves, seed=seed * 53 + i,
            n_blobs=4, squash=0.85, spread=0.5, subdiv=1,
        )
        parts.append(clump)

    tree = _join_all("tree_birch", parts)
    return _export(tree, "tree_birch")


def build() -> None:
    results = {
        "pine": build_pine(),
        "oak": build_oak(),
        "birch": build_birch(),
    }
    global result
    result = {"status": "ok", "exported": results}


if __name__ == "__main__":
    build()

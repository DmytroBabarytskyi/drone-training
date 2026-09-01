"""Blender-скрипт (bpy): детальна модель артилерійської установки — доп.
фаза "графіка 2.0" (заміна коробок+циліндра render/models.py::build_artillery).
Той самий підхід/матеріали, що й build_vehicle.py (див. його докстрінг).

ЕТИКА (SKILL.md розділ 0): лише технічний силует (основа/лафет/ствол/щит),
без геометрії людських фігур.
"""

import math

import bpy

_OUT_PATH = "D:/PycharmProjects/drone-training/assets/models/artillery.glb"


def _clear_scene():
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete(use_global=False)
    for coll in list(bpy.data.collections):
        if coll.name != "Collection":
            bpy.data.collections.remove(coll)


def _make_material(name, base_color, roughness, metallic):
    mat = bpy.data.materials.new(name)
    mat.use_nodes = True
    bsdf = mat.node_tree.nodes.get("Principled BSDF")
    bsdf.inputs["Base Color"].default_value = (*base_color, 1.0)
    bsdf.inputs["Roughness"].default_value = roughness
    bsdf.inputs["Metallic"].default_value = metallic
    return mat


def _bevel(obj, width=0.02, segments=2):
    mod = obj.modifiers.new("bevel", "BEVEL")
    mod.width = width
    mod.segments = segments
    mod.limit_method = "ANGLE"
    mod.angle_limit = math.radians(35)
    bpy.ops.object.select_all(action="DESELECT")
    obj.select_set(True)
    bpy.context.view_layer.objects.active = obj
    bpy.ops.object.modifier_apply(modifier=mod.name)


def _box(name, size, location, scale, material):
    bpy.ops.mesh.primitive_cube_add(size=size, location=location)
    obj = bpy.context.active_object
    obj.name = name
    obj.scale = scale
    bpy.ops.object.transform_apply(location=False, rotation=False, scale=True)
    obj.data.materials.append(material)
    return obj


def _cylinder(name, radius, depth, location, rotation, material, vertices=10):
    bpy.ops.mesh.primitive_cylinder_add(radius=radius, depth=depth, location=location, vertices=vertices)
    obj = bpy.context.active_object
    obj.name = name
    obj.rotation_euler = rotation
    obj.data.materials.append(material)
    return obj


def build():
    _clear_scene()

    mat_paint = _make_material("hull_paint", (0.18, 0.20, 0.13), roughness=0.45, metallic=0.25)
    mat_bare = _make_material("metal_bare", (0.10, 0.10, 0.10), roughness=0.30, metallic=0.85)
    mat_rubber = _make_material("rubber", (0.03, 0.03, 0.03), roughness=0.95, metallic=0.0)

    base = _box("base", 1.0, (0.0, 0.0, 0.25), (0.8, 0.8, 0.25), mat_paint)
    _bevel(base, width=0.02, segments=2)

    mount = _box("mount", 1.0, (0.0, 0.0, 0.65), (0.28, 0.28, 0.25), mat_paint)
    _bevel(mount, width=0.015, segments=2)

    # Захисний щит (типовий для буксируваної артилерії) — нахилена плита перед лафетом.
    shield = _box("shield", 1.0, (0.55, 0.0, 0.85), (0.04, 0.75, 0.55), mat_bare)
    shield.rotation_euler = (0.0, math.radians(-12), 0.0)

    barrel = _cylinder("barrel", 0.07, 1.9, (0.0, 0.0, 0.7), (0.0, math.radians(90 - 35), 0.0), mat_bare, vertices=10)
    barrel.location = (
        math.cos(math.radians(35)) * 0.95,
        0.0,
        0.7 + math.sin(math.radians(35)) * 0.95,
    )

    muzzle = _cylinder("muzzle_brake", 0.10, 0.16, (0.0, 0.0, 0.0), (0.0, math.radians(90 - 35), 0.0), mat_bare, vertices=10)
    tip_dist = 1.8
    muzzle.location = (
        math.cos(math.radians(35)) * tip_dist,
        0.0,
        0.7 + math.sin(math.radians(35)) * tip_dist,
    )

    wheels = []
    for y_sign in (-1, 1):
        tire = _cylinder("wheel_tire", 0.32, 0.16, (-0.15, y_sign * 0.55, 0.32), (math.radians(90), 0.0, 0.0), mat_rubber, vertices=14)
        hub = _cylinder("wheel_hub", 0.14, 0.18, (-0.15, y_sign * 0.55, 0.32), (math.radians(90), 0.0, 0.0), mat_bare, vertices=10)
        wheels += [tire, hub]

    parts = [base, mount, shield, barrel, muzzle, *wheels]

    bpy.ops.object.select_all(action="DESELECT")
    for p in parts:
        p.select_set(True)
    bpy.context.view_layer.objects.active = parts[0]
    bpy.ops.object.join()
    obj = bpy.context.active_object
    obj.name = "artillery"

    bpy.ops.export_scene.gltf(
        filepath=_OUT_PATH,
        export_format="GLB",
        use_selection=False,
        export_apply=True,
        export_yup=True,
    )
    return {"status": "ok", "exported": _OUT_PATH, "objects": len(parts)}


result = build()

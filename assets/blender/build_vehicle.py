"""Blender-скрипт (bpy), що будує детальну модель військової техніки —
доп. фаза "графіка 2.0" (детальні меші, замість коробок+циліндрів
render/models.py::build_vehicle).

ЗАПУСК: виконується ВСЕРЕДИНІ Blender (через MCP execute_blender_code або
``blender --background --python assets/blender/build_vehicle.py``) — сам
скрипт нічого не знає про Panda3D/проєкт, лише будує сцену й ЕКСПОРТУЄ
``assets/models/vehicle.glb``. Конвертація .glb -> .bam (Panda3D-формат) —
окремий крок через ``gltf2bam`` (panda3d-gltf), не тут.

ЕТИКА (SKILL.md розділ 0, той самий принцип, що й render/models.py): НЕМАЄ
геометрії людських фігур — лише технічний силует (корпус/вежа/ствол/колеса).

Матеріали (Principled BSDF: base_color/roughness/metallic) підібрані ТАК
САМО, як пресети ``render/materials.py`` (METAL_PAINTED/METAL_BARE/RUBBER)
— щоб Blender-модель і процедурний fallback виглядали узгоджено.
"""

import math

import bmesh
import bpy

# Абсолютний шлях (НЕ "//"-відносний): сцена в цьому скрипті може бути ще
# не збереженою (.blend без імені файлу), тож відносний шлях від
# bpy.data.filepath був би ненадійним.
_OUT_PATH = "D:/PycharmProjects/drone-training/assets/models/vehicle.glb"


def _clear_scene() -> None:
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete(use_global=False)
    for coll in list(bpy.data.collections):
        if coll.name != "Collection":
            bpy.data.collections.remove(coll)


def _make_material(name: str, base_color, roughness: float, metallic: float):
    mat = bpy.data.materials.new(name)
    mat.use_nodes = True
    bsdf = mat.node_tree.nodes.get("Principled BSDF")
    bsdf.inputs["Base Color"].default_value = (*base_color, 1.0)
    bsdf.inputs["Roughness"].default_value = roughness
    bsdf.inputs["Metallic"].default_value = metallic
    return mat


def _bevel(obj, width: float = 0.02, segments: int = 2) -> None:
    mod = obj.modifiers.new("bevel", "BEVEL")
    mod.width = width
    mod.segments = segments
    mod.limit_method = "ANGLE"
    mod.angle_limit = math.radians(35)
    bpy.ops.object.select_all(action="DESELECT")
    obj.select_set(True)
    bpy.context.view_layer.objects.active = obj
    bpy.ops.object.modifier_apply(modifier=mod.name)


def _new_mesh_obj(name: str, verts, faces, material):
    mesh = bpy.data.meshes.new(name)
    bm = bmesh.new()
    bm_verts = [bm.verts.new(v) for v in verts]
    for f in faces:
        bm.faces.new([bm_verts[i] for i in f])
    bm.normal_update()
    bm.to_mesh(mesh)
    bm.free()
    obj = bpy.data.objects.new(name, mesh)
    bpy.context.collection.objects.link(obj)
    obj.data.materials.append(material)
    return obj


def _build_hull(mat_paint):
    """Корпус: трапецієподібний профіль (звужується й нижчає до носа —
    "glacis"-плита, як у реальної БМП/БТР), не проста коробка."""
    hl, hw, h_low, h_high = 1.2, 0.7, 0.35, 0.75  # half-length, half-width, висота носа/корми
    verts = [
        (-hl, -hw, 0.0), (-hl, hw, 0.0), (hl, -hw, 0.0), (hl, hw, 0.0),  # низ (прямокутник)
        (-hl, -hw, h_high), (-hl, hw, h_high),  # верх-корма (повна висота)
        (hl * 0.35, -hw, h_low), (hl * 0.35, hw, h_low),  # верх-перехід (глясіс починається тут)
        (hl, -hw * 0.7, h_low * 0.4), (hl, hw * 0.7, h_low * 0.4),  # верх-ніс (низько, вузько - скошений)
    ]
    faces = [
        (0, 1, 3, 2),  # низ
        (0, 4, 5, 1),  # корма
        (4, 6, 7, 5),  # дах
        (6, 8, 9, 7),  # глясіс
        (0, 2, 8, 6, 4),  # правий борт (низ->ніс->глясіс->дах->корма)
        (1, 5, 7, 9, 3),  # лівий борт
        (2, 3, 9, 8),  # ніс (торець)
    ]
    hull = _new_mesh_obj("hull", verts, faces, mat_paint)
    hull.location = (0.0, 0.0, 0.32)
    _bevel(hull, width=0.025, segments=2)
    return hull


def _build_turret(mat_paint, mat_bare):
    turret = bpy.ops.mesh.primitive_cube_add(size=1.0, location=(0.25, 0.0, 1.05))
    obj = bpy.context.active_object
    obj.name = "turret"
    obj.scale = (0.32, 0.32, 0.18)
    bpy.ops.object.transform_apply(location=False, rotation=False, scale=True)
    obj.data.materials.append(mat_paint)
    _bevel(obj, width=0.02, segments=2)

    bpy.ops.mesh.primitive_cylinder_add(radius=0.12, depth=0.08, location=(0.15, 0.0, 1.19), vertices=12)
    hatch = bpy.context.active_object
    hatch.name = "hatch"
    hatch.data.materials.append(mat_bare)

    bpy.ops.mesh.primitive_cylinder_add(radius=0.045, depth=0.85, location=(0.65, 0.0, 1.05), vertices=10)
    barrel = bpy.context.active_object
    barrel.name = "barrel"
    barrel.rotation_euler = (0.0, math.radians(90), 0.0)
    barrel.data.materials.append(mat_bare)

    bpy.ops.mesh.primitive_cylinder_add(radius=0.06, depth=0.12, location=(1.02, 0.0, 1.05), vertices=10)
    muzzle = bpy.context.active_object
    muzzle.name = "muzzle_brake"
    muzzle.rotation_euler = (0.0, math.radians(90), 0.0)
    muzzle.data.materials.append(mat_bare)

    return [obj, hatch, barrel, muzzle]


def _build_wheel(x, y, mat_rubber, mat_bare):
    bpy.ops.mesh.primitive_cylinder_add(radius=0.3, depth=0.22, location=(x, y, 0.3), vertices=14)
    tire = bpy.context.active_object
    tire.name = "wheel_tire"
    tire.rotation_euler = (math.radians(90), 0.0, 0.0)
    tire.data.materials.append(mat_rubber)

    bpy.ops.mesh.primitive_cylinder_add(radius=0.15, depth=0.24, location=(x, y, 0.3), vertices=10)
    hub = bpy.context.active_object
    hub.name = "wheel_hub"
    hub.rotation_euler = (math.radians(90), 0.0, 0.0)
    hub.data.materials.append(mat_bare)
    return [tire, hub]


def _build_track(y_side, mat_bare):
    bpy.ops.mesh.primitive_cube_add(size=1.0, location=(0.0, y_side * 0.72, 0.22))
    track = bpy.context.active_object
    track.name = f"track_{'r' if y_side > 0 else 'l'}"
    track.scale = (1.15, 0.08, 0.16)
    bpy.ops.object.transform_apply(location=False, rotation=False, scale=True)
    track.data.materials.append(mat_bare)
    return track


def build() -> None:
    _clear_scene()

    mat_paint = _make_material("hull_paint", (0.16, 0.19, 0.12), roughness=0.45, metallic=0.25)
    mat_bare = _make_material("metal_bare", (0.08, 0.08, 0.08), roughness=0.30, metallic=0.85)
    mat_rubber = _make_material("rubber", (0.03, 0.03, 0.03), roughness=0.95, metallic=0.0)

    parts = [_build_hull(mat_paint), *_build_turret(mat_paint, mat_bare)]
    for x in (-0.75, -0.25, 0.25, 0.75):
        for y_sign in (-1, 1):
            parts += _build_wheel(x, y_sign * 0.75, mat_rubber, mat_bare)
    parts += [_build_track(-1, mat_bare), _build_track(1, mat_bare)]

    bpy.ops.object.select_all(action="DESELECT")
    for p in parts:
        p.select_set(True)
    bpy.context.view_layer.objects.active = parts[0]
    bpy.ops.object.join()
    vehicle = bpy.context.active_object
    vehicle.name = "vehicle"

    out_path = _OUT_PATH
    bpy.ops.export_scene.gltf(
        filepath=out_path,
        export_format="GLB",
        use_selection=False,
        export_apply=True,
        export_yup=True,
    )
    result = {"status": "ok", "exported": out_path, "objects": len(parts)}
    return result


if __name__ == "__main__":
    build()

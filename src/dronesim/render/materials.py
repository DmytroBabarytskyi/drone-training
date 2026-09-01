"""PBR-матеріали (доп. фаза "графіка 2.0"): реальні ``panda3d.core.Material``
(roughness/metallic) замість голого кольору.

ПЕРЕВІРЕНО ЕМПІРИЧНО: без явного ``NodePath.setMaterial()`` Panda3D підставляє
ГЛОБАЛЬНИЙ дефолт (``Material().getRoughness() == 1.0``, повністю матовий,
``metallic == 0.0``) — саме тому ВСІ об'єкти сцени досі виглядали однаково
"пластиково-матовими" незалежно від того, що вони представляють (метал
техніки, гума коліс, земля, листя). ``simplepbr.frag`` рахує колір як
``p3d_Material.baseColor * v_color * texture`` — щоб НЕ зламати наявне
тонування через ``NodePath.setColor()``/вершинний колір, тут ``baseColor``
ЗАВЖДИ лишається нейтральним (1,1,1,1); Material відповідає ЛИШЕ за
roughness/metallic/emission.

Значення roughness/metallic — ОРІЄНТОВНІ (типові для класу матеріалу зі
спільнот PBR-текстурування), не виміряні на фізичному зразку — той самий
рівень точності, що й інші орієнтовні фізичні константи цього проєкту
(docs/DECISIONS.md, "реалізм фізики").
"""

from __future__ import annotations

from panda3d.core import Material, NodePath, Vec4

_NEUTRAL_BASE_COLOR = Vec4(1.0, 1.0, 1.0, 1.0)


def make_material(roughness: float, metallic: float, emission: tuple[float, float, float, float] = (0.0, 0.0, 0.0, 0.0)) -> Material:
    """PBR-матеріал із НЕЙТРАЛЬНИМ baseColor (1,1,1,1) — фактичний колір і
    далі йде через ``NodePath.setColor()``/текстуру (``simplepbr.frag``
    множить ``Material.baseColor * v_color * texture``, тож зміна тут НІЯК
    не впливає на вже підібрані кольори моделей)."""
    mat = Material()
    mat.setBaseColor(_NEUTRAL_BASE_COLOR)
    mat.setRoughness(roughness)
    mat.setMetallic(metallic)
    if any(emission):
        mat.setEmission(Vec4(*emission))
    return mat


# Пресети за класом поверхні (roughness 0=дзеркало..1=повністю матовий).
METAL_PAINTED = make_material(roughness=0.45, metallic=0.25)  # пофарбований метал (корпус техніки/будівлі)
METAL_BARE = make_material(roughness=0.30, metallic=0.85)  # голий метал (стволи, обода коліс)
RUBBER = make_material(roughness=0.95, metallic=0.0)  # шини/гумові деталі
WOOD = make_material(roughness=0.80, metallic=0.0)  # паркан/стовбур дерева
DIRT = make_material(roughness=0.97, metallic=0.0)  # земля/окоп
CONCRETE = make_material(roughness=0.75, metallic=0.0)  # стіни будівлі
FOLIAGE = make_material(roughness=0.85, metallic=0.0)  # крони/трава/кущі
ASPHALT = make_material(roughness=0.80, metallic=0.05)  # дорога
GENERIC = make_material(roughness=0.6, metallic=0.05)  # дефолт для не категоризованих коробок


def apply_material(node: NodePath, material: Material) -> None:
    """Застосувати PBR-пресет до вже пофарбованого вузла (не змінює колір)."""
    node.setMaterial(material)

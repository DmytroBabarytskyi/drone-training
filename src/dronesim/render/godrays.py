"""Справжні god rays (доп. фаза "графіка 2.0", пост-процесинг) — заміна
гало+туману на реальний радіальний розмив яскравої маски сонця, адитивно
накладений на основне зображення (стандартна техніка "volumetric light
scattering" / crepuscular rays, той самий підхід, що й приклад Panda3D
"Volumetric Lighting").

ІЗОЛЯЦІЯ ВІД simplepbr: НЕ використовує ``simplepbr``'s власний
``FilterManager`` (той уже зайнятий tonemap-постобробкою) — натомість
ОКРЕМИЙ, повністю ІЗОЛЬОВАНИЙ корінь сцени (``NodePath``, НЕ дочірній
``base.render``!) з ЄДИНИМ об'єктом — копією диска сонця. Камера маски
(``base.makeCamera(..., scene=цей_корінь)``) рендерить ЛИШЕ цей корінь —
``Camera.setScene()`` (перевірено в джерелі ``ShowBase.makeCamera``) робить
непотрібним будь-яке маскування бітами видимості: маска-камера просто НІКОЛИ
не бачить основну сцену, бо в неї INШИЙ корінь traversal. Той самий принцип
ізоляції, що й ``FPVCamera``/chase-cam (render/camera.py) — окремий буфер,
власна камера, нуль втручання в уже готовий рендер-пайплайн.
"""

from __future__ import annotations

from panda3d.core import (
    CardMaker,
    ColorBlendAttrib,
    NodePath,
    Point2,
    Point3,
    Shader,
    TransparencyAttrib,
)

from dronesim.render.sky import _make_sun_texture, sun_direction

_VERTEX_SHADER = """
#version 120
uniform mat4 p3d_ModelViewProjectionMatrix;
attribute vec4 p3d_Vertex;
attribute vec2 p3d_MultiTexCoord0;
varying vec2 v_texcoord;
void main() {
    v_texcoord = p3d_MultiTexCoord0;
    gl_Position = p3d_ModelViewProjectionMatrix * p3d_Vertex;
}
"""

# Радіальний розмив маски до джерела (sun_screen_pos, 0..1 текстурний
# простір) — стандартна формула GPU Gems "Volumetric Light Scattering",
# NUM_SAMPLES=24 — досить для м'яких променів без помітної продуктивної
# ціни (маска рендериться в МАЛЕНЬКИЙ буфер, не в повну роздільність вікна).
_FRAGMENT_SHADER = """
#version 120
uniform sampler2D mask_tex;
uniform vec2 sun_screen_pos;
uniform float exposure;
uniform float decay;
uniform float density;
uniform float ray_weight;
varying vec2 v_texcoord;

const int NUM_SAMPLES = 24;

void main() {
    vec2 tex_coord = v_texcoord;
    vec2 delta = (tex_coord - sun_screen_pos) * (density / float(NUM_SAMPLES));
    vec3 color = texture2D(mask_tex, tex_coord).rgb;
    float illum_decay = 1.0;
    for (int i = 0; i < NUM_SAMPLES; i++) {
        tex_coord -= delta;
        vec3 sample_color = texture2D(mask_tex, clamp(tex_coord, 0.0, 1.0)).rgb;
        sample_color *= illum_decay * ray_weight;
        color += sample_color;
        illum_decay *= decay;
    }
    gl_FragColor = vec4(color * exposure, 1.0);
}
"""


class GodRays:
    """Пост-процес god rays. ``update(dt)`` викликати щокадру — перераховує
    екранну позицію сонця (проєкція через об'єктив ГОЛОВНОЇ камери) для
    шейдера радіального розмиву; сама маска рендериться автоматично щокадру
    Panda3D (окрема камера/буфер, як і FPVCamera)."""

    def __init__(
        self,
        base,
        mask_size: int = 256,
        sun_azimuth_deg: float = 45.0,
        sun_elevation_deg: float = 48.0,
        sun_distance: float = 1000.0,
        exposure: float = 0.12,
        decay: float = 0.93,
        density: float = 1.4,
        ray_weight: float = 0.4,
    ):
        self._base = base
        self._sun_world_pos = Point3(*(sun_direction(sun_azimuth_deg, sun_elevation_deg) * sun_distance))

        # ІЗОЛЬОВАНИЙ корінь сцени (НЕ base.render!) — єдиний вміст: диск
        # сонця. Маска-камера рендерить ТІЛЬКИ це дерево (Camera.setScene),
        # тож фон буфера природно чорний, без потреби в occlusion-масках.
        self._mask_root = NodePath("godrays_mask_scene")
        tex = _make_sun_texture()
        cm = CardMaker("godrays_sun")
        size = sun_distance * 0.14
        cm.setFrame(-size, size, -size, size)
        sun_card = self._mask_root.attachNewNode(cm.generate())
        sun_card.setTexture(tex)
        # MAlpha (не MNone): текстура сонця має радіальний альфа-спад
        # (render/sky.py::_make_sun_texture) — без альфи диск рендерився б
        # ЯК ХАРАКТЕРНИЙ ГОСТРОКУТНИЙ КВАДРАТ картки, а не м'яке сяйво
        # (виявлено емпірично: перший варіант з MNone дав квадратну маску).
        sun_card.setTransparency(TransparencyAttrib.MAlpha)
        sun_card.setBillboardPointEye()
        sun_card.setPos(self._sun_world_pos)
        sun_card.setLightOff(1)
        sun_card.setShaderOff(1)
        sun_card.setColor(1.0, 0.95, 0.8, 1.0)

        self._buffer = base.win.makeTextureBuffer("godrays_mask", mask_size, mask_size, None, True)
        self._buffer.setClearColor((0.0, 0.0, 0.0, 1.0))
        self._mask_cam = base.makeCamera(self._buffer, scene=self._mask_root, lens=base.camLens)
        self._mask_tex = self._buffer.getTexture()

        # Повноекранна картка-компоновка: адитивне накладання на ГОЛОВНЕ вікно.
        shader = Shader.make(Shader.SL_GLSL, vertex=_VERTEX_SHADER, fragment=_FRAGMENT_SHADER)
        cm2 = CardMaker("godrays_composite")
        cm2.setFrameFullscreenQuad()
        self._composite = base.render2d.attachNewNode(cm2.generate())
        self._composite.setShader(shader)
        self._composite.setShaderInput("mask_tex", self._mask_tex)
        self._composite.setShaderInput("exposure", float(exposure))
        self._composite.setShaderInput("decay", float(decay))
        self._composite.setShaderInput("density", float(density))
        self._composite.setShaderInput("ray_weight", float(ray_weight))
        self._composite.setShaderInput("sun_screen_pos", (0.5, 0.5))
        self._composite.setTransparency(TransparencyAttrib.MNone)
        self._composite.setAttrib(
            ColorBlendAttrib.make(ColorBlendAttrib.MAdd, ColorBlendAttrib.OOne, ColorBlendAttrib.OOne)
        )
        self._composite.setBin("fixed", 1000)  # малювати ПІСЛЯ основної сцени (поверх)
        self._composite.setDepthTest(False)
        self._composite.setDepthWrite(False)

    def update(self, dt: float) -> None:
        lens = self._base.camLens
        rel_pos = self._base.camera.getRelativePoint(self._base.render, self._sun_world_pos)
        screen = Point2()
        if lens.project(rel_pos, screen):
            u = (screen.x + 1.0) * 0.5
            v = (screen.y + 1.0) * 0.5
            self._composite.setShaderInput("sun_screen_pos", (float(u), float(v)))

    def remove(self) -> None:
        self._composite.removeNode()
        self._mask_root.removeNode()
        self._base.graphicsEngine.removeWindow(self._buffer)

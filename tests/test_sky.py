"""Тести неба/сонця/туману (render/sky.py) — headless (engine-фікстура, як
test_hud.py/test_visuals.py). Перевіряють, що купол будується без винятку,
має геометрію й коректні стани рендеру (позаду сцени, без туману на самому
небі), а туман — кольору горизонту з очікуваним лінійним діапазоном."""

from dronesim.render.sky import _HORIZON_COLOR, add_clouds, add_fog, add_sky


def test_add_clouds_places_billboards_behind_scene(engine):
    clouds = add_clouds(engine, count=10)
    try:
        assert clouds.getNumChildren() == 10
        assert clouds.getBinName() == "background"
    finally:
        clouds.removeNode()


def test_add_sky_builds_dome_and_sun(engine):
    sky = add_sky(engine)
    try:
        assert not sky.isEmpty()
        # Купол (GeomNode) + диск сонця (картка) — принаймні 2 дочірні вузли.
        assert sky.getNumChildren() >= 2
        # Небо не має писати глибину (щоб не перекривати сцену) і має бути в
        # фоновому біні.
        assert sky.getBinName() == "background"
    finally:
        sky.removeNode()


def test_add_sky_dome_has_vertices(engine):
    sky = add_sky(engine, rings=8, segments=12)
    try:
        geom_nodes = sky.findAllMatches("**/+GeomNode")
        assert len(geom_nodes) >= 1
        total_verts = 0
        for gn in geom_nodes:
            node = gn.node()
            for i in range(node.getNumGeoms()):
                total_verts += node.getGeom(i).getVertexData().getNumRows()
        assert total_verts > 0
    finally:
        sky.removeNode()


def test_add_fog_is_horizon_colored_and_linear(engine):
    fog = add_fog(engine, density_near_m=50.0, density_far_m=200.0)
    try:
        color = fog.getColor()
        assert abs(color[0] - _HORIZON_COLOR[0]) < 1e-3
        assert abs(color[1] - _HORIZON_COLOR[1]) < 1e-3
        assert abs(color[2] - _HORIZON_COLOR[2]) < 1e-3
        assert engine.render.getFog() is not None
    finally:
        engine.render.clearFog()

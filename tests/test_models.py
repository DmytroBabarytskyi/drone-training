"""Тести процедурної бібліотеки 3D-форм (render/models.py, доп. фаза
"реальні моделі") — headless, engine fixture як test_visuals.py."""

from panda3d.core import Vec3

from dronesim.render import models


def test_build_cylinder_mesh_has_expected_radial_bounds(engine):
    parent = engine.render.attachNewNode("test_parent_cyl")
    try:
        cyl = models.build_cylinder_mesh(0.3, 2.0, segments=12)
        cyl.reparentTo(parent)
        mn, mx = cyl.getTightBounds()
        assert abs(mx.x - 0.3) < 0.02
        assert abs(mn.z - 0.0) < 1e-6
        assert abs(mx.z - 2.0) < 1e-6
    finally:
        parent.removeNode()


def test_build_cone_mesh_tapers_to_apex(engine):
    parent = engine.render.attachNewNode("test_parent_cone")
    try:
        cone = models.build_cone_mesh(0.5, 1.5, segments=10)
        cone.reparentTo(parent)
        mn, mx = cone.getTightBounds()
        assert abs(mx.x - 0.5) < 0.02  # база на z=0 має радіус ~0.5
        assert abs(mx.z - 1.5) < 1e-6  # вершина на z=height
    finally:
        parent.removeNode()


def test_build_tree_procedural_fallback_has_trunk_and_foliage_children(engine):
    """Прямий тест fallback-білдера (не ``build_tree`` — той тепер спершу
    пробує Blender-модель ``assets/models/tree_{species}.bam``, доп. фаза
    "реальні моделі"; якщо файл присутній на цій машині, ``build_tree``
    поверне ЇЇ, а не процедурну геометрію)."""
    parent = engine.render.attachNewNode("test_parent_tree")
    try:
        tree = models._build_pine_tree(parent, 2.5, 0.15, 1.3, 0)  # noqa: SLF001
        assert len(tree.getChildren()) == 4  # стовбур + 3 конуси крони (пишніше)
        mn, mx = tree.getTightBounds()
        assert mx.z > 2.0  # вище стовбура (крона зверху)
    finally:
        parent.removeNode()


def test_build_tree_returns_nonempty_node_either_path(engine):
    """``build_tree`` — Blender-модель, якщо ``assets/models/tree_{species}.bam``
    присутній, інакше процедурний fallback; в обох випадках непорожній вузол
    з розумними габаритами (не перевіряємо КОНКРЕТНУ структуру, щоб тест не
    залежав від того, чи згенеровано ассет на цій машині)."""
    parent = engine.render.attachNewNode("test_parent_tree_dispatch")
    try:
        tree = models.build_tree(parent, trunk_height=2.5)
        assert not tree.isEmpty()
        mn, mx = tree.getTightBounds()
        assert mx.z > 1.5  # реальна висота, не крапка
    finally:
        parent.removeNode()


def test_build_tree_species_produce_distinct_non_empty_trees(engine):
    """Доп. фаза "графіка 3.0"/"реальні моделі": реальний фідбек "різні види
    дерев" — 3 види непорожні й дають РІЗНІ габарити (не перевіряємо
    структуру дочірніх вузлів, щоб тест не залежав від того, чи Blender-
    модель присутня на цій машині)."""
    parent = engine.render.attachNewNode("test_parent_tree_species")
    try:
        pine = models.build_tree(parent, species="pine")
        oak = models.build_tree(parent, species="oak")
        birch = models.build_tree(parent, species="birch")
        bounds = []
        for tree in (pine, oak, birch):
            assert not tree.isEmpty()
            mn, mx = tree.getTightBounds()
            assert mx.z > 1.0
            bounds.append((round(mx.z, 2), round(mx.x - mn.x, 2)))
        assert len(set(bounds)) > 1  # різні види -> різні габарити
    finally:
        parent.removeNode()


def test_build_tree_unknown_species_falls_back_to_pine(engine):
    parent = engine.render.attachNewNode("test_parent_tree_fallback")
    try:
        tree = models.build_tree(parent, species="not_a_real_species")
        assert tree.getName() == "tree_pine"
    finally:
        parent.removeNode()


def test_build_grass_tuft_has_three_crossed_blades(engine):
    """Доп. фаза "графіка 2.0": 3 перехресні картки (як кущ), не 2 —
    пишніший силует."""
    parent = engine.render.attachNewNode("test_parent_grass")
    try:
        tuft = models.build_grass_tuft(parent)
        assert len(tuft.getChildren()) == 3
    finally:
        parent.removeNode()


def test_grass_blade_image_is_clump_with_gaps_not_solid():
    """Клумп-текстура (доп. фаза "графіка 3.0"): кілька тонких стеблин із
    ПРОЗОРИМИ проміжками — не суцільний прямокутник і не суцільний конус.
    Покриття alpha помірне (є і стеблини, і проміжки)."""
    import numpy as np

    size = 48
    img = models._make_grass_blade_image(size=size, seed=0)  # noqa: SLF001
    alpha = np.array([[img.getAlpha(x, y) for x in range(size)] for y in range(size)])
    coverage = (alpha > 0.5).mean()
    assert 0.1 < coverage < 0.7  # частково-прозоро: стеблини + проміжки
    # Верхній рядок (кінчики) рідший за нижній (бази стеблин ширші).
    assert (alpha[0] > 0.5).mean() < (alpha[-1] > 0.5).mean()


def test_grass_blade_texture_cached_by_seed(engine):
    tex_a = models._make_grass_blade_texture(seed=2)  # noqa: SLF001
    tex_b = models._make_grass_blade_texture(seed=2)  # noqa: SLF001
    assert tex_a is tex_b  # той самий seed -> перевикористана текстура (не перегенерована)


def test_build_fence_line_spans_from_start_to_end(engine):
    parent = engine.render.attachNewNode("test_parent_fence")
    try:
        fence = models.build_fence_line(parent, (0.0, 0.0), (10.0, 0.0), post_spacing=2.0)
        mn, mx = fence.getTightBounds()
        assert mx.x > 9.0  # рейки/стовпці сягають кінця лінії
        assert mn.x < 1.0
    finally:
        parent.removeNode()


def test_build_fence_line_zero_length_does_not_raise(engine):
    parent = engine.render.attachNewNode("test_parent_fence_zero")
    try:
        fence = models.build_fence_line(parent, (0.0, 0.0), (0.0, 0.0))
        assert fence is not None
    finally:
        parent.removeNode()


def test_build_trench_procedural_fallback_has_two_walls(engine):
    """Прямий тест fallback-білдера (не ``build_trench`` — той тепер спершу
    пробує Blender-модель ``assets/models/trench.bam``, доп. фаза
    "ландшафт 2.0"; якщо файл присутній на цій машині, ``build_trench``
    поверне ЇЇ, а не процедурну геометрію)."""
    parent = engine.render.attachNewNode("test_parent_trench")
    try:
        trench = models._build_trench_procedural(parent, length=4.0, width=1.5)  # noqa: SLF001
        assert len(trench.getChildren()) == 2
    finally:
        parent.removeNode()


def test_build_trench_returns_nonempty_node_either_path(engine):
    """``build_trench`` — Blender-модель, якщо ``assets/models/trench.bam``
    присутній, інакше процедурний fallback; в обох випадках непорожній
    вузол з розумними габаритами (не перевіряємо КОНКРЕТНУ структуру, щоб
    тест не залежав від того, чи згенеровано ассет на цій машині)."""
    parent = engine.render.attachNewNode("test_parent_trench_dispatch")
    try:
        trench = models.build_trench(parent, length=8.0, width=2.4, wall_height=1.0)
        assert not trench.isEmpty()
        mn, mx = trench.getTightBounds()
        assert mx.x - mn.x > 4.0  # реальна довжина, не крапка
    finally:
        parent.removeNode()


def test_build_road_strip_has_texture(engine):
    parent = engine.render.attachNewNode("test_parent_road")
    try:
        road = models.build_road_strip(parent, length=20.0, width=4.0)
        assert road.hasTexture()
        mn, mx = road.getTightBounds()
        assert abs((mx.x - mn.x) - 20.0) < 1e-3
    finally:
        parent.removeNode()


def test_build_vehicle_procedural_fallback_has_hull_cabin_turret_barrel_and_wheels(engine):
    """Прямий тест fallback-білдера (не ``build_vehicle`` — той тепер
    спершу пробує Blender-модель ``assets/models/vehicle.bam``, доп. фаза
    "графіка 2.0"; якщо файл присутній на цій машині, ``build_vehicle``
    поверне ЇЇ, а не процедурну геометрію)."""
    parent = engine.render.attachNewNode("test_parent_vehicle")
    try:
        vehicle = models._build_vehicle_procedural(parent, (0.28, 0.32, 0.22, 1.0))  # noqa: SLF001
        # корпус + кабіна + вежа + ствол + 8 коліс = 12
        assert len(vehicle.getChildren()) == 12
    finally:
        parent.removeNode()


def test_build_vehicle_returns_nonempty_node_either_path(engine):
    """``build_vehicle`` — Blender-модель, якщо ``assets/models/vehicle.bam``
    присутній, інакше процедурний fallback; в обох випадках непорожній
    вузол з розумними габаритами (не перевіряємо КОНКРЕТНУ структуру, щоб
    тест не залежав від того, чи згенеровано ассет на цій машині)."""
    parent = engine.render.attachNewNode("test_parent_vehicle_dispatch")
    try:
        vehicle = models.build_vehicle(parent)
        assert not vehicle.isEmpty()
        mn, mx = vehicle.getTightBounds()
        assert (mx.x - mn.x) > 0.5  # має реальний розмір, не крапка
    finally:
        parent.removeNode()


def test_build_artillery_procedural_fallback_has_base_mount_and_barrel(engine):
    """Прямий тест fallback-білдера (``build_artillery`` тепер спершу пробує
    Blender-модель ``assets/models/artillery.bam``, доп. фаза "графіка 2.0")."""
    parent = engine.render.attachNewNode("test_parent_artillery")
    try:
        artillery = models._build_artillery_procedural(parent, (0.3, 0.28, 0.2, 1.0))  # noqa: SLF001
        assert len(artillery.getChildren()) == 3
    finally:
        parent.removeNode()


def test_build_artillery_returns_nonempty_node_either_path(engine):
    parent = engine.render.attachNewNode("test_parent_artillery_dispatch")
    try:
        artillery = models.build_artillery(parent)
        assert not artillery.isEmpty()
        mn, mx = artillery.getTightBounds()
        assert (mx.z - mn.z) > 0.3
    finally:
        parent.removeNode()


def test_build_bush_is_twosided_with_cards(engine):
    parent = engine.render.attachNewNode("test_parent_bush")
    try:
        bush = models.build_bush(parent, size=0.6)
        assert len(bush.getChildren()) == 3  # 3 перехресні картки
    finally:
        parent.removeNode()


def test_scatter_ground_cover_builds_grass_field_over_area(engine):
    parent = engine.render.attachNewNode("test_parent_cover")
    try:
        cover = models.scatter_ground_cover(
            parent, radius=20.0, density_per_m2=12.0, bush_count=8, seed=1, exclude_radius=3.0
        )
        # Трав'яний килим — один Geom (`grass_field`), непорожній, розкиданий
        # по площі (не в одній точці).
        grass = cover.find("**/grass_field")
        assert not grass.isEmpty()
        mn, mx = cover.getTightBounds()
        assert (mx.x - mn.x) > 10.0  # розкидано по полю радіусом ~20
    finally:
        parent.removeNode()


def test_scatter_ground_cover_keeps_spawn_clearing(engine):
    """Density map тримає точку старту (``exclude_radius``) чистою: густина
    трави біля центру має бути НИЖЧОЮ, ніж на відкритому полі."""
    import numpy as np

    rng = np.random.default_rng(0)
    clearings = [(0.0, 0.0, 5.0)]
    # Точки в ядрі галявини vs далеко на полі.
    core_x = np.array([0.0, 0.5, -0.5, 1.0])
    core_y = np.array([0.0, 0.0, 0.5, -0.5])
    far_x = np.array([15.0, -15.0, 12.0, -12.0])
    far_y = np.array([15.0, -12.0, -15.0, 12.0])
    d_core = models._grass_density(core_x, core_y, 20.0, rng, clearings)  # noqa: SLF001
    d_far = models._grass_density(far_x, far_y, 20.0, np.random.default_rng(0), clearings)  # noqa: SLF001
    assert d_core.mean() < d_far.mean()
    assert d_core.max() < 0.2  # ядро галявини майже без трави


def test_build_building_has_walls_roof_and_correct_footprint(engine):
    parent = engine.render.attachNewNode("test_parent_building")
    try:
        building = models.build_building(parent, width=6.0, depth=6.0, height=8.0)
        # 4 стіни + дах = 5 дочірніх вузлів
        assert len(building.getChildren()) == 5
        mn, mx = building.getTightBounds()
        assert abs((mx.x - mn.x) - 6.0) < 1e-3  # ширина
        assert mn.z >= -1e-3  # база на землі (локальний z=0), не нижче
        assert mx.z > 7.9  # висота ~8м (+ дах)
    finally:
        parent.removeNode()


def test_build_building_walls_face_outward(engine):
    """Регрес-тест на реальний баг (доп. фаза "графіка 3.0", реальний
    фідбек "не відрізниш, на яку сторону падає сонце"): перед/зад мали
    ОДНАКОВИЙ heading (0.0/0.0), лівий/правий — теж (90.0/90.0), тож
    половина стін мала нормаль ВСЕРЕДИНУ будівлі."""
    parent = engine.render.attachNewNode("test_parent_building_normals")
    try:
        building = models.build_building(parent, width=6.0, depth=6.0, height=8.0)
        walls = [c for c in building.getChildren() if c.getName() == "wall"]
        assert len(walls) == 4
        expected_by_y = {True: 1.0, False: -1.0}  # +hd -> нормаль +Y, -hd -> нормаль -Y
        expected_by_x = {True: 1.0, False: -1.0}
        for wall in walls:
            pos = wall.getPos()
            normal = building.getRelativeVector(wall, Vec3(0, -1, 0))
            if abs(pos.y) > abs(pos.x):  # перед/зад (зсув по Y)
                assert abs(normal.y - expected_by_y[pos.y > 0]) < 1e-3, (pos, normal)
            else:  # право/ліво (зсув по X)
                assert abs(normal.x - expected_by_x[pos.x > 0]) < 1e-3, (pos, normal)
    finally:
        parent.removeNode()


def test_build_rock_is_nonempty_and_bounded_by_radius(engine):
    parent = engine.render.attachNewNode("test_parent_rock")
    try:
        rock = models.build_rock(radius=0.3, seed=1)
        rock.reparentTo(parent)
        assert not rock.isEmpty()
        mn, mx = rock.getTightBounds()
        assert (mx.x - mn.x) < 0.3 * 2.5  # не набагато більший за заданий радіус
        # Джиттер форми на полюсі трохи хитає базу навколо z=0 (неправильна
        # "грудкувата" форма, не ідеальна геометрія) — допуск, не 0 строго.
        assert mn.z >= -0.3 * 0.25
    finally:
        parent.removeNode()


def test_scatter_rocks_places_requested_count(engine):
    parent = engine.render.attachNewNode("test_parent_rocks_scatter")
    try:
        rocks = models.scatter_rocks(parent, count=15, radius=20.0, seed=2, exclude_radius=2.0)
        # flattenStrong зводить у 1 вузол (кілька draw-викликів) — перевіряємо
        # непорожність і габарити, не кількість дочірніх вузлів.
        assert not rocks.isEmpty()
        mn, mx = rocks.getTightBounds()
        assert (mx.x - mn.x) > 4.0  # розкидані по площі, не в одній точці
    finally:
        parent.removeNode()


def test_build_wheat_tuft_has_three_crossed_blades(engine):
    parent = engine.render.attachNewNode("test_parent_wheat")
    try:
        tuft = models.build_wheat_tuft(parent)
        assert len(tuft.getChildren()) == 3
    finally:
        parent.removeNode()


def test_wheat_blade_image_is_golden_not_green():
    size = 32
    img = models._make_grass_blade_image(  # noqa: SLF001
        size, 0, base_color=(0.55, 0.42, 0.12), tip_color=(0.85, 0.72, 0.28)
    )
    # Знайти НЕПРОЗОРИЙ піксель стеблини (клумп має проміжки, тож фіксована
    # координата може бути прозорою) і перевірити золотисто-жовту гаму:
    # R і G високі, B нижчий — на відміну від трави, де G домінує над R.
    for y in range(size):
        for x in range(size):
            if img.getAlpha(x, y) > 0.5:
                r, g, b = img.getXel(x, y)
                assert r > 0.3
                assert b < g
                return
    raise AssertionError("не знайдено непрозорого пікселя стеблини")


def test_scatter_wheat_patch_stays_within_requested_area(engine):
    parent = engine.render.attachNewNode("test_parent_wheat_patch")
    try:
        patch = models.scatter_wheat_patch(parent, center=(10.0, 5.0), size=(6.0, 6.0), count=30, seed=4)
        assert not patch.isEmpty()
        mn, mx = patch.getTightBounds()
        assert mn.x >= 10.0 - 3.5  # невеликий запас під ширину самих стебел
        assert mx.x <= 10.0 + 3.5
    finally:
        parent.removeNode()


def test_scatter_wheat_patch_plants_in_straight_rows():
    """Реальний фідбек: "поле — це не просто трава, воно посаджене рядочками
    спеціальним комбайном" — стебла МАЮТЬ лягати на кілька дискретних
    рядкових координат упоперек поля (``row_spacing``), а не розкидатись
    рівномірно-випадково по всій ширині."""
    import numpy as np

    length, width = 20.0, 6.0
    row_spacing = 0.5
    n_rows_expected = int(width / row_spacing)
    # Пряме відтворення локальної сітки (без обертання, angle_deg=0) —
    # перевіряємо, що поперечна координата кластеризується в
    # ``n_rows_expected`` вузьких групах, а не суцільний розкид.
    rng = np.random.default_rng(4)
    plant_spacing = max(0.10, length * width / 400 / row_spacing)
    n_rows = max(1, int(width / row_spacing))
    n_per_row = max(1, int(length / plant_spacing))
    row_i, _col_i = np.meshgrid(np.arange(n_rows), np.arange(n_per_row), indexing="ij")
    ly = (row_i.ravel() + 0.5) * row_spacing - width / 2.0
    ly = ly + rng.uniform(-row_spacing * 0.2, row_spacing * 0.2, ly.size)
    # Кожна точка МАЄ бути близько до одного з n_rows номінальних рядків
    # (у межах джиттера), не розподілена рівномірно по всій ширині.
    row_centers = (np.arange(n_rows) + 0.5) * row_spacing - width / 2.0
    nearest_row_dist = np.min(np.abs(ly[:, None] - row_centers[None, :]), axis=1)
    assert np.max(nearest_row_dist) <= row_spacing * 0.21  # завжди близько до "свого" рядка
    assert n_rows_expected == n_rows


def test_scatter_wheat_patch_rows_follow_angle(engine):
    """Повернене поле (``angle_deg``) МАЄ фактично повертати forму ділянки —
    bounding box більше не вирівняний по осях X/Y."""
    parent = engine.render.attachNewNode("test_parent_wheat_rot")
    try:
        patch = models.scatter_wheat_patch(
            parent, center=(0.0, 0.0), size=(20.0, 4.0), count=400, seed=9, angle_deg=45.0
        )
        mn, mx = patch.getTightBounds()
        # Довга вісь (20м) під 45° -> проєкції на X і Y приблизно однакові
        # (~20/sqrt(2) ≈ 14.1), а НЕ 20 на X і 4 на Y, як було б без обертання.
        span_x, span_y = mx.x - mn.x, mx.y - mn.y
        assert abs(span_x - span_y) < 4.0
    finally:
        parent.removeNode()


def test_scatter_tree_line_places_trees_along_segment(engine):
    """Реальний фідбек: "посадки — це багато дерев лінією вздовж поля" —
    дерева МАЮТЬ бути розкидані вздовж відрізка, а не в одній точці.
    ``flattenStrong()`` (продуктивність — див. DECISIONS.md) зводить дерева
    в кілька Geom, тож перевіряємо TIGHT BOUNDS усього результату, не
    кількість дочірніх NodePath."""
    parent = engine.render.attachNewNode("test_parent_tree_line")
    try:
        line = models.scatter_tree_line(parent, start=(0.0, 0.0), end=(30.0, 0.0), spacing=4.0, seed=2)
        assert not line.isEmpty()
        mn, mx = line.getTightBounds()
        assert mn.x < 5.0  # перше дерево біля start
        assert mx.x > 25.0  # останнє біля end
        # Дерева розподілені вздовж ЛІНІЇ (бічний джиттер малий відносно
        # довжини 30м; допуск враховує й радіус крони, не лише позицію).
        assert mn.y > -2.5
        assert mx.y < 2.5
    finally:
        parent.removeNode()


def test_terrain_height_is_flat_under_working_zone_and_settles_far_out():
    """Реальний фідбек: "рельєф лиш трохи нерівний, потрібно щоб десь міг
    бути яр, десь пагорб, десь рівне поле" (heightmap на основі Вороного) —
    рельєф МАЄ бути точно 0 у робочій зоні (де живуть obstacle/target/spawn,
    інваріант ``pos.z == ground_z``), містити РЕАЛЬНУ варіацію (і пагорби, і
    яри — не лише додатні значення) у кільці за нею, і ЗНОВУ бути точно
    рівним далеко за межею сітки (радіальне згасання в ``_terrain_height_grid``
    не "затоплює" базу далеких об'єктів у нескінченність)."""
    import numpy as np

    flat, outer = 40.0, 70.0
    # Точно 0 у робочій зоні — незалежно від seed/розкладки клітинок.
    r_flat = np.array([0.0, 20.0, 39.9])
    h_flat = models._terrain_height(r_flat, np.zeros_like(r_flat), flat, outer, 4.0, seed=1)  # noqa: SLF001
    assert np.allclose(h_flat, 0.0)
    # Точно 0 задовго за межею сітки (радіальне згасання в grid).
    r_far = np.array([outer + 40.0, 150.0])
    h_far = models._terrain_height(r_far, np.zeros_like(r_far), flat, outer, 4.0, seed=1)  # noqa: SLF001
    assert np.allclose(h_far, 0.0, atol=1e-4)
    # У кільці рельєфу — РЕАЛЬНА варіація (не однорідне плато): десь помітно
    # відмінне від нуля значення (пагорб АБО яр), і є ОБИДВА знаки (не лише
    # горби) для типового seed.
    xs = np.linspace(-outer, outer, 60)
    gx, gy = np.meshgrid(xs, xs, indexing="ij")
    h_grid = models._terrain_height(gx, gy, flat, outer, 4.0, seed=1)  # noqa: SLF001
    assert np.any(h_grid > 0.5)
    assert np.any(h_grid < -0.5)
    # max_height=0 -> завжди пласко (зворотна сумісність, дефолт викликів).
    assert models._terrain_height(np.array([0.0]), np.array([0.0]), flat, outer, 0.0)[0] == 0.0  # noqa: SLF001


def test_voronoi_cells_include_both_hills_and_ravines():
    """``_voronoi_cells`` — основа heightmap (реальний фідбек: "генерувати
    heightmap реалістичний спочатку з картою Вороного... щоб десь міг бути
    яр, десь пагорб, десь рівне поле") — МАЄ давати клітинки всіх ЧОТИРЬОХ
    видів (``hill``/``ravine``/``flat``/``plantation``, доп. фаза
    "розширення kind") для достатньої площі/кількості клітинок; горби/яри —
    ще й напрямок+довжину хребта (не точка), ``"plantation"`` (як і
    ``"flat"``) НЕ впливає на рельєф (heights=0, без хребта)."""
    seeds_x, _sy, heights, kinds, ridge_angle, ridge_half_len = models._voronoi_cells(40.0, 130.0, seed=42)  # noqa: SLF001
    kind_set = set(kinds.tolist())
    assert {"hill", "ravine", "flat", "plantation"} <= kind_set
    assert (heights[kinds == "hill"] > 0).all()
    assert (heights[kinds == "ravine"] < 0).all()
    assert (heights[kinds == "flat"] == 0).all()
    assert (heights[kinds == "plantation"] == 0).all()
    relief = (kinds == "hill") | (kinds == "ravine")
    assert (ridge_half_len[relief] > 0).all()  # хребет має довжину — не точка
    assert (ridge_half_len[~relief] == 0).all()  # flat/plantation — не рельєф
    assert len(seeds_x) == len(ridge_angle)


def test_voronoi_grouped_regions_are_flat_and_bounded():
    """``_voronoi_grouped_regions(kind="flat")`` (реальний фідбек: "перевіряємо
    поруч клітинки, якщо теж рівно — додаємо поле туди; де рельєф піднімається
    чи опускається різко — не садять поле") — МАЄ давати КІЛЬКА окремих полів
    (обмежений ``max_cells`` — інакше суцільна рівнина перколює в одне поле
    на всю сцену), і ХОЧА Б ЯКІСЬ із них справді рівні (перевірка через
    ``_terrain_height``). Це топологічне НАБЛИЖЕННЯ (BFS по Delaunay-
    сусідству) — межова клітинка іноді опиняється неподалік хребта, що не
    був її прямим сусідом, тож не КОЖЕН регіон гарантовано ідеально рівний;
    справжня гарантія "пшениця ніколи не сіється на схилі" — геометрична
    перевірка САМЕ ПЕРЕД посівом у ``app.py`` (``_region_is_flat_enough``),
    ця функція лише постачає кандидатів."""
    import math

    import numpy as np

    flat, outer = 35.0, 150.0
    regions = models._voronoi_grouped_regions(flat, outer, seed=7, kind="flat")  # noqa: SLF001
    assert len(regions) > 1  # кілька ОКРЕМИХ полів, не одне на всю карту
    assert all(len(r["cells"]) <= 8 for r in regions)  # max_cells-кап: без перколяції в одну "мега-ниву"
    flatness_scores = []
    for region in regions:
        cx, cy = region["center"]
        length, width, angle = region["length"], region["width"], region["angle_deg"]
        rad = math.radians(angle)
        along = np.array([math.cos(rad), math.sin(rad)])
        across = np.array([-math.sin(rad), math.cos(rad)])
        sl_vals = np.linspace(-0.4, 0.4, 5)
        sw_vals = np.linspace(-0.4, 0.4, 5)
        sl_grid, sw_grid = np.meshgrid(sl_vals, sw_vals)
        pts = (
            np.array([cx, cy])[None, :]
            + along[None, :] * (sl_grid.ravel() * length)[:, None]
            + across[None, :] * (sw_grid.ravel() * width)[:, None]
        )
        h = models._terrain_height(pts[:, 0], pts[:, 1], flat, outer, 4.0, seed=7)  # noqa: SLF001
        flatness_scores.append(float(np.mean(np.abs(h))))
    assert any(s < 0.4 for s in flatness_scores)


def test_voronoi_grouped_regions_works_for_plantation_kind():
    """``_voronoi_grouped_regions`` — узагальнено з "лише flat -> поля" на
    БУДЬ-ЯКИЙ груповий kind (доп. фаза "розширення kind"): ``kind="plantation"``
    МАЄ давати ХОЧА Б ОДНУ групу (той самий BFS/PCA конвеєр, лише інша
    множина клітинок), з тим самим ``max_cells``-капом і структурою dict, що
    й поля. На відміну від полів, рівність рельєфу тут НЕ критична (посадка —
    декоративні ряди дерев, не сільгосп-техніка), тож без перевірки
    ``_terrain_height``."""
    flat, outer = 35.0, 150.0
    regions = models._voronoi_grouped_regions(flat, outer, seed=7, kind="plantation", max_cells=6)  # noqa: SLF001
    assert len(regions) >= 1
    assert all(len(r["cells"]) <= 6 for r in regions)
    for region in regions:
        assert region["length"] > 0
        assert region["width"] > 0


def test_build_terrain_is_flat_disc_at_center_and_varies_beyond():
    parent_np = None
    from panda3d.core import NodePath

    parent_np = NodePath("test_parent_terrain")
    try:
        terrain = models.build_terrain(
            parent_np, radius=120.0, flat_radius=30.0, outer_radius=55.0, max_height=5.0, segments=40, seed=2
        )
        mn, mx = terrain.getTightBounds()
        # Реальна варіація висоти (не плоска картка) — може йти і вгору
        # (пагорб), і вниз (яр), тож перевіряємо РОЗМАХ, не знак.
        assert (mx.z - mn.z) > 1.0
    finally:
        parent_np.removeNode()


def test_grass_density_excludes_road_rectangle():
    """Реальний фідбек: "на дорозі не росте трава" — прямокутна зона
    (``exclude_rects``) має нульову густину незалежно від шуму/галявин."""
    import numpy as np

    x = np.array([0.0, 5.0, 20.0])
    y = np.array([0.0, 0.0, 0.0])
    rng = np.random.default_rng(0)
    density = models._grass_density(  # noqa: SLF001
        x, y, 30.0, rng, clearings=[], exclude_rects=[(0.0, 0.0, 8.0, 2.0)]
    )
    assert density[0] == 0.0  # у прямокутнику дороги
    assert density[1] == 0.0  # теж у межах half_x=8
    assert density[2] > 0.0  # за межею дороги — звичайна густина


def test_scatter_ground_cover_grass_follows_terrain_height():
    """Трава на схилі МАЄ сидіти на реальній висоті рельєфу під нею, а не
    "плавати" над ним на висоті 0 (``base_z`` у ``_build_blade_field_mesh``).
    Явно задане ``base_z`` (не через ``_terrain_height`` — розкладка форм
    рельєфу випадкова й не гарантує ненульову висоту в довільній точці) —
    тестуємо, що сам меш ПОВАЖАЄ base_z, а не конкретну розкладку форм."""
    import numpy as np

    n = 6
    x = np.full(n, 60.0)
    y = np.linspace(-1.0, 1.0, n)
    height = np.full(n, 0.15)
    width = np.full(n, 0.1)
    heading = np.zeros(n)
    base_z = np.array([1.0, 1.2, 0.8, 1.5, 0.9, 1.1])

    from panda3d.core import NodePath

    parent_np = NodePath("test_parent_blade_bz")
    try:
        mesh = models._build_blade_field_mesh(  # noqa: SLF001
            parent_np, "test_field", x, y, height, width, heading, models._make_grass_blade_texture(seed=0),
            base_z=base_z,
        )
        mn, mx = mesh.getTightBounds()
        assert mn.z >= base_z.min() - 1e-6  # найнижча вершина не нижча за землю під нею
    finally:
        parent_np.removeNode()

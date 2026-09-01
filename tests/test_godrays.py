"""Тести god rays пост-процесу (render/godrays.py, доп. фаза "графіка 2.0").

Перевіряють, що ефект будується/оновлюється/прибирається без винятку і що
екранна позиція сонця РЕАЛЬНО перераховується (не залишається статичним
дефолтом) — глибша перевірка коректності радіального розмиву вимагала б
візуального порівняння скріншотів (перевірено вручну під час розробки, див.
docs/DECISIONS.md), тут — контракт "не падає, реагує на положення камери"."""

from dronesim.render.godrays import GodRays


def test_godrays_builds_without_raising(engine):
    godrays = GodRays(engine)
    try:
        assert not godrays._mask_root.isEmpty()  # noqa: SLF001
        assert not godrays._composite.isEmpty()  # noqa: SLF001
    finally:
        godrays.remove()


def test_godrays_update_does_not_raise(engine):
    godrays = GodRays(engine)
    try:
        for _ in range(5):
            engine.taskMgr.step()
            godrays.update(1 / 60)
    finally:
        godrays.remove()


def test_godrays_sun_screen_pos_centers_when_camera_looks_directly_at_sun(engine):
    """Дивлячись ТОЧНО на сонце, його екранна позиція має проєктуватись
    близько до центру (0.5, 0.5) — перевірка, що ``lens.project`` реально
    викликається й оновлює shader input (не лишається дефолтним значенням
    з __init__, встановленим ДО першого ``update()``)."""
    engine.camLens.setFov(100.0)  # широкий FOV, як у реальній грі (не вузький дефолт)
    godrays = GodRays(engine)
    try:
        engine.camera.reparentTo(engine.render)
        engine.camera.setPos(0, 0, 0)
        engine.camera.lookAt(godrays._sun_world_pos)  # noqa: SLF001
        godrays.update(1 / 60)
        pos = godrays._composite.getShaderInput("sun_screen_pos").getVector()  # noqa: SLF001
        assert abs(pos[0] - 0.5) < 0.05
        assert abs(pos[1] - 0.5) < 0.05

        # Розвернувшись на 180° (сонце тепер ПОЗАДУ), ``lens.project``
        # має провалитись (повернути False), тож shader input лишається
        # ОСТАННІМ УСПІШНИМ значенням (не скидається на щось довільне) —
        # той самий контракт, що й ``update()``'s докстрінг обіцяє.
        engine.camera.lookAt(-godrays._sun_world_pos)  # noqa: SLF001
        godrays.update(1 / 60)
        pos_behind = godrays._composite.getShaderInput("sun_screen_pos").getVector()  # noqa: SLF001
        assert pos_behind == pos  # незмінено, бо project() провалився
    finally:
        godrays.remove()


def test_godrays_remove_cleans_up_nodes(engine):
    godrays = GodRays(engine)
    mask_root = godrays._mask_root  # noqa: SLF001
    composite = godrays._composite  # noqa: SLF001
    godrays.remove()
    assert mask_root.isEmpty()
    assert composite.isEmpty()

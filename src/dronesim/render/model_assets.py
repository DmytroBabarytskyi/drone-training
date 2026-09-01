"""Завантаження попередньо запечених 3D-моделей (``.bam``), АВТОРОВАНИХ у
Blender (доп. фаза "графіка 2.0") — на відміну від ``render/models.py``
(повністю процедурні Panda3D-меші, побудовані НА ЛЬОТУ через Geom-API), тут
модель побудована ОДИН РАЗ у Blender через ЧЕКНУТИЙ У РЕПО bpy-скрипт
(``assets/blender/build_*.py`` — відтворюваний код, не одноразова ручна
робота в GUI), експортована в ``.glb``, сконвертована в ``.bam``
(``gltf2bam``, пакет ``panda3d-gltf``) і лежить у ``assets/models/*.bam``.

Це НЕ порушує принцип "усе процедурне, без готових ассетів ззовні"
(``render/models.py``, docs/DECISIONS.md) — ассет не ЗАВАНТАЖЕНО ЗЗОВНІ
(ліцензійні асset-стори тощо), а АВТОРОВАНО НАМИ САМИМИ, відтворювано, з
відкритим вихідним скриптом у репозиторії.

FALLBACK: якщо ``.bam``-файл відсутній (свіжий клон репо без згенерованих
ассетів, CI, чи Blender-пайплайн ще не запускали) — повертає ``None``, і
виклик має впасти на процедурний білдер (``render/models.py``) — гра
НІКОЛИ не падає через відсутній файл моделі.
"""

from __future__ import annotations

import builtins

from panda3d.core import Filename, NodePath

from dronesim.core.config import REPO_ROOT

_MODELS_DIR = REPO_ROOT / "assets" / "models"


def load_model_asset(parent: NodePath, name: str) -> NodePath | None:
    """Завантажити ``assets/models/{name}.bam`` як дочірній вузол ``parent``.

    ``builtins.loader`` — стандартний Panda3D-глобал (``ShowBase.__init__``
    реєструє його в ``builtins`` для КОЖНОГО коду, що будується поверх
    ShowBase, той самий патерн, що й у всіх туторіалах Panda3D) — тут це
    єдиний спосіб дістатись до ``Loader`` без зміни сигнатури КОЖНОГО
    виклику ``attach_scenery_visual``/``_sync_target_markers`` (app.py),
    жоден з яких зараз не передає ``engine``/``base`` явно.
    """
    path = _MODELS_DIR / f"{name}.bam"
    if not path.exists():
        return None
    model = builtins.loader.loadModel(Filename.fromOsSpecific(str(path)))
    if model is None or model.isEmpty():
        return None
    model.reparentTo(parent)
    return model

"""Спільні фікстури pytest.

``Engine`` (render/engine.py) обгортає ``ShowBase``, а Panda3D тримає багато
глобального стану (builtins на кшталт ``render``/``taskMgr``) — тож у тестовому
процесі створюємо ЛИШЕ ОДИН екземпляр на всю сесію (``scope="session"``), а не
по одному на тест, щоб уникнути конфліктів графічного контексту.

КРИТИЧНО (виявлено емпірично у фазі 5, docs/DECISIONS.md): на цій Windows-
машині ``torch`` МАЄ бути імпортований ДО Panda3D в межах одного процесу —
інакше зіштовхуються нативні DLL (torch падає з `OSError: DLL init routine
failed`, якщо Panda3D вже завантажений). Порядок УСВІДОМЛЕННЯ, не використання:
досить імпортувати `torch` тут першим рядком; використовувати можна пізніше.
Тому цей імпорт — перед ``dronesim.render.engine`` — навіть якщо жоден тест
у файлі напряму не використовує torch.
"""

from __future__ import annotations

import contextlib

with contextlib.suppress(ImportError):
    import torch  # noqa: F401  — див. докстрінг модуля: порядок імпорту важливий, не використання

import pytest

from dronesim.render.engine import Engine


@pytest.fixture(scope="session")
def engine() -> Engine:
    return Engine(offscreen=True, physics_hz=240.0)

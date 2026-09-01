"""Мінімальний Qt-смоук тест GUI-лаунчера (доп. фаза "GUI-лаунчер") —
``QT_QPA_PLATFORM=offscreen`` (без реального дисплея, headless CI/тести),
підтверджує лише що вікно БУДУЄТЬСЯ без винятку. Повне інтерактивне
тестування кліків — поза автоматичним обсягом (див. план); чиста логіка
(сканування/argv/калібрування) повністю покрита в tests/test_gui_logic.py.
"""

from __future__ import annotations

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtWidgets import QApplication  # noqa: E402

from dronesim.gui.main_window import MainWindow  # noqa: E402


def test_main_window_constructs_without_raising():
    app = QApplication.instance() or QApplication([])  # noqa: F841 — тримати живим, інакше GC вікна
    window = MainWindow()
    try:
        assert window.windowTitle() == "dronesim — лаунчер"
        assert window.centralWidget() is not None
        assert window.centralWidget().count() == 4  # 4 вкладки
    finally:
        window.close()

"""Вхідна точка GUI-лаунчера (доп. фаза "GUI-лаунчер").

Запуск: python -m dronesim.gui.launcher
"""

from __future__ import annotations

import sys


def main() -> int:
    from PySide6.QtWidgets import QApplication

    from dronesim.gui.main_window import MainWindow

    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())

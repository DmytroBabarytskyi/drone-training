"""Запис подій сценарію у файл (JSON Lines — один запис на рядок).

Критерій приймання фази 4 (ROADMAP.md): "лог подій пишеться у файл". Формат
JSONL обраний за простоту (append-friendly, по одному events за рядок,
grep-able), без залежності від сторонніх бібліотек серіалізації.
"""

from __future__ import annotations

import json
from pathlib import Path


def write_event_log(events: list[dict], path: str | Path) -> None:
    """Записати список подій (dict) у файл ``path`` — по одному JSON-об'єкту на рядок.

    Створює батьківські директорії за потреби (типово ``logs/``, у .gitignore).
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for event in events:
            f.write(json.dumps(event, ensure_ascii=False) + "\n")


def read_event_log(path: str | Path) -> list[dict]:
    """Прочитати назад лог подій, записаний ``write_event_log`` (для тестів/аналізу)."""
    path = Path(path)
    events = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                events.append(json.loads(line))
    return events

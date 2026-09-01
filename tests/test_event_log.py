"""Тести запису/читання логу подій сценарію (scenarios/event_log.py)."""

from dronesim.scenarios.event_log import read_event_log, write_event_log


def test_write_then_read_roundtrip(tmp_path):
    events = [
        {"t": 0.5, "event": "gate", "gate_id": 0},
        {"t": 1.2, "event": "lap", "lap": 1, "lap_time": 1.2},
    ]
    path = tmp_path / "logs" / "race.jsonl"
    write_event_log(events, path)

    assert path.exists()
    loaded = read_event_log(path)
    assert loaded == events


def test_write_creates_parent_directories(tmp_path):
    path = tmp_path / "a" / "b" / "c" / "events.jsonl"
    write_event_log([{"t": 0.0, "event": "start"}], path)
    assert path.exists()


def test_write_empty_events_creates_empty_file(tmp_path):
    path = tmp_path / "empty.jsonl"
    write_event_log([], path)
    assert path.exists()
    assert read_event_log(path) == []


def test_jsonl_format_one_object_per_line(tmp_path):
    path = tmp_path / "events.jsonl"
    write_event_log([{"a": 1}, {"b": 2}], path)
    lines = path.read_text(encoding="utf-8").strip().split("\n")
    assert len(lines) == 2

import logging

from chess.log import LogEntry, LogHandler, LogRing, setup_logging


def entry(level, name, message, ts):
    return LogEntry(timestamp=ts, level=level, name=name, message=message)


def test_ring_keeps_newest_last_and_truncates_to_maxlen():
    ring = LogRing(maxlen=3)
    for ts in range(5):
        ring.append(entry("DEBUG", "chess.game", f"msg{ts}", float(ts)))
    recent = ring.recent()
    assert [e.message for e in recent] == ["msg2", "msg3", "msg4"]
    assert [e.message for e in ring.recent(2)] == ["msg3", "msg4"]


def test_handler_forwards_only_chess_records():
    ring = LogRing(maxlen=50)
    handler = LogHandler(ring)
    chess_root = logging.getLogger("chess")
    chess_root.addHandler(handler)
    try:
        probe = logging.getLogger("chess.test_probe")
        probe.setLevel(logging.DEBUG)
        probe.debug("probe fired")
        assert len(ring.recent()) == 1
        captured = ring.recent()[0]
        assert captured.level == "DEBUG"
        assert captured.name == "chess.test_probe"
        assert captured.message == "probe fired"
        assert isinstance(captured.timestamp, float)

        foreign = logging.LogRecord(
            "unrelated", logging.WARNING, __file__, 1, "nope", None, None
        )
        handler.emit(foreign)
        assert len(ring.recent()) == 1
    finally:
        chess_root.removeHandler(handler)


def test_setup_logging_is_idempotent():
    setup_logging()
    count = len(logging.getLogger("chess").handlers)
    assert count >= 1
    setup_logging()
    assert len(logging.getLogger("chess").handlers) == count

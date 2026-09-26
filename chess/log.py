"""Central logging plumbing for the chess package.

Design
------
The package keeps its own ring buffer of recent log entries *in addition to*
the standard-library ``logging`` output, so the TUI and the web API can
surface the last N events without callers having to install their own
handlers.

Architecture:

- ``LogRing`` is a bounded ring buffer of ``LogEntry`` values.  It never
  grows past ``maxlen`` entries; the oldest entry is evicted on overflow
  (``collections.deque`` semantics), which makes the cost of ``append``
  O(1) and memory usage bounded.
- ``LogHandler`` is a ``logging.Handler`` that forwards every record whose
  logger name starts with ``chess`` into the ring.  It is attached by
  callers (the TUI/web server) that own a ring, not by this module itself.
- ``setup_logging`` configures the ``"chess"`` root logger once with a
  stream handler.  It is idempotent: repeated calls do not stack
  duplicate handlers, and it never touches unrelated loggers (e.g.
  ``uvicorn``) so embedding the package in a server does not hijack the
  host application's logging configuration.
- All library modules obtain their logger via ``get_logger("name")``,
  which returns ``logging.getLogger("chess.<name>")``.  Child loggers
  propagate to the ``"chess"`` root, so a single handler set on the root
  sees every package log line.

Library code MUST NOT call ``logging.basicConfig`` or attach handlers
itself; only the CLI entrypoints (``chess.engine``, ``chess.stockfish``,
``chess.web``) may configure the top-level root logger.
"""

import logging
from collections import deque
from dataclasses import dataclass
from typing import Literal

type LogLevel = Literal["DEBUG", "INFO", "WARNING", "ERROR"]
_LEVEL_TO_LOGLEVEL: dict[int, LogLevel] = {
    logging.DEBUG: "DEBUG",
    logging.INFO: "INFO",
    logging.WARNING: "WARNING",
    logging.ERROR: "ERROR",
}


@dataclass(frozen=True, slots=True)
class LogEntry:
    """A single captured log line with its timestamp, level, and source.

    ``name`` is the logger name (e.g. ``"chess.game"``); keeping it lets
    consumers group or filter entries per module without re-parsing the
    message.
    """

    timestamp: float  # time.time()
    level: LogLevel
    name: str
    message: str


class LogRing:
    """Bounded, newest-on-read ring buffer of :class:`LogEntry` values.

    ``recent`` returns entries with the *newest* entry last, which matches
    how a terminal log panel renders (scrolling output).  When ``count``
    is ``None`` the whole window (up to ``maxlen`` entries) is returned.
    """

    def __init__(self, maxlen: int = 200) -> None:
        self._entries: deque[LogEntry] = deque(maxlen=maxlen)

    def append(self, entry: LogEntry) -> None:
        """Append an entry, evicting the oldest one once full."""
        self._entries.append(entry)

    def recent(self, count: int | None = None) -> tuple[LogEntry, ...]:
        """Return up to ``count`` most recent entries, newest last."""
        if count is None:
            count = len(self._entries)
        return tuple(self._entries)[-count:] if count else ()


class LogHandler(logging.Handler):
    """Forwards ``chess.*`` records into a :class:`LogRing`.

    Records from other logger hierarchies (e.g. ``uvicorn``,
    ``fastapi``) are ignored so a host application's log traffic cannot
    pollute the ring.  ``record.levelno`` is mapped to :data:`LogLevel`
    via a module-level dict; levels outside the four known values
    (e.g. ``CRITICAL`` or a custom level) clamp to ``WARNING`` so the
    entry is still visible without inventing a new literal value.
    """

    def __init__(self, ring: LogRing) -> None:
        super().__init__()
        self._ring = ring

    def emit(self, record: logging.LogRecord) -> None:
        if not record.name.startswith("chess"):
            return
        level = _LEVEL_TO_LOGLEVEL.get(record.levelno, "WARNING")
        self._ring.append(
            LogEntry(
                timestamp=record.created,
                level=level,
                name=record.name,
                message=record.getMessage(),
            )
        )


def setup_logging(level: int = logging.INFO) -> None:
    """Attach a stream handler to the ``"chess"`` root logger.

    Idempotent: if the root ``"chess"`` logger already has a handler, the
    existing one's level is simply updated, so repeated calls (e.g. from a
    test plus a CLI entrypoint) never produce duplicate lines.  Other
    loggers are left untouched.
    """
    root = logging.getLogger("chess")
    existing = next(
        (h for h in root.handlers if isinstance(h, logging.StreamHandler)),
        None,
    )
    if existing is not None:
        existing.setLevel(level)
    else:
        handler = logging.StreamHandler()
        handler.setLevel(level)
        handler.setFormatter(logging.Formatter("%(levelname)s %(name)s: %(message)s"))
        root.addHandler(handler)
    root.setLevel(level)


def get_logger(name: str) -> logging.Logger:
    """Return the ``chess.<name>`` module logger used across the package."""
    return logging.getLogger(f"chess.{name}")

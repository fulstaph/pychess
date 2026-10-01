"""SQLite persistence for game sessions.

Each game is a *session*: a ``games`` row plus one ``moves`` row per played
half-move (UCI notation).  A session is created explicitly when the client
opens a board, so even idle sessions persist across restarts; undo
truncates the move tail, reset rewrites ``start_fen`` and clears the moves,
and replay reconstructs the position from ``start_fen`` + UCI list, which
are the source of truth.

Concurrency model: one ``sqlite3`` connection owned by the ``GameStore``
instance.  The Python ``sqlite3`` connection is not thread-safe even with
``check_same_thread`` off, so every method takes ``_lock`` before touching
it.  The web layer still serializes mutations of one session on that
session's own lock; the store lock is what keeps two sessions from
interleaving statements on the shared connection.
"""

from __future__ import annotations

import sqlite3
import threading
from dataclasses import dataclass
from datetime import UTC, datetime

from .log import get_logger

logger = get_logger("store")

_SCHEMA = """
CREATE TABLE IF NOT EXISTS games (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT NOT NULL,
    created_at TEXT NOT NULL,
    start_fen TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS moves (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    game_id INTEGER NOT NULL REFERENCES games(id) ON DELETE CASCADE,
    ply INTEGER NOT NULL,
    uci TEXT NOT NULL,
    UNIQUE (game_id, ply)
);
CREATE INDEX IF NOT EXISTS idx_moves_game ON moves (game_id, ply);
"""


@dataclass(frozen=True, slots=True)
class GameRecord:
    """A game row with its half-move count (for the listing view)."""

    id: int
    name: str
    created_at: str
    start_fen: str
    move_count: int


class GameStore:
    """Thin repository over the ``games`` + ``moves`` tables."""

    def __init__(self, path: str | None = None) -> None:
        # ``path=None`` -> in-memory database (used by tests).
        # RLock: public methods call each other (append_move -> _move_count).
        self._lock = threading.RLock()
        self._conn = sqlite3.connect(path or ":memory:", check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA foreign_keys = ON")
        self._conn.executescript(_SCHEMA)
        self._conn.commit()

    # ------------------------------------------------------------------
    # writes
    # ------------------------------------------------------------------

    def create_game(self, start_fen: str, name: str | None = None) -> int:
        """Insert a fresh game; returns its id.

        ``name`` defaults to a timestamp when omitted, which keeps the
        listing readable without a rename feature.
        """
        label = name or datetime.now(UTC).strftime("%Y-%m-%d %H:%M:%S")
        with self._lock:
            cur = self._conn.execute(
                "INSERT INTO games (name, created_at, start_fen) VALUES (?, ?, ?)",
                (label, datetime.now(UTC).isoformat(), start_fen),
            )
            self._conn.commit()
            lastrowid = cur.lastrowid
        if lastrowid is None:
            raise RuntimeError("Game INSERT returned no id")
        logger.info("Created game %s (%s)", lastrowid, label)
        return lastrowid

    def append_move(self, game_id: int, uci: str) -> None:
        """Append half-move ``len + 1`` to ``game_id``."""
        with self._lock:
            ply = self._move_count(game_id) + 1
            self._conn.execute(
                "INSERT INTO moves (game_id, ply, uci) VALUES (?, ?, ?)",
                (game_id, ply, uci),
            )
            self._conn.commit()

    def set_start_fen(self, game_id: int, start_fen: str) -> None:
        """Rewrite the session's starting position (reset to FEN)."""
        with self._lock:
            self._conn.execute(
                "UPDATE games SET start_fen = ? WHERE id = ?", (start_fen, game_id)
            )
            self._conn.commit()

    def delete_game(self, game_id: int) -> bool:
        """Remove a game and its moves; False when no such id existed."""
        with self._lock:
            cur = self._conn.execute("DELETE FROM games WHERE id = ?", (game_id,))
            self._conn.commit()
            return cur.rowcount > 0

    def truncate_moves(self, game_id: int, keep_plies: int) -> None:
        """Undo: drop every stored half-move with ply > ``keep_plies``."""
        with self._lock:
            self._conn.execute(
                "DELETE FROM moves WHERE game_id = ? AND ply > ?",
                (game_id, keep_plies),
            )
            self._conn.commit()

    # ------------------------------------------------------------------
    # reads
    # ------------------------------------------------------------------

    def list_games(self) -> list[GameRecord]:
        """All games, most recent first, with half-move counts."""
        with self._lock:
            rows = self._conn.execute(
                """
                SELECT g.id, g.name, g.created_at, g.start_fen,
                       COUNT(m.id) AS move_count
                FROM games g
                LEFT JOIN moves m ON m.game_id = g.id
                GROUP BY g.id
                ORDER BY g.id DESC
                """
            ).fetchall()
        return [
            GameRecord(
                id=row["id"],
                name=row["name"],
                created_at=row["created_at"],
                start_fen=row["start_fen"],
                move_count=row["move_count"],
            )
            for row in rows
        ]

    def get_moves(self, game_id: int) -> list[str]:
        """The game's UCI half-moves in order; [] when unknown or empty."""
        with self._lock:
            rows = self._conn.execute(
                "SELECT uci FROM moves WHERE game_id = ? ORDER BY ply", (game_id,)
            ).fetchall()
        return [row["uci"] for row in rows]

    def get_start_fen(self, game_id: int) -> str | None:
        with self._lock:
            rows = self._conn.execute(
                "SELECT start_fen FROM games WHERE id = ?", (game_id,)
            ).fetchall()
        return rows[0]["start_fen"] if rows else None

    # ------------------------------------------------------------------
    # replay
    # ------------------------------------------------------------------

    def get_game(self, game_id: int) -> tuple[str, list[str]]:
        """Return ``(start_fen, ucis)`` for a stored game.

        Raises ``LookupError`` for an unknown id.  The caller replays the
        UCI list itself so it can rebuild ``Move`` objects for its own
        history bookkeeping.
        """
        with self._lock:
            start_fen = self.get_start_fen(game_id)
            if start_fen is None:
                raise LookupError(f"Game {game_id} not found")
            return start_fen, self.get_moves(game_id)

    def close(self) -> None:
        with self._lock:
            self._conn.close()

    def _move_count(self, game_id: int) -> int:
        with self._lock:
            row = self._conn.execute(
                "SELECT COUNT(*) AS n FROM moves WHERE game_id = ?", (game_id,)
            ).fetchone()
        return int(row["n"]) if row else 0

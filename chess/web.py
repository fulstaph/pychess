"""Local Web UI server (FastAPI) with Lichess Chessground board and REST API.

Architecture
------------
The server supports any number of simultaneous game *sessions*.  Each
session is backed by a ``games`` row in SQLite (``--db`` path, default
``games.db``; ``create_app`` without ``db_path`` keeps it in memory) plus
one ``moves`` row per played half-move (UCI notation), and cached in a
per-session in-memory object:

- ``session.game``          -- the live ``ChessGame`` (authoritative position)
- ``session.history``       -- every executed ``Move`` since the last reset
- ``session.history_states`` -- game snapshots, one per history entry plus the
                                initial position; undo restores an older
                                snapshot from this list

The database (``start_fen`` + UCI move list) is the source of truth; the
cache is replayed from it on first access and re-synced on every request
whenever the stored ply count drifts from the cache (e.g. another process
edited the same row).  A cold replay of an N-ply game costs N legal-move
generations — milliseconds for a local game.

Concurrency: handlers run in FastAPI's threadpool, and the AI-reply path
holds a session open for a full engine turn, so ``session.lock`` serializes
requests on the *same* session — without it a concurrent move/undo could
interleave with an in-flight engine reply and corrupt the history.  Each
store write happens inside that lock, so a session's rows are never
mutated by two threads at once.  Different sessions hold different locks
and proceed in parallel.

Endpoints (all JSON unless noted):

    GET  /                      -> the self-contained Chessground HTML page
    GET  /api/sessions          -> all sessions (newest first) with live state
    POST /api/sessions          -> {"fen": "..."} or {}; create a session
    GET  /api/sessions/{id}     -> full serialized position of one session
    POST /api/sessions/{id}/move     -> {"move": "e2e4", "ai_reply": bool,
                          "depth": int?, "stockfish": bool}; optional depth
                          overrides the app's configured search depth
    POST /api/sessions/{id}/ai_move  -> {"depth": int?, "stockfish": bool};
                          optional depth overrides the app default
    POST /api/sessions/{id}/undo     -> {"steps": int}; rolls the session
                          back ``steps`` half-moves (the UI sends 2 in AI
                          mode so the human regains the move)
    POST /api/sessions/{id}/reset    -> {"fen": "..."} or {}; loads a FEN
                          position or the standard start and clears history
    DELETE /api/sessions/{id}   -> remove a session and its moves

Error contract: invalid moves / FENs / bodies return HTTP 400 with
``{"detail": "..."}``; the session is left untouched (fail fast, no partial
state).  Unknown session ids return 404.  Engine failure (no Stockfish
binary, malformed UCI reply) degrades to the built-in minimax instead of
failing the request.
"""

import argparse
import socket
import sys
import threading
import webbrowser
from collections.abc import AsyncIterator, Sequence
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from typing import Any

import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field

from .engine import choose_move, evaluate
from .game import ChessGame, from_fen
from .helpers import square_notation
from .log import get_logger, setup_logging
from .move import Move
from .search import DEFAULT_ENGINE_CONFIG, MAX_SEARCH_DEPTH, EngineConfig
from .stockfish import Stockfish, find_stockfish
from .store import GameStore
from .tui import get_captured_pieces
from .uci import to_uci
from .ui import get_html

logger = get_logger("web")

# Beyond this many plies, listing a session skips the replay needed to show
# its live FEN and reports only the stored move count.
_MAX_LIST_REPLAY = 2000


@dataclass
class WebSession:
    """Mutable per-session game state shared by all requests on that session.

    ``history_states[0]`` is always the position the session started (or was
    reset) with; each executed move appends one more *distinct* sibling
    (``ChessGame.after``), because ``make_move`` mutates the game in place
    and aliasing the live object would leave undo with nothing to restore.
    ``len(history_states) == len(history) + 1`` holds at all times.

    ``lock`` serializes requests on *this* session (see the module
    docstring); sessions never share a lock.
    """

    game_id: int
    game: ChessGame
    history: list[Move] = field(default_factory=list)
    history_states: list[ChessGame] = field(default_factory=list)
    lock: threading.Lock = field(
        default_factory=threading.Lock, repr=False, compare=False
    )

    def __post_init__(self) -> None:
        self.history_states = [self.game]


class SessionRegistry:
    """In-memory cache of sessions keyed by their ``games`` row id.

    The SQLite store is the source of truth; the cache replays
    ``start_fen`` + UCI moves on first access and re-syncs whenever the
    stored ply count drifts from the cache.  ``store is None`` (in-memory
    mode) keeps sessions in the cache only; ids are negative so they can
    never collide with real row ids.
    """

    def __init__(self, store: GameStore | None) -> None:
        self._store = store
        self._cache: dict[int, WebSession] = {}
        self._next_mem_id = 0
        self._lock = threading.Lock()

    # ------------------------------------------------------------------
    # session management
    # ------------------------------------------------------------------

    def list_ids(self) -> list[int]:
        """Known session ids, newest first."""
        with self._lock:
            cached = list(self._cache)
        if self._store is None:
            return sorted(cached)
        return [g.id for g in self._store.list_games()] + [i for i in cached if i < 0]

    def create(self, start_fen: str) -> WebSession:
        """Create a session row (or in-memory session) and cache it.

        The FEN is parsed first, so a rejected FEN raises ``ValueError``
        before any row is inserted — no orphan rows on invalid input.  The
        normalized FEN (``game.to_fen()``) is what gets stored and replayed.
        """
        game = from_fen(start_fen)
        if self._store is not None:
            game_id = self._store.create_game(game.to_fen())
        else:
            self._next_mem_id -= 1
            game_id = self._next_mem_id
        session = WebSession(game_id=game_id, game=game)
        with self._lock:
            self._cache[game_id] = session
        logger.info("Created session %d (%s)", game_id, game.to_fen())
        return session

    def delete(self, game_id: int) -> bool:
        """Delete a session row and drop it from the cache; False when unknown."""
        if self._store is not None and not self._store.delete_game(game_id):
            return False
        with self._lock:
            self._cache.pop(game_id, None)
        logger.info("Deleted session %d", game_id)
        return True

    # ------------------------------------------------------------------
    # replay
    # ------------------------------------------------------------------

    def get(self, game_id: int) -> WebSession | None:
        """Return the cached session, replaying it from the store if cold."""
        with self._lock:
            session = self._cache.get(game_id)
        if session is not None:
            return session
        store = self._store
        if store is None:
            return None
        try:
            start_fen, ucis = store.get_game(game_id)
        except LookupError:
            return None
        session = WebSession(game_id=game_id, game=from_fen(start_fen))
        self._replay_into(session, start_fen, ucis)
        with self._lock:
            existing = self._cache.get(game_id)
            if existing is None:
                self._cache[game_id] = session
            logger.info("Replayed session %d from %d plies", game_id, len(ucis))
        return self._cache[game_id]

    def resync(self, session: WebSession) -> None:
        """Rebuild the cache from the store when the stored ply count drifts."""
        store = self._store
        if store is None:
            return
        try:
            start_fen, ucis = store.get_game(session.game_id)
        except LookupError:
            return
        if len(ucis) == len(session.history):
            return
        logger.info(
            "Re-syncing session %d from %d stored plies",
            session.game_id,
            len(ucis),
        )
        self._replay_into(session, start_fen, ucis)

    def _replay_into(
        self, session: WebSession, start_fen: str, ucis: list[str]
    ) -> None:
        """Replace the cached position/history with a replay of ``ucis``."""
        game = from_fen(start_fen)
        history: list[Move] = []
        states: list[ChessGame] = [game]
        for uci in ucis:
            move = game._select_move(uci)
            game = game.after(move)
            history.append(move)
            states.append(game)
        session.game = game
        session.history = history
        session.history_states = states


def serialize_game_state(
    game: ChessGame,
    moves_history: Sequence[Move] = (),
) -> dict[str, Any]:
    """Serialize complete chess game state into a JSON-friendly dictionary.

    ``dests`` maps source square -> target squares for every legal move; the
    Chessground frontend uses it to highlight legal destinations when a piece
    is dragged.  ``legal_moves`` is the same set as plain UCI strings, which
    the promotion detector (``a7a8q`` probe in ``app.js``) relies on.
    """
    cap_w, cap_b, diff = get_captured_pieces(game.board)
    dests: dict[str, list[str]] = {}
    legal_moves_uci: list[str] = []
    # A draw still has geometric moves; the play surface must not offer them.
    if not game.is_draw() and not game.is_checkmate():
        for move in game.legal_moves():
            from_sq = square_notation(*move.from_square)
            to_sq = square_notation(*move.to_square)
            dests.setdefault(from_sq, []).append(to_sq)
            legal_moves_uci.append(to_uci(move))

    return {
        "fen": game.to_fen(),
        "turn": game.turn,
        "status": game.status.value,
        "is_check": game.is_check(),
        "is_checkmate": game.is_checkmate(),
        "is_stalemate": game.is_stalemate(),
        "is_draw": game.is_draw(),
        "winner": game.get_winner(),
        "move_num": game.move_num,
        "halfmove_clock": game.halfmove_clock,
        "dests": dests,
        "legal_moves": legal_moves_uci,
        "captured_w": [p.type.value for p in cap_w],
        "captured_b": [p.type.value for p in cap_b],
        "material_diff": diff,
        "eval": evaluate(game.board),
        "history": [to_uci(m) for m in moves_history],
    }


class MoveRequest(BaseModel):
    """Body for POST /api/sessions/{id}/move."""

    move: str = ""
    ai_reply: bool = False
    depth: int | None = Field(default=None, ge=1, le=MAX_SEARCH_DEPTH)
    stockfish: bool = False


class UndoRequest(BaseModel):
    """Body for POST /api/sessions/{id}/undo."""

    steps: int = 1


class ResetRequest(BaseModel):
    """Body for POST /api/sessions/{id}/reset and POST /api/sessions."""

    fen: str = ""


class AIMoveRequest(BaseModel):
    """Body for POST /api/sessions/{id}/ai_move."""

    depth: int | None = Field(default=None, ge=1, le=MAX_SEARCH_DEPTH)
    stockfish: bool = False


def _engine_move(
    game: ChessGame,
    history: list[Move],
    depth: int | None,
    use_stockfish: bool,
    stockfish_path: str | None,
    engine_config: EngineConfig = DEFAULT_ENGINE_CONFIG,
) -> Move | None:
    """Ask Stockfish when requested, otherwise fall back to minimax.

    Stockfish is spawned per request (context manager closes the UCI
    process) — cheap for a local server, and it avoids leaking
    subprocesses if the server shuts down mid-game.  Any startup/protocol
    failure (missing binary, invalid ``bestmove`` reply) logs a warning and
    transparently degrades to the built-in alpha-beta engine so the UI never
    sees a 500 from the engine layer.
    """
    search_depth = engine_config.search_depth if depth is None else depth
    if use_stockfish:
        try:
            sf_path = find_stockfish(stockfish_path)
            logger.info("Using Stockfish at %s for AI move", sf_path)
            with Stockfish(path=sf_path, skill_level=0) as sf:
                return sf.get_move(game, moves_history=history)
        except FileNotFoundError:
            logger.warning("Stockfish binary not found; falling back to minimax")
        except RuntimeError as err:
            logger.warning("Stockfish failed (%s); falling back to minimax", err)
    else:
        logger.info("Using built-in minimax (depth=%d, quiescence)", search_depth)
    return choose_move(
        game,
        depth=search_depth,
        quiescence=engine_config.quiescence,
        config=engine_config,
    )


def create_app(
    stockfish_path: str | None = None,
    db_path: str | None = None,
    engine_config: EngineConfig = DEFAULT_ENGINE_CONFIG,
) -> FastAPI:
    """Build a FastAPI app with its own session registry.

    The registry lives on ``app.state`` (not a module global) so each app
    instance — e.g. in tests via ``TestClient`` — gets its own isolated
    sessions.  ``db_path`` selects the SQLite file for persistence;
    ``None`` keeps everything in memory (the default, and what tests use).
    """

    @asynccontextmanager
    async def lifespan(_: FastAPI) -> AsyncIterator[None]:
        yield
        app.state.store.close()

    app = FastAPI(title="pychess Web UI", lifespan=lifespan)
    app.state.store = GameStore(db_path)
    app.state.registry = SessionRegistry(app.state.store)
    app.state.stockfish_path = stockfish_path
    app.state.engine_config = engine_config

    def require_session(game_id: int) -> WebSession:
        """Resolve a session id, 404 when unknown."""
        registry: SessionRegistry = app.state.registry
        session = registry.get(game_id)
        if session is None:
            raise HTTPException(status_code=404, detail="Session not found")
        return session

    @app.get("/", response_class=HTMLResponse)
    def index() -> str:
        """Serve the self-contained frontend (CSS/JS inlined by ``chess.ui``)."""
        return get_html()

    # ------------------------------------------------------------------
    # session management
    # ------------------------------------------------------------------

    @app.get("/api/sessions")
    def list_sessions() -> list[dict[str, Any]]:
        """All sessions, newest first, with their live state."""
        registry: SessionRegistry = app.state.registry
        store: GameStore = app.state.store
        rows = {g.id: g for g in store.list_games()} if store is not None else {}
        out: list[dict[str, Any]] = []
        for game_id in registry.list_ids():
            session = registry.get(game_id)
            if session is None:
                continue
            with session.lock:
                registry.resync(session)
                move_count = len(session.history)
                entry: dict[str, Any] = {"id": game_id, "move_count": move_count}
                row = rows.get(game_id)
                if row is not None:
                    entry["name"] = row.name
                    entry["created_at"] = row.created_at
                if move_count > _MAX_LIST_REPLAY:
                    entry["fen"] = None
                else:
                    entry["state"] = serialize_game_state(session.game, session.history)
                out.append(entry)
        return out

    @app.post("/api/sessions")
    def create_session(body: ResetRequest) -> dict[str, Any]:
        """Create a fresh session from a FEN (default: standard start)."""
        registry: SessionRegistry = app.state.registry
        fen = body.fen.strip()
        if fen:
            try:
                session = registry.create(fen)
            except ValueError as err:
                logger.info("Rejected FEN %r: %s", fen, err)
                raise HTTPException(
                    status_code=400, detail=f"Invalid FEN: {err}"
                ) from err
        else:
            session = registry.create(ChessGame().to_fen())
        return {
            "id": session.game_id,
            **serialize_game_state(session.game, session.history),
        }

    @app.get("/api/sessions/{game_id}")
    def get_session_state(game_id: int) -> dict[str, Any]:
        """Full serialized position for one session."""
        registry: SessionRegistry = app.state.registry
        session = require_session(game_id)
        with session.lock:
            registry.resync(session)
            return serialize_game_state(session.game, session.history)

    @app.delete("/api/sessions/{game_id}")
    def delete_session(game_id: int) -> dict[str, Any]:
        """Remove a session row and its moves."""
        registry: SessionRegistry = app.state.registry
        if not registry.delete(game_id):
            raise HTTPException(status_code=404, detail="Session not found")
        return {"deleted": game_id}

    # ------------------------------------------------------------------
    # per-session play
    # ------------------------------------------------------------------

    @app.post("/api/sessions/{game_id}/move")
    def move(game_id: int, body: MoveRequest) -> dict[str, Any]:
        """Execute a human move, optionally followed by an engine reply.

        The move is applied first and recorded; only then is the engine
        consulted, so a slow or failing engine can never roll back the
        human's move.  An invalid move raises 400 before any state changes.
        """
        store: GameStore = app.state.store
        registry: SessionRegistry = app.state.registry
        session = require_session(game_id)
        with session.lock:
            registry.resync(session)
            if session.game.is_draw() or session.game.is_checkmate():
                raise HTTPException(status_code=400, detail="Game is over")
            if not body.move.strip():
                raise HTTPException(status_code=400, detail="No move specified")
            try:
                executed = session.game._select_move(body.move)
                session.game = session.game.after(executed)
            except ValueError as err:
                logger.info("Rejected move %r: %s", body.move, err)
                raise HTTPException(
                    status_code=400, detail=f"Invalid move: {err}"
                ) from err

            session.history.append(executed)
            session.history_states.append(session.game)
            if store is not None:
                store.append_move(session.game_id, to_uci(executed))
            logger.info("Executed %s; turn now %s", to_uci(executed), session.game.turn)

            if body.ai_reply and session.game.legal_moves():
                ai_move = _engine_move(
                    session.game,
                    session.history,
                    body.depth,
                    body.stockfish,
                    app.state.stockfish_path,
                    engine_config=app.state.engine_config,
                )
                if ai_move is not None:
                    session.game = session.game.after(ai_move)
                    session.history.append(ai_move)
                    session.history_states.append(session.game)
                    if store is not None:
                        store.append_move(session.game_id, to_uci(ai_move))
                    logger.info(
                        "AI replied %s; turn now %s",
                        to_uci(ai_move),
                        session.game.turn,
                    )
                else:
                    logger.warning("Engine returned no move; game state unchanged")

            return serialize_game_state(session.game, session.history)

    @app.post("/api/sessions/{game_id}/ai_move")
    def ai_move(game_id: int, body: AIMoveRequest) -> dict[str, Any]:
        """Play one engine move for the side to move (no human move involved)."""
        store: GameStore = app.state.store
        registry: SessionRegistry = app.state.registry
        session = require_session(game_id)
        with session.lock:
            registry.resync(session)
            if session.game.is_draw() or session.game.is_checkmate():
                raise HTTPException(status_code=400, detail="Game is over")
            if not session.game.legal_moves():
                raise HTTPException(status_code=400, detail="No legal moves available")
            ai_move = _engine_move(
                session.game,
                session.history,
                body.depth,
                body.stockfish,
                app.state.stockfish_path,
                engine_config=app.state.engine_config,
            )
            if ai_move is not None:
                session.game = session.game.after(ai_move)
                session.history.append(ai_move)
                session.history_states.append(session.game)
                if store is not None:
                    store.append_move(session.game_id, to_uci(ai_move))
                logger.info(
                    "AI move %s; turn now %s", to_uci(ai_move), session.game.turn
                )
            return serialize_game_state(session.game, session.history)

    @app.post("/api/sessions/{game_id}/undo")
    def undo(game_id: int, body: UndoRequest) -> dict[str, Any]:
        """Roll back ``steps`` half-moves; extra steps are silently ignored
        once the session has unwound to its initial position."""
        store: GameStore = app.state.store
        registry: SessionRegistry = app.state.registry
        session = require_session(game_id)
        with session.lock:
            registry.resync(session)
            for _ in range(body.steps):
                if len(session.history_states) > 1:
                    session.history_states.pop()
                    session.history.pop()
            session.game = session.history_states[-1]
            if store is not None:
                store.truncate_moves(session.game_id, len(session.history))
            logger.info(
                "Undo %d step(s); history now %d move(s)",
                body.steps,
                len(session.history),
            )
            return serialize_game_state(session.game, session.history)

    @app.post("/api/sessions/{game_id}/reset")
    def reset(game_id: int, body: ResetRequest) -> dict[str, Any]:
        """Load a FEN position, or the standard start when the body is empty.

        Invalid FENs return 400 and leave the session untouched.
        """
        store: GameStore = app.state.store
        registry: SessionRegistry = app.state.registry
        session = require_session(game_id)
        with session.lock:
            registry.resync(session)
            if body.fen.strip():
                try:
                    session.game = from_fen(body.fen)
                except ValueError as err:
                    logger.info("Rejected FEN %r: %s", body.fen, err)
                    raise HTTPException(
                        status_code=400, detail=f"Invalid FEN: {err}"
                    ) from err
                if store is not None:
                    store.set_start_fen(session.game_id, session.game.to_fen())
                logger.info(
                    "Session %d reset to FEN: %s",
                    session.game_id,
                    session.game.to_fen(),
                )
            else:
                session.game = ChessGame()
                if store is not None:
                    store.set_start_fen(session.game_id, session.game.to_fen())
                logger.info(
                    "Session %d reset to standard start position", session.game_id
                )
            if store is not None:
                store.truncate_moves(session.game_id, 0)
            session.history = []
            session.history_states = [session.game]
            return serialize_game_state(session.game, session.history)

    return app


def _find_free_port(host: str, port: int) -> int:
    """Bind-probe ``port`` and the nine following ports for a free one.

    The bind is released immediately (context manager) so there is a small
    race window before ``uvicorn`` re-binds; acceptable for a local tool.
    Exits with a message if none of the candidates can be bound.
    """
    for candidate in range(port, port + 10):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            try:
                sock.bind((host, candidate))
                if candidate != port:
                    logger.info("Port %d busy; using %d", port, candidate)
                return candidate
            except OSError:
                continue
    sys.stderr.write(f"Error: Could not bind to {host}:{port} or subsequent ports.\n")
    sys.exit(1)


def run_web_server(
    host: str = "127.0.0.1",
    port: int = 8000,
    open_browser: bool = True,
    stockfish_path: str | None = None,
    db_path: str = "games.db",
    engine_config: EngineConfig = DEFAULT_ENGINE_CONFIG,
) -> None:
    """Start local web chessboard UI server and open in browser."""
    # Configure logging once at the process entry point so both uvicorn's
    # own records and chess.web records share one consistent format.
    setup_logging()

    actual_port = _find_free_port(host, port)
    url = f"http://{host}:{actual_port}"
    print(f"pychess Web UI running at: {url}")
    print("Press Ctrl+C to stop the server.")
    logger.info("Binding FastAPI app to %s", url)

    if open_browser:
        webbrowser.open(url)

    try:
        # ``uvicorn.run`` blocks until the loop is cancelled (Ctrl+C) or the
        # process receives EOF on stdin; the ASGI app is passed as an object
        # (not an import string) because it carries the live session state.
        uvicorn.run(
            create_app(
                stockfish_path=stockfish_path,
                db_path=db_path,
                engine_config=engine_config,
            ),
            host=host,
            port=actual_port,
            log_level="warning",
        )
    except KeyboardInterrupt, EOFError:
        print("\nShutting down web server...")
        logger.info("Server stopped")


def main(argv: Sequence[str] | None = None) -> None:
    """CLI entrypoint for running pychess Web UI."""
    parser = argparse.ArgumentParser(
        prog="python -m chess.web",
        description="Run local pychess Web UI with Lichess Chessground.",
    )
    parser.add_argument(
        "--port",
        "-p",
        type=int,
        default=8000,
        help="Local port to bind server (default: 8000)",
    )
    parser.add_argument(
        "--depth",
        type=int,
        choices=range(1, MAX_SEARCH_DEPTH + 1),
        default=DEFAULT_ENGINE_CONFIG.search_depth,
        help=f"Built-in engine search depth (1-{MAX_SEARCH_DEPTH})",
    )
    parser.add_argument(
        "--host",
        type=str,
        default="127.0.0.1",
        help="Host address to bind server (default: 127.0.0.1)",
    )
    parser.add_argument(
        "--no-browser",
        action="store_true",
        help="Do not automatically open default browser",
    )
    parser.add_argument(
        "--stockfish-path",
        type=str,
        default=None,
        help="Optional path to Stockfish executable",
    )
    parser.add_argument(
        "--db",
        type=str,
        default="games.db",
        help="SQLite file for persisted sessions (default: games.db)",
    )
    args = parser.parse_args(argv)

    run_web_server(
        host=args.host,
        port=args.port,
        open_browser=not args.no_browser,
        stockfish_path=args.stockfish_path,
        db_path=args.db,
        engine_config=EngineConfig(search_depth=args.depth),
    )


if __name__ == "__main__":
    main()

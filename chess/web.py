"""Local Web UI server (FastAPI) with Lichess Chessground board and REST API.

Architecture
------------
The server keeps exactly ONE in-memory game per process (``WebSession`` on
``app.state``).  This matches the single-user, single-board scope of the UI:

- ``session.game``          -- the live ``ChessGame`` (authoritative position)
- ``session.history``       -- every executed ``Move`` since the last reset
- ``session.history_states`` -- game snapshots, one per history entry plus the
                                initial position; undo restores an older
                                snapshot from this list

All state is mutated in place on purpose: ``app.state`` lives on the ASGI app
object, which outlives any single request, and FastAPI request handlers must
share one session.  ``ChessGame`` itself is still only mutated through its
own atomic ``make_move`` / property interface.

Endpoints (all JSON unless noted):

    GET  /            -> the self-contained Chessground HTML page
    GET  /api/state   -> full serialized position for the current session
    POST /api/move    -> {"move": "e2e4", "ai_reply": bool, "depth": int,
                          "stockfish": bool}; executes the human move and,
                          when ``ai_reply`` is set, immediately plays the
                          engine's reply in the same request
    POST /api/ai_move -> {"depth": int, "stockfish": bool}; plays one engine
                          move for the side to move
    POST /api/undo    -> {"steps": int}; rolls the session back ``steps``
                          half-moves (the UI sends 2 in AI mode so the human
                          regains the move)
    POST /api/reset   -> {"fen": "..."} or {}; loads a FEN position or the
                          standard start and clears the history

Error contract: invalid moves / FENs / bodies return HTTP 400 with
``{"detail": "..."}``; the session is left untouched (fail fast, no partial
state).  Engine failure (no Stockfish binary, malformed UCI reply) degrades
to the built-in minimax instead of failing the request.
"""

import argparse
import socket
import sys
import webbrowser
from collections.abc import Sequence
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
from .stockfish import Stockfish, find_stockfish
from .tui import get_captured_pieces
from .uci import to_uci
from .ui import get_html

logger = get_logger("web")


@dataclass
class WebSession:
    """Mutable per-server game state shared by all requests.

    ``history_states[0]`` is always the position the session started (or was
    reset) with; each executed move appends one more snapshot.  Undo pops
    from the tail of both lists, so ``len(history_states) ==
    len(history) + 1`` holds at all times.
    """

    game: ChessGame
    history: list[Move] = field(default_factory=list)
    history_states: list[ChessGame] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.history_states = [self.game]


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
    """Body for POST /api/move."""

    move: str = ""
    ai_reply: bool = False
    depth: int = Field(default=2, ge=1)
    stockfish: bool = False


class UndoRequest(BaseModel):
    """Body for POST /api/undo."""

    steps: int = Field(default=1, ge=1)


class ResetRequest(BaseModel):
    """Body for POST /api/reset."""

    fen: str = ""


class AIMoveRequest(BaseModel):
    """Body for POST /api/ai_move."""

    depth: int = Field(default=2, ge=1)
    stockfish: bool = False


def _engine_move(
    game: ChessGame,
    history: list[Move],
    depth: int,
    use_stockfish: bool,
    stockfish_path: str | None,
) -> Move | None:
    """Ask Stockfish when requested, otherwise fall back to minimax.

    Stockfish is spawned per request (context manager closes the UCI
    process) — cheap for a local single-user server, and it avoids leaking
    subprocesses if the server shuts down mid-game.  Any startup/protocol
    failure (missing binary, invalid ``bestmove`` reply) logs a warning and
    transparently degrades to the built-in alpha-beta engine so the UI never
    sees a 500 from the engine layer.
    """
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
        logger.info("Using built-in minimax (depth=%d, quiescence)", depth)
    return choose_move(game, depth=depth, quiescence=True)


def create_app(stockfish_path: str | None = None) -> FastAPI:
    """Build a FastAPI app with a fresh game session on ``app.state``.

    The session lives on ``app.state`` (not a module global) so each app
    instance — e.g. in tests via ``TestClient`` — gets its own isolated game.
    """
    app = FastAPI(title="pychess Web UI")
    app.state.session = WebSession(game=ChessGame())
    app.state.stockfish_path = stockfish_path

    @app.get("/", response_class=HTMLResponse)
    def index() -> str:
        """Serve the self-contained frontend (CSS/JS inlined by ``chess.ui``)."""
        return get_html()

    @app.get("/api/state")
    def api_state() -> dict[str, Any]:
        """Return the current position; the UI polls it after load."""
        session: WebSession = app.state.session
        return serialize_game_state(session.game, session.history)

    @app.post("/api/move")
    def api_move(body: MoveRequest) -> dict[str, Any]:
        """Execute a human move, optionally followed by an engine reply.

        The move is applied first and its ``Move`` recorded; only then is the
        engine consulted, so a slow or failing engine can never roll back the
        human's move.  An invalid move raises 400 before any state changes.
        """
        session: WebSession = app.state.session
        if not body.move.strip():
            raise HTTPException(status_code=400, detail="No move specified")
        try:
            executed = session.game.make_move(body.move)
        except ValueError as err:
            logger.info("Rejected move %r: %s", body.move, err)
            raise HTTPException(status_code=400, detail=f"Invalid move: {err}") from err

        session.history.append(executed)
        session.history_states.append(session.game)
        logger.info("Executed %s; turn now %s", to_uci(executed), session.game.turn)

        if body.ai_reply and session.game.legal_moves():
            ai_move = _engine_move(
                session.game,
                session.history,
                body.depth,
                body.stockfish,
                app.state.stockfish_path,
            )
            if ai_move is not None:
                session.game.make_move(ai_move)
                session.history.append(ai_move)
                session.history_states.append(session.game)
                logger.info(
                    "AI replied %s; turn now %s", to_uci(ai_move), session.game.turn
                )
            else:
                logger.warning("Engine returned no move; game state unchanged")

        return serialize_game_state(session.game, session.history)

    @app.post("/api/ai_move")
    def api_ai_move(body: AIMoveRequest) -> dict[str, Any]:
        """Play one engine move for the side to move (no human move involved)."""
        session: WebSession = app.state.session
        if not session.game.legal_moves():
            raise HTTPException(status_code=400, detail="No legal moves available")
        ai_move = _engine_move(
            session.game,
            session.history,
            body.depth,
            body.stockfish,
            app.state.stockfish_path,
        )
        if ai_move is not None:
            session.game.make_move(ai_move)
            session.history.append(ai_move)
            session.history_states.append(session.game)
            logger.info("AI move %s; turn now %s", to_uci(ai_move), session.game.turn)
        return serialize_game_state(session.game, session.history)

    @app.post("/api/undo")
    def api_undo(body: UndoRequest) -> dict[str, Any]:
        """Roll back ``steps`` half-moves; extra steps are silently ignored
        once the session has unwound to its initial position."""
        session: WebSession = app.state.session
        for _ in range(body.steps):
            if len(session.history_states) > 1:
                session.history_states.pop()
                if session.history:
                    session.history.pop()
        session.game = session.history_states[-1]
        logger.info(
            "Undo %d step(s); history now %d move(s)",
            body.steps,
            len(session.history),
        )
        return serialize_game_state(session.game, session.history)

    @app.post("/api/reset")
    def api_reset(body: ResetRequest) -> dict[str, Any]:
        """Load a FEN position, or the standard start when the body is empty.

        Invalid FENs return 400 and leave the current game untouched.
        """
        session: WebSession = app.state.session
        if body.fen.strip():
            try:
                session.game = from_fen(body.fen)
            except ValueError as err:
                logger.info("Rejected FEN %r: %s", body.fen, err)
                raise HTTPException(
                    status_code=400, detail=f"Invalid FEN: {err}"
                ) from err
            logger.info("Reset to FEN: %s", body.fen)
        else:
            session.game = ChessGame()
            logger.info("Reset to standard start position")
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
            create_app(stockfish_path=stockfish_path),
            host=host,
            port=actual_port,
            log_level="warning",
        )
    except (KeyboardInterrupt, EOFError):
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
    args = parser.parse_args(argv)

    run_web_server(
        host=args.host,
        port=args.port,
        open_browser=not args.no_browser,
        stockfish_path=args.stockfish_path,
    )


if __name__ == "__main__":
    main()

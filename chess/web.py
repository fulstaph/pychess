"""Local Web UI server with Lichess Chessground board and REST API."""

import argparse
import http.server
import json
import socketserver
import sys
import webbrowser
from collections.abc import Sequence
from typing import ClassVar

from .engine import choose_move, evaluate
from .game import ChessGame, from_fen
from .helpers import square_notation
from .move import Move
from .stockfish import Stockfish, find_stockfish
from .tui import get_captured_pieces
from .uci import to_uci
from .ui import get_html


def serialize_game_state(
    game: ChessGame,
    moves_history: Sequence[Move] = (),
) -> dict[str, object]:
    """Serialize complete chess game state into a JSON-friendly dictionary."""
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


def get_html_template() -> str:
    """Return the frontend HTML document loaded from chess.ui package."""
    return get_html()


class ChessWebHandler(http.server.BaseHTTPRequestHandler):
    """HTTP request handler providing REST API and Lichess Chessground frontend."""

    game: ClassVar[ChessGame] = ChessGame()
    history: ClassVar[list[Move]] = []
    history_states: ClassVar[list[ChessGame]] = [game]
    stockfish_path: ClassVar[str | None] = None

    def log_message(self, format_str: str, *args: object) -> None:
        """Suppress default stdout request logging."""

    def _send_json(self, payload: dict[str, object], status: int = 200) -> None:
        raw = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(raw)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(raw)

    def do_GET(self) -> None:
        if self.path in ("/", "/index.html"):
            raw_html = get_html().encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(raw_html)))
            self.end_headers()
            self.wfile.write(raw_html)
            return

        if self.path == "/api/state":
            state_data = serialize_game_state(self.game, self.history)
            self._send_json(state_data)
            return

        self.send_error(404, "Endpoint not found")

    def do_POST(self) -> None:
        content_length = int(self.headers.get("Content-Length", 0))
        post_body = self.rfile.read(content_length) if content_length > 0 else b"{}"
        try:
            body: dict[str, object] = (
                json.loads(post_body.decode("utf-8")) if post_body else {}
            )
        except json.JSONDecodeError:
            body = {}

        if self.path == "/api/move":
            move_str = str(body.get("move", "")).strip()
            if not move_str:
                self._send_json({"error": "No move specified"}, status=400)
                return
            try:
                executed = self.game.make_move(move_str)
            except ValueError as err:
                self._send_json({"error": f"Invalid move: {err}"}, status=400)
                return

            self.history.append(executed)
            self.history_states.append(self.game)

            # Trigger AI reply if requested
            ai_reply = bool(body.get("ai_reply", False))
            if ai_reply and self.game.legal_moves():
                depth = int(str(body.get("depth", "2")))
                use_sf = bool(body.get("stockfish", False))
                ai_move: Move | None = None
                if use_sf:
                    try:
                        sf_path = find_stockfish(self.stockfish_path)
                        with Stockfish(path=sf_path, skill_level=0) as sf:
                            ai_move = sf.get_move(self.game, moves_history=self.history)
                    except FileNotFoundError, RuntimeError:
                        ai_move = choose_move(self.game, depth=depth, quiescence=True)
                else:
                    ai_move = choose_move(self.game, depth=depth, quiescence=True)

                if ai_move:
                    self.game.make_move(ai_move)
                    self.history.append(ai_move)
                    self.history_states.append(self.game)

            self._send_json(serialize_game_state(self.game, self.history))
            return

        if self.path == "/api/ai_move":
            if not self.game.legal_moves():
                self._send_json({"error": "No legal moves available"}, status=400)
                return
            depth = int(str(body.get("depth", "2")))
            use_sf = bool(body.get("stockfish", False))
            ai_move = None
            if use_sf:
                try:
                    sf_path = find_stockfish(self.stockfish_path)
                    with Stockfish(path=sf_path, skill_level=0) as sf:
                        ai_move = sf.get_move(self.game, moves_history=self.history)
                except FileNotFoundError, RuntimeError:
                    ai_move = choose_move(self.game, depth=depth, quiescence=True)
            else:
                ai_move = choose_move(self.game, depth=depth, quiescence=True)

            if ai_move:
                self.game.make_move(ai_move)
                self.history.append(ai_move)
                self.history_states.append(self.game)

            self._send_json(serialize_game_state(self.game, self.history))
            return

        if self.path == "/api/undo":
            steps = int(str(body.get("steps", "1")))
            for _ in range(steps):
                if len(self.history_states) > 1:
                    self.history_states.pop()
                    if self.history:
                        self.history.pop()
            ChessWebHandler.game = self.history_states[-1]
            self._send_json(serialize_game_state(self.game, self.history))
            return

        if self.path == "/api/reset":
            fen_input = str(body.get("fen", "")).strip()
            if fen_input:
                try:
                    new_game = from_fen(fen_input)
                except ValueError as err:
                    self._send_json({"error": f"Invalid FEN: {err}"}, status=400)
                    return
                ChessWebHandler.game = new_game
            else:
                ChessWebHandler.game = ChessGame()

            ChessWebHandler.history = []
            ChessWebHandler.history_states = [ChessWebHandler.game]
            self._send_json(serialize_game_state(self.game, self.history))
            return

        self.send_error(404, "Endpoint not found")


def create_web_server(
    host: str = "127.0.0.1",
    port: int = 8000,
    stockfish_path: str | None = None,
) -> http.server.HTTPServer:
    """Create a configured HTTPServer instance for pychess web UI."""
    ChessWebHandler.game = ChessGame()
    ChessWebHandler.history = []
    ChessWebHandler.history_states = [ChessWebHandler.game]
    ChessWebHandler.stockfish_path = stockfish_path

    class ThreadedHTTPServer(socketserver.ThreadingMixIn, http.server.HTTPServer):
        daemon_threads = True

    return ThreadedHTTPServer((host, port), ChessWebHandler)


def run_web_server(
    host: str = "127.0.0.1",
    port: int = 8000,
    open_browser: bool = True,
    stockfish_path: str | None = None,
) -> None:
    """Start local web chessboard UI server and open in browser."""
    actual_port = port
    server: http.server.HTTPServer | None = None

    for attempt in range(10):
        try:
            server = create_web_server(
                host=host, port=actual_port, stockfish_path=stockfish_path
            )
            break
        except OSError:
            actual_port = port + attempt + 1

    if server is None:
        sys.stderr.write(
            f"Error: Could not bind to {host}:{port} or subsequent ports.\n"
        )
        sys.exit(1)

    url = f"http://{host}:{actual_port}"
    print(f"pychess Web UI running at: {url}")
    print("Press Ctrl+C to stop the server.")

    if open_browser:
        webbrowser.open(url)

    try:
        server.serve_forever()
    except KeyboardInterrupt, EOFError:
        print("\nShutting down web server...")
    finally:
        server.server_close()


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

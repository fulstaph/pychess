"""Chess engine with immutable boards, legal moves, and a terminal player."""

from typing import TYPE_CHECKING

from .game import ChessGame, GameStatus, from_fen, print_board, to_fen
from .helpers import PlayerColor, notation_to_coords, square_notation
from .move import Move
from .piece import Board, Color, Piece, PieceType, Square

if TYPE_CHECKING:
    from .log import (
        LogEntry,
        LogRing,
        get_logger,
        setup_logging,
    )
    from .stockfish import (
        MatchResult,
        SeriesResult,
        Stockfish,
        find_stockfish,
        format_series_summary,
        play_match,
        run_series,
    )
    from .tui import (
        format_pgn,
        get_captured_pieces,
        render_dashboard,
    )
    from .uci import (
        UCIEngine,
        UCIEngineError,
        run_uci_server,
        to_uci,
    )
    from .web import (
        create_app,
        run_web_server,
    )

__all__ = [
    "Board",
    "ChessGame",
    "Color",
    "GameStatus",
    "LogEntry",
    "LogRing",
    "MatchResult",
    "Move",
    "Piece",
    "PieceType",
    "PlayerColor",
    "SeriesResult",
    "Square",
    "Stockfish",
    "UCIEngine",
    "UCIEngineError",
    "create_app",
    "find_stockfish",
    "format_pgn",
    "format_series_summary",
    "from_fen",
    "get_captured_pieces",
    "get_logger",
    "notation_to_coords",
    "play_match",
    "print_board",
    "render_dashboard",
    "run_series",
    "run_uci_server",
    "run_web_server",
    "setup_logging",
    "square_notation",
    "to_fen",
    "to_uci",
]

__version__ = "0.1.0"


# Lazily-imported attributes. ``web`` (FastAPI/uvicorn) and the other heavy
# modules are only imported on first attribute access, so a plain
# ``import chess`` stays dependency-free at import time.
def __getattr__(name: str) -> object:
    if name in ("LogEntry", "LogRing", "get_logger", "setup_logging"):
        from . import log

        return getattr(log, name)
    if name in (
        "MatchResult",
        "SeriesResult",
        "Stockfish",
        "find_stockfish",
        "format_series_summary",
        "play_match",
        "run_series",
    ):
        from . import stockfish

        return getattr(stockfish, name)
    if name in (
        "UCIEngine",
        "UCIEngineError",
        "run_uci_server",
        "to_uci",
    ):
        from . import uci

        return getattr(uci, name)
    if name in (
        "format_pgn",
        "get_captured_pieces",
        "render_dashboard",
    ):
        from . import tui

        return getattr(tui, name)
    if name in ("run_web_server", "create_app"):
        from . import web

        return getattr(web, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

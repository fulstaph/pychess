"""Tests for UCI protocol communication, move serialization, and FEN generation."""

import io
import sys
from pathlib import Path

import pytest

from chess import (
    Board,
    ChessGame,
    Piece,
    PieceType,
    UCIEngine,
    UCIEngineError,
    notation_to_coords,
    run_uci_server,
    to_fen,
    to_uci,
)

FAKE_ENGINE_SCRIPT = str(Path(__file__).parent / "fake_uci_engine.py")


def sq(name: str) -> tuple[int, int]:
    return notation_to_coords(name)


def test_to_uci_formats():
    game = ChessGame()

    # Normal pawn push
    move_e4 = game._select_move("e4")
    assert to_uci(move_e4) == "e2e4"

    # Knight move
    move_nf3 = game._select_move("Nf3")
    assert to_uci(move_nf3) == "g1f3"

    # Castling kingside (White)
    board_k = (
        Board()
        .with_piece(sq("e1"), Piece(PieceType.KING, "w"))
        .with_piece(sq("h1"), Piece(PieceType.ROOK, "w"))
        .with_piece(sq("e8"), Piece(PieceType.KING, "b"))
    )
    game_k = ChessGame(board_k, castling_rights=frozenset("K"))
    move_castle_k = game_k._select_move("O-O")
    assert to_uci(move_castle_k) == "e1g1"

    # Castling queenside (White)
    board_q = (
        Board()
        .with_piece(sq("e1"), Piece(PieceType.KING, "w"))
        .with_piece(sq("a1"), Piece(PieceType.ROOK, "w"))
        .with_piece(sq("e8"), Piece(PieceType.KING, "b"))
    )
    game_q = ChessGame(board_q, castling_rights=frozenset("Q"))
    move_castle_q = game_q._select_move("O-O-O")
    assert to_uci(move_castle_q) == "e1c1"

    # Castling kingside and queenside (Black)
    board_b = (
        Board()
        .with_piece(sq("e8"), Piece(PieceType.KING, "b"))
        .with_piece(sq("h8"), Piece(PieceType.ROOK, "b"))
        .with_piece(sq("a8"), Piece(PieceType.ROOK, "b"))
        .with_piece(sq("e1"), Piece(PieceType.KING, "w"))
    )
    game_bk = ChessGame(board_b, turn="b", castling_rights=frozenset("k"))
    assert to_uci(game_bk._select_move("o-o")) == "e8g8"

    game_bq = ChessGame(board_b, turn="b", castling_rights=frozenset("q"))
    assert to_uci(game_bq._select_move("o-o-o")) == "e8c8"

    # Promotion
    board_promo = (
        Board()
        .with_piece(sq("a7"), Piece(PieceType.PAWN, "w"))
        .with_piece(sq("h1"), Piece(PieceType.KING, "w"))
        .with_piece(sq("h8"), Piece(PieceType.KING, "b"))
    )
    game_promo = ChessGame(board_promo)
    assert to_uci(game_promo._select_move("a8=Q")) == "a7a8q"
    assert to_uci(game_promo._select_move("a8=R")) == "a7a8r"
    assert to_uci(game_promo._select_move("a8=B")) == "a7a8b"
    assert to_uci(game_promo._select_move("a8=N")) == "a7a8n"

    # En passant
    game_ep = ChessGame()
    game_ep.make_move("e4")
    game_ep.make_move("a6")
    game_ep.make_move("e5")
    game_ep.make_move("d5")
    move_ep = game_ep._select_move("exd6")
    assert to_uci(move_ep) == "e5d6"


def test_to_fen_serialization():
    game = ChessGame()
    initial_fen = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
    assert to_fen(game) == initial_fen
    assert game.to_fen() == initial_fen

    game.make_move("e4")
    assert game.to_fen() == (
        "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq e3 0 1"
    )

    game.make_move("d5")
    assert (
        game.to_fen() == "rnbqkbnr/ppp1pppp/8/3p4/4P3/8/PPPP1PPP/RNBQKBNR w KQkq d6 0 2"
    )


def test_uci_engine_lifecycle_and_protocol():
    with UCIEngine(sys.executable, args=[FAKE_ENGINE_SCRIPT]) as engine:
        engine.set_option("Threads", 2)
        engine.set_option("Hash", 32)
        engine.new_game()

        # Set position with startpos and moves
        engine.set_position(moves=["e2e4", "e7e5"])
        best = engine.go(movetime_ms=50, depth=1)
        assert isinstance(best, str)
        assert len(best) >= 4

        # Set position with FEN
        test_fen = "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq e3 0 1"
        engine.set_position(fen=test_fen)
        best2 = engine.go(movetime_ms=50, depth=1)
        assert isinstance(best2, str)
        assert len(best2) >= 4

    # Process closed after with block
    assert engine._closed


def test_uci_engine_error_handling():
    # Spawning nonexistent binary
    with pytest.raises(UCIEngineError):
        UCIEngine("/nonexistent/path/to/engine")

    # Command on closed engine
    engine = UCIEngine(sys.executable, args=[FAKE_ENGINE_SCRIPT])
    engine.close()
    with pytest.raises(UCIEngineError):
        engine.send_command("isready")

    # Engine terminating unexpectedly
    with UCIEngine(sys.executable, args=[FAKE_ENGINE_SCRIPT]) as eng:
        eng.send_command("quit")
        with pytest.raises((UCIEngineError, TimeoutError)):
            eng.wait_for("readyok", timeout=1.0)


def test_run_uci_server_in_memory():
    commands = (
        "uci\n"
        "isready\n"
        "setoption name Depth value 1\n"
        "setoption name Quiescence value false\n"
        "ucinewgame\n"
        "position startpos moves e2e4\n"
        "go depth 1\n"
        "position fen rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq e3 0 1\n"
        "go depth 1\n"
        "quit\n"
    )
    in_stream = io.StringIO(commands)
    out_stream = io.StringIO()
    run_uci_server(input_stream=in_stream, output_stream=out_stream)
    output = out_stream.getvalue()
    assert "id name pychess" in output
    assert "uciok" in output
    assert "readyok" in output
    assert "bestmove" in output


def test_uci_server_as_subprocess_with_client():
    with UCIEngine(sys.executable, args=["-m", "chess.uci"]) as engine:
        engine.set_option("Depth", 1)
        engine.new_game()
        engine.set_position(moves=["e2e4"])
        best = engine.go(depth=1)
        assert isinstance(best, str)
        assert len(best) >= 4

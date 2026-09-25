"""Tests for Terminal User Interface (TUI) layout, rendering, and styling."""

import pytest

from chess import (
    Board,
    ChessGame,
    PieceType,
    notation_to_coords,
)
from chess.tui import (
    format_eval_breakdown,
    format_legal_moves,
    format_pgn,
    format_recent_moves,
    get_captured_pieces,
    get_piece_art_lines,
    render_board_lines,
    render_dashboard,
)


def sq(name: str) -> tuple[int, int]:
    return notation_to_coords(name)


def test_get_captured_pieces():
    # Initial board: no captures
    board_init = Board.from_notation()
    cap_w, cap_b, diff = get_captured_pieces(board_init)
    assert cap_w == ()
    assert cap_b == ()
    assert diff == 0

    # Board with black queen captured (cleared) and white pawn captured (cleared)
    board_mod = board_init.with_piece(sq("d8"), None).with_piece(sq("a2"), None)
    cap_w2, cap_b2, diff2 = get_captured_pieces(board_mod)
    assert len(cap_w2) == 1
    assert cap_w2[0].type == PieceType.QUEEN
    assert len(cap_b2) == 1
    assert cap_b2[0].type == PieceType.PAWN
    # Queen is 900, Pawn is 100 -> diff is +800
    assert diff2 == 800


def test_format_recent_moves():
    assert format_recent_moves([]) == ["  (No moves yet)"]

    game = ChessGame()
    m1 = game.make_move("e4")
    m2 = game.make_move("e5")
    m3 = game.make_move("Nf3")

    lines = format_recent_moves([m1, m2, m3])
    assert len(lines) == 2
    assert "1. e2e4" in lines[0]
    assert "e7e5" in lines[0]
    assert "2. g1f3" in lines[1]


def test_format_legal_moves():
    game = ChessGame()
    all_formatted = format_legal_moves(game)
    assert "Pawn" in all_formatted
    assert "Knight" in all_formatted

    # Filter by specific square
    filtered = format_legal_moves(game, "e2")
    assert "e2e3" in filtered
    assert "e2e4" in filtered

    # Invalid square
    invalid = format_legal_moves(game, "z9")
    assert "Invalid square" in invalid


def test_format_eval_breakdown():
    board = Board.from_notation()
    output = format_eval_breakdown(board)
    assert "Material:" in output
    assert "Positional:" in output
    assert "Total Eval:" in output


def test_format_pgn():
    game = ChessGame()
    m1 = game.make_move("e4")
    m2 = game.make_move("e5")

    pgn = format_pgn([m1, m2], white_name="Alice", black_name="Bob", result="1-0")
    assert '[White "Alice"]' in pgn
    assert '[Black "Bob"]' in pgn
    assert '[Result "1-0"]' in pgn
    assert "1. e2e4 e7e5 1-0" in pgn


def test_render_board_lines_and_options():
    game = ChessGame()
    last_move = game.make_move("e4")

    # Standard ASCII rendering
    lines_ascii = render_board_lines(
        game.board,
        flip=False,
        unicode_pieces=False,
        ansi_colors=False,
        last_move=last_move,
    )
    assert len(lines_ascii) == 10  # 8 ranks + 2 coordinate headers
    assert "8" in lines_ascii[1]
    assert "1" in lines_ascii[8]

    # Flipped orientation (rank 1 at top)
    lines_flip = render_board_lines(
        game.board,
        flip=True,
        unicode_pieces=False,
        ansi_colors=False,
    )
    assert "1" in lines_flip[1]
    assert "8" in lines_flip[8]

    # Unicode + ANSI colors with highlights
    lines_color = render_board_lines(
        game.board,
        flip=False,
        unicode_pieces=True,
        ansi_colors=True,
        last_move=last_move,
        check_square=sq("e1"),
    )
    combined = "\n".join(lines_color)
    assert "\033[" in combined  # ANSI escapes
    assert any(sym in combined for sym in ("♙", "♟", "♖", "♜"))


def test_render_dashboard():
    game = ChessGame()
    m1 = game.make_move("e4")
    dash = render_dashboard(
        game,
        moves_history=[m1],
        flip=False,
        unicode_pieces=True,
        ansi_colors=True,
        white_name="Player1",
        black_name="AI",
    )
    assert "pychess" in dash
    assert "Player1" in dash
    assert "AI" in dash
    assert "Recent Moves:" in dash
    assert "Commands:" in dash


def test_render_board_sizes_and_validation():
    game = ChessGame()
    lines_s1 = render_board_lines(game.board, size=1)
    assert len(lines_s1) == 10

    lines_s2 = render_board_lines(game.board, size=2)
    assert len(lines_s2) == 18

    lines_s3 = render_board_lines(game.board, size=3)
    assert len(lines_s3) == 26

    lines_s4 = render_board_lines(game.board, size=4)
    assert len(lines_s4) == 34

    with pytest.raises(ValueError, match="Board size must be 1, 2, 3, or 4"):
        render_board_lines(game.board, size=0)
    with pytest.raises(ValueError, match="Board size must be 1, 2, 3, or 4"):
        render_board_lines(game.board, size=5)

    dash_large = render_dashboard(game, size=2)
    assert "pychess" in dash_large
    dash_giant = render_dashboard(game, size=4)
    assert "pychess" in dash_giant


def test_scaled_piece_art_dimensions():
    expected_widths = {1: 1, 2: 5, 3: 7, 4: 9}
    for size in (1, 2, 3, 4):
        for pt in (
            PieceType.PAWN,
            PieceType.ROOK,
            PieceType.KNIGHT,
            PieceType.BISHOP,
            PieceType.QUEEN,
            PieceType.KING,
        ):
            for color in ("w", "b"):
                art_lines = get_piece_art_lines(pt, color, size, unicode_pieces=True)
                assert len(art_lines) == size
                if size > 1:
                    for line in art_lines:
                        assert len(line) == expected_widths[size]

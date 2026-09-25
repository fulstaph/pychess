import pytest

from chess import Board, ChessGame, GameStatus, Piece, PieceType, notation_to_coords
from chess.game import to_fen


def sq(name):
    return notation_to_coords(name)


def board_of(*placements):
    board = Board()
    for name, kind, color in placements:
        board = board.with_piece(sq(name), Piece(kind, color))
    return board


def test_documented_opening_and_fools_mate_san():
    game = ChessGame()
    for text in ("e4", "e5", "Nf3", "Nc6"):
        game.make_move(text)
    assert game.board.get(sq("e4")) == Piece(PieceType.PAWN, "w")
    assert game.board.get(sq("e5")) == Piece(PieceType.PAWN, "b")
    assert game.turn == "w" and game.move_num == 3
    game = ChessGame()
    for text in ("f3", "e5", "g4", "Qh4#"):
        game.make_move(text)
    assert game.status == GameStatus.CHECKMATE and game.get_winner() == "b"


def test_capture_and_castling_san_use_generated_moves():
    game = ChessGame()
    for text in ("e4", "d5", "exd5"):
        move = game.make_move(text)
    assert move.captured_piece == Piece(PieceType.PAWN, "b")
    assert game.board.get(sq("d5")) == Piece(PieceType.PAWN, "w")
    board = board_of(
        ("e1", PieceType.KING, "w"),
        ("h1", PieceType.ROOK, "w"),
        ("e8", PieceType.KING, "b"),
    )
    game = ChessGame(board, castling_rights=frozenset("K"))
    assert game.make_move("o-o").special == "castle_k"
    assert game.board.get(sq("f1")) == Piece(PieceType.ROOK, "w")


def test_san_disambiguates_by_file_rank_and_full_square():
    board = board_of(
        ("h1", PieceType.KING, "w"),
        ("a8", PieceType.KING, "b"),
        ("g6", PieceType.KNIGHT, "w"),
        ("c6", PieceType.KNIGHT, "w"),
    )
    game = ChessGame(board)
    with pytest.raises(ValueError):
        game.make_move("Ne7")
    assert game.make_move("Nge7").from_square == sq("g6")
    board = board_of(
        ("h1", PieceType.KING, "w"),
        ("h8", PieceType.KING, "b"),
        ("b1", PieceType.KNIGHT, "w"),
        ("b3", PieceType.KNIGHT, "w"),
        ("f1", PieceType.KNIGHT, "w"),
    )
    game = ChessGame(board)
    assert game.make_move("Nb1d2").from_square == sq("b1")


def test_claimed_check_or_mate_must_be_true_and_failed_parse_is_atomic():
    game = ChessGame()
    original = game.board
    for text in ("e4+", "e4#", "z4", "e2e5", "e7e5", "e2e4=Q", "Qh4#"):
        with pytest.raises(ValueError):
            game.make_move(text)
        assert game.board is original and game.turn == "w"
    game.make_move("e4")
    assert game.turn == "b"


def test_promotion_san_requires_choice():
    board = board_of(
        ("h1", PieceType.KING, "w"),
        ("h8", PieceType.KING, "b"),
        ("a7", PieceType.PAWN, "w"),
    )
    game = ChessGame(board)
    with pytest.raises(ValueError):
        game.make_move("a8")
    assert game.make_move("a8=Q").promotion_to == PieceType.QUEEN


def test_promotion_uci_coordinate_formats():
    board = board_of(
        ("h1", PieceType.KING, "w"),
        ("h8", PieceType.KING, "b"),
        ("a7", PieceType.PAWN, "w"),
    )
    game = ChessGame(board)
    move_lower = game.make_move("a7a8q")
    assert move_lower.promotion_to == PieceType.QUEEN
    assert game.board.get(sq("a8")) == Piece(PieceType.QUEEN, "w")

    game2 = ChessGame(board)
    move_eq_lower = game2.make_move("a7a8=r")
    assert move_eq_lower.promotion_to == PieceType.ROOK

    game3 = ChessGame(board)
    move_upper = game3.make_move("a7a8N")
    assert move_upper.promotion_to == PieceType.KNIGHT


def test_to_fen_initial_and_transitions():
    game = ChessGame()
    assert game.to_fen() == "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
    assert (
        to_fen(game._state)
        == "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
    )

    game.make_move("e4")
    assert (
        game.to_fen() == "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq e3 0 1"
    )

    game.make_move("c5")
    assert (
        game.to_fen() == "rnbqkbnr/pp1ppppp/8/2p5/4P3/8/PPPP1PPP/RNBQKBNR w KQkq c6 0 2"
    )

    # Custom position with partial castling rights and no en passant
    board = board_of(
        ("e1", PieceType.KING, "w"),
        ("h1", PieceType.ROOK, "w"),
        ("e8", PieceType.KING, "b"),
    )
    custom_game = ChessGame(board, castling_rights=frozenset("K"))
    assert custom_game.to_fen() == "4k3/8/8/8/8/8/8/4K2R w K - 0 1"

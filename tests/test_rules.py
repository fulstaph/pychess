import pytest

from chess import Board, ChessGame, GameStatus, Piece, PieceType, notation_to_coords
from chess.attacks import is_in_check


def sq(text):
    return notation_to_coords(text)


def board_of(*placements):
    board = Board()
    for name, kind, color in placements:
        board = board.with_piece(sq(name), Piece(kind, color))
    return board


def perft(game, depth):
    if depth == 0:
        return 1
    total = 0
    for move in game.legal_moves():
        total += perft(game.after(move), depth - 1)
    return total


def test_opening_perft_counts_all_legal_replies():
    game = ChessGame()
    assert perft(game, 1) == 20
    assert perft(game, 2) == 400
    assert perft(game, 3) == 8902


def test_fools_mate_and_terminal_atomicity():
    game = ChessGame()
    for move in ("f2f3", "e7e5", "g2g4", "d8h4"):
        game.make_move(move)
    assert game.status == GameStatus.CHECKMATE and game.get_winner() == "b"
    board = game.board
    with pytest.raises(ValueError):
        game.make_move("a2a3")
    assert game.board is board and game.turn == "w"


def test_pinned_piece_and_en_passant_must_not_expose_king():
    board = board_of(
        ("e1", PieceType.KING, "w"),
        ("a8", PieceType.KING, "b"),
        ("e8", PieceType.ROOK, "b"),
        ("e2", PieceType.ROOK, "w"),
    )
    game = ChessGame(board)
    initial = game.board
    with pytest.raises(ValueError):
        game.make_move("e2d2")
    assert game.board is initial
    board = board_of(
        ("e1", PieceType.KING, "w"),
        ("a8", PieceType.KING, "b"),
        ("e8", PieceType.ROOK, "b"),
        ("e5", PieceType.PAWN, "w"),
        ("d5", PieceType.PAWN, "b"),
    )
    game = ChessGame(board, en_passant=sq("d6"))
    assert all(move.special != "en_passant" for move in game.legal_moves())
    with pytest.raises(ValueError):
        game.make_move("e5d6")


def test_black_castling_and_rook_capture_rights():
    board = board_of(
        ("e1", PieceType.KING, "w"),
        ("a1", PieceType.ROOK, "w"),
        ("h1", PieceType.ROOK, "w"),
        ("e8", PieceType.KING, "b"),
        ("a8", PieceType.ROOK, "b"),
        ("h8", PieceType.ROOK, "b"),
    )
    for destination, rook_destination in (("g8", "f8"), ("c8", "d8")):
        game = ChessGame(board, turn="b", castling_rights=frozenset("KQkq"))
        move = next(
            move
            for move in game.legal_moves()
            if move.piece.type == PieceType.KING and move.to_square == sq(destination)
        )
        game.make_move(move)
        assert game.board.get(sq(rook_destination)) == Piece(PieceType.ROOK, "b")
        assert game.move_num == 2 and game.castles("w", "queenside")
        assert not game.castles("b", "kingside") and not game.castles("b", "queenside")
    board = board.with_piece(sq("b7"), Piece(PieceType.BISHOP, "w"))
    game = ChessGame(board, castling_rights=frozenset("KQkq"))
    game.make_move("b7a8")
    assert not game.castles("b", "queenside") and game.castles("b", "kingside")


def test_castling_through_attack_is_forbidden_and_missing_king_is_invalid():
    board = board_of(
        ("e1", PieceType.KING, "w"),
        ("h1", PieceType.ROOK, "w"),
        ("a8", PieceType.KING, "b"),
        ("f8", PieceType.ROOK, "b"),
    )
    game = ChessGame(board, castling_rights=frozenset("K"))
    assert all(move.special != "castle_k" for move in game.legal_moves())
    with pytest.raises(ValueError):
        is_in_check(Board(), "w")


def test_black_underpromotion_and_invalid_promotion_atomicity():
    board = board_of(
        ("h1", PieceType.KING, "w"),
        ("h8", PieceType.KING, "b"),
        ("a2", PieceType.PAWN, "b"),
    )
    game = ChessGame(board, turn="b")
    with pytest.raises(ValueError):
        game.make_move("a2a1=X")
    assert game.board is board and game.turn == "b"
    game.make_move("a2a1=N")
    assert game.board.get(sq("a1")) == Piece(PieceType.KNIGHT, "b")

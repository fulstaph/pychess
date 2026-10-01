import pytest

from chess import (
    Board,
    ChessGame,
    GameStatus,
    Move,
    Piece,
    PieceType,
    notation_to_coords,
)
from chess.engine import move_notation


def sq(text):
    return notation_to_coords(text)


def test_move_is_pure_and_checks_capture_metadata():
    board = Board().with_piece(sq("e2"), Piece(PieceType.PAWN, "w"))
    move = Move(Piece(PieceType.PAWN, "w"), sq("e2"), sq("e4"))
    result = move.execute(board)
    assert board.get(sq("e2")) == move.piece
    assert result.get(sq("e4")) == move.piece
    assert result.get(sq("e2")) is None
    with pytest.raises(ValueError):
        move.execute(result)
    with pytest.raises(ValueError):
        Move(move.piece, sq("e2"), sq("e4"), Piece(PieceType.ROOK, "b")).execute(board)


def test_move_rejects_invalid_special_metadata_without_touching_board():
    board = Board().with_piece(sq("a7"), Piece(PieceType.PAWN, "w"))
    with pytest.raises(ValueError):
        Move(Piece(PieceType.PAWN, "w"), sq("a7"), sq("a8")).execute(board)
    with pytest.raises(ValueError):
        Move(Piece(PieceType.PAWN, "w"), sq("a7"), sq("a8"), special="unknown").execute(
            board
        )
    assert board.get(sq("a7")) == Piece(PieceType.PAWN, "w")


def test_initial_legal_moves_and_turn_transition():
    game = ChessGame()
    assert len(game.legal_moves()) == 20
    move = next(
        move
        for move in game.legal_moves()
        if move.from_square == sq("e2") and move.to_square == sq("e4")
    )
    previous = game.board
    assert game.make_move(move) == move
    assert game.board is not previous and previous.get(sq("e2")) == move.piece
    assert game.board.get(sq("e4")) == move.piece
    assert game.turn == "b" and game.move_num == 1 and game.status == GameStatus.ACTIVE
    assert game.get_turn() == "b" and game.get_board() is game.board
    assert len(game.legal_moves()) == 20


def test_invalid_move_does_not_consume_turn_or_change_board():
    game = ChessGame()
    initial = (
        game.board,
        game.turn,
        game.status,
        game.move_num,
        game.castles("w", "kingside"),
    )
    with pytest.raises(ValueError):
        game.make_move(Move(Piece(PieceType.PAWN, "w"), sq("e2"), sq("e5")))
    assert (
        game.board,
        game.turn,
        game.status,
        game.move_num,
        game.castles("w", "kingside"),
    ) == initial
    with pytest.raises(ValueError):
        ChessGame(Board())


def test_game_status_queries_and_init_validation():
    game = ChessGame()
    assert not game.is_check()
    assert not game.is_checkmate()
    assert not game.is_stalemate()

    with pytest.raises(ValueError, match="Invalid player or castling side"):
        game.castles("x", "kingside")
    with pytest.raises(ValueError, match="Invalid player or castling side"):
        game.castles("w", "invalid_side")
    with pytest.raises(ValueError, match="Invalid castling rights"):
        ChessGame(castling_rights="invalid")
    with pytest.raises(ValueError, match="Invalid castling rights"):
        ChessGame(castling_rights=frozenset("XYZ"))
    with pytest.raises(ValueError, match="Invalid en passant target"):
        ChessGame(en_passant=(0, 0))
    with pytest.raises(ValueError, match="Illegal move"):
        game.make_move(123)
    with pytest.raises(ValueError, match="Illegal move"):
        game.after(123)


def test_king_safety_memo_is_reset_by_make_move_and_after():
    game = ChessGame()
    for move in ("e4", "f6", "Qh5+"):
        # Warm the per-position memo, then advance: the child must not reuse it.
        game.is_check()
        game.legal_moves()
        game.make_move(move)
    assert game.is_check() is True
    assert [move_notation(m) for m in game.legal_moves()] == ["g7g6"]
    game.make_move("g6")
    assert game.is_check() is False

    parent = ChessGame()
    assert parent.is_check() is False and len(parent.legal_moves()) == 20
    child = parent.after("e4")
    assert child.is_check() is False and len(child.legal_moves()) == 20
    checked = child.after("f6").after("Qh5+")
    assert checked.is_check() is True

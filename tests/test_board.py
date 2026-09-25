from dataclasses import FrozenInstanceError

import pytest

from chess import (
    Board,
    Piece,
    PieceType,
    PlayerColor,
    notation_to_coords,
    square_notation,
)


def test_starting_board_has_both_complete_armies():
    board = Board.from_notation()
    assert sum(piece is not None for row in board.squares for piece in row) == 32
    assert board.get(notation_to_coords("e1")) == Piece(PieceType.KING, "w")
    assert board.get(notation_to_coords("e8")) == Piece(PieceType.KING, "b")
    assert all(board.get((6, col)) == Piece(PieceType.PAWN, "w") for col in range(8))
    assert all(board.get((1, col)) == Piece(PieceType.PAWN, "b") for col in range(8))


def test_board_replacement_does_not_change_original_or_expose_mutable_rows():
    board = Board()
    piece = Piece(PieceType.KING, "w")
    changed = board.with_piece((7, 4), piece)
    assert board.get((7, 4)) is None
    assert changed.get((7, 4)) == piece
    with pytest.raises(TypeError):
        changed.squares[7][4] = None
    with pytest.raises(FrozenInstanceError):
        piece.color = "b"


def test_invalid_piece_and_coordinates_are_rejected():
    with pytest.raises(ValueError):
        Piece("p", "w")
    with pytest.raises(ValueError):
        Piece(PieceType.PAWN, "x")
    for square in ((-1, 0), (8, 2), (1, 8), (True, 0)):
        with pytest.raises(ValueError):
            Board().get(square)
    for text in ("z4", "a0", "a9", "a1x", "a٤"):
        with pytest.raises(ValueError):
            notation_to_coords(text)
    for row, col in ((-1, 0), (8, 0), (0, 8)):
        with pytest.raises(ValueError):
            square_notation(row, col)


def test_player_colors_and_square_conversions():
    assert PlayerColor.WHITE.value == "w"
    assert PlayerColor.WHITE.opposite() is PlayerColor.BLACK
    assert notation_to_coords("h8") == (0, 7)
    assert square_notation(7, 0) == "a1"


def test_piece_methods_and_board_validation_edges():
    piece = Piece(PieceType.KNIGHT, "b")
    assert str(piece) == "N"
    assert piece.is_black()
    assert not piece.is_white()
    assert PlayerColor.BLACK.is_black()
    assert not PlayerColor.BLACK.is_white()
    assert PlayerColor.BLACK.opposite() is PlayerColor.WHITE

    board = Board()
    assert board.is_empty((0, 0))
    with pytest.raises(ValueError, match="eight by eight"):
        Board(((None,),))
    with pytest.raises(ValueError, match="eight by eight"):
        Board(tuple((None,) * 7 for _ in range(8)))
    with pytest.raises(ValueError, match="pieces or None"):
        Board(tuple((123,) * 8 for _ in range(8)))
    with pytest.raises(ValueError, match="pieces or None"):
        board._updated({(0, 0): "invalid"})

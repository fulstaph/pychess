"""
Helper functions and classes for coordinate conversion.
"""

from enum import StrEnum


class PlayerColor(StrEnum):
    """Player colors."""

    WHITE = "w"
    BLACK = "b"

    def is_white(self) -> bool:
        return self == PlayerColor.WHITE

    def is_black(self) -> bool:
        return self == PlayerColor.BLACK

    def opposite(self) -> PlayerColor:
        return PlayerColor.BLACK if self == PlayerColor.WHITE else PlayerColor.WHITE


def notation_to_coords(square: str) -> tuple[int, int]:
    """
    Convert algebraic notation square to array coordinates.

    Args:
        square: Algebraic notation (e.g., 'e4', 'h8')

    Returns:
        (row, col) coordinates where row 0 = rank 8, col 0 = file a

    Examples:
        >>> notation_to_coords('a1')
        (7, 0)

        >>> notation_to_coords('e4')
        (4, 4)

        >>> notation_to_coords('h8')
        (0, 7)
    """
    if (
        not isinstance(square, str)
        or len(square) != 2
        or square[0] not in "abcdefgh"
        or square[1] not in "12345678"
    ):
        raise ValueError(f"Invalid square notation: {square!r}")
    return 8 - int(square[1]), ord(square[0]) - ord("a")


def square_notation(row: int, col: int) -> str:
    """
    Convert array coordinates to algebraic notation.

    Args:
        row: Row index (0-7), 0 = rank 8
        col: Column index (0-7), 0 = file a

    Returns:
        Algebraic notation (e.g., 'e4', 'a1')
    """
    if (
        type(row) is not int
        or type(col) is not int
        or not 0 <= row < 8
        or not 0 <= col < 8
    ):
        raise ValueError(f"Invalid square coordinates: {(row, col)!r}")
    return f"{'abcdefgh'[col]}{8 - row}"

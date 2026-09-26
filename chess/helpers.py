"""Coordinate conversion and shared color vocabulary.

This module bridges the two coordinate systems the package uses:

- **Array coordinates** ``(row, col)`` where row 0 = rank 8 and
  col 0 = file ``a`` (matching how ``Board`` stores its nested tuples).
- **Algebraic notation** such as ``"e4"``, used in FEN, UCI, and display.

Mapping table (corners and center):

==============  ============
Notation        (row, col)
==============  ============
``a1``          ``(7, 0)``
``a8``          ``(0, 0)``
``e4``          ``(4, 4)``
``h1``          ``(7, 7)``
``h8``          ``(0, 7)``
==============  ============

All functions are pure: no logging, no I/O, no mutation.
"""

from enum import StrEnum


class PlayerColor(StrEnum):
    """Player colors, keyed by the FEN/UCI single-letter value.

    Value semantics: because this is a ``StrEnum``, each member *is* a
    string, so ``PlayerColor.WHITE == "w"`` is ``True``.  This lets code
    accept plain ``"w"``/``"b"`` literals (as FEN/UCI do) interchangeably
    with the enum without any conversion, while still giving a readable
    name in logs and error messages.
    """

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
    # Validate the exact 2-character form up front (file letter then rank
    # digit) so a bad token raises here with a clear message instead of
    # surfacing later as an IndexError or a wrong square.
    if (
        not isinstance(square, str)
        or len(square) != 2
        or square[0] not in "abcdefgh"
        or square[1] not in "12345678"
    ):
        raise ValueError(f"Invalid square notation: {square!r}")
    # Rank '8' is row 0 and rank '1' is row 7, hence the 8 - rank
    # inversion; the file letter is an offset from 'a' via its code point.
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
    # ``type(x) is not int`` (not isinstance) rejects bools: True/False
    # are ints in Python and would silently map to col 1 / rank 8, hiding
    # a caller bug.  Range checks keep the f-string below index-safe.
    if (
        type(row) is not int
        or type(col) is not int
        or not 0 <= row < 8
        or not 0 <= col < 8
    ):
        raise ValueError(f"Invalid square coordinates: {(row, col)!r}")
    # Inverse of notation_to_coords: file letter by column offset, rank
    # digit by 8 - row (row 0 = rank 8).
    return f"{'abcdefgh'[col]}{8 - row}"

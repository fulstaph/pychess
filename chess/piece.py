"""Chess pieces and an immutable eight-by-eight board.

Squares are (row, column): a8 is (0, 0), a1 is (7, 0).
"""

from collections.abc import Mapping
from dataclasses import dataclass
from enum import StrEnum
from typing import Literal

Color = Literal["w", "b"]
Square = tuple[int, int]


class PieceType(StrEnum):
    PAWN = "p"
    ROOK = "r"
    KNIGHT = "n"
    BISHOP = "b"
    QUEEN = "q"
    KING = "k"


@dataclass(frozen=True, slots=True)
class Piece:
    type: PieceType
    color: Color

    def __post_init__(self) -> None:
        if not isinstance(self.type, PieceType) or self.color not in ("w", "b"):
            raise ValueError("A piece needs a PieceType and color 'w' or 'b'")

    def is_white(self) -> bool:
        return self.color == "w"

    def is_black(self) -> bool:
        return self.color == "b"

    def __str__(self) -> str:
        return self.type.value.upper()


class Board:
    """Persistent board; every move or fixture update creates a new value."""

    __slots__ = ("_squares",)
    BOARD_SIZE = 8

    def __init__(
        self, squares: tuple[tuple[Piece | None, ...], ...] | None = None
    ) -> None:
        if squares is None:
            squares = tuple((None,) * 8 for _ in range(8))
        if len(squares) != 8 or any(len(row) != 8 for row in squares):
            raise ValueError("Board must be eight by eight")
        if any(
            piece is not None and not isinstance(piece, Piece)
            for row in squares
            for piece in row
        ):
            raise ValueError("Board squares must contain pieces or None")
        self._squares = tuple(tuple(row) for row in squares)

    @property
    def squares(self) -> tuple[tuple[Piece | None, ...], ...]:
        return self._squares

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Board):
            return NotImplemented
        return self._squares == other._squares

    def __hash__(self) -> int:
        return hash(self._squares)

    @staticmethod
    def _validate_square(square: Square) -> None:
        if (
            not isinstance(square, tuple)
            or len(square) != 2
            or any(type(index) is not int or not 0 <= index < 8 for index in square)
        ):
            raise ValueError(f"Invalid square: {square!r}")

    def get(self, square: Square) -> Piece | None:
        self._validate_square(square)
        return self._squares[square[0]][square[1]]

    def is_empty(self, square: Square) -> bool:
        return self.get(square) is None

    def with_piece(self, square: Square, piece: Piece | None) -> Board:
        return self._updated({square: piece})

    def _updated(self, changes: Mapping[Square, Piece | None]) -> Board:
        for square, piece in changes.items():
            self._validate_square(square)
            if piece is not None and not isinstance(piece, Piece):
                raise ValueError("Board squares must contain pieces or None")
        rows = list(self._squares)
        for row in {square[0] for square in changes}:
            updated_row = list(rows[row])
            for (changed_row, col), piece in changes.items():
                if changed_row == row:
                    updated_row[col] = piece
            rows[row] = tuple(updated_row)
        return Board(tuple(rows))

    @classmethod
    def from_notation(cls) -> Board:
        back_rank = (
            PieceType.ROOK,
            PieceType.KNIGHT,
            PieceType.BISHOP,
            PieceType.QUEEN,
            PieceType.KING,
            PieceType.BISHOP,
            PieceType.KNIGHT,
            PieceType.ROOK,
        )
        changes: dict[Square, Piece] = {}
        placements: tuple[tuple[Color, int, int], ...] = (
            ("w", 7, 6),
            ("b", 0, 1),
        )
        for color, home, pawns in placements:
            for col, piece_type in enumerate(back_rank):
                changes[(home, col)] = Piece(piece_type, color)
                changes[(pawns, col)] = Piece(PieceType.PAWN, color)
        return cls()._updated(changes)

    @classmethod
    def from_fen(cls, fen: str) -> Board:
        """Create a Board from a FEN string or board placement portion."""
        board_part = fen.strip().split()[0] if " " in fen.strip() else fen.strip()
        ranks = board_part.split("/")
        if len(ranks) != 8:
            raise ValueError(
                f"FEN must contain 8 ranks separated by '/', got {len(ranks)}"
            )
        rows: list[list[Piece | None]] = []
        for rank_str in ranks:
            row: list[Piece | None] = []
            for ch in rank_str:
                if ch.isdigit():
                    count = int(ch)
                    if count < 1 or count > 8:
                        raise ValueError(f"Invalid empty square count in FEN: {ch}")
                    row.extend([None] * count)
                else:
                    try:
                        pt = PieceType(ch.lower())
                    except ValueError:
                        raise ValueError(
                            f"Invalid piece character in FEN: {ch}"
                        ) from None
                    color: Color = "w" if ch.isupper() else "b"
                    row.append(Piece(pt, color))
            if len(row) != 8:
                raise ValueError(f"Rank in FEN does not contain 8 squares: {rank_str}")
            rows.append(row)
        return cls(tuple(tuple(r) for r in rows))

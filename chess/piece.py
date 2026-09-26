"""Chess pieces and an immutable eight-by-eight board.

Coordinate system
-----------------
Squares are ``(row, col)`` tuples with **row 0 = rank 8** and col 0 = file
``a``:

- ``a8 -> (0, 0)`` (black's back rank, top-left from White's view)
- ``h1 -> (7, 7)``
- ``e4 -> (4, 4)``

The row axis therefore grows downward toward White's rank 1, which is why
White pieces "move up" by decrementing the row.

Immutability model
------------------
``Board`` is a persistent value: the underlying storage is a tuple of
tuples (one inner tuple per rank), and every mutation helper
(``with_piece``, ``_updated``) rebuilds only the affected rows and returns
a *new* ``Board``.  Copying a full row per change is deliberate: rows are
8 cells, so the copy is cheap, and value semantics make boards hashable,
comparable, and safe to share across the search engine without locking.

Validation policy
-----------------
All public entry points (``Board.__init__``, ``get``, ``_updated``, the
``from_fen``/``from_notation`` classmethods) validate eagerly and raise
``ValueError`` on malformed input.  Invalid *squares* and *pieces* are
programmer/fixture errors, not normal flow, so they raise immediately
rather than poisoning a partially built board.
"""

from collections.abc import Mapping
from dataclasses import dataclass
from enum import StrEnum
from typing import Literal

from .log import get_logger

logger = get_logger("piece")

Color = Literal["w", "b"]
Square = tuple[int, int]


class PieceType(StrEnum):
    """FEN single-letter piece codes, reused everywhere as the type value."""

    PAWN = "p"
    ROOK = "r"
    KNIGHT = "n"
    BISHOP = "b"
    QUEEN = "q"
    KING = "k"


@dataclass(frozen=True, slots=True)
class Piece:
    """An immutable, hashable piece: a ``PieceType`` plus a color.

    ``StrEnum``-based types make a piece comparable to plain FEN
    characters in tests and protocol code.
    """

    type: PieceType
    color: Color

    def __post_init__(self) -> None:
        # Guard the constructor so no code path can smuggle an invalid
        # piece into a board; PieceType already covers the "type" side.
        if not isinstance(self.type, PieceType) or self.color not in ("w", "b"):
            raise ValueError("A piece needs a PieceType and color 'w' or 'b'")

    def is_white(self) -> bool:
        return self.color == "w"

    def is_black(self) -> bool:
        return self.color == "b"

    def __str__(self) -> str:
        # Upper-case = FEN "display" form; color is lost on purpose, it is
        # carried separately by the ``color`` field.
        return self.type.value.upper()


class Board:
    """Persistent board; every move or fixture update creates a new value."""

    __slots__ = ("_squares",)
    BOARD_SIZE = 8

    def __init__(
        self, squares: tuple[tuple[Piece | None, ...], ...] | None = None
    ) -> None:
        if squares is None:
            # Empty board: 8 ranks of 8 empty squares.  The double
            # multiplication is safe because cells hold only immutable
            # (Piece | None) values that are never mutated in place.
            squares = tuple((None,) * 8 for _ in range(8))
        if len(squares) != 8 or any(len(row) != 8 for row in squares):
            raise ValueError("Board must be eight by eight")
        if any(
            piece is not None and not isinstance(piece, Piece)
            for row in squares
            for piece in row
        ):
            raise ValueError("Board squares must contain pieces or None")
        # Normalize to immutable nested tuples so later copies never see a
        # caller-mutable list alias.
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
        # ``type(index) is not int`` rejects bools (True/False are ints in
        # Python) and subclasses; a bool square index would silently land
        # on row/col 0 or 1 and hide a real bug.
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
        # Validate every change up front so a bad entry cannot leave a
        # half-applied board behind.
        for square, piece in changes.items():
            self._validate_square(square)
            if piece is not None and not isinstance(piece, Piece):
                raise ValueError("Board squares must contain pieces or None")
        # Copy-on-write: only rows touched by `changes` are copied; the
        # other rows are shared with the previous board version for free.
        rows = list(self._squares)
        for row in {square[0] for square in changes}:
            updated_row = list(rows[row])
            for (changed_row, col), piece in changes.items():
                if changed_row == row:
                    updated_row[col] = piece
            rows[row] = tuple(updated_row)
        # Rebuilding through __init__ re-validates; with the checks above
        # this is a no-cost guarantee, not a second defense layer.
        return Board(tuple(rows))

    @classmethod
    def from_notation(cls) -> Board:
        """Standard starting position (the "notation" initial board).

        Back ranks are built once and mirrored: white home rank is row 7
        (rank 1), black home rank is row 0 (rank 8).  The pawn ranks sit
        one row in front of each home rank: row 6 for white, row 1 for
        black.  Both ranks are laid out as (color, home_row, pawn_row).
        """
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
        # Start from the empty board and apply all placements in one
        # copy-on-write pass instead of 32 individual updates.
        return cls()._updated(changes)

    @classmethod
    def from_fen(cls, fen: str) -> Board:
        """Create a Board from a FEN string or board placement portion.

        Accepts either a full FEN (only the first whitespace-separated
        token is used) or a bare placement string.  FEN ranks are written
        top-down: the first rank string is rank 8 (row 0), the last is
        rank 1 (row 7), so the parsed rows are stored in the same order
        with no flipping.
        """
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
                    # '1'-'8' = that many consecutive empty squares.
                    # FEN only ever encodes runs of at most 8, so anything
                    # else is a malformed placement.
                    count = int(ch)
                    if count < 1 or count > 8:
                        raise ValueError(f"Invalid empty square count in FEN: {ch}")
                    row.extend([None] * count)
                else:
                    # Uppercase letter = white piece, lowercase = black;
                    # the letter (case-folded) is the FEN piece code.
                    try:
                        pt = PieceType(ch.lower())
                    except ValueError:
                        raise ValueError(
                            f"Invalid piece character in FEN: {ch}"
                        ) from None
                    color: Color = "w" if ch.isupper() else "b"
                    row.append(Piece(pt, color))
            # The per-rank length check catches FENs with missing digits
            # or extra pieces that would otherwise shift every later
            # square in the rank.
            if len(row) != 8:
                raise ValueError(f"Rank in FEN does not contain 8 squares: {rank_str}")
            rows.append(row)
        # Parsed rows are passed through __init__, which re-validates
        # shape and cell contents.
        return cls(tuple(tuple(r) for r in rows))

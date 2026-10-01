"""A generated legal move and its immutable board transition.

``Move`` is a *value* describing a transition, not a rule engine.  Full
legality (check, pin, double-step availability, castling rights, etc.) is
decided by ``ChessGame`` when it generates moves; ``execute`` only
enforces the *structural* preconditions that must hold for the stored
value to apply to the given board (the right piece is on the source
square, the capture matches, promotion geometry is correct, ...).  If a
caller hands ``execute`` a move that was never generated against that
board, it raises ``ValueError`` — that is a programmer error, not user
input, which is why the failures are debug-logged rather than treated
as recoverable state.

``Move`` is a frozen dataclass: moves are produced in bulk by the search
and shared across tree nodes, so they must be immutable and hashable.
``slots`` keeps the per-move footprint small at engine scale.
"""

from dataclasses import dataclass
from typing import Literal

from .log import get_logger
from .piece import Board, Piece, PieceType, Square

logger = get_logger("move")

type MoveSpecial = Literal["none", "castle_k", "castle_q", "en_passant", "promotion"]


@dataclass(frozen=True, slots=True)
class Move:
    piece: Piece
    from_square: Square
    to_square: Square
    captured_piece: Piece | None = None
    special: MoveSpecial = "none"
    promotion_to: PieceType | None = None

    def is_capture(self) -> bool:
        return self.captured_piece is not None

    def execute(self, board: Board) -> Board:
        """Apply a generated legal move; full legality belongs to ChessGame."""
        if self.special not in (
            "none",
            "castle_k",
            "castle_q",
            "en_passant",
            "promotion",
        ):
            logger.debug("execute failed: invalid special %r", self.special)
            raise ValueError("Invalid special move")
        source = board._get_fast(self.from_square)
        target = board._get_fast(self.to_square)
        if source != self.piece or self.from_square == self.to_square:
            logger.debug(
                "execute failed: source mismatch for %s -> %s",
                self.from_square,
                self.to_square,
            )
            raise ValueError("Move source does not match the board")
        # A pawn landing on the last rank must have been generated with
        # special="promotion"; promotion_to itself is only mandatory for
        # that special (checked below).
        last_rank = 0 if self.piece.color == "w" else 7
        if (
            self.piece.type == PieceType.PAWN
            and self.to_square[0] == last_rank
            and self.special != "promotion"
        ):
            logger.debug(
                "execute failed: pawn on last rank %d without promotion",
                last_rank,
            )
            raise ValueError("Pawn must promote on the last rank")
        if self.special == "en_passant":
            # En passant geometry: the captured pawn sits on the *from*
            # rank of the moving pawn but on the *destination* file, i.e.
            # (from_row, to_col).  The landing square itself is empty.
            capture_square = (self.from_square[0], self.to_square[1])
            captured = board._get_fast(capture_square)
            if (
                self.piece.type != PieceType.PAWN
                or target is not None
                or captured != self.captured_piece
                or captured is None
                or captured.type != PieceType.PAWN
                or captured.color == self.piece.color
            ):
                logger.debug(
                    "execute failed: invalid en passant capture at %s",
                    capture_square,
                )
                raise ValueError("Invalid en passant capture")
        else:
            capture_square = self.to_square
            if target != self.captured_piece or (
                target and target.color == self.piece.color
            ):
                logger.debug("execute failed: capture mismatch at %s", self.to_square)
                raise ValueError("Move capture does not match the board")
        # Start from clearing the source square; each special case adds
        # its extra removals/placements before the promoted/normal piece
        # lands on the destination.
        changes: dict[Square, Piece | None] = {self.from_square: None}
        if self.special in ("castle_k", "castle_q"):
            # Kingside: king e->g (col 4->6), rook h->f (col 7->5).
            # Queenside: king e->c (col 4->2), rook a->d (col 0->3).
            if (
                self.piece.type != PieceType.KING
                or self.captured_piece is not None
                or self.from_square != (7 if self.piece.color == "w" else 0, 4)
                or self.to_square
                != (self.from_square[0], 6 if self.special == "castle_k" else 2)
            ):
                logger.debug("execute failed: invalid castling move")
                raise ValueError("Invalid castling move")
            row = self.from_square[0]
            rook_from = (row, 7 if self.special == "castle_k" else 0)
            rook_to = (row, 5 if self.special == "castle_k" else 3)
            rook = board._get_fast(rook_from)
            if (
                rook != Piece(PieceType.ROOK, self.piece.color)
                or board._get_fast(rook_to) is not None
            ):
                logger.debug("execute failed: castling rook unavailable")
                raise ValueError("Castling rook is unavailable")
            changes[rook_from] = None
            changes[rook_to] = rook
        if self.special == "en_passant":
            # The captured pawn is off the board at a *different* square
            # than the destination, so it needs an explicit removal.
            changes[capture_square] = None
        if self.special == "promotion":
            # promotion_to was validated as part of the "last rank" rule
            # above; only the four legal promotion types are accepted and
            # the pawn must actually be arriving on the last rank.
            if (
                self.piece.type != PieceType.PAWN
                or self.to_square[0] != (0 if self.piece.color == "w" else 7)
                or self.promotion_to
                not in (
                    PieceType.QUEEN,
                    PieceType.ROOK,
                    PieceType.BISHOP,
                    PieceType.KNIGHT,
                )
            ):
                logger.debug("execute failed: invalid promotion")
                raise ValueError("Invalid promotion")
            promoted = Piece(self.promotion_to, self.piece.color)
        else:
            if self.promotion_to is not None:
                logger.debug("execute failed: unexpected promotion choice")
                raise ValueError("Unexpected promotion choice")
            promoted = self.piece
        if len(changes) == 1 and (target is None or target.type != PieceType.KING):
            # Plain move or promotion: only source and destination change.
            return board._moved(self.from_square, self.to_square, promoted)
        changes[self.to_square] = promoted
        # Coordinate/piece invariants have been checked above; skip the
        # duplicate validation pass while retaining copy-on-write rows.
        return board._updated_validated(changes)

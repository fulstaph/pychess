"""A generated legal move and its immutable board transition."""

from dataclasses import dataclass
from typing import Literal

from .piece import Board, Piece, PieceType, Square

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
            raise ValueError("Invalid special move")
        source = board.get(self.from_square)
        target = board.get(self.to_square)
        if source != self.piece or self.from_square == self.to_square:
            raise ValueError("Move source does not match the board")
        last_rank = 0 if self.piece.color == "w" else 7
        if (
            self.piece.type == PieceType.PAWN
            and self.to_square[0] == last_rank
            and self.special != "promotion"
        ):
            raise ValueError("Pawn must promote on the last rank")
        if self.special == "en_passant":
            capture_square = (self.from_square[0], self.to_square[1])
            captured = board.get(capture_square)
            if (
                self.piece.type != PieceType.PAWN
                or target is not None
                or captured != self.captured_piece
                or captured is None
                or captured.type != PieceType.PAWN
                or captured.color == self.piece.color
            ):
                raise ValueError("Invalid en passant capture")
        else:
            capture_square = self.to_square
            if target != self.captured_piece or (
                target and target.color == self.piece.color
            ):
                raise ValueError("Move capture does not match the board")
        changes: dict[Square, Piece | None] = {self.from_square: None}
        if self.special in ("castle_k", "castle_q"):
            if (
                self.piece.type != PieceType.KING
                or self.captured_piece is not None
                or self.from_square != (7 if self.piece.color == "w" else 0, 4)
                or self.to_square
                != (self.from_square[0], 6 if self.special == "castle_k" else 2)
            ):
                raise ValueError("Invalid castling move")
            row = self.from_square[0]
            rook_from = (row, 7 if self.special == "castle_k" else 0)
            rook_to = (row, 5 if self.special == "castle_k" else 3)
            rook = board.get(rook_from)
            if (
                rook != Piece(PieceType.ROOK, self.piece.color)
                or board.get(rook_to) is not None
            ):
                raise ValueError("Castling rook is unavailable")
            changes[rook_from] = None
            changes[rook_to] = rook
        if self.special == "en_passant":
            changes[capture_square] = None
        if self.special == "promotion":
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
                raise ValueError("Invalid promotion")
            promoted = Piece(self.promotion_to, self.piece.color)
        else:
            if self.promotion_to is not None:
                raise ValueError("Unexpected promotion choice")
            promoted = self.piece
        changes[self.to_square] = promoted
        return board._updated(changes)

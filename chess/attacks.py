"""Attack detection independent of move legality or turn history.

Pure geometry: these functions answer "is square X attacked by color C
on this board position?" with no knowledge of whose turn it is, the move
counter, or castling rights.  That makes them safe to call from any
context (check detection, castling-through-check tests, move
generation) and trivially correct: an attack is a property of the
position, not of the game state.

Sliding pieces are handled by raycasting: walk one step at a time along
a direction and stop at the *first* piece encountered.  If that piece is
an attacker of the right type, the square is attacked; if it is anything
else (any color), it blocks the ray — a rook cannot attack through a
friend or a foe.
"""

from .log import get_logger
from .piece import Board, Color, PieceType, Square

logger = get_logger("attacks")

# (row, col) deltas.  Row grows downward (row 0 = rank 8), so a *white*
# pawn moving "up" the board decreases its row.
KNIGHT_STEPS = ((-2, -1), (-2, 1), (-1, -2), (-1, 2), (1, -2), (1, 2), (2, -1), (2, 1))
KING_STEPS = tuple((dr, dc) for dr in (-1, 0, 1) for dc in (-1, 0, 1) if dr or dc)
ORTHOGONAL = ((-1, 0), (1, 0), (0, -1), (0, 1))
DIAGONAL = ((-1, -1), (-1, 1), (1, -1), (1, 1))


def on_board(row: int, col: int) -> bool:
    return 0 <= row < 8 and 0 <= col < 8


type SquareTable = tuple[tuple[tuple[Square, ...], ...], ...]
type RayTable = tuple[tuple[tuple[tuple[Square, ...], ...], ...], ...]


def _step_targets(steps: tuple[tuple[int, int], ...]) -> SquareTable:
    return tuple(
        tuple(
            tuple(
                (row + dr, col + dc) for dr, dc in steps if on_board(row + dr, col + dc)
            )
            for col in range(8)
        )
        for row in range(8)
    )


def _rays(directions: tuple[tuple[int, int], ...]) -> RayTable:
    return tuple(
        tuple(
            tuple(
                tuple(
                    (row + k * dr, col + k * dc)
                    for k in range(1, 8)
                    if on_board(row + k * dr, col + k * dc)
                )
                for dr, dc in directions
            )
            for col in range(8)
        )
        for row in range(8)
    )


# Precomputed once at import; entries are indexed ``[row][col]``.  Targets keep
# the order of ``KNIGHT_STEPS`` / ``KING_STEPS``; rays keep the order of
# ``ORTHOGONAL`` / ``DIAGONAL`` and list squares nearest-first to the edge.
KNIGHT_TARGETS = _step_targets(KNIGHT_STEPS)
KING_TARGETS = _step_targets(KING_STEPS)
ORTHOGONAL_RAYS = _rays(ORTHOGONAL)
DIAGONAL_RAYS = _rays(DIAGONAL)
# Squares from which a pawn of the keyed color attacks ``[row][col]``.
PAWN_ATTACKERS: dict[Color, SquareTable] = {
    "w": _step_targets(((1, -1), (1, 1))),
    "b": _step_targets(((-1, -1), (-1, 1))),
}


def find_king(board: Board, color: Color) -> Square:
    """Validated king lookup — scans only when the board has no cache hit.

    ``ChessGame.__init__`` calls this to enforce exactly-one-king; the
    hot-path ``is_in_check`` uses the cached ``Board.king_square``
    directly instead.
    """
    cached = board.king_square(color)
    if cached is not None:
        # Verify uniqueness only when the full-scan contract matters
        # (game construction).  The cache stores the *last* king found,
        # so a second king on the board would slip by without this loop.
        count = sum(
            1
            for row in board.squares
            for piece in row
            if piece is not None
            and piece.type == PieceType.KING
            and piece.color == color
        )
        if count != 1:
            raise ValueError(f"Expected exactly one {color} king")
        return cached
    # No king on the board at all.
    raise ValueError(f"Expected exactly one {color} king")


def is_square_attacked(square: Square, attacker_color: Color, board: Board) -> bool:
    # Hot path: cache the raw tuple grid and use direct indexing to
    # avoid Board.get → _validate_square overhead on every ray step.
    # Callers (is_in_check, castling check) always pass valid squares.
    sq = board.squares  # single property access
    row, col = square
    # Pawn attackers sit on the row *behind* the target from their own
    # perspective (row + 1 for white, row - 1 for black), one column either side.
    for r, c in PAWN_ATTACKERS[attacker_color][row][col]:
        piece = sq[r][c]
        if (
            piece is not None
            and piece.color == attacker_color
            and piece.type is PieceType.PAWN
        ):
            return True
    # Knights and kings have fixed, non-sliding target sets: check each
    # precomputed target directly; no ray walk, no blocking.
    for r, c in KNIGHT_TARGETS[row][col]:
        piece = sq[r][c]
        if (
            piece is not None
            and piece.color == attacker_color
            and piece.type is PieceType.KNIGHT
        ):
            return True
    for r, c in KING_TARGETS[row][col]:
        piece = sq[r][c]
        if (
            piece is not None
            and piece.color == attacker_color
            and piece.type is PieceType.KING
        ):
            return True
    # One table unifies the sliding pieces: orthogonal rays carry rook
    # and queen power, diagonal rays carry bishop and queen power.  Each
    # ray stops at the first piece — friend or foe — see module docstring.
    for rays, types in (
        (ORTHOGONAL_RAYS[row][col], (PieceType.ROOK, PieceType.QUEEN)),
        (DIAGONAL_RAYS[row][col], (PieceType.BISHOP, PieceType.QUEEN)),
    ):
        for ray in rays:
            for r, c in ray:
                piece = sq[r][c]
                if piece is not None:
                    if piece.color == attacker_color and piece.type in types:
                        return True
                    break
    return False


def is_in_check(board: Board, player: Color) -> bool:
    """Fast check detection using cached king position."""
    king_sq = board.king_square(player)
    if king_sq is None:
        # Fallback for boards constructed outside ChessGame (tests).
        king_sq = find_king(board, player)
    return is_square_attacked(king_sq, "b" if player == "w" else "w", board)

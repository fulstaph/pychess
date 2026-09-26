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


def find_king(board: Board, color: Color) -> Square:
    # A legal position always has exactly one king per side; anything
    # else means a corrupted board, so fail loudly with a count.
    found = [
        (row, col)
        for row, squares in enumerate(board.squares)
        for col, piece in enumerate(squares)
        if piece is not None and piece.type == PieceType.KING and piece.color == color
    ]
    if len(found) != 1:
        raise ValueError(f"Expected exactly one {color} king")
    return found[0]


def is_square_attacked(square: Square, attacker_color: Color, board: Board) -> bool:
    # Validate the target via board.get before any indexing below: a bad
    # square would otherwise raise IndexError/KeyError with no context.
    board.get(square)  # Validate the target before indexing.
    if attacker_color not in ("w", "b"):
        raise ValueError("Invalid attacker color")
    row, col = square
    # Pawn attacks are asymmetric: a white pawn on row r attacks from
    # that row upward, so it hits square (row, col) only if it sits on
    # the row *behind* the target from its own perspective —
    # row + 1 for white (below the target), row - 1 for black.  The
    # diagonal offset is exactly one column either side.
    pawn_row = row + (1 if attacker_color == "w" else -1)
    for pawn_col in (col - 1, col + 1):
        if on_board(pawn_row, pawn_col):
            piece = board.get((pawn_row, pawn_col))
            if piece and piece.color == attacker_color and piece.type == PieceType.PAWN:
                return True
    # Knights and kings have fixed, non-sliding step sets: check the
    # destination of each step directly; no ray walk, no blocking.
    for dr, dc in KNIGHT_STEPS:
        r, c = row + dr, col + dc
        if on_board(r, c):
            piece = board.get((r, c))
            if (
                piece
                and piece.color == attacker_color
                and piece.type == PieceType.KNIGHT
            ):
                return True
    for dr, dc in KING_STEPS:
        r, c = row + dr, col + dc
        if on_board(r, c):
            piece = board.get((r, c))
            if piece and piece.color == attacker_color and piece.type == PieceType.KING:
                return True
    # One table unifies the sliding pieces: orthogonal rays carry rook
    # and queen power, diagonal rays carry bishop and queen power.  Each
    # ray stops at the first piece — friend or foe — see module docstring.
    for directions, types in (
        (ORTHOGONAL, (PieceType.ROOK, PieceType.QUEEN)),
        (DIAGONAL, (PieceType.BISHOP, PieceType.QUEEN)),
    ):
        for dr, dc in directions:
            r, c = row + dr, col + dc
            while on_board(r, c):
                piece = board.get((r, c))
                if piece is not None:
                    if piece.color == attacker_color and piece.type in types:
                        return True
                    break
                r, c = r + dr, c + dc
    return False


def is_in_check(board: Board, player: Color) -> bool:
    if player not in ("w", "b"):
        raise ValueError("Invalid player color")
    return is_square_attacked(
        find_king(board, player), "b" if player == "w" else "w", board
    )

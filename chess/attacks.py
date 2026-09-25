"""Attack detection independent of move legality or turn history."""

from .piece import Board, Color, PieceType, Square

KNIGHT_STEPS = ((-2, -1), (-2, 1), (-1, -2), (-1, 2), (1, -2), (1, 2), (2, -1), (2, 1))
KING_STEPS = tuple((dr, dc) for dr in (-1, 0, 1) for dc in (-1, 0, 1) if dr or dc)
ORTHOGONAL = ((-1, 0), (1, 0), (0, -1), (0, 1))
DIAGONAL = ((-1, -1), (-1, 1), (1, -1), (1, 1))


def on_board(row: int, col: int) -> bool:
    return 0 <= row < 8 and 0 <= col < 8


def find_king(board: Board, color: Color) -> Square:
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
    board.get(square)  # Validate the target before indexing.
    if attacker_color not in ("w", "b"):
        raise ValueError("Invalid attacker color")
    row, col = square
    pawn_row = row + (1 if attacker_color == "w" else -1)
    for pawn_col in (col - 1, col + 1):
        if on_board(pawn_row, pawn_col):
            piece = board.get((pawn_row, pawn_col))
            if piece and piece.color == attacker_color and piece.type == PieceType.PAWN:
                return True
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

"""Terminal User Interface (TUI) components, layout, and rendering."""

from collections import Counter
from collections.abc import Sequence

from .attacks import find_king
from .engine import PIECE_VALUES, evaluate, move_notation
from .game import UNICODE_PIECES, ChessGame, GameStatus
from .helpers import notation_to_coords
from .move import Move
from .piece import Board, Color, Piece, PieceType, Square

STARTING_PIECE_COUNTS: dict[PieceType, int] = {
    PieceType.PAWN: 8,
    PieceType.ROOK: 2,
    PieceType.KNIGHT: 2,
    PieceType.BISHOP: 2,
    PieceType.QUEEN: 1,
}

# ANSI Escape sequences
RESET = "\033[0m"
BOLD = "\033[1m"
DIM = "\033[2m"

# Background colors for board squares
BG_LIGHT = "\033[48;5;250m"
BG_DARK = "\033[48;5;240m"
BG_HIGHLIGHT = "\033[48;5;178m"  # Gold / Amber for last move
BG_CHECK = "\033[48;5;196m"  # Red for checked king

# Foreground colors
FG_BLACK_PIECE = "\033[38;5;16m"
FG_WHITE_PIECE = "\033[38;5;231m"
FG_GOLD = "\033[38;5;220m"
FG_CYAN = "\033[38;5;75m"
FG_GREEN = "\033[38;5;114m"
FG_RED = "\033[38;5;203m"


def get_captured_pieces(
    board: Board,
) -> tuple[tuple[Piece, ...], tuple[Piece, ...], int]:
    """Calculate pieces captured by White, pieces captured by Black, and material diff.

    Returns:
        (captured_by_white, captured_by_black, material_diff)
    """
    counts_w: Counter[PieceType] = Counter()
    counts_b: Counter[PieceType] = Counter()
    for row in board.squares:
        for p in row:
            if p is not None and p.type != PieceType.KING:
                if p.color == "w":
                    counts_w[p.type] += 1
                else:
                    counts_b[p.type] += 1

    captured_by_w: list[Piece] = []
    captured_by_b: list[Piece] = []
    # If black is missing pieces, white captured them
    for pt, starting_count in STARTING_PIECE_COUNTS.items():
        missing_b = starting_count - counts_b[pt]
        if missing_b > 0:
            captured_by_w.extend([Piece(pt, "b")] * missing_b)
        missing_w = starting_count - counts_w[pt]
        if missing_w > 0:
            captured_by_b.extend([Piece(pt, "w")] * missing_w)

    val_w = sum(PIECE_VALUES[p.type] for p in captured_by_w)
    val_b = sum(PIECE_VALUES[p.type] for p in captured_by_b)
    diff = val_w - val_b

    # Sort high to low value
    captured_by_w.sort(key=lambda p: PIECE_VALUES[p.type], reverse=True)
    captured_by_b.sort(key=lambda p: PIECE_VALUES[p.type], reverse=True)
    return tuple(captured_by_w), tuple(captured_by_b), diff


def format_recent_moves(moves: Sequence[Move], max_pairs: int = 4) -> list[str]:
    """Format recent moves into algebraic notation pairs."""
    if not moves:
        return ["  (No moves yet)"]

    pairs: list[str] = []
    total = len(moves)
    # Group moves by move number
    move_pairs: list[tuple[int, str, str | None]] = []
    for i in range(0, total, 2):
        move_num = (i // 2) + 1
        w_move = move_notation(moves[i])
        b_move = move_notation(moves[i + 1]) if i + 1 < total else None
        move_pairs.append((move_num, w_move, b_move))

    recent = move_pairs[-max_pairs:]
    for num, w_str, b_str in recent:
        if b_str is not None:
            pairs.append(f"  {num:2d}. {w_str:<7s} {b_str}")
        else:
            pairs.append(f"  {num:2d}. {w_str}")
    return pairs


def format_legal_moves(game: ChessGame, from_square_str: str | None = None) -> str:
    """Format available legal moves for the current position."""
    all_moves = game.legal_moves()
    if from_square_str:
        try:
            target_sq = notation_to_coords(from_square_str.strip().lower())
            all_moves = tuple(m for m in all_moves if m.from_square == target_sq)
        except ValueError:
            return f"Invalid square: {from_square_str!r}"

    if not all_moves:
        return "No legal moves available."

    by_type: dict[str, list[str]] = {}
    for m in all_moves:
        name = m.piece.type.name.capitalize()
        by_type.setdefault(name, []).append(move_notation(m))

    lines: list[str] = []
    for piece_name, move_list in sorted(by_type.items()):
        lines.append(f"{piece_name:<8s}: {', '.join(sorted(move_list))}")
    return "\n".join(lines)


def format_eval_breakdown(board: Board) -> str:
    """Calculate and format positional and material score breakdown."""
    mat_w = sum(
        PIECE_VALUES[p.type]
        for row in board.squares
        for p in row
        if p and p.color == "w"
    )
    mat_b = sum(
        PIECE_VALUES[p.type]
        for row in board.squares
        for p in row
        if p and p.color == "b"
    )
    total_eval = evaluate(board)
    pos_diff = total_eval - (mat_w - mat_b)

    return (
        f"Material:   White {mat_w} vs Black {mat_b} (diff: {mat_w - mat_b:+d})\n"
        f"Positional: {pos_diff:+d}\n"
        f"Total Eval: {total_eval:+d} (from White's perspective)"
    )


def format_pgn(
    moves: Sequence[Move],
    white_name: str = "White",
    black_name: str = "Black",
    result: str = "*",
) -> str:
    """Format game into a standard Portable Game Notation (PGN) string."""
    headers = [
        '[Event "pychess Terminal Game"]',
        '[Site "Terminal"]',
        f'[White "{white_name}"]',
        f'[Black "{black_name}"]',
        f'[Result "{result}"]',
        "",
    ]
    move_tokens: list[str] = []
    for i in range(0, len(moves), 2):
        num = (i // 2) + 1
        w_not = move_notation(moves[i])
        if i + 1 < len(moves):
            b_not = move_notation(moves[i + 1])
            move_tokens.append(f"{num}. {w_not} {b_not}")
        else:
            move_tokens.append(f"{num}. {w_not}")

    if result != "*":
        move_tokens.append(result)

    return "\n".join(headers) + " ".join(move_tokens) + "\n"


def get_piece_art_lines(
    piece_type: PieceType,
    color: Color,
    size: int,
    unicode_pieces: bool = True,
) -> list[str]:
    """Return scaled multi-line piece art for a given piece and board size."""
    glyph = (
        UNICODE_PIECES[(piece_type, color)]
        if unicode_pieces
        else (piece_type.value.upper() if color == "w" else piece_type.value.lower())
    )
    if size == 1:
        return [glyph]

    pt = piece_type.value
    if size == 2:
        if pt == "p":
            return [" ( ) ", f" /{glyph}\\ "]
        if pt == "r":
            return [" [U] ", f" |{glyph}| "]
        if pt == "n":
            return [" /\\_ ", f" |{glyph}\\ "]
        if pt == "b":
            return [" (o) ", f" /{glyph}\\ "]
        if pt == "q":
            return [" ^^^ ", f" ({glyph}) "]
        return ["  +  ", f" ({glyph}) "]

    if size == 3:
        if pt == "p":
            return ["       ", "  (o)  ", f"  /{glyph}\\  "]
        if pt == "r":
            return ["  [U]  ", f"  |{glyph}|  ", "  ===  "]
        if pt == "n":
            return ["  /\\_  ", f"  |{glyph}\\  ", "  ===  "]
        if pt == "b":
            return ["   ()  ", f"  ({glyph})  ", "  ===  "]
        if pt == "q":
            return ["  ^^^  ", f"  ({glyph})  ", "  ===  "]
        return ["   +   ", f"  ({glyph})  ", "  ===  "]

    # size == 4
    if pt == "p":
        return ["         ", "   (o)   ", f"   /{glyph}\\   ", "  =====  "]
    if pt == "r":
        return ["  [U U]  ", f"  | {glyph} |  ", "  |   |  ", "  =====  "]
    if pt == "n":
        return ["   /|\\   ", f"  //{glyph} \\  ", "  // \\|  ", "  =====  "]
    if pt == "b":
        return ["   (o)   ", f"   /{glyph}\\   ", "   \\ /   ", "  =====  "]
    if pt == "q":
        return ["  \\\\_v_//", f"  / {glyph} \\  ", "  \\___/  ", "  =====  "]
    return ["    +    ", f"  ( {glyph} )  ", "  /___\\  ", "  =====  "]


def get_empty_square_lines(size: int, ansi_colors: bool) -> list[str]:
    """Return lines for an empty board square of given size."""
    dot = "·" if ansi_colors else "."
    if size == 1:
        return [f" {dot} "]
    if size == 2:
        return ["     ", f"  {dot}  "]
    if size == 3:
        return ["       ", f"   {dot}   ", "       "]
    return ["         ", "         ", f"    {dot}    ", "         "]


def render_board_lines(
    board: Board,
    flip: bool = False,
    unicode_pieces: bool = True,
    ansi_colors: bool = True,
    last_move: Move | None = None,
    check_square: Square | None = None,
    size: int = 1,
) -> list[str]:
    """Render the 8x8 chess board into formatted terminal string lines."""
    if size not in (1, 2, 3, 4):
        raise ValueError(f"Board size must be 1, 2, 3, or 4, got {size}")

    rows = list(range(8)) if not flip else list(range(7, -1, -1))
    cols = list(range(8)) if not flip else list(range(7, -1, -1))

    highlighted_squares: set[Square] = set()
    if last_move is not None:
        highlighted_squares.add(last_move.from_square)
        highlighted_squares.add(last_move.to_square)

    lines: list[str] = []
    file_labels = [chr(ord("a") + c) for c in cols]
    if size == 1:
        gap = "  " if ansi_colors else " "
        header = f"   {gap}" + gap.join(file_labels) + f"{gap}   "
    elif size == 2:
        gap = "    "
        header = f"   {gap}" + gap.join(file_labels) + f"{gap}   "
    elif size == 3:
        gap = "      "
        header = f"   {gap}" + gap.join(file_labels) + f"{gap}   "
    else:
        gap = "        "
        header = f"   {gap}" + gap.join(file_labels) + f"{gap}   "
    lines.append(header)

    sq_height = size
    for r in rows:
        rank_num = 8 - r
        for h in range(sq_height):
            is_center_row = h == (sq_height // 2)
            left_label = f" {rank_num} " if is_center_row else "   "
            right_label = f" {rank_num}" if is_center_row else "   "

            cells: list[str] = []
            for c in cols:
                piece = board.squares[r][c]
                sq = (r, c)

                if size == 1 and not ansi_colors:
                    if piece is not None:
                        content = (
                            piece.type.value.upper()
                            if piece.color == "w"
                            else piece.type.value
                        )
                    else:
                        content = "."
                elif piece is not None:
                    art = get_piece_art_lines(
                        piece.type, piece.color, size, unicode_pieces
                    )
                    content = f" {art[0]} " if size == 1 and ansi_colors else art[h]
                else:
                    art = get_empty_square_lines(size, ansi_colors)
                    content = art[h]

                if ansi_colors:
                    if sq == check_square:
                        bg = BG_CHECK
                        fg = FG_WHITE_PIECE + BOLD
                    elif sq in highlighted_squares:
                        bg = BG_HIGHLIGHT
                        fg = FG_BLACK_PIECE + BOLD
                    else:
                        is_light = (r + c) % 2 == 0
                        bg = BG_LIGHT if is_light else BG_DARK
                        fg = (
                            FG_WHITE_PIECE
                            if piece and piece.color == "w"
                            else FG_BLACK_PIECE
                        )
                    cells.append(f"{bg}{fg}{content}{RESET}")
                else:
                    cells.append(content)

            sep = "" if ansi_colors else " "
            lines.append(f"{left_label}{sep.join(cells)}{right_label}")

    lines.append(header)
    return lines


def render_dashboard(
    game: ChessGame,
    moves_history: Sequence[Move] = (),
    flip: bool = False,
    unicode_pieces: bool = True,
    ansi_colors: bool = True,
    white_name: str = "White",
    black_name: str = "Black",
    size: int = 1,
) -> str:
    """Render a side-by-side terminal dashboard containing board and game status."""
    last_move = moves_history[-1] if moves_history else None
    check_sq = (
        find_king(game.board, game.turn)
        if game.status in (GameStatus.CHECK, GameStatus.CHECKMATE)
        else None
    )

    board_lines = render_board_lines(
        game.board,
        flip=flip,
        unicode_pieces=unicode_pieces,
        ansi_colors=ansi_colors,
        last_move=last_move,
        check_square=check_sq,
        size=size,
    )

    captured_w, captured_b, diff = get_captured_pieces(game.board)
    cap_w_str = (
        "".join(UNICODE_PIECES[(p.type, p.color)] for p in captured_w)
        if captured_w
        else "-"
    )
    cap_b_str = (
        "".join(UNICODE_PIECES[(p.type, p.color)] for p in captured_b)
        if captured_b
        else "-"
    )
    if diff > 0:
        lead_str = f"White +{diff}"
    elif diff < 0:
        lead_str = f"Black +{-diff}"
    else:
        lead_str = "Equal (0)"

    # Build sidebar lines
    sidebar: list[str] = [
        f"{BOLD}pychess{RESET} {DIM}v0.1.0{RESET}",
        "─" * 36,
        f"Match:   {white_name} (W) vs {black_name} (B)",
        f"Turn:    {'White' if game.turn == 'w' else 'Black'} (Move {game.move_num})",
        f"Status:  {game.status.value.upper()}",
        "─" * 36,
        f"Lead:    {lead_str}",
        f"Captures W: {cap_w_str}",
        f"Captures B: {cap_b_str}",
        "─" * 36,
        "Recent Moves:",
    ]
    sidebar.extend(format_recent_moves(moves_history, max_pairs=3))
    sidebar.append("─" * 36)
    sidebar.append("Commands: undo | moves [sq] | eval | pgn | flip | size [1-4] | q")

    # Merge board lines and sidebar lines side-by-side
    combined_lines: list[str] = []
    max_rows = max(len(board_lines), len(sidebar))
    for i in range(max_rows):
        b_line = board_lines[i] if i < len(board_lines) else " " * len(board_lines[0])
        s_line = sidebar[i] if i < len(sidebar) else ""
        combined_lines.append(f"{b_line}    {s_line}")

    return "\n".join(combined_lines)

"""Terminal User Interface (TUI) components, layout, and rendering.

Architecture
------------
This module is a strictly render-only layer: every function takes plain
data (a ``Board``, ``ChessGame``, or a move sequence) and returns ANSI-
formatted strings. There is no state, no I/O, and no mutation — callers
``print`` the results, so the same helpers can be reused by the CLI
(``chess.engine``), tests, and future TUI runtimes.

Palette
    Color is expressed as a fixed set of ANSI 256-color escape constants
    (``BG_*`` for square backgrounds, ``FG_*`` for piece/level foregrounds)
    plus ``BOLD``/``DIM``/``RESET``. When ``ansi_colors`` is False the
    renderers degrade to plain text with "." for empty squares.

Board scaling
    Pieces are drawn as multi-line ASCII/Unicode art that scales with
    ``size`` (1 = one glyph per square, 2/3/4 = 2/3/4 lines per square),
    so the same board renders from a compact in-line grid to a giant
    poster without changing the caller's code.

Dashboard layout
    ``render_dashboard`` composes two columns side by side: the board on
    the left, a status sidebar on the right (match info, lead, captures,
    recent moves, command hints). An optional ``ring`` parameter appends a
    box-drawn LOG panel below the sidebar so live log entries stay
    visible during play.
"""

from collections import Counter
from collections.abc import Sequence

from .attacks import find_king
from .engine import PIECE_VALUES, evaluate, move_notation
from .game import UNICODE_PIECES, ChessGame, GameStatus
from .helpers import notation_to_coords
from .log import LogRing, get_logger
from .move import Move
from .piece import Board, Color, Piece, PieceType, Square

logger = get_logger("tui")

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
    """Derive captured-piece lists and the material difference.

    The board only shows what is ON it, so captures are reconstructed by
    diffing the current piece counts against the starting set (kings are
    excluded — they cannot be captured). Each missing black piece is
    attributed to White's capture list and vice versa.

    ``material_diff`` = value captured by White minus value captured by
    Black, i.e. positive means White is ahead in the capture race, from
    White's perspective. Both lists are sorted by descending piece value
    so the most important trophies print first.

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
    """Format the most recent moves as numbered "N. e4 e5" pairs.

    Moves are grouped into full moves (white then black) starting from
    the FIRST move, then only the last ``max_pairs`` pairs are shown so
    the panel tracks the latest play. A trailing half-move (white played,
    black not yet) is printed without a second column.
    """
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
    """Format legal moves grouped by piece type for the current position.

    When ``from_square_str`` is given (e.g. "e4"), the list is filtered to
    moves of the piece on that square; an unparseable square returns an
    "Invalid square" message instead of raising. Output is one line per
    piece type, alphabetized, with moves sorted in coordinate order.
    """
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
    """Split the engine's total score into material vs positional parts.

    Material totals are computed directly from piece values; the
    positional component is what remains after subtracting the pure
    material difference from ``evaluate``'s white-positive total, so the
    three lines always add up (material diff + positional == total).
    """
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
    """Build a PGN string: five tag pair headers, then the move text.

    Move tokens use coordinate notation grouped into full moves
    ("1. e4 e5"); a lone trailing white move is printed without its
    black reply. The game result marker ("1-0", "0-1", "1/2-1/2", ...)
    is appended after the moves unless it is the default "*" (ongoing),
    which PGN convention keeps in the Result header only.
    """
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
    """Return the multi-line art for one piece at the given board size.

    Scaling rule: size 1 is a single glyph line; sizes 2/3/4 return
    2/3/4 lines of ASCII art with the glyph embedded on the middle-ish
    row, so every piece occupies a square that is ``size`` lines tall
    (matching ``get_empty_square_lines``). Unicode mode uses the
    Unicode chess glyphs; plain mode falls back to upper-case (white) /
    lower-case (black) single letters.
    """
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
    """Return the ``size`` blank lines of a vacant square.

    Must match the height of the piece art for the same size so filled
    and empty squares align; a centered dot (middle line) marks the
    square, styled for ANSI or plain text.
    """
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
    """Render the 8x8 board as a list of terminal lines, framed by file labels.

    Layout: a header/footer row of file letters (a-h), and per board row
    the rank number is printed on the vertical middle line of the square
    (squares are ``size`` lines tall). ``flip`` reverses both row and
    column order so Black can be drawn at the bottom.

    Highlights (ANSI mode only): the from/to squares of ``last_move``
    get a gold background (``BG_HIGHLIGHT``); ``check_square`` (the
    king currently in check) gets red (``BG_CHECK``) and takes priority
    over the last-move highlight. Without ANSI colors, squares degrade
    to plain text with single-letter pieces and "." for empty.
    """
    if size not in (1, 2, 3, 4):
        raise ValueError(f"Board size must be 1, 2, 3, or 4, got {size}")
    logger.debug("rendering board size=%d", size)

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
        # board.squares is indexed with row 0 == rank 8 (white's back
        # rank), hence the inversion.
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


_LOG_LEVEL_COLORS: dict[str, str] = {
    "INFO": FG_CYAN,
    "WARNING": FG_GOLD,
    "ERROR": FG_RED,
    "DEBUG": DIM,
}


def render_log_panel(ring: LogRing, height: int = 6) -> str:
    """Render a box-drawn "LOG" panel with the ring's most recent entries.

    The panel is exactly ``height`` lines tall (top border + content +
    bottom border) and 36 columns wide, matching the dashboard sidebar.
    Entries are colored by level (INFO cyan, WARNING gold, ERROR red,
    DEBUG dim), with the oldest entry at the top and the newest at the
    bottom; the ring is trimmed to the most recent entries that fit the
    content area. An empty ring renders an empty box with a dim
    "no events" line.
    """
    inner_w = 34  # sidebar width (36) minus the two border columns
    inner_h = max(height - 2, 0)
    entries = ring.recent(inner_h) if inner_h > 0 else ()

    body: list[str] = []
    if entries:
        # ring.recent() returns newest LAST, so iterating in order puts
        # the newest entry on the bottom content line of the box.
        for entry in entries:
            text = f"{entry.level:<7} {entry.message}"[:inner_w]
            color = _LOG_LEVEL_COLORS.get(entry.level, RESET)
            body.append(f"{color}{text}{RESET}")
    elif inner_h > 0:
        body.append(f"{DIM}  no events{RESET}")
    body.extend([" " * inner_w] * (inner_h - len(body)))

    lines = [
        "┌─ LOG " + "─" * (inner_w - 6) + "┐",
        *body,
        "└" + "─" * inner_w + "┘",
    ]
    return "\n".join(lines)


def render_dashboard(
    game: ChessGame,
    moves_history: Sequence[Move] = (),
    flip: bool = False,
    unicode_pieces: bool = True,
    ansi_colors: bool = True,
    white_name: str = "White",
    black_name: str = "Black",
    size: int = 1,
    ring: LogRing | None = None,
) -> str:
    """Render a side-by-side terminal dashboard: board left, status right.

    Column composition: the left column is ``render_board_lines`` (with
    last-move and check highlights); the right column is a fixed-width
    sidebar (36 columns) holding the match header, turn/status, material
    lead, captured pieces, recent moves, and command hints. The two
    columns are merged row by row, padding the shorter one so the output
    is a ragged-free block; four spaces separate the columns.

    When ``ring`` is given, a box-drawn LOG panel (see
    :func:`render_log_panel`) is appended below the sidebar so recent
    log entries stay visible during play; with ``ring=None`` (the
    default) the layout is exactly as before and no panel is printed.
    """
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

    # Optional LOG panel: appended below the status sections so recent
    # entries stay visible while playing; omitted entirely when no ring
    # is supplied, keeping the legacy layout byte-identical.
    if ring is not None:
        sidebar.append("")
        sidebar.extend(render_log_panel(ring).split("\n"))
    # Merge board lines and sidebar lines side-by-side
    combined_lines: list[str] = []
    max_rows = max(len(board_lines), len(sidebar))
    for i in range(max_rows):
        b_line = board_lines[i] if i < len(board_lines) else " " * len(board_lines[0])
        s_line = sidebar[i] if i < len(sidebar) else ""
        combined_lines.append(f"{b_line}    {s_line}")

    return "\n".join(combined_lines)

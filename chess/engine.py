"""Play chess against an alpha-beta minimax engine using the shared game rules.

Architecture
------------
Evaluation model
    ``evaluate`` is a white-positive static score: sum of material values
    (``PIECE_VALUES``) plus small piece-square-table bonuses for central and
    home-rank squares. Black's table lookups are mirrored vertically
    (``effective_row = 7 - row``) so a black piece enjoys the same positional
    bonus on its own central squares.

Search
    ``choose_move`` runs an alpha-beta search (default depth 2) over the root
    move list. Root moves are ordered by MVV-LVA (see ``order_moves``) so the
    most promising captures are tried first, maximizing pruning. Optional
    quiescence search (``_quiescence``) extends leaf nodes through captures
    only, preventing the horizon effect where a player "defends" by moving
    out of attack instead of losing the piece.

Tie-break policy
    On equal scores the FIRST ordered move wins: ``best_move`` is only
    replaced on a strict improvement. This keeps the engine fully
    deterministic for a given position and depth.

``choose_move`` is a pure query over ``ChessGame`` — it never mutates the
game; each candidate is examined via the immutable ``game.after(move)``.

Run with ``python -m chess.engine``.
"""

import argparse
import logging
import sys
from collections.abc import Sequence

from .game import ChessGame, GameStatus, print_board
from .helpers import square_notation
from .log import LogHandler, LogRing, get_logger, setup_logging
from .move import Move
from .piece import Board, Color, PieceType

logger = get_logger("engine")

PIECE_VALUES: dict[PieceType, int] = {
    PieceType.PAWN: 100,
    PieceType.KNIGHT: 320,
    PieceType.BISHOP: 330,
    PieceType.ROOK: 500,
    PieceType.QUEEN: 900,
    PieceType.KING: 0,
}

PAWN_TABLE: dict[tuple[int, int], int] = {
    (3, 3): 20,
    (3, 4): 20,
    (4, 3): 30,
    (4, 4): 30,
}

KNIGHT_TABLE: dict[tuple[int, int], int] = {
    (5, 2): 15,
    (5, 5): 15,
    (2, 2): 15,
    (2, 5): 15,
}

BISHOP_TABLE: dict[tuple[int, int], int] = {
    (5, 2): 10,
    (5, 5): 10,
    (4, 2): 10,
    (4, 3): 10,
    (4, 4): 10,
    (4, 5): 10,
    (3, 2): 10,
    (3, 3): 10,
    (3, 4): 10,
    (3, 5): 10,
}

ROOK_TABLE: dict[tuple[int, int], int] = {
    (1, 0): 20,
    (1, 1): 20,
    (1, 2): 20,
    (1, 3): 20,
    (1, 4): 20,
    (1, 5): 20,
    (1, 6): 20,
    (1, 7): 20,
}

QUEEN_TABLE: dict[tuple[int, int], int] = {
    (3, 3): 10,
    (3, 4): 10,
    (4, 3): 10,
    (4, 4): 10,
}

KING_TABLE: dict[tuple[int, int], int] = {
    (7, 6): 20,
    (7, 2): 15,
}

PIECE_TABLES: dict[PieceType, dict[tuple[int, int], int]] = {
    PieceType.PAWN: PAWN_TABLE,
    PieceType.KNIGHT: KNIGHT_TABLE,
    PieceType.BISHOP: BISHOP_TABLE,
    PieceType.ROOK: ROOK_TABLE,
    PieceType.QUEEN: QUEEN_TABLE,
    PieceType.KING: KING_TABLE,
}


def mvv_lva_score(move: Move) -> int:
    """Score a move for search ordering (higher = try earlier).

    Most Valuable Victim / Least Valuable Attacker heuristic:
    ``victim_value * 10 - attacker_value``. Capturing with a pawn while
    taking a queen scores far higher than a rook taking a pawn, so the
    "safely best" captures are explored first and the alpha-beta window
    closes sooner.

    The flat ``+10000`` guarantees every capture outranks every quiet
    move, and ``+9000`` (just under 10000) makes promotions the second
    most attractive category without ever beating a capture.
    """
    score = 0
    if move.captured_piece is not None:
        # victim * 10 dominates the attacker term, so a pawn taking a
        # rook (5000 - 100) still beats a queen taking a pawn (9000 - 900).
        victim = PIECE_VALUES.get(move.captured_piece.type, 0)
        attacker = PIECE_VALUES.get(move.piece.type, 0)
        score += 10000 + (victim * 10 - attacker)
    if move.special == "promotion":
        score += 9000
    return score


def order_moves(moves: Sequence[Move]) -> list[Move]:
    """Sort moves best-first by MVV-LVA score to maximize alpha-beta pruning.

    Captures/promotions come first, so the most likely window-closing lines
    are explored before quiet moves.
    """
    return sorted(moves, key=mvv_lva_score, reverse=True)


def _quiescence(game: ChessGame, alpha: int, beta: int, qdepth: int = 3) -> int:
    """Search captures only at the horizon to prevent the horizon effect.

    At the end of a normal search a losing side can simply move out of
    attack, so static eval overstates its position. Extending the search
    through forced captures fixes that. Invariants:

    - Terminal positions short-circuit: checkmate is +-100000 (sign by who
      is mated — the side to move has no escape), stalemate is 0.
    - ``stand_pat`` (quiet eval) is the alpha (max) or beta (min) bound for
      the side to move; a stand-pat cutoff returns the window bound, not
      the exact score, which is all alpha-beta needs.
    - Only ``is_capture()`` moves and promotions are searched, and only up
      to ``qdepth`` plies deep, so the horizon stays shallow and bounded.
    """
    if game.status == GameStatus.CHECKMATE:
        return 100000 if game.turn == "b" else -100000
    if game.status == GameStatus.STALEMATE:
        return 0

    stand_pat = evaluate(game.board)
    if game.turn == "w":
        if stand_pat >= beta:
            return beta
        if stand_pat > alpha:
            alpha = stand_pat
        if qdepth == 0:
            return stand_pat
        captures = [
            m for m in game.legal_moves() if m.is_capture() or m.special == "promotion"
        ]
        for move in order_moves(captures):
            score = _quiescence(game.after(move), alpha, beta, qdepth - 1)
            if score >= beta:
                return beta
            if score > alpha:
                alpha = score
        return alpha
    if stand_pat <= alpha:
        return alpha
    if stand_pat < beta:
        beta = stand_pat
    if qdepth == 0:
        return stand_pat
    captures = [
        m for m in game.legal_moves() if m.is_capture() or m.special == "promotion"
    ]
    for move in order_moves(captures):
        score = _quiescence(game.after(move), alpha, beta, qdepth - 1)
        if score <= alpha:
            return alpha
        if score < beta:
            beta = score
    return beta


def evaluate(board: Board) -> int:
    """White-positive static eval: material plus piece-square-table bonus.

    One pass over the 8x8 grid: white pieces add ``material + bonus``,
    black pieces subtract it. The bonus is a sparse table lookup keyed by
    ``(row, col)``; black rows are mirrored (``7 - row``) so both colors
    value their own central squares identically. King value is 0, so kings
    only contribute positional safety bonuses.
    """
    score = 0
    for row, squares in enumerate(board.squares):
        for col, piece in enumerate(squares):
            if piece is None:
                continue
            material = PIECE_VALUES[piece.type]
            table = PIECE_TABLES.get(piece.type)
            if piece.color == "w":
                bonus = table.get((row, col), 0) if table else 0
                score += material + bonus
            else:
                effective_row = 7 - row
                bonus = table.get((effective_row, col), 0) if table else 0
                score -= material + bonus
    return score


def _minimax(
    game: ChessGame,
    depth: int,
    alpha: int,
    beta: int,
    ply: int,
    quiescence: bool = False,
) -> int:
    """Alpha-beta minimax recursion from White's point of view.

    Returns a white-positive score for ``game`` at ``depth`` plies.
    Terminal handling comes first: CHECKMATE is +-100000 (positive when the
    mated side to move is Black) minus/plus ``ply`` so that faster mates
    are preferred over slower ones; STALEMATE is 0. At ``depth == 0`` the
    leaf is a static evaluation — optionally extended through captures via
    ``_quiescence`` when the caller requested quiescent search. The empty-
    move fallback re-checks ``is_check()`` so a side with no legal moves is
    scored as mated, not stalemated.

    The two branches are mirror images of each other (explicit max for
    white, explicit min for black rather than negamax). After each child:
    the side's ``value`` is tightened, then the shared alpha/beta window
    is raised (max) or lowered (min); ``alpha >= beta`` means the window
    has closed and the remaining siblings can be pruned (fail-soft: the
    returned bound may overshoot the true score).
    """
    if game.status == GameStatus.CHECKMATE:
        return 100000 - ply if game.turn == "b" else -100000 + ply
    if game.status == GameStatus.STALEMATE:
        return 0
    if depth == 0:
        return _quiescence(game, alpha, beta) if quiescence else evaluate(game.board)

    moves = game.legal_moves()
    if not moves:
        if game.is_check():
            return 100000 - ply if game.turn == "b" else -100000 + ply
        return 0

    ordered = order_moves(moves)
    if game.turn == "w":
        value = -1_000_000
        for move in ordered:
            score = _minimax(
                game.after(move),
                depth - 1,
                alpha,
                beta,
                ply + 1,
                quiescence=quiescence,
            )
            if score > value:
                value = score
            if value > alpha:
                alpha = value
            if alpha >= beta:
                break
        return value
    value = 1_000_000
    for move in ordered:
        score = _minimax(
            game.after(move),
            depth - 1,
            alpha,
            beta,
            ply + 1,
            quiescence=quiescence,
        )
        if score < value:
            value = score
        if value < beta:
            beta = value
        if alpha >= beta:
            break
    return value


def choose_move(game: ChessGame, depth: int = 2, quiescence: bool = False) -> Move:
    """Select the best move by alpha-beta search (pure query, no mutation).

    The root position is never scored itself; each candidate is examined
    through ``game.after(move)`` at ``depth - 1`` plies. Root moves are
    ordered by MVV-LVA. ``best_move`` is replaced only on a STRICT
    improvement, so on equal scores the first ordered move wins — the
    engine is deterministic for a given position and depth. The alpha/beta
    window is widened as the search progresses and the loop exits early
    once the window closes (pruning).
    """
    if depth < 1:
        raise ValueError("Search depth must be at least 1")
    moves = game.legal_moves()
    if not moves:
        raise ValueError("No legal moves available")

    best_move: Move = moves[0]
    alpha = -1_000_000
    beta = 1_000_000
    ordered = order_moves(moves)

    # Local node counter: a thin closure over _minimax keeps the public
    # search signatures untouched while still giving the log line a
    # useful "how much work did this search take" number.
    nodes = 0

    def _counted_minimax(
        search_game: ChessGame,
        search_depth: int,
        alpha_: int,
        beta_: int,
        ply: int,
    ) -> int:
        nonlocal nodes
        nodes += 1
        return _minimax(
            search_game,
            search_depth,
            alpha_,
            beta_,
            ply,
            quiescence=quiescence,
        )

    if game.turn == "w":
        best_score = -1_000_000
        for move in ordered:
            score = _counted_minimax(
                game.after(move),
                depth - 1,
                alpha,
                beta,
                ply=1,
            )
            if score > best_score:
                best_score = score
                best_move = move
            if score > alpha:
                alpha = score
            if alpha >= beta:
                break
    else:
        best_score = 1_000_000
        for move in ordered:
            score = _counted_minimax(
                game.after(move),
                depth - 1,
                alpha,
                beta,
                ply=1,
            )
            if score < best_score:
                best_score = score
                best_move = move
            if score < beta:
                beta = score
            if alpha >= beta:
                break

    logger.info(
        "chose %s depth=%d nodes=%d",
        move_notation(best_move),
        depth,
        nodes,
    )

    return best_move


def move_notation(move: Move) -> str:
    """Display an engine move in coordinate notation accepted by ChessGame."""
    source = square_notation(*move.from_square)
    target = square_notation(*move.to_square)
    promotion = f"={move.promotion_to.value.upper()}" if move.promotion_to else ""
    return f"{source}{target}{promotion}"


def main(argv: Sequence[str] | None = None) -> None:
    """CLI entry point for the interactive terminal game.

    Flow:

    1. Parse flags (``--stockfish`` switches the opponent to an external
       Stockfish process; ``--verbose`` raises the log level to DEBUG).
    2. Set up console logging and an in-memory :class:`LogRing` for the
       dashboard LOG panel.
    3. Side selection with a retry loop on invalid input.
    4. Render loop: each iteration draws the board (plain or dashboard
       with the log ring), reports terminal status (checkmate/stalemate/
       check), then either reads a human move or command (undo/moves/
       eval/pgn/flip/size/fen/q with parse-error retry) or lets the
       engine pick and play a move. ``history`` holds the full list of
       immutable ``ChessGame`` snapshots so ``undo`` can rewind one full
       move (two plies) in one step.
    """
    parser = argparse.ArgumentParser(
        prog="python -m chess.engine",
        description="Play chess in the terminal against minimax AI or Stockfish.",
    )
    parser.add_argument(
        "--stockfish",
        "-s",
        action="store_true",
        help="Play against Stockfish instead of built-in minimax engine",
    )
    parser.add_argument(
        "--stockfish-path",
        type=str,
        default=None,
        help="Path to Stockfish executable",
    )
    parser.add_argument(
        "--unicode",
        "-u",
        action="store_true",
        help="Render pieces using Unicode chess characters",
    )
    parser.add_argument(
        "--color-board",
        action="store_true",
        help="Render board with alternating colored square backgrounds",
    )
    parser.add_argument(
        "--dashboard",
        action="store_true",
        help="Render side-by-side terminal dashboard",
    )
    parser.add_argument(
        "--flip",
        action="store_true",
        help="Flip board orientation (Black at bottom)",
    )
    parser.add_argument(
        "--size",
        "-S",
        type=int,
        choices=[1, 2, 3, 4],
        default=1,
        help="TUI board size: 1 (compact), 2 (medium), 3 (large), 4 (giant)",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Enable DEBUG-level logging",
    )
    args = parser.parse_args(argv)

    # Console logging goes to stderr via the "chess" root logger; the ring
    # keeps recent entries in memory for the TUI LOG panel (dashboard mode).
    setup_logging(logging.DEBUG if args.verbose else logging.INFO)
    log_ring = LogRing()
    logging.getLogger("chess").addHandler(LogHandler(log_ring))

    print("Choose your side: 1) White  2) Black  3) Quit")
    while True:
        choice = input("Selection: ").strip().lower()
        if choice in ("3", "q"):
            print("Goodbye!")
            return
        if choice in ("1", "2"):
            break
        print("Invalid selection. Choose 1, 2, or 3.")

    human_color: Color = "w" if choice == "1" else "b"
    if args.stockfish:
        # Stockfish mode delegates the whole game loop to the stockfish
        # module; the built-in engine loop below only runs for minimax.
        from .stockfish import Stockfish, play_human_vs_stockfish

        try:
            with Stockfish(path=args.stockfish_path) as sf:
                play_human_vs_stockfish(sf, human_color=human_color)
        except FileNotFoundError as err:
            sys.stderr.write(f"Error: {err}\n")
            sys.exit(1)
        return
    game = ChessGame()
    # history stores every immutable ChessGame snapshot ever reached, so
    # undo is just "pop back"; moves_history is the flat move list used
    # for PGN, recent-moves display, and the same rewind on undo.
    history: list[ChessGame] = [game]
    moves_history: list[Move] = []
    # Playing Black in dashboard mode starts with the board flipped so
    # the human's side is at the bottom; --flip always wins.
    flipped = args.flip or (human_color == "b" and args.dashboard)
    board_size = args.size
    while True:
        if args.dashboard:
            from .tui import render_dashboard

            white_label = "Human" if human_color == "w" else "Engine"
            black_label = "Engine" if human_color == "w" else "Human"
            print(
                render_dashboard(
                    game,
                    moves_history=moves_history,
                    flip=flipped,
                    unicode_pieces=args.unicode,
                    ansi_colors=args.color_board,
                    white_name=white_label,
                    black_name=black_label,
                    size=board_size,
                    ring=log_ring,
                )
            )
        else:
            print_board(
                game.board,
                unicode_pieces=args.unicode,
                ansi_colors=args.color_board,
            )
            print(f"Status: {game.status.value}")

        # Terminal status: end the game on mate or stalemate, surface a
        # "Check!" notice, then hand control to the side to move.

        if game.status == GameStatus.CHECKMATE:
            winner = "White" if game.get_winner() == "w" else "Black"
            print(f"Checkmate! {winner} wins.")
            return
        if game.status == GameStatus.STALEMATE:
            print("Stalemate! Draw.")
            return
        if game.status == GameStatus.CHECK:
            print("Check!")

        if game.turn == human_color:
            # Command parsing: anything that is not a move is a recognized
            # command (help/fen/flip/size/eval/pgn/moves/undo/q).
            # Unrecognized text falls through to make_move below, which
            # raises ValueError on illegal input so we can retry.
            text = input("Your move: ").strip()
            lower_text = text.lower()
            if lower_text == "q":
                print("Goodbye!")
                return
            if lower_text in ("help", "?"):
                print(
                    "Commands: 'undo', 'moves' [sq], 'eval', 'pgn', 'flip', "
                    "'size' [1-3], 'fen', 'q' (quit).\n"
                    "Move formats: 'e4', 'Nf3', 'e2e4', 'O-O'"
                )
                continue
            if lower_text == "fen":
                print(f"FEN: {game.to_fen()}")
                continue
            if lower_text == "flip":
                flipped = not flipped
                print("Board flipped.")
                continue
            if lower_text.startswith("size"):
                tokens = text.split()
                if len(tokens) > 1 and tokens[1] in ("1", "2", "3", "4"):
                    board_size = int(tokens[1])
                else:
                    board_size = (board_size % 4) + 1
                size_names = {
                    1: "1 (compact)",
                    2: "2 (medium)",
                    3: "3 (large)",
                    4: "4 (giant)",
                }
                print(f"Board size set to {size_names[board_size]}.")
                continue
            if lower_text == "eval":
                from .tui import format_eval_breakdown

                print(format_eval_breakdown(game.board))
                continue
            if lower_text == "pgn":
                from .tui import format_pgn

                w_name = "Human" if human_color == "w" else "Engine"
                b_name = "Engine" if human_color == "w" else "Human"
                print(
                    format_pgn(
                        moves_history,
                        white_name=w_name,
                        black_name=b_name,
                    )
                )
                continue
            if lower_text.startswith("moves"):
                from .tui import format_legal_moves

                tokens = text.split()
                sq_filter = tokens[1] if len(tokens) > 1 else None
                print(format_legal_moves(game, sq_filter))
                continue
            if lower_text == "undo":
                if len(history) <= 1:
                    print("No moves to undo.")
                    continue
                # Rewind one FULL move (two plies) so it is the human's
                # turn again; only one ply exists when the engine just
                # replied to the very first human move.
                steps = 2 if len(history) >= 3 else 1
                for _ in range(steps):
                    history.pop()
                    if moves_history:
                        moves_history.pop()
                game = history[-1]
                print("Move undone.")
                continue
            try:
                selected = game.make_move(text)
            except ValueError as error:
                print(f"Invalid move: {error}")
                continue
            moves_history.append(selected)
            # Human move accepted: record it in both the move list
            # (for PGN/undo display) and the immutable game history.
            history.append(game)
            print(f"You played: {text}")
        else:
            # Engine turn: the search is a pure query, so choose_move
            # runs first and the move is only applied afterwards.
            selected = choose_move(game)
            game.make_move(selected)
            moves_history.append(selected)
            history.append(game)
            print(f"Engine: {move_notation(selected)}")


if __name__ == "__main__":
    try:
        main()
    except EOFError, KeyboardInterrupt:
        print("\nGoodbye!")

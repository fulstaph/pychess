"""Terminal chess player backed by the configurable built-in search engine."""

import argparse
import logging
import sys
from collections.abc import Sequence

from .game import ChessGame, print_board
from .log import LogHandler, LogRing, get_logger, setup_logging
from .move import Move
from .piece import Color
from .search import (
    DEFAULT_ENGINE_CONFIG,
    MAX_SEARCH_DEPTH,
    PIECE_TABLES,
    PIECE_VALUES,
    EngineConfig,
    SearchEngine,
    SearchStats,
    choose_move,
    evaluate,
    move_notation,
    mvv_lva_score,
    order_moves,
)

logger = get_logger("engine")

__all__ = [
    "DEFAULT_ENGINE_CONFIG",
    "PIECE_TABLES",
    "PIECE_VALUES",
    "EngineConfig",
    "SearchEngine",
    "SearchStats",
    "choose_move",
    "evaluate",
    "move_notation",
    "mvv_lva_score",
    "order_moves",
]


def main(
    argv: Sequence[str] | None = None,
    *,
    engine_config: EngineConfig = DEFAULT_ENGINE_CONFIG,
) -> None:
    """CLI entry point for the interactive terminal game.

    Flow:

    1. Parse flags (``--depth`` selects the built-in search depth;
       ``--stockfish`` switches opponents; ``--verbose`` raises the log level).
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
        "--depth",
        type=int,
        choices=range(1, MAX_SEARCH_DEPTH + 1),
        default=engine_config.search_depth,
        help=f"Built-in engine search depth (1-{MAX_SEARCH_DEPTH})",
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

    search_engine = SearchEngine(engine_config)
    game = ChessGame()
    # Each entry is a distinct game from ``after``. ``make_move`` mutates
    # in place, so storing the live object would make undo a no-op.
    # moves_history is the flat move list for PGN and the same rewind.
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

        # Mate and every draw end the game. Check is only a notice.
        if game.is_checkmate():
            winner = "White" if game.get_winner() == "w" else "Black"
            print(f"Checkmate! {winner} wins.")
            return
        announcement = game.draw_announcement()
        if announcement is not None:
            print(announcement)
            return
        if game.is_check():
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
                selected = game._select_move(text)
            except ValueError as error:
                print(f"Invalid move: {error}")
                continue
            game = game.after(selected)
            moves_history.append(selected)
            history.append(game)
            print(f"You played: {text}")
        else:
            # Engine turn: the search is a pure query, so choose_move
            # runs first and the move is only applied afterwards.
            selected = search_engine.choose_move(game, depth=args.depth)
            game = game.after(selected)
            moves_history.append(selected)
            history.append(game)
            print(f"Engine: {move_notation(selected)}")


if __name__ == "__main__":
    try:
        main()
    except EOFError, KeyboardInterrupt:
        print("\nGoodbye!")

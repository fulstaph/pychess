"""Play chess against an alpha-beta minimax engine using the shared game rules.

Run with ``python -m chess.engine``.
"""

import argparse
import sys
from collections.abc import Sequence

from .game import ChessGame, GameStatus, print_board
from .helpers import square_notation
from .move import Move
from .piece import Board, Color, PieceType

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
    """Calculate Most Valuable Victim - Least Valuable Attacker move score."""
    score = 0
    if move.captured_piece is not None:
        victim = PIECE_VALUES.get(move.captured_piece.type, 0)
        attacker = PIECE_VALUES.get(move.piece.type, 0)
        score += 10000 + (victim * 10 - attacker)
    if move.special == "promotion":
        score += 9000
    return score


def order_moves(moves: Sequence[Move]) -> list[Move]:
    """Order moves to maximize alpha-beta pruning efficiency."""
    return sorted(moves, key=mvv_lva_score, reverse=True)


def _quiescence(game: ChessGame, alpha: int, beta: int, qdepth: int = 3) -> int:
    """Search capture moves at horizon depth to prevent the horizon effect."""
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
    """Evaluate position with fixed material and central bonuses (white-positive)."""
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
    """Select best move using alpha-beta minimax with move ordering."""
    if depth < 1:
        raise ValueError("Search depth must be at least 1")
    moves = game.legal_moves()
    if not moves:
        raise ValueError("No legal moves available")

    best_move: Move = moves[0]
    alpha = -1_000_000
    beta = 1_000_000
    ordered = order_moves(moves)

    if game.turn == "w":
        best_score = -1_000_000
        for move in ordered:
            score = _minimax(
                game.after(move),
                depth - 1,
                alpha,
                beta,
                ply=1,
                quiescence=quiescence,
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
            score = _minimax(
                game.after(move),
                depth - 1,
                alpha,
                beta,
                ply=1,
                quiescence=quiescence,
            )
            if score < best_score:
                best_score = score
                best_move = move
            if score < beta:
                beta = score
            if alpha >= beta:
                break

    return best_move


def move_notation(move: Move) -> str:
    """Display an engine move in coordinate notation accepted by ChessGame."""
    source = square_notation(*move.from_square)
    target = square_notation(*move.to_square)
    promotion = f"={move.promotion_to.value.upper()}" if move.promotion_to else ""
    return f"{source}{target}{promotion}"


def main(argv: Sequence[str] | None = None) -> None:
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
    args = parser.parse_args(argv)

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
        from .stockfish import Stockfish, play_human_vs_stockfish

        try:
            with Stockfish(path=args.stockfish_path) as sf:
                play_human_vs_stockfish(sf, human_color=human_color)
        except FileNotFoundError as err:
            sys.stderr.write(f"Error: {err}\n")
            sys.exit(1)
        return
    game = ChessGame()
    history: list[ChessGame] = [game]
    moves_history: list[Move] = []
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
                )
            )
        else:
            print_board(
                game.board,
                unicode_pieces=args.unicode,
                ansi_colors=args.color_board,
            )
            print(f"Status: {game.status.value}")

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
            history.append(game)
            print(f"You played: {text}")
        else:
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

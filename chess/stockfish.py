"""Stockfish engine integration and tournament match runner."""

import argparse
import os
import shutil
import sys
from collections.abc import Callable, Sequence
from dataclasses import dataclass

from .engine import choose_move
from .game import ChessGame, GameStatus, print_board
from .move import Move
from .piece import Color
from .uci import UCIEngine, UCIEngineError, to_uci

type PlayerCallable = Callable[[ChessGame, Sequence[Move]], Move | str]


def _other(color: Color) -> Color:
    return "b" if color == "w" else "w"


def find_stockfish(custom_path: str | None = None) -> str:
    """Locate the Stockfish chess engine executable on the host system."""
    if custom_path is not None:
        expanded = os.path.expanduser(custom_path)
        if os.path.isfile(expanded) and os.access(expanded, os.X_OK):
            return os.path.abspath(expanded)
        resolved = shutil.which(custom_path)
        if resolved:
            return os.path.abspath(resolved)
        raise FileNotFoundError(
            f"Stockfish executable not found at specified path: {custom_path}"
        )

    env_path = os.environ.get("STOCKFISH_PATH")
    if env_path:
        expanded = os.path.expanduser(env_path)
        if os.path.isfile(expanded) and os.access(expanded, os.X_OK):
            return os.path.abspath(expanded)
        resolved = shutil.which(env_path)
        if resolved:
            return os.path.abspath(resolved)
        raise FileNotFoundError(
            f"Stockfish executable specified by STOCKFISH_PATH not found: {env_path}"
        )
    which_path = shutil.which("stockfish")
    if which_path:
        return os.path.abspath(which_path)

    standard_candidates = [
        "/opt/homebrew/bin/stockfish",
        "/usr/local/bin/stockfish",
        "/usr/games/stockfish",
        "/usr/bin/stockfish",
        shutil.which("stockfish.exe") or "C:\\Program Files\\Stockfish\\stockfish.exe",
    ]
    for candidate in standard_candidates:
        if candidate and os.path.isfile(candidate) and os.access(candidate, os.X_OK):
            return os.path.abspath(candidate)

    raise FileNotFoundError(
        "Stockfish executable could not be found. Please install Stockfish "
        "(e.g., 'brew install stockfish' or 'apt install stockfish') or specify "
        "its location using the --stockfish-path argument or STOCKFISH_PATH "
        "environment variable."
    )


class Stockfish(UCIEngine):
    """Stockfish chess engine interface implementing UCI protocol."""

    def __init__(
        self,
        path: str | None = None,
        skill_level: int | None = None,
        threads: int = 1,
        hash_mb: int = 16,
        args: list[str] | None = None,
    ) -> None:
        resolved_path = find_stockfish(path)
        if skill_level is not None and not (0 <= skill_level <= 20):
            raise ValueError(f"Skill level must be between 0 and 20, got {skill_level}")
        super().__init__(binary_path=resolved_path, args=args)
        self.set_option("Threads", threads)
        self.set_option("Hash", hash_mb)
        if skill_level is not None:
            self.set_option("Skill Level", skill_level)

    def get_move(
        self,
        game: ChessGame,
        moves_history: Sequence[Move | str] | None = None,
        movetime_ms: int = 100,
        depth: int | None = None,
    ) -> Move:
        """Query Stockfish for the best move in the given position."""
        if moves_history is not None:
            uci_moves = [
                to_uci(m) if isinstance(m, Move) else str(m) for m in moves_history
            ]
            self.set_position(moves=uci_moves)
        else:
            self.set_position(fen=game.to_fen())

        uci_str = self.go(movetime_ms=movetime_ms, depth=depth)
        return game._select_move(uci_str)


@dataclass(frozen=True, slots=True)
class MatchResult:
    """Outcome and moves of a single game between two engines or players."""

    white_name: str
    black_name: str
    winner: Color | None
    status: GameStatus
    termination: str
    moves: tuple[Move, ...]
    uci_moves: tuple[str, ...] = ()
    san_moves: tuple[str, ...] = ()
    halfmove_count: int = 0


@dataclass(frozen=True, slots=True)
class SeriesResult:
    """Aggregated statistics from a tournament match series."""

    games: tuple[MatchResult, ...]
    scores: dict[str, float]
    wins: dict[str, int]
    draws: int
    white_wins: int
    black_wins: int
    average_moves: float


def format_series_summary(result: SeriesResult) -> str:
    """Format series results into an ASCII tournament summary table."""
    total_games = len(result.games)
    s = "s" if total_games != 1 else ""
    lines: list[str] = [
        "=" * 60,
        f"Tournament Match Summary ({total_games} game{s})",
        "-" * 60,
    ]
    for engine, score in result.scores.items():
        wins = result.wins.get(engine, 0)
        draws = result.draws
        losses = total_games - wins - draws
        pct = (score / total_games * 100.0) if total_games > 0 else 0.0
        lines.append(
            f"{engine}: {score:.1f} / {total_games} ({pct:.1f}%)  "
            f"[{wins} wins, {losses} losses, {draws} draws]"
        )
    lines.append("")
    lines.append("Stats:")
    lines.append(
        f"  White wins: {result.white_wins} | Black wins: {result.black_wins} "
        f"| Draws: {result.draws}"
    )
    lines.append(f"  Average moves per game: {result.average_moves:.1f}")

    terminations: dict[str, int] = {}
    for g in result.games:
        terminations[g.termination] = terminations.get(g.termination, 0) + 1
    term_str = ", ".join(f"{count} {term}" for term, count in terminations.items())
    lines.append(f"  Terminations: {term_str}")
    lines.append("=" * 60)
    return "\n".join(lines)


def play_match(
    white_player: PlayerCallable,
    black_player: PlayerCallable,
    white_name: str = "White",
    black_name: str = "Black",
    max_moves: int = 150,
    on_move: Callable[[ChessGame, Move, str], None] | None = None,
) -> MatchResult:
    """Play a complete game between two players or engines."""
    game = ChessGame()
    played_moves: list[Move] = []
    uci_moves: list[str] = []
    san_moves: list[str] = []

    winner: Color | None = None
    termination = "unknown"

    while True:
        if game.status == GameStatus.CHECKMATE:
            winner = game.get_winner()
            termination = "checkmate"
            break
        if game.status == GameStatus.STALEMATE:
            winner = None
            termination = "stalemate"
            break
        if len(played_moves) >= max_moves * 2:
            winner = None
            termination = f"draw by move limit ({max_moves} moves)"
            break

        current_color = game.turn
        current_player = white_player if current_color == "w" else black_player
        current_name = white_name if current_color == "w" else black_name
        opponent_color: Color = _other(current_color)

        try:
            chosen = current_player(game, tuple(played_moves))
        except Exception as exc:  # noqa: BLE001
            winner = opponent_color
            termination = f"{current_name} forfeited ({exc})"
            break

        try:
            executed = game.make_move(chosen)
        except ValueError as exc:
            winner = opponent_color
            termination = f"{current_name} illegal move ({exc})"
            break

        played_moves.append(executed)
        uci_str = to_uci(executed)
        uci_moves.append(uci_str)
        san_moves.append(uci_str)

        if on_move is not None:
            on_move(game, executed, uci_str)

    return MatchResult(
        white_name=white_name,
        black_name=black_name,
        winner=winner,
        status=game.status,
        termination=termination,
        moves=tuple(played_moves),
        uci_moves=tuple(uci_moves),
        san_moves=tuple(san_moves),
        halfmove_count=len(played_moves),
    )


def run_series(
    stockfish: Stockfish,
    games: int = 2,
    movetime_ms: int = 100,
    depth: int = 2,
    max_moves: int = 150,
    engine_name: str = "pychess",
    stockfish_name: str = "Stockfish",
    on_game_start: Callable[[int, str, str], None] | None = None,
    on_game_end: Callable[[int, MatchResult], None] | None = None,
    on_move: Callable[[ChessGame, Move, str], None] | None = None,
) -> SeriesResult:
    """Run a tournament series alternating colors between pychess and Stockfish."""
    if games < 1:
        raise ValueError("Number of games must be at least 1")

    match_results: list[MatchResult] = []

    def pychess_player(g: ChessGame, _history: Sequence[Move]) -> Move:
        return choose_move(g, depth=depth)

    def stockfish_player(g: ChessGame, history: Sequence[Move]) -> Move:
        return stockfish.get_move(g, moves_history=history, movetime_ms=movetime_ms)

    for i in range(1, games + 1):
        stockfish.new_game()
        w_player: PlayerCallable
        b_player: PlayerCallable
        if i % 2 == 1:
            w_player = pychess_player
            b_player = stockfish_player
            w_name = engine_name
            b_name = stockfish_name
        else:
            w_player = stockfish_player
            b_player = pychess_player
            w_name = stockfish_name
            b_name = engine_name

        if on_game_start is not None:
            on_game_start(i, w_name, b_name)

        result = play_match(
            white_player=w_player,
            black_player=b_player,
            white_name=w_name,
            black_name=b_name,
            max_moves=max_moves,
            on_move=on_move,
        )
        match_results.append(result)

        if on_game_end is not None:
            on_game_end(i, result)

    scores = {engine_name: 0.0, stockfish_name: 0.0}
    wins = {engine_name: 0, stockfish_name: 0}
    draws = 0
    white_wins = 0
    black_wins = 0

    for r in match_results:
        if r.winner == "w":
            white_wins += 1
            scores[r.white_name] += 1.0
            wins[r.white_name] += 1
        elif r.winner == "b":
            black_wins += 1
            scores[r.black_name] += 1.0
            wins[r.black_name] += 1
        else:
            draws += 1
            scores[engine_name] += 0.5
            scores[stockfish_name] += 0.5

    avg_moves = (
        sum((len(r.moves) + 1) // 2 for r in match_results) / float(games)
        if games > 0
        else 0.0
    )

    return SeriesResult(
        games=tuple(match_results),
        scores=scores,
        wins=wins,
        draws=draws,
        white_wins=white_wins,
        black_wins=black_wins,
        average_moves=avg_moves,
    )


def play_human_vs_stockfish(
    stockfish: Stockfish,
    human_color: Color = "w",
    movetime_ms: int = 100,
    depth: int | None = None,
) -> MatchResult:
    """Interactive terminal game loop for human vs Stockfish."""
    game = ChessGame()
    played_moves: list[Move] = []
    stockfish.new_game()

    human_name = "Human"
    sf_name = "Stockfish"
    white_name = human_name if human_color == "w" else sf_name
    black_name = sf_name if human_color == "w" else human_name

    while True:
        print_board(game.board)
        print(f"Status: {game.status.value}")
        if game.status == GameStatus.CHECKMATE:
            winner_side = game.get_winner()
            winner_str = "White" if winner_side == "w" else "Black"
            print(f"Checkmate! {winner_str} wins.")
            return MatchResult(
                white_name=white_name,
                black_name=black_name,
                winner=winner_side,
                status=game.status,
                termination="checkmate",
                moves=tuple(played_moves),
                halfmove_count=len(played_moves),
            )
        if game.status == GameStatus.STALEMATE:
            print("Stalemate! Draw.")
            return MatchResult(
                white_name=white_name,
                black_name=black_name,
                winner=None,
                status=game.status,
                termination="stalemate",
                moves=tuple(played_moves),
                halfmove_count=len(played_moves),
            )
        if game.status == GameStatus.CHECK:
            print("Check!")

        if game.turn == human_color:
            text = input("Your move: ").strip()
            if text.lower() == "q":
                print("Goodbye!")
                return MatchResult(
                    white_name=white_name,
                    black_name=black_name,
                    winner=_other(human_color),
                    status=game.status,
                    termination="resignation",
                    moves=tuple(played_moves),
                    halfmove_count=len(played_moves),
                )
            try:
                selected = game.make_move(text)
            except ValueError as error:
                print(f"Invalid move: {error}")
                continue
            played_moves.append(selected)
            print(f"You played: {text}")
        else:
            selected = stockfish.get_move(
                game,
                moves_history=played_moves,
                movetime_ms=movetime_ms,
                depth=depth,
            )
            game.make_move(selected)
            played_moves.append(selected)
            print(f"Stockfish: {to_uci(selected)}")


def main(argv: Sequence[str] | None = None) -> None:
    """CLI entrypoint for running matches against Stockfish."""
    parser = argparse.ArgumentParser(
        prog="python -m chess.stockfish",
        description="Run matches or tournaments against Stockfish.",
    )
    parser.add_argument(
        "--games",
        "-n",
        type=int,
        default=2,
        help="Number of tournament games to play (default: 2)",
    )
    parser.add_argument(
        "--skill-level",
        "-s",
        type=int,
        default=0,
        help="Stockfish skill level 0-20 (default: 0)",
    )
    parser.add_argument(
        "--movetime",
        "-t",
        type=int,
        default=100,
        help="Stockfish search time in ms (default: 100)",
    )
    parser.add_argument(
        "--depth",
        "-d",
        type=int,
        default=2,
        help="Pychess minimax depth (default: 2)",
    )
    parser.add_argument(
        "--stockfish-path",
        "-p",
        type=str,
        default=None,
        help="Path to Stockfish executable",
    )
    parser.add_argument(
        "--max-moves",
        "-m",
        type=int,
        default=150,
        help="Maximum moves per game before draw (default: 150)",
    )
    parser.add_argument(
        "--human",
        action="store_true",
        help="Play interactively as human against Stockfish",
    )
    parser.add_argument(
        "--color",
        "-c",
        choices=["w", "b"],
        default="w",
        help="Human player color (w or b, default: w)",
    )
    parser.add_argument(
        "--quiet",
        "-q",
        action="store_true",
        help="Suppress move logging, only show game outcomes and summary",
    )

    args = parser.parse_args(argv)

    if args.games < 1:
        parser.error("Number of games must be at least 1")

    try:
        sf = Stockfish(
            path=args.stockfish_path,
            skill_level=args.skill_level,
        )
    except FileNotFoundError as err:
        sys.stderr.write(f"Error: {err}\n")
        sys.exit(1)
    except UCIEngineError as err:
        sys.stderr.write(f"UCI Error: {err}\n")
        sys.exit(1)

    with sf:
        if args.human:
            try:
                play_human_vs_stockfish(
                    sf,
                    human_color=args.color,
                    movetime_ms=args.movetime,
                )
            except KeyboardInterrupt, EOFError:
                print("\nGoodbye!")
                sys.exit(0)
            return

        def on_game_end(idx: int, result: MatchResult) -> None:
            if result.winner == "w":
                score_str = "1-0"
            elif result.winner == "b":
                score_str = "0-1"
            else:
                score_str = "1/2-1/2"
            moves_count = (len(result.moves) + 1) // 2
            print(
                f"Game {idx}/{args.games}: {result.white_name} (White) vs "
                f"{result.black_name} (Black) -> {score_str} "
                f"({result.termination}, {moves_count} moves)"
            )

        def on_move(_g: ChessGame, _m: Move, uci_str: str) -> None:
            if not args.quiet:
                sys.stdout.write(f"{uci_str} ")
                sys.stdout.flush()

        try:
            series_result = run_series(
                stockfish=sf,
                games=args.games,
                movetime_ms=args.movetime,
                depth=args.depth,
                max_moves=args.max_moves,
                on_game_end=on_game_end,
                on_move=on_move if not args.quiet else None,
            )
        except KeyboardInterrupt, EOFError:
            print("\nTournament interrupted.")
            sys.exit(0)

        print()
        print(format_series_summary(series_result))


if __name__ == "__main__":
    main()

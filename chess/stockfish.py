"""Stockfish engine integration and tournament match runner.

Architecture
------------
This module sits on top of :mod:`chess.uci` and provides three layers:

1. **Discovery** (``find_stockfish``): resolves the Stockfish binary in
   the order ``--stockfish-path``/constructor path → ``STOCKFISH_PATH``
   environment variable → ``shutil.which("stockfish")`` → well-known
   install locations (Homebrew, /usr/local, /usr/games, /usr/bin).  All
   lookups accept ``~`` expansion and return an absolute path.

2. **Engine wrapper** (``Stockfish``): a thin ``UCIEngine`` subclass
   that spawns the process once, then configures ``Threads``, ``Hash``,
   and optionally ``Skill Level`` (0-20, where low values degrade the
   engine for casual play).  ``get_move`` replays the move history as
   UCI strings (or a FEN when no history is supplied), runs ``go`` with
   a movetime/depth budget, and converts the returned UCI move back
   into a validated ``Move`` via the game's own parser.

3. **Match orchestration** (``play_match``, ``run_series``,
   ``play_human_vs_stockfish``, ``main``): players are plain
   ``PlayerCallable`` functions, so a human, the built-in minimax, and
   Stockfish are interchangeable sides.  ``run_series`` alternates
   colors across games and aggregates wins/draws into a ``SeriesResult``.

Error model
-----------
``find_stockfish`` raises ``FileNotFoundError``; ``Stockfish.__init__``
raises ``ValueError`` for an out-of-range skill level and
``UCIEngineError`` for spawn/handshake failures.  ``play_match``
never raises for in-game failures: an exception from a player, or an
illegal move, forfeits the game to the opponent and is recorded in
``MatchResult.termination``.

Logging
-------
The module logger ``chess.stockfish`` logs the user-facing match flow
at INFO (spawn, game start/end, results, moves played), degradations
and rejected input at WARNING, and fatal engine/discovery failures at
ERROR.  The ``main`` entrypoint is the only place that calls
``setup_logging``; library callers never configure the root logger.
"""

import argparse
import os
import shutil
import sys
from collections.abc import Callable, Sequence
from dataclasses import dataclass

from .game import ChessGame, GameStatus, print_board
from .log import get_logger, setup_logging
from .move import Move
from .piece import Color
from .search import DEFAULT_ENGINE_CONFIG, MAX_SEARCH_DEPTH, EngineConfig, SearchEngine
from .uci import UCIEngine, UCIEngineError, to_uci

logger = get_logger("stockfish")

type PlayerCallable = Callable[[ChessGame, Sequence[Move]], Move | str]


def _other(color: Color) -> Color:
    return "b" if color == "w" else "w"


def find_stockfish(custom_path: str | None = None) -> str:
    """Locate the Stockfish chess engine executable on the host system.

    Resolution order: explicit path argument, ``STOCKFISH_PATH``
    environment variable, ``PATH`` lookup, then well-known install
    directories.  Raises ``FileNotFoundError`` with an actionable
    message when nothing resolves.
    """
    if custom_path is not None:
        # Explicit paths are trusted first: expanduser supports ~, and
        # an isfile+X_OK check rejects typos before they reach spawn.
        expanded = os.path.expanduser(custom_path)
        if os.path.isfile(expanded) and os.access(expanded, os.X_OK):
            return os.path.abspath(expanded)
        resolved = shutil.which(custom_path)
        if resolved:
            return os.path.abspath(resolved)
        logger.error("Explicit Stockfish path %r not found", custom_path)
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
        logger.error("STOCKFISH_PATH %r not found", env_path)
        raise FileNotFoundError(
            f"Stockfish executable specified by STOCKFISH_PATH not found: {env_path}"
        )
    which_path = shutil.which("stockfish")
    if which_path:
        return os.path.abspath(which_path)

    # Last resort: probe well-known install locations (macOS Homebrew,
    # Linux distro paths, Windows).  The shutil.which fallback keeps the
    # list from hardcoding a non-existent Windows path on other OSes.
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

    logger.error("Stockfish executable could not be found on this system")
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
        logger.info(
            "Stockfish configured: %s (skill=%s, threads=%d, hash=%dMB)",
            resolved_path,
            skill_level,
            threads,
            hash_mb,
        )

    def get_move(
        self,
        game: ChessGame,
        moves_history: Sequence[Move | str] | None = None,
        movetime_ms: int | None = 100,
        depth: int | None = None,
    ) -> Move:
        """Query Stockfish for the best move in the given position.

        Positions are communicated by replaying the move history (the
        cheap, allocation-free path for live games) or by FEN when the
        caller has no history.  The answer is validated through
        ``game._select_move`` so an engine hallucination surfaces as a
        ``ValueError`` instead of corrupting the game state.
        """
        if moves_history is not None:
            uci_moves = [
                to_uci(m) if isinstance(m, Move) else str(m) for m in moves_history
            ]
            self.set_position(moves=uci_moves)
        else:
            self.set_position(fen=game.to_fen())

        uci_str = self.go(movetime_ms=movetime_ms, depth=depth)
        logger.debug("Stockfish -> %s", uci_str)
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
    """Play a complete game between two players or engines.

    The loop is strictly turn-based on ``game.turn``: each side's
    ``PlayerCallable`` receives the current game plus the full move
    history and must return a ``Move`` or coordinate string.  Any
    failure (engine exception, illegal move) forfeits the game to the
    opponent rather than propagating, so a tournament series always
    yields a result for every scheduled game.
    """
    game = ChessGame()
    played_moves: list[Move] = []
    uci_moves: list[str] = []
    san_moves: list[str] = []

    winner: Color | None = None
    termination = "unknown"

    logger.info("Match start: %s vs %s", white_name, black_name)

    while True:
        if game.is_checkmate():
            winner = game.get_winner()
            termination = "checkmate"
            break
        reason = game.draw_reason()
        if reason is not None:
            winner = None
            termination = reason
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
            # A crashing player loses on forfeit: the match runner must
            # stay deterministic even if one side's engine dies mid-game.
            logger.warning("%s forfeited: %s", current_name, exc)
            winner = opponent_color
            termination = f"{current_name} forfeited ({exc})"
            break

        try:
            executed = game.make_move(chosen)
        except ValueError as exc:
            # Same contract as above, but for a *well-behaved* player
            # that simply proposed an illegal move (e.g. engine bug).
            logger.warning("%s played illegal move: %s", current_name, exc)
            winner = opponent_color
            termination = f"{current_name} illegal move ({exc})"
            break

        played_moves.append(executed)
        uci_str = to_uci(executed)
        uci_moves.append(uci_str)
        san_moves.append(uci_str)
        logger.debug("move %d: %s", len(played_moves), uci_str)

        if on_move is not None:
            on_move(game, executed, uci_str)

    result = MatchResult(
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
    winner_str = _color_name(result.winner) if result.winner else "draw"
    logger.info(
        "Match end: %s (%s) — %d moves", winner_str, termination, len(played_moves)
    )
    return result


def _color_name(color: Color | None) -> str:
    return {"w": "White", "b": "Black"}.get(color or "", "None")


def run_series(
    stockfish: Stockfish,
    games: int = 2,
    movetime_ms: int = 100,
    depth: int = DEFAULT_ENGINE_CONFIG.search_depth,
    max_moves: int = 150,
    engine_name: str = "pychess",
    stockfish_name: str = "Stockfish",
    on_game_start: Callable[[int, str, str], None] | None = None,
    on_game_end: Callable[[int, MatchResult], None] | None = None,
    on_move: Callable[[ChessGame, Move, str], None] | None = None,
) -> SeriesResult:
    """Run a tournament series alternating colors between pychess and Stockfish.

    Odd games put the built-in engine on White; even games flip colors,
    so each side plays one of each color.  ``stockfish.new_game()`` is
    called between games to clear its search state.  Aggregation
    awards 1.0 / 0.5 / 0.0 points per win / draw / loss.
    """
    if games < 1:
        raise ValueError("Number of games must be at least 1")

    match_results: list[MatchResult] = []
    search_engine = SearchEngine(EngineConfig(search_depth=depth))
    logger.info(
        "Series start: %d game(s), pychess depth=%d, stockfish movetime=%dms",
        games,
        depth,
        movetime_ms,
    )

    def pychess_player(g: ChessGame, _history: Sequence[Move]) -> Move:
        return search_engine.choose_move(g, depth=depth)

    def stockfish_player(g: ChessGame, history: Sequence[Move]) -> Move:
        return stockfish.get_move(g, moves_history=history, movetime_ms=movetime_ms)

    for i in range(1, games + 1):
        stockfish.new_game()
        search_engine.clear()
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

    series = SeriesResult(
        games=tuple(match_results),
        scores=scores,
        wins=wins,
        draws=draws,
        white_wins=white_wins,
        black_wins=black_wins,
        average_moves=avg_moves,
    )
    logger.info(
        "Series end: %s %.1f — %s %.1f (%d draws)",
        engine_name,
        scores[engine_name],
        stockfish_name,
        scores[stockfish_name],
        draws,
    )
    return series


def play_human_vs_stockfish(
    stockfish: Stockfish,
    human_color: Color = "w",
    movetime_ms: int = 100,
    depth: int | None = None,
) -> MatchResult:
    """Interactive terminal game loop for human vs Stockfish.

    Blocks on ``input()`` for the human side; the engine side queries
    Stockfish with the shared ``movetime_ms`` budget.  Typing ``q``
    resigns immediately.  Invalid human input prints an error and
    re-prompts without ending the game.
    """
    game = ChessGame()
    played_moves: list[Move] = []
    stockfish.new_game()

    human_name = "Human"
    sf_name = "Stockfish"
    white_name = human_name if human_color == "w" else sf_name
    black_name = sf_name if human_color == "w" else human_name
    logger.info(
        "Human vs Stockfish: human plays %s, movetime=%dms", human_color, movetime_ms
    )

    while True:
        print_board(game.board)
        print(f"Status: {game.status.value}")
        if game.is_checkmate():
            winner_side = game.get_winner()
            winner_str = "White" if winner_side == "w" else "Black"
            print(f"Checkmate! {winner_str} wins.")
            logger.info("Game ended: checkmate, %s wins", winner_str)
            return MatchResult(
                white_name=white_name,
                black_name=black_name,
                winner=winner_side,
                status=game.status,
                termination="checkmate",
                moves=tuple(played_moves),
                halfmove_count=len(played_moves),
            )
        announcement = game.draw_announcement()
        if announcement is not None:
            print(announcement)
            reason = game.draw_reason() or "draw"
            logger.info("Game ended: %s", reason)
            return MatchResult(
                white_name=white_name,
                black_name=black_name,
                winner=None,
                status=game.status,
                termination=reason,
                moves=tuple(played_moves),
                halfmove_count=len(played_moves),
            )
        if game.is_check():
            print("Check!")

        if game.turn == human_color:
            text = input("Your move: ").strip()
            if text.lower() == "q":
                print("Goodbye!")
                logger.info("Human resigned")
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
                # Invalid input is not game-ending: print the reason and
                # let the human retry from the same position.
                logger.warning("Rejected human input %r: %s", text, error)
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
    setup_logging()
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
        choices=range(1, MAX_SEARCH_DEPTH + 1),
        default=DEFAULT_ENGINE_CONFIG.search_depth,
        help=f"Pychess search depth (1-{MAX_SEARCH_DEPTH})",
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
        logger.error("Stockfish not found: %s", err)
        sys.stderr.write(f"Error: {err}\n")
        sys.exit(1)
    except UCIEngineError as err:
        logger.error("UCI engine failure: %s", err)
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
                logger.info("Human game interrupted")
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
            logger.warning("Tournament interrupted by user")
            print("\nTournament interrupted.")
            sys.exit(0)

        print()
        print(format_series_summary(series_result))


if __name__ == "__main__":
    main()

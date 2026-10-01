"""Run a deterministic, fixed-position search benchmark from the CLI."""

import argparse
from collections.abc import Sequence

from .engine import move_notation
from .game import ChessGame
from .search import (
    DEFAULT_ENGINE_CONFIG,
    MAX_SEARCH_DEPTH,
    EngineConfig,
    EvaluationProfile,
    SearchEngine,
)

BENCHMARK_POSITIONS: tuple[tuple[str, str], ...] = (
    (
        "start",
        "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
    ),
    (
        "tactical",
        "r1bqk2r/pppp1ppp/2n5/4p3/2B1n3/5N2/PPPP1PPP/RNBQK2R w KQkq - 0 5",
    ),
    (
        "middlegame",
        "r2q1rk1/pp1n1ppp/2p1pn2/3p4/2PP4/1PN1PN2/PBQ2PPP/R3KB1R w KQ - 0 10",
    ),
    (
        "endgame",
        "8/5pk1/6p1/8/8/6P1/5PK1/8 w - - 0 1",
    ),
)


def run_benchmark(
    depth: int = DEFAULT_ENGINE_CONFIG.search_depth,
    repeats: int = 3,
    profile: EvaluationProfile = DEFAULT_ENGINE_CONFIG.evaluation_profile,
) -> None:
    """Print cold-table node/time samples for each fixed benchmark position."""
    if type(depth) is not int or not 1 <= depth <= MAX_SEARCH_DEPTH:
        raise ValueError(f"Depth must be between 1 and {MAX_SEARCH_DEPTH}")
    if type(repeats) is not int or repeats < 1:
        raise ValueError("Repeats must be at least 1")
    config = EngineConfig(search_depth=depth, evaluation_profile=profile)
    for name, fen in BENCHMARK_POSITIONS:
        for run in range(1, repeats + 1):
            engine = SearchEngine(config)
            move = engine.choose_move(ChessGame.from_fen(fen))
            stats = engine.last_stats
            if stats is None:
                raise RuntimeError("Search completed without statistics")
            print(
                f"{name} run={run} depth={stats.completed_depth} "
                f"move={move_notation(move)} nodes={stats.nodes} "
                f"qnodes={stats.quiescence_nodes} "
                f"tt_hits={stats.transposition_hits} "
                f"elapsed_ms={stats.elapsed_ms:.1f} nps={stats.nodes_per_second}"
            )


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--depth",
        type=int,
        choices=range(1, MAX_SEARCH_DEPTH + 1),
        default=DEFAULT_ENGINE_CONFIG.search_depth,
    )
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument(
        "--profile",
        choices=("basic", "positional"),
        default=DEFAULT_ENGINE_CONFIG.evaluation_profile,
    )
    args = parser.parse_args(argv)
    run_benchmark(args.depth, args.repeats, args.profile)


if __name__ == "__main__":
    main()

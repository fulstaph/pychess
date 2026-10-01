"""Run balanced, instrumented matches between evaluator profiles and Stockfish."""

import argparse
import json
import logging
import os
import random
import subprocess
import sys
import tempfile
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from time import perf_counter
from typing import Literal

from .game import ChessGame
from .log import setup_logging
from .move import Move
from .piece import Color
from .search import MAX_SEARCH_DEPTH, EngineConfig, EvaluationProfile, SearchEngine
from .stockfish import MatchResult, PlayerCallable, Stockfish, play_match
from .uci import UCIEngineError, to_uci

MatchMode = Literal["depth", "time"]
PROFILES: tuple[EvaluationProfile, ...] = ("basic", "positional")
COLORS: tuple[Color, ...] = ("w", "b")


@dataclass(frozen=True, slots=True)
class OpeningLine:
    name: str
    san_moves: tuple[str, ...]


OPENING_LINES: tuple[OpeningLine, ...] = (
    OpeningLine("open_game", ("e4", "e5", "Nf3", "Nc6")),
    OpeningLine("queens_gambit", ("d4", "d5", "c4", "e6", "Nc3", "Nf6")),
    OpeningLine("sicilian", ("e4", "c5", "Nf3", "d6", "d4", "cxd4", "Nxd4", "Nf6")),
    OpeningLine("english", ("c4", "e5", "Nc3", "Nf6", "g3", "d5", "cxd5", "Nxd5")),
)


@dataclass(frozen=True, slots=True)
class ScheduledMatch:
    pair_id: str
    profile: EvaluationProfile
    stockfish_skill_level: int
    opening: OpeningLine
    repeat: int
    pychess_color: Color


@dataclass(frozen=True, slots=True)
class MatchBenchConfig:
    """Settings shared by every game in one experiment."""

    mode: MatchMode = "time"
    skill_levels: tuple[int, ...] = (0, 5)
    repeats: int = 4
    seed: int = 211
    max_moves: int = 100
    depth: int = 3
    time_ms: int = 100

    def __post_init__(self) -> None:
        if self.mode not in ("depth", "time"):
            raise ValueError("Mode must be 'depth' or 'time'")
        if not self.skill_levels:
            raise ValueError("At least one Stockfish skill level is required")
        if any(
            type(level) is not int or not 0 <= level <= 20
            for level in self.skill_levels
        ):
            raise ValueError("Stockfish skill levels must be integers from 0 to 20")
        if len(set(self.skill_levels)) != len(self.skill_levels):
            raise ValueError("Stockfish skill levels must be unique")
        if type(self.repeats) is not int or self.repeats < 1:
            raise ValueError("Repeats must be at least 1")
        if type(self.seed) is not int:
            raise ValueError("Seed must be an integer")
        if type(self.max_moves) is not int or self.max_moves < 1:
            raise ValueError("Maximum moves must be at least 1")
        if type(self.depth) is not int or not 1 <= self.depth <= MAX_SEARCH_DEPTH:
            raise ValueError(f"Depth must be between 1 and {MAX_SEARCH_DEPTH}")
        if type(self.time_ms) is not int or self.time_ms < 1:
            raise ValueError("Time budget must be a positive number of milliseconds")


def build_schedule(
    skill_levels: Sequence[int], repeats: int, seed: int
) -> tuple[ScheduledMatch, ...]:
    """Create a seeded schedule paired on opening, repeat, skill, and color."""
    config = MatchBenchConfig(
        skill_levels=tuple(skill_levels), repeats=repeats, seed=seed
    )
    schedule: list[ScheduledMatch] = []
    for skill_level in config.skill_levels:
        for opening in OPENING_LINES:
            for repeat in range(1, config.repeats + 1):
                for color in COLORS:
                    pair_id = f"sf{skill_level}-{opening.name}-r{repeat}-{color}"
                    schedule.extend(
                        [
                            ScheduledMatch(
                                pair_id,
                                profile,
                                skill_level,
                                opening,
                                repeat,
                                color,
                            )
                            for profile in PROFILES
                        ]
                    )
    random.Random(config.seed).shuffle(schedule)  # noqa: S311
    return tuple(schedule)


def _stockfish_version(path: str) -> str | None:
    try:
        result = subprocess.run(  # noqa: S603
            [path, "--version"],
            capture_output=True,
            check=False,
            text=True,
            timeout=5,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    output = (result.stdout or result.stderr).strip()
    if result.returncode != 0 or not output:
        return None
    return output.splitlines()[0]


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as temporary:
            temporary_path = Path(temporary.name)
            json.dump(payload, temporary, indent=2, sort_keys=True)
            temporary.write("\n")
            temporary.flush()
            os.fsync(temporary.fileno())
        os.replace(temporary_path, path)
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)


def _match_record(
    scheduled: ScheduledMatch,
    game_id: int,
    result: MatchResult,
    ply_records: list[dict[str, object]],
) -> dict[str, object]:
    if result.winner is None:
        score = "1/2-1/2"
        profile_score = 0.5
    elif result.winner == scheduled.pychess_color:
        score = "1-0" if result.winner == "w" else "0-1"
        profile_score = 1.0
    else:
        score = "1-0" if result.winner == "w" else "0-1"
        profile_score = 0.0
    return {
        "game_id": game_id,
        "pair_id": scheduled.pair_id,
        "profile": scheduled.profile,
        "stockfish_skill_level": scheduled.stockfish_skill_level,
        "opening": scheduled.opening.name,
        "opening_san_moves": list(scheduled.opening.san_moves),
        "repeat": scheduled.repeat,
        "pychess_color": scheduled.pychess_color,
        "white": result.white_name,
        "black": result.black_name,
        "result": score,
        "pychess_score": profile_score,
        "termination": result.termination,
        "status": result.status.value,
        "ply_count": len(result.uci_moves),
        "uci_moves": list(result.uci_moves),
        "per_move": ply_records,
    }


def _make_player(
    engine_name: str,
    scheduled: ScheduledMatch,
    ply_records: list[dict[str, object]],
    search_engine: SearchEngine,
    search_depth: int,
    config: MatchBenchConfig,
    stockfish: Stockfish,
) -> PlayerCallable:
    def player(game: ChessGame, history: Sequence[Move]) -> Move:
        ply = len(history) + 1
        if ply <= len(scheduled.opening.san_moves):
            move = game._select_move(scheduled.opening.san_moves[ply - 1])
            ply_records.append(
                {
                    "ply": ply,
                    "side": game.turn,
                    "engine": engine_name,
                    "source": "opening",
                    "uci": to_uci(move),
                    "elapsed_ms": 0.0,
                    "completed_depth": None,
                    "nodes": None,
                    "quiescence_nodes": None,
                    "score_white_cp": None,
                }
            )
            return move

        started = perf_counter()
        if engine_name == "stockfish":
            move = stockfish.get_move(
                game,
                moves_history=history,
                movetime_ms=config.time_ms if config.mode == "time" else None,
                depth=config.depth if config.mode == "depth" else None,
            )
            elapsed_ms = (perf_counter() - started) * 1000
            ply_records.append(
                {
                    "ply": ply,
                    "side": game.turn,
                    "engine": engine_name,
                    "source": "search",
                    "uci": to_uci(move),
                    "elapsed_ms": elapsed_ms,
                    "completed_depth": None,
                    "nodes": None,
                    "quiescence_nodes": None,
                    "score_white_cp": None,
                }
            )
            return move

        move = search_engine.choose_move(
            game,
            depth=search_depth,
            time_limit_ms=config.time_ms if config.mode == "time" else None,
        )
        elapsed_ms = (perf_counter() - started) * 1000
        stats = search_engine.last_stats
        if stats is None:
            raise RuntimeError("Search completed without statistics")
        ply_records.append(
            {
                "ply": ply,
                "side": game.turn,
                "engine": engine_name,
                "source": "search",
                "uci": to_uci(move),
                "elapsed_ms": elapsed_ms,
                "search_elapsed_ms": stats.elapsed_ms,
                "completed_depth": stats.completed_depth,
                "nodes": stats.nodes,
                "quiescence_nodes": stats.quiescence_nodes,
                "score_white_cp": stats.score,
            }
        )
        return move

    return player


def run_match_benchmark(
    config: MatchBenchConfig,
    output_path: Path,
    stockfish: Stockfish,
    stockfish_version: str | None = None,
) -> dict[str, object]:
    """Play and incrementally persist the complete scheduled match set."""
    if output_path.exists():
        raise FileExistsError(
            f"Refusing to overwrite existing match data: {output_path}"
        )
    schedule = build_schedule(config.skill_levels, config.repeats, config.seed)
    stockfish_path = stockfish.binary_path
    payload: dict[str, object] = {
        "schema_version": 1,
        "complete": False,
        "started_at": datetime.now(UTC).isoformat(),
        "completed_at": None,
        "config": {
            "mode": config.mode,
            "skill_levels": list(config.skill_levels),
            "repeats": config.repeats,
            "seed": config.seed,
            "max_moves": config.max_moves,
            "depth": config.depth if config.mode == "depth" else None,
            "time_ms": config.time_ms if config.mode == "time" else None,
            "profiles": list(PROFILES),
            "quiescence": True,
            "openings": [
                {"name": opening.name, "san_moves": list(opening.san_moves)}
                for opening in OPENING_LINES
            ],
            "stockfish": {
                "path": stockfish_path,
                "version": stockfish_version,
                "threads": 1,
                "hash_mb": 16,
            },
        },
        "matches": [],
    }
    matches: list[dict[str, object]] = []
    configured_skill: int | None = None
    _write_json(output_path, payload)

    for game_id, scheduled in enumerate(schedule, start=1):
        if scheduled.stockfish_skill_level != configured_skill:
            stockfish.set_option("Skill Level", scheduled.stockfish_skill_level)
            configured_skill = scheduled.stockfish_skill_level
        stockfish.new_game()
        search_depth = config.depth if config.mode == "depth" else MAX_SEARCH_DEPTH
        search_engine = SearchEngine(
            EngineConfig(
                search_depth=search_depth,
                evaluation_profile=scheduled.profile,
            )
        )
        ply_records: list[dict[str, object]] = []

        profile_player = _make_player(
            scheduled.profile,
            scheduled,
            ply_records,
            search_engine,
            search_depth,
            config,
            stockfish,
        )
        stockfish_player = _make_player(
            "stockfish",
            scheduled,
            ply_records,
            search_engine,
            search_depth,
            config,
            stockfish,
        )
        white_name: str
        black_name: str
        if scheduled.pychess_color == "w":
            white_player, black_player = profile_player, stockfish_player
            white_name, black_name = scheduled.profile, "Stockfish"
        else:
            white_player, black_player = stockfish_player, profile_player
            white_name, black_name = "Stockfish", scheduled.profile

        result = play_match(
            white_player,
            black_player,
            white_name=white_name,
            black_name=black_name,
            max_moves=config.max_moves,
        )
        matches.append(_match_record(scheduled, game_id, result, ply_records))
        payload["matches"] = matches
        _write_json(output_path, payload)
        result_score = matches[-1]["result"]
        print(
            f"{game_id}/{len(schedule)} {scheduled.profile} vs Stockfish "
            f"skill={scheduled.stockfish_skill_level} "
            f"opening={scheduled.opening.name} color={scheduled.pychess_color} "
            f"result={result_score} termination={result.termination}"
        )

    payload["complete"] = True
    payload["completed_at"] = datetime.now(UTC).isoformat()
    _write_json(output_path, payload)
    return payload


def main(argv: Sequence[str] | None = None) -> None:
    setup_logging(logging.WARNING)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mode",
        choices=("depth", "time"),
        default="time",
        help="Shared search limit",
    )
    parser.add_argument("--depth", type=int, default=3, help="Depth used in depth mode")
    parser.add_argument(
        "--time-ms", type=int, default=100, help="Budget used in time mode"
    )
    parser.add_argument("--repeats", type=int, default=4)
    parser.add_argument("--seed", type=int, default=211)
    parser.add_argument("--max-moves", type=int, default=100)
    parser.add_argument(
        "--skill-level",
        type=int,
        action="append",
        dest="skill_levels",
        help="Stockfish skill level 0-20; repeat for multiple levels (default: 0, 5)",
    )
    parser.add_argument("--stockfish-path", type=str)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    skill_levels = tuple(args.skill_levels) if args.skill_levels else (0, 5)
    try:
        config = MatchBenchConfig(
            mode=args.mode,
            skill_levels=skill_levels,
            repeats=args.repeats,
            seed=args.seed,
            max_moves=args.max_moves,
            depth=args.depth,
            time_ms=args.time_ms,
        )
    except ValueError as error:
        parser.error(str(error))

    try:
        with Stockfish(
            path=args.stockfish_path,
            skill_level=skill_levels[0],
            threads=1,
            hash_mb=16,
        ) as stockfish:
            run_match_benchmark(
                config,
                args.output,
                stockfish,
                stockfish_version=_stockfish_version(stockfish.binary_path),
            )
    except (FileNotFoundError, FileExistsError, UCIEngineError) as error:
        sys.stderr.write(f"Match benchmark failed: {error}\n")
        raise SystemExit(1) from error


if __name__ == "__main__":
    main()

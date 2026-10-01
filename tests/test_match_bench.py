"""Tests for paired match schedules and persisted move-level results."""

import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import pytest

from chess.game import ChessGame
from chess.match_bench import (
    OPENING_LINES,
    MatchBenchConfig,
    build_schedule,
    run_match_benchmark,
)
from chess.stockfish import Stockfish

FAKE_ENGINE_SCRIPT = str(Path(__file__).parent / "fake_uci_engine.py")


def test_schedule_pairs_profiles_and_balances_colors() -> None:
    schedule = build_schedule((0, 5), repeats=2, seed=211)
    assert schedule == build_schedule((0, 5), repeats=2, seed=211)
    assert len(schedule) == 64

    pairs: dict[str, list[object]] = defaultdict(list)
    for match in schedule:
        pairs[match.pair_id].append(match)
    assert len(pairs) == 32
    for pair in pairs.values():
        assert len(pair) == 2
        assert {match.profile for match in pair} == {"basic", "positional"}
        assert len({match.stockfish_skill_level for match in pair}) == 1
        assert len({match.opening.name for match in pair}) == 1
        assert len({match.repeat for match in pair}) == 1
        assert len({match.pychess_color for match in pair}) == 1

    profile_color_counts = Counter(
        (match.profile, match.stockfish_skill_level, match.pychess_color)
        for match in schedule
    )
    assert set(profile_color_counts.values()) == {8}


def test_opening_lines_are_legal_from_start_position() -> None:
    for opening in OPENING_LINES:
        game = ChessGame()
        for move in opening.san_moves:
            game.make_move(move)
        assert game.turn == "w"


def test_match_benchmark_persists_complete_uci_and_search_records(
    tmp_path: Path,
) -> None:
    output_path = tmp_path / "matches.json"
    config = MatchBenchConfig(
        mode="depth",
        skill_levels=(0, 1),
        repeats=1,
        seed=42,
        max_moves=5,
        depth=1,
    )
    with Stockfish(
        path=sys.executable,
        skill_level=0,
        threads=1,
        hash_mb=16,
        args=[FAKE_ENGINE_SCRIPT],
    ) as stockfish:
        payload = run_match_benchmark(
            config,
            output_path,
            stockfish,
            stockfish_version="Fake UCI test engine",
        )

    stored = json.loads(output_path.read_text(encoding="utf-8"))
    assert payload["complete"] is True
    assert stored["complete"] is True
    assert len(stored["matches"]) == 32
    assert stored["config"]["mode"] == "depth"
    assert stored["config"]["stockfish"]["version"] == "Fake UCI test engine"

    engine_searches: Counter[str] = Counter()
    for match in stored["matches"]:
        assert match["ply_count"] == len(match["uci_moves"])
        assert [ply["uci"] for ply in match["per_move"]] == match["uci_moves"]
        for ply in match["per_move"]:
            if ply["source"] == "search":
                engine_searches[ply["engine"]] += 1
                if ply["engine"] != "stockfish":
                    assert ply["completed_depth"] == 1
                    assert ply["nodes"] + ply["quiescence_nodes"] > 0
    assert engine_searches["basic"] + engine_searches["positional"] > 0
    assert engine_searches["stockfish"] > 0

    with pytest.raises(FileExistsError, match="Refusing to overwrite"):
        run_match_benchmark(config, output_path, stockfish)


def test_match_benchmark_config_rejects_invalid_limits() -> None:
    invalid_configs = (
        {"skill_levels": ()},
        {"skill_levels": (0, 0)},
        {"skill_levels": (21,)},
        {"repeats": 0},
        {"max_moves": 0},
        {"depth": 9},
        {"time_ms": 0},
    )
    for values in invalid_configs:
        with pytest.raises(ValueError):
            MatchBenchConfig(**values)

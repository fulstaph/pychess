"""Tests for Stockfish integration, match series runners, and CLI entrypoint."""

import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from chess import (
    ChessGame,
    GameStatus,
    MatchResult,
    Move,
    SeriesResult,
    Stockfish,
    find_stockfish,
    format_series_summary,
    play_match,
    run_series,
)
from chess.engine import choose_move
from chess.engine import main as engine_main
from chess.stockfish import main as stockfish_main
from chess.stockfish import play_human_vs_stockfish

FAKE_ENGINE_SCRIPT = str(Path(__file__).parent / "fake_uci_engine.py")


def test_find_stockfish_resolution(monkeypatch):
    # Custom existing path
    resolved = find_stockfish(sys.executable)
    assert Path(resolved).exists()

    # Nonexistent custom path
    with pytest.raises(FileNotFoundError, match="Stockfish executable not found"):
        find_stockfish("/nonexistent/stockfish/path/bin")

    # STOCKFISH_PATH environment variable
    monkeypatch.setenv("STOCKFISH_PATH", sys.executable)
    assert find_stockfish() == sys.executable

    # STOCKFISH_PATH set to nonexistent
    monkeypatch.setenv("STOCKFISH_PATH", "/nonexistent/from/env")
    with pytest.raises(
        FileNotFoundError, match="specified by STOCKFISH_PATH not found"
    ):
        find_stockfish()

    # No binary found anywhere
    monkeypatch.delenv("STOCKFISH_PATH", raising=False)
    monkeypatch.setattr(shutil, "which", lambda _: None)
    monkeypatch.setattr(os.path, "isfile", lambda _: False)
    with pytest.raises(
        FileNotFoundError, match="Stockfish executable could not be found"
    ):
        find_stockfish()


def test_stockfish_skill_level_validation():
    with pytest.raises(ValueError, match="Skill level must be between 0 and 20"):
        Stockfish(path=sys.executable, skill_level=-1, args=[FAKE_ENGINE_SCRIPT])

    with pytest.raises(ValueError, match="Skill level must be between 0 and 20"):
        Stockfish(path=sys.executable, skill_level=21, args=[FAKE_ENGINE_SCRIPT])


def test_stockfish_get_move_with_fen_and_history():
    with Stockfish(path=sys.executable, skill_level=0, args=[FAKE_ENGINE_SCRIPT]) as sf:
        game = ChessGame()
        # Move from starting position via FEN
        move1 = sf.get_move(game, movetime_ms=50)
        assert isinstance(move1, Move)
        assert move1 in game.legal_moves()

        # Move with move history
        game.make_move(move1)
        move2 = sf.get_move(game, moves_history=[move1], movetime_ms=50)
        assert isinstance(move2, Move)
        assert move2 in game.legal_moves()


def test_play_match_completion_and_termination():
    with Stockfish(path=sys.executable, skill_level=0, args=[FAKE_ENGINE_SCRIPT]) as sf:

        def white_player(g: ChessGame, _hist: tuple[Move, ...]) -> Move:
            return choose_move(g, depth=1)

        def black_player(g: ChessGame, hist: tuple[Move, ...]) -> Move:
            return sf.get_move(g, moves_history=hist, movetime_ms=50)

        # Normal match to completion or limit
        result = play_match(
            white_player=white_player,
            black_player=black_player,
            white_name="MiniMax",
            black_name="FakeSF",
            max_moves=10,
        )
        assert isinstance(result, MatchResult)
        assert result.white_name == "MiniMax"
        assert result.black_name == "FakeSF"
        assert len(result.moves) > 0
        assert result.halfmove_count == len(result.moves)

        # Forfeit when player raises exception
        def failing_player(_g: ChessGame, _hist: tuple[Move, ...]) -> Move:
            raise RuntimeError("Engine crashed")

        forfeit_result = play_match(
            white_player=failing_player,
            black_player=black_player,
            white_name="Crasher",
            black_name="Survivor",
        )
        assert forfeit_result.winner == "b"
        assert "forfeited" in forfeit_result.termination

        # Forfeit when player returns illegal move
        def illegal_player(_g: ChessGame, _hist: tuple[Move, ...]) -> str:
            return "e2e5"

        illegal_result = play_match(
            white_player=illegal_player,
            black_player=black_player,
            white_name="RuleBreaker",
            black_name="Defender",
        )
        assert illegal_result.winner == "b"
        assert "illegal move" in illegal_result.termination


def test_run_series_alternation_and_stats():
    with Stockfish(path=sys.executable, skill_level=0, args=[FAKE_ENGINE_SCRIPT]) as sf:
        with pytest.raises(ValueError, match="Number of games must be at least 1"):
            run_series(stockfish=sf, games=0)

        series = run_series(
            stockfish=sf,
            games=2,
            movetime_ms=50,
            depth=1,
            max_moves=5,
            engine_name="pychess",
            stockfish_name="Stockfish",
        )
        assert isinstance(series, SeriesResult)
        assert len(series.games) == 2
        # Game 1: pychess White, Stockfish Black
        assert series.games[0].white_name == "pychess"
        assert series.games[0].black_name == "Stockfish"
        # Game 2: Stockfish White, pychess Black
        assert series.games[1].white_name == "Stockfish"
        assert series.games[1].black_name == "pychess"

        total_score = series.scores["pychess"] + series.scores["Stockfish"]
        assert total_score == 2.0


def test_format_series_summary():
    match1 = MatchResult(
        white_name="pychess",
        black_name="Stockfish",
        winner="w",
        status=GameStatus.CHECKMATE,
        termination="checkmate",
        moves=(),
    )
    match2 = MatchResult(
        white_name="Stockfish",
        black_name="pychess",
        winner=None,
        status=GameStatus.STALEMATE,
        termination="stalemate",
        moves=(),
    )
    series = SeriesResult(
        games=(match1, match2),
        scores={"pychess": 1.5, "Stockfish": 0.5},
        wins={"pychess": 1, "Stockfish": 0},
        draws=1,
        white_wins=1,
        black_wins=0,
        average_moves=24.5,
    )
    summary = format_series_summary(series)
    assert "Tournament Match Summary (2 games)" in summary
    assert "pychess: 1.5 / 2 (75.0%)" in summary
    assert "Stockfish: 0.5 / 2 (25.0%)" in summary
    assert "White wins: 1 | Black wins: 0 | Draws: 1" in summary
    assert "Average moves per game: 24.5" in summary
    assert "checkmate" in summary
    assert "stalemate" in summary


def test_cli_subprocesses():
    # Help flag
    res_help = subprocess.run(
        [sys.executable, "-m", "chess.stockfish", "--help"],
        capture_output=True,
        text=True,
        timeout=10,
    )
    assert res_help.returncode == 0
    assert "Run matches or tournaments against Stockfish" in res_help.stdout

    # Games 0 rejection (exit code 2)
    res_zero = subprocess.run(
        [sys.executable, "-m", "chess.stockfish", "--games", "0"],
        capture_output=True,
        text=True,
        timeout=10,
    )
    assert res_zero.returncode == 2

    # Nonexistent binary exit code 1
    res_nobin = subprocess.run(
        [
            sys.executable,
            "-m",
            "chess.stockfish",
            "--stockfish-path",
            "/nonexistent/engine/binary",
        ],
        capture_output=True,
        text=True,
        timeout=10,
    )
    assert res_nobin.returncode == 1
    assert "Error:" in res_nobin.stderr

    # Engine CLI --stockfish flag with nonexistent binary
    res_engine_nobin = subprocess.run(
        [
            sys.executable,
            "-m",
            "chess.engine",
            "--stockfish",
            "--stockfish-path",
            "/nonexistent/binary",
        ],
        input="1\n",
        capture_output=True,
        text=True,
        timeout=10,
    )
    assert res_engine_nobin.returncode == 1
    assert "Error:" in res_engine_nobin.stderr

    # Series run using fake UCI engine via python
    # We create a temporary launcher wrapper script so stockfish-path
    # can be an executable
    launcher_path = Path(__file__).parent / "_run_fake_engine.sh"
    launcher_path.write_text(f"#!/bin/sh\nexec {sys.executable} {FAKE_ENGINE_SCRIPT}\n")
    launcher_path.chmod(0o755)

    try:
        res_series = subprocess.run(  # noqa: S603
            [
                sys.executable,
                "-m",
                "chess.stockfish",
                "--stockfish-path",
                str(launcher_path),
                "--games",
                "2",
                "--max-moves",
                "5",
                "--quiet",
            ],
            capture_output=True,
            text=True,
            timeout=15,
        )
        assert res_series.returncode == 0, res_series.stderr
        assert "Game 1/2:" in res_series.stdout
        assert "Game 2/2:" in res_series.stdout
        assert "Tournament Match Summary (2 games)" in res_series.stdout
    finally:
        if launcher_path.exists():
            launcher_path.unlink()


def test_real_stockfish_integration():
    real_stockfish = shutil.which("stockfish") or (
        "/opt/homebrew/bin/stockfish"
        if os.path.isfile("/opt/homebrew/bin/stockfish")
        else None
    )
    if not real_stockfish:
        pytest.skip("Stockfish binary not installed on host")

    with Stockfish(path=real_stockfish, skill_level=0) as sf:
        game = ChessGame()
        move = sf.get_move(game, movetime_ms=50, depth=1)
        assert isinstance(move, Move)
        assert move in game.legal_moves()


def test_play_human_vs_stockfish(monkeypatch):
    with Stockfish(path=sys.executable, skill_level=0, args=[FAKE_ENGINE_SCRIPT]) as sf:
        # Human as White: invalid move, valid move e4, then resign
        inputs = iter(["invalid_move", "e4", "q"])
        monkeypatch.setattr("builtins.input", lambda _: next(inputs))
        res_white = play_human_vs_stockfish(sf, human_color="w", movetime_ms=50)
        assert res_white.winner == "b"
        assert res_white.termination == "resignation"
        assert len(res_white.moves) >= 1

        # Human as Black: engine plays first, then human resigns
        inputs_black = iter(["q"])
        monkeypatch.setattr("builtins.input", lambda _: next(inputs_black))
        res_black = play_human_vs_stockfish(sf, human_color="b", movetime_ms=50)
        assert res_black.winner == "w"
        assert res_black.termination == "resignation"


def test_stockfish_cli_main_and_engine_delegation(monkeypatch, capsys):
    real_stockfish = shutil.which("stockfish") or (
        "/opt/homebrew/bin/stockfish"
        if os.path.isfile("/opt/homebrew/bin/stockfish")
        else None
    )
    if not real_stockfish:
        pytest.skip("Stockfish binary not installed")

    # Tournament mode with 1 game, quiet
    stockfish_main(["--games", "1", "--max-moves", "2", "--movetime", "20", "--quiet"])
    captured = capsys.readouterr()
    assert "Tournament Match Summary (1 game)" in captured.out

    # Human mode: select quit immediately
    monkeypatch.setattr("builtins.input", lambda _: "q")
    stockfish_main(["--human", "--movetime", "20"])
    captured_human = capsys.readouterr()
    assert "Goodbye!" in captured_human.out

    # Engine CLI delegating to stockfish then quit
    inputs = iter(["1", "q"])
    monkeypatch.setattr("builtins.input", lambda _: next(inputs))
    engine_main(["--stockfish"])
    captured_engine = capsys.readouterr()
    assert "Goodbye!" in captured_engine.out

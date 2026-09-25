import os
import subprocess
import sys


def run_cli(
    inputs: str, args: list[str] | None = None
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "-m", "chess.engine", *(args or [])],
        input=inputs,
        text=True,
        capture_output=True,
        timeout=15,
        env={**os.environ, "PYTHONUNBUFFERED": "1"},
    )


def test_white_can_retry_invalid_move_without_losing_turn():
    result = run_cli("1\nnotamove\ne4\nq\n")
    assert result.returncode == 0, result.stderr
    assert "Invalid move" in result.stdout
    assert "e4" in result.stdout
    assert "Engine: e7e5" in result.stdout
    assert "Traceback" not in result.stderr


def test_black_choice_makes_white_ai_move_first():
    result = run_cli("2\nq\n")
    assert result.returncode == 0, result.stderr
    assert result.stdout.index("Engine: d2d4") < result.stdout.index("Your move")
    assert "Traceback" not in result.stderr


def test_menu_quit_and_eof_exit_without_traceback():
    for inputs in ("3\n", "q\n", ""):
        result = run_cli(inputs)
        assert result.returncode == 0, result.stderr
        assert "Traceback" not in result.stderr


def test_interactive_help_and_fen():
    result = run_cli("1\nhelp\nfen\nq\n")
    assert result.returncode == 0, result.stderr
    assert "Commands: 'undo'" in result.stdout
    assert (
        "FEN: rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1" in result.stdout
    )


def test_interactive_undo_command():
    result = run_cli("1\ne4\nundo\nq\n")
    assert result.returncode == 0, result.stderr
    assert "You played: e4" in result.stdout
    assert "Engine: e7e5" in result.stdout
    assert "Move undone." in result.stdout

    result_empty = run_cli("1\nundo\nq\n")
    assert "No moves to undo." in result_empty.stdout


def test_visual_flags_unicode_and_color_board():
    res_unicode = run_cli("1\nq\n", args=["--unicode"])
    assert res_unicode.returncode == 0, res_unicode.stderr
    assert "♙" in res_unicode.stdout or "♟" in res_unicode.stdout

    res_color = run_cli("1\nq\n", args=["--color-board"])
    assert res_color.returncode == 0, res_color.stderr
    assert "\033[" in res_color.stdout


def test_interactive_commands_moves_eval_pgn_flip():
    inputs = "1\nmoves\nmoves e2\neval\npgn\nflip\nq\n"
    res = run_cli(inputs)
    assert res.returncode == 0, res.stderr
    assert "Pawn" in res.stdout
    assert "e2e4" in res.stdout
    assert "Material:" in res.stdout
    assert '[Event "pychess Terminal Game"]' in res.stdout
    assert "Board flipped." in res.stdout


def test_cli_dashboard_flag():
    res = run_cli("1\nq\n", args=["--dashboard"])
    assert res.returncode == 0, res.stderr
    assert "pychess" in res.stdout
    assert "Recent Moves:" in res.stdout


def test_cli_size_scaling():
    res = run_cli("1\nsize 2\nsize 4\nq\n", args=["--dashboard", "--size", "2"])
    assert res.returncode == 0, res.stderr
    assert "Board size set to 2 (medium)." in res.stdout
    assert "Board size set to 4 (giant)." in res.stdout

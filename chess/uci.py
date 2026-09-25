"""UCI protocol engine communication wrapper."""

import contextlib
import queue
import subprocess
import sys
import threading
import time
from collections.abc import Sequence
from types import TracebackType
from typing import Self, TextIO

from .engine import choose_move
from .game import ChessGame
from .helpers import square_notation
from .move import Move


def to_uci(move: Move) -> str:
    """Format a Move into UCI coordinate string notation (e.g., 'e2e4', 'e7e8q')."""
    source = square_notation(*move.from_square)
    target = square_notation(*move.to_square)
    promotion = move.promotion_to.value.lower() if move.promotion_to else ""
    return f"{source}{target}{promotion}"


class UCIEngineError(RuntimeError):
    """Raised when a UCI chess engine fails or produces invalid output."""


class UCIEngine:
    """Universal Chess Interface (UCI) protocol client for external chess engines."""

    def __init__(
        self,
        binary_path: str,
        args: list[str] | None = None,
        startup_timeout: float = 10.0,
    ) -> None:
        self.binary_path = binary_path
        self._closed = False
        command = [binary_path, *(args or [])]
        try:
            self._process = subprocess.Popen(  # noqa: S603
                command,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                bufsize=1,
            )
        except OSError as exc:
            raise UCIEngineError(
                f"Failed to spawn engine process {binary_path!r}: {exc}"
            ) from exc

        self._queue: queue.Queue[str | None] = queue.Queue()
        self._reader_thread = threading.Thread(
            target=self._read_stdout,
            name=f"UCIEngine-reader-{self._process.pid}",
            daemon=True,
        )
        self._reader_thread.start()

        # Perform initial UCI handshake
        try:
            self.send_command("uci")
            self.wait_for("uciok", timeout=startup_timeout)
            self.send_command("isready")
            self.wait_for("readyok", timeout=startup_timeout)
        except Exception:
            self.close()
            raise

    def _read_stdout(self) -> None:
        if self._process.stdout is None:
            self._queue.put(None)
            return
        try:
            for raw_line in iter(self._process.stdout.readline, ""):
                line = raw_line.strip()
                if line:
                    self._queue.put(line)
        except ValueError, OSError:
            pass
        finally:
            self._queue.put(None)

    def _read_line(self, timeout: float) -> str:
        deadline = time.monotonic() + timeout
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError("Timed out waiting for engine response")
            try:
                item = self._queue.get(timeout=min(remaining, 0.1))
            except queue.Empty:
                if self._process.poll() is not None and self._queue.empty():
                    code = self._process.returncode
                    raise UCIEngineError(
                        f"Engine process terminated with code {code}"
                    ) from None
                continue
            if item is None:
                exit_code = self._process.poll()
                raise UCIEngineError(
                    f"Engine process closed output stream (exit code: {exit_code})"
                )
            return item

    def send_command(self, cmd: str) -> None:
        """Send a single UCI command line to the engine."""
        if self._closed or self._process.poll() is not None:
            code = self._process.returncode
            msg = f"Cannot send command to terminated engine (code {code})"
            raise UCIEngineError(msg)
        if self._process.stdin is None or self._process.stdin.closed:
            raise UCIEngineError("Engine stdin stream is closed")
        try:
            self._process.stdin.write(f"{cmd}\n")
            self._process.stdin.flush()
        except OSError as exc:
            raise UCIEngineError(f"Failed to write to engine: {exc}") from exc

    def wait_for(self, prefix_or_exact: str, timeout: float = 10.0) -> list[str]:
        """Read stdout lines until one matches prefix_or_exact."""
        deadline = time.monotonic() + timeout
        collected: list[str] = []
        while True:
            remaining = max(0.001, deadline - time.monotonic())
            line = self._read_line(remaining)
            collected.append(line)
            if line == prefix_or_exact or line.startswith(prefix_or_exact):
                return collected

    def set_option(self, name: str, value: str | int | bool) -> None:
        """Set a UCI engine option and wait for readiness confirmation."""
        val_str = (
            "true" if value is True else ("false" if value is False else str(value))
        )
        self.send_command(f"setoption name {name} value {val_str}")
        self.send_command("isready")
        self.wait_for("readyok")

    def new_game(self) -> None:
        """Signal ucinewgame and wait for the engine to acknowledge readiness."""
        self.send_command("ucinewgame")
        self.send_command("isready")
        self.wait_for("readyok")

    def set_position(
        self, moves: Sequence[str] | None = None, fen: str | None = None
    ) -> None:
        """Set current position via FEN or starting position with move history."""
        cmd = f"position fen {fen}" if fen else "position startpos"
        if moves:
            cmd += f" moves {' '.join(moves)}"
        self.send_command(cmd)

    def go(
        self,
        movetime_ms: int | None = None,
        depth: int | None = None,
        timeout: float | None = None,
    ) -> str:
        """Request the engine search the position and return the best move string."""
        cmd_parts = ["go"]
        if movetime_ms is not None:
            cmd_parts.extend(["movetime", str(movetime_ms)])
        if depth is not None:
            cmd_parts.extend(["depth", str(depth)])

        if timeout is None:
            if movetime_ms is not None:
                calc_timeout = max(5.0, (movetime_ms / 1000.0) * 3 + 2.0)
            else:
                calc_timeout = 30.0
        else:
            calc_timeout = timeout

        self.send_command(" ".join(cmd_parts))
        lines = self.wait_for("bestmove", timeout=calc_timeout)
        bestmove_line = lines[-1]
        tokens = bestmove_line.split()
        if len(tokens) < 2 or tokens[1] == "(none)":
            raise UCIEngineError("No legal moves available")
        return tokens[1]

    def close(self) -> None:
        """Cleanly terminate the engine process and close communication streams."""
        if self._closed:
            return
        self._closed = True
        if self._process.poll() is None:
            try:
                if self._process.stdin and not self._process.stdin.closed:
                    self._process.stdin.write("quit\n")
                    self._process.stdin.flush()
                self._process.wait(timeout=2.0)
            except OSError, subprocess.TimeoutExpired:
                try:
                    self._process.terminate()
                    self._process.wait(timeout=1.0)
                except OSError, subprocess.TimeoutExpired:
                    self._process.kill()
        if self._process.stdin and not self._process.stdin.closed:
            self._process.stdin.close()
        if self._process.stdout and not self._process.stdout.closed:
            self._process.stdout.close()
        if self._process.stderr and not self._process.stderr.closed:
            self._process.stderr.close()

    def __enter__(self) -> Self:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        self.close()


def run_uci_server(
    input_stream: TextIO | None = None,
    output_stream: TextIO | None = None,
) -> None:
    """Run the pychess UCI engine server loop communicating over text streams."""
    stdin = input_stream if input_stream is not None else sys.stdin
    stdout = output_stream if output_stream is not None else sys.stdout

    game = ChessGame()
    search_depth = 2
    quiescence_enabled = True

    for raw_line in stdin:
        line = raw_line.strip()
        if not line:
            continue

        if line == "uci":
            stdout.write("id name pychess\n")
            stdout.write("id author pychess developers\n")
            stdout.write("option name Depth type spin default 2 min 1 max 8\n")
            stdout.write("option name Quiescence type check default true\n")
            stdout.write("uciok\n")
            stdout.flush()
        elif line == "isready":
            stdout.write("readyok\n")
            stdout.flush()
        elif line == "ucinewgame":
            game = ChessGame()
        elif line.startswith("setoption"):
            parts = line.split()
            if "name" in parts:
                name_idx = parts.index("name") + 1
                val_idx = parts.index("value") + 1 if "value" in parts else len(parts)
                opt_name = " ".join(
                    parts[name_idx : val_idx - 1 if "value" in parts else len(parts)]
                ).lower()
                opt_val = " ".join(parts[val_idx:]) if "value" in parts else ""
                if opt_name == "depth":
                    with contextlib.suppress(ValueError):
                        search_depth = max(1, int(opt_val))
                elif opt_name == "quiescence":
                    quiescence_enabled = opt_val.lower() in ("true", "1", "yes")
        elif line.startswith("position"):
            tokens = line.split()
            moves_idx = tokens.index("moves") if "moves" in tokens else -1
            if "startpos" in tokens:
                game = ChessGame()
            elif "fen" in tokens:
                fen_tokens = tokens[2:moves_idx] if moves_idx != -1 else tokens[2:]
                game = ChessGame.from_fen(" ".join(fen_tokens))

            if moves_idx != -1:
                for move_str in tokens[moves_idx + 1 :]:
                    try:
                        game.make_move(move_str)
                    except ValueError:
                        break
        elif line.startswith("go"):
            tokens = line.split()
            depth = search_depth
            if "depth" in tokens:
                try:
                    d_idx = tokens.index("depth") + 1
                    if d_idx < len(tokens):
                        depth = int(tokens[d_idx])
                except ValueError, IndexError:
                    pass
            legal = game.legal_moves()
            if not legal:
                stdout.write("bestmove (none)\n")
            else:
                try:
                    selected = choose_move(
                        game, depth=depth, quiescence=quiescence_enabled
                    )
                except ValueError:
                    selected = legal[0]
                stdout.write(f"bestmove {to_uci(selected)}\n")
            stdout.flush()
        elif line == "quit":
            break


if __name__ == "__main__":
    with contextlib.suppress(KeyboardInterrupt, EOFError):
        run_uci_server()

"""UCI protocol engine communication wrapper.

Architecture
------------
A UCI engine is a plain-text protocol over two pipes: the client sends
one command per line on the engine's stdin, the engine answers line-wise
on stdout (option lines, ``info`` progress, ``bestmove``, ...), and any
diagnostics go to stderr.

Handshake sequence after spawning the process:

    client -> "uci"          engine -> id lines + "uciok"
    client -> "isready"      engine -> "readyok"

Per game: ``ucinewgame`` + ``isready``/``readyok``, then
``position startpos moves ...`` (or ``position fen ...``), then
``go depth/movetime``; the engine answers with ``info`` lines while
searching and finally ``bestmove <uci-move>`` (or ``bestmove (none)``
when there is no legal move).  Shutdown is a ``quit`` command.

Reads are line-buffered because UCI is strictly line-oriented: every
protocol line is a complete unit of meaning, so the reader thread
drains ``readline()`` from the pipe into a queue and the protocol
methods consume whole lines.  A ``None`` sentinel marks a closed pipe.

Error model
-----------
- ``UCIEngineError``: spawn failure, write failure, protocol timeout,
  or the engine process dying mid-conversation (the raised message
  includes the last observed stderr output where available).
- ``EOFError`` semantics: a reader thread that hits EOF on stdout puts
  the ``None`` sentinel; the next ``_read_line`` call surfaces it as
  ``UCIEngineError`` ("closed output stream") carrying the exit code.
  Callers treat both as "the engine is dead" and must not reuse it.
"""

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
from .log import get_logger, setup_logging
from .move import Move

logger = get_logger("uci")


def to_uci(move: Move) -> str:
    """Format a Move into UCI coordinate string notation (e.g., 'e2e4', 'e7e8q').

    UCI encodes a move as source square + target square + optional
    promotion piece character (lowercase).  Castling has no dedicated
    syntax: it is sent as a king-square move (e.g. 'e1g1' for O-O,
    'e1c1' for O-O-O) because the king physically lands on that file.
    """
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
            # text=True + bufsize=1 give line-buffered text I/O, matching
            # the line-oriented UCI protocol on both directions.
            self._process = subprocess.Popen(  # noqa: S603
                command,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                bufsize=1,
            )
        except OSError as exc:
            logger.error("Failed to spawn engine %r: %s", binary_path, exc)
            raise UCIEngineError(
                f"Failed to spawn engine process {binary_path!r}: {exc}"
            ) from exc

        logger.info("Engine started: %s (pid %d)", " ".join(command), self._process.pid)

        self._queue: queue.Queue[str | None] = queue.Queue()
        # Daemon reader thread decouples pipe draining from protocol
        # calls: if the caller stops reading, the thread exits with the
        # process anyway (daemon=True) and cannot hold shutdown up.
        self._reader_thread = threading.Thread(
            target=self._read_stdout,
            name=f"UCIEngine-reader-{self._process.pid}",
            daemon=True,
        )
        self._reader_thread.start()

        # Perform initial UCI handshake: "uci" must be answered with
        # "uciok" before any game command is accepted by spec.
        try:
            self.send_command("uci")
            self.wait_for("uciok", timeout=startup_timeout)
            self.send_command("isready")
            self.wait_for("readyok", timeout=startup_timeout)
        except Exception:
            # Handshake failure means the engine never came up; tear
            # down the process before propagating so nothing leaks.
            self.close()
            raise

    def _read_stdout(self) -> None:
        """Reader-thread body: pump engine stdout lines into the queue.

        Blank lines are dropped (engines pad with newlines), and a
        ``None`` sentinel is always queued on exit so blocked readers
        cannot hang on a dead pipe.
        """
        if self._process.stdout is None:
            self._queue.put(None)
            return
        try:
            # readline() returns "" only at EOF, which terminates the
            # loop; the sentinel below is the canonical "pipe closed".
            for raw_line in iter(self._process.stdout.readline, ""):
                line = raw_line.strip()
                if line:
                    logger.debug("Engine <- %s", line)
                    self._queue.put(line)
        except ValueError, OSError:
            # Broken pipe / text decode error mid-read: treat as EOF.
            pass
        finally:
            self._queue.put(None)

    def _read_line(self, timeout: float) -> str:
        """Block until one stdout line is available, or the engine dies.

        Raises ``TimeoutError`` on timeout and ``UCIEngineError`` when
        the reader thread signalled a closed pipe (engine exit) — the
        message includes the observed exit code so callers can tell a
        clean ``quit`` (0) from a crash apart.
        """
        deadline = time.monotonic() + timeout
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                logger.error("Timed out waiting for engine response")
                raise TimeoutError("Timed out waiting for engine response")
            try:
                # Poll in 100 ms slices so a dead-but-unsignalled
                # process is noticed via poll() instead of waiting out
                # the whole timeout.
                item = self._queue.get(timeout=min(remaining, 0.1))
            except queue.Empty:
                if self._process.poll() is not None and self._queue.empty():
                    code = self._process.returncode
                    logger.error("Engine process terminated with code %s", code)
                    raise UCIEngineError(
                        f"Engine process terminated with code {code}"
                    ) from None
                continue
            if item is None:
                exit_code = self._process.poll()
                logger.error(
                    "Engine process closed output stream (exit code: %s)",
                    exit_code,
                )
                raise UCIEngineError(
                    f"Engine process closed output stream (exit code: {exit_code})"
                )
            return item

    def send_command(self, cmd: str) -> None:
        """Send a single UCI command line to the engine."""
        if self._closed or self._process.poll() is not None:
            code = self._process.returncode
            msg = f"Cannot send command to terminated engine (code {code})"
            logger.error("Refusing to send %r: %s", cmd, msg)
            raise UCIEngineError(msg)
        if self._process.stdin is None or self._process.stdin.closed:
            logger.error("Engine stdin stream is closed; cannot send %r", cmd)
            raise UCIEngineError("Engine stdin stream is closed")
        try:
            # UCI commands are newline-terminated; flush immediately so
            # the engine does not wait for a block of buffered text.
            self._process.stdin.write(f"{cmd}\n")
            self._process.stdin.flush()
        except OSError as exc:
            logger.error("Failed to write %r to engine: %s", cmd, exc)
            raise UCIEngineError(f"Failed to write to engine: {exc}") from exc
        logger.debug("Engine -> %s", cmd)

    def wait_for(self, prefix_or_exact: str, timeout: float = 10.0) -> list[str]:
        """Read stdout lines until one matches prefix_or_exact.

        The match is prefix-based so it accepts ``bestmove e2e4`` as well
        as bare ``readyok``; all lines seen up to and including the match
        are returned so callers can inspect ``info`` output.
        """
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
        # Booleans map to UCI's textual "true"/"false"; everything else
        # (ints, strings) is stringified verbatim.
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
        # ``position startpos moves ...`` replays a move list from the
        # initial position; ``position fen ...`` places pieces directly.
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
        """Request the engine search the position and return the best move string.

        ``movetime_ms`` caps engine thinking time; ``depth`` caps
        principal-variation depth.  When no explicit ``timeout`` is
        given it is derived from movetime (3x budget + 2 s margin,
        floored at 5 s) or a flat 30 s for unbounded searches — always
        more generous than the engine's own search budget so the client
        never kills a legitimate search.
        """
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
        # "bestmove" is the first token; the move (or "(none)") follows.
        tokens = bestmove_line.split()
        if len(tokens) < 2 or tokens[1] == "(none)":
            logger.error("Engine reported no legal moves: %s", bestmove_line)
            raise UCIEngineError("No legal moves available")
        return tokens[1]

    def close(self) -> None:
        """Cleanly terminate the engine process and close communication streams.

        Idempotent.  Tries, in order: cooperative ``quit`` with a 2 s
        wait, ``terminate()`` (SIGTERM) with a 1 s wait, then a hard
        ``kill()``.  All pipes are closed afterwards so the reader
        thread sees EOF promptly.
        """
        if self._closed:
            return
        self._closed = True
        if self._process.poll() is None:
            logger.debug("Stopping engine process %d", self._process.pid)
            try:
                if self._process.stdin and not self._process.stdin.closed:
                    self._process.stdin.write("quit\n")
                    self._process.stdin.flush()
                self._process.wait(timeout=2.0)
            except OSError, subprocess.TimeoutExpired:
                logger.warning(
                    "Engine did not exit on quit; terminating (pid %d)",
                    self._process.pid,
                )
                try:
                    self._process.terminate()
                    self._process.wait(timeout=1.0)
                except OSError, subprocess.TimeoutExpired:
                    logger.warning(
                        "Engine ignored SIGTERM; killing (pid %d)",
                        self._process.pid,
                    )
                    self._process.kill()
        if self._process.stdin and not self._process.stdin.closed:
            self._process.stdin.close()
        if self._process.stdout and not self._process.stdout.closed:
            self._process.stdout.close()
        if self._process.stderr and not self._process.stderr.closed:
            self._process.stderr.close()
        logger.info("Engine stopped (exit code: %s)", self._process.returncode)

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
    """Run the pychess UCI engine server loop communicating over text streams.

    Acts as a UCI *engine*: reads commands from ``input_stream``
    (default stdin) one line at a time and writes UCI responses to
    ``output_stream`` (default stdout).  Every response is flushed
    immediately because GUI clients wait for it before proceeding.

    Exit paths:
    - ``quit`` command: clean break out of the loop.
    - EOF on the input stream (client closed the pipe): the for-loop
      terminates naturally.
    - ``KeyboardInterrupt`` / ``EOFError``: suppressed by the
      ``__main__`` wrapper below.
    """
    stdin = input_stream if input_stream is not None else sys.stdin
    stdout = output_stream if output_stream is not None else sys.stdout

    game = ChessGame()
    search_depth = 2
    quiescence_enabled = True

    for raw_line in stdin:
        line = raw_line.strip()
        if not line:
            continue
        logger.debug("Server <- %s", line)

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
            # "setoption name <Name> value <value>" — the option name
            # may contain spaces, so it is reconstructed from the tokens
            # between "name" and "value" (or the end of the line).
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
                        # A bad replay line means the position is
                        # already diverged; stop replaying rather than
                        # guessing.
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
                    # Malformed "go" line: keep the configured default.
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
                    # Search found nothing usable; any legal move is a
                    # valid protocol answer, so fall back to the first.
                    selected = legal[0]
                stdout.write(f"bestmove {to_uci(selected)}\n")
            stdout.flush()
        elif line == "quit":
            break


if __name__ == "__main__":
    # CLI entrypoint: only place allowed to configure logging output.
    setup_logging()
    with contextlib.suppress(KeyboardInterrupt, EOFError):
        run_uci_server()

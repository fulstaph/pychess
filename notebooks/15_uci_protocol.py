# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.5
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # 15. The UCI protocol: engines as line-oriented services
#
# **Goals:** distinguish a UCI client from an engine server, trace the handshake and search conversation, and exercise pychess's real server using both in-memory streams and a child process. UCI is a text protocol, not a chess algorithm; it standardizes communication between a GUI/client and an engine.
#
# Primary references: [Chessprogramming Wiki: UCI](https://www.chessprogramming.org/UCI) and [UCI protocol description](https://backscattering.de/chess/uci/).

# %% [markdown]
# ## Conversation phases and command meaning
#
# A typical lifecycle is `uci` → engine identity/options → `uciok`, then `isready` → `readyok`. The readiness ping lets a client wait for initialization without starting a search. A client can send `ucinewgame` and configure options, then describe a position and ask the engine to search it.
#
# - `position startpos [moves ...]` or `position fen ... [moves ...]` replaces the engine's current board context. The move suffix is a sequence of coordinate moves.
# - `go depth N`, `go movetime MS`, or clock-based forms start a search. The engine may stream `info` progress and eventually returns `bestmove MOVE` (or `bestmove 0000` if no legal move exists).
# - `stop` asks an ongoing search to finish promptly and report a best move; it is not the same as shutting down. `quit` requests process shutdown.
#
# These commands are complete newline-terminated records. Both sides must flush protocol output: a client blocked waiting for `uciok` cannot proceed if it remains in a userspace buffer. A UCI process must keep stdout protocol-clean; diagnostic logs belong on stderr, not mixed into response lines.

# %% [markdown]
# ## Server versus client
#
# A **server/engine** reads commands from its input and writes protocol replies. pychess exposes `run_uci_server(input_stream, output_stream)` for that role. A **client** launches or connects to an engine, writes commands, parses asynchronous output, handles timeouts/lifecycle, and validates the returned move; `UCIEngine` in `chess.uci` is a client wrapper. Below, it launches this repository's own UCI server in a child Python process—not an external engine.
#
# In the real protocol, searches can run asynchronously, so a server needs to process `stop` while search is underway. pychess's current server loop searches synchronously while handling `go`; its simple implementation does not implement the `stop` command. The examples use fixed shallow searches and finite client timeouts to stay bounded.

# %%
# Inspect the client wrapper and use its process command as documented.
import inspect
import io
import sys

from chess import ChessGame
from chess.uci import UCIEngine, _go_limits, run_uci_server, to_uci

print('UCIEngine client constructor:', inspect.signature(UCIEngine))
print('pychess server function:', inspect.signature(run_uci_server))

# %% [markdown]
# Feed a sequence of complete protocol lines to the actual pychess server. `StringIO` provides deterministic text streams, and the server flushes every reply just as it would over a pipe.

# %%
commands = io.StringIO(
    'uci\n'
    'isready\n'
    'position startpos moves e2e4 e7e5\n'
    'go depth 1\n'
    'quit\n'
)
responses = io.StringIO()
run_uci_server(commands, responses)
protocol_lines = responses.getvalue().splitlines()
print('\n'.join(protocol_lines))

assert protocol_lines[0].startswith('id name ')
assert protocol_lines[-1].startswith('bestmove ')
position = ChessGame().after('e2e4').after('e7e5')
legal_uci_moves = {to_uci(move) for move in position.legal_moves()}
assert protocol_lines[-1].split()[1] in legal_uci_moves
assert 'uciok' in protocol_lines
assert 'readyok' in protocol_lines
assert all(not line.startswith(('DEBUG', 'INFO', 'WARNING')) for line in protocol_lines)

# %% [markdown]
# ## A real client round trip
#
# The in-memory exchange above isolates the server loop. Here the client performs the UCI handshake with a child process running this same repository's server, sends a legal position, and waits for a shallow search result. Both startup and search waits are explicitly bounded; the context manager closes the child even if an assertion fails.

# %%
position = ChessGame().after('e2e4').after('e7e5')
legal_uci_moves = {to_uci(move) for move in position.legal_moves()}
with UCIEngine(
    sys.executable,
    args=['-m', 'chess.uci'],
    startup_timeout=5.0,
) as client:
    client.set_position(moves=['e2e4', 'e7e5'])
    bestmove = client.go(depth=1, timeout=5.0)

print('Child-process bestmove:', bestmove)
assert bestmove in legal_uci_moves

# %% [markdown]
# ## Clock allocation in this implementation
#
# `_go_limits` is a private helper imported explicitly here for source-reading purposes. This is pychess's current server policy, not a UCI-mandated time-allocation formula. With a clock, it divides the side-to-move's remaining milliseconds by `movestogo` (defaulting to 30 when absent/invalid), then adds that side's increment. Without an explicit `depth`, the clock search may iterate up to the server's depth ceiling; an explicit `depth` caps that ceiling.

# %%
white_clock = _go_limits(
    ['go', 'wtime', '60000', 'btime', '45000', 'winc', '1000',
     'binc', '500', 'movestogo', '20'],
    turn='w',
    default_depth=4,
)
black_clock = _go_limits(
    ['go', 'wtime', '60000', 'btime', '45000', 'winc', '1000',
     'binc', '500', 'movestogo', '15'],
    turn='b',
    default_depth=4,
)
depth_capped = _go_limits(
    ['go', 'wtime', '60000', 'winc', '1000', 'movestogo', '20',
     'depth', '2'],
    turn='w',
    default_depth=4,
)
print('White clock (depth ceiling, budget ms):', white_clock)
print('Black clock (depth ceiling, budget ms):', black_clock)
print('White clock with explicit depth 2:', depth_capped)
assert white_clock == (8, 4000)  # 60000 // 20 + 1000
assert black_clock == (8, 3500)  # 45000 // 15 + 500
assert depth_capped == (2, 4000)  # explicit depth caps the ceiling

# %% [markdown]
# ## Parsing is stateful, not just splitting tokens
#
# The server first consumes whole lines, then interprets each command in context: a `position` changes the board used by a later `go`; `go` produces a best-move response. `position fen ...` contains six FEN fields, so a parser must know where an optional `moves` suffix begins instead of treating each token as an independent command. `setoption name ... value ...` likewise permits option names containing spaces.
# The response assertions check legality in the position requested, rather than pinning the particular engine move. Another compliant engine may choose a different legal move.

# %% [markdown]
# ## Takeaways
#
# - `uci` negotiates identity/options; `isready` is a synchronization barrier; `position` establishes context; `go` begins work; `bestmove` completes a search.
# - `stop` requests an ongoing search to finish, while `quit` ends the engine session. pychess's current server searches synchronously during `go`, so it cannot service `stop` until that search returns.
# - The client and server are opposite roles. pychess's `UCIEngine` client can launch its own server (`python -m chess.uci`) as well as other engines; `run_uci_server` is the server entry point.
#
# **Reflect:** Why might `readyok` be needed after changing engine options? What deadlock can result if either peer forgets to flush? Which part of the pychess server would need architectural change to honor `stop` during a long search?
#
# Next: [Endgame programming](16_endgame_programming.ipynb) explores rule-sensitive search and exact results.

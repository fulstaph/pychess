# Chess Engine in Python

A dependency-free educational chess engine for Python 3.14 and newer. The `chess` package provides an immutable board, legal-move generation, notation parsing, and a human-versus-engine terminal game.

## Install and play

Run commands from the repository root. Install [uv](https://docs.astral.sh/uv/getting-started/installation/) first, then:

```bash
uv sync --python 3.14 --locked
uv run python -m chess.engine
```

Alternatively, `./install.sh` syncs the locked environment and prints the play command. For a conventional editable package install from the repository root, use `python -m pip install -e .` in a Python 3.14+ environment.

## Learn in notebooks

Install Python 3.14 or newer and run the lessons from the repository root:

```bash
uv run --locked --extra notebooks jupyter lab notebooks
```

The curriculum contains 20 notebooks in order. Run each notebook's cells top-to-bottom in a separate kernel. Percent-format `.py` files accompany the notebooks as editable sources.

- **Core:** [01 — board and coordinates](notebooks/01_board_and_coordinates.ipynb), [02 — moves and rules](notebooks/02_moves_and_rules.ipynb), [03 — engine decisions](notebooks/03_engine_decisions.ipynb), [04 — Zobrist hashing](notebooks/04_zobrist_hashing.ipynb), [05 — evaluation deep dive](notebooks/05_evaluation_deep_dive.ipynb).
- **Representation and search:** [06 — board representations](notebooks/06_board_representations.ipynb), [07 — minimax, negamax, and alpha-beta](notebooks/07_search_algorithms.ipynb), [08 — search heuristics](notebooks/08_search_heuristics.ipynb), [09 — transposition tables](notebooks/09_transposition_tables.ipynb), [10 — draw state and repetition](notebooks/10_draw_state_and_repetition.ipynb), [11 — advanced evaluation](notebooks/11_advanced_evaluation.ipynb).
- **Validation and engine systems:** [12 — testing and benchmarking](notebooks/12_engine_testing_and_benchmarking.ipynb), [13 — evaluation tuning](notebooks/13_evaluation_tuning.ipynb), [14 — chess formats](notebooks/14_chess_formats.ipynb), [15 — UCI protocol](notebooks/15_uci_protocol.ipynb), [16 — endgame programming](notebooks/16_endgame_programming.ipynb), [17 — NNUE foundations](notebooks/17_nnue_foundations.ipynb), [18 — opening books](notebooks/18_opening_books.ipynb), [19 — parallel search](notebooks/19_parallel_search.ipynb).
- **Performance:** [20 — move-generation performance](notebooks/20_move_generation_performance.ipynb).

If [go-task](https://taskfile.dev/) is installed, run `task` to list available tasks:

* `task play` / `task play:tui`: Start terminal chess game (optional rich side-by-side dashboard with Unicode pieces).
* `task play:large` / `task play:giant`: Start game with scaled multi-line piece art (sizes 2 and 4).
* `task web`: Start local web chessboard UI in browser (`http://localhost:8000`) powered by Lichess Chessground.
* `task play:stockfish`: Play interactively against Stockfish in the terminal.
* `task uci`: Run pychess as a standard Universal Chess Interface (UCI) engine server.
* `task bench`: Run canonical perft move-generation benchmarks.
* `task bench:search -- --depth 3 --repeats 3`: Benchmark cold search across fixed start, tactical, middlegame, and endgame positions.
* `task bench:matches -- --output /tmp/matches.json`: Run seeded, paired
  evaluator matches against Stockfish and save move-level JSON records.
* `task test` / `task test:unit` / `task test:cov`: Run test suite with 80% coverage check, fast unit mode, or detailed coverage report.
* `task check`: Run complete verification gate (lint, format-check, strict typecheck, tests).
* `task clean`: Remove build artifacts and caches.

In dashboard mode the side panel shows a live LOG view of recent engine events; add `--verbose` (`-v`) to raise console logging to DEBUG level.

The CLI menu accepts `1` to play White, `2` to play Black, and `3` or `q` to quit. In-game commands include `undo` (take back move), `moves [sq]` (list legal moves), `eval` (position breakdown), `pgn` (export PGN), `flip` (flip board view), `fen` (export FEN), and `help`.

## Python API

```python
from chess import ChessGame, GameStatus, print_board

game = ChessGame()  # Standard position, White to move
assert len(game.legal_moves()) == 20
print_board(game.board)

for notation in ("e4", "e7e5", "Nf3", "Nc6"):
    move = game.make_move(notation)  # Returns the Move that was played

assert game.turn == "w"
assert game.get_turn() == "w"
assert game.status == GameStatus.ACTIVE
assert game.get_winner() is None
print_board(game.get_board())
```

`legal_moves()` returns a tuple of legal `Move` values for the side to move. `make_move()` takes a notation string or a generated legal `Move`; an illegal move raises `ValueError` without changing the game. `board`, `turn`, `status`, and `move_num` expose the current position, player (`"w"` or `"b"`), state, and full-move number (starting at 1; incremented after Black). `get_board()` and `get_turn()` return the same board and turn. `castles(player, side)` reports castling rights for `"w"`/`"b"` and `"kingside"`/`"queenside"`.

The four states are `GameStatus.ACTIVE`, `CHECK`, `CHECKMATE`, and `STALEMATE`. `is_check()`, `is_checkmate()`, and `is_stalemate()` inspect the current position; `get_winner()` returns the winning color after checkmate and `None` otherwise. `is_draw()` evaluates stalemate, the 50-move rule (`is_fifty_moves()`), threefold repetition (`is_threefold_repetition()`), and insufficient material (`is_insufficient_material()`). Those rule draws do not change `status` and do not make `is_stalemate()` true; `legal_moves()` still lists any escapes. Positions can be serialized and loaded via `to_fen()` and `from_fen()`. Checkmate, stalemate, and `is_draw()` end a played game; the library rejects further moves only for checkmate and stalemate.

A `Move` records `piece`, `from_square`, `to_square`, `captured_piece`, `special`, and `promotion_to`. Its `special` value is one of `"none"`, `"castle_k"`, `"castle_q"`, `"en_passant"`, or `"promotion"`. For a move generated from the current position, `move.execute(board)` returns a new board; use `game.make_move(move)` to update the game, including turn, rights, en-passant target, and status.

## Search and engine configuration

The built-in engine defaults to depth 3, quiescence search, a bounded
transposition table, and the `basic` evaluator. The `positional` evaluator is
available as an opt-in profile; current paired matches do not establish a
strength improvement. Depth accepts integers from 1 through 8. Configure a
reusable searcher directly:

```python
from chess import ChessGame
from chess.engine import EngineConfig, SearchEngine

engine = SearchEngine(EngineConfig(search_depth=4))
game = ChessGame()
move = engine.choose_move(game)
print(move)
```

`EngineConfig` also accepts `quiescence`, `quiescence_depth`,
`transposition_table_size`, and `evaluation_profile` (`"basic"` or
`"positional"`). `SearchEngine` retains bounded transpositions and move-ordering
history between searches; call `clear()` at match boundaries. It is not
thread-safe; use one `SearchEngine` per concurrent worker. Per-search `depth`
and `quiescence` arguments override the configured defaults.

Set the same default depth at application entry points with
`python -m chess.engine --depth 4`, `python -m chess.web --depth 4`, or
`python -m chess.stockfish --depth 4`. The UCI server advertises a `Depth`
option; web AI requests may supply an optional `depth` to override the app's
configured value.

Search uses iterative deepening, alpha-beta bounds, hash, capture/promotion,
killer, and history ordering, mate-distance scores, and capture quiescence.
The `positional` evaluator tapers king activity by game phase and scores bishop
pairs plus isolated, doubled, and passed pawns.
Run `uv run --locked python -m chess.bench --depth 3 --repeats 3` to benchmark
fixed positions; timings are observations, not test assertions.
Run `python -m chess.match_bench --output /tmp/matches.json` for seeded,
color-balanced `basic`/`positional` matches against Stockfish with a shared
requested 100 ms per-move budget across skill levels 0 and 5. The JSON stores
schedule seed, complete UCI histories, per-ply elapsed time, and pychess search
stats. Use `--mode depth --depth 3` for fixed-depth comparisons.

## Boards and squares

Squares are `(row, column)` pairs: row 0 is rank 8, row 7 is rank 1; column 0 is file a, column 7 is file h. Thus `a8` is `(0, 0)`, `a1` is `(7, 0)`, and `e4` is `(4, 4)`.

```python
from chess import Board, Piece, PieceType, notation_to_coords, square_notation

board = Board.from_notation()  # Standard 32-piece position; Board() is empty
assert notation_to_coords("e4") == (4, 4)
assert square_notation(4, 4) == "e4"
pawn = Piece(PieceType.PAWN, "w")
updated = board.with_piece(notation_to_coords("e4"), pawn)
assert board.get(notation_to_coords("e4")) is None
assert updated.get(notation_to_coords("e4")) == pawn
```

`Board.with_piece(square, piece_or_none)` creates a new board and never changes the original; use `None` to clear a square. `Board.squares` is read-only. `Board.get()` and the coordinate conversion helpers raise `ValueError` for invalid squares. A custom `ChessGame(board=..., turn="b", castling_rights=frozenset(...), en_passant=...)` requires both kings; a custom board has no castling rights by default.

## Moves and notation

Coordinate strings use source and destination, such as `e2e4` or `e7e8=Q`. SAN-like strings include `e4`, `Nf3`, `exd5`, `Bxh7`, disambiguated moves such as `Nge7`, promotion `e8=Q`, and castling `O-O` or `O-O-O` (case-insensitive for castling). Promotion requires `Q`, `R`, `B`, or `N`. Optional `+` and `#` suffixes must correctly describe check or checkmate. Only currently legal moves are accepted; castling requires an unobstructed, unattacked king path and available rights, while en passant must be taken immediately after the opponent's two-square pawn push.

For example, Fool's Mate ends in Black checkmating White:

```python
from chess import ChessGame, GameStatus

game = ChessGame()
for notation in ("f3", "e5", "g4", "Qh4#"):
    game.make_move(notation)
assert game.status == GameStatus.CHECKMATE
assert game.get_winner() == "b"
```

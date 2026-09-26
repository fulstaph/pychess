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

the notebooks are ordered: [01 — board and coordinates](notebooks/01_board_and_coordinates.ipynb), [02 — moves and rules](notebooks/02_moves_and_rules.ipynb), then [03 — engine decisions](notebooks/03_engine_decisions.ipynb). run each notebook's cells top-to-bottom in a separate kernel.

If [go-task](https://taskfile.dev/) is installed, run `task` to list available tasks:

* `task play` / `task play:tui`: Start terminal chess game (optional rich side-by-side dashboard with Unicode pieces).
* `task play:large` / `task play:giant`: Start game with scaled multi-line piece art (sizes 2 and 4).
* `task web`: Start local web chessboard UI in browser (`http://localhost:8000`) powered by Lichess Chessground.
* `task play:stockfish`: Play interactively against Stockfish in the terminal.
* `task uci`: Run pychess as a standard Universal Chess Interface (UCI) engine server.
* `task bench`: Run canonical perft move-generation benchmarks.
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

The four states are `GameStatus.ACTIVE`, `CHECK`, `CHECKMATE`, and `STALEMATE`. `is_check()`, `is_checkmate()`, and `is_stalemate()` inspect the current position; `get_winner()` returns the winning color after checkmate and `None` otherwise. `is_draw()` evaluates stalemate, the 50-move rule (`is_fifty_moves()`), threefold repetition (`is_threefold_repetition()`), and insufficient material (`is_insufficient_material()`). Positions can be serialized and loaded via `to_fen()` and `from_fen()`. Checkmate and stalemate end the game; further moves are rejected.

A `Move` records `piece`, `from_square`, `to_square`, `captured_piece`, `special`, and `promotion_to`. Its `special` value is one of `"none"`, `"castle_k"`, `"castle_q"`, `"en_passant"`, or `"promotion"`. For a move generated from the current position, `move.execute(board)` returns a new board; use `game.make_move(move)` to update the game, including turn, rights, en-passant target, and status.

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

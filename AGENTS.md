# Repository Guidelines

## Project Overview

`chess` (`pychess`) is a type-safe chess engine and terminal game built for **Python 3.14+**. The package provides purely functional board representations, two-stage legal move generation, algebraic/coordinate notation parsing, and an interactive human-versus-engine terminal interface alongside educational Jupyter notebooks.

Key characteristics:
- **Strict Immutability**: All core domain models (`Board`, `Piece`, `Move`, `GameState`) are immutable or frozen. Board transitions produce fresh board instances via copy-on-write persistent data structures.
- **Engine Dependencies**: The chess engine core (`piece`, `move`, `game`, `attacks`, `helpers`, `engine`, `uci`, `stockfish`, `tui`) relies exclusively on Python 3.14 standard library primitives. The web UI layer (`chess/web.py`) is a FastAPI + Uvicorn application — the only non-engine third-party dependencies.
- **Strict Static Typing**: Annotated with modern Python 3.14 syntax (`type` aliases, pipe unions, `Literal`), type-checked strictly with `mypy`, and distributed with a PEP 561 `py.typed` marker.

---

## Architecture & Data Flow

The codebase is organized into four distinct functional layers:

```
┌────────────────────────────────────────────────────────┐
│             Application & Interface Layer              │
│          chess/engine.py     |  notebooks/*.ipynb     │
└───────────────────────────┬────────────────────────────┘
                            │
┌───────────────────────────▼────────────────────────────┐
│              Game State & Legality Façade              │
│         chess/game.py (ChessGame, GameState)           │
└───────────────────────────┬────────────────────────────┘
                            │
┌───────────────────────────▼────────────────────────────┐
│             Attack & Geometry Query Layer              │
│         chess/attacks.py (Raycasting, Checks)          │
└───────────────────────────┬────────────────────────────┘
                            │
┌───────────────────────────▼────────────────────────────┐
│            Representation & Value Object Layer         │
│   chess/piece.py (Board, Piece) | chess/move.py (Move) │
│              chess/helpers.py (Coordinates)            │
└────────────────────────────────────────────────────────┘
```

### 1. Representation & Value Objects (`chess/piece.py`, `chess/move.py`, `chess/helpers.py`)
- **Coordinates (`Square`)**: 0-indexed matrix coordinate `(row, col)`:
  - `row 0` is Rank 8; `row 7` is Rank 1.
  - `col 0` is File a; `col 7` is File h.
  - Example: `a8` $\rightarrow$ `(0, 0)`, `e4` $\rightarrow$ `(4, 4)`, `a1` $\rightarrow$ `(7, 0)`.
- **`Board`**: Immutable 8×8 grid stored as `tuple[tuple[Piece | None, ...], ...]`. Methods `with_piece(square, piece)` and `_updated(changes)` return new `Board` instances without mutating existing state.
- **`Piece`**: `@dataclass(frozen=True, slots=True)` with `type: PieceType` and `color: Color` (`"w"` or `"b"`).
- **`Move`**: `@dataclass(frozen=True, slots=True)` capturing `piece`, `from_square`, `to_square`, `captured_piece`, `special` (`"none"`, `"castle_k"`, `"castle_q"`, `"en_passant"`, `"promotion"`), and `promotion_to`.
  - `move.execute(board) -> Board`: Pure local transformation returning a new `Board` after verifying move preconditions.

### 2. Spatial Attack Geometry (`chess/attacks.py`)
- Independent of turn order or game history; operates strictly on `(board, square, color)`.
- Stepping pieces (Knights, Kings) use direct directional delta lookups (`KNIGHT_STEPS`, `KING_STEPS`).
- Sliding pieces (Rooks, Bishops, Queens) use raycasting (`ORTHOGONAL`, `DIAGONAL`) that immediately stops at the first encountered piece.
- `is_square_attacked(square, attacker_color, board) -> bool` and `is_in_check(board, player) -> bool`.

### 3. Move Generation & Game State Management (`chess/game.py`)
- **Two-Stage Move Generation**:
  1. `_piece_moves(state, source, piece)`: Generates pseudo-legal candidate moves (pawn pushes, captures, en passant, sliders, and unblocked/unattacked castling paths).
  2. `_legal_moves(state)`: Simulates each candidate move via `move.execute(board)` and rejects any transition leaving the moving player's king in check (`is_in_check`).
- **`GameState`**: `@dataclass(frozen=True, slots=True)` snapshot containing `board`, `turn`, `move_num`, `status` (`GameStatus.ACTIVE`, `CHECK`, `CHECKMATE`, `STALEMATE`), `castling_rights` (`frozenset[str]`), and `en_passant` (`Square | None`).
- **`ChessGame`**: Mutable state wrapper exposing public properties (`board`, `turn`, `status`, `move_num`).
  - `make_move(move_or_notation)`: Accepts coordinate strings (e.g., `e2e4`, `e7e8=Q`), SAN notation (e.g., `e4`, `Nf3`, `exd5`, `O-O`, `e8=Q#`), or `Move` instances.
  - Validates move legality, computes updated castling rights and en passant targets, evaluates new check/mate status, and atomically swaps `self._state`.

### 4. Engine & Terminal Loop (`chess/engine.py`)
- **`choose_move(game: ChessGame) -> Move`**: Evaluates `game.legal_moves()` using a greedy capture heuristic (`PIECE_VALUES`: Q=9, R=5, B=3, N=3, P=1), defaulting to the first generated move on ties. (Roadmap: upgrading to depth-2 alpha-beta minimax per `SEARCH_OPPONENT_PLAN.md`).
- **`main()`**: Synchronous REPL driving side selection, rendering boards via `print_board()`, processing human input with retry on invalid moves, and executing AI responses.

### 5. Logging (`chess/log.py`)
- **Module loggers** live under the `"chess"` root (e.g. `chess.game`, `chess.web`); the standard library propagates them upward, so no handler is ever attached inside library modules.
- **`LogRing`** is a bounded FIFO of `LogEntry` records (timestamp, level, name, message) fed by `LogHandler`; the TUI dashboard renders its tail as a live LOG panel.
- **`setup_logging(level)`** is idempotent and called only by CLI entry points (`chess.engine`, `chess.stockfish`, `chess.web`), never by library code.

---

## Key Directories

```
pychess/
├── chess/                 # Core engine package
│   ├── attacks.py         # Attack tables, raycasting, check detection
│   ├── game.py            # ChessGame façade, GameState, legal move generator, SAN parser
│   ├── helpers.py         # Coordinate conversions, square notation, PlayerColor enum
│   ├── log.py             # LogRing ring buffer, LogHandler, setup_logging, get_logger
│   ├── move.py            # Move dataclass and pure board execution logic
│   ├── piece.py           # Piece dataclass, PieceType enum, immutable 8x8 Board
│   ├── py.typed           # PEP 561 marker for inline type annotations
│   └── engine.py          # Minimax AI and terminal CLI game loop
├── notebooks/             # Educational Jupyter notebooks (01_board, 02_moves, 03_engine)
├── tests/                 # Complete pytest test suite (zero-mock, functional)
│   ├── test_board.py      # Board construction, coordinates, immutability
│   ├── test_cli.py        # Subprocess integration tests for the CLI loop
│   ├── test_game.py       # Move execution purity, turn transitions, invalid move atomicity
│   ├── test_notation.py   # SAN parsing, disambiguation, check/mate suffix validation
│   ├── test_rules.py      # Perft tree walks (depth 1-3), pins, checkmate, castling rights
│   └── test_special_moves.py # En passant expiration, promotion, slider obstruction, stalemate
└── .github/workflows/     # CI workflow definitions (tests.yml)
```

---

## Development Commands

All tasks can be executed via [go-task](https://taskfile.dev/) (`task <cmd>`) or directly with Astral `uv`:

### Environment & Installation
```bash
# Install locked development environment (Python 3.14, mypy, pytest, ruff)
task install
# Or via uv directly:
uv sync --python 3.14 --extra dev --locked

# Install notebook dependencies (jupyterlab, ipykernel, nbclient)
uv sync --python 3.14 --extra dev --extra notebooks --locked
```

### Running the Application
```bash
# Start the interactive terminal chess game
task play
# Or via uv directly:
uv run --locked python -m chess.engine

# Launch educational Jupyter notebooks
task notebooks
# Or via uv directly:
uv run --locked --extra notebooks jupyter lab notebooks
```

### Quality Verification Pipeline
```bash
# Run complete verification gate (lint, format-check, typecheck, test)
task check

# 1. Linting
task lint
# uv: uv run --locked --extra dev ruff check chess tests notebooks

# 2. Formatting check (read-only)
task format-check
# uv: uv run --locked --extra dev ruff format --check chess tests

# 3. Apply formatting
task format
# uv: uv run --locked --extra dev ruff format chess tests

# 4. Strict Type Checking (chess/ package only)
task typecheck
# uv: uv run --locked --extra dev mypy chess

# 5. Tests with 80% coverage enforcement
task test
# uv: uv run --locked --extra dev pytest --cov=chess --cov-fail-under=80 -q
```

### Packaging & Builds
```bash
# Build wheel and sdist distributions
task build
# uv: uv build
```

---

## Code Conventions & Common Patterns

### 1. Immutability & Persistence (Critical)
- **Never mutate objects in-place**. Always return a new instance or copy:
  ```python
  # CORRECT: functional update returns a new Board instance
  updated_board = board.with_piece(square, piece)

  # WRONG: modifying internal state in-place
  board.squares[row][col] = piece  # Raises TypeError (tuples are immutable)
  ```
- Use `@dataclass(frozen=True, slots=True)` for domain models (`Piece`, `Move`, `GameState`).
- State mutation in `ChessGame` is atomic: construct the entire next `GameState` before assigning `self._state`.

### 2. Error Handling & State Atomicity
- **Fail Fast & Preserve State**: Functions must raise `ValueError` for invalid moves, coordinates, piece placements, or notation before applying changes.
- If `game.make_move()` raises `ValueError`, the game state (`board`, `turn`, `move_num`, `castling_rights`) must remain completely unchanged.
- Validate notation suffixes: If SAN notation contains `+` (check) or `#` (checkmate), the parser must verify that the resulting position actually results in `CHECK` or `CHECKMATE`; otherwise, raise `ValueError`.

### 3. Naming Conventions
- **Classes**: `PascalCase` (`ChessGame`, `PieceType`, `GameState`, `Board`, `Move`).
- **Functions & Methods**: `snake_case` (`is_square_attacked`, `make_move`, `notation_to_coords`, `square_notation`).
- **Constants**: `UPPER_SNAKE_CASE` (`KNIGHT_STEPS`, `KING_STEPS`, `ORTHOGONAL`, `DIAGONAL`, `PIECE_VALUES`).
- **Private/Internal Helpers**: Leading underscore (`_piece_moves`, `_castling_moves`, `_status`, `_san`, `_other`).
- **Type Aliases**: Modern Python 3.14 statement syntax:
  ```python
  type MoveSpecial = Literal["none", "castle_k", "castle_q", "en_passant", "promotion"]
  Color = Literal["w", "b"]
  Square = tuple[int, int]
  ```

### 4. Dependency & Execution Patterns
- **Pure Functions**: Attack queries (`attacks.py`) and move transitions (`move.py`) are pure functions taking snapshots and returning values.
- **Synchronous Engine Execution**: The engine core and CLI are fully synchronous — no `asyncio`, threads, or background event loops. The CLI operates via standard blocking `input()` / `print()` in a loop. (The FastAPI web layer is the only asynchronous surface.)
- **Subprocess CLI Interaction**: Interactive terminal tests use `subprocess.run` with `PYTHONUNBUFFERED=1` and explicit `timeout` parameters.

---

## Important Files

| File Path | Description & Purpose |
|---|---|
| `chess/__init__.py` | Public library entry point exporting `ChessGame`, `Board`, `Piece`, `Move`, and helpers. |
| `chess/game.py` | Core game loop façade, `GameState` snapshot, legal move generation, and SAN parsing. |
| `chess/piece.py` | `Piece` and persistent 8×8 `Board` class definition. |
| `chess/move.py` | `Move` dataclass and pure board execution logic. |
| `chess/attacks.py` | Threat detection, raycasting, check tests, and king location algorithms. |
| `chess/helpers.py` | Coordinate string to `(row, col)` conversion (`notation_to_coords`, `square_notation`). |
| `chess/engine.py` | Application CLI entry point (`main()`) and AI move selector (`choose_move`). |
| `chess/py.typed` | PEP 561 package marker declaring inline typing support. |
| `pyproject.toml` | Build definitions, runtime constraints (`>=3.14`), optional dependencies, Mypy and Ruff configs. |
| `Taskfile.yml` | Standardized task runner definitions (`task test`, `task check`, `task play`, etc.). |
| `uv.lock` | Pinned dependency lockfile for deterministic development environments. |
| `.python-version` | Pinned Python runtime version (`3.14`). |
| `CHESS_TYPE_SAFETY_PLAN.md` | Architectural specification for strict typing and Mypy boundary rules. |
| `SEARCH_OPPONENT_PLAN.md` | Roadmap specification for depth-2 minimax search and positional evaluation. |

---

## Runtime & Tooling Preferences

- **Python Runtime**: **Python `>=3.14`** is strictly required. Features such as Python 3.14 type aliases (`type Name = ...`) and standard library improvements are actively used.
- **Environment & Package Manager**: **Astral `uv`** is the mandatory tool for environment sync and execution (`uv sync`, `uv run`).
- **Build Backend**: Standard PEP 517/621 packaging using `setuptools>=61` and `wheel`.
- **Zero GUI/Display Requirements**: No GTK, PyGObject, Cairo, or X11/Wayland dependencies. The app runs headlessly and in pure terminal environments.
- **Linter & Formatter**: **Ruff** targeting Python 3.14 with an 88-character line limit. Comprehensive strict lint rules enforced across 30 rule families (`E`, `W`, `F`, `I`, `B`, `UP`, `RUF`, `S`, `SIM`, `PERF`, `PIE`, `C4`, `RET`, `RSE`, `SLOT`, `FLY`, `YTT`, `A`, `ASYNC`, `BLE`, `PGH`, `FURB`, `LOG`, `G`, `ICN`, `T10`, `DTZ`, `EXE`, `FA`, `N`).
- **Static Type Checking**: **Mypy in strict mode** (`strict = true`).
  - **CRITICAL**: Mypy is intentionally scoped only to `files = ["chess"]`. **Never run Mypy on `tests/`**. The test suite deliberately passes malformed types to verify runtime boundary validation (e.g. `Board().get((True, 0))`).

---

## Testing & QA

### Test Architecture
- **Framework**: `pytest>=8,<9` with `pytest-cov>=5,<8`.
- **Coverage Gate**: Minimum **80% code coverage** enforced via `--cov=chess --cov-fail-under=80`.
- **Zero-Mock Policy**: No `unittest.mock`, stubs, or monkeypatching. Because the core engine is pure and immutable, tests construct real boards and games directly.
- **No Test Classes**: Tests use standalone functions (`def test_*():`) following the Arrange-Act-Assert pattern.
- **In-Test Iteration**: Parameterization is typically implemented via loops within test functions rather than `@pytest.mark.parametrize` to guarantee atomic state rollback verification.

### Verification Techniques
1. **Perft (Performance Test)**: `tests/test_rules.py` implements a recursive legal move tree traversal (`perft`) matching canonical chess benchmarks:
   - Depth 1: 20 nodes
   - Depth 2: 400 nodes
   - Depth 3: 8,902 nodes
2. **Atomic Failure Testing**: Tests verify that attempting invalid moves, illegal notation, or moves into check raises `ValueError` while leaving `game.board` and `game.turn` identical to the pre-move state.
3. **Headless CLI Integration**: `tests/test_cli.py` spawns `python -m chess.engine` via `subprocess.run`, piping scripted inputs and asserting output/exit codes without traceback.
4. **CI Smoke Verification**: `.github/workflows/tests.yml` builds an isolated distribution wheel into a fresh virtual environment and asserts module importability and move generation.

### Running Tests
```bash
# Run full suite with coverage check
task test

# Run a specific test module
uv run --locked --extra dev pytest tests/test_rules.py

# Run a single test function with verbose output
uv run --locked --extra dev pytest tests/test_rules.py::test_opening_perft_counts_all_legal_replies -v
```

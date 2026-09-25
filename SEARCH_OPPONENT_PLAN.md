# Search opponent

## Context
The CLI opponent in `chess/simple_engine.py` picks the legal move with the highest captured-piece value and otherwise keeps `legal_moves()` order. From the start that order begins `a2a3`, so White plays `a2a3`. Replace that selector with a depth-limited minimax over the existing `ChessGame` rules, still with no runtime dependencies, so the advertised engine develops and sees one reply.

## Approach

1. **Branch a game without mutating it.** In `chess/game.py`, extract the transition currently in `ChessGame.make_move` (`_next_rights`, en-passant target, `Move.execute`, turn flip, full-move increment, `_status`) into `_advanced(state: GameState, move: Move) -> GameState`. `make_move` still selects one current legal move and assigns `self._state = _advanced(...)`. Add `ChessGame.after(self, move: str | Move) -> ChessGame`: same selection and the same `ValueError` on an illegal string or a `Move` that is not exactly one current legal move; return a new instance whose `_state` is `_advanced`; leave the receiver's board, turn, status, move number, and castling rights unchanged. Construct the sibling with `object.__new__(ChessGame)` so `__init__` does not reset the move number or rights. No other public game method changes.

2. **Score positions with fixed material and central bonuses.** In `chess/simple_engine.py`, replace `PIECE_VALUES` and the capture-only `choose_move`. Add `evaluate(board: Board) -> int`, white-positive. Material: pawn 100, knight 320, bishop 330, rook 500, queen 900, king 0. Add the piece-square value from the white-orientation tables below; for a black piece, look up row `7 - row` and the same column, then subtract. Every other square is 0. No mobility, no castling bonus, no passed-pawn term.

   Pawn table, row 0 = rank 8:

   - row 3, columns 3 and 4 (`d5`, `e5`): 20
   - row 4, columns 3 and 4 (`d4`, `e4`): 30

   Knight table:

   - row 5, columns 2 and 5 (`c3`, `f3`): 15
   - row 2, columns 2 and 5 (`c6`, `f6`): 15

3. **Search two plies with alpha-beta minimax.** `choose_move(game: ChessGame, depth: int = 2) -> Move`. Reject `depth < 1` with `ValueError`. If `legal_moves()` is empty, raise `ValueError`; the CLI must still return before calling it on checkmate or stalemate. Search only through `game.after(move)`, never `make_move` and never `copy.deepcopy`. Iterate moves in `legal_moves()` order and replace the selected move only when the score is strictly better for the side to move, so equal scores keep the earlier move. White maximizes and Black minimizes `evaluate`. At a node, checkmate returns `100000 - ply` if the side to move is Black and `-100000 + ply` if it is White (`ply` is 0 at the root). Stalemate returns 0, not `evaluate`. Depth 0 returns `evaluate`. Alpha-beta window is `-1000000` to `1000000`; pruning must not change the move selected by the strict-improvement rule. No quiescence, iterative deepening, opening book, or transposition table. `main` calls `choose_move(game)` and still prints `Engine: ` plus the existing coordinate string from `move_notation`.

   With these values and depth 2, the unique selected move from the standard start is `d2d4`. After `e2e4`, Black's unique selected move is `e7e5`. A one-move checkmate outranks any capture. A free queen outranks a quiet move.

4. **Lock those results in the existing CLI tests and one engine test module.** Extend `tests/test_cli.py`: `printf` input `2\nq` must print `Engine: d2d4` before `Your move`; input `1\nnotamove\ne4\nq` must print `Engine: e7e5`. Add `tests/test_engine.py` importing `choose_move` from `chess.simple_engine`. Assert start-position `choose_move` is `d2d4` and does not change the original game; a custom position with Black to move and an immediate `Qh4` mate selects that mate rather than a pawn capture; a position with a hanging white queen selects the capture. Run the new assertions before replacing `choose_move`, then make them pass. Keep the 15-second CLI timeout; depth 2 stays inside it.

## Critical files & anchors
- `chess/simple_engine.py:21-28` — capture-only `choose_move` to replace.
- `chess/game.py:192-209` — transition to share between `make_move` and `after`.
- `tests/test_cli.py:23-27` — Black-side CLI proof that currently allows `a2a3`.

## Verification
From the repository root, with `uv` and Python 3.12:

- `uv run --extra dev pytest tests/test_engine.py tests/test_cli.py -q` passes.
- `uv run --extra dev pytest --cov=chess --cov-fail-under=80 -q` passes.
- `printf '%s\n' 2 q | uv run python -m chess.simple_engine` prints `Engine: d2d4` and exits 0 with no traceback.
- `printf '%s\n' 1 e4 q | uv run python -m chess.simple_engine` prints `Engine: e7e5`.

## Assumptions & contingencies
- Depth stays 2. If a depth-2 search exceeds the 15-second CLI test, do not lower depth or remove alpha-beta; the search is accidentally regenerating moves or copying games, and that implementation is wrong.
- The piece-square values above are the evaluation. Do not tune them if `d2d4` / `e7e5` fail; the move generator or minimax tie-break is wrong.

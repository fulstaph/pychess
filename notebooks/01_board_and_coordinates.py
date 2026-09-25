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
# # 1. Boards and coordinates
#
# This lesson uses the repository's `chess` package throughout. By the end, you will be able to read and convert square coordinates, inspect the starting board, and explain why changing a board or applying a move creates a new value.
#
# Chess notation names a square by **file** (a-h, left to right) and **rank** (1-8, White's side to Black's side). The package stores squares as zero-based `(row, column)` pairs in a matrix viewed from Black's side: row `0` is rank `8`, row `7` is rank `1`; column `0` is file `a`.

# %%
from chess import (
    Board,
    Move,
    Piece,
    PieceType,
    notation_to_coords,
    print_board,
    square_notation,
)

sq = notation_to_coords

# %% [markdown]
# ## Translate between chess and matrix coordinates
#
# The rank is counted from the top of the matrix, so convert a rank `r` to row `8 - r`. Files map directly to columns: `a → 0`, `b → 1`, and so on. The inverse helper turns a row and column back into chess notation.

# %%
assert sq('a8') == (0, 0)
assert sq('e4') == (4, 4)
assert square_notation(7, 0) == 'a1'
print('a8 ->', sq('a8'))
print('e4 ->', sq('e4'))
print('(7, 0) ->', square_notation(7, 0))

# %% [markdown]
# ## Inspect the starting position
#
# `Board.from_notation()` builds the standard chess position. `Board.get()` accepts the same `(row, column)` representation used by the conversion helper, and `print_board()` displays the board with rank and file labels.

# %%
starting_board = Board.from_notation()
piece_count = sum(piece is not None for row in starting_board.squares for piece in row)
assert piece_count == 32
assert starting_board.get(sq('e1')) == Piece(PieceType.KING, 'w')
assert starting_board.get(sq('e2')) == Piece(PieceType.PAWN, 'w')
print('Pieces on the board:', piece_count)
print('White king on e1:', starting_board.get(sq('e1')))
print('White pawn on e2:', starting_board.get(sq('e2')))
print_board(starting_board)

# %% [markdown]
# ## Update a board without changing it
#
# `Board()` creates an empty board. `with_piece(square, piece)` returns a new board snapshot; it does not edit the original. This persistent-value design makes it safe to retain earlier positions while exploring alternatives.

# %%
empty_board = Board()
white_pawn = Piece(PieceType.PAWN, 'w')
pawn_board = empty_board.with_piece(sq('e2'), white_pawn)
assert empty_board.get(sq('e2')) is None
assert pawn_board.get(sq('e2')) == white_pawn
print('Original e2:', empty_board.get(sq('e2')))
print('New board e2:', pawn_board.get(sq('e2')))

# %% [markdown]
# ## A move is another board transition
#
# **Predict, then run:** after moving the pawn from e2 to e4, what will `pawn_board` contain on e2 and e4? What will the returned board contain?
#
# `Move.execute(board)` applies the transition described by a `Move` and returns a new `Board`. It does **not** check complete chess legality, whose turn it is, or game history; use `ChessGame.make_move()` for those rules.

# %%
pawn_move = Move(white_pawn, sq('e2'), sq('e4'))
moved_board = pawn_move.execute(pawn_board)
assert pawn_board.get(sq('e2')) == white_pawn
assert pawn_board.get(sq('e4')) is None
assert moved_board.get(sq('e2')) is None
assert moved_board.get(sq('e4')) == white_pawn
print('Source board: e2 =', pawn_board.get(sq('e2')), ', e4 =', pawn_board.get(sq('e4')))
print('Returned board: e2 =', moved_board.get(sq('e2')), ', e4 =', moved_board.get(sq('e4')))

# %% [markdown]
# ## Takeaways
#
# - A square is stored as `(row, column)`; ranks run opposite to matrix row numbers.
# - `Board.from_notation()` creates the familiar 32-piece starting position, while `Board()` is empty.
# - `with_piece()` and `Move.execute()` return new boards, preserving the source snapshot.
# - Board transitions and complete game rules are separate: legal play belongs to `ChessGame`.
#
# Next: [Moves and rules](02_moves_and_rules.ipynb).

# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # 6. Board representations and attack geometry
#
# Goals: relate human coordinates to integer indices; compare mailbox, 0x88, and bitboards; understand why sliding attacks stop at blockers; and compare a tiny independent ray walker with the project's actual attack detector.
#
# A representation is not just storage: it shapes the cost and complexity of move generation. The [Chess Programming Wiki: Board Representation](https://www.chessprogramming.org/Board_Representation), [0x88](https://www.chessprogramming.org/0x88), [Bitboards](https://www.chessprogramming.org/Bitboards), and [Magic Bitboards](https://www.chessprogramming.org/Magic_Bitboards) describe the design space.

# %%
from chess import Board, Piece, PieceType, notation_to_coords
from chess.attacks import is_square_attacked

# %% [markdown]
# ## Coordinate conventions
#
# In this project, a square is `(row, column)` on an 8-by-8 tuple-of-tuples: row 0 is rank 8, and column 0 is file a. Bitboard examples conventionally number a1 as bit 0, increasing across ranks. These conventions are both valid, but conversions must be explicit.

# %%
def coords_to_bit_index(square: tuple[int, int]) -> int:
    row, col = square
    if not (0 <= row < 8 and 0 <= col < 8):
        raise ValueError("square outside board")
    return (7 - row) * 8 + col


def bit_index_to_coords(index: int) -> tuple[int, int]:
    if not 0 <= index < 64:
        raise ValueError("bit index outside board")
    rank_from_bottom, col = divmod(index, 8)
    return 7 - rank_from_bottom, col

assert coords_to_bit_index(notation_to_coords("a1")) == 0
assert coords_to_bit_index(notation_to_coords("h8")) == 63
assert bit_index_to_coords(28) == notation_to_coords("e4")
print("e4 maps to bit", coords_to_bit_index(notation_to_coords("e4")))

# %% [markdown]
# ## Three representation families
#
# **Mailbox** stores a piece per square. The project's `Board` is specifically an immutable tuple-of-tuples, not a mutable 10-by-12 mailbox: simple indexing and clear snapshots are advantages; scanning/ray steps use ordinary coordinate checks. **0x88** uses a 128-cell array; indices with `index & 0x88` set are off-board, simplifying directional steps. **Bitboards** encode square sets as integers; unions, intersections, and population counts are compact, but updating and extracting individual squares requires bit operations and consistent indexing. Python integers are arbitrary precision (the teaching demo below keeps values within 64 bits); the project does not implement bitboards, magic indexing, or magic-bitboard move generation.
#
# Magic tables trade memory and initialization for fast occupancy-to-attack lookup. They are one strategy for sliding moves, not a different chess rule. A ray walker is far easier to audit and handles blockers naturally, but repeats work.

# %%
# A 0x88 index reserves a four-bit stride per rank.
def square_0x88(row: int, col: int) -> int:
    return row * 16 + col

assert square_0x88(0, 0) == 0
assert square_0x88(7, 7) == 0x77
assert (square_0x88(7, 8) & 0x88) != 0

# Rank 2 pawns as a conventional a1=0 bitboard.
white_home_pawns = 0xFF00
assert white_home_pawns.bit_count() == 8
print("0x88 detects an off-board step; home-rank bit count:", white_home_pawns.bit_count())

# %% [markdown]
# ## Sliding rays and the first blocker
#
# Rooks, bishops, and queens trace orthogonal or diagonal rays. Every square before the first occupied square is attacked; the occupied square itself is attacked, but squares beyond it are not. Color does not affect blocking: a friendly piece blocks just as an enemy piece does. This independent demonstration returns attacked squares for one rook and can be checked against `chess.attacks.is_square_attacked`.

# %%
ORTHOGONAL = ((-1, 0), (1, 0), (0, -1), (0, 1))

def rook_attack_mask(board: Board, origin: tuple[int, int]) -> set[tuple[int, int]]:
    attacked = set()
    for dr, dc in ORTHOGONAL:
        row, col = origin
        while True:
            row, col = row + dr, col + dc
            if not (0 <= row < 8 and 0 <= col < 8):
                break
            attacked.add((row, col))
            if board.get((row, col)) is not None:
                break
    return attacked

rook_board = Board().with_piece(notation_to_coords("d4"), Piece(PieceType.ROOK, "w"))
rook_board = rook_board.with_piece(notation_to_coords("d6"), Piece(PieceType.PAWN, "b"))
mask = rook_attack_mask(rook_board, notation_to_coords("d4"))
assert notation_to_coords("d6") in mask
assert notation_to_coords("d7") not in mask
assert notation_to_coords("d5") in mask
# Compare the independent mask with the package's attack query on empty ray squares.
for target in ("d5", "d6", "d7", "a4", "h4"):
    coords = notation_to_coords(target)
    assert (coords in mask) == is_square_attacked(coords, "w", rook_board)
print("Rook ray contains d5 and blocker d6, but not d7")

# %% [markdown]
# ## Attack masks versus legal moves
#
# An attack map describes geometric control, not a move list. Pinned pieces still attack squares for king-safety purposes; an attacked square can contain a friendly piece; and legal moves additionally depend on turn, check, castling rights, en-passant state, and promotion. The project's attack module uses coordinate stepping and stops at the first blocker, rather than bitboard lookup tables.
#
# Tradeoff summary: mailbox is approachable and easy to validate; 0x88 makes off-board detection cheap in a flat array; bitboards make set-wide operations fast and often simplify bulk pawn/king calculations. No representation is universally best: complexity, language/runtime, and the operations a search performs matter.

# %% [markdown]
# ## Takeaways
#
# - Convert coordinate conventions at boundaries; a1=0 in bitboards differs from row 0 being rank 8 in this package.
# - Sliding rays include their first blocker and exclude everything behind it.
# - `Board` is an immutable tuple grid; this project does not use bitboards or magic indexing.
# - Attack geometry is not the same as legal move generation.
#
# Reflect: which representation makes debugging easiest? Which operations would benefit most from set-wide bitwise calculations?
#
# Next: [Search algorithms](07_search_algorithms.ipynb).

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
# # 20. Move-generation performance: implementation tradeoffs
#
# This lesson follows the introductory coordinate/ray comparison in lesson 06. Here we focus on the shape of a move generator: generate pseudo-legal candidates, then reject moves that leave the moving side's king attacked; represent many squares as integer masks; and weigh state-update strategies used by search. The examples are small, deterministic, and illustrative—not a replacement for the package's chess rules.
#
# References: [CPW Bitboards](https://www.chessprogramming.org/Bitboards), [CPW Move Generation](https://www.chessprogramming.org/Move_Generation), and [CPW Make_Move](https://www.chessprogramming.org/Make_Move).

# %% [markdown]
# ## Candidate generation and king safety
#
# A common architecture first emits pseudo-legal moves: piece geometry, blockers, and basic occupancy permit the move, but king safety may not. A second step tests each candidate and keeps only those that do not expose the mover's king. This separates local movement rules from a global constraint. Pin-aware generators can avoid some rejected candidates, but still need exact rule handling.
#
# This project follows that broad candidate-then-filter model: its piece generator documents pseudo-legal moves, and legal generation applies king-safety checking. Special rules and state (castling, en passant, promotion) make a full rules generator considerably more involved than geometry alone.

# %%
from chess import ChessGame

opening = ChessGame()
legal = opening.legal_moves()
assert len(legal) == 20
print("Starting-position legal candidates:", len(legal))

# %% [markdown]
# ## Bitboard knight attacks
#
# Use the conventional mapping a1=0 through h8=63. A bitboard is a Python integer whose set bits name squares. For a knight on bit `s`, shifts by 6, 10, 15, or 17 encode the eight L-shaped offsets. File masks are essential: without them, a shift can wrap from one board edge to the other. The coordinate reference below computes the same geometry directly; assertions compare the two definitions on every square.

# %%
BOARD_MASK = (1 << 64) - 1
FILE_A = sum(1 << (rank * 8) for rank in range(8))
FILE_H = FILE_A << 7
NOT_FILE_A = BOARD_MASK ^ FILE_A
NOT_FILE_H = BOARD_MASK ^ FILE_H
NOT_FILES_AB = BOARD_MASK ^ (FILE_A | (FILE_A << 1))
NOT_FILES_GH = BOARD_MASK ^ (FILE_H | (FILE_H >> 1))


def knight_attacks_bitboard(square: int) -> int:
    """Return attacked squares for a1=0..h8=63 using masked shifts."""
    if not 0 <= square < 64:
        raise ValueError("square index must be in 0..63")
    source = 1 << square
    attacks = (
        (source << 17 & NOT_FILE_A)
        | (source << 15 & NOT_FILE_H)
        | (source << 10 & NOT_FILES_AB)
        | (source << 6 & NOT_FILES_GH)
        | (source >> 17 & NOT_FILE_H)
        | (source >> 15 & NOT_FILE_A)
        | (source >> 10 & NOT_FILES_GH)
        | (source >> 6 & NOT_FILES_AB)
    )
    return attacks & BOARD_MASK


def knight_attacks_coordinates(square: int) -> set[int]:
    """Simple independent coordinate implementation, also a1=0."""
    if not 0 <= square < 64:
        raise ValueError("square index must be in 0..63")
    rank, file = divmod(square, 8)
    return {
        (rank + dr) * 8 + file + df
        for dr, df in ((2, 1), (2, -1), (-2, 1), (-2, -1),
                       (1, 2), (1, -2), (-1, 2), (-1, -2))
        if 0 <= rank + dr < 8 and 0 <= file + df < 8
    }


def bits(mask: int) -> set[int]:
    """Extract set-bit indices for comparison/inspection."""
    result = set()
    while mask:
        least_bit = mask & -mask
        result.add(least_bit.bit_length() - 1)
        mask ^= least_bit
    return result


for square in range(64):
    assert bits(knight_attacks_bitboard(square)) == knight_attacks_coordinates(square)

assert knight_attacks_bitboard(0).bit_count() == 2  # a1 -> b3, c2
print("Bitboard and coordinate knight attacks agree on all 64 squares")

# %% [markdown]
# ## Occupancy is not legality
#
# Given an attack mask, `attacks & ~own` removes destinations occupied by the moving side. This is useful for ordinary destination generation, while captures can be separated by intersecting with the opponent's occupancy. The mask remains geometric: it does not decide whether moving a pinned piece exposes its king, whether the destination is attacked, or whether castling/en-passant conditions hold. Attack maps themselves intentionally include defended friendly-occupied squares in many chess algorithms, especially for king safety.

# %%
# A small synthetic occupancy example: knight on d4, friendly piece on e6,
# enemy piece on c6. The mask arithmetic classifies destinations only.
knight_from_d4 = 3 + 3 * 8
knight_mask = knight_attacks_bitboard(knight_from_d4)
e6 = 4 + 5 * 8
c6 = 2 + 5 * 8
own_occupancy = 1 << e6
enemy_occupancy = 1 << c6
candidate_destinations = knight_mask & ~own_occupancy & BOARD_MASK
capture_destinations = candidate_destinations & enemy_occupancy
quiet_destinations = candidate_destinations & ~(own_occupancy | enemy_occupancy) & BOARD_MASK
assert not (candidate_destinations & own_occupancy)
assert capture_destinations == 1 << c6
assert not (quiet_destinations & enemy_occupancy)
print("Friendly destination removed; enemy square is a capture candidate")

# %% [markdown]
# ## Coordinates, bitboards, and lookup tables
#
# Coordinate tuples are explicit and easy to inspect; tuple-grid indexing and stepping require ordinary Python operations and boundary checks. Bitboards make union, intersection, exclusion, and counting whole square sets compact. They also demand conversion at API boundaries and careful masks/index conventions. In Python, arbitrary-precision integers are not hardware 64-bit values, so the representation's performance depends on workload and runtime; this lesson makes no timing claim.
#
# Sliding attacks are harder than a knight's fixed shifts because blockers truncate rays. Magic-bitboard designs use occupancy bits relevant to a rook/bishop origin to index precomputed attack tables. Lookup can replace repeated ray walking, at the cost of table memory, initialization or generated data, indexing/mask complexity, and cache pressure. Any occupancy or position cache also has invalidation/update costs. This package's rules generator uses coordinate-based board operations; it does not implement bitboards or magic sliding tables.

# %% [markdown]
# ## Make/unmake versus immutable successors
#
# Search engines often mutate a position with make/unmake: apply a move, recurse, then restore every changed field. This can avoid allocating a new position at each node, but restoration must correctly cover captures, rights, en-passant, clocks, hashes, and special moves; one missed field corrupts later branches. An immutable successor such as `ChessGame.after(move)` leaves its parent unchanged, which simplifies sibling search and reasoning, but creates new state/snapshot data and carries history. `make_move` mutates the game façade, while `after` returns a sibling. These are API/semantic tradeoffs, not a benchmark verdict; neither should be assumed faster without measurements on the actual workload.

# %%
parent = ChessGame()
move = parent.legal_moves()[0]
sibling = parent.after(move)
assert parent.turn == "w"
assert sibling.turn == "b"
assert parent.board != sibling.board
print("after() keeps parent to move:", parent.turn, "; sibling:", sibling.turn)

# %% [markdown]
# ## Takeaways
#
# - Separating pseudo-legal geometry from king-safety filtering keeps concerns distinct, but full legal rules need state and special-move handling.
# - Bitboards accelerate set-shaped operations conceptually; explicit file masks prevent edge wrap in shifted knight attacks.
# - Occupancy-filtered destinations are not legal moves, and attack maps are not move lists.
# - Magic sliding tables trade repeated ray work for lookup data, initialization, and cache/update costs.
# - Make/unmake and immutable successors trade restoration complexity against allocation and branch clarity.
#
# This notebook does not replace the project's rules generator. Lesson 06 introduces board representations and ray geometry; this lesson builds on those ideas to discuss candidate filtering, bitboard mask computation, lookup-table costs, and search-state transitions.

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
# # 4. Zobrist hashing and position identity
#
# This lesson builds a **Zobrist hash** — a compact fingerprint of a chess
# position. You will generate deterministic 64-bit keys, hash positions, and
# update a hash after an ordinary move by XOR-ing only the changed features.
#
# The method is described on the [Chess Programming Wiki](https://www.chessprogramming.org/Zobrist_Hashing).
# Hashes support transposition tables and position-history checks, but they are
# not collision-free. This notebook is educational; it does not add hashing to
# the engine.

# %%
import math
from collections.abc import Iterator

from chess import ChessGame, Move, PieceType, notation_to_coords

sq = notation_to_coords

# %% [markdown]
# ## The key table
#
# Zobrist hashing assigns a **pseudo-random 64-bit integer** to every
# (square, piece) combination. With 64 squares x 12 piece/color combinations,
# we need 768 random keys, plus one key for "Black to move", 16 keys for
# castling-right combinations, and eight keys for en-passant files.
#
# A fixed seed makes this demonstration reproducible across runs.

# %%
PIECE_TYPES = (
    PieceType.PAWN,
    PieceType.KNIGHT,
    PieceType.BISHOP,
    PieceType.ROOK,
    PieceType.QUEEN,
    PieceType.KING,
)
PIECE_INDEX: dict[tuple[str, str], int] = {}
for color in ('w', 'b'):
    for piece_type in PIECE_TYPES:
        PIECE_INDEX[(piece_type.value, color)] = len(PIECE_INDEX)

assert len(PIECE_INDEX) == 12

MASK64 = (1 << 64) - 1


def splitmix64(seed: int) -> Iterator[int]:
    """Yield reproducible pseudo-random 64-bit keys."""
    while True:
        seed = (seed + 0x9E3779B97F4A7C15) & MASK64
        value = seed
        value = ((value ^ (value >> 30)) * 0xBF58476D1CE4E5B9) & MASK64
        value = ((value ^ (value >> 27)) * 0x94D049BB133111EB) & MASK64
        yield (value ^ (value >> 31)) & MASK64


keys = iter(splitmix64(42))
PIECE_KEYS: list[list[int]] = [
    [next(keys) for _ in range(12)]
    for _ in range(64)
]
SIDE_KEY: int = next(keys)
CASTLING_KEYS: list[int] = [next(keys) for _ in range(16)]
EP_KEYS: list[int] = [next(keys) for _ in range(8)]

print(f'Piece keys:    {len(PIECE_KEYS)} squares x {len(PIECE_KEYS[0])} pieces')
print(f'Side key:      0x{SIDE_KEY:016X}')
print(f'Castling keys: {len(CASTLING_KEYS)}')
print(f'EP keys:       {len(EP_KEYS)}')

# %% [markdown]
# ## Hash a full position from scratch
#
# XOR the key for every occupied square, then add keys for side to move,
# castling rights, and the en-passant file. The public game API does not expose
# castling or en-passant state directly, so this lesson reads those fields from
# FEN. Production engines keep these features in game state and do not
# serialize the board for each update.

# %%
CASTLING_BITS: dict[str, int] = {'K': 1, 'Q': 2, 'k': 4, 'q': 8}


def square_index(row: int, col: int) -> int:
    """Map (row, col) to a 0-63 index (a8=0, h1=63)."""
    return row * 8 + col


def castling_mask(rights: frozenset[str]) -> int:
    return sum(CASTLING_BITS.get(right, 0) for right in rights)


def game_features(game: ChessGame) -> tuple[int, int | None]:
    """Extract castling mask and en-passant file from public FEN output."""
    _, _, rights, ep_square, *_ = game.to_fen().split()
    mask = castling_mask(frozenset() if rights == '-' else frozenset(rights))
    ep_file = ord(ep_square[0]) - ord('a') if ep_square != '-' else None
    return mask, ep_file


def zobrist_hash(game: ChessGame) -> int:
    """Compute a 64-bit Zobrist key for a full game position."""
    h = 0
    for row, squares in enumerate(game.board.squares):
        for col, piece in enumerate(squares):
            if piece is not None:
                si = square_index(row, col)
                pi = PIECE_INDEX[(piece.type.value, piece.color)]
                h ^= PIECE_KEYS[si][pi]

    if game.turn == 'b':
        h ^= SIDE_KEY

    castling, ep_file = game_features(game)
    h ^= CASTLING_KEYS[castling]
    if ep_file is not None:
        h ^= EP_KEYS[ep_file]
    return h


start_hash = zobrist_hash(ChessGame())
assert start_hash.bit_length() <= 64
print(f'Starting position hash: 0x{start_hash:016X}')

# %% [markdown]
# ## XOR is its own inverse
#
# The crucial property: `x ^ x == 0`. XOR-ing a feature key a second time
# removes that feature. An ordinary move therefore updates a hash in constant
# time: remove the moving piece from its source, remove a captured piece (if
# any), add the moving piece at its destination, and update state keys.

# %%
key = next(keys)
h = 0
h ^= key  # place the piece
h ^= key  # remove it
assert h == 0
print(f'key = 0x{key:016X}')
print(f'0 ^ key ^ key = {h}  (zero — piece removed)')

h1 = 0xAAAA_BBBB_CCCC_DDDD
h2 = h1 ^ key
h3 = h2 ^ key
assert h3 == h1
print(f'Hash survives round-trip: {h3 == h1}')

# %% [markdown]
# ## Incremental update after an ordinary move
#
# **Predict, then run:** after `1. e4`, the hash should change. The update
# function XORs only the moved/captured piece keys and old/new state keys.
# This example covers ordinary moves; castling, en passant, and promotion
# need additional piece-square XORs.
#
# The demonstration extracts FEN fields for verification because the package
# does not expose castling and en-passant state as public properties. The
# update itself receives those small features directly and does not scan the
# board.

# %%
def update_zobrist_hash(
    old_hash: int,
    move: Move,
    old_features: tuple[int, int | None],
    new_features: tuple[int, int | None],
) -> int:
    """Update a hash for an ordinary single-piece move."""
    h = old_hash
    from_idx = square_index(*move.from_square)
    to_idx = square_index(*move.to_square)
    piece_index = PIECE_INDEX[(move.piece.type.value, move.piece.color)]

    h ^= PIECE_KEYS[from_idx][piece_index]
    if move.captured_piece is not None:
        captured_index = PIECE_INDEX[
            (move.captured_piece.type.value, move.captured_piece.color)
        ]
        h ^= PIECE_KEYS[to_idx][captured_index]
    h ^= PIECE_KEYS[to_idx][piece_index]
    h ^= SIDE_KEY

    old_castling, old_ep = old_features
    new_castling, new_ep = new_features
    h ^= CASTLING_KEYS[old_castling]
    h ^= CASTLING_KEYS[new_castling]
    if old_ep is not None:
        h ^= EP_KEYS[old_ep]
    if new_ep is not None:
        h ^= EP_KEYS[new_ep]
    return h


game = ChessGame()
hash_before = zobrist_hash(game)
move_e4 = next(
    move for move in game.legal_moves()
    if move.from_square == sq('e2') and move.to_square == sq('e4')
)
assert move_e4.special == 'none'
game_after_e4 = game.after(move_e4)
hash_after_full = zobrist_hash(game_after_e4)
hash_after_incremental = update_zobrist_hash(
    hash_before,
    move_e4,
    game_features(game),
    game_features(game_after_e4),
)

assert hash_after_incremental == hash_after_full
assert hash_after_incremental != hash_before
print(f'Before 1.e4:      0x{hash_before:016X}')
print(f'After (full):      0x{hash_after_full:016X}')
print(f'After (increment): 0x{hash_after_incremental:016X}')
print(f'Incremental matches full scan: {hash_after_incremental == hash_after_full}')

# %% [markdown]
# ## Collision probability
#
# For one pair of positions, a random 64-bit key collides with probability
# about $1 / 2^{64} \approx 5.4 \times 10^{-20}$. The birthday bound grows
# with the number of distinct keys: at one billion keys the probability is
# about 2.7%, not zero. Transposition tables hold far fewer entries than the
# total nodes searched and store key signatures to reject table-index
# collisions; a true full-key collision remains possible.

# %%
n = 10**9  # one billion distinct positions
bits = 64
p_collision = 1 - math.exp(-(n * (n - 1)) / (2 * 2**bits))
print(f'Positions hashed:        {n:,}')
print(f'Hash space:              2^{bits} ≈ {2**bits:.2e}')
print(f'Birthday collision prob:  {p_collision:.2%}')
assert 0.02 < p_collision < 0.03

# %% [markdown]
# ## Transposition detection
#
# Two different move orders that reach the same position produce the same
# hash. A **transposition table** stores a search result so a position reached
# through another move order need not be searched again.
#
# **Predict, then run:** do the Italian Game positions `1.e4 e5 2.Nf3 Nc6 3.Bc4`
# and `1.e4 e5 2.Bc4 Nc6 3.Nf3` produce the same hash?

# %%
game_a = ChessGame()
for move in ('e4', 'e5', 'Nf3', 'Nc6', 'Bc4'):
    game_a.make_move(move)

game_b = ChessGame()
for move in ('e4', 'e5', 'Bc4', 'Nc6', 'Nf3'):
    game_b.make_move(move)

hash_a = zobrist_hash(game_a)
hash_b = zobrist_hash(game_b)
assert hash_a == hash_b
print(f'Path A (e4 e5 Nf3 Nc6 Bc4): 0x{hash_a:016X}')
print(f'Path B (e4 e5 Bc4 Nc6 Nf3): 0x{hash_b:016X}')
print(f'Same position, same hash:   {hash_a == hash_b}')

# %% [markdown]
# ## Threefold repetition sketch
#
# A hash history can quickly find repeated position keys. A real draw detector
# must compare the complete repetition-relevant state and handle hash
# collisions; it also needs the full game history. En-passant state must be
# normalized when no legal en-passant capture exists, and the halfmove clock
# matters separately for the fifty-move rule. This knight shuffle returns to
# the starting position twice.

# %%
def has_threefold(history: list[int]) -> bool:
    """Check whether the last position has occurred three times."""
    if not history:
        return False
    return history.count(history[-1]) >= 3


rep_game = ChessGame()
history: list[int] = [zobrist_hash(rep_game)]
knight_shuffle = [
    'Nf3', 'Nf6',
    'Ng1', 'Ng8',
    'Nf3', 'Nf6',
    'Ng1', 'Ng8',
]

for move in knight_shuffle:
    rep_game.make_move(move)
    history.append(zobrist_hash(rep_game))

assert has_threefold(history)
print(f'Moves played: {len(knight_shuffle)}')
print(f'History length: {len(history)}')
print(f'Threefold repetition detected: {has_threefold(history)}')

# %% [markdown]
# ## Takeaways
#
# - Zobrist hashing combines reproducible pseudo-random keys with XOR to fingerprint position features ([CPW: Zobrist Hashing](https://www.chessprogramming.org/Zobrist_Hashing)).
# - XOR is self-inverse, so ordinary move updates change only the affected feature keys instead of rescanning all 64 squares.
# - Transposition tables cache search results by position hash; entries also carry depth, score bounds, and a key signature ([CPW: Transposition Table](https://www.chessprogramming.org/Transposition_Table)).
# - Repetition detection needs position history and collision-aware state comparison; a hash alone is not proof of position identity.
# - The birthday bound is visible at scale: a billion distinct 64-bit keys have roughly a 2.7% chance of at least one collision.
#
# Next: [Evaluation deep dive](05_evaluation_deep_dive.ipynb).

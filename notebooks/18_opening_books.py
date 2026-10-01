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
# # 18. Opening books and book selection
#
# An opening book supplies moves before, alongside, or instead of tree search.
# This lesson builds a tiny book from recorded lines, merges transpositions,
# and selects moves by integer weights. It does not load a commercial or
# PolyGlot binary book and is not integrated into the engine.
#
# See the Chess Programming Wiki pages on [Opening Books](https://www.chessprogramming.org/Opening_Book),
# [PolyGlot](https://www.chessprogramming.org/PolyGlot), and
# [Transpositions](https://www.chessprogramming.org/Transposition).

# %%
from collections import Counter, defaultdict
from collections.abc import Mapping

from chess import ChessGame
from chess.engine import move_notation

# %% [markdown]
# ## What a book stores
#
# A book maps a position to one or more legal moves, often with weights or
# empirical statistics. The book can save search time and add variety. Its
# move choice is only as good as its source games and selection policy.
#
# A transposition can reach the same position through different move orders.
# A compact text key made from the first four FEN fields includes piece
# placement, side to move, castling rights, and the en-passant target. The
# move counters are excluded because they do not change which moves are legal
# in the opening-book example. This is a teaching key, not a complete game
# history or repetition key.

# %%
def book_position_key(game: ChessGame) -> str:
    """Return the position fields relevant to this tiny opening book."""
    return " ".join(game.to_fen().split()[:4])


same_position_a = ChessGame()
for token in ('g1f3', 'g8f6', 'g2g3', 'g7g6'):
    same_position_a.make_move(token)

same_position_b = ChessGame()
for token in ('g2g3', 'g7g6', 'g1f3', 'g8f6'):
    same_position_b.make_move(token)

assert book_position_key(same_position_a) == book_position_key(same_position_b)
print('Transposed opening position:', book_position_key(same_position_a))

# %% [markdown]
# ## Build a book from a small game collection
#
# Each example line is a sequence of coordinate moves. At every position we
# record the move played next. Repeated observations raise its weight. Real
# books need much larger, curated data; this tiny corpus only demonstrates the
# data flow and uses no downloads.

# %%
opening_lines: tuple[tuple[str, ...], ...] = (
    ('e2e4', 'e7e5', 'g1f3', 'b8c6', 'f1b5'),
    ('e2e4', 'c7c5', 'g1f3', 'd7d6', 'd2d4', 'c5d4', 'f3d4'),
    ('e2e4', 'c7c5', 'g1f3', 'd7d6', 'd2d4', 'c5d4', 'f3d4'),
    ('d2d4', 'd7d5', 'c2c4'),
    ('d2d4', 'g8f6', 'c2c4', 'e7e6'),
)

book: dict[str, Counter[str]] = defaultdict(Counter)
for line in opening_lines:
    game = ChessGame()
    for token in line:
        key = book_position_key(game)
        played = game.make_move(token)
        book[key][move_notation(played)] += 1

root_moves = book[book_position_key(ChessGame())]
assert root_moves == Counter({'e2e4': 3, 'd2d4': 2})
print('Recorded starting-position weights:', dict(root_moves))

# %% [markdown]
# ## Weighted selection without hidden randomness
#
# A weighted picker maps each integer ticket from `0` to `sum(weights) - 1`
# into a move interval. Production code can draw the ticket from a seeded or
# system random generator; accepting the ticket directly makes this example
# reproducible and easy to inspect.

# %%
def choose_weighted_move(weights: Mapping[str, int], ticket: int) -> str:
    """Choose a move for a ticket in [0, total weight)."""
    total = sum(weights.values())
    if total <= 0 or any(weight < 0 for weight in weights.values()):
        raise ValueError('Weights must be non-negative with positive total')
    if not 0 <= ticket < total:
        raise ValueError('Ticket is outside the cumulative weight range')

    for move, weight in weights.items():
        if ticket < weight:
            return move
        ticket -= weight
    raise AssertionError('A valid ticket must select a move')


sample_weights = {'e2e4': 6, 'd2d4': 3, 'c2c4': 1}
assert [choose_weighted_move(sample_weights, n) for n in (0, 6, 9)] == [
    'e2e4', 'd2d4', 'c2c4'
]
for ticket in range(sum(sample_weights.values())):
    print(f'ticket {ticket}: {choose_weighted_move(sample_weights, ticket)}')

# %% [markdown]
# ## A book move still has to be legal
#
# A file can be corrupt, stale, or keyed incorrectly. Treat a book suggestion
# as a candidate, not as an instruction to bypass the rules layer. Here we
# compare the selected move with the current game's generated legal moves.

# %%
opening_game = ChessGame()
root_key = book_position_key(opening_game)
book_suggestion = choose_weighted_move(book[root_key], ticket=0)
legal_notations = {move_notation(move) for move in opening_game.legal_moves()}
assert book_suggestion in legal_notations
played = opening_game.make_move(book_suggestion)
assert move_notation(played) == book_suggestion
print('Book suggests:', book_suggestion)
print('Legal and played:', move_notation(played))

# %% [markdown]
# ## Book policy and data risks
#
# - **Most-played** is stable but can repeat the same line.
# - **Weighted random** adds variety but may select a weak or dubious move.
# - **Score-based** policies need a trustworthy score source and enough games;
#   raw win rate is biased by player strength, color, and opening selection.
# - A book is not a transposition table: a book supplies opening choices from
#   curated data, while a transposition table caches search results.
# - Binary formats such as PolyGlot store compact position/move records; this
#   notebook deliberately uses transparent Python dictionaries instead.
#
# **Try:** add another opening line and predict which root weight changes.
# Then compare deterministic ticket selection with a seeded random draw.
#
# Next: [Parallel search](19_parallel_search.ipynb) examines concurrency and shared search work.

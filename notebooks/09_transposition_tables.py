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
# # 9. Transposition tables: caching search knowledge
#
# **Goals:** distinguish a table slot from a position key; read depth, best move,
# score, and bound fields; use exact/lower/upper bounds safely; and see why a
# position key is not always a complete search-state key. Lesson 04 introduced
# Zobrist hashing and showed that move orders can transpose. Here we build a
# tiny, deterministic teaching example—not an engine implementation.
#
# A table entry summarizes a search to a stated depth. A score is not always an
# exact value: alpha-beta cutoffs can prove only a lower or upper bound. A
# shallow entry may still suggest a move for ordering, but cannot answer a
# deeper search. Real engines make replacement and collision policies more
# The shipped engine uses a bounded transposition table keyed by the board,
# turn, castling/en-passant rights, halfmove clock, and repetition counts,
# plus quiescence and evaluation settings. Entries store depth, bound,
# mate-normalized score, and best move; full-key equality resolves hash collisions.
#
# Sources: [CPW: Transposition Table](https://www.chessprogramming.org/Transposition_Table),
# [CPW: Zobrist Hashing](https://www.chessprogramming.org/Zobrist_Hashing).

# %%
from dataclasses import dataclass

from chess import ChessGame

# Two different move orders from lesson 04 reach the same board/state.
left = ChessGame()
for move in ('e4', 'e5', 'Nf3', 'Nc6', 'Bc4'):
    left.make_move(move)
right = ChessGame()
for move in ('e4', 'e5', 'Bc4', 'Nc6', 'Nf3'):
    right.make_move(move)
assert left.to_fen().split()[:4] == right.to_fen().split()[:4]
print('Transposition FEN state fields match:', left.to_fen().split()[:4])

# %% [markdown]
# ## Entry structure and probing
#
# `key` is the full position signature stored to verify a slot; `depth` is the
# searched depth; `best_move` can help order a later search; `score` is in the
# caller's score convention; `bound` is EXACT, LOWER (true score is at least
# this), or UPPER (at most this). Probe only uses a bound if depth is sufficient.
# This tiny example uses readable strings as keys instead of the lesson 04 hash.

# %%
@dataclass(frozen=True)
class Entry:
    key: int
    depth: int
    best_move: str | None
    score: int
    bound: str


def probe(entry: Entry | None, key: int, depth: int, alpha: int, beta: int):
    """Return cutoff score or None; reject shallow and mismatching entries."""
    if entry is None or entry.key != key or entry.depth < depth:
        return None
    if entry.bound == 'EXACT':
        return entry.score
    if entry.bound == 'LOWER' and entry.score >= beta:
        return entry.score
    if entry.bound == 'UPPER' and entry.score <= alpha:
        return entry.score
    return None


exact = Entry(key=0x1234, depth=4, best_move='Nf3', score=27, bound='EXACT')
lower = Entry(key=0x1234, depth=4, best_move='Nf3', score=50, bound='LOWER')
upper = Entry(key=0x1234, depth=4, best_move='Nf3', score=-20, bound='UPPER')
assert probe(exact, 0x1234, 4, -10, 10) == 27
assert probe(lower, 0x1234, 4, -10, 40) == 50  # beta cutoff
assert probe(lower, 0x1234, 4, -10, 60) is None
assert probe(upper, 0x1234, 4, -10, 10) == -20  # alpha cutoff
assert probe(upper, 0x1234, 4, -30, 10) is None
assert probe(exact, 0x1234, 5, -100, 100) is None  # too shallow
assert probe(exact, 0x9999, 4, -100, 100) is None  # key mismatch
print('Entry:', exact)
print('Exact, bound cutoffs, depth rejection, and key check passed.')

# %% [markdown]
# ## A deterministic transposition in a toy search
#
# In this small directed acyclic game tree, root paths `A-X` and `B-X` share
# node X. Leaf scores are already expressed from the root player's perspective.
# The search is intentionally tiny and uses exact minimax, not chess move rules.
# `visited` shows the second encounter of X can reuse its depth-1 result.

# %%
TREE = {
    'root': [('A', 'a'), ('B', 'b')],
    'a': [('to-X', 'X'), ('to-Y', 'Y')],
    'b': [('to-X', 'X'), ('to-Z', 'Z')],
    'X': [('x1', 'x1'), ('x2', 'x2')],
    'Y': [('y1', 'y1'), ('y2', 'y2')],
    'Z': [('z1', 'z1'), ('z2', 'z2')],
}
LEAVES = {'x1': 7, 'x2': 3, 'y1': 1, 'y2': 5, 'z1': 2, 'z2': 4}
NODE_KEYS = {'root': 1, 'a': 2, 'b': 3, 'X': 4, 'Y': 5, 'Z': 6}
toy_tt: dict[str, Entry] = {}
visits: list[str] = []
expanded: list[str] = []


def toy_search(node: str, depth: int, maximizing: bool) -> int:
    visits.append(node)
    if node in LEAVES:
        return LEAVES[node]
    cached = toy_tt.get(node)
    if (
        cached is not None
        and cached.key == NODE_KEYS[node]
        and cached.depth >= depth
        and cached.bound == 'EXACT'
    ):
        return cached.score
    expanded.append(node)
    child_scores = [
        (move, toy_search(child, depth - 1, not maximizing))
        for move, child in TREE[node]
    ]
    choose = max if maximizing else min
    best_move, score = choose(child_scores, key=lambda pair: pair[1])
    toy_tt[node] = Entry(NODE_KEYS[node], depth, best_move, score, 'EXACT')
    return score


score = toy_search('root', 3, maximizing=True)
assert score == 5
assert visits.count('X') == 2  # node reached via two parents
assert expanded.count('X') == 1  # second encounter reused the exact entry
assert toy_tt['X'].best_move == 'x1'
assert toy_tt['X'].score == 7
print(f'Toy minimax score: {score}; X visits: {visits.count("X")}')
print('X entry:', toy_tt['X'])
# Stable explicit keys keep the example reproducible across Python processes.

# %% [markdown]
# ## Index collisions are not key collisions
#
# A fixed-size table maps many full keys to the same slot (index collision).
# Keeping and checking the full key prevents a false hit. But if two different
# positions have the same stored key signature (a true key collision), that
# check cannot distinguish them. This deliberately tiny table makes both ideas
# concrete; it does not estimate real 64-bit collision rates.

# %%
slots: list[Entry | None] = [None] * 2

def slot_index(key: int) -> int:
    return key % len(slots)


def store(entry: Entry) -> None:
    """Always replace the selected slot (a deliberately simple policy)."""
    slots[slot_index(entry.key)] = entry


def lookup(key: int) -> Entry | None:
    entry = slots[slot_index(key)]
    return entry if entry is not None and entry.key == key else None


store(Entry(10, 3, 'e4', 12, 'EXACT'))
store(Entry(12, 2, 'd4', 5, 'EXACT'))  # same index, different full key
assert slot_index(10) == slot_index(12)
assert lookup(10) is None and lookup(12) is not None
# Distinct conceptual positions that share a key are indistinguishable to lookup.
collision_key = 14
store(Entry(collision_key, 2, 'Nc3', 9, 'EXACT'))
assert lookup(collision_key).score == 9
print('Index collision rejected by full-key check; true key collision is undetectable.')

# %% [markdown]
# ## Replacement and search history
#
# Always-replace is cheap but a shallow result can evict a valuable deep one.
# Depth-preferred replacement protects deeper analysis, but can leave an old
# entry occupying a hot slot; generations/age, buckets, and move quality are
# common additional tradeoffs. Also, repetition is path-dependent: identical
# board, turn, rights, and en-passant state can have different repetition
# histories (and halfmove clocks). A cache keyed only by a position hash can
# therefore reuse a score whose draw context is not equivalent. Production
# engines address this carefully; this notebook's toy table does not.

# %%
def replace_if_deeper(old: Entry | None, new: Entry) -> Entry:
    return new if old is None or new.depth >= old.depth else old


deep = Entry(99, 8, 'Qh5', 40, 'EXACT')
shallow = Entry(99, 2, 'Nf3', 10, 'EXACT')
assert replace_if_deeper(deep, shallow) == deep
assert replace_if_deeper(shallow, deep) == deep
print('Depth-preferred policy keeps depth 8 over depth 2.')

# Same current FEN identity fields, distinct prior histories: the current
# position key cannot encode how often it appeared earlier in the game.
cycle_a = ChessGame()
for move in ('Nf3', 'Nf6', 'Ng1', 'Ng8'):
    cycle_a.make_move(move)
cycle_b = ChessGame()
assert cycle_a.to_fen().split()[:4] == cycle_b.to_fen().split()[:4]
assert cycle_a.to_fen().split()[4:] != cycle_b.to_fen().split()[4:]
print('Same repetition-relevant position fields, different game-history context.')

# %% [markdown]
# ## Takeaways
#
# - A TT stores a verified key, depth, best move, score, and bound; only a
#   sufficiently deep entry with a compatible bound can safely cut off.
# - Full-key verification catches slot/index collisions, not genuine key
#   collisions. Replacement balances freshness, depth, and table pressure.
# - Repetition and move-count draws depend on history/counters in addition to
#   board position; TT reuse must account for that context.
# - This educational table and toy minimax are not integrated into pychess's
#   shipped search.
#
# Reflect: when might a shallow cached best move help even if its score cannot
# cut off? What policy would you choose if memory were extremely constrained?
# Next: [Draw state and repetition](10_draw_state_and_repetition.ipynb).

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
# # 17. NNUE foundations for engine programmers
#
# **Goals:** understand how a neural evaluation can coexist with incremental chess search, why king-relative features are useful, and how a tiny accumulator update works. NNUE means *Efficiently Updatable Neural Network*; it is a learned evaluation architecture designed so a move usually changes only a small part of the input representation.
#
# References: [Chessprogramming Wiki: NNUE](https://www.chessprogramming.org/NNUE) and the verified [official Stockfish NNUE documentation](https://official-stockfish.github.io/docs/nnue-pytorch-wiki/docs/nnue.html).

# %% [markdown]
# ## Handcrafted evaluation and NNUE
#
# A classical evaluator computes explicit terms such as material, piece-square values, pawn structure, mobility, and phase interpolation. These terms are interpretable and compact, but their weights and interactions are designed by people. NNUE evaluation instead maps a sparse encoding of a position through learned layers. It can represent nonlinear interactions that are tedious to hand-code, but its behavior is less transparent and depends on a trained parameter set.
#
# A common feature family describes each piece by type, color, and square **relative to a king's square** (often with separate perspectives for each side). Most possible features are absent in any one position, so the input is sparse. The first-layer affine contributions for active features are summed into an accumulator. After a move, piece features are added or removed; when the perspective king moves, many king-relative feature indices change, so an accumulator refresh may be needed. Search can then reuse a cheaply updated representation across nearby positions.
#
# Real NNUE networks use multiple quantized layers and carefully chosen integer arithmetic/activation ranges to make inference fast. Training starts from position/evaluation examples (often generated or filtered using engine analysis), optimizes weights, and validates strength through testing. Quantization and feature layout are architectural details—not just rounding arbitrary floats at runtime.

# %% [markdown]
# ## A tiny sparse accumulator: a didactic toy, not NNUE
#
# This intentionally tiny linear example has three declared features and two outputs. The accumulator is the sum of feature weight rows plus a bias. Adding or removing one active feature updates it by one row addition/subtraction rather than recomputing every active contribution. This illustrates the bookkeeping idea only: it is **not an NNUE model, not a chess evaluator, has no king-relative feature map, no hidden activation or quantized network, and is not competitive**.

# %%
FEATURE_WEIGHTS = {
    'white_knight_central': (2, -1),
    'black_pawn_passed': (-3, 4),
    'white_king_active': (1, 2),
}
BIAS = (5, 0)


def add_vectors(left: tuple[int, ...], right: tuple[int, ...]) -> tuple[int, ...]:
    return tuple(a + b for a, b in zip(left, right, strict=True))


def subtract_vectors(left: tuple[int, ...], right: tuple[int, ...]) -> tuple[int, ...]:
    return tuple(a - b for a, b in zip(left, right, strict=True))


def recompute(active: set[str]) -> tuple[int, int]:
    """Compute the toy linear output directly from all active features."""
    result = BIAS
    for feature in sorted(active):
        result = add_vectors(result, FEATURE_WEIGHTS[feature])
    return result


active = {'white_knight_central', 'white_king_active'}
accumulator = recompute(active)
print('Initial active features:', sorted(active))
print('Accumulator:', accumulator)
assert accumulator == (8, 1)

# Add one active feature, then remove another. Feature names here are abstract labels, not a claim that this represents a legal chess move.
active.add('black_pawn_passed')
accumulator = add_vectors(accumulator, FEATURE_WEIGHTS['black_pawn_passed'])
assert accumulator == recompute(active) == (5, 5)

active.remove('white_knight_central')
accumulator = subtract_vectors(accumulator, FEATURE_WEIGHTS['white_knight_central'])
assert accumulator == recompute(active) == (3, 6)
print('After incremental add/remove:', accumulator)

# %% [markdown]
# The set is kept alongside the accumulator so this toy can recompute a reference result and assert its incremental updates. Production systems must preserve exact feature semantics through captures, promotions, en passant, castling, and king-bucket changes. The small arithmetic is simple; ensuring every state transition applies the right feature delta is the engineering challenge.

# %% [markdown]
# ## Where it fits in search
#
# Search repeatedly evaluates related positions. A handcrafted evaluator may rescan terms for each leaf, while an incrementally updatable network can reuse much of its feature accumulator. This does not remove the need for legal move generation, terminal/draw handling, search, or sound state restoration. Nor does “neural” mean an engine searches deeper automatically: evaluation speed, feature-update cost, network size, cache behavior, and playing strength are coupled tradeoffs.
#
# NNUE should also not be conflated with this repository's shipped evaluation lesson: the existing engine uses explicit classical terms, and this notebook introduces NNUE concepts without integrating a network or changing engine behavior.

# %% [markdown]
# ## Takeaways
#
# - Handcrafted evaluation makes chess knowledge explicit; NNUE learns feature interactions from data.
# - Sparse piece features let an accumulator update by adding/removing affected rows, with king-relative feature remapping as an important exception.
# - Quantized inference and training data are central to practical NNUE systems; a tiny integer sum is only an intuition pump.
# - Incremental state correctness is crucial because search explores many transitions.
#
# **Reflect:** Which move types change more than one piece feature? Why does a king move potentially invalidate a whole perspective's feature mapping? How would you test an incremental accumulator against full recomputation over a search tree?
#
# Next: [Opening books](18_opening_books.ipynb) shows a data-driven alternative to searching early moves.

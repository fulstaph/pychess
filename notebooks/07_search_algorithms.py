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
# # 7. Minimax, negamax, and alpha-beta
#
# Goals: compute a minimax choice on a transparent tree, derive negamax's sign convention, interpret alpha-beta windows/cutoffs and principal variations, and relate branching factor to search cost. This tiny tree is educational and separate from the shipped engine implementation.
#
# Primary references: [Minimax](https://www.chessprogramming.org/Minimax), [Negamax](https://www.chessprogramming.org/Negamax), [Alpha-Beta](https://www.chessprogramming.org/Alpha-Beta), [Branching Factor](https://www.chessprogramming.org/Branching_Factor).

# %%
# A depth-two tree: root chooses A or B; opponent then chooses a leaf.
# Leaf values are always from the root player's perspective.
tree = {"root": ["A", "B"], "A": ["A1", "A2"], "B": ["B1", "B2"]}
values = {"A1": 3, "A2": 5, "B1": 2, "B2": 9}

# %% [markdown]
# ## Minimax and a consistent negamax sign
#
# A score is meaningful only with a perspective. Here every leaf's value is from the root player's perspective; because leaves are two plies down, that is also the side-to-move perspective at each leaf. Negamax expresses the same zero-sum choice by always maximizing the current player's score and negating the child score whenever the side changes. Every recursive return is from the perspective of the player to move at that node.

# %%
def minimax(node: str, maximizing: bool) -> int:
    if node in values:  # depth-zero leaf
        return values[node]
    child_scores = [minimax(child, not maximizing) for child in tree[node]]
    return max(child_scores) if maximizing else min(child_scores)


def negamax_root_tree(node: str) -> int:
    """Return the score from the side-to-move perspective at this node."""
    if node in values:
        return values[node]
    return max(-negamax_root_tree(child) for child in tree[node])

assert minimax("root", True) == 3
assert negamax_root_tree("root") == 3
print("A scores", [minimax(move, False) for move in tree["root"]], "=> choose A")

# %% [markdown]
# ## Alpha-beta windows and pruning
#
# Alpha is the best lower bound already available to the maximizing side; beta is the best upper bound imposed by an ancestor. A node whose value cannot improve that window need not be searched further. Alpha-beta preserves minimax's result; move ordering determines how early useful bounds arise, not the value itself. The root principal variation (PV) is a best line consistent with the searched scores; ties can produce multiple equally good lines. At a cutoff, an unfinished line provides only a bound, not an exact score or a fully searched best continuation. The returned root PV follows the selected explored moves.

# %%
# The recursive result carries both the node's minimax score and its selected line.
visited = []
def alphabeta(node: str, maximizing: bool, alpha: float, beta: float) -> tuple[int, tuple[str, ...]]:
    if node in values:
        visited.append(node)
        return values[node], ()
    best = -float("inf") if maximizing else float("inf")
    best_line = ()
    for child in tree[node]:
        child_score, child_line = alphabeta(
            child, not maximizing, alpha, beta
        )
        if (maximizing and child_score > best) or (
            not maximizing and child_score < best
        ):
            best = child_score
            best_line = (child, *child_line)
        if maximizing:
            alpha = max(alpha, best)
        else:
            beta = min(beta, best)
        if alpha >= beta:
            break
    return int(best), best_line

score, principal_variation = alphabeta(
    "root", True, -float("inf"), float("inf")
)
assert score == 3
assert principal_variation == ("A", "A1")
assert visited == ["A1", "A2", "B1"]
print(
    "Score:", score, "PV:", principal_variation,
    "visited leaves:", visited, "pruned leaves:", 1,
)

# %% [markdown]
# ## Cost and chess-specific terminal scores
#
# With branching factor `b` and depth `d`, an unpruned uniform tree has about `b**d` leaves; chess's legal move counts vary by position and ply. Ideal ordering can make alpha-beta approach roughly `b**(d/2)` leaf work, while poor ordering may save little. These are intuition-building bounds, not runtime guarantees.
#
# Real chess search also distinguishes checkmate/stalemate from depth-zero evaluation. A mate score must use the same perspective convention as ordinary evaluations; many engines adjust mate scores by ply to prefer faster wins and delayed losses. A horizon leaf is evaluated rather than expanded, except when a search explicitly extends tactical continuations.

# %%
for branching, depth in ((3, 4), (30, 4)):
    print(f"uniform tree b={branching}, d={depth}: {branching ** depth} leaves without pruning")

# %% [markdown]
# ## Relating the toy tree to this project
#
# The repository exposes `choose_move(game, depth=...)` and a reusable
# `SearchEngine(EngineConfig(...))`. It defaults to depth 3 with quiescence
# search, a bounded transposition table, and the legacy `basic` evaluation.
# The opt-in `positional` profile adds phase-aware terms; paired matches remain
# inconclusive. This example uses white-relative alpha-beta scores; the toy tree
# still isolates search logic from real evaluation and move generation.
#
# Takeaways: minimax models alternating preferences; negamax is the same zero-sum recurrence with a carefully maintained perspective; alpha-beta prunes only when bounds prove a branch irrelevant; ordering affects efficiency. Reflect: if the second child of A scored 1 instead of 5, what value would root choose?
#
# Next: [Search heuristics](08_search_heuristics.ipynb).

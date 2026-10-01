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
# # 19. Parallel search and concurrency tradeoffs
#
# Game-tree search is not embarrassingly parallel: alpha-beta learns from
# earlier siblings, and that information can make later work unnecessary.
# This lesson compares sequential root search with independent speculative
# root workers. It then surveys shared transposition tables, Lazy SMP, and
# why CPU-bound parallelism in Python has different costs from native engines.
#
# The examples are a deterministic toy tree. The project engine remains
# synchronous and does not implement parallel search.
#
# Sources: CPW [Parallel Search](https://www.chessprogramming.org/Parallel_Search),
# [Lazy SMP](https://www.chessprogramming.org/Lazy_SMP), and the Python
# [`concurrent.futures`](https://docs.python.org/3/library/concurrent.futures.html)
# documentation.

# %%
from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor

# %% [markdown]
# ## Why alpha-beta's first child matters
#
# At a maximizing root, each child is a minimizing node. Once the first child
# establishes a root alpha score, a later minimizing child can stop as soon as
# its value is no better than alpha. That cutoff is valid only because the
# root has already found a result at least as good.

# %%
NEG_INF = -10**9
root_tree: dict[str, tuple[int, ...]] = {
    'first': (3, 2, 0),
    'second': (2, 1, -8),
    'third': (0, -100, -200),
}


def search_min_node(scores: Sequence[int], alpha: int) -> tuple[int, int]:
    """Search one minimizing child; return its bound and visited leaves."""
    value = -NEG_INF
    visited = 0
    for score in scores:
        visited += 1
        value = min(value, score)
        if value <= alpha:
            break
    return value, visited


def sequential_root(
    tree: dict[str, tuple[int, ...]],
) -> tuple[int, dict[str, int]]:
    alpha = NEG_INF
    best = NEG_INF
    visits: dict[str, int] = {}
    for name, replies in tree.items():
        score, visits[name] = search_min_node(replies, alpha)
        best = max(best, score)
        alpha = max(alpha, best)
    return best, visits


sequential_score, sequential_visits = sequential_root(root_tree)
assert sequential_score == 0
assert sequential_visits == {'first': 3, 'second': 3, 'third': 1}
print('Sequential alpha-beta score:', sequential_score)
print('Leaves visited by root child:', sequential_visits)
print('Total leaves:', sum(sequential_visits.values()))

# %% [markdown]
# ## Speculative root splitting
#
# Each root child can be searched independently if workers receive the same
# initial alpha. This is safe for the final minimax score, but workers cannot
# exploit cutoffs discovered by other workers until results are combined.
# The example intentionally models that wasted speculative work; it does not
# claim that a thread pool makes this tiny Python tree faster.

# %%
def independent_child(
    item: tuple[str, tuple[int, ...]],
) -> tuple[str, int, int]:
    name, replies = item
    score, visited = search_min_node(replies, NEG_INF)
    return name, score, visited


with ThreadPoolExecutor(max_workers=3) as executor:
    independent_results = tuple(executor.map(independent_child, root_tree.items()))

parallel_scores = {name: score for name, score, _ in independent_results}
parallel_visits = {name: visited for name, _, visited in independent_results}
parallel_score = max(parallel_scores.values())
assert parallel_score == sequential_score
assert sum(parallel_visits.values()) == 9
assert sum(sequential_visits.values()) == 7
print('Independent worker scores:', parallel_scores)
print('Speculative leaves visited:', sum(parallel_visits.values()))
print('Sequential leaves visited: ', sum(sequential_visits.values()))

# %% [markdown]
# ## Ways engines share search work
#
# - **Root splitting:** search some root moves in workers. Search order and
#   current alpha affect how much speculative work is wasted.
# - **Young Brothers Wait Concept:** search the eldest/best-ordered child first;
#   launch siblings after it supplies a useful bound.
# - **Lazy SMP:** workers search the same root with different depths or move
#   orders and share a transposition table. They coordinate less, while
#   useful hash-table entries can influence one another's searches.
#
# A shared table is a communication channel, not just a cache. Entries need
# enough information to reject a mismatched key; concurrent readers and
# writers need a documented consistency strategy. A stale or partial bound
# can change search behavior, even if the board objects themselves are
# immutable.

# %% [markdown]
# ## Python-specific constraints
#
# In conventional CPython builds, the Global Interpreter Lock means Python
# threads do not generally execute CPU-bound Python bytecode in parallel.
# A process pool can run work on multiple cores, but sending board states,
# search stacks, and results between processes costs time and memory. A native
# engine can use compact shared tables and low-level atomics that are not a
# direct fit for this Python implementation.
#
# Do not infer strength from nodes per second alone. Parallel searches can
# reach a deeper iteration sooner, produce more transpositions, or search a
# different tree. Measure fixed-depth node counts, time-to-depth, and match
# results separately; keep correctness checks independent of speed claims.

# %% [markdown]
# ## Takeaways
#
# - Alpha-beta pruning is order-sensitive; independent workers may repeat work.
# - Shared transposition data improves cooperation but adds consistency and
#   collision concerns.
# - A thread pool proves that root tasks can be expressed independently; it
#   does not prove CPU speedup or competitive parallel search.
# - The current pychess engine is synchronous. Treat native-engine algorithms
#   as design references, not as features this repository already has.
#
# **Try:** reorder the root children and recalculate sequential visits. The
# score should remain the same while cutoff work can change.

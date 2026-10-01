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
# # 12. Engine testing and benchmarking
#
# **Learning goals:** use perft as a move-generation oracle; investigate a divide mismatch; test state transitions and query purity; distinguish correctness from playing strength; and measure performance without flaky timing thresholds.
#
# Earlier, lesson 02 introduced perft and special-move rules. Here the focus is a repeatable validation workflow, not another rules tour. [CPW: Perft](https://www.chessprogramming.org/Perft) · [CPW: Perft Results](https://www.chessprogramming.org/Perft_Results)
# %%
from time import perf_counter

from chess import ChessGame
from chess.engine import choose_move, move_notation


# %% [markdown]
# ## Perft counts legal move paths
#
# Perft(depth) counts the leaf positions reached by making every legal move for exactly `depth` plies. It does **not** evaluate chess or decide who is winning. With an established reference count, it is a powerful move-generator regression oracle: a mismatch proves some path differs, but the total alone does not identify which move is wrong.
#
# `after` makes a sibling state, so this small recursive implementation does not mutate its input. Keep depths shallow: the tree grows exponentially. The initial-position counts below are standard references.
# %%
def perft(game: ChessGame, depth: int) -> int:
    if depth == 0:
        return 1
    return sum(perft(game.after(move), depth - 1) for move in game.legal_moves())


start = ChessGame()
depth_1 = perft(start, 1)
depth_2 = perft(start, 2)
depth_3 = perft(start, 3)
assert (depth_1, depth_2, depth_3) == (20, 400, 8902)
print(
    f'Start position: depth 1 = {depth_1:,}; '
    f'depth 2 = {depth_2:,}; depth 3 = {depth_3:,}'
)

# %% [markdown]
# ## Divide localizes a discrepancy
#
# A divide reports the subtree count after each root move. Comparing the sorted mapping with a trusted table or another implementation narrows a failing total to one opening branch. It is often the fastest first diagnostic, though a bad branch may still require a deeper divide or move-by-move inspection.
#
# Kiwipete is a commonly used castling/tactical stress position; its depth-one reference is 48. Depth one lists each legal root move, which can be inspected directly.
# %%

def perft_divide(game: ChessGame, depth: int) -> dict[str, int]:
    if depth < 1:
        raise ValueError('divide depth must be at least one')
    return {
        move_notation(move): perft(game.after(move), depth - 1)
        for move in game.legal_moves()
    }


kiwipete = ChessGame.from_fen(
    'r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1'
)
kiwi_divide = perft_divide(kiwipete, 1)
assert len(kiwi_divide) == 48
assert sum(kiwi_divide.values()) == 48
print(f'Kiwipete root moves: {len(kiwi_divide)}; divide total: {sum(kiwi_divide.values())}')
print('First five branches:', sorted(kiwi_divide.items())[:5])

# %% [markdown]
# ## Edge-case fixtures and transition invariants
#
# Perft fixtures should deliberately include rule-state boundaries, not just the initial position: castling rights, an en-passant target, promotions, checks/pins, and sparse endgames. These small assertions check that the corresponding legal choices appear in known positions. They complement (rather than replace) deeper reference perft suites.
#
# A transition test checks more than the destination board: turn and counters advance, state serialization is coherent, `after` preserves its parent, and a rejected move is atomic. `choose_move` is a query and should likewise leave game state untouched.
# %%
castle_position = ChessGame.from_fen('r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 0 1')
assert {move.special for move in castle_position.legal_moves()} >= {'castle_k', 'castle_q'}

en_passant_position = ChessGame.from_fen('4k3/8/8/3pP3/8/8/8/4K3 w - d6 0 1')
assert en_passant_position.after('exd6').board.get((2, 3)) is not None
assert en_passant_position.after('exd6').board.get((3, 3)) is None

promotion_position = ChessGame.from_fen('4k3/P7/8/8/8/8/8/4K3 w - - 0 1')
promotion_moves = [move for move in promotion_position.legal_moves() if move.from_square == (1, 0)]
assert {move.promotion_to.value for move in promotion_moves} == {'q', 'r', 'b', 'n'}
print('Castles available: 2; en passant and four promotions checked')

before = start.to_fen()
child = start.after('e2e4')
assert start.to_fen() == before
assert child.turn == 'b' and child.move_num == 1 and child.halfmove_clock == 0
assert child.to_fen().split()[3] == 'e3'
try:
    start.make_move('e2e5')
except ValueError:
    pass
else:
    raise AssertionError('illegal move should be rejected')
assert start.to_fen() == before

query_before = child.to_fen()
selected = choose_move(child, depth=1)
assert child.to_fen() == query_before
assert selected in child.legal_moves()
print(f'Atomic rejection and query purity checked; depth-1 choice: {selected}')

# %% [markdown]
# ## Differential testing: compare, then minimize
#
# Differential testing runs the same positions through independent implementations and compares legal move sets, resulting positions, or shallow perft counts. An independent implementation is useful because two wrappers around the same move generator can share the same defect. If an external oracle is used in a separate development environment, pin its version/options and normalize rule conventions before comparing; this notebook does not install or invoke one.
#
# A productive loop is: generate/select a position, compare move sets, stop at the first divergence, then reduce the position or move sequence while preserving the failure. Preserve special state (side to move, rights, en-passant square, counters), otherwise the minimized example may no longer reproduce the bug. Random testing supplements curated fixtures; deterministic seeds and saved failures make regressions reproducible.
#
# **Correctness is not strength.** Perft and transition invariants can find rule/search bugs, but a correct engine can still choose weak moves. Playing strength needs match-based evidence and a separate methodology (lesson 13).
#
# ## Measure performance without overclaiming
#
# Record the exact position, depth, options, runtime/build, warm-up policy, and repeat count. Report distributions (for example median), and separate node-count changes from elapsed-time changes; time is affected by machine load and interpreter/runtime conditions. Profilers answer *where* time is spent, while benchmarks answer *how long* a fixed workload takes. Avoid timing assertions in tests: they are hardware- and load-sensitive.
#
# `uv run --locked python -m chess.bench --depth 3 --repeats 3` runs cold-table
# searches over fixed start, tactical, middlegame, and endgame FENs. It reports
# depth, move, alpha-beta and quiescence nodes, TT hits, elapsed time, and NPS.
# Compare `--profile basic` with `--profile positional`; keep time observational.
# %%
benchmark_position = ChessGame()
repeats = 5
samples = []
for _ in range(repeats):
    t0 = perf_counter()
    count = perft(benchmark_position, 2)
    samples.append(perf_counter() - t0)
assert count == 400
print(f'Deterministic workload: start-position perft(2) = {count} leaves')
print(f'Illustrative elapsed times ({repeats} runs, seconds): {[round(s, 6) for s in samples]}')

# %% [markdown]
# ## Takeaways
#
# - Perft is a move-generation oracle only when compared with trusted counts; divide helps locate the branch that differs.
# - Exercise unusual state and verify transition boundaries, not only legal-move totals.
# - A query/purity test and an illegal-move atomicity test protect useful API contracts.
# - Deterministic counts are suitable assertions; wall-clock performance is an observation, not a pass/fail threshold.
# - Correctness validation and strength evaluation answer different questions.
#
# Sources: [CPW Perft](https://www.chessprogramming.org/Perft), [CPW Perft Results](https://www.chessprogramming.org/Perft_Results), [CPW Testing](https://www.chessprogramming.org/Testing)
#
# Next: [Evaluation tuning](13_evaluation_tuning.ipynb) studies playing-strength evidence.

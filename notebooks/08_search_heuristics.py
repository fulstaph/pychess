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
# # 8. Search heuristics beyond move ordering basics
#
# Goals: understand iterative deepening and PV reuse; see how static exchange evaluation (SEE) extends MVV-LVA; distinguish horizon extension from pruning; and survey selective-search techniques with their failure modes. The demonstrations are small and deterministic, not a production chess search. Lesson 03 already covers the shipped engine's MVV-LVA ordering and quiescence basics; these topics explain what can build on those ideas.
#
# References: [Iterative Deepening](https://www.chessprogramming.org/Iterative_Deepening), [Move Ordering](https://www.chessprogramming.org/Move_Ordering), [Static Exchange Evaluation](https://www.chessprogramming.org/Static_Exchange_Evaluation), [Quiescence Search](https://www.chessprogramming.org/Quiescence_Search), [Aspiration Windows](https://www.chessprogramming.org/Aspiration_Windows), [Late Move Reductions](https://www.chessprogramming.org/Late_Move_Reductions), [Null Move Pruning](https://www.chessprogramming.org/Null_Move_Pruning).

# %% [markdown]
# ## Iterative deepening: search shallow, then reuse information
#
# Search depths 1, 2, 3, … until the time budget expires. Each completed iteration supplies a legal fallback move and a principal variation (PV); the prior best move can be searched first at the next depth. This may look like repeating work, but a well-ordered deeper search can prune much more, and the shallow result provides useful ordering. This deterministic toy ranks a prior PV move first without claiming to evaluate chess.

# %%
root_moves = ("quiet", "tactical", "other")
# Toy deeper-search values; only ordering is demonstrated, not chess evaluation.
score = {"quiet": 1, "tactical": 8, "other": 3}
previous_pv_move = "tactical"
ordered = tuple(sorted(root_moves, key=lambda move: (move != previous_pv_move, move)))
assert ordered[0] == previous_pv_move
assert max(root_moves, key=score.__getitem__) == "tactical"
print("Prior PV first:", ordered)

# %% [markdown]
# ## SEE: exchanges, not just victim-attacker labels
#
# MVV-LVA cheaply ranks a capture using the victim and attacker values, but it does not model the recapture sequence on the destination square. SEE repeatedly chooses the least valuable available attacker and works backward through the exchange gains, approximating whether the capture wins or loses material. It can still miss pins, king legality, x-rays, and positional compensation; it is a tactical ordering/evaluation aid, not a full search. No SEE is implemented in the shipped engine here.
#
# This tiny swap-list example isolates the backward minimax accounting: White captures a 3-point pawn with a 5-point rook, Black recaptures the rook with a 3-point bishop. The net exchange for White is pawn value minus rook value, so the apparently attractive first capture loses 2.

# %%
# This deliberately simplified swap-list is not legal move generation.
# White wins a pawn (3), then loses its rook (5): net -2.
# MVV-LVA sees only the first capture; an exchange sequence sees the recapture.
net_exchange = 3 - 5
assert net_exchange == -2
print("Simplified capture then recapture net:", net_exchange)


# %% [markdown]
# ## Horizon: quiescence and alternatives
#
# At a fixed-depth leaf, a position with a hanging queen can look falsely stable just before a capture. Quiescence continues selected tactical moves (commonly captures, sometimes checks) until a quieter position, reducing this horizon effect. It is not free: noisy positions can branch widely, and careless check handling or stand-pat use in check is incorrect. This is why a larger quiescence tree needs its own limits and correctness rules. Lesson 03 already demonstrates the project's optional quiescence behavior; this lesson emphasizes its horizon motivation and caveats.

# %% [markdown]
# ## Selectivity survey: bounds, reductions, and risk
#
# - **Aspiration windows** search around the previous iteration's score instead of using a wide window. If the true score falls outside, fail-low/high requires a wider re-search. A too-narrow window can cause repeated work; it must not change the final exact result.
# - **Principal Variation Search (PVS)** gives the first (presumed best) move a full window and later moves a narrow test window, re-searching any move that improves alpha. Incorrect re-search conditions can lose the best line.
# - **Killer/history ordering** promotes quiet moves that caused cutoffs at the same ply (killer) or historically caused cutoffs (history). These are ordering hints, not proofs; stale correlations can worsen ordering.
# - **Null-move pruning** searches after a pass-like null move and may cut off if even that handicapped position exceeds beta. Zugzwang (where every legal move is worse than passing), check, and endgame material make this especially risky; implementations restrict when it is allowed and often verify.
# - **Late-move reductions (LMR)** search later, presumed less promising moves at reduced depth, restoring full depth if they challenge alpha. Strong late tactical moves can be missed if reduction/re-search policy is unsound.
#
# These techniques depend on accurate bounds, consistent score perspective, and carefully tested edge cases. They are not interchangeable with exact alpha-beta, and none is added to this lesson as a pile of speculative code. The project engine's documented API and behavior remain distinct from these general engine techniques.

# %%
# An aspiration-window toy: a narrow window misses the true score, so widen and repeat.
true_score = 42
alpha, beta = 40, 41
assert true_score >= beta  # fail-high: this result is only a lower bound
alpha, beta = 40, 50
assert alpha < true_score < beta  # re-search window now contains exact value
print("Aspiration fail-high at beta=41; widened window contains", true_score)

# %% [markdown]
# ## Takeaways
#
# - Iterative deepening makes prior PV information useful for deeper move ordering.
# - SEE models capture sequences beyond MVV-LVA's one-ply victim/attacker comparison.
# - Quiescence addresses tactical instability at the horizon but needs careful rules and limits.
# - Aspiration/PVS can save work through re-search; killer/history order; null move and LMR prune/reduce with position-dependent risks.
# - These are engine techniques, not claims about features integrated in this educational repository.
#
# Reflect: which techniques can change work ordering only, and which can make search selective enough to miss a tactical line?
#
# Next: [Transposition tables](09_transposition_tables.ipynb) build on these search techniques.

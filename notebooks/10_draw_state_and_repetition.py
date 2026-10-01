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
# # 10. Chess state, FEN, and draw semantics
#
# **Goals:** read all six FEN fields, separate position identity from game
# history, inspect castling/en-passant rights and move counters, and use
# `ChessGame`'s draw APIs. These details extend the basic move/rule tour in
# lesson 02: a board diagram alone does not specify a complete game state.
#
# Sources: [CPW: Forsyth-Edwards Notation](https://www.chessprogramming.org/Forsyth-Edwards_Notation),
# [CPW: Repetitions](https://www.chessprogramming.org/Repetitions),
# [CPW: Fifty-move Rule](https://www.chessprogramming.org/Fifty-move_Rule),
# [CPW: Insufficient Material](https://www.chessprogramming.org/Insufficient_Material),
# [FIDE Laws of Chess](https://handbook.fide.com/chapter/E012023).

# %%
from chess import ChessGame

# %% [markdown]
# ## Six FEN fields
#
# FEN contains piece placement, active color, castling availability, en-passant
# target, halfmove clock, and fullmove number. `ChessGame.from_fen` parses them;
# `to_fen()` emits the state. For hash identity, board, side to move, castling,
# and en-passant matter; the halfmove clock affects the 50-move rule, and
# repetition requires prior positions. Fullmove number is notation bookkeeping.
#
# Castling rights can change legal options without changing piece placement.
# En-passant target is recorded after a two-square pawn move; whether it changes
# effective repetition identity can depend on whether capture is actually legal.

# %%
fen_rights = 'r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 0 1'
fen_no_rights = 'r3k2r/8/8/8/8/8/8/R3K2R w - - 0 1'
rights_game = ChessGame.from_fen(fen_rights)
no_rights_game = ChessGame.from_fen(fen_no_rights)
assert rights_game.board == no_rights_game.board
assert rights_game.to_fen().split()[2] == 'KQkq'
assert no_rights_game.to_fen().split()[2] == '-'
assert rights_game.castles('w', 'kingside')
assert not no_rights_game.castles('w', 'kingside')
print('Same placement, different castling rights:')
print('  with rights:', rights_game.to_fen())
print('  without:    ', no_rights_game.to_fen())

# %%
fen_ep = '4k3/8/8/3pP3/8/8/8/4K3 w - d6 0 2'
ep_game = ChessGame.from_fen(fen_ep)
no_ep_game = ChessGame.from_fen(fen_ep.replace(' d6 ', ' - '))
assert ep_game.to_fen().split()[3] == 'd6'
assert no_ep_game.to_fen().split()[3] == '-'
print('En-passant field is state information:', ep_game.to_fen().split()[3])

# %% [markdown]
# ## Halfmove and fullmove counters
#
# The halfmove clock counts plies since the last pawn move or capture; it resets
# after either. `is_fifty_moves()` in this project returns true at 100 halfmoves.
# The fullmove number increments after Black's move and does not affect legality.
# Here the project policy marks a draw as soon as the threshold is reached.

# %%
clock_game = ChessGame.from_fen('4k3/8/8/8/8/8/8/4K3 w - - 100 37')
assert clock_game.to_fen().split()[4:] == ['100', '37']
assert clock_game.is_fifty_moves()
assert clock_game.draw_reason() == 'fifty-move rule'
print('FEN counters:', clock_game.to_fen().split()[4:])
print('Draw reason:', clock_game.draw_reason())

# %% [markdown]
# ## Repetition depends on history
#
# A repeated position means the same piece placement, side to move, and
# castling availability. An en-passant target distinguishes positions only if
# the side to move has a legal en-passant capture. Unavailable or pinned
# captures do not change repetition identity. `is_threefold_repetition()`
# counts the current position in recorded history, so one FEN snapshot cannot
# establish threefold repetition. Play the same knight cycle twice: the initial
# state plus two returns is three occurrences.

# %%
repetition_game = ChessGame()
cycle = ('Nf3', 'Nf6', 'Ng1', 'Ng8')
for _ in range(2):
    for move in cycle:
        repetition_game.make_move(move)
assert repetition_game.is_threefold_repetition()
assert repetition_game.draw_reason() == 'threefold repetition'
print('After 8 plies:', repetition_game.to_fen())
print('Threefold:', repetition_game.is_threefold_repetition())
print('Draw reason:', repetition_game.draw_reason())

# A target square with a pinned capturer is not a legal en-passant right:
pinned_ep_game = ChessGame.from_fen(
    'k3r1n1/8/8/3pP3/8/8/8/4K1N1 w - d6 0 2'
)
assert pinned_ep_game.to_fen().split()[3] == 'd6'
assert not any(
    move.special == 'en_passant' for move in pinned_ep_game.legal_moves()
)
for _ in range(2):
    for move in cycle:
        pinned_ep_game.make_move(move)
assert pinned_ep_game.is_threefold_repetition()
print(
    'Threefold with an unusable en-passant target:',
    pinned_ep_game.is_threefold_repetition(),
)


# Same current position fields, but this newly loaded FEN has no prior cycle.
snapshot_only = ChessGame.from_fen(repetition_game.to_fen())
assert snapshot_only.to_fen().split()[:4] == repetition_game.to_fen().split()[:4]
assert not snapshot_only.is_threefold_repetition()
print('FEN snapshot alone establishes repetition:', snapshot_only.is_threefold_repetition())

# %% [markdown]
# ## Insufficient material and draw policy
#
# Bare kings cannot produce checkmate, so the project identifies this as
# insufficient material. The public API exposes `is_draw()`, `draw_reason()`,
# `is_threefold_repetition()`, `is_fifty_moves()`, and
# `is_insufficient_material()`. Checkmate is explicitly not reported as a draw.
#
# **Rules caveat:** FIDE distinguishes claimable draws (including threefold
# repetition and the 50-move rule) from automatic endings at fivefold repetition
# and 75 moves, with details about intended moves. This implementation applies
# its draw policy directly: it reports draw at threefold/100 halfmoves rather
# than modeling the claim-vs-automatic procedure. It is a useful application
# policy, not a complete arbiter of FIDE procedure.

# %%
bare_kings = ChessGame.from_fen('4k3/8/8/8/8/8/8/4K3 w - - 0 1')
assert bare_kings.is_insufficient_material()
assert bare_kings.is_draw()
assert bare_kings.draw_reason() == 'insufficient material'
print('Bare kings:', bare_kings.draw_reason())

# %% [markdown]
# ## Hash identity versus history
#
# Lesson 04's educational Zobrist key includes board pieces, turn, castling,
# and en-passant fields. That is a position fingerprint, not a complete game
# record. A repetition-aware search also needs the reversible-move history (or
# an equivalent repetition context); 50-move adjudication needs the halfmove
# clock. TT entries keyed only by position can therefore be unsafe across
# histories even when the board fingerprint matches.
#
# **Takeaways:** read all six FEN fields; distinguish position state from path
# history; use the project API to inspect its explicit draw policy. Reflect:
# should a game-loading API infer repetition history from a FEN? Why or why not?
# Next: [Advanced evaluation](11_advanced_evaluation.ipynb).

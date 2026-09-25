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
# # 3. Engine decisions and search
#
# This lesson examines the repository's small terminal engine and a separate toy search. You will see how a greedy engine values captures, how to count short move sequences without changing the original game, and why looking ahead can prefer a quiet move over an immediate capture.

# %%
from copy import deepcopy

from chess import ChessGame, Piece, PieceType
from chess.engine import PIECE_VALUES, choose_move, move_notation

# %% [markdown]
# ## The shipped engine: material evaluation and minimax search
#
# The package's engine scores positions with standard material values (queen `900`, rook `500`, bishop `330`, knight `320`, pawn `100`) and central bonuses, evaluating replies with a two-ply alpha-beta minimax search. From the standard opening, White develops with `d2d4`.

# %%
assert PIECE_VALUES == {
    PieceType.PAWN: 100,
    PieceType.KNIGHT: 320,
    PieceType.BISHOP: 330,
    PieceType.ROOK: 500,
    PieceType.QUEEN: 900,
    PieceType.KING: 0,
}
initial_game = ChessGame()
assert move_notation(choose_move(initial_game)) == 'd2d4'
print('Material values:', PIECE_VALUES)
print('Opening move:', move_notation(choose_move(initial_game)))

# %% [markdown]
# After `1. e4 d5`, White can capture the pawn on d5. The engine chooses that capture because it is worth more by its immediate scoring rule than any quiet move. The returned `Move` is still submitted through `ChessGame.make_move()`, so the game applies normal legal-state updates.

# %%
capture_game = ChessGame()
capture_game.make_move('e4')
capture_game.make_move('d5')
selected = choose_move(capture_game)
assert move_notation(selected) == 'e4d5'
played = capture_game.make_move(selected)
assert played.captured_piece == Piece(PieceType.PAWN, 'b')
print('Engine selected:', move_notation(selected))
print('Captured piece:', played.captured_piece)


# %% [markdown]
# ## Count positions at a fixed depth
#
# A simple way to explore a game tree is to visit every legal move, copy the current `ChessGame`, apply one move to the copy, and recurse. At depth zero, the current position counts as one leaf. Copying the game preserves its current board, turn, castling rights, en-passant target, move number, and status. `Move.execute()` alone receives only a board and cannot update that rule-relevant game state. The package does not keep a complete list of historical moves.

# %%
def count_positions(game: ChessGame, depth: int) -> int:
    if depth == 0:
        return 1
    total = 0
    for move in game.legal_moves():
        branch = deepcopy(game)
        branch.make_move(move)
        total += count_positions(branch, depth - 1)
    return total

counting_game = ChessGame()
assert count_positions(counting_game, 1) == 20
assert count_positions(counting_game, 2) == 400
assert counting_game.turn == 'w'
assert len(counting_game.legal_moves()) == 20
print('Positions after one ply:', count_positions(counting_game, 1))
print('Positions after two plies:', count_positions(counting_game, 2))

# %% [markdown]
# ## A separate toy two-ply minimax
#
# Now consider a tiny abstract example, separate from the shipped engine. Each tuple gives White-relative outcomes after Black's possible replies. Black chooses the lower score, so take the minimum for each White move; White then chooses the line with the highest worst-case result.

# %%
reply_scores = {
    'greedy_capture': (-5, -2),
    'quiet_move': (1, 0),
}
worst_case_by_move = {
    move: min(replies)
    for move, replies in reply_scores.items()
}
selected_by_minimax = max(worst_case_by_move, key=worst_case_by_move.get)
assert worst_case_by_move == {'greedy_capture': -5, 'quiet_move': 0}
assert selected_by_minimax == 'quiet_move'
print('Worst-case White scores:', worst_case_by_move)
print('Toy minimax chooses:', selected_by_minimax)

# %% [markdown]
# ## Reflect and try the engine
#
# - What tactical information does a capture-only score ignore?
# - How might a one-ply engine be tricked into taking a poisoned piece?
# - What additional evaluation or search rule would you add first?
#
# For an interactive game against the shipped engine, run `python -m chess.engine` from the repository root. The notebook task launches JupyterLab with the optional dependencies installed.

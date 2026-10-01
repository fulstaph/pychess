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
# # 2. Legal moves and chess rules
#
# This lesson explores how the package turns possible piece movements into legal game moves. We will inspect generated `Move` values, update game state, see how attacks constrain moves, and exercise castling, en passant, promotion, checkmate, and stalemate. Each scenario starts with a fresh game.

# %%
from copy import deepcopy

from chess import Board, ChessGame, GameStatus, Piece, PieceType, notation_to_coords
from chess.attacks import is_in_check
from chess.engine import move_notation

sq = notation_to_coords

def position(*placements):
    board = Board()
    for text, kind, color in placements:
        board = board.with_piece(sq(text), Piece(kind, color))
    return board


# %% [markdown]
# ## Generated moves and game state
#
# A `Move` is a value describing a piece, its source and destination squares, any captured piece, and special-move metadata such as promotion. `ChessGame.legal_moves()` returns the legal moves for the current player. At the start of a game, White has 20 legal choices.

# %%
opening_game = ChessGame()
opening_moves = opening_game.legal_moves()
assert len(opening_moves) == 20
e4_candidate = next(
    move for move in opening_moves
    if move.from_square == sq('e2') and move.to_square == sq('e4')
)
assert e4_candidate.captured_piece is None
assert e4_candidate.special == 'none'
print('Opening legal moves:', len(opening_moves))
print('One generated move:', e4_candidate)

# %% [markdown]
# **Predict, then run:** after White plays `e4`, whose turn is next, what is the game status, and how many replies does Black have? `make_move()` updates more than the board: it also commits turn, rights, en-passant state, and status.

# %%
opening_game.make_move('e4')
black_replies = opening_game.legal_moves()
assert opening_game.turn == 'b'
assert opening_game.status == GameStatus.ACTIVE
assert len(black_replies) == 20
print('Next to move:', opening_game.turn)
print('Status:', opening_game.status)
print('Black replies:', len(black_replies))

# %% [markdown]
# ## Attacks, blockers, and check
#
# An attack is geometric: a rook attacks along its rank or file until the first piece blocks the ray. The king at e1 is in check from a rook on e8 when the e-file is clear. A piece at e4 stops that attack. `ChessGame` uses attack detection when deciding which generated moves are legal.

# %%
clear_file = position(
    ('e1', PieceType.KING, 'w'),
    ('a8', PieceType.KING, 'b'),
    ('e8', PieceType.ROOK, 'b'),
)
assert is_in_check(clear_file, 'w')
checked_game = ChessGame(clear_file)
assert checked_game.status == GameStatus.CHECK
assert all(
    not is_in_check(move.execute(checked_game.board), 'w')
    for move in checked_game.legal_moves()
)

blocked_file = clear_file.with_piece(
    sq('e4'), Piece(PieceType.PAWN, 'w')
)
blocked_game = ChessGame(blocked_file)
assert not is_in_check(blocked_file, 'w')
assert blocked_game.status == GameStatus.ACTIVE
print('Clear e-file puts White in check:', is_in_check(clear_file, 'w'))
print('Pawn on e4 blocks the rook:', not is_in_check(blocked_file, 'w'))

# %% [markdown]
# ## A pinned piece cannot expose its king
#
# The rook on e2 shields the White king on e1 from the Black rook on e8. The rook moves geometrically to d2, but that move uncovers the e-file and leaves the king in check, so it is not a legal move. A rook move that keeps blocking the file, such as e2-e3, remains legal.

# %%
pinned_game = ChessGame(position(
    ('e1', PieceType.KING, 'w'),
    ('e2', PieceType.ROOK, 'w'),
    ('a8', PieceType.KING, 'b'),
    ('e8', PieceType.ROOK, 'b'),
))
rook_moves = tuple(
    move for move in pinned_game.legal_moves()
    if move.from_square == sq('e2')
)
rook_targets = {move.to_square for move in rook_moves}
assert sq('d2') not in rook_targets
assert sq('e3') in rook_targets
assert not is_in_check(pinned_game.board, 'w')
print('Rook may move to d2:', sq('d2') in rook_targets)
print('Rook may move to e3:', sq('e3') in rook_targets)

# %% [markdown]
# ## Stateful special moves
#
# These rules depend on game state as well as the board. A custom position has no castling rights unless they are explicitly supplied. En passant is available only immediately after the opposing pawn's two-square push. A pawn reaching its last rank must choose a promotion piece.

# %% [markdown]
# **Predict, then run:** in the position below, which squares will the White king and rook occupy after `O-O`? The game is explicitly given the `K` kingside castling right.

# %%
castle_game = ChessGame(
    position(
        ('e1', PieceType.KING, 'w'),
        ('h1', PieceType.ROOK, 'w'),
        ('e8', PieceType.KING, 'b'),
    ),
    castling_rights=frozenset('K'),
)
castle_game.make_move('O-O')
assert castle_game.board.get(sq('g1')) == Piece(PieceType.KING, 'w')
assert castle_game.board.get(sq('f1')) == Piece(PieceType.ROOK, 'w')
print('King on g1:', castle_game.board.get(sq('g1')))
print('Rook on f1:', castle_game.board.get(sq('f1')))

# %%
en_passant_game = ChessGame()
for notation in ('e2e4', 'a7a6', 'e4e5', 'd7d5'):
    en_passant_game.make_move(notation)
assert any(move.special == 'en_passant' for move in en_passant_game.legal_moves())
en_passant_game.make_move('e5d6')
assert en_passant_game.board.get(sq('d5')) is None
assert en_passant_game.board.get(sq('d6')) == Piece(PieceType.PAWN, 'w')
print('Captured pawn square d5:', en_passant_game.board.get(sq('d5')))
print('White pawn after en passant:', en_passant_game.board.get(sq('d6')))

# %%
promotion_game = ChessGame(position(
    ('h1', PieceType.KING, 'w'),
    ('h8', PieceType.KING, 'b'),
    ('a7', PieceType.PAWN, 'w'),
))
promotion_choices = tuple(
    move.promotion_to for move in promotion_game.legal_moves()
    if move.from_square == sq('a7') and move.to_square == sq('a8')
)
assert promotion_choices == (
    PieceType.QUEEN, PieceType.ROOK, PieceType.BISHOP, PieceType.KNIGHT
)
promotion_game.make_move('a7a8=Q')
assert promotion_game.board.get(sq('a8')) == Piece(PieceType.QUEEN, 'w')
print('Available promotions:', promotion_choices)
print('Chosen piece on a8:', promotion_game.board.get(sq('a8')))

# %% [markdown]
# ## Checkmate, stalemate, and draws
#
# Checkmate is a win; stalemate is a draw. `is_draw()` also recognizes the
# 50-move threshold, threefold repetition, and insufficient material.
# This implementation treats the 50-move and threefold thresholds as draws
# immediately; it does not model FIDE's claim workflow or 75-move/fivefold
# automatic-draw distinction. Rule draws do not change `GameStatus`.
#
# **Predict, then run:** which draw reason applies to each of these positions?
#
# %%
fifty_move_game = ChessGame.from_fen(
    'k7/8/8/8/8/8/8/7K w - - 100 51'
)
insufficient_game = ChessGame.from_fen(
    'k7/8/8/8/8/8/8/7K w - - 0 1'
)
repetition_game = ChessGame()
for _ in range(2):
    for notation in ('g1f3', 'g8f6', 'f3g1', 'f6g8'):
        repetition_game.make_move(notation)

assert fifty_move_game.is_fifty_moves()
assert fifty_move_game.draw_reason() == 'fifty-move rule'
assert insufficient_game.is_insufficient_material()
assert insufficient_game.draw_reason() == 'insufficient material'
assert repetition_game.is_threefold_repetition()
assert repetition_game.draw_reason() == 'threefold repetition'
print('100 halfmoves:', fifty_move_game.draw_reason())
print('Bare kings:', insufficient_game.draw_reason())
print('Repeated start position:', repetition_game.draw_reason())
print('Draw status remains separate:', repetition_game.status)

# %%
mate_game = ChessGame()
for notation in ('f3', 'e5', 'g4', 'Qh4#'):
    mate_game.make_move(notation)
assert mate_game.status == GameStatus.CHECKMATE
assert mate_game.get_winner() == 'b'
print("Fool's Mate status:", mate_game.status)
print('Winner:', mate_game.get_winner())

# %%
stalemate_game = ChessGame(
    position(
        ('a8', PieceType.KING, 'b'),
        ('c6', PieceType.KING, 'w'),
        ('b6', PieceType.QUEEN, 'w'),
    ),
    turn='b',
)
assert stalemate_game.status == GameStatus.STALEMATE
assert stalemate_game.is_draw()
print('Black to move:', stalemate_game.turn)
print('Stalemate:', stalemate_game.status)
print('Recognized as draw:', stalemate_game.is_draw())


# %% [markdown]
# ## Perft: verifying move generation by counting
#
# Perft (Performance Test) counts legal move paths ending at a fixed depth. The [Chess Programming Wiki](https://www.chessprogramming.org/Perft) documents the algorithm and published results. Matching canonical counts across several positions and depths gives strong confidence in move generation, but does not prove correctness for every position.
#
# **Perft Divide** reports the leaf count for each root move, isolating a bug to one branch. See the [Perft Results](https://www.chessprogramming.org/Perft_Results) page for canonical values.

# %%

def perft(game: ChessGame, depth: int) -> int:
    if depth == 0:
        return 1
    nodes = 0
    for move in game.legal_moves():
        branch = deepcopy(game)
        branch.make_move(move)
        nodes += perft(branch, depth - 1)
    return nodes

def perft_divide(game: ChessGame, depth: int) -> dict[str, int]:
    results = {}
    for move in game.legal_moves():
        branch = deepcopy(game)
        branch.make_move(move)
        results[move_notation(move)] = perft(branch, depth - 1)
    return results

game = ChessGame()
# CPW canonical counts for the starting position
assert perft(game, 1) == 20
assert perft(game, 2) == 400
print('Depth 1:', perft(game, 1))
print('Depth 2:', perft(game, 2))

# %%
divide_result = perft_divide(ChessGame(), 1)
for mv, count in sorted(divide_result.items()):
    print(f'  {mv}: {count}')
print(f'Total moves: {sum(divide_result.values())}')
assert sum(divide_result.values()) == 20

# %% [markdown]
# **Predict, then run:** [Kiwipete](https://www.chessprogramming.org/Perft_Results) is a tricky CPW test position that exercises castling, en passant, and promotions. How many legal moves does White have?

# %%
kiwipete = ChessGame.from_fen(
    'r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq -'
)
kiwipete_depth1 = perft(kiwipete, 1)
assert kiwipete_depth1 == 48
print(f'Kiwipete depth 1: {kiwipete_depth1} (expected 48)')

# %% [markdown]
# If any count disagrees with the reference, `perft_divide` at that depth isolates which root move is wrong — then recurse into that branch at depth-1.

# %% [markdown]
# ## Takeaways
#
# - A generated `Move` carries capture and special-move metadata.
# - `ChessGame` filters geometrically possible moves so they cannot leave your king in check, and commits state when a move is made.
# - Castling rights, en-passant timing, and promotion choice are part of legal game state.
# - `is_draw()` recognizes stalemate, the 50-move threshold, threefold repetition, and insufficient material; its claim semantics are simplified.
# - Perft is the standard correctness test for move generators: recursive leaf counts must match published reference values.
#
# Next: [Engine decisions](03_engine_decisions.ipynb).

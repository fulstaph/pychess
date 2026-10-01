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
# This lesson explores material, piece-square tables, move ordering, and alpha-beta search. You will trace the engine's evaluation inputs, compare search work with and without MVV-LVA ordering, and see how quiescence extends tactical lines at the search horizon.

# %%
from copy import deepcopy

from chess import Board, ChessGame, Piece, PieceType, notation_to_coords
from chess.engine import (
    PIECE_TABLES,
    PIECE_VALUES,
    EngineConfig,
    SearchEngine,
    choose_move,
    evaluate,
    move_notation,
    mvv_lva_score,
    order_moves,
)

# %% [markdown]
# ## The shipped engine: evaluation and search
#
# The default `basic` evaluator scores material and piece-square tables. The
# optional `positional` profile adds phase-tapered king activity, bishop-pair
# and pawn-structure terms. Paired matches were inconclusive, so positional
# evaluation remains opt-in. `EngineConfig` defaults to depth 3 with quiescence
# and a bounded transposition table; reusable `SearchEngine` instances retain
# search knowledge between calls.

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
searcher = SearchEngine(EngineConfig(search_depth=3))
opening_move = searcher.choose_move(initial_game)
assert opening_move in initial_game.legal_moves()
print('Material values:', PIECE_VALUES)
print('Opening move:', move_notation(opening_move))

# %% [markdown]
# After `1. e4 d5`, White can capture the pawn on d5. Compare the engine's selected move after its search; `make_move()` still applies the normal legal-state updates.

# %%
capture_game = ChessGame()
capture_game.make_move('e4')
capture_game.make_move('d5')
selected = searcher.choose_move(capture_game)
assert selected.is_capture()
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
# ## Piece-square tables: where pieces want to stand
#
# Beyond raw material value, engines reward pieces for occupying strategically good squares. Our engine stores **sparse tables** — only bonus squares are recorded, so most squares default to zero. Full 8x8 piece-square tables (PSTs), such as those from Michniewski's [Simplified Evaluation Function](https://www.chessprogramming.org/Simplified_Evaluation_Function), assign a value to every square for every piece type.
#
# See the [Chess Programming Wiki on Piece-Square Tables](https://www.chessprogramming.org/Piece-Square_Tables) for a survey of table designs.
#
# **Predict:** Which piece types do you expect to have the largest central bonuses?

# %%
sq = notation_to_coords


def show_table(name: str, table: dict[tuple[int, int], int]) -> None:
    print(f"\n{name} positional bonuses (from White's perspective):")
    print('     a    b    c    d    e    f    g    h')
    for row in range(8):
        rank = 8 - row
        values = [f'{table.get((row, col), 0):4d}' for col in range(8)]
        print(f"  {rank}  {'  '.join(values)}")


for piece_type, table in PIECE_TABLES.items():
    show_table(piece_type.value.title(), table)

# %%
# A knight on c3 earns a PST bonus; one on the rim at a1 does not.
center_knight = Board()
center_knight = center_knight.with_piece(sq('c3'), Piece(PieceType.KNIGHT, 'w'))
center_knight = center_knight.with_piece(sq('a8'), Piece(PieceType.KING, 'w'))
center_knight = center_knight.with_piece(sq('h8'), Piece(PieceType.KING, 'b'))

rim_knight = Board()
rim_knight = rim_knight.with_piece(sq('a1'), Piece(PieceType.KNIGHT, 'w'))
rim_knight = rim_knight.with_piece(sq('a8'), Piece(PieceType.KING, 'w'))
rim_knight = rim_knight.with_piece(sq('h8'), Piece(PieceType.KING, 'b'))

print(f'Developed knight (c3) eval: {evaluate(center_knight)}')
print(f'Rim knight (a1) eval:       {evaluate(rim_knight)}')
assert evaluate(center_knight) > evaluate(rim_knight)

# %% [markdown]
# ## MVV-LVA: searching the best captures first
#
# **Most Valuable Victim / Least Valuable Attacker** ([CPW](https://www.chessprogramming.org/MVV-LVA)) is a move-ordering heuristic. Captures of valuable pieces by low-value attackers are ranked highly. Alpha-beta can search fewer nodes when promising moves are examined first, though the benefit depends on the position and ordering.
#
# **Predict:** After `1. e4 d5`, which move should MVV-LVA rank first?

# %%
mvv_game = ChessGame()
mvv_game.make_move('e4')
mvv_game.make_move('d5')
moves = mvv_game.legal_moves()
ordered = order_moves(moves)

print(f'Total legal moves: {len(moves)}')
print('Top 5 after ordering:')
for move in ordered[:5]:
    capture = (
        f' captures {move.captured_piece.type.value}'
        if move.captured_piece else ''
    )
    print(
        f'  {move_notation(move)}{capture} '
        f'(MVV-LVA score: {mvv_lva_score(move)})'
    )
assert ordered[0].captured_piece is not None

# %% [markdown]
# ## Move ordering and alpha-beta work
#
# This small search counter evaluates the same fixed-depth position twice:
# once in the game's natural legal-move order, and once with MVV-LVA ordering.
# Both should return the same minimax score; alpha-beta may visit different
# numbers of nodes because its cutoffs depend on move order.

# %%
def alpha_beta_nodes(
    game: ChessGame,
    depth: int,
    alpha: int,
    beta: int,
    use_mvv_lva: bool,
) -> tuple[int, int]:
    if depth == 0:
        return evaluate(game.board), 1

    moves = list(game.legal_moves())
    if not moves:
        return evaluate(game.board), 1
    if use_mvv_lva:
        moves = order_moves(moves)

    nodes = 1
    if game.turn == 'w':
        value = -1_000_000
        for move in moves:
            score, child_nodes = alpha_beta_nodes(
                game.after(move), depth - 1, alpha, beta, use_mvv_lva
            )
            nodes += child_nodes
            value = max(value, score)
            alpha = max(alpha, value)
            if alpha >= beta:
                break
    else:
        value = 1_000_000
        for move in moves:
            score, child_nodes = alpha_beta_nodes(
                game.after(move), depth - 1, alpha, beta, use_mvv_lva
            )
            nodes += child_nodes
            value = min(value, score)
            beta = min(beta, value)
            if alpha >= beta:
                break
    return value, nodes


plain_score, plain_nodes = alpha_beta_nodes(
    mvv_game, 3, -1_000_000, 1_000_000, False
)
ordered_score, ordered_nodes = alpha_beta_nodes(
    mvv_game, 3, -1_000_000, 1_000_000, True
)
assert plain_score == ordered_score
print(f'Natural move order: {plain_nodes:,} nodes')
print(f'MVV-LVA order:      {ordered_nodes:,} nodes')

# %% [markdown]
# ## Quiescence search: seeing past the horizon
#
# A fixed-depth search can stop before an obvious recapture, making a losing
# trade look profitable. This is the **horizon effect** ([CPW](https://www.chessprogramming.org/Horizon_Effect)).
# **Quiescence search** ([CPW](https://www.chessprogramming.org/Quiescence_Search))
# reduces it by extending forcing captures at the leaves. This engine also
# searches promotions and legal evasions while in check, with a bounded
# quiescence depth.
#
# **Predict:** Will enabling quiescence change the engine's move after `1. e4 d5`?

# %%
q_game = ChessGame()
q_game.make_move('e4')
q_game.make_move('d5')

without_q = choose_move(q_game, depth=2, quiescence=False)
with_q = choose_move(q_game, depth=2, quiescence=True)
print(f'Without quiescence: {move_notation(without_q)}')
print(f'With quiescence:    {move_notation(with_q)}')

# %% [markdown]
# ## Reflect and try the engine
#
# **Takeaways:**
#
# - Piece-square tables encode positional knowledge compactly — the [Chess Programming Wiki](https://www.chessprogramming.org/Piece-Square_Tables) catalogs popular table designs.
# - MVV-LVA can reduce alpha-beta work by finding cutoffs earlier; its effect depends on the position.
# - Quiescence search reduces horizon errors by extending tactical captures and checking evasions.
#
# **Questions to consider:**
#
# - Which squares receive a bonus for each piece type in this engine?
# - When could MVV-LVA order a bad capture ahead of a useful quiet move?
# - What tactical information does a fixed-depth, static evaluation miss?
#
# For an interactive game against the shipped engine, run `python -m chess.engine` from the repository root. The notebook task launches JupyterLab with the optional dependencies installed.

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
# # 16. Endgame programming: geometry, tempo, and exact knowledge
#
# **Goals:** reason about king opposition and triangulation, calculate pawn races with the rule of the square, recognize zugzwang, and separate strategic heuristics from exact tablebase information. This goes beyond general-purpose material/PST evaluation: in sparse positions, king activity, tempo, and the move rule can dominate static piece values.
#
# References: [Opposition](https://www.chessprogramming.org/Opposition), [Triangulation](https://www.chessprogramming.org/Triangulation), [Rule of the Square](https://www.chessprogramming.org/Rule_of_the_Square), [Zugzwang](https://www.chessprogramming.org/Zugzwang), [Endgame Tablebases](https://www.chessprogramming.org/Endgame_Tablebases), and [Syzygy Bases](https://www.chessprogramming.org/Syzygy_Bases).

# %% [markdown]
# ## Read a small position as a diagram
#
# FEN is a compact diagram plus turn and rule state. In the first example White's king on e4 faces Black's king on e6; the pawn on e5 is protected by the king. A king's opposition is not just distance: the side *not* to move can often retain a crucial barrier, while moving first may have to give way. The actual result depends on the complete position, especially pawn placement and whose turn it is.

# %%
from chess import ChessGame, notation_to_coords
from chess.game import print_board

opposition_fen = '8/8/4k3/4P3/4K3/8/8/8 w - - 0 1'
opposition = ChessGame.from_fen(opposition_fen)
print('White to move:')
print_board(opposition.board)
assert opposition.turn == 'w'
assert opposition.board.squares[notation_to_coords('e4')[0]][notation_to_coords('e4')[1]].type.value == 'k'

# %% [markdown]
# Kings on the same file/rank or diagonal can oppose one another with an odd number of squares between them. Triangulation is a related tempo maneuver: a king takes a three-step route to return to a nearby square while handing the move to the opponent. These are strategic ideas, not a simple distance score; legal paths, opposition direction, and pawn constraints matter. A static evaluator can reward king proximity to useful squares, but cannot safely declare a win from that heuristic alone.

# %% [markdown]
# ## Rule of the square: calculate a pawn race
#
# For an unobstructed pawn with no help from its king, draw the square from its current square to its promotion rank; the square's side length equals the pawn's remaining forward moves. If the defending king can enter the square by its next move, it can usually catch the pawn. This is a geometric shortcut, not a full legal-move proof: side to move, a pawn's initial two-square option, blocking pieces, and checks can alter the race.

# %%
FILES = 'abcdefgh'


def square(file_index: int, rank: int) -> tuple[int, int]:
    """Return a board coordinate as (file index, rank), using chess ranks 1..8."""
    return file_index, rank


def king_distance(a: tuple[int, int], b: tuple[int, int]) -> int:
    """Chebyshev distance: minimum king moves on an empty board."""
    return max(abs(a[0] - b[0]), abs(a[1] - b[1]))


def pawn_square_contains(pawn: tuple[int, int], king: tuple[int, int], color: str) -> bool:
    """Whether the king lies in the pawn's promotion square (not a full solver)."""
    file_index, rank = pawn
    moves_left = 8 - rank if color == 'w' else rank - 1
    if moves_left <= 0:
        return king == pawn
    forward = 1 if color == 'w' else -1
    target_rank = rank + forward * moves_left
    low_file = max(0, file_index - moves_left)
    high_file = min(7, file_index + moves_left)
    return low_file <= king[0] <= high_file and min(rank, target_rank) <= king[1] <= max(rank, target_rank)


white_pawn = square(FILES.index('a'), 5)  # a5: three pushes remain
black_king = square(FILES.index('d'), 7)
print('a5 promotion square includes d7:', pawn_square_contains(white_pawn, black_king, 'w'))
assert pawn_square_contains(white_pawn, black_king, 'w')
assert king_distance(white_pawn, black_king) == 3

# Move the king one square farther away: it cannot catch up by king moves alone.
far_king = square(FILES.index('e'), 7)
assert not pawn_square_contains(white_pawn, far_king, 'w')

# %% [markdown]
# Our coordinate convention above is deliberately explicit: file `a` is 0, rank 1 is 1. This is distinct from the repository's row/column helper convention, where row zero is rank eight. Keep board-coordinate transforms visible in chess code to avoid off-by-one race errors.

# %% [markdown]
# ## Zugzwang, conversion, and drawing technique
#
# **Zugzwang** is a position where every legal move worsens the mover's outcome. In many king-and-pawn endings, a defender can hold a key square by waiting; a triangulation can force that defender to move and lose the opposition. Conversion technique means improving the king, creating a passed pawn, and avoiding stalemate or perpetual counterplay rather than simply trading pieces. Drawing technique reverses those priorities: blockade pawns, seek an active king, and avoid passive positions where every move yields ground.
#
# Endgame-specific king activity is therefore more than a generic king-safety term. With queens absent, centralization and access to passed pawns often become assets. Yet these are heuristics and plans; the legal state tree determines the truth, and turn/halfmove history can affect draw claims.

# %% [markdown]
# ## Tablebases: exact outcomes with rule-sensitive limits
#
# Endgame tablebases exhaustively solve positions within a supported piece-count/material domain. Syzygy separates **WDL** (win/draw/loss under its rule conventions) from **DTZ** (distance to the next zeroing move: pawn move or capture), useful for managing the 50-move rule. WDL is not an instruction that every practical line wins under every clock/rule setting. A nominal win may require a zeroing move before the claim threshold; a long non-zeroing sequence can turn a theoretical result into a draw under the applicable move-count rule. Claim/automatic-draw details and tablebase conventions matter.
#
# A tablebase answer is exact for the encoded position, supported tablebase set, and its rule model. A heuristic evaluator's king-activity bonus is not equivalent. This lesson downloads or probes no tablebase and adds no dependency.

# %%
# A position's FEN records the halfmove clock, one input to 50-move adjudication.
clock_example = ChessGame.from_fen('8/8/4k3/8/4K3/8/8/8 w - - 99 1')
print('Halfmove-clock position:', clock_example.to_fen())
assert clock_example.to_fen().split()[4] == '99'

# %% [markdown]
# ## Takeaways
#
# - Opposition and triangulation are tempo/geometry concepts; the side to move can change the value of an otherwise similar diagram.
# - The square rule is a fast race test, not a replacement for legal search.
# - Zugzwang explains why a position may be lost despite no immediate tactical threat. Conversion and defense both demand purposeful king activity.
# - Tablebases provide exact results within their domain; WDL and DTZ answer different questions, and the 50-move rule is a practical caveat.
#
# **Reflect:** Which assumptions did the square-rule helper make? How could an initial two-square pawn push, a blocking piece, or a checking move change its conclusion? What information would an endgame evaluator need to distinguish an active king from a king merely near the center?
#
# Next: [NNUE foundations](17_nnue_foundations.ipynb) introduces incrementally updated learned evaluation.

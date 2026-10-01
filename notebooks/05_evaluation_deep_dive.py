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

# %% [markdown] cell:0
# 5. Evaluation deep dive
#
# This lesson goes inside the engine's **static evaluation function** — the
# score it assigns to a position without searching any further.  We build
# up from raw material counting through piece-square tables (PSTs) to
# pawn-structure heuristics and mobility, following the progression
# described on the [Chess Programming Wiki](https://www.chessprogramming.org/Evaluation).
#
# Along the way we will construct a richer evaluation from scratch and
# compare it against the engine's built-in `evaluate()`.
# %% [code] cell:1
from chess import Board, ChessGame, Piece, PieceType, notation_to_coords, print_board
from chess.engine import PIECE_TABLES, PIECE_VALUES, evaluate

sq = notation_to_coords


def position(*placements: tuple[str, PieceType, str]) -> Board:
    board = Board()
    for text, kind, color in placements:
        board = board.with_piece(sq(text), Piece(kind, color))
    return board
# %% [markdown] cell:2
# ## Material: the foundation
#
# The simplest evaluation counts the value of pieces on the board.
# Standard centipawn values (pawn = 100) originated in early computer
# chess and are documented on the
# [CPW Simplified Evaluation page](https://www.chessprogramming.org/Simplified_Evaluation_Function):
#
# | Piece  | Value |
# |--------|------:|
# | Pawn   |   100 |
# | Knight |   320 |
# | Bishop |   330 |
# | Rook   |   500 |
# | Queen  |   900 |
# | King   |     0 |
#
# The king is invaluable (checkmate ends the game), so its material
# weight is zero — only its positional safety matters.
# %% [code] cell:3
def material_score(board: Board) -> int:
    """White-positive material balance."""
    score = 0
    for row in range(8):
        for col in range(8):
            piece = board.get((row, col))
            if piece is None:
                continue
            value = PIECE_VALUES[piece.type]
            score += value if piece.color == 'w' else -value
    return score


starting = Board.from_notation()
assert material_score(starting) == 0  # symmetric
print(f'Starting material balance: {material_score(starting)} (symmetric)')

# White up a queen
up_queen = position(
    ('e1', PieceType.KING, 'w'),
    ('d1', PieceType.QUEEN, 'w'),
    ('e8', PieceType.KING, 'b'),
)
assert material_score(up_queen) == 900
print(f'White queen vs bare king:  +{material_score(up_queen)} cp')
# %% [markdown] cell:4
# ## Piece-square tables: where pieces want to stand
#
# Material alone cannot distinguish between a knight stuck on the rim
# and one dominating the centre.  **Piece-square tables** (PSTs) add a
# small positional bonus or penalty depending on where each piece sits.
#
# The concept is explained in detail on the
# [CPW Piece-Square Tables page](https://www.chessprogramming.org/Piece-Square_Tables).
# Tomasz Michniewski's [Simplified Evaluation Function](https://www.chessprogramming.org/Simplified_Evaluation_Function)
# popularised a full set of 8x8 tables for educational engines.
#
# Our engine uses **sparse** tables — only squares with a non-zero bonus
# are stored.  Let's visualise them.
# %% [code] cell:5
def show_table(name: str, table: dict[tuple[int, int], int]) -> None:
    print(f'\n{name} bonuses (White perspective, centipawns):')
    print('     a    b    c    d    e    f    g    h')
    for row in range(8):
        rank = 8 - row
        values = [f'{table.get((row, col), 0):4d}' for col in range(8)]
        print(f"  {rank} {''.join(values)}")


for piece_type, table in PIECE_TABLES.items():
    show_table(piece_type.value.title(), table)
# %% [markdown] cell:6
# **Predict, then run:** which is worth more to the engine — a knight
# developed to c3 or one on a1 (corner)? The material is identical
# (320 cp); only the PST bonus differs.

# %% [code] cell:7
centre_knight = position(
    ('a8', PieceType.KING, 'w'),
    ('h8', PieceType.KING, 'b'),
    ('c3', PieceType.KNIGHT, 'w'),
)
corner_knight = position(
    ('a8', PieceType.KING, 'w'),
    ('h8', PieceType.KING, 'b'),
    ('a1', PieceType.KNIGHT, 'w'),
)

eval_centre = evaluate(centre_knight)
eval_corner = evaluate(corner_knight)
assert eval_centre > eval_corner
print(f'Knight c3: {eval_centre} cp  (material 320 + bonus {eval_centre - 320})')
print(f'Knight a1: {eval_corner} cp  (material 320 + bonus {eval_corner - 320})')
print(f'c3 advantage: +{eval_centre - eval_corner} cp')
# %% [markdown] cell:8
# ## Full 8x8 PSTs: Michniewski's simplified tables
#
# The tables below come from the CPW
# [Simplified Evaluation Function](https://www.chessprogramming.org/Simplified_Evaluation_Function).
# They are defined from White's perspective (row 0 = rank 8); Black's
# values are mirrored by reading the table upside-down (`row → 7 - row`).
#
# We build them here so we can compare a full PST evaluation against
# the engine's sparse version.
# %% [code] cell:9
# Michniewski tables — centipawns, from White's viewpoint (rank 8 at top)
FULL_PAWN_TABLE = [
    [  0,   0,   0,   0,   0,   0,   0,   0],
    [ 50,  50,  50,  50,  50,  50,  50,  50],
    [ 10,  10,  20,  30,  30,  20,  10,  10],
    [  5,   5,  10,  25,  25,  10,   5,   5],
    [  0,   0,   0,  20,  20,   0,   0,   0],
    [  5,  -5, -10,   0,   0, -10,  -5,   5],
    [  5,  10,  10, -20, -20,  10,  10,   5],
    [  0,   0,   0,   0,   0,   0,   0,   0],
]

FULL_KNIGHT_TABLE = [
    [-50, -40, -30, -30, -30, -30, -40, -50],
    [-40, -20,   0,   0,   0,   0, -20, -40],
    [-30,   0,  10,  15,  15,  10,   0, -30],
    [-30,   5,  15,  20,  20,  15,   5, -30],
    [-30,   0,  15,  20,  20,  15,   0, -30],
    [-30,   5,  10,  15,  15,  10,   5, -30],
    [-40, -20,   0,   5,   5,   0, -20, -40],
    [-50, -40, -30, -30, -30, -30, -40, -50],
]

FULL_BISHOP_TABLE = [
    [-20, -10, -10, -10, -10, -10, -10, -20],
    [-10,   0,   0,   0,   0,   0,   0, -10],
    [-10,   0,   5,  10,  10,   5,   0, -10],
    [-10,   5,   5,  10,  10,   5,   5, -10],
    [-10,   0,  10,  10,  10,  10,   0, -10],
    [-10,  10,  10,  10,  10,  10,  10, -10],
    [-10,   5,   0,   0,   0,   0,   5, -10],
    [-20, -10, -10, -10, -10, -10, -10, -20],
]

FULL_ROOK_TABLE = [
    [  0,   0,   0,   0,   0,   0,   0,   0],
    [  5,  10,  10,  10,  10,  10,  10,   5],
    [ -5,   0,   0,   0,   0,   0,   0,  -5],
    [ -5,   0,   0,   0,   0,   0,   0,  -5],
    [ -5,   0,   0,   0,   0,   0,   0,  -5],
    [ -5,   0,   0,   0,   0,   0,   0,  -5],
    [ -5,   0,   0,   0,   0,   0,   0,  -5],
    [  0,   0,   0,   5,   5,   0,   0,   0],
]

FULL_QUEEN_TABLE = [
    [-20, -10, -10,  -5,  -5, -10, -10, -20],
    [-10,   0,   0,   0,   0,   0,   0, -10],
    [-10,   0,   5,   5,   5,   5,   0, -10],
    [ -5,   0,   5,   5,   5,   5,   0,  -5],
    [  0,   0,   5,   5,   5,   5,   0,  -5],
    [-10,   5,   5,   5,   5,   5,   0, -10],
    [-10,   0,   5,   0,   0,   0,   0, -10],
    [-20, -10, -10,  -5,  -5, -10, -10, -20],
]

# King middlegame table — prefers castled positions
FULL_KING_MG_TABLE = [
    [-30, -40, -40, -50, -50, -40, -40, -30],
    [-30, -40, -40, -50, -50, -40, -40, -30],
    [-30, -40, -40, -50, -50, -40, -40, -30],
    [-30, -40, -40, -50, -50, -40, -40, -30],
    [-20, -30, -30, -40, -40, -30, -30, -20],
    [-10, -20, -20, -20, -20, -20, -20, -10],
    [ 20,  20,   0,   0,   0,   0,  20,  20],
    [ 20,  30,  10,   0,   0,  10,  30,  20],
]

# King endgame table — prefers centralisation
FULL_KING_EG_TABLE = [
    [-50, -40, -30, -20, -20, -30, -40, -50],
    [-30, -20, -10,   0,   0, -10, -20, -30],
    [-30, -10,  20,  30,  30,  20, -10, -30],
    [-30, -10,  30,  40,  40,  30, -10, -30],
    [-30, -10,  30,  40,  40,  30, -10, -30],
    [-30, -10,  20,  30,  30,  20, -10, -30],
    [-30, -30,   0,   0,   0,   0, -30, -30],
    [-50, -30, -30, -30, -30, -30, -30, -50],
]

FULL_TABLES: dict[PieceType, list[list[int]]] = {
    PieceType.PAWN: FULL_PAWN_TABLE,
    PieceType.KNIGHT: FULL_KNIGHT_TABLE,
    PieceType.BISHOP: FULL_BISHOP_TABLE,
    PieceType.ROOK: FULL_ROOK_TABLE,
    PieceType.QUEEN: FULL_QUEEN_TABLE,
    PieceType.KING: FULL_KING_MG_TABLE,
}

print('Full 8x8 pawn table (White perspective):')
print('     a    b    c    d    e    f    g    h')
for row in range(8):
    rank = 8 - row
    values = [f'{FULL_PAWN_TABLE[row][col]:4d}' for col in range(8)]
    print(f"  {rank} {''.join(values)}")
# %% [markdown] cell:10
# ## Comparing sparse vs full PST evaluation
#
# The engine's sparse tables only reward a few key squares.  A full
# PST evaluation also penalises bad squares (edge, back rank).
# Let's score a middlegame position both ways.
# %% [code] cell:11
def full_pst_evaluate(board: Board) -> int:
    """Material + full Michniewski PST bonus, White-positive."""
    score = 0
    for row in range(8):
        for col in range(8):
            piece = board.get((row, col))
            if piece is None:
                continue
            mat = PIECE_VALUES[piece.type]
            table = FULL_TABLES.get(piece.type)
            if piece.color == 'w':
                bonus = table[row][col] if table else 0
                score += mat + bonus
            else:
                bonus = table[7 - row][col] if table else 0
                score -= mat + bonus
    return score


italian_game = ChessGame()
for mv in ('e4', 'e5', 'Nf3', 'Nc6', 'Bc4', 'Bc5'):
    italian_game.make_move(mv)

engine_eval = evaluate(italian_game.board)
full_eval = full_pst_evaluate(italian_game.board)
print('Italian Game (after 3...Bc5):')
print(f'  Engine (sparse PST): {engine_eval:+d} cp')
print(f'  Full Michniewski:    {full_eval:+d} cp')
print(f'  Difference:          {full_eval - engine_eval:+d} cp')
# %% [markdown] cell:12
# ## Pawn structure: isolated, doubled, and passed pawns
#
# Pawns are the soul of chess (Philidor).  Three classical structural
# weaknesses are documented on the
# [CPW Pawn Structure page](https://www.chessprogramming.org/Pawn_Structure):
#
# - **Isolated pawn**: no friendly pawn on adjacent files. Weak because
#   it cannot be defended by another pawn.
# - **Doubled pawns**: two pawns of the same colour on the same file.
#   One blocks the other's advance.
# - **Passed pawn**: no opposing pawn ahead on the same or adjacent
#   files. Dangerous because it can march to promotion.
# %% [code] cell:13
def pawn_structure(board: Board, color: str) -> dict[str, list[str]]:
    """Identify isolated, doubled, and passed pawns for a colour."""
    files: dict[int, list[int]] = {}  # file -> list of rows
    for row in range(8):
        for col in range(8):
            piece = board.get((row, col))
            if piece and piece.type == PieceType.PAWN and piece.color == color:
                files.setdefault(col, []).append(row)

    opp = 'b' if color == 'w' else 'w'
    opp_files: dict[int, list[int]] = {}
    for row in range(8):
        for col in range(8):
            piece = board.get((row, col))
            if piece and piece.type == PieceType.PAWN and piece.color == opp:
                opp_files.setdefault(col, []).append(row)

    file_names = 'abcdefgh'
    isolated: list[str] = []
    doubled: list[str] = []
    passed: list[str] = []

    for col, rows in files.items():
        # Doubled: more than one pawn on this file
        if len(rows) > 1:
            doubled.extend(f'{file_names[col]}{8 - r}' for r in rows)

        # Isolated: no friendly pawn on adjacent files
        has_neighbour = any(
            adj in files for adj in (col - 1, col + 1)
        )
        if not has_neighbour:
            isolated.extend(f'{file_names[col]}{8 - r}' for r in rows)
        # Passed: no opposing pawn ahead on same or adjacent files
        for r in rows:
            is_passed = True
            advance = range(r - 1, -1, -1) if color == 'w' else range(r + 1, 8)
            for check_col in (col - 1, col, col + 1):
                if check_col < 0 or check_col > 7:
                    continue
                for check_row in advance:
                    opp_piece = board.get((check_row, check_col))
                    if opp_piece and opp_piece.type == PieceType.PAWN and opp_piece.color == opp:
                        is_passed = False
                        break
                if not is_passed:
                    break
            if is_passed:
                passed.append(f'{file_names[col]}{8 - r}')

    return {'isolated': isolated, 'doubled': doubled, 'passed': passed}


# Fixture: a passed a6 pawn, doubled c-pawns, an isolated f4 pawn, and a
# white e5 pawn blocked by the black pawn on e6.
structure_board = position(
    ('a8', PieceType.KING, 'w'),
    ('h8', PieceType.KING, 'b'),
    ('a6', PieceType.PAWN, 'w'),
    ('c2', PieceType.PAWN, 'w'),
    ('c3', PieceType.PAWN, 'w'),
    ('e5', PieceType.PAWN, 'w'),
    ('e6', PieceType.PAWN, 'b'),
    ('f4', PieceType.PAWN, 'w'),
)
print_board(structure_board)
w_struct = pawn_structure(structure_board, 'w')
b_struct = pawn_structure(structure_board, 'b')
print(f'\\nWhite pawns: {w_struct}')
print(f'Black pawns: {b_struct}')
assert 'a6' in w_struct['passed']
assert 'e5' not in w_struct['passed']
assert {'c2', 'c3'} <= set(w_struct['doubled'])
assert 'f4' not in w_struct['isolated']

# French Advance used for the evaluation comparison below.
french = ChessGame()
for mv in ('e4', 'e6', 'd4', 'd5', 'e5'):
    french.make_move(mv)
# %% [markdown] cell:14
# ## Mobility: counting safe squares
#
# A piece that can move to many squares is more active than one that is
# boxed in.  The [CPW Mobility page](https://www.chessprogramming.org/Mobility)
# explains how engines count available squares per piece as an evaluation
# term.  We can approximate mobility using `legal_moves()`.
# %% [code] cell:15
def piece_mobility(game: ChessGame) -> dict[str, int]:
    """Count legal moves per piece type for the side to move."""
    counts: dict[str, int] = {}
    for move in game.legal_moves():
        name = move.piece.type.value.title()
        counts[name] = counts.get(name, 0) + 1
    return counts


# Compare mobility in the Italian vs a cramped position
italian = ChessGame()
for mv in ('e4', 'e5', 'Nf3', 'Nc6', 'Bc4'):
    italian.make_move(mv)

cramped = ChessGame()
for mv in ('d4', 'd5', 'c4', 'e6', 'Nc3', 'c6', 'Nf3'):
    cramped.make_move(mv)

mob_italian = piece_mobility(italian)
mob_cramped = piece_mobility(cramped)
print(f'Italian (Black to move): {mob_italian}')
print(f'  Total: {sum(mob_italian.values())} legal moves')
print("Queen's Gambit Declined setup (Black to move):", mob_cramped)
print(f'  Total: {sum(mob_cramped.values())} legal moves')
# %% [markdown] cell:16
# ## Tapered evaluation: blending middlegame and endgame
#
# Piece values and PSTs should change as pieces leave the board.  A king
# hiding in the corner is ideal in the middlegame but terrible in the
# endgame (it should centralise).  **Tapered evaluation** blends two
# scores:
#
# $$\text{eval} = \frac{\text{phase} \times \text{mg} + (256 - \text{phase}) \times \text{eg}}{256}$$
#
# where `phase` is derived from the remaining material:
#
# | Piece  | Phase weight |
# |--------|-------------:|
# | Pawn   |            0 |
# | Knight |            1 |
# | Bishop |            1 |
# | Rook   |            2 |
# | Queen  |            4 |
#
# Total = 24 (all minor + major pieces).  As pieces are captured,
# `phase` drops from 256 toward 0 and the endgame score dominates.
#
# Reference: [CPW Tapered Eval](https://www.chessprogramming.org/Tapered_Eval).
# %% [code] cell:17
PHASE_WEIGHTS = {
    PieceType.PAWN: 0,
    PieceType.KNIGHT: 1,
    PieceType.BISHOP: 1,
    PieceType.ROOK: 2,
    PieceType.QUEEN: 4,
    PieceType.KING: 0,
}
MAX_PHASE = 24  # 4N + 4B + 4R + 2Q = 4+4+8+8


def game_phase(board: Board) -> int:
    """Return phase weight 0 (endgame) to 256 (opening)."""
    total = 0
    for row in range(8):
        for col in range(8):
            piece = board.get((row, col))
            if piece is not None:
                total += PHASE_WEIGHTS.get(piece.type, 0)
    # Scale to 0-256
    return (total * 256 + MAX_PHASE // 2) // MAX_PHASE


def tapered_king_bonus(board: Board, color: str) -> int:
    """King PST bonus that blends middlegame and endgame tables."""
    phase = game_phase(board)
    bonus = 0
    for row in range(8):
        for col in range(8):
            piece = board.get((row, col))
            if piece and piece.type == PieceType.KING and piece.color == color:
                r = row if color == 'w' else 7 - row
                mg = FULL_KING_MG_TABLE[r][col]
                eg = FULL_KING_EG_TABLE[r][col]
                bonus = (phase * mg + (256 - phase) * eg) // 256
    return bonus


# Opening: full material → king should hide
opening_board = Board.from_notation()
phase_opening = game_phase(opening_board)
king_bonus_opening = tapered_king_bonus(opening_board, 'w')

# Endgame: only kings and pawns → king should centralise
endgame_board = position(
    ('e4', PieceType.KING, 'w'),
    ('e5', PieceType.KING, 'b'),
    ('d2', PieceType.PAWN, 'w'),
    ('d7', PieceType.PAWN, 'b'),
)
phase_endgame = game_phase(endgame_board)
king_bonus_endgame = tapered_king_bonus(endgame_board, 'w')

print(f'Opening phase:  {phase_opening}/256  king(e1) bonus: {king_bonus_opening}')
print(f'Endgame phase:  {phase_endgame}/256  king(e4) bonus: {king_bonus_endgame}')
assert phase_opening > phase_endgame
assert king_bonus_endgame > king_bonus_opening
# %% [markdown] cell:18
# ## Putting it all together
#
# A competitive evaluation combines material, PSTs, pawn structure,
# mobility, king safety, and more.  The function below demonstrates
# material + full PSTs + a simple pawn-structure penalty.
# %% [code] cell:19
ISOLATED_PENALTY = -15   # centipawns
DOUBLED_PENALTY = -10
PASSED_BONUS = 20


def rich_evaluate(game: ChessGame) -> int:
    """Material + full PST + pawn structure, White-positive."""
    board = game.board
    score = full_pst_evaluate(board)

    for color, sign in (('w', 1), ('b', -1)):
        struct = pawn_structure(board, color)
        score += sign * len(struct['isolated']) * ISOLATED_PENALTY
        score += sign * len(struct['doubled']) * DOUBLED_PENALTY
        score += sign * len(struct['passed']) * PASSED_BONUS

    return score


# Compare evaluations on the French Advance
french_engine = evaluate(french.board)
french_rich = rich_evaluate(french)

print('French Advance (after 3.e5):')
print(f'  Engine eval:  {french_engine:+d} cp')
print(f'  Rich eval:    {french_rich:+d} cp')
print(f'  Difference:   {french_rich - french_engine:+d} cp')
# %% [markdown] cell:20
# ## Takeaways
#
# - **Material** is the bedrock of evaluation; centipawn scaling (P=100) is universal ([CPW: Material](https://www.chessprogramming.org/Material)).
# - **Piece-square tables** add positional awareness without expensive computation ([CPW: PST](https://www.chessprogramming.org/Piece-Square_Tables)).
# - **Pawn structure** penalties (isolated, doubled) and bonuses (passed) capture strategic value ([CPW: Pawn Structure](https://www.chessprogramming.org/Pawn_Structure)).
# - **Mobility** rewards active pieces and punishes cramped positions ([CPW: Mobility](https://www.chessprogramming.org/Mobility)).
# - **Tapered evaluation** smoothly blends middlegame and endgame scoring as material leaves the board ([CPW: Tapered Eval](https://www.chessprogramming.org/Tapered_Eval)).
# - Production engines layer dozens more terms (king safety, bishop pair, rook on open files, etc.) and tune weights automatically via the [Texel method](https://www.chessprogramming.org/Texel%27s_Tuning_Method).

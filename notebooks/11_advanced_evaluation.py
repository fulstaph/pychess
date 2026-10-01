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
# # 11. Advanced evaluation: beyond material and activity
#
# **Goals:** reason about king safety, bishop pair, rook files, outposts, passed-
# pawn support, tempo, phase, and interactions; implement several transparent
# white-relative features on real `Board` objects; and identify their limits.
# Lesson 05 already covered material, piece-square tables, pawn structure,
# mobility, and tapered evaluation. The optional positional profile adds
# tapered king activity, bishop-pair, and isolated/doubled/passed-pawn terms.
# It remains opt-in pending stronger match evidence. This lesson's added terms
# remain experiments until separately measured.
#
# Sources: [CPW: Evaluation](https://www.chessprogramming.org/Evaluation),
# [King Safety](https://www.chessprogramming.org/King_Safety),
# [Bishop Pair](https://www.chessprogramming.org/Bishop_Pair),
# [Rook Open File](https://www.chessprogramming.org/Rook_Open_File),
# [Outpost](https://www.chessprogramming.org/Outpost),
# [Passed Pawns](https://www.chessprogramming.org/Passed_Pawns).

# %%
from chess import Board, Piece, PieceType

WHITE, BLACK = 'w', 'b'


def board_from_fen(fen: str) -> Board:
    return Board.from_fen(fen)


def count_pieces(board: Board, color: str, piece_type: PieceType) -> int:
    return sum(piece is not None and piece.color == color and piece.type == piece_type
               for row in board.squares for piece in row)


def pawn_squares(board: Board, color: str) -> list[tuple[int, int]]:
    return [(r, c) for r, row in enumerate(board.squares)
            for c, p in enumerate(row)
            if p is not None and p.color == color and p.type == PieceType.PAWN]


def king_shelter(board: Board, color: str) -> int:
    """Reward own pawns one or two ranks ahead of king on nearby files."""
    king = board.king_square(color)
    if king is None:
        return 0
    row, col = king
    forward = -1 if color == WHITE else 1
    score = 0
    pawns = set(pawn_squares(board, color))
    for distance, weight in ((1, 12), (2, 5)):
        target_row = row + forward * distance
        if not 0 <= target_row < 8:
            continue
        for target_col in range(max(0, col - 1), min(7, col + 1) + 1):
            if (target_row, target_col) in pawns:
                score += weight
    return score


def rook_file_score(board: Board, color: str) -> int:
    """Reward rooks on fully open or own-pawn-free files (simple file scan)."""
    own_files = {c for _, c in pawn_squares(board, color)}
    enemy_files = {c for _, c in pawn_squares(board, 'b' if color == WHITE else WHITE)}
    score = 0
    for row in board.squares:
        for col, piece in enumerate(row):
            if piece is not None and piece.color == color and piece.type == PieceType.ROOK:
                score += 18 if col not in own_files | enemy_files else (9 if col not in own_files else 0)
    return score


def bishop_pair_score(board: Board, color: str) -> int:
    return 28 if count_pieces(board, color, PieceType.BISHOP) >= 2 else 0


def outpost_score(board: Board, color: str) -> int:
    """Reward a supported central minor on a square no enemy pawn can chase."""
    enemy = 'b' if color == WHITE else WHITE
    own_pawns = pawn_squares(board, color)
    enemy_pawns = pawn_squares(board, enemy)
    score = 0
    for r, row in enumerate(board.squares):
        for c, piece in enumerate(row):
            if piece is None or piece.color != color or piece.type not in (PieceType.KNIGHT, PieceType.BISHOP):
                continue
            # Central advanced outpost ranks: white rows 2..4 (ranks 6..4),
            # black rows 3..5. A friendly pawn diagonally behind supports it.
            advanced = range(2, 5) if color == WHITE else range(3, 6)
            supported = any(pr == r + (1 if color == WHITE else -1) and abs(pc - c) == 1
                            for pr, pc in own_pawns)
            chasable = any(abs(pc - c) <= 1 and ((pr < r) if color == WHITE else (pr > r))
                           for pr, pc in enemy_pawns)
            if r in advanced and supported and not chasable:
                score += 20 if piece.type == PieceType.KNIGHT else 12
    return score


def supported_passers(board: Board, color: str) -> int:
    """Bonus passed pawns backed by a friendly pawn on an adjacent file."""
    enemy = 'b' if color == WHITE else WHITE
    own = pawn_squares(board, color)
    opposing = pawn_squares(board, enemy)
    bonus = 0
    for r, c in own:
        passed = not any(
            abs(ec - c) <= 1
            and ((er < r) if color == WHITE else (er > r))
            for er, ec in opposing
        )
        supported = any(abs(pc - c) == 1 and (pr > r if color == WHITE else pr < r)
                        for pr, pc in own)
        if passed:
            bonus += 18 + (12 if supported else 0)
    return bonus


def phase(board: Board) -> int:
    """Coarse 0..24 phase: queens 4, rooks 2, minors 1 each."""
    weights = {PieceType.QUEEN: 4, PieceType.ROOK: 2,
               PieceType.BISHOP: 1, PieceType.KNIGHT: 1}
    return sum(weights[pt] * (count_pieces(board, WHITE, pt) + count_pieces(board, BLACK, pt))
               for pt in weights)


def advanced_terms(board: Board, turn: str = WHITE) -> dict[str, int]:
    """Illustrative white-positive terms; deliberately not a full evaluator."""
    pair = bishop_pair_score(board, WHITE) - bishop_pair_score(board, BLACK)
    shelter = king_shelter(board, WHITE) - king_shelter(board, BLACK)
    rooks = rook_file_score(board, WHITE) - rook_file_score(board, BLACK)
    outposts = outpost_score(board, WHITE) - outpost_score(board, BLACK)
    passers = supported_passers(board, WHITE) - supported_passers(board, BLACK)
    # Shelter matters more with queens and heavy pieces still present.
    shelter = shelter * min(phase(board), 12) // 12
    return {'bishop_pair': pair, 'king_shelter': shelter, 'rook_files': rooks,
            'outposts': outposts, 'supported_passers': passers,
            'tempo': 8 if turn == WHITE else -8}


def advanced_score(board: Board, turn: str = WHITE) -> int:
    return sum(advanced_terms(board, turn).values())

# %% [markdown]
# ## Reading the terms and their interactions
#
# - **King shelter/safety:** nearby pawns can shield a king, but pawn advances
#   can also create holes; attackers, open lines, and queen presence matter.
# - **Bishop pair:** two bishops often gain value as lines open, but the bonus
#   depends on pawn structure and is not universally decisive.
# - **Rooks:** open files contain no pawns; semi-open files lack friendly pawns.
#   A rook still needs access to useful targets before a bonus means much.
# - **Outposts/weak squares:** an advanced minor piece supported by a pawn and
#   immune to enemy-pawn challenge can be persistent; piece attacks still matter.
# - **Passed-pawn support:** a passed pawn has no opposing pawn ahead on its or
#   adjacent files; a friendly pawn behind/alongside can support its advance.
# - **Tempo/phase:** tempo is a small side-to-move initiative bonus. Phase can
#   scale terms: king shelter generally matters more in a middlegame, while king
#   activity and passer advancement matter more in endings. Lesson 05 derived
#   phase for tapered evaluation; here a coarse phase only scales shelter.
# - **Interactions:** an open rook file may attack a king and increase danger;
#   a bishop pair benefits from open diagonals; a passer may divert a rook.
#   Adding independent bonuses can double-count the same underlying advantage.
#
# These terms are intentionally local and transparent, not tuned or tactical.
# They omit attacks, king checks, pawn races, pins, zugzwang, and nonlinear
# danger. Compare scenarios, not absolute scores.

# %%
# White has two bishops (pair), a rook on an open a-file, and a supported
# advanced passer. Black has only king and rook; terms remain independently
# inspectable rather than collapsed into an opaque total.
strategic = board_from_fen('4k3/8/8/8/P7/8/1P6/R3KBBR w - - 0 1')
terms = advanced_terms(strategic, WHITE)
assert terms['bishop_pair'] == 28
assert terms['rook_files'] > 0
assert terms['supported_passers'] > 0
print('Strategic fixture terms (cp):', terms)
print('Illustrative total:', advanced_score(strategic, WHITE))

# Shelter comparison: same kings/rooks, white pawns shield the kingside king.
sheltered = board_from_fen('4k3/8/8/8/8/5PPP/8/4K2R w - - 0 1')
exposed = board_from_fen('4k3/8/8/8/8/8/8/4K2R w - - 0 1')
assert king_shelter(sheltered, WHITE) > king_shelter(exposed, WHITE)
print('Sheltered vs exposed white king:', king_shelter(sheltered, WHITE),
      'vs', king_shelter(exposed, WHITE))

# %% [markdown]
# ## Symmetry and boundaries
#
# A white-relative evaluator should reverse sign when board colors and ranks are
# mirrored and the side to move is mirrored. This fixture check catches common
# color-orientation mistakes. It is an invariant of this implementation, not a
# claim that the chosen weights are strategically correct.

# %%
def color_mirror(board: Board) -> Board:
    squares = [[None for _ in range(8)] for _ in range(8)]
    for r, row in enumerate(board.squares):
        for c, piece in enumerate(row):
            if piece is not None:
                squares[7 - r][c] = Piece(piece.type, 'b' if piece.color == WHITE else WHITE)
    return Board(tuple(tuple(row) for row in squares))


mirrored = color_mirror(strategic)
assert advanced_score(mirrored, BLACK) == -advanced_score(strategic, WHITE)
assert phase(mirrored) == phase(strategic)
print('Color/rank mirror score:', advanced_score(strategic, WHITE),
      'and', advanced_score(mirrored, BLACK))

# Tempo changes sign without changing board features.
without_tempo = sum(value for name, value in terms.items() if name != 'tempo')
assert advanced_score(strategic, WHITE) == without_tempo + 8
assert advanced_score(strategic, BLACK) == without_tempo - 8

# %% [markdown]
# ## Takeaways
#
# Advanced terms are hypotheses about positional value. Their usefulness depends
# on context, phase, interactions, and tuning against actual outcomes. The
# optional `positional` profile adds tapered king activity, bishop-pair, and
# basic pawn-structure terms; paired matches remain inconclusive. The
# king-shelter, rook-file, outpost, supported-passer, and tempo functions here
# remain experiments outside `evaluate()`.
#
# Reflect: which feature should fade in a pawn ending? When might an open-file
# rook bonus be misleading?
#
# Next: [Engine testing and benchmarking](12_engine_testing_and_benchmarking.ipynb) turns evaluation ideas into measurable validation.

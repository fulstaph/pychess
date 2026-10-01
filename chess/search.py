"""Configurable alpha-beta chess search with tactical and positional heuristics."""

from collections.abc import Sequence
from dataclasses import dataclass
from time import perf_counter
from typing import Literal

from .game import ChessGame, GameStatus, SearchPositionKey
from .helpers import square_notation
from .log import get_logger
from .move import Move
from .piece import Board, Color, Piece, PieceType

logger = get_logger("engine")

PIECE_VALUES: dict[PieceType, int] = {
    PieceType.PAWN: 100,
    PieceType.KNIGHT: 320,
    PieceType.BISHOP: 330,
    PieceType.ROOK: 500,
    PieceType.QUEEN: 900,
    PieceType.KING: 0,
}

PAWN_TABLE: dict[tuple[int, int], int] = {
    (3, 3): 20,
    (3, 4): 20,
    (4, 3): 30,
    (4, 4): 30,
}

KNIGHT_TABLE: dict[tuple[int, int], int] = {
    (5, 2): 15,
    (5, 5): 15,
    (2, 2): 15,
    (2, 5): 15,
}

BISHOP_TABLE: dict[tuple[int, int], int] = {
    (5, 2): 10,
    (5, 5): 10,
    (4, 2): 10,
    (4, 3): 10,
    (4, 4): 10,
    (4, 5): 10,
    (3, 2): 10,
    (3, 3): 10,
    (3, 4): 10,
    (3, 5): 10,
}

ROOK_TABLE: dict[tuple[int, int], int] = {
    (1, 0): 20,
    (1, 1): 20,
    (1, 2): 20,
    (1, 3): 20,
    (1, 4): 20,
    (1, 5): 20,
    (1, 6): 20,
    (1, 7): 20,
}

QUEEN_TABLE: dict[tuple[int, int], int] = {
    (3, 3): 10,
    (3, 4): 10,
    (4, 3): 10,
    (4, 4): 10,
}

KING_TABLE: dict[tuple[int, int], int] = {
    (7, 6): 20,
    (7, 2): 15,
}

KING_ENDGAME_TABLE: tuple[tuple[int, ...], ...] = (
    (-50, -40, -30, -20, -20, -30, -40, -50),
    (-30, -20, -10, 0, 0, -10, -20, -30),
    (-30, -10, 20, 30, 30, 20, -10, -30),
    (-30, -10, 30, 40, 40, 30, -10, -30),
    (-30, -10, 30, 40, 40, 30, -10, -30),
    (-30, -10, 20, 30, 30, 20, -10, -30),
    (-30, -30, 0, 0, 0, 0, -30, -30),
    (-50, -30, -30, -30, -30, -30, -30, -50),
)

PIECE_TABLES: dict[PieceType, dict[tuple[int, int], int]] = {
    PieceType.PAWN: PAWN_TABLE,
    PieceType.KNIGHT: KNIGHT_TABLE,
    PieceType.BISHOP: BISHOP_TABLE,
    PieceType.ROOK: ROOK_TABLE,
    PieceType.QUEEN: QUEEN_TABLE,
    PieceType.KING: KING_TABLE,
}

_COLOR_SIGNS: tuple[tuple[Color, int], ...] = (("w", 1), ("b", -1))

# White-positive material plus table score per piece and square, precomputed
# so ``_basic_evaluate`` is one lookup per piece.
_BASIC_SQUARE_SCORES: dict[Piece, tuple[tuple[int, ...], ...]] = {
    Piece(kind, color): tuple(
        tuple(
            sign * (PIECE_VALUES[kind] + PIECE_TABLES[kind].get((oriented, col), 0))
            for col in range(8)
        )
        for oriented in (range(8) if color == "w" else range(7, -1, -1))
    )
    for kind in PIECE_TABLES
    for color, sign in _COLOR_SIGNS
}

PHASE_WEIGHTS: dict[PieceType, int] = {
    PieceType.PAWN: 0,
    PieceType.KNIGHT: 1,
    PieceType.BISHOP: 1,
    PieceType.ROOK: 2,
    PieceType.QUEEN: 4,
    PieceType.KING: 0,
}
MAX_PHASE = 24
BISHOP_PAIR_BONUS = 28
ISOLATED_PAWN_PENALTY = 15
DOUBLED_PAWN_PENALTY = 10
PASSED_PAWN_BONUSES = (0, 5, 10, 15, 20, 30, 40)
MATE_SCORE = 100_000
MATE_THRESHOLD = 90_000
INFINITY = 1_000_000
MAX_SEARCH_DEPTH = 8

EvaluationProfile = Literal["basic", "positional"]
TTBound = Literal["exact", "lower", "upper"]
HistoryKey = tuple[Color, tuple[int, int], tuple[int, int]]
TranspositionKey = tuple[SearchPositionKey, bool, int, EvaluationProfile]


@dataclass(frozen=True, slots=True)
class EngineConfig:
    """Default engine behavior; per-search arguments override these values."""

    search_depth: int = 3
    quiescence: bool = True
    quiescence_depth: int = 3
    transposition_table_size: int = 65_536
    evaluation_profile: EvaluationProfile = "basic"

    def __post_init__(self) -> None:
        if (
            type(self.search_depth) is not int
            or not 1 <= self.search_depth <= MAX_SEARCH_DEPTH
        ):
            raise ValueError(f"Search depth must be between 1 and {MAX_SEARCH_DEPTH}")
        if type(self.quiescence) is not bool:
            raise ValueError("Quiescence must be a bool")
        if type(self.quiescence_depth) is not int or self.quiescence_depth < 0:
            raise ValueError("Quiescence depth must be a non-negative integer")
        if (
            type(self.transposition_table_size) is not int
            or self.transposition_table_size < 0
        ):
            raise ValueError("Transposition table size must be a non-negative integer")
        if self.evaluation_profile not in ("basic", "positional"):
            raise ValueError("Evaluation profile must be 'basic' or 'positional'")


DEFAULT_ENGINE_CONFIG = EngineConfig()


@dataclass(frozen=True, slots=True)
class SearchStats:
    """Work and result from the most recent :class:`SearchEngine` search."""

    completed_depth: int
    nodes: int
    quiescence_nodes: int
    transposition_probes: int
    transposition_hits: int
    transposition_cutoffs: int
    beta_cutoffs: int
    elapsed_ms: float
    score: int | None
    best_move: Move

    @property
    def nodes_per_second(self) -> int:
        if self.elapsed_ms <= 0:
            return 0
        return int((self.nodes + self.quiescence_nodes) * 1000 / self.elapsed_ms)


@dataclass(frozen=True, slots=True)
class _TTEntry:
    depth: int
    score: int
    bound: TTBound
    best_move: Move | None


@dataclass(slots=True)
class _SearchContext:
    table: dict[TranspositionKey, _TTEntry]
    table_limit: int
    deadline: float | None
    killers: dict[int, list[Move]]
    history: dict[HistoryKey, int]
    nodes: int = 0
    quiescence_nodes: int = 0
    transposition_probes: int = 0
    transposition_hits: int = 0
    transposition_cutoffs: int = 0
    beta_cutoffs: int = 0


class _SearchStoppedError(Exception):
    """Internal signal that a timed search should return its last iteration."""


def game_phase(board: Board) -> int:
    """Return remaining non-pawn phase on a 0 (endgame) to 256 scale."""
    phase = 0
    for row in board.squares:
        for piece in row:
            if piece is not None:
                phase += PHASE_WEIGHTS[piece.type]
    phase = min(phase, MAX_PHASE)
    return (phase * 256 + MAX_PHASE // 2) // MAX_PHASE


def _basic_evaluate(board: Board) -> int:
    scores = _BASIC_SQUARE_SCORES
    score = 0
    for row, squares in enumerate(board.squares):
        for col, piece in enumerate(squares):
            if piece is not None:
                score += scores[piece][row][col]
    return score


def _basic_delta(move: Move) -> int:
    """Change in ``_basic_evaluate`` caused by ``move`` (white-positive)."""
    scores = _BASIC_SQUARE_SCORES
    from_row, from_col = move.from_square
    to_row, to_col = move.to_square
    piece = move.piece
    special = move.special
    placed = (
        Piece(move.promotion_to, piece.color)
        if special == "promotion" and move.promotion_to is not None
        else piece
    )
    delta = scores[placed][to_row][to_col] - scores[piece][from_row][from_col]
    captured = move.captured_piece
    if captured is not None:
        # En passant removes the pawn beside the mover, not on the target.
        capture_row = from_row if special == "en_passant" else to_row
        delta -= scores[captured][capture_row][to_col]
    if special == "castle_k" or special == "castle_q":
        rook = scores[Piece(PieceType.ROOK, piece.color)]
        rook_from, rook_to = (7, 5) if special == "castle_k" else (0, 3)
        delta += rook[from_row][rook_to] - rook[from_row][rook_from]
    return delta


def _static_after(static: int | None, move: Move) -> int | None:
    """Child static score, or ``None`` when no static score is being carried."""
    return None if static is None else static + _basic_delta(move)


def evaluate(board: Board, profile: EvaluationProfile = "basic") -> int:
    """White-positive material and piece-square score with a selectable profile.

    ``basic`` preserves the original material-plus-sparse-PST evaluation for
    controlled ablation. ``positional`` tapers king activity by phase and adds
    bishop-pair and pawn-structure terms.
    """
    if profile == "basic":
        return _basic_evaluate(board)
    if profile != "positional":
        raise ValueError(f"Unknown evaluation profile: {profile!r}")

    score = 0
    king_mg = 0
    king_eg = 0
    phase = 0
    bishops = {"w": 0, "b": 0}
    pawns: dict[Color, list[tuple[int, int]]] = {"w": [], "b": []}
    for row, squares in enumerate(board.squares):
        for col, piece in enumerate(squares):
            if piece is None:
                continue
            color_sign = 1 if piece.color == "w" else -1
            oriented_row = row if piece.color == "w" else 7 - row
            piece_square = PIECE_TABLES[piece.type].get((oriented_row, col), 0)
            phase += PHASE_WEIGHTS[piece.type]
            if piece.type == PieceType.KING:
                king_mg += color_sign * piece_square
                king_eg += color_sign * KING_ENDGAME_TABLE[oriented_row][col]
                score += color_sign * piece_square
            else:
                score += color_sign * (PIECE_VALUES[piece.type] + piece_square)
            if piece.type == PieceType.BISHOP:
                bishops[piece.color] += 1
            elif piece.type == PieceType.PAWN:
                pawns[piece.color].append((row, col))

    phase = min(phase, MAX_PHASE)
    phase = (phase * 256 + MAX_PHASE // 2) // MAX_PHASE
    king_score = (phase * king_mg + (256 - phase) * king_eg) // 256
    score += king_score - king_mg
    if bishops["w"] >= 2:
        score += BISHOP_PAIR_BONUS
    if bishops["b"] >= 2:
        score -= BISHOP_PAIR_BONUS

    for color, own_pawns in pawns.items():
        sign = 1 if color == "w" else -1
        enemy_pawns = pawns["b" if color == "w" else "w"]
        files: list[list[int]] = [[] for _ in range(8)]
        for row, col in own_pawns:
            files[col].append(row)
        for col, rows in enumerate(files):
            for row in rows:
                if not any(
                    files[adjacent]
                    for adjacent in (col - 1, col + 1)
                    if 0 <= adjacent < 8
                ):
                    score -= sign * ISOLATED_PAWN_PENALTY
                if len(rows) > 1:
                    score -= sign * DOUBLED_PAWN_PENALTY
                passed = not any(
                    abs(enemy_col - col) <= 1
                    and (enemy_row < row if color == "w" else enemy_row > row)
                    for enemy_row, enemy_col in enemy_pawns
                )
                if passed:
                    advance = 6 - (row if color == "w" else 7 - row)
                    score += sign * PASSED_PAWN_BONUSES[advance]
    return score


def mvv_lva_score(move: Move) -> int:
    """Score captures and promotions for stable, best-first public ordering."""
    score = 0
    if move.captured_piece is not None:
        victim = PIECE_VALUES.get(move.captured_piece.type, 0)
        attacker = PIECE_VALUES.get(move.piece.type, 0)
        score += 10_000 + victim * 10 - attacker
    if move.special == "promotion":
        score += 9_000
    return score


def order_moves(moves: Sequence[Move]) -> list[Move]:
    """Order captures and promotions first; stable ties preserve input order."""
    return sorted(moves, key=mvv_lva_score, reverse=True)


def move_notation(move: Move) -> str:
    """Display a move in coordinate notation accepted by ``ChessGame``."""
    source = square_notation(*move.from_square)
    target = square_notation(*move.to_square)
    promotion = f"={move.promotion_to.value.upper()}" if move.promotion_to else ""
    return f"{source}{target}{promotion}"


def _mate_score(turn: Color, ply: int) -> int:
    """White-positive mate score; a shorter mate is closer to +-100000."""
    return MATE_SCORE - ply if turn == "b" else -MATE_SCORE + ply


def _score_to_table(score: int, ply: int) -> int:
    if score >= MATE_THRESHOLD:
        return score + ply
    if score <= -MATE_THRESHOLD:
        return score - ply
    return score


def _score_from_table(score: int, ply: int) -> int:
    if score >= MATE_THRESHOLD:
        return score - ply
    if score <= -MATE_THRESHOLD:
        return score + ply
    return score


def _check_time(context: _SearchContext) -> None:
    if context.deadline is not None and perf_counter() >= context.deadline:
        raise _SearchStoppedError


def _table_key(
    game: ChessGame,
    quiescence: bool,
    config: EngineConfig,
) -> TranspositionKey:
    return (
        game._search_key(),
        quiescence,
        config.quiescence_depth,
        config.evaluation_profile,
    )


def _store_entry(
    context: _SearchContext,
    key: TranspositionKey,
    entry: _TTEntry,
) -> None:
    if context.table_limit == 0:
        return
    current = context.table.get(key)
    if current is not None:
        if current.depth > entry.depth:
            return
        if (
            current.depth == entry.depth
            and current.bound == "exact"
            and entry.bound != "exact"
        ):
            return
    elif len(context.table) >= context.table_limit:
        context.table.pop(next(iter(context.table)))
    context.table[key] = entry


def _ordered_search_moves(
    moves: Sequence[Move],
    *,
    hash_move: Move | None,
    ply: int,
    color: Color,
    context: _SearchContext,
) -> list[Move]:
    killers = context.killers.get(ply, ())

    def score(move: Move) -> int:
        if hash_move is not None and move == hash_move:
            return 2_000_000
        if move.is_capture():
            return 1_000_000 + mvv_lva_score(move)
        if move.special == "promotion":
            return 900_000 + (
                PIECE_VALUES[move.promotion_to] if move.promotion_to is not None else 0
            )
        if killers and move == killers[0]:
            return 800_000
        if len(killers) > 1 and move == killers[1]:
            return 700_000
        return min(
            600_000,
            context.history.get((color, move.from_square, move.to_square), 0),
        )

    return sorted(moves, key=score, reverse=True)


def _record_quiet_cutoff(
    move: Move, ply: int, color: Color, depth: int, context: _SearchContext
) -> None:
    if move.is_capture() or move.special == "promotion":
        return
    killers = context.killers.setdefault(ply, [])
    if move not in killers:
        killers.insert(0, move)
        del killers[2:]
    key = (color, move.from_square, move.to_square)
    context.history[key] = min(1_000_000, context.history.get(key, 0) + depth * depth)


def _rule_draw_score(game: ChessGame, ply: int) -> int | None:
    is_draw = (
        game.is_fifty_moves()
        or game.is_threefold_repetition()
        or game.is_insufficient_material()
    )
    if not is_draw:
        return None
    if game.is_check() and not game.legal_moves():
        return _mate_score(game.turn, ply)
    return 0


def _quiescence_search(
    game: ChessGame,
    alpha: int,
    beta: int,
    qdepth: int,
    ply: int,
    context: _SearchContext,
    config: EngineConfig,
    static: int | None,
) -> int:
    context.quiescence_nodes += 1
    _check_time(context)

    draw_score = _rule_draw_score(game, ply)
    if draw_score is not None:
        return draw_score

    in_check = game.is_check()
    if in_check:
        moves = game.legal_moves()
        if not moves:
            return _mate_score(game.turn, ply)
    elif qdepth <= 0:
        if game.status == GameStatus.STALEMATE:
            return 0
        return (
            static
            if static is not None
            else evaluate(game.board, config.evaluation_profile)
        )
    else:
        if not game._has_legal_move():
            return 0
        stand_pat = (
            static
            if static is not None
            else evaluate(game.board, config.evaluation_profile)
        )
        if game.turn == "w":
            if stand_pat >= beta:
                context.beta_cutoffs += 1
                return beta
            alpha = max(alpha, stand_pat)
        else:
            if stand_pat <= alpha:
                context.beta_cutoffs += 1
                return alpha
            beta = min(beta, stand_pat)
        moves = game._legal_tactical_moves()

    ordered = _ordered_search_moves(
        moves, hash_move=None, ply=ply, color=game.turn, context=context
    )
    child_qdepth = qdepth - 1
    if game.turn == "w":
        value = -INFINITY if in_check else alpha
        for move in ordered:
            score = _quiescence_search(
                game._after_generated(move),
                alpha,
                beta,
                child_qdepth,
                ply + 1,
                context,
                config,
                _static_after(static, move),
            )
            value = max(value, score)
            alpha = max(alpha, value)
            if alpha >= beta:
                context.beta_cutoffs += 1
                return beta
        return value

    value = INFINITY if in_check else beta
    for move in ordered:
        score = _quiescence_search(
            game._after_generated(move),
            alpha,
            beta,
            child_qdepth,
            ply + 1,
            context,
            config,
            _static_after(static, move),
        )
        value = min(value, score)
        beta = min(beta, value)
        if alpha >= beta:
            context.beta_cutoffs += 1
            return alpha
    return value


def _quiescence(
    game: ChessGame,
    alpha: int,
    beta: int,
    qdepth: int = 3,
    ply: int = 0,
) -> int:
    """Search captures/promotions; search every legal evasion when checked."""
    context = _SearchContext({}, 0, None, {}, {})
    config = EngineConfig(
        search_depth=1,
        quiescence_depth=max(0, qdepth),
        transposition_table_size=0,
    )
    return _quiescence_search(game, alpha, beta, qdepth, ply, context, config, None)


def _search(
    game: ChessGame,
    depth: int,
    alpha: int,
    beta: int,
    ply: int,
    quiescence: bool,
    context: _SearchContext,
    config: EngineConfig,
    static: int | None,
) -> int:
    _check_time(context)
    if depth == 0 and quiescence:
        return _quiescence_search(
            game,
            alpha,
            beta,
            config.quiescence_depth,
            ply,
            context,
            config,
            static,
        )
    context.nodes += 1

    draw_score = _rule_draw_score(game, ply)
    if draw_score is not None:
        return draw_score
    if depth == 0:
        if game.status == GameStatus.CHECKMATE:
            return _mate_score(game.turn, ply)
        if game.status == GameStatus.STALEMATE:
            return 0
        return (
            static
            if static is not None
            else evaluate(game.board, config.evaluation_profile)
        )

    original_alpha = alpha
    original_beta = beta
    key: TranspositionKey | None = None
    hash_move: Move | None = None
    if context.table_limit:
        key = _table_key(game, quiescence, config)
        context.transposition_probes += 1
        entry = context.table.get(key)
        if entry is not None:
            context.transposition_hits += 1
            hash_move = entry.best_move
            table_score = _score_from_table(entry.score, ply)
            if entry.depth >= depth:
                if entry.bound == "exact":
                    context.transposition_cutoffs += 1
                    return table_score
                if entry.bound == "lower":
                    alpha = max(alpha, table_score)
                else:
                    beta = min(beta, table_score)
                if alpha >= beta:
                    context.transposition_cutoffs += 1
                    return table_score

    moves = game.legal_moves()
    if not moves:
        return _mate_score(game.turn, ply) if game.is_check() else 0

    ordered = _ordered_search_moves(
        moves, hash_move=hash_move, ply=ply, color=game.turn, context=context
    )
    best_move: Move | None = None
    if game.turn == "w":
        value = -INFINITY
        for move in ordered:
            score = _search(
                game._after_generated(move),
                depth - 1,
                alpha,
                beta,
                ply + 1,
                quiescence,
                context,
                config,
                _static_after(static, move),
            )
            if score > value:
                value = score
                best_move = move
            alpha = max(alpha, value)
            if alpha >= beta:
                context.beta_cutoffs += 1
                _record_quiet_cutoff(move, ply, game.turn, depth, context)
                break
    else:
        value = INFINITY
        for move in ordered:
            score = _search(
                game._after_generated(move),
                depth - 1,
                alpha,
                beta,
                ply + 1,
                quiescence,
                context,
                config,
                _static_after(static, move),
            )
            if score < value:
                value = score
                best_move = move
            beta = min(beta, value)
            if alpha >= beta:
                context.beta_cutoffs += 1
                _record_quiet_cutoff(move, ply, game.turn, depth, context)
                break

    if key is not None:
        if value <= original_alpha:
            bound: TTBound = "upper"
        elif value >= original_beta:
            bound = "lower"
        else:
            bound = "exact"
        _store_entry(
            context,
            key,
            _TTEntry(depth, _score_to_table(value, ply), bound, best_move),
        )
    return value


def _search_root(
    game: ChessGame,
    moves: Sequence[Move],
    depth: int,
    quiescence: bool,
    context: _SearchContext,
    config: EngineConfig,
) -> tuple[Move, int]:
    static = (
        _basic_evaluate(game.board) if config.evaluation_profile == "basic" else None
    )
    alpha = -INFINITY
    beta = INFINITY
    original_alpha = alpha
    original_beta = beta
    key: TranspositionKey | None = None
    hash_move: Move | None = None
    if context.table_limit:
        key = _table_key(game, quiescence, config)
        context.transposition_probes += 1
        entry = context.table.get(key)
        if entry is not None:
            context.transposition_hits += 1
            hash_move = entry.best_move

    ordered = _ordered_search_moves(
        moves, hash_move=hash_move, ply=0, color=game.turn, context=context
    )
    best_move = ordered[0]
    if game.turn == "w":
        best_score = -INFINITY
        for move in ordered:
            score = _search(
                game._after_generated(move),
                depth - 1,
                alpha,
                beta,
                1,
                quiescence,
                context,
                config,
                _static_after(static, move),
            )
            if score > best_score:
                best_score, best_move = score, move
            alpha = max(alpha, best_score)
            if alpha >= beta:
                context.beta_cutoffs += 1
                _record_quiet_cutoff(move, 0, game.turn, depth, context)
                break
    else:
        best_score = INFINITY
        for move in ordered:
            score = _search(
                game._after_generated(move),
                depth - 1,
                alpha,
                beta,
                1,
                quiescence,
                context,
                config,
                _static_after(static, move),
            )
            if score < best_score:
                best_score, best_move = score, move
            beta = min(beta, best_score)
            if alpha >= beta:
                context.beta_cutoffs += 1
                _record_quiet_cutoff(move, 0, game.turn, depth, context)
                break

    if key is not None:
        if best_score <= original_alpha:
            bound: TTBound = "upper"
        elif best_score >= original_beta:
            bound = "lower"
        else:
            bound = "exact"
        _store_entry(
            context,
            key,
            _TTEntry(depth, _score_to_table(best_score, 0), bound, best_move),
        )
    return best_move, best_score


class SearchEngine:
    """Stateful, non-thread-safe searcher with bounded cross-search reuse."""

    def __init__(self, config: EngineConfig = DEFAULT_ENGINE_CONFIG) -> None:
        self.config = config
        self._table: dict[TranspositionKey, _TTEntry] = {}
        self._killers: dict[int, list[Move]] = {}
        self._history: dict[HistoryKey, int] = {}
        self.last_stats: SearchStats | None = None

    def clear(self) -> None:
        """Discard cached search results and move-ordering heuristics."""
        self._table.clear()
        self._killers.clear()
        self._history.clear()
        self.last_stats = None

    def choose_move(
        self,
        game: ChessGame,
        depth: int | None = None,
        quiescence: bool | None = None,
        *,
        time_limit_ms: int | None = None,
    ) -> Move:
        search_depth = self.config.search_depth if depth is None else depth
        if type(search_depth) is not int or not 1 <= search_depth <= MAX_SEARCH_DEPTH:
            raise ValueError(f"Search depth must be between 1 and {MAX_SEARCH_DEPTH}")
        if time_limit_ms is not None and (
            type(time_limit_ms) is not int or time_limit_ms < 1
        ):
            raise ValueError(
                "Time limit must be a positive integer number of milliseconds"
            )
        use_quiescence = self.config.quiescence if quiescence is None else quiescence
        if type(use_quiescence) is not bool:
            raise ValueError("Quiescence must be a bool")

        started = perf_counter()
        deadline = started + time_limit_ms / 1000 if time_limit_ms is not None else None
        moves = game.legal_moves()
        if not moves:
            raise ValueError("No legal moves available")

        context = _SearchContext(
            self._table,
            self.config.transposition_table_size,
            deadline,
            self._killers,
            self._history,
        )
        best_move = moves[0]
        best_score: int | None = None
        completed_depth = 0
        for current_depth in range(1, search_depth + 1):
            try:
                candidate, score = _search_root(
                    game,
                    moves,
                    current_depth,
                    use_quiescence,
                    context,
                    self.config,
                )
            except _SearchStoppedError:
                break
            best_move, best_score = candidate, score
            completed_depth = current_depth
            if abs(score) >= MATE_SCORE - 1:
                break

        elapsed_ms = (perf_counter() - started) * 1000
        self.last_stats = SearchStats(
            completed_depth,
            context.nodes,
            context.quiescence_nodes,
            context.transposition_probes,
            context.transposition_hits,
            context.transposition_cutoffs,
            context.beta_cutoffs,
            elapsed_ms,
            best_score,
            best_move,
        )
        logger.info(
            "chose %s depth=%d nodes=%d qnodes=%d tt_hits=%d elapsed_ms=%.1f",
            move_notation(best_move),
            completed_depth,
            context.nodes,
            context.quiescence_nodes,
            context.transposition_hits,
            elapsed_ms,
        )
        return best_move


def choose_move(
    game: ChessGame,
    depth: int | None = None,
    quiescence: bool | None = None,
    *,
    config: EngineConfig = DEFAULT_ENGINE_CONFIG,
) -> Move:
    """Convenience search using a fresh table and configurable engine defaults."""
    return SearchEngine(config).choose_move(game, depth=depth, quiescence=quiescence)

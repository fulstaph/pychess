"""Legal chess play over immutable board snapshots.

Design overview
---------------
Legality is computed by a two-stage pipeline:

1. *Pseudo-legal generation* (``_piece_moves``, ``_castling_moves``)
   emits every geometrically legal move for a piece, ignoring whether
   the move exposes the moving king.
2. *King-safety filtering* (``_legal_candidates``) rejects candidates that
   leave the moving side in check. Outside check, only king moves, moves
   of pinned pieces, and en passant can expose the king, so only those are
   simulated with the pure ``Move.execute``; every move while in check is.

``GameState`` is a frozen dataclass: transitions never mutate state
in place, and ``ChessGame`` builds a fully-advanced state before
swapping the reference atomically. ``ChessGame.after`` therefore
returns a sibling game (created via ``object.__new__`` to bypass
``__init__`` validation) that shares no mutable state with its parent.

FEN round-trip policy: game status is never serialized — check,
checkmate and stalemate are always recomputed from the position on
parse — and the fullmove number is the only move counter persisted.

Public surface: ``ChessGame`` (state accessors, ``legal_moves``,
``make_move``, ``after``, ``castles``, status predicates, ``from_fen``,
``to_fen``) plus module-level ``to_fen`` / ``from_fen`` / ``print_board``
helpers. ``select_move`` resolves a ``str | Move`` to one legal candidate
without applying it; ``make_move`` and ``after`` accept ``str | Move``: strings are
matched against the current legal candidates as coordinate or SAN
notation; a ``ValueError`` is raised, state left untouched, when the
input matches no legal move.
"""

import re
from collections import Counter
from collections.abc import Iterator
from dataclasses import dataclass, replace
from enum import StrEnum

from .attacks import (
    DIAGONAL,
    DIAGONAL_RAYS,
    KING_STEPS,
    KNIGHT_STEPS,
    ORTHOGONAL,
    ORTHOGONAL_RAYS,
    find_king,
    is_in_check,
    is_square_attacked,
)
from .helpers import notation_to_coords, square_notation
from .log import get_logger
from .move import Move, MoveSpecial
from .piece import Board, Color, Piece, PieceType, Square

logger = get_logger("game")

type PositionKey = tuple[Board, Color, frozenset[str], Square | None]
type RepetitionContext = frozenset[tuple[PositionKey, int]]
type SearchPositionKey = tuple[
    Board,
    Color,
    frozenset[str],
    Square | None,
    int,
    RepetitionContext,
]


class GameStatus(StrEnum):
    ACTIVE = "active"
    CHECK = "check"
    CHECKMATE = "checkmate"
    STALEMATE = "stalemate"


@dataclass(frozen=True, slots=True)
class GameState:
    """Immutable snapshot of a game: board, turn, counters, rules state.

    ``frozen`` + ``slots`` makes instances cheap to share and guarantees
    transitions produce a new instance (see ``_advanced``) rather than
    mutating a live one.
    """

    board: Board
    turn: Color
    move_num: int
    status: GameStatus
    castling_rights: frozenset[str]
    en_passant: Square | None
    halfmove_clock: int = 0


def _other(color: Color) -> Color:
    """Return the opposite color."""
    return "b" if color == "w" else "w"


def _move_variants(
    piece: Piece, source: Square, target: Square, captured: Piece | None = None
) -> tuple[Move, ...]:
    """Expand (piece, from, to) into one or more concrete moves.

    A pawn reaching the final rank fans out into one Move per
    promotion piece; everything else is a single-element tuple.
    """
    if piece.type == PieceType.PAWN and target[0] == (0 if piece.color == "w" else 7):
        return tuple(
            Move(piece, source, target, captured, "promotion", kind)
            for kind in (
                PieceType.QUEEN,
                PieceType.ROOK,
                PieceType.BISHOP,
                PieceType.KNIGHT,
            )
        )
    return (Move(piece, source, target, captured),)


def _piece_moves(
    state: GameState, source: Square, piece: Piece, *, tactical: bool = False
) -> Iterator[Move]:
    """Pseudo-legal moves for one piece; king safety is checked later.

    Yields one Move per geometrically legal destination (promotions
    fanned out by ``_move_variants``). The result may still expose the
    moving king — ``_legal_candidates`` is responsible for filtering those
    out. With ``tactical`` set, only captures and promotions are yielded,
    in the same relative order, so quiet moves are never constructed.
    """
    board = state.board
    sq = board.squares  # raw grid; callers own bounds checks
    row, col = source
    if piece.type == PieceType.PAWN:
        step = -1 if piece.color == "w" else 1
        fr = row + step
        if 0 <= fr < 8 and sq[fr][col] is None:
            forward = (fr, col)
            if not tactical or fr == (0 if piece.color == "w" else 7):
                yield from _move_variants(piece, source, forward)
            two_r = row + 2 * step
            # Double push is only legal from the start rank (row 6 for
            # white, row 1 for black); `forward` is known empty above
            # and `two` must be empty as well.
            if (
                not tactical
                and row == (6 if piece.color == "w" else 1)
                and sq[two_r][col] is None
            ):
                yield Move(piece, source, (two_r, col))
        else:
            fr = row + step  # may be off-board; captures below still need it
        for dc in (-1, 1):
            tc = col + dc
            if not (0 <= fr < 8 and 0 <= tc < 8):
                continue
            target = (fr, tc)
            victim = sq[fr][tc]
            if victim and victim.color != piece.color and victim.type != PieceType.KING:
                yield from _move_variants(piece, source, target, victim)
            elif target == state.en_passant and victim is None:
                # En passant: `target` is the empty square the pawn jumps
                # to; the captured pawn lives on the origin's row in the
                # target's column.
                captured = sq[row][tc]
                if captured == Piece(PieceType.PAWN, _other(piece.color)):
                    yield Move(piece, source, target, captured, "en_passant")
        return
    if piece.type in (PieceType.KNIGHT, PieceType.KING):
        steps = KNIGHT_STEPS if piece.type == PieceType.KNIGHT else KING_STEPS
        for dr, dc in steps:
            tr, tc = row + dr, col + dc
            if 0 <= tr < 8 and 0 <= tc < 8:
                victim = sq[tr][tc]
                if victim is None:
                    if not tactical:
                        yield Move(piece, source, (tr, tc))
                elif victim.color != piece.color and victim.type != PieceType.KING:
                    yield Move(piece, source, (tr, tc), victim)
        if piece.type == PieceType.KING and not tactical:
            yield from _castling_moves(state, source, piece)
        return
    directions = (
        ORTHOGONAL
        if piece.type == PieceType.ROOK
        else DIAGONAL
        if piece.type == PieceType.BISHOP
        else ORTHOGONAL + DIAGONAL
    )
    for dr, dc in directions:
        r, c = row + dr, col + dc
        while 0 <= r < 8 and 0 <= c < 8:
            target = (r, c)
            victim = sq[r][c]
            if victim is not None:
                # The ray stops here: capture if the victim is a hostile
                # non-king, otherwise the piece merely blocks the ray.
                if victim.color != piece.color and victim.type != PieceType.KING:
                    yield Move(piece, source, target, victim)
                break
            if not tactical:
                yield Move(piece, source, target)
            r, c = r + dr, c + dc


def _castling_moves(state: GameState, source: Square, king: Piece) -> Iterator[Move]:
    """Castling candidates for a king still on its home square.

    A side is offered only when all of these hold:

    - the right is still present in ``state.castling_rights``;
    - the home rook is still on its corner square;
    - every square between king and rook is empty;
    - the king is not currently in check;
    - with the king temporarily vacating its home square, none of the
      squares it crosses (including the destination) are attacked.
    """
    row = 7 if king.color == "w" else 0
    if source != (row, 4) or is_in_check(state.board, king.color):
        return
    opponent = _other(king.color)
    castle_options: tuple[
        tuple[MoveSpecial, str, int, tuple[int, ...], tuple[int, ...], int], ...
    ] = (
        ("castle_k", "K" if king.color == "w" else "k", 7, (5, 6), (5, 6), 6),
        ("castle_q", "Q" if king.color == "w" else "q", 0, (1, 2, 3), (3, 2), 2),
    )
    for side, right, rook_col, empty_cols, transit_cols, destination in castle_options:
        if right not in state.castling_rights:
            continue
        if state.board._get_fast((row, rook_col)) != Piece(PieceType.ROOK, king.color):
            continue
        if any(state.board._get_fast((row, col)) is not None for col in empty_cols):
            continue
        # The king vacates its home square while crossing the transit squares.
        exposed = state.board._updated_validated({source: None})
        if any(
            is_square_attacked((row, col), opponent, exposed) for col in transit_cols
        ):
            continue
        yield Move(king, source, (row, destination), special=side)


def _pinned_squares(board: Board, color: Color, king: Square) -> frozenset[Square]:
    """Squares of ``color`` pieces that an enemy slider pins to its king.

    Walks the eight rays out of the king. A friendly piece is pinned when
    it is the first piece on its ray and the next piece behind it is an
    enemy slider moving along that ray's axis.
    """
    sq = board.squares
    pinned: set[Square] = set()
    king_row, king_col = king
    for rays, sliders in (
        (ORTHOGONAL_RAYS[king_row][king_col], (PieceType.ROOK, PieceType.QUEEN)),
        (DIAGONAL_RAYS[king_row][king_col], (PieceType.BISHOP, PieceType.QUEEN)),
    ):
        for ray in rays:
            blocker: Square | None = None
            for r, c in ray:
                piece = sq[r][c]
                if piece is None:
                    continue
                if blocker is not None:
                    if piece.color != color and piece.type in sliders:
                        pinned.add(blocker)
                    break
                if piece.color != color:
                    break
                blocker = (r, c)
    return frozenset(pinned)


def _king_square(board: Board, color: Color) -> Square:
    """Cached king square, raising ``ValueError`` when ``color`` has no king."""
    king = board.king_square(color)
    return king if king is not None else find_king(board, color)


def _legal_candidates(
    state: GameState,
    *,
    tactical: bool = False,
    in_check: bool | None = None,
    pinned: frozenset[Square] | None = None,
) -> Iterator[Move]:
    """Legal moves for ``state.turn`` in board-scan order.

    With ``tactical`` set, only captures and promotions are produced; other
    candidates are dropped before any king-safety work.

    King safety is established by simulation (``Move.execute`` plus
    ``is_in_check``) only where a move can expose the king: every move
    while in check, king moves, moves of pinned pieces, and en passant
    (which removes two pawns from one rank). All other moves are legal by
    construction. Castling is also accepted as generated, because
    ``_castling_moves`` already proved the king is not in check and that
    no square it crosses or lands on is attacked.
    """
    board = state.board
    turn = state.turn
    king = _king_square(board, turn)
    if in_check is None:
        in_check = is_square_attacked(king, _other(turn), board)
    if pinned is None:
        pinned = frozenset[Square]() if in_check else _pinned_squares(board, turn, king)
    for row, squares in enumerate(board.squares):
        for col, piece in enumerate(squares):
            if piece is None or piece.color != turn:
                continue
            source = (row, col)
            exposed = in_check or piece.type == PieceType.KING or source in pinned
            for move in _piece_moves(state, source, piece, tactical=tactical):
                special = move.special
                if (
                    special != "castle_k"
                    and special != "castle_q"
                    and (exposed or special == "en_passant")
                    and is_in_check(move.execute(board), turn)
                ):
                    continue
                yield move


def _legal_moves(
    state: GameState,
    *,
    in_check: bool | None = None,
    pinned: frozenset[Square] | None = None,
) -> tuple[Move, ...]:
    """All legal moves for ``state.turn``, in board-scan order."""
    moves = tuple(_legal_candidates(state, in_check=in_check, pinned=pinned))
    logger.debug("legal moves: %d for %s", len(moves), state.turn)
    return moves


def _legal_tactical_moves(
    state: GameState,
    *,
    in_check: bool | None = None,
    pinned: frozenset[Square] | None = None,
) -> tuple[Move, ...]:
    """Legal captures and promotions only, for quiescence search."""
    return tuple(
        _legal_candidates(state, tactical=True, in_check=in_check, pinned=pinned)
    )


def _position_key(state: GameState) -> PositionKey:
    """Build repetition identity with only legally usable en-passant rights."""
    en_passant = state.en_passant
    if en_passant is not None:
        target_row, target_col = en_passant
        source_row = target_row + (1 if state.turn == "w" else -1)
        for source_col in (target_col - 1, target_col + 1):
            if not (0 <= source_row < 8 and 0 <= source_col < 8):
                continue
            source = (source_row, source_col)
            piece = state.board.get(source)
            if (
                piece is None
                or piece.type != PieceType.PAWN
                or piece.color != state.turn
            ):
                continue
            if any(
                move.special == "en_passant"
                and not is_in_check(move.execute(state.board), state.turn)
                for move in _piece_moves(state, source, piece)
            ):
                break
        else:
            en_passant = None
    return state.board, state.turn, state.castling_rights, en_passant


def _has_legal_move(
    state: GameState,
    *,
    in_check: bool | None = None,
    pinned: frozenset[Square] | None = None,
) -> bool:
    """True if the side to move has at least one legal move.

    Short-circuits on the first legal candidate, avoiding the full
    ``_legal_moves`` allocation.
    """
    return (
        next(_legal_candidates(state, in_check=in_check, pinned=pinned), None)
        is not None
    )


def _is_insufficient_material(board: Board) -> bool:
    """True if the remaining pieces can never force checkmate.

    Any pawn, rook or queen rules the position out immediately; the
    mateless shapes that remain are K+K, K+minor vs K, and K+B vs K+B
    on same-colored bishops.
    """
    total = 0
    minors: list[tuple[Piece, int]] = []  # non-king pieces with square parity
    for r, row in enumerate(board.squares):
        for c, piece in enumerate(row):
            if piece is None:
                continue
            kind = piece.type
            if kind in (PieceType.PAWN, PieceType.ROOK, PieceType.QUEEN):
                return False
            total += 1
            if kind != PieceType.KING:
                minors.append((piece, (r + c) % 2))
    if total <= 3:
        return True
    if total == 4 and len(minors) == 2:
        (first, first_parity), (second, second_parity) = minors
        return (
            first.type == PieceType.BISHOP
            and second.type == PieceType.BISHOP
            and first.color != second.color
            and first_parity == second_parity
        )
    return False


def _status(state: GameState) -> GameStatus:
    """Classify the position's check state, not claimable draws."""
    check = is_in_check(state.board, state.turn)
    if not _has_legal_move(state):
        return GameStatus.CHECKMATE if check else GameStatus.STALEMATE
    return GameStatus.CHECK if check else GameStatus.ACTIVE


def _next_rights(rights: frozenset[str], move: Move) -> frozenset[str]:
    """Castling rights after ``move`` has been made."""
    kind = move.piece.type
    captured = move.captured_piece
    if not rights or (
        kind != PieceType.KING
        and kind != PieceType.ROOK
        and (captured is None or captured.type != PieceType.ROOK)
    ):
        return rights  # neither a king/rook move nor a rook capture
    removed: set[str] = set()
    if move.piece.type == PieceType.KING:
        removed.update("KQ" if move.piece.color == "w" else "kq")
    home_rooks: tuple[tuple[Color, int, tuple[str, str]], ...] = (
        ("w", 7, ("Q", "K")),
        ("b", 0, ("q", "k")),
    )
    for color, row, sides in home_rooks:
        for col, right in zip((0, 7), sides, strict=True):
            home = (row, col)
            if (
                move.piece.type == PieceType.ROOK
                and move.piece.color == color
                and move.from_square == home
            ) or (
                move.captured_piece == Piece(PieceType.ROOK, color)
                and move.to_square == home
            ):
                removed.add(right)
    return rights.difference(removed)


def _advanced(
    state: GameState, move: Move, *, recompute_status: bool = True
) -> GameState:
    """Build a state after a (already-legal) move; ``state`` untouched.

    Turn flips, the fullmove number increments only after Black's
    reply, en passant targets are set only on double pushes, and the
    halfmove clock resets on any pawn move or capture. Search can defer
    status calculation until it has generated the node's legal moves.
    """
    rights = _next_rights(state.castling_rights, move)
    en_passant: Square | None = None
    if (
        move.piece.type == PieceType.PAWN
        and abs(move.to_square[0] - move.from_square[0]) == 2
    ):
        # The en passant target is the square the pawn jumped over.
        en_passant = (
            (move.from_square[0] + move.to_square[0]) // 2,
            move.from_square[1],
        )
    if move.piece.type == PieceType.PAWN or move.is_capture():
        next_halfmove = 0
    else:
        next_halfmove = state.halfmove_clock + 1

    next_state = GameState(
        move.execute(state.board),
        _other(state.turn),
        state.move_num + (state.turn == "b"),
        GameStatus.ACTIVE,
        rights,
        en_passant,
        next_halfmove,
    )
    return (
        replace(next_state, status=_status(next_state))
        if recompute_status
        else next_state
    )


class ChessGame:
    """A game façade that commits only complete, legal state transitions.

    The game owns a single immutable ``GameState`` reference.
    ``make_move`` replaces it in place; ``after`` returns a sibling
    game built via ``object.__new__`` — bypassing ``__init__``
    validation, which would only re-check an already-legal position —
    so the sibling shares no mutable state with its parent. Position
    history is append-only, which is what lets threefold-repetition
    detection span sibling branches.
    """

    def __init__(
        self,
        board: Board | None = None,
        turn: Color = "w",
        castling_rights: frozenset[str] | None = None,
        en_passant: Square | None = None,
        halfmove_clock: int = 0,
        move_num: int = 1,
    ) -> None:
        """Validate inputs and build a game with a computed status.

        Both kings must exist; castling rights must be a subset of
        ``KQkq``; an en passant target must sit on the correct rank
        with the opposing pawn beside it; the halfmove clock must not
        be negative and the move number must be at least 1. All
        validation happens here — ``after`` deliberately skips it.
        """
        custom = board is not None
        board = board if custom else Board.from_notation()
        if not isinstance(board, Board) or turn not in ("w", "b"):
            raise ValueError("Invalid board or turn")
        find_king(board, "w")
        find_king(board, "b")
        rights = (
            castling_rights
            if castling_rights is not None
            else (frozenset() if custom else frozenset("KQkq"))
        )
        if not isinstance(rights, frozenset) or not rights <= frozenset("KQkq"):
            raise ValueError("Invalid castling rights")
        if en_passant is not None:
            board.get(en_passant)
            pawn_row = 3 if turn == "w" else 4
            if (
                en_passant[0] != (2 if turn == "w" else 5)
                or board.get(en_passant) is not None
                or board.get((pawn_row, en_passant[1]))
                != Piece(PieceType.PAWN, _other(turn))
            ):
                raise ValueError("Invalid en passant target")
        if halfmove_clock < 0:
            raise ValueError("Halfmove clock cannot be negative")
        if move_num < 1:
            raise ValueError("Move number must be at least 1")
        state = GameState(
            board, turn, move_num, GameStatus.ACTIVE, rights, en_passant, halfmove_clock
        )
        self._state = replace(state, status=_status(state))
        self._position_history: tuple[PositionKey, ...] = (_position_key(self._state),)
        self._legal_moves_cache: tuple[Move, ...] | None = None
        self._in_check: bool | None = None
        self._pinned_cache: frozenset[Square] | None = None
        self._status_current = True
        self._position_counts_cache: Counter[PositionKey] | None = None
        self._repetition_context_cache: RepetitionContext | None = None

    @property
    def board(self) -> Board:
        return self._state.board

    @property
    def turn(self) -> Color:
        return self._state.turn

    @property
    def status(self) -> GameStatus:
        """Current status, calculated lazily for search-created positions."""
        if not self._status_current:
            cached_moves = self._legal_moves_cache
            if cached_moves is None:
                status = _status(self._state)
            else:
                check = is_in_check(self.board, self.turn)
                if not cached_moves:
                    status = GameStatus.CHECKMATE if check else GameStatus.STALEMATE
                else:
                    status = GameStatus.CHECK if check else GameStatus.ACTIVE
            self._state = replace(self._state, status=status)
            self._status_current = True
        return self._state.status

    @property
    def move_num(self) -> int:
        return self._state.move_num

    @property
    def halfmove_clock(self) -> int:
        return self._state.halfmove_clock

    def legal_moves(self) -> tuple[Move, ...]:
        """Legal moves for the side to move; empty once the game is
        decided by checkmate or stalemate.  Cached after the first call
        so the search never regenerates the same list twice.
        """
        if self._legal_moves_cache is not None:
            return self._legal_moves_cache
        if self._status_current and self._state.status in (
            GameStatus.CHECKMATE,
            GameStatus.STALEMATE,
        ):
            self._legal_moves_cache = ()
            return ()
        result = _legal_moves(
            self._state, in_check=self.is_check(), pinned=self._pinned()
        )
        self._legal_moves_cache = result
        return result

    def _has_legal_move(self) -> bool:
        """Whether any legal move exists, without generating the full list."""
        if self._legal_moves_cache is not None:
            return bool(self._legal_moves_cache)
        return _has_legal_move(
            self._state, in_check=self.is_check(), pinned=self._pinned()
        )

    def _legal_tactical_moves(self) -> tuple[Move, ...]:
        """Legal captures and promotions; reuses the full list when cached."""
        if self._legal_moves_cache is not None:
            return tuple(
                move
                for move in self._legal_moves_cache
                if move.captured_piece is not None or move.special == "promotion"
            )
        return _legal_tactical_moves(
            self._state, in_check=self.is_check(), pinned=self._pinned()
        )

    def select_move(self, move: str | Move) -> Move:
        """Resolve a ``str | Move`` request to exactly one legal candidate.

        Strings go through the SAN/coordinate parser; ``Move`` objects
        must equal a candidate exactly. Raises ``ValueError`` when
        nothing matches; never mutates state.
        """
        candidates = self.legal_moves()
        if isinstance(move, str):
            selected = self._parse_move(move, candidates)
        elif isinstance(move, Move):
            selected = next(
                (candidate for candidate in candidates if candidate == move), None
            )
        else:
            selected = None
        if selected is None:
            raise ValueError(f"Illegal move: {move!r}")
        return selected

    def make_move(self, move: str | Move) -> Move:
        """Apply a legal move, given as SAN/coordinate string or a Move.

        Parse, select, advance, then swap atomically: the replacement
        state is fully built before ``self._state`` is reassigned, so an
        illegal input raises ``ValueError`` and leaves the state exactly
        as it was — the fail-fast contract callers rely on.
        """
        try:
            selected = self.select_move(move)
        except ValueError as exc:
            logger.warning("rejected move %r: %s", move, exc)
            raise
        logger.debug(
            "move %s %s -> %s",
            selected.piece.type.value,
            square_notation(*selected.from_square),
            square_notation(*selected.to_square),
        )
        self._state = _advanced(self._state, selected)
        self._legal_moves_cache = None
        self._in_check = None
        self._pinned_cache = None
        self._status_current = True
        pos_key = _position_key(self._state)
        self._position_history = (*self._position_history, pos_key)
        self._position_counts_cache = None
        self._repetition_context_cache = None
        logger.debug("status after move: %s", self._state.status.value)
        return selected

    def after(self, move: str | Move) -> ChessGame:
        """Return a sibling game with the move applied; this game is
        left unchanged.

        ``object.__new__`` skips ``__init__`` (which would re-validate
        an already-legal position); the sibling inherits the parent's
        position history plus its own new position key, which is what
        lets threefold-repetition detection span sibling branches.
        """
        selected = self.select_move(move)
        return self._after_selected(selected, recompute_status=True)

    def _after_generated(self, move: Move) -> ChessGame:
        """Advance a move already yielded by this position's legal generator.

        Search uses this private path to skip duplicate move selection and
        status probing. The child resolves its status lazily if a caller asks.
        """
        return self._after_selected(move, recompute_status=False)

    def _after_selected(self, selected: Move, *, recompute_status: bool) -> ChessGame:
        sibling = object.__new__(ChessGame)
        sibling._state = _advanced(
            self._state, selected, recompute_status=recompute_status
        )
        sibling._legal_moves_cache = None
        sibling._in_check = None
        sibling._pinned_cache = None
        sibling._status_current = recompute_status
        pos_key = _position_key(sibling._state)
        sibling._position_history = (*self._position_history, pos_key)
        sibling._position_counts_cache = None
        sibling._repetition_context_cache = None
        return sibling

    def _parse_move(self, text: str, candidates: tuple[Move, ...]) -> Move | None:
        """Match ``text`` against the legal candidates; ``None`` if not
        exactly one matches.

        Two accepted forms: coordinate notation (``e2e4``, ``e7e8q``
        with an optional ``=`` before the promotion piece) and SAN,
        where castling is additionally accepted case-insensitively
        (``o-o``). A trailing ``+`` or ``#`` is validated against the
        simulated result: ``#`` requires checkmate, ``+`` requires
        check but not mate.
        """
        suffix = text[-1] if text.endswith(("+", "#")) else ""
        body = text[:-1] if suffix else text
        coordinate = re.fullmatch(r"([a-h][1-8])([a-h][1-8])(?:=?([qrbnQRBN]))?", body)
        if coordinate:
            source, target = (
                notation_to_coords(coordinate[1]),
                notation_to_coords(coordinate[2]),
            )
            promotion = PieceType(coordinate[3].lower()) if coordinate[3] else None
            matches = [
                move
                for move in candidates
                if move.from_square == source
                and move.to_square == target
                and move.promotion_to == promotion
            ]
        else:
            matches = [
                move
                for move in candidates
                if self._san(move, candidates) == body
                or (
                    move.special.startswith("castle_")
                    and self._san(move, candidates).lower() == body.lower()
                )
            ]
        if len(matches) != 1:
            return None
        selected = matches[0]
        if suffix:
            outcome = _advanced(self._state, selected).status
            if (suffix == "#" and outcome != GameStatus.CHECKMATE) or (
                suffix == "+" and outcome != GameStatus.CHECK
            ):
                return None
        return selected

    def _san(self, move: Move, candidates: tuple[Move, ...]) -> str:
        """Render ``move`` in Standard Algebraic Notation.

        Pawn moves carry a file prefix only on captures; other pieces
        disambiguate among rival pieces reaching the same square by
        file first, then by rank, else by the full origin square.
        """
        if move.special == "castle_k":
            return "O-O"
        if move.special == "castle_q":
            return "O-O-O"
        destination = square_notation(*move.to_square)
        capture = "x" if move.is_capture() else ""
        promotion = f"={move.promotion_to.value.upper()}" if move.promotion_to else ""
        origin = square_notation(*move.from_square)
        if move.piece.type == PieceType.PAWN:
            return (origin[0] if capture else "") + capture + destination + promotion
        rivals = [
            other
            for other in candidates
            if other != move
            and other.piece.type == move.piece.type
            and other.to_square == move.to_square
        ]
        disambiguation = ""
        if rivals:
            if all(other.from_square[1] != move.from_square[1] for other in rivals):
                disambiguation = origin[0]
            elif all(other.from_square[0] != move.from_square[0] for other in rivals):
                disambiguation = origin[1]
            else:
                disambiguation = origin
        symbol = move.piece.type.value
        if not isinstance(symbol, str):
            raise TypeError("Piece type notation must be a string")
        return symbol.upper() + disambiguation + capture + destination

    def get_board(self) -> Board:
        return self.board

    def get_turn(self) -> Color:
        return self.turn

    def castles(self, player: Color, side: str) -> bool:
        """Whether the side may still castle on the given flank."""
        if player not in ("w", "b") or side not in ("kingside", "queenside"):
            raise ValueError("Invalid player or castling side")
        right = (
            ("K" if side == "kingside" else "Q")
            if player == "w"
            else ("k" if side == "kingside" else "q")
        )
        return right in self._state.castling_rights

    def is_check(self) -> bool:
        in_check = self._in_check
        if in_check is None:
            in_check = self._in_check = is_in_check(self.board, self.turn)
        return in_check

    def _pinned(self) -> frozenset[Square]:
        """Pinned friendly squares, memoised per position (empty when in check)."""
        pinned = self._pinned_cache
        if pinned is None:
            pinned = self._pinned_cache = (
                frozenset[Square]()
                if self.is_check()
                else _pinned_squares(
                    self.board, self.turn, _king_square(self.board, self.turn)
                )
            )
        return pinned

    def is_checkmate(self) -> bool:
        return self.status == GameStatus.CHECKMATE

    def is_stalemate(self) -> bool:
        return self.status == GameStatus.STALEMATE

    def is_fifty_moves(self) -> bool:
        return self._state.halfmove_clock >= 100

    def is_threefold_repetition(self) -> bool:
        pos_key = _position_key(self._state)
        return self._position_counts().get(pos_key, 0) >= 3

    def _position_counts(self) -> Counter[PositionKey]:
        if self._position_counts_cache is None:
            self._position_counts_cache = Counter(self._position_history)
        return self._position_counts_cache

    def _repetition_context(self) -> RepetitionContext:
        if self._repetition_context_cache is None:
            self._repetition_context_cache = frozenset(self._position_counts().items())
        return self._repetition_context_cache

    def _search_key(self) -> SearchPositionKey:
        state = self._state
        return (
            state.board,
            state.turn,
            state.castling_rights,
            state.en_passant,
            state.halfmove_clock,
            self._repetition_context(),
        )

    def is_insufficient_material(self) -> bool:
        return _is_insufficient_material(self.board)

    def draw_reason(self) -> str | None:
        """Why the game is drawn, or None. Checkmate is never a draw."""
        if self.is_checkmate():
            return None
        if self.is_stalemate():
            return "stalemate"
        if self.is_fifty_moves():
            return "fifty-move rule"
        if self.is_threefold_repetition():
            return "threefold repetition"
        if self.is_insufficient_material():
            return "insufficient material"
        return None

    def draw_announcement(self) -> str | None:
        """Player-facing draw text, or None when the game is not drawn."""
        reason = self.draw_reason()
        if reason is None:
            return None
        if reason == "stalemate":
            return "Stalemate! Draw."
        if reason == "fifty-move rule":
            return "Draw by the 50-move rule."
        return f"Draw by {reason}."

    def is_draw(self) -> bool:
        return self.draw_reason() is not None

    @classmethod
    def from_fen(cls, fen: str) -> ChessGame:
        """Create a ChessGame from a Forsyth-Edwards Notation (FEN) string.

        Field order: piece placement (ranks 8 to 1), active color,
        castling availability (letters ``KQkq``, ``-`` if none),
        en passant target square, halfmove clock, fullmove number.
        Fields after the first are optional and default to a fresh
        game.

        Raises:
            ValueError: on any malformed field (logged at WARNING).
        """
        try:
            tokens = fen.strip().split()
            if not tokens:
                raise ValueError("FEN string cannot be empty")
            board = Board.from_fen(tokens[0])
            turn: Color = "w"
            if len(tokens) > 1:
                if tokens[1] not in ("w", "b"):
                    raise ValueError(f"Invalid turn in FEN: {tokens[1]!r}")
                turn = "w" if tokens[1] == "w" else "b"

            castling_rights: frozenset[str] = frozenset()
            if len(tokens) > 2 and tokens[2] != "-":
                if not set(tokens[2]) <= set("KQkq"):
                    raise ValueError(f"Invalid castling rights in FEN: {tokens[2]!r}")
                castling_rights = frozenset(tokens[2])

            en_passant: Square | None = None
            if len(tokens) > 3 and tokens[3] != "-":
                try:
                    en_passant = notation_to_coords(tokens[3])
                except ValueError as exc:
                    raise ValueError(
                        f"Invalid en passant square in FEN: {tokens[3]!r}"
                    ) from exc

            halfmove = 0
            if len(tokens) > 4:
                try:
                    halfmove = int(tokens[4])
                    if halfmove < 0:
                        raise ValueError
                except ValueError:
                    raise ValueError(
                        f"Invalid halfmove clock in FEN: {tokens[4]!r}"
                    ) from None

            move_num = 1
            if len(tokens) > 5:
                try:
                    move_num = int(tokens[5])
                    if move_num < 1:
                        raise ValueError
                except ValueError:
                    raise ValueError(
                        f"Invalid fullmove number in FEN: {tokens[5]!r}"
                    ) from None

            game = cls(
                board=board,
                turn=turn,
                castling_rights=castling_rights,
                en_passant=en_passant,
                halfmove_clock=halfmove,
                move_num=move_num,
            )
        except ValueError as exc:
            # Public entry point: surface the offending input for debugging.
            logger.warning("FEN parse failed for %r: %s", fen, exc)
            raise
        logger.debug("parsed FEN %r", fen)
        return game

    def get_winner(self) -> Color | None:
        return _other(self.turn) if self.is_checkmate() else None

    def to_fen(self) -> str:
        return to_fen(self)


# Unicode glyphs: outlined symbols for white pieces, filled for black.
UNICODE_PIECES: dict[tuple[PieceType, Color], str] = {
    (PieceType.KING, "w"): "\u2654",
    (PieceType.QUEEN, "w"): "\u2655",
    (PieceType.ROOK, "w"): "\u2656",
    (PieceType.BISHOP, "w"): "\u2657",
    (PieceType.KNIGHT, "w"): "\u2658",
    (PieceType.PAWN, "w"): "\u2659",
    (PieceType.KING, "b"): "\u265a",
    (PieceType.QUEEN, "b"): "\u265b",
    (PieceType.ROOK, "b"): "\u265c",
    (PieceType.BISHOP, "b"): "\u265d",
    (PieceType.KNIGHT, "b"): "\u265e",
    (PieceType.PAWN, "b"): "\u265f",
}


def print_board(
    board: Board, unicode_pieces: bool = False, ansi_colors: bool = False
) -> None:
    """Print the board, ranks 8 to 1 top to bottom, plus an a-h header.

    Symbols default to uppercase (white) / lowercase (black) letters;
    ``unicode_pieces`` switches to Unicode chess glyphs and
    ``ansi_colors`` adds 256-color square backgrounds (250 light,
    240 dark).
    """
    for row, squares in enumerate(board.squares):
        cells: list[str] = []
        for col, piece in enumerate(squares):
            if unicode_pieces and piece is not None:
                symbol = UNICODE_PIECES[(piece.type, piece.color)]
            elif piece is not None:
                symbol = (
                    piece.type.value.upper() if piece.color == "w" else piece.type.value
                )
            else:
                symbol = "."

            if ansi_colors:
                is_light = (row + col) % 2 == 0
                bg = "\033[48;5;250m" if is_light else "\033[48;5;240m"
                fg = "\033[30m" if is_light else "\033[97m"
                cells.append(f"{bg}{fg} {symbol} \033[0m")
            else:
                cells.append(symbol)

        sep = "" if ansi_colors else " "
        print(f"{8 - row} " + sep.join(cells))
    if ansi_colors:
        print("   a  b  c  d  e  f  g  h")
    else:
        print("  a b c d e f g h")


def to_fen(state_or_game: GameState | ChessGame) -> str:
    """Serialize a game state or game to Forsyth-Edwards Notation (FEN).

    Field order mirrors ``from_fen``: placement (rank 8 first, runs of
    empty squares collapsed into a digit), active color, castling
    letters in fixed ``KQkq`` order (``-`` if none), en passant square,
    halfmove clock, fullmove number. Status is deliberately omitted —
    it is recomputed on parse.
    """
    state = (
        state_or_game._state if isinstance(state_or_game, ChessGame) else state_or_game
    )
    board = state.board
    rank_strings: list[str] = []
    for row in range(8):
        empty_count = 0
        rank_chars: list[str] = []
        for col in range(8):
            piece = board.squares[row][col]
            if piece is None:
                empty_count += 1
            else:
                if empty_count > 0:
                    rank_chars.append(str(empty_count))
                    empty_count = 0
                symbol = (
                    piece.type.value.upper() if piece.color == "w" else piece.type.value
                )
                rank_chars.append(symbol)
        if empty_count > 0:
            rank_chars.append(str(empty_count))
        rank_strings.append("".join(rank_chars))
    board_part = "/".join(rank_strings)
    castling_part = "".join(c for c in "KQkq" if c in state.castling_rights) or "-"
    ep_part = (
        square_notation(*state.en_passant) if state.en_passant is not None else "-"
    )
    return (
        f"{board_part} {state.turn} {castling_part} {ep_part} "
        f"{state.halfmove_clock} {state.move_num}"
    )


def from_fen(fen: str) -> ChessGame:
    """Create a ChessGame from a Forsyth-Edwards Notation (FEN) string.

    Module-level convenience delegating to ``ChessGame.from_fen``,
    which performs all validation and logging.
    """
    return ChessGame.from_fen(fen)

"""Legal chess play over immutable board snapshots.

Design overview
---------------
Legality is computed by a two-stage pipeline:

1. *Pseudo-legal generation* (``_piece_moves``, ``_castling_moves``)
   emits every geometrically legal move for a piece, ignoring whether
   the move exposes the moving king.
2. *Execute-and-reject* (``_legal_moves``) simulates each candidate
   with the pure ``Move.execute`` and rejects moves that leave the
   moving side in check.

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
helpers. ``make_move`` and ``after`` accept ``str | Move``: strings are
matched against the current legal candidates as coordinate or SAN
notation; a ``ValueError`` is raised, state left untouched, when the
input matches no legal move.
"""

import re
from collections.abc import Iterator
from dataclasses import dataclass, replace
from enum import StrEnum

from .attacks import (
    DIAGONAL,
    KING_STEPS,
    KNIGHT_STEPS,
    ORTHOGONAL,
    find_king,
    is_in_check,
    is_square_attacked,
    on_board,
)
from .helpers import notation_to_coords, square_notation
from .log import get_logger
from .move import Move, MoveSpecial
from .piece import Board, Color, Piece, PieceType, Square

logger = get_logger("game")


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


def _piece_moves(state: GameState, source: Square, piece: Piece) -> Iterator[Move]:
    """Pseudo-legal moves for one piece; king safety is checked later.

    Yields one Move per geometrically legal destination (promotions
    fanned out by ``_move_variants``). The result may still expose the
    moving king — ``_legal_moves`` is responsible for filtering those
    out.
    """
    board = state.board
    row, col = source
    if piece.type == PieceType.PAWN:
        step = -1 if piece.color == "w" else 1
        forward = (row + step, col)
        if on_board(*forward) and board.get(forward) is None:
            yield from _move_variants(piece, source, forward)
            two = (row + 2 * step, col)
            # Double push is only legal from the start rank (row 6 for
            # white, row 1 for black); `forward` is known empty above
            # and `two` must be empty as well.
            if row == (6 if piece.color == "w" else 1) and board.get(two) is None:
                yield Move(piece, source, two)
        for dc in (-1, 1):
            target = (row + step, col + dc)
            if not on_board(*target):
                continue
            victim = board.get(target)
            if victim and victim.color != piece.color and victim.type != PieceType.KING:
                yield from _move_variants(piece, source, target, victim)
            elif target == state.en_passant and victim is None:
                # En passant: `target` is the empty square the pawn jumps
                # to; the captured pawn lives on the origin's row in the
                # target's column.
                captured = board.get((row, col + dc))
                if captured == Piece(PieceType.PAWN, _other(piece.color)):
                    yield Move(piece, source, target, captured, "en_passant")
        return
    if piece.type in (PieceType.KNIGHT, PieceType.KING):
        steps = KNIGHT_STEPS if piece.type == PieceType.KNIGHT else KING_STEPS
        for dr, dc in steps:
            target = (row + dr, col + dc)
            if on_board(*target):
                victim = board.get(target)
                if victim is None or (
                    victim.color != piece.color and victim.type != PieceType.KING
                ):
                    yield Move(piece, source, target, victim)
        if piece.type == PieceType.KING:
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
        while on_board(r, c):
            target = (r, c)
            victim = board.get(target)
            if victim is not None:
                # The ray stops here: capture if the victim is a hostile
                # non-king, otherwise the piece merely blocks the ray.
                if victim.color != piece.color and victim.type != PieceType.KING:
                    yield Move(piece, source, target, victim)
                break
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
        if state.board.get((row, rook_col)) != Piece(PieceType.ROOK, king.color):
            continue
        if any(state.board.get((row, col)) is not None for col in empty_cols):
            continue
        # The king vacates its home square while crossing the transit squares.
        exposed = state.board.with_piece(source, None)
        if any(
            is_square_attacked((row, col), opponent, exposed) for col in transit_cols
        ):
            continue
        yield Move(king, source, (row, destination), special=side)


def _legal_moves(state: GameState) -> tuple[Move, ...]:
    """All legal moves for ``state.turn``, check-filtered.

    Pseudo-legal candidates are executed and rejected if the mover ends
    up in check. ``Move.execute`` is pure (returns a new board, never
    mutates), so simulating every candidate is cheap and needs no
    restore step.
    """
    moves: list[Move] = []
    for row, squares in enumerate(state.board.squares):
        for col, piece in enumerate(squares):
            if piece is None or piece.color != state.turn:
                continue
            moves.extend(
                move
                for move in _piece_moves(state, (row, col), piece)
                if not is_in_check(move.execute(state.board), state.turn)
            )
    logger.debug("legal moves: %d for %s", len(moves), state.turn)
    return tuple(moves)


def _is_insufficient_material(board: Board) -> bool:
    """True if the remaining pieces can never force checkmate.

    Any pawn, rook or queen rules the position out immediately; the
    mateless shapes that remain are K+K, K+minor vs K, and K+B vs K+B
    on same-colored bishops.
    """
    pieces: list[tuple[Piece, Square]] = []
    for r, row in enumerate(board.squares):
        for c, piece in enumerate(row):
            if piece is not None:
                pieces.append((piece, (r, c)))
    if any(
        p.type in (PieceType.PAWN, PieceType.ROOK, PieceType.QUEEN) for p, _ in pieces
    ):
        return False
    if len(pieces) <= 2:
        return True
    if len(pieces) == 3:
        return True
    if len(pieces) == 4:
        minors = [(p, sq) for p, sq in pieces if p.type != PieceType.KING]
        if (
            len(minors) == 2
            and minors[0][0].type == PieceType.BISHOP
            and minors[1][0].type == PieceType.BISHOP
            and minors[0][0].color != minors[1][0].color
        ):
            sq_color1 = (minors[0][1][0] + minors[0][1][1]) % 2
            sq_color2 = (minors[1][1][0] + minors[1][1][1]) % 2
            return sq_color1 == sq_color2
    return False


def _status(state: GameState) -> GameStatus:
    """Classify a position; the detection order is load-bearing.

    No-legal-move positions are terminal, so checkmate/stalemate is
    decided first; only then do the fifty-move and insufficient-
    material draws apply; otherwise check beats active.
    """
    check = is_in_check(state.board, state.turn)
    if not _legal_moves(state):
        return GameStatus.CHECKMATE if check else GameStatus.STALEMATE
    if state.halfmove_clock >= 100 or _is_insufficient_material(state.board):
        return GameStatus.STALEMATE
    return GameStatus.CHECK if check else GameStatus.ACTIVE


def _next_rights(rights: frozenset[str], move: Move) -> frozenset[str]:
    """Castling rights after ``move`` has been made.

    - A king move drops both of that side's rights.
    - A home rook (a1/h1 for white, a8/h8 for black) moving off its
      corner, or being captured on that corner, drops that side's
      right.
    """
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


def _advanced(state: GameState, move: Move) -> GameState:
    """Build the state after a (already-legal) move; ``state`` untouched.

    Turn flips, the fullmove number increments only after Black's
    reply, en passant targets are set only on double pushes, the
    halfmove clock resets on any pawn move or capture, and status is
    recomputed from scratch.
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
    return replace(next_state, status=_status(next_state))


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
        self._position_history: tuple[
            tuple[Board, Color, frozenset[str], Square | None], ...
        ] = (
            (
                self._state.board,
                self._state.turn,
                self._state.castling_rights,
                self._state.en_passant,
            ),
        )

    @property
    def board(self) -> Board:
        return self._state.board

    @property
    def turn(self) -> Color:
        return self._state.turn

    @property
    def status(self) -> GameStatus:
        return self._state.status

    @property
    def move_num(self) -> int:
        return self._state.move_num

    @property
    def halfmove_clock(self) -> int:
        return self._state.halfmove_clock

    def legal_moves(self) -> tuple[Move, ...]:
        """Legal moves for the side to move; empty once the game is
        decided by checkmate or stalemate.
        """
        if self.status in (GameStatus.CHECKMATE, GameStatus.STALEMATE):
            return ()
        return _legal_moves(self._state)

    def _select_move(self, move: str | Move) -> Move:
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
        state (including status and the threefold-repetition override)
        is fully built before ``self._state`` is reassigned, so an
        illegal input raises ``ValueError`` and leaves the state
        exactly as it was — the fail-fast contract callers rely on.
        """
        try:
            selected = self._select_move(move)
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
        pos_key = (
            self._state.board,
            self._state.turn,
            self._state.castling_rights,
            self._state.en_passant,
        )
        self._position_history = (*self._position_history, pos_key)
        if (
            self._position_history.count(pos_key) >= 3
            and self._state.status != GameStatus.CHECKMATE
        ):
            self._state = replace(self._state, status=GameStatus.STALEMATE)
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
        selected = self._select_move(move)
        sibling = object.__new__(ChessGame)
        sibling._state = _advanced(self._state, selected)
        pos_key = (
            sibling._state.board,
            sibling._state.turn,
            sibling._state.castling_rights,
            sibling._state.en_passant,
        )
        sibling._position_history = (*self._position_history, pos_key)
        if (
            sibling._position_history.count(pos_key) >= 3
            and sibling._state.status != GameStatus.CHECKMATE
        ):
            sibling._state = replace(sibling._state, status=GameStatus.STALEMATE)
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
        return is_in_check(self.board, self.turn)

    def is_checkmate(self) -> bool:
        return self.status == GameStatus.CHECKMATE

    def is_stalemate(self) -> bool:
        return self.status == GameStatus.STALEMATE

    def is_fifty_moves(self) -> bool:
        return self._state.halfmove_clock >= 100

    def is_threefold_repetition(self) -> bool:
        pos_key = (
            self._state.board,
            self._state.turn,
            self._state.castling_rights,
            self._state.en_passant,
        )
        return self._position_history.count(pos_key) >= 3

    def is_insufficient_material(self) -> bool:
        return _is_insufficient_material(self.board)

    def is_draw(self) -> bool:
        return (
            self.is_stalemate()
            or self.is_fifty_moves()
            or self.is_threefold_repetition()
            or self.is_insufficient_material()
        )

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

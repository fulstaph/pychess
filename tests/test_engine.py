import pytest

from chess import (
    Board,
    ChessGame,
    GameStatus,
    Move,
    Piece,
    PieceType,
    PlayerColor,
    notation_to_coords,
)
from chess.engine import (
    DEFAULT_ENGINE_CONFIG,
    EngineConfig,
    SearchEngine,
    choose_move,
    evaluate,
    move_notation,
    mvv_lva_score,
    order_moves,
)
from chess.search import (
    BISHOP_PAIR_BONUS,
    DOUBLED_PAWN_PENALTY,
    ISOLATED_PAWN_PENALTY,
    MATE_SCORE,
    PASSED_PAWN_BONUSES,
    _basic_delta,
    _ordered_search_moves,
    _quiescence,
    _score_from_table,
    _score_to_table,
    _SearchContext,
)


def sq(name: str) -> tuple[int, int]:
    return notation_to_coords(name)


def test_start_position_engine_returns_legal_move_and_preserves_game():
    game = ChessGame()
    board_before = game.board
    turn_before = game.turn
    move = choose_move(game)
    assert move in game.legal_moves()
    assert game.board == board_before
    assert game.turn == turn_before


def test_black_reply_to_e4_is_legal():
    game = ChessGame()
    game.make_move("e4")
    move = choose_move(game)
    assert move in game.legal_moves()
    assert game.turn == "b"


def test_immediate_checkmate_outranks_pawn_capture():
    # White: Ke1, Pf3, Pg4, Pd4 (hanging pawn)
    # Black: Ke8, Qd8, Pe5
    # Black can play Qh4# (mate) or exd4 (pawn capture)
    board = (
        Board.from_notation()
        .with_piece(sq("f2"), None)
        .with_piece(sq("f3"), Piece(PieceType.PAWN, "w"))
        .with_piece(sq("g2"), None)
        .with_piece(sq("g4"), Piece(PieceType.PAWN, "w"))
        .with_piece(sq("e7"), None)
        .with_piece(sq("e5"), Piece(PieceType.PAWN, "b"))
        .with_piece(sq("d4"), Piece(PieceType.PAWN, "w"))
    )
    game = ChessGame(board, turn="b")
    move = choose_move(game)
    assert game.after(move).is_checkmate()


def test_hanging_queen_capture_outranks_quiet_move():
    # White queen on e4 is hanging, attacked by black pawn on d5
    board = (
        Board()
        .with_piece(sq("h1"), Piece(PieceType.KING, "w"))
        .with_piece(sq("e4"), Piece(PieceType.QUEEN, "w"))
        .with_piece(sq("h8"), Piece(PieceType.KING, "b"))
        .with_piece(sq("d5"), Piece(PieceType.PAWN, "b"))
        .with_piece(sq("a7"), Piece(PieceType.PAWN, "b"))
    )
    game = ChessGame(board, turn="b")
    move = choose_move(game)
    assert move.is_capture()
    assert move.captured_piece == Piece(PieceType.QUEEN, "w")


def test_choose_move_validates_depth_and_empty_legal_moves():
    game = ChessGame()
    with pytest.raises(ValueError, match="depth"):
        choose_move(game, depth=0)

    # Checkmate position (Fool's mate) has no legal moves
    for m in ("f3", "e5", "g4", "Qh4#"):
        game.make_move(m)
    assert game.status == GameStatus.CHECKMATE
    with pytest.raises(ValueError, match="No legal moves"):
        choose_move(game)


def test_evaluate_symmetry_and_values():
    initial_board = Board.from_notation()
    assert evaluate(initial_board) == 0

    # Board with white pawn in center has positive score
    board_with_d4 = initial_board.with_piece(sq("d4"), Piece(PieceType.PAWN, "w"))
    assert evaluate(board_with_d4) > 0


def test_basic_delta_matches_full_reevaluation_across_special_moves():
    fens = (
        "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1",
        "r3k2r/Pppp1ppp/1b3nbN/nP6/BBP1P3/q4N2/Pp1P2PP/R2Q1RK1 w kq - 0 1",
        "rnbq1k1r/pp1Pbppp/2p5/8/2B5/8/PPP1NnPP/RNBQK2R w KQ - 1 8",
        "4k3/8/8/8/3pP3/8/8/4K2R b K e3 0 1",
    )
    seen = set()

    def walk(game, depth):
        base = evaluate(game.board)
        for move in game.legal_moves():
            child = game.after(move)
            assert evaluate(child.board) - base == _basic_delta(move)
            seen.add(move.special)
            if depth:
                walk(child, depth - 1)

    for fen in fens:
        walk(ChessGame.from_fen(fen), 2)
    assert seen == {"none", "castle_k", "castle_q", "en_passant", "promotion"}


def test_move_notation_with_promotion():
    move = Move(
        Piece(PieceType.PAWN, "w"),
        sq("a7"),
        sq("a8"),
        special="promotion",
        promotion_to=PieceType.QUEEN,
    )
    assert move_notation(move) == "a7a8=Q"


def test_en_passant_suffix_parsing_regression():
    # Position where pawn move e2-e4+ gives check and only defense is en passant d4xe3
    board = (
        Board()
        .with_piece(sq("a1"), Piece(PieceType.KING, "w"))
        .with_piece(sq("e2"), Piece(PieceType.PAWN, "w"))
        .with_piece(sq("d5"), Piece(PieceType.KING, "b"))
        .with_piece(sq("d4"), Piece(PieceType.PAWN, "b"))
        .with_piece(sq("c1"), Piece(PieceType.ROOK, "w"))
        .with_piece(sq("e8"), Piece(PieceType.ROOK, "w"))
        .with_piece(sq("a6"), Piece(PieceType.ROOK, "w"))
    )
    game = ChessGame(board, turn="w")
    # Must accept e4+ because it gives check
    game.make_move("e4+")
    assert game.board.get(sq("e4")) == Piece(PieceType.PAWN, "w")

    # Must reject e4# because it is not checkmate (en passant capture is legal)
    game_for_mate_test = ChessGame(board, turn="w")
    with pytest.raises(ValueError):
        game_for_mate_test.make_move("e4#")


def test_board_equality_and_hash():
    b1 = Board.from_notation()
    b2 = Board.from_notation()
    assert b1 == b2
    assert hash(b1) == hash(b2)
    assert b1 != "not a board"

    b3 = b1.with_piece(sq("e4"), Piece(PieceType.PAWN, "w"))
    assert b1 != b3
    assert hash(b1) != hash(b3)


def test_player_color_str_enum_compatibility():
    assert PlayerColor.WHITE == "w"
    assert PlayerColor.BLACK == "b"
    p = Piece(PieceType.PAWN, PlayerColor.WHITE)
    assert p.is_white()
    assert not p.is_black()
    assert p.color == "w"
    game = ChessGame(turn=PlayerColor.BLACK)
    assert game.turn == "b"


def test_full_piece_tables_and_symmetry():
    initial = Board.from_notation()
    assert evaluate(initial) == 0
    b_c4 = initial.with_piece(sq("c4"), Piece(PieceType.BISHOP, "w"))
    assert evaluate(b_c4) > evaluate(initial.with_piece(sq("c4"), None))


def test_mvv_lva_and_move_ordering():
    pawn_capture_q = Move(
        Piece(PieceType.PAWN, "w"),
        sq("d4"),
        sq("e5"),
        captured_piece=Piece(PieceType.QUEEN, "b"),
    )
    queen_capture_p = Move(
        Piece(PieceType.QUEEN, "w"),
        sq("d4"),
        sq("e5"),
        captured_piece=Piece(PieceType.PAWN, "b"),
    )
    quiet_move = Move(Piece(PieceType.KNIGHT, "w"), sq("b1"), sq("c3"))

    assert mvv_lva_score(pawn_capture_q) > mvv_lva_score(queen_capture_p)
    assert mvv_lva_score(queen_capture_p) > mvv_lva_score(quiet_move)

    ordered = order_moves([quiet_move, queen_capture_p, pawn_capture_q])
    assert ordered[0] == pawn_capture_q
    assert ordered[1] == queen_capture_p
    assert ordered[2] == quiet_move


def test_search_move_ordering_preserves_tactical_priority():
    hash_move = Move(Piece(PieceType.KNIGHT, "w"), sq("b1"), sq("c3"))
    capture = Move(
        Piece(PieceType.PAWN, "w"),
        sq("d4"),
        sq("e5"),
        Piece(PieceType.QUEEN, "b"),
    )
    killer = Move(Piece(PieceType.BISHOP, "w"), sq("c1"), sq("g5"))
    history = Move(Piece(PieceType.ROOK, "w"), sq("a1"), sq("a2"))
    context = _SearchContext(
        {},
        0,
        None,
        {0: [killer]},
        {("w", history.from_square, history.to_square): 1_000_000},
    )

    ordered = _ordered_search_moves(
        [history, killer, capture, hash_move],
        hash_move=hash_move,
        ply=0,
        color="w",
        context=context,
    )
    assert ordered == [hash_move, capture, killer, history]


def test_choose_move_with_quiescence():
    fen = "r1bqk2r/pppp1ppp/2n5/4p3/2B1n3/5N2/PPPP1PPP/RNBQK2R w KQkq - 0 5"
    game = ChessGame.from_fen(fen)
    move_q = choose_move(game, depth=2, quiescence=True)
    assert isinstance(move_q, Move)
    assert move_q in game.legal_moves()


def test_quiescence_does_not_stand_pat_in_check():
    # Rook on a1 skewers Ke1 and Qh1. Every evasion loses the queen.
    game = ChessGame.from_fen("4k3/8/8/8/8/8/8/r3K2Q w - - 0 1")
    assert game.is_check()
    score = _quiescence(game, -1_000_000, 1_000_000)
    assert score < 0
    assert score < evaluate(game.board)


def test_quiescence_mate_score_uses_ply():
    game = ChessGame()
    for text in ("f3", "e5", "g4", "Qh4"):
        game.make_move(text)
    assert game.is_checkmate()
    assert _quiescence(game, -1_000_000, 1_000_000, ply=4) == -100000 + 4


def test_engine_config_controls_default_and_per_search_depth():
    assert DEFAULT_ENGINE_CONFIG.search_depth == 3
    assert DEFAULT_ENGINE_CONFIG.quiescence is True
    assert DEFAULT_ENGINE_CONFIG.evaluation_profile == "basic"
    for depth in (0, 9, True):
        with pytest.raises(ValueError, match="depth"):
            EngineConfig(search_depth=depth)
    for options in (
        {"quiescence": 1},
        {"quiescence_depth": -1},
        {"transposition_table_size": -1},
        {"evaluation_profile": "unknown"},
    ):
        with pytest.raises(ValueError):
            EngineConfig(**options)

    game = ChessGame()
    engine = SearchEngine(
        EngineConfig(search_depth=1, quiescence=False, transposition_table_size=0)
    )
    first = engine.choose_move(game)
    assert first in game.legal_moves()
    assert engine.last_stats is not None
    assert engine.last_stats.completed_depth == 1
    assert engine.last_stats.quiescence_nodes == 0

    engine.choose_move(game, depth=2)
    assert engine.last_stats is not None
    assert engine.last_stats.completed_depth == 2
    with pytest.raises(ValueError, match="depth"):
        engine.choose_move(game, depth=9)


def test_quiescence_is_on_by_default_and_can_be_disabled():
    game = ChessGame()
    engine = SearchEngine(EngineConfig(search_depth=1, transposition_table_size=0))
    engine.choose_move(game)
    assert engine.last_stats is not None
    assert engine.last_stats.quiescence_nodes > 0

    engine.choose_move(game, quiescence=False)
    assert engine.last_stats is not None
    assert engine.last_stats.quiescence_nodes == 0


def test_transposition_table_reuses_bounds_without_changing_score():
    game = ChessGame()
    config = EngineConfig(search_depth=3, quiescence=False)
    cached = SearchEngine(config)
    cached.choose_move(game)
    cached.choose_move(game)
    cached_stats = cached.last_stats
    assert cached_stats is not None
    assert cached_stats.transposition_hits > 0
    assert cached_stats.transposition_cutoffs > 0

    uncached = SearchEngine(
        EngineConfig(
            search_depth=3,
            quiescence=False,
            transposition_table_size=0,
        )
    )
    uncached.choose_move(game)
    uncached_stats = uncached.last_stats
    assert uncached_stats is not None
    assert cached_stats.score == uncached_stats.score


def test_search_key_includes_draw_history_and_fen_rule_fields():
    repeated = ChessGame()
    for _ in range(2):
        for move in ("Nf3", "Nf6", "Ng1", "Ng8"):
            repeated.make_move(move)
    replayed_from_fen = ChessGame.from_fen(repeated.to_fen())
    assert repeated.to_fen().split()[:5] == replayed_from_fen.to_fen().split()[:5]
    assert repeated.is_threefold_repetition()
    assert not replayed_from_fen.is_threefold_repetition()
    assert repeated._search_key() != replayed_from_fen._search_key()

    start_fen = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
    no_rights = ChessGame.from_fen(start_fen.replace("KQkq", "-"))
    assert ChessGame().board == no_rights.board
    assert ChessGame()._search_key() != no_rights._search_key()

    ep_fen = "4k3/8/8/3pP3/8/8/8/4K3 w - d6 0 2"
    with_ep = ChessGame.from_fen(ep_fen)
    without_ep = ChessGame.from_fen(ep_fen.replace("d6", "-"))
    assert with_ep.board == without_ep.board
    assert with_ep._search_key() != without_ep._search_key()

    high_clock = ChessGame.from_fen(start_fen.replace(" - 0 1", " - 99 1"))
    assert ChessGame()._search_key() != high_clock._search_key()


def test_mate_scores_keep_distance_when_cached_at_another_ply():
    for score in (MATE_SCORE - 7, -MATE_SCORE + 7):
        cached = _score_to_table(score, 4)
        restored = _score_from_table(cached, 9)
        expected = score - 5 if score > 0 else score + 5
        assert restored == expected


def test_positional_evaluation_is_color_mirror_symmetric_and_scores_bishop_pair():
    single = (
        Board()
        .with_piece(sq("e1"), Piece(PieceType.KING, "w"))
        .with_piece(sq("e8"), Piece(PieceType.KING, "b"))
        .with_piece(sq("a1"), Piece(PieceType.BISHOP, "w"))
    )
    pair = single.with_piece(sq("b1"), Piece(PieceType.BISHOP, "w"))
    basic_delta = evaluate(pair, "basic") - evaluate(single, "basic")
    positional_delta = evaluate(pair, "positional") - evaluate(single, "positional")
    assert positional_delta - basic_delta == BISHOP_PAIR_BONUS

    position = pair.with_piece(sq("c4"), Piece(PieceType.PAWN, "w")).with_piece(
        sq("h6"), Piece(PieceType.PAWN, "b")
    )
    mirrored = Board()
    for row, squares in enumerate(position.squares):
        for col, piece in enumerate(squares):
            if piece is not None:
                color = "b" if piece.color == "w" else "w"
                mirrored = mirrored.with_piece((7 - row, col), Piece(piece.type, color))
    assert evaluate(mirrored, "positional") == -evaluate(position, "positional")


def test_positional_evaluation_scores_pawn_structure_terms():
    kings = (
        Board()
        .with_piece(sq("e1"), Piece(PieceType.KING, "w"))
        .with_piece(sq("e8"), Piece(PieceType.KING, "b"))
    )

    separated = kings.with_piece(sq("a4"), Piece(PieceType.PAWN, "w")).with_piece(
        sq("c4"), Piece(PieceType.PAWN, "w")
    )
    connected = kings.with_piece(sq("a4"), Piece(PieceType.PAWN, "w")).with_piece(
        sq("b4"), Piece(PieceType.PAWN, "w")
    )
    assert (
        evaluate(connected, "positional") - evaluate(separated, "positional")
        == 2 * ISOLATED_PAWN_PENALTY
    )

    doubled = kings.with_piece(sq("a4"), Piece(PieceType.PAWN, "w")).with_piece(
        sq("a3"), Piece(PieceType.PAWN, "w")
    )
    spread = kings.with_piece(sq("a4"), Piece(PieceType.PAWN, "w")).with_piece(
        sq("b3"), Piece(PieceType.PAWN, "w")
    )
    assert (
        evaluate(spread, "positional") - evaluate(doubled, "positional")
        == 2 * ISOLATED_PAWN_PENALTY + 2 * DOUBLED_PAWN_PENALTY
    )

    passer = kings.with_piece(sq("a4"), Piece(PieceType.PAWN, "w"))
    passed_and_isolated = evaluate(passer, "positional") - evaluate(passer, "basic")
    assert passed_and_isolated == PASSED_PAWN_BONUSES[2] - ISOLATED_PAWN_PENALTY

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
    choose_move,
    evaluate,
    move_notation,
    mvv_lva_score,
    order_moves,
)


def sq(name: str) -> tuple[int, int]:
    return notation_to_coords(name)


def test_start_position_engine_chooses_d2d4_and_preserves_game():
    game = ChessGame()
    board_before = game.board
    turn_before = game.turn
    move = choose_move(game)
    assert move_notation(move) == "d2d4"
    assert game.board == board_before
    assert game.turn == turn_before


def test_black_reply_to_e4_chooses_e7e5():
    game = ChessGame()
    game.make_move("e4")
    move = choose_move(game)
    assert move_notation(move) == "e7e5"


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
    assert move_notation(move) == "d8h4"


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
    assert move_notation(move) == "d5e4"


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


def test_move_notation_with_promotion():
    move = choose_move(
        ChessGame(
            Board()
            .with_piece(sq("h1"), Piece(PieceType.KING, "w"))
            .with_piece(sq("h8"), Piece(PieceType.KING, "b"))
            .with_piece(sq("a7"), Piece(PieceType.PAWN, "w")),
            turn="w",
        )
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


def test_choose_move_with_quiescence():
    fen = "r1bqk2r/pppp1ppp/2n5/4p3/2B1n3/5N2/PPPP1PPP/RNBQK2R w KQkq - 0 5"
    game = ChessGame.from_fen(fen)
    move_q = choose_move(game, depth=2, quiescence=True)
    assert isinstance(move_q, Move)
    assert move_q in game.legal_moves()

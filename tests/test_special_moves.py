import pytest

from chess import (
    Board,
    ChessGame,
    GameStatus,
    Piece,
    PieceType,
    from_fen,
    notation_to_coords,
)
from chess.attacks import is_in_check, is_square_attacked


def square(text):
    return notation_to_coords(text)


def position(*pieces):
    board = Board()
    for text, kind, color in pieces:
        board = board.with_piece(square(text), Piece(kind, color))
    return board


def test_long_range_attack_stops_at_first_piece():
    board = position(
        ("e1", PieceType.KING, "w"),
        ("a8", PieceType.KING, "b"),
        ("e8", PieceType.ROOK, "b"),
    )
    assert is_square_attacked(square("e1"), "b", board)
    assert is_in_check(board, "w")
    assert not is_in_check(
        board.with_piece(square("e4"), Piece(PieceType.PAWN, "w")), "w"
    )


def test_castling_moves_both_rooks_and_retains_other_player_rights():
    board = position(
        ("e1", PieceType.KING, "w"),
        ("a1", PieceType.ROOK, "w"),
        ("h1", PieceType.ROOK, "w"),
        ("e8", PieceType.KING, "b"),
        ("a8", PieceType.ROOK, "b"),
        ("h8", PieceType.ROOK, "b"),
    )
    for side, destination, rook_destination in (
        ("kingside", "g1", "f1"),
        ("queenside", "c1", "d1"),
    ):
        game = ChessGame(board, castling_rights=frozenset("KQkq"))
        move = next(
            move
            for move in game.legal_moves()
            if move.piece.type == PieceType.KING
            and move.to_square == square(destination)
        )
        game.make_move(move)
        assert game.board.get(square(destination)) == Piece(PieceType.KING, "w")
        assert game.board.get(square(rook_destination)) == Piece(PieceType.ROOK, "w")
        assert not game.castles("w", side) and game.castles("b", side)


def test_en_passant_removes_pawn_and_expires():
    game = ChessGame()
    for text in ("e2e4", "a7a6", "e4e5", "d7d5"):
        game.make_move(text)
    game.make_move("e5d6")
    assert game.board.get(square("d5")) is None
    assert game.board.get(square("d6")) == Piece(PieceType.PAWN, "w")
    game = ChessGame()
    for text in ("e2e4", "a7a6", "e4e5", "d7d5", "g1f3", "a6a5"):
        game.make_move(text)
    assert not any(move.special == "en_passant" for move in game.legal_moves())


def test_promotion_requires_choice_and_stalemate_is_draw():
    board = position(
        ("h1", PieceType.KING, "w"),
        ("h8", PieceType.KING, "b"),
        ("a7", PieceType.PAWN, "w"),
    )
    game = ChessGame(board)
    with pytest.raises(ValueError):
        game.make_move("a7a8")
    game.make_move("a7a8=Q")
    assert game.board.get(square("a8")) == Piece(PieceType.QUEEN, "w")
    stalemate = position(
        ("a8", PieceType.KING, "b"),
        ("c6", PieceType.KING, "w"),
        ("b6", PieceType.QUEEN, "w"),
    )
    game = ChessGame(stalemate, turn="b")
    assert game.status == GameStatus.STALEMATE and game.is_draw()


def test_board_from_fen_valid_and_invalid():
    init_fen = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR"
    board = Board.from_fen(init_fen)
    assert board == Board.from_notation()

    # Trailing fields stripped cleanly
    board_full = Board.from_fen(
        "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
    )
    assert board_full == Board.from_notation()

    # Invalid rank count
    with pytest.raises(ValueError, match="must contain 8 ranks"):
        Board.from_fen("8/8/8")

    # Invalid empty square digit
    with pytest.raises(ValueError, match="Invalid empty square count"):
        Board.from_fen("9/8/8/8/8/8/8/8")

    # Invalid piece character
    with pytest.raises(ValueError, match="Invalid piece character"):
        Board.from_fen("x7/8/8/8/8/8/8/8")

    # Wrong rank length (e.g., 7 squares)
    with pytest.raises(ValueError, match="does not contain 8 squares"):
        Board.from_fen("7/8/8/8/8/8/8/8")


def test_chessgame_from_fen_roundtrip_and_validation():
    start_fen = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
    game = ChessGame.from_fen(start_fen)
    assert game.to_fen() == start_fen
    assert from_fen(start_fen).to_fen() == start_fen

    custom_fen = "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq e3 5 12"
    game_custom = ChessGame.from_fen(custom_fen)
    assert game_custom.turn == "b"
    assert game_custom.castles("w", "kingside")
    assert game_custom.halfmove_clock == 5
    assert game_custom.move_num == 12
    assert game_custom.to_fen() == custom_fen

    # Invalid inputs
    with pytest.raises(ValueError, match="cannot be empty"):
        ChessGame.from_fen("   ")
    with pytest.raises(ValueError, match="Invalid turn in FEN"):
        ChessGame.from_fen("8/8/8/8/8/8/8/4K2k x - - 0 1")
    with pytest.raises(ValueError, match="Invalid castling rights in FEN"):
        ChessGame.from_fen("8/8/8/8/8/8/8/4K2k w XYZ - 0 1")
    with pytest.raises(ValueError, match="Invalid en passant square in FEN"):
        ChessGame.from_fen("8/8/8/8/8/8/8/4K2k w - e9 0 1")
    with pytest.raises(ValueError, match="Invalid halfmove clock in FEN"):
        ChessGame.from_fen("8/8/8/8/8/8/8/4K2k w - - -1 1")
    with pytest.raises(ValueError, match="Invalid fullmove number in FEN"):
        ChessGame.from_fen("8/8/8/8/8/8/8/4K2k w - - 0 0")


def test_fifty_move_rule_draw_and_resets():
    board = position(
        ("e1", PieceType.KING, "w"),
        ("e8", PieceType.KING, "b"),
        ("a1", PieceType.ROOK, "w"),
    )
    # Initialize at 99 halfmoves
    game = ChessGame(board, turn="w", halfmove_clock=99)
    assert not game.is_fifty_moves()

    # Quiet move pushes clock to 100
    game.make_move("a1a2")
    assert game.halfmove_clock == 100
    assert game.is_fifty_moves()
    assert game.is_draw()
    assert game.draw_reason() == "fifty-move rule"
    assert not game.is_stalemate()
    assert game.status == GameStatus.ACTIVE
    assert game.legal_moves()

    # Pawn move resets halfmove clock
    pawn_game = ChessGame(halfmove_clock=50)
    pawn_game.make_move("e4")
    assert pawn_game.halfmove_clock == 0

    # Capture resets halfmove clock
    capture_board = position(
        ("e1", PieceType.KING, "w"),
        ("e8", PieceType.KING, "b"),
        ("a1", PieceType.ROOK, "w"),
        ("a8", PieceType.ROOK, "b"),
    )
    capture_game = ChessGame(capture_board, turn="w", halfmove_clock=40)
    capture_game.make_move("a1a8")
    assert capture_game.halfmove_clock == 0


def test_check_on_the_fifty_move_boundary_keeps_the_check_suffix():
    board = position(
        ("a1", PieceType.KING, "w"),
        ("a7", PieceType.ROOK, "w"),
        ("h8", PieceType.KING, "b"),
    )
    game = ChessGame(board, turn="w", halfmove_clock=99)
    game.make_move("a7h7+")
    assert game.is_check()
    assert game.is_fifty_moves()
    assert game.is_draw()
    assert not game.is_stalemate()
    assert game.status == GameStatus.CHECK
    assert game.legal_moves()


def test_threefold_repetition_draw():
    game = ChessGame()
    moves = ["Nf3", "Nf6", "Ng1", "Ng8", "Nf3", "Nf6", "Ng1", "Ng8"]
    for m in moves:
        game.make_move(m)

    assert game.is_threefold_repetition()
    assert game.is_draw()
    assert game.draw_reason() == "threefold repetition"
    assert not game.is_stalemate()
    assert game.status == GameStatus.ACTIVE
    assert game.legal_moves()

    # Test after() also triggers threefold repetition
    game2 = ChessGame()
    for m in moves[:-1]:
        game2 = game2.after(m)
    assert not game2.is_threefold_repetition()
    game2 = game2.after(moves[-1])
    assert game2.is_threefold_repetition()
    assert game2.is_draw()
    assert not game2.is_stalemate()


def test_threefold_repetition_ignores_unusable_en_passant_rights():
    pinned_ep = ChessGame.from_fen("k3r1n1/8/8/3pP3/8/8/8/4K1N1 w - d6 0 2")
    assert pinned_ep.to_fen().split()[3] == "d6"
    assert not any(move.special == "en_passant" for move in pinned_ep.legal_moves())

    cycle = ("Nf3", "Nf6", "Ng1", "Ng8")
    for _ in range(2):
        for move in cycle:
            pinned_ep.make_move(move)

    assert pinned_ep.is_threefold_repetition()
    assert pinned_ep.draw_reason() == "threefold repetition"

    legal_ep = ChessGame.from_fen("4k1n1/8/8/3pP3/8/8/8/4K1N1 w - d6 0 2")
    assert any(move.special == "en_passant" for move in legal_ep.legal_moves())
    for _ in range(2):
        for move in cycle:
            legal_ep.make_move(move)
    assert not legal_ep.is_threefold_repetition()


def test_unusable_en_passant_repetition_is_tracked_by_after():
    game = ChessGame.from_fen("4k1n1/8/8/3p4/8/8/8/4K1N1 w - d6 0 2")
    cycle = ("Nf3", "Nf6", "Ng1", "Ng8")
    for _ in range(2):
        for move in cycle:
            game = game.after(move)

    assert game.is_threefold_repetition()
    assert game.draw_reason() == "threefold repetition"


def test_insufficient_material_draw_variations():
    # K vs K
    kvk = position(("e1", PieceType.KING, "w"), ("e8", PieceType.KING, "b"))
    game_kvk = ChessGame(kvk)
    assert game_kvk.is_insufficient_material()
    assert game_kvk.is_draw()
    assert game_kvk.draw_reason() == "insufficient material"
    assert not game_kvk.is_stalemate()
    assert game_kvk.status == GameStatus.ACTIVE
    assert game_kvk.legal_moves()

    # K+N vs K
    knvk = position(
        ("e1", PieceType.KING, "w"),
        ("e8", PieceType.KING, "b"),
        ("c3", PieceType.KNIGHT, "w"),
    )
    assert ChessGame(knvk).is_insufficient_material()

    # K+B vs K
    kbvk = position(
        ("e1", PieceType.KING, "w"),
        ("e8", PieceType.KING, "b"),
        ("c1", PieceType.BISHOP, "w"),
    )
    assert ChessGame(kbvk).is_insufficient_material()

    # K+B vs K+B same color bishops (c1 is dark, f8 is dark)
    kbvkb_same = position(
        ("e1", PieceType.KING, "w"),
        ("c1", PieceType.BISHOP, "w"),
        ("e8", PieceType.KING, "b"),
        ("f8", PieceType.BISHOP, "b"),
    )
    assert ChessGame(kbvkb_same).is_insufficient_material()

    # K+B vs K+B opposite color bishops (c1 is dark, c8 is light)
    kbvkb_opp = position(
        ("e1", PieceType.KING, "w"),
        ("c1", PieceType.BISHOP, "w"),
        ("e8", PieceType.KING, "b"),
        ("c8", PieceType.BISHOP, "b"),
    )
    assert not ChessGame(kbvkb_opp).is_insufficient_material()

    # K+P vs K
    kpvk = position(
        ("e1", PieceType.KING, "w"),
        ("e8", PieceType.KING, "b"),
        ("e2", PieceType.PAWN, "w"),
    )
    assert not ChessGame(kpvk).is_insufficient_material()

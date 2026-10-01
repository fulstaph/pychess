import pytest

from chess import Board, ChessGame, GameStatus, Piece, PieceType, notation_to_coords
from chess.attacks import is_in_check
from chess.game import _piece_moves


def sq(text):
    return notation_to_coords(text)


def board_of(*placements):
    board = Board()
    for name, kind, color in placements:
        board = board.with_piece(sq(name), Piece(kind, color))
    return board


def perft(game, depth):
    if depth == 0:
        return 1
    total = 0
    for move in game.legal_moves():
        total += perft(game.after(move), depth - 1)
    return total


def test_opening_perft_counts_all_legal_replies():
    game = ChessGame()
    assert perft(game, 1) == 20
    assert perft(game, 2) == 400
    assert perft(game, 3) == 8902


def test_kiwipete_perft_covers_castling_and_tactical_replies():
    game = ChessGame.from_fen(
        "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1"
    )
    assert perft(game, 1) == 48
    assert perft(game, 2) == 2039


def test_fools_mate_and_terminal_atomicity():
    game = ChessGame()
    for move in ("f2f3", "e7e5", "g2g4", "d8h4"):
        game.make_move(move)
    assert game.status == GameStatus.CHECKMATE and game.get_winner() == "b"
    board = game.board
    with pytest.raises(ValueError):
        game.make_move("a2a3")
    assert game.board is board and game.turn == "w"


def test_pinned_piece_and_en_passant_must_not_expose_king():
    board = board_of(
        ("e1", PieceType.KING, "w"),
        ("a8", PieceType.KING, "b"),
        ("e8", PieceType.ROOK, "b"),
        ("e2", PieceType.ROOK, "w"),
    )
    game = ChessGame(board)
    initial = game.board
    with pytest.raises(ValueError):
        game.make_move("e2d2")
    assert game.board is initial
    board = board_of(
        ("e1", PieceType.KING, "w"),
        ("a8", PieceType.KING, "b"),
        ("e8", PieceType.ROOK, "b"),
        ("e5", PieceType.PAWN, "w"),
        ("d5", PieceType.PAWN, "b"),
    )
    game = ChessGame(board, en_passant=sq("d6"))
    assert all(move.special != "en_passant" for move in game.legal_moves())
    with pytest.raises(ValueError):
        game.make_move("e5d6")


def test_black_castling_and_rook_capture_rights():
    board = board_of(
        ("e1", PieceType.KING, "w"),
        ("a1", PieceType.ROOK, "w"),
        ("h1", PieceType.ROOK, "w"),
        ("e8", PieceType.KING, "b"),
        ("a8", PieceType.ROOK, "b"),
        ("h8", PieceType.ROOK, "b"),
    )
    for destination, rook_destination in (("g8", "f8"), ("c8", "d8")):
        game = ChessGame(board, turn="b", castling_rights=frozenset("KQkq"))
        move = next(
            move
            for move in game.legal_moves()
            if move.piece.type == PieceType.KING and move.to_square == sq(destination)
        )
        game.make_move(move)
        assert game.board.get(sq(rook_destination)) == Piece(PieceType.ROOK, "b")
        assert game.move_num == 2 and game.castles("w", "queenside")
        assert not game.castles("b", "kingside") and not game.castles("b", "queenside")
    board = board.with_piece(sq("b7"), Piece(PieceType.BISHOP, "w"))
    game = ChessGame(board, castling_rights=frozenset("KQkq"))
    game.make_move("b7a8")
    assert not game.castles("b", "queenside") and game.castles("b", "kingside")


def test_castling_through_attack_is_forbidden_and_missing_king_is_invalid():
    board = board_of(
        ("e1", PieceType.KING, "w"),
        ("h1", PieceType.ROOK, "w"),
        ("a8", PieceType.KING, "b"),
        ("f8", PieceType.ROOK, "b"),
    )
    game = ChessGame(board, castling_rights=frozenset("K"))
    assert all(move.special != "castle_k" for move in game.legal_moves())
    with pytest.raises(ValueError):
        is_in_check(Board(), "w")


def test_black_underpromotion_and_invalid_promotion_atomicity():
    board = board_of(
        ("h1", PieceType.KING, "w"),
        ("h8", PieceType.KING, "b"),
        ("a2", PieceType.PAWN, "b"),
    )
    game = ChessGame(board, turn="b")
    with pytest.raises(ValueError):
        game.make_move("a2a1=X")
    assert game.board is board and game.turn == "b"
    game.make_move("a2a1=N")
    assert game.board.get(sq("a1")) == Piece(PieceType.KNIGHT, "b")


def assert_tactical_generation_matches_full(game, depth):
    # Query the uncached generators first; legal_moves() then fills the cache.
    tactical = game._legal_tactical_moves()
    has_move = game._has_legal_move()
    legal = game.legal_moves()
    assert tactical == tuple(
        move for move in legal if move.is_capture() or move.special == "promotion"
    )
    assert has_move == bool(legal)
    if depth:
        for move in legal:
            assert_tactical_generation_matches_full(game.after(move), depth - 1)


def test_tactical_generation_matches_filtered_legal_moves_across_positions():
    positions = (
        # Promotions, discovered checks and castling rights (perft position 4).
        "r3k2r/Pppp1ppp/1b3nbN/nP6/BBP1P3/q4N2/Pp1P2PP/R2Q1RK1 w kq - 0 1",
        # Pinned pawns, en passant exposing the king along a rank.
        "8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1",
        # Kiwipete: dense captures and castling.
        "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1",
        # Stalemate: no legal move although not in check.
        "7k/5Q2/6K1/8/8/8/8/8 b - - 0 1",
        # Checkmate: no legal move while in check.
        "rnb1kbnr/pppp1ppp/8/4p3/6Pq/5P2/PPPPP2P/RNBQKBNR w KQkq - 1 3",
    )
    for fen in positions:
        assert_tactical_generation_matches_full(ChessGame.from_fen(fen), 1)
    assert not ChessGame.from_fen(positions[3])._has_legal_move()
    assert not ChessGame.from_fen(positions[4])._has_legal_move()


def reference_legal_moves(game):
    """Execute-and-reject over every pseudo-legal move, with no shortcuts."""
    state = game._state
    return tuple(
        move
        for row, squares in enumerate(state.board.squares)
        for col, piece in enumerate(squares)
        if piece is not None and piece.color == state.turn
        for move in _piece_moves(state, (row, col), piece)
        if not is_in_check(move.execute(state.board), state.turn)
    )


def assert_legal_generation_matches_reference(game, depth):
    legal = reference_legal_moves(game)
    assert game.legal_moves() == legal
    if depth:
        for move in legal:
            assert_legal_generation_matches_reference(game.after(move), depth - 1)


def test_pin_aware_legality_matches_execute_and_reject_reference():
    positions = (
        "r3k2r/Pppp1ppp/1b3nbN/nP6/BBP1P3/q4N2/Pp1P2PP/R2Q1RK1 w kq - 0 1",
        "8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1",
        "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1",
        "rnbq1k1r/pp1Pbppp/2p5/8/2B5/8/PPP1NnPP/RNBQK2R w KQ - 1 8",
        # En passant that would expose the king along the fourth rank.
        "8/8/8/8/k2Pp2Q/8/8/3K4 b - d3 0 1",
        # Rook pinned on a file, knight pinned on a diagonal, and two
        # blockers on one ray (no pin).
        "4r1k1/8/8/8/8/8/4R3/4K3 w - - 0 1",
        "6k1/8/8/8/1b6/8/3N4/4K3 w - - 0 1",
        "4r1k1/8/8/8/4N3/8/4R3/4K3 w - - 0 1",
    )
    for fen in positions:
        assert_legal_generation_matches_reference(ChessGame.from_fen(fen), 2)
    ep_pinned = ChessGame.from_fen(positions[4])
    assert all(move.special != "en_passant" for move in ep_pinned.legal_moves())

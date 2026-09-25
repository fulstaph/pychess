"""Tests for the FastAPI Web UI and REST API endpoints."""

from fastapi.testclient import TestClient

from chess import ChessGame
from chess.web import create_app, serialize_game_state


def test_serialize_game_state_structure():
    game = ChessGame()
    data = serialize_game_state(game)
    assert data["turn"] == "w"
    assert data["status"] == "active"
    assert data["is_check"] is False
    assert data["winner"] is None
    assert "e2" in data["dests"]
    assert "e4" in data["dests"]["e2"]
    assert "e2e4" in data["legal_moves"]
    assert data["captured_w"] == []
    assert data["captured_b"] == []


def test_web_endpoints_full_flow():
    client = TestClient(create_app())

    # GET / (HTML UI page)
    res_root = client.get("/")
    assert res_root.status_code == 200
    html = res_root.text
    assert "Chessground" in html
    assert "pychess" in html
    assert '<div id="cg-board">' in html

    # GET /api/state
    state = client.get("/api/state").json()
    assert state["turn"] == "w"
    assert state["status"] == "active"

    # POST /api/move (human move e2e4)
    res_move = client.post("/api/move", json={"move": "e2e4", "ai_reply": False})
    assert res_move.status_code == 200
    state_after_e4 = res_move.json()
    assert state_after_e4["turn"] == "b"
    assert "e2e4" in state_after_e4["history"]

    # POST /api/move with AI reply
    res_move_ai = client.post(
        "/api/move", json={"move": "e7e5", "ai_reply": True, "depth": 1}
    )
    assert res_move_ai.status_code == 200
    state_after_ai = res_move_ai.json()
    # e2e4 (1), e7e5 (2), White AI reply (3)
    assert len(state_after_ai["history"]) >= 3

    # POST /api/move with invalid move -> 400
    res_bad = client.post("/api/move", json={"move": "invalid_move"})
    assert res_bad.status_code == 400

    # POST /api/undo
    res_undo = client.post("/api/undo", json={"steps": 1})
    assert res_undo.status_code == 200

    # POST /api/ai_move
    res_ai = client.post("/api/ai_move", json={"depth": 1})
    assert res_ai.status_code == 200

    # POST /api/reset (custom FEN)
    custom_fen = "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq e3 0 1"
    res_reset_fen = client.post("/api/reset", json={"fen": custom_fen})
    assert res_reset_fen.status_code == 200
    state_custom = res_reset_fen.json()
    assert state_custom["fen"] == custom_fen
    assert state_custom["turn"] == "b"

    # POST /api/reset with knight on c7, then move it
    knight_fen = "r2k1bnr/ppN1pppp/2n5/8/8/2N5/PPPP1PPP/R1B1K2R w KQ - 0 1"
    res_reset_kn = client.post("/api/reset", json={"fen": knight_fen})
    assert res_reset_kn.status_code == 200

    res_kn_move = client.post("/api/move", json={"move": "c7a8", "ai_reply": False})
    assert res_kn_move.status_code == 200
    state_kn = res_kn_move.json()
    assert "c7a8" in state_kn["history"]

    # POST /api/reset (initial position)
    res_reset_init = client.post("/api/reset", json={})
    assert res_reset_init.status_code == 200
    state_init = res_reset_init.json()
    assert state_init["turn"] == "w"
    assert state_init["history"] == []

    # 404 for unknown endpoint
    assert client.get("/nonexistent/endpoint").status_code == 404


def test_api_move_rejects_empty_and_missing_move():
    client = TestClient(create_app())

    res_empty = client.post("/api/move", json={"move": "   "})
    assert res_empty.status_code == 400

    res_missing = client.post("/api/move", json={})
    assert res_missing.status_code == 400


def test_api_reset_rejects_invalid_fen():
    client = TestClient(create_app())
    res = client.post("/api/reset", json={"fen": "not-a-fen"})
    assert res.status_code == 400
    state = client.get("/api/state").json()
    assert state["turn"] == "w"
    assert state["history"] == []


def test_api_ai_move_rejects_finished_game():
    client = TestClient(create_app())
    # Fool's mate: f3 e5 g4 Qh4#
    for move in ("f2f3", "e7e5", "g2g4"):
        res = client.post("/api/move", json={"move": move, "ai_reply": False})
        assert res.status_code == 200
    res_mate = client.post("/api/move", json={"move": "d8h4", "ai_reply": False})
    assert res_mate.status_code == 200
    state_mate = res_mate.json()
    assert state_mate["is_checkmate"] is True
    assert state_mate["winner"] == "b"

    res_ai = client.post("/api/ai_move", json={"depth": 1})
    assert res_ai.status_code == 400

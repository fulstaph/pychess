"""Tests for local Web UI server and REST API endpoints."""

import json
import threading
import urllib.error
import urllib.request

import pytest

from chess import ChessGame
from chess.web import create_web_server, serialize_game_state


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


def test_web_server_endpoints():
    server = create_web_server(host="127.0.0.1", port=0)
    host, port = server.server_address
    base_url = f"http://{host}:{port}"

    server_thread = threading.Thread(target=server.serve_forever, daemon=True)
    server_thread.start()

    try:
        # GET / (HTML UI page)
        req_root = urllib.request.urlopen(f"{base_url}/", timeout=5)
        assert req_root.status == 200
        html = req_root.read().decode("utf-8")
        assert "Chessground" in html
        assert "pychess" in html
        assert '<div id="cg-board">' in html

        # GET /api/state
        req_state = urllib.request.urlopen(f"{base_url}/api/state", timeout=5)
        assert req_state.status == 200
        state = json.loads(req_state.read().decode("utf-8"))
        assert state["turn"] == "w"
        assert state["status"] == "active"

        # POST /api/move (Human move e2e4)
        req_move = urllib.request.Request(
            f"{base_url}/api/move",
            data=json.dumps({"move": "e2e4", "ai_reply": False}).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        res_move = urllib.request.urlopen(req_move, timeout=5)
        assert res_move.status == 200
        state_after_e4 = json.loads(res_move.read().decode("utf-8"))
        assert state_after_e4["turn"] == "b"
        assert "e2e4" in state_after_e4["history"]

        # POST /api/move with AI reply
        payload_ai = {"move": "e7e5", "ai_reply": True, "depth": 1}
        req_move_ai = urllib.request.Request(
            f"{base_url}/api/move",
            data=json.dumps(payload_ai).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        res_move_ai = urllib.request.urlopen(req_move_ai, timeout=5)
        assert res_move_ai.status == 200
        state_after_ai = json.loads(res_move_ai.read().decode("utf-8"))
        # e2e4 (history 1), e7e5 (history 2), White AI reply (history 3)
        assert len(state_after_ai["history"]) >= 3

        # POST /api/move with invalid move -> 400
        req_bad_move = urllib.request.Request(
            f"{base_url}/api/move",
            data=json.dumps({"move": "invalid_move"}).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with pytest.raises(urllib.error.HTTPError) as exc_info:
            urllib.request.urlopen(req_bad_move, timeout=5)
        assert exc_info.value.code == 400

        # POST /api/undo
        req_undo = urllib.request.Request(
            f"{base_url}/api/undo",
            data=json.dumps({"steps": 1}).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        res_undo = urllib.request.urlopen(req_undo, timeout=5)
        assert res_undo.status == 200

        # POST /api/ai_move
        req_ai = urllib.request.Request(
            f"{base_url}/api/ai_move",
            data=json.dumps({"depth": 1}).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        res_ai = urllib.request.urlopen(req_ai, timeout=5)
        assert res_ai.status == 200

        # POST /api/reset (custom FEN)
        custom_fen = "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq e3 0 1"
        req_reset_fen = urllib.request.Request(
            f"{base_url}/api/reset",
            data=json.dumps({"fen": custom_fen}).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        res_reset_fen = urllib.request.urlopen(req_reset_fen, timeout=5)
        assert res_reset_fen.status == 200
        state_custom = json.loads(res_reset_fen.read().decode("utf-8"))
        assert state_custom["fen"] == custom_fen
        assert state_custom["turn"] == "b"

        # POST /api/reset with knight on c7
        knight_fen = "r2k1bnr/ppN1pppp/2n5/8/8/2N5/PPPP1PPP/R1B1K2R w KQ - 0 1"
        req_reset_kn = urllib.request.Request(
            f"{base_url}/api/reset",
            data=json.dumps({"fen": knight_fen}).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        res_reset_kn = urllib.request.urlopen(req_reset_kn, timeout=5)
        assert res_reset_kn.status == 200

        # Move knight c7 to a8 (succeeds as c7a8 without promotion)
        req_kn_move = urllib.request.Request(
            f"{base_url}/api/move",
            data=json.dumps({"move": "c7a8", "ai_reply": False}).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        res_kn_move = urllib.request.urlopen(req_kn_move, timeout=5)
        assert res_kn_move.status == 200
        state_kn = json.loads(res_kn_move.read().decode("utf-8"))
        assert "c7a8" in state_kn["history"]
        # POST /api/reset (initial position)
        req_reset_init = urllib.request.Request(
            f"{base_url}/api/reset",
            data=json.dumps({}).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        res_reset_init = urllib.request.urlopen(req_reset_init, timeout=5)
        assert res_reset_init.status == 200
        state_init = json.loads(res_reset_init.read().decode("utf-8"))
        assert state_init["turn"] == "w"
        assert state_init["history"] == []

        # 404 for unknown endpoint
        with pytest.raises(urllib.error.HTTPError) as err_404:
            urllib.request.urlopen(f"{base_url}/nonexistent/endpoint", timeout=5)
        assert err_404.value.code == 404

    finally:
        server.shutdown()
        server.server_close()

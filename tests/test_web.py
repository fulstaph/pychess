"""Tests for the FastAPI Web UI and REST API endpoints."""

import threading

from fastapi.testclient import TestClient

from chess.engine import EngineConfig
from chess.game import ChessGame
from chess.uci import to_uci
from chess.web import create_app, serialize_game_state

MATE_FEN = "7k/4rppp/8/8/8/8/5PPP/R5K1 b - - 0 1"
RESET_FEN = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"


def test_serialize_game_state_structure():
    game = ChessGame()
    data = serialize_game_state(game)
    assert data["turn"] == "w"
    assert data["status"] == "active"
    assert data["is_checkmate"] is False
    assert data["captured_w"] == []
    assert data["captured_b"] == []
    assert data["history"] == []
    assert "e2e4" in data["legal_moves"]
    assert data["dests"]["e2"] == ["e3", "e4"]


def _create(client, fen: str = "") -> dict:
    res = client.post("/api/sessions", json={"fen": fen})
    assert res.status_code == 200, res.text
    return res.json()


def _session_id(client, n: int = 0) -> int:
    res = client.get("/api/sessions")
    assert res.status_code == 200
    sessions = res.json()
    assert len(sessions) > n
    return sessions[n]["id"]


def test_web_endpoints_full_flow():
    client = TestClient(create_app())

    # GET / -> HTML page
    page = client.get("/")
    assert page.status_code == 200
    assert "Chessground" in page.text

    # No sessions yet
    assert client.get("/api/sessions").json() == []

    # Create the first session
    data = _create(client)
    session_id = data["id"]
    assert data["turn"] == "w"

    # List shows one session with its live state
    sessions = client.get("/api/sessions").json()
    assert len(sessions) == 1
    assert sessions[0]["id"] == session_id
    assert sessions[0]["move_count"] == 0
    assert sessions[0]["state"]["fen"].startswith("rnbqkbnr")

    # GET /api/sessions/{id} -> full state
    state = client.get(f"/api/sessions/{session_id}").json()
    assert state["fen"].startswith("rnbqkbnr")

    # Move e2e4 without an engine reply
    res = client.post(
        f"/api/sessions/{session_id}/move",
        json={"move": "e2e4", "ai_reply": False},
    )
    assert res.status_code == 200
    assert res.json()["turn"] == "b"
    assert res.json()["history"] == ["e2e4"]

    # Invalid move is rejected and the position is unchanged
    res = client.post(
        f"/api/sessions/{session_id}/move",
        json={"move": "e5e4"},
    )
    assert res.status_code == 400
    assert "Invalid move" in res.json()["detail"]
    state = client.get(f"/api/sessions/{session_id}").json()
    assert state["history"] == ["e2e4"]
    assert state["turn"] == "b"

    # Unknown session ids 404 on every per-session endpoint
    assert client.get("/api/sessions/999").status_code == 404
    assert (
        client.post("/api/sessions/999/move", json={"move": "e2e4"}).status_code == 404
    )
    assert client.delete("/api/sessions/999").status_code == 404

    # Delete the session
    assert client.delete(f"/api/sessions/{session_id}").json() == {
        "deleted": session_id
    }
    assert client.delete(f"/api/sessions/{session_id}").status_code == 404
    assert client.get("/api/sessions").json() == []

    # Unknown routes still 404
    assert client.get("/nonexistent/endpoint").status_code == 404


def test_api_move_rejects_empty_and_missing_move():
    client = TestClient(create_app())
    sid = _create(client)["id"]
    res_empty = client.post(f"/api/sessions/{sid}/move", json={"move": ""})
    res_missing = client.post(f"/api/sessions/{sid}/move", json={})
    assert res_empty.status_code == 400
    assert res_missing.status_code == 400


def test_api_create_session_rejects_invalid_fen():
    client = TestClient(create_app())
    res = client.post("/api/sessions", json={"fen": "not-a-fen"})
    assert res.status_code == 400
    assert "Invalid FEN" in res.json()["detail"]
    assert client.get("/api/sessions").json() == []


def test_api_ai_move_rejects_finished_game():
    client = TestClient(create_app())
    data = _create(client)
    sid = data["id"]
    # Fool's mate: f3 e5 g4 Qh4#
    for move in ("f2f3", "e7e5", "g2g4", "d8h4"):
        res = client.post(
            f"/api/sessions/{sid}/move",
            json={"move": move, "ai_reply": False},
        )
        assert res.status_code == 200, res.text
    state = res.json()
    assert state["status"] == "checkmate"
    assert state["winner"] == "b"
    res_ai = client.post(f"/api/sessions/{sid}/ai_move", json={})
    assert res_ai.status_code == 400


def test_api_undo_steps_back_and_over_undoes_to_initial():
    client = TestClient(create_app())
    data = _create(client)
    sid = data["id"]
    start = data["fen"]
    for move in ("e2e4", "e7e5", "d2d4"):
        res = client.post(
            f"/api/sessions/{sid}/move",
            json={"move": move, "ai_reply": False},
        )
        assert res.status_code == 200, res.text
    # After three plies it is Black to move.
    assert res.json()["turn"] == "b"
    assert res.json()["history"] == ["e2e4", "e7e5", "d2d4"]
    res_undo = client.post(f"/api/sessions/{sid}/undo", json={"steps": 2})
    assert res_undo.status_code == 200
    assert res_undo.json()["history"] == ["e2e4"]
    assert res_undo.json()["turn"] == "b"
    # Over-undoing past the start position is clamped to the initial FEN.
    res_over = client.post(f"/api/sessions/{sid}/undo", json={"steps": 5})
    assert res_over.status_code == 200
    assert res_over.json()["history"] == []
    state = client.get(f"/api/sessions/{sid}").json()
    assert state["fen"].startswith(start)


def test_api_reset_with_fen_replaces_position():
    client = TestClient(create_app())
    data = _create(client)
    sid = data["id"]
    client.post(f"/api/sessions/{sid}/move", json={"move": "e2e4", "ai_reply": False})
    custom_fen = RESET_FEN
    res = client.post(f"/api/sessions/{sid}/reset", json={"fen": custom_fen})
    assert res.status_code == 200
    state = client.get(f"/api/sessions/{sid}").json()
    assert state["fen"] == custom_fen
    assert state["history"] == []


def test_api_reset_rejects_invalid_fen():
    client = TestClient(create_app())
    sid = _create(client)["id"]
    res = client.post(f"/api/sessions/{sid}/reset", json={"fen": "garbage"})
    assert res.status_code == 400
    state = client.get(f"/api/sessions/{sid}").json()
    assert state["history"] == []


def test_api_reset_clears_history_after_undo():
    client = TestClient(create_app())
    sid = _create(client)["id"]
    client.post(f"/api/sessions/{sid}/move", json={"move": "d2d4", "ai_reply": False})
    client.post(f"/api/sessions/{sid}/undo", json={"steps": 1})
    res = client.post(f"/api/sessions/{sid}/reset", json={})
    assert res.status_code == 200
    assert res.json()["history"] == []
    res2 = client.post(
        f"/api/sessions/{sid}/move", json={"move": "d2d4", "ai_reply": False}
    )
    assert res2.json()["history"] == ["d2d4"]


def test_sessions_are_isolated():
    client = TestClient(create_app())
    first = _create(client)
    second = _create(client)
    sid_a, sid_b = first["id"], second["id"]
    assert sid_a != sid_b

    res = client.post(
        f"/api/sessions/{sid_a}/move",
        json={"move": "e2e4", "ai_reply": False},
    )
    assert res.status_code == 200

    state_a = client.get(f"/api/sessions/{sid_a}").json()
    state_b = client.get(f"/api/sessions/{sid_b}").json()
    assert state_a["history"] == ["e2e4"]
    assert state_b["history"] == []
    assert state_b["turn"] == "w"

    # Undoing on one session leaves the other untouched.
    res_undo = client.post(f"/api/sessions/{sid_a}/undo", json={"steps": 1})
    assert res_undo.status_code == 200
    assert client.get(f"/api/sessions/{sid_a}").json()["history"] == []
    assert client.get(f"/api/sessions/{sid_b}").json()["history"] == []


def test_file_backed_session_survives_restart(tmp_path):
    db = tmp_path / "games.db"
    with TestClient(create_app(db_path=str(db))) as first:
        data = _create(first)
        sid = data["id"]
        first.post(
            f"/api/sessions/{sid}/move",
            json={"move": "e2e4", "ai_reply": False},
        )

    with TestClient(create_app(db_path=str(db))) as second:
        state = second.get(f"/api/sessions/{sid}").json()
        assert state["history"] == ["e2e4"]
        assert state["turn"] == "b"
        # Undoing after the cold replay truncates the stored plies.
        res_undo = second.post(f"/api/sessions/{sid}/undo", json={"steps": 1})
        assert res_undo.json()["history"] == []


def test_delete_session_removes_stored_plies():
    client = TestClient(create_app())
    data = _create(client)
    sid = data["id"]
    client.post(
        f"/api/sessions/{sid}/move",
        json={"move": "e2e4", "ai_reply": False},
    )
    store = client.app.state.store
    assert len(store.get_moves(sid)) == 1
    assert client.delete(f"/api/sessions/{sid}").status_code == 200
    assert store.get_moves(sid) is None or store.get_start_fen(sid) is None


def test_concurrent_sessions_serialize_cleanly():
    """Two threads on different sessions run in parallel; each session's
    stored plies stay consistent with its own history."""
    client = TestClient(create_app())
    sid_a = _create(client)["id"]
    sid_b = _create(client)["id"]
    store = client.app.state.store

    def play(sid, moves):
        for m in moves:
            res = client.post(
                f"/api/sessions/{sid}/move",
                json={"move": m, "ai_reply": False},
            )
            assert res.status_code == 200, res.text

    threads = [
        threading.Thread(target=play, args=(sid_a, ["e2e4", "e7e5", "d2d4", "d7d5"])),
        threading.Thread(target=play, args=(sid_b, ["c2c4", "c7c5", "g1f3", "g8f6"])),
    ]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    for sid, expected in (
        (sid_a, ["e2e4", "e7e5", "d2d4", "d7d5"]),
        (sid_b, ["c2c4", "c7c5", "g1f3", "g8f6"]),
    ):
        state = client.get(f"/api/sessions/{sid}").json()
        assert state["history"] == expected
        assert store.get_moves(sid) == expected


def test_ai_move_uses_app_search_config_and_validates_depth_overrides():
    config = EngineConfig(search_depth=1, quiescence=False)
    client = TestClient(create_app(engine_config=config))
    sid = _create(client)["id"]

    for depth in (0, 9):
        invalid = client.post(
            f"/api/sessions/{sid}/ai_move",
            json={"depth": depth},
        )
        assert invalid.status_code == 422
    assert client.get(f"/api/sessions/{sid}").json()["history"] == []

    result = client.post(f"/api/sessions/{sid}/ai_move", json={})
    assert result.status_code == 200, result.text
    state = result.json()
    assert len(state["history"]) == 1
    valid_moves = {to_uci(move) for move in ChessGame().legal_moves()}
    assert state["history"][0] in valid_moves
    assert state["turn"] == "b"

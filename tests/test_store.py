"""Tests for the SQLite GameStore persistence layer."""

import os
import tempfile

import pytest

from chess.store import GameStore


def test_create_and_list_games():
    store = GameStore()
    assert store.list_games() == []

    first = store.create_game("start-fen-1")
    second = store.create_game("start-fen-2", name="my game")
    assert first == 1
    assert second == 2

    games = store.list_games()
    # Newest first, with the default timestamped name on the first row.
    assert [g.id for g in games] == [2, 1]
    assert games[0].name == "my game"
    assert games[0].start_fen == "start-fen-2"
    assert games[0].move_count == 0
    assert games[1].start_fen == "start-fen-1"
    store.close()


def test_append_truncate_and_replay_data():
    store = GameStore()
    gid = store.create_game("start")
    store.append_move(gid, "e2e4")
    store.append_move(gid, "e7e5")
    store.append_move(gid, "g1f3")

    fen, ucis = store.get_game(gid)
    assert fen == "start"
    assert ucis == ["e2e4", "e7e5", "g1f3"]

    # Undo truncates the tail; remaining plies are untouched.
    store.truncate_moves(gid, 2)
    assert store.get_moves(gid) == ["e2e4", "e7e5"]

    # Re-appending after a truncate reuses the next ply number.
    store.append_move(gid, "d2d4")
    assert store.get_moves(gid) == ["e2e4", "e7e5", "d2d4"]

    # Reset rewrites the start FEN in place; the row keeps its id.
    store.set_start_fen(gid, "custom-fen")
    assert store.get_start_fen(gid) == "custom-fen"
    assert store.list_games()[0].start_fen == "custom-fen"
    assert store.list_games()[0].id == gid
    store.close()


def test_delete_game_cascades_and_reports():
    store = GameStore()
    gid = store.create_game("start")
    store.append_move(gid, "e2e4")

    assert store.delete_game(gid) is True
    assert store.list_games() == []
    assert store.get_start_fen(gid) is None
    # Deleting again reports no row removed.
    assert store.delete_game(gid) is False
    store.close()


def test_get_game_unknown_id_raises():
    store = GameStore()
    with pytest.raises(LookupError):
        store.get_game(999)
    store.close()


def test_persistence_across_connections():
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "games.db")
        store = GameStore(path)
        gid = store.create_game("start")
        store.append_move(gid, "e2e4")
        store.close()

        reopened = GameStore(path)
        fen, ucis = reopened.get_game(gid)
        assert fen == "start"
        assert ucis == ["e2e4"]
        reopened.close()

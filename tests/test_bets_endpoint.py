"""Tests for GET /bt/api/bets (recent resolved-bet history).

Locks two behaviours the UI relies on:
- ``db.rounds_history`` only queries true wagering outcomes
  (status in settled/cashed_out) — never open/abandoned/voided rounds,
  which would render misleading 0.00x multipliers in the panel;
- the endpoint returns exactly the fields the panel consumes, with
  bet/payout coerced to ints (payout may be NULL in the DB).
"""

import pytest
from fastapi.testclient import TestClient

from api import db
from api.auth import require_user
from api.main import app


@pytest.fixture()
def client():
    app.dependency_overrides[require_user] = lambda: {
        "tg_id": 1, "user": {"id": 1}, "username": "u", "display_name": "U",
    }
    c = TestClient(app)
    yield c
    app.dependency_overrides.clear()


def test_rounds_history_filters_to_resolved_bets_only(monkeypatch):
    captured = {}

    async def fake_get(table, params):
        captured["table"] = table
        captured["params"] = params
        return []

    monkeypatch.setattr(db, "_get", fake_get)

    import asyncio
    asyncio.run(db.rounds_history(1, limit=50))

    assert captured["table"] == "bt_game_rounds"
    p = captured["params"]
    assert p["tg_id"] == "eq.1"
    assert p["status"] == "in.(settled,cashed_out)"
    assert p["order"] == "created_at.desc"
    assert p["limit"] == "50"


def test_bets_endpoint_shape_and_int_coercion(client, monkeypatch):
    async def fake_history(tg_id, limit=50):
        assert tg_id == 1 and limit == 50
        return [
            {"id": "r2", "game": "dice", "bet": 100, "payout": 196,
             "status": "settled", "created_at": "2026-07-08T10:00:00Z",
             "settled_at": "2026-07-08T10:00:01Z"},
            # payout NULL from DB (never credited) must serialise as 0.
            {"id": "r1", "game": "towers", "bet": 50, "payout": None,
             "status": "settled", "created_at": "2026-07-08T09:00:00Z",
             "settled_at": "2026-07-08T09:00:05Z"},
        ]

    monkeypatch.setattr(db, "rounds_history", fake_history)

    r = client.get("/bt/api/bets")
    assert r.status_code == 200
    rows = r.json()["rows"]
    assert len(rows) == 2
    assert rows[0] == {"id": "r2", "game": "dice", "bet": 100, "payout": 196,
                       "status": "settled", "created_at": "2026-07-08T10:00:00Z",
                       "settled_at": "2026-07-08T10:00:01Z"}
    assert rows[1]["payout"] == 0
    assert isinstance(rows[1]["bet"], int)

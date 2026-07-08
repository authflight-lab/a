"""Regression tests for _open_bet after the hot-path round-trip removal.

Scenarios:
1. Clean bet opens — and does NOT call get_user or get_open_round (hot path).
2. Genuine insufficient balance (no leftover) -> insufficient_balance.
3. First-time user (no seed pair) -> upsert_user + create_seed_pair, bet opens.
4. Leftover open round holding the points -> RPC raises insufficient_balance,
   leftover is voided (refund) and the retry succeeds.
5. Leftover with SUFFICIENT balance -> RPC raises open_round_exists,
   leftover is voided (refund) and the retry succeeds.
6. Leftover that can never be cleared -> the retry loop stays bounded and
   returns try_again (no infinite loop, no double refund).
"""

import pytest
from fastapi.testclient import TestClient

from api import db, main
from api.auth import require_user
from api.main import app

PAIR = {"server_seed": "s" * 64, "client_seed": "c" * 16, "nonce": 7,
        "server_hash": "h" * 64}
OPEN_RESULT = {"round_id": "r-1", "server_hash": PAIR["server_hash"],
               "nonce": 7, "balance": 900}


@pytest.fixture()
def client():
    app.dependency_overrides[require_user] = lambda: {
        "tg_id": 42, "user": {"id": 42}, "username": "u", "display_name": "U",
    }
    c = TestClient(app)
    yield c
    app.dependency_overrides.clear()


def _bet(client):
    return client.post("/bt/api/game/dice/bet",
                       json={"bet": 100, "params": {"target": 50}})


def test_clean_bet_opens_without_pre_reads(client, monkeypatch):
    calls = []

    async def fake_get_seed_pair(tg_id):
        calls.append("get_seed_pair")
        return dict(PAIR)

    async def fake_open_round(tg_id, game, bet, nonce, params, outcome):
        calls.append("open_round")
        assert (tg_id, game, bet, nonce) == (42, "dice", 100, 7)
        return dict(OPEN_RESULT)

    async def boom(*a, **k):  # any pre-read = regression
        raise AssertionError("hot path must not touch get_user/get_open_round")

    monkeypatch.setattr(db, "get_seed_pair", fake_get_seed_pair)
    monkeypatch.setattr(db, "open_round", fake_open_round)
    monkeypatch.setattr(db, "get_user", boom)
    monkeypatch.setattr(db, "get_open_round", boom)
    monkeypatch.setattr(db, "upsert_user", boom)

    r = _bet(client)
    assert r.status_code == 200 and r.json().get("error") is None
    assert r.json()["round_id"] == "r-1"
    assert calls == ["get_seed_pair", "open_round"]


def test_insufficient_balance_no_leftover(client, monkeypatch):
    async def fake_get_seed_pair(tg_id):
        return dict(PAIR)

    async def fake_open_round(*a, **k):
        raise db.InsufficientBalance("insufficient_balance")

    async def fake_get_open_round(tg_id, game):
        return None  # no leftover to reclaim

    monkeypatch.setattr(db, "get_seed_pair", fake_get_seed_pair)
    monkeypatch.setattr(db, "open_round", fake_open_round)
    monkeypatch.setattr(db, "get_open_round", fake_get_open_round)

    r = _bet(client)
    assert r.json() == {"ok": False, "error": "insufficient_balance"}


def test_first_time_user_gets_row_then_bets(client, monkeypatch):
    calls = []

    async def fake_get_seed_pair(tg_id):
        calls.append("get_seed_pair")
        return None  # first bet ever

    async def fake_upsert_user(tg_id, username=None, display_name=None):
        calls.append("upsert_user")
        assert tg_id == 42
        return {"tg_id": 42, "balance": 0}

    async def fake_create_seed_pair(tg_id, pair):
        calls.append("create_seed_pair")
        return dict(PAIR)

    async def fake_open_round(tg_id, game, bet, nonce, params, outcome):
        calls.append("open_round")
        return dict(OPEN_RESULT)

    monkeypatch.setattr(db, "get_seed_pair", fake_get_seed_pair)
    monkeypatch.setattr(db, "upsert_user", fake_upsert_user)
    monkeypatch.setattr(db, "create_seed_pair", fake_create_seed_pair)
    monkeypatch.setattr(db, "open_round", fake_open_round)

    r = _bet(client)
    assert r.status_code == 200 and r.json().get("error") is None
    assert calls == ["get_seed_pair", "upsert_user", "create_seed_pair",
                     "open_round"]


def test_leftover_round_voided_then_retry_succeeds(client, monkeypatch):
    calls = []
    attempts = {"n": 0}
    leftover = {"id": "old-1", "bet": 100, "outcome": {}}

    async def fake_get_seed_pair(tg_id):
        return dict(PAIR)

    async def fake_open_round(tg_id, game, bet, nonce, params, outcome):
        attempts["n"] += 1
        if attempts["n"] == 1:
            # Leftover holds the points: debit fails before the unique index.
            raise db.InsufficientBalance("insufficient_balance")
        calls.append("open_round_retry")
        return dict(OPEN_RESULT)

    async def fake_get_open_round(tg_id, game):
        calls.append("get_open_round")
        return dict(leftover)

    async def fake_settle_round(round_id, tg_id, outcome, payout, status):
        calls.append(f"void:{round_id}:{payout}:{status}")
        return {"balance": 1000}

    monkeypatch.setattr(db, "get_seed_pair", fake_get_seed_pair)
    monkeypatch.setattr(db, "open_round", fake_open_round)
    monkeypatch.setattr(db, "get_open_round", fake_get_open_round)
    monkeypatch.setattr(db, "settle_round", fake_settle_round)

    r = _bet(client)
    assert r.status_code == 200 and r.json().get("error") is None
    assert calls == ["get_open_round", "void:old-1:100:voided",
                     "open_round_retry"]


def test_leftover_open_round_exists_voided_then_retry_succeeds(client, monkeypatch):
    """Leftover + SUFFICIENT balance: the debit succeeds, the unique open-round
    index fires open_round_exists, and the loop voids + retries."""
    calls = []
    attempts = {"n": 0}
    leftover = {"id": "old-2", "bet": 60, "outcome": {}}

    async def fake_get_seed_pair(tg_id):
        return dict(PAIR)

    async def fake_open_round(tg_id, game, bet, nonce, params, outcome):
        attempts["n"] += 1
        if attempts["n"] == 1:
            raise db.OpenRoundExists("open_round_exists")
        calls.append("open_round_retry")
        return dict(OPEN_RESULT)

    async def fake_get_open_round(tg_id, game):
        calls.append("get_open_round")
        return dict(leftover)

    async def fake_settle_round(round_id, tg_id, outcome, payout, status):
        calls.append(f"void:{round_id}:{payout}:{status}")
        return {"balance": 1000}

    monkeypatch.setattr(db, "get_seed_pair", fake_get_seed_pair)
    monkeypatch.setattr(db, "open_round", fake_open_round)
    monkeypatch.setattr(db, "get_open_round", fake_get_open_round)
    monkeypatch.setattr(db, "settle_round", fake_settle_round)

    r = _bet(client)
    assert r.status_code == 200 and r.json().get("error") is None
    assert calls == ["get_open_round", "void:old-2:60:voided",
                     "open_round_retry"]


def test_unclearable_leftover_is_bounded_and_returns_try_again(client, monkeypatch):
    """If the leftover can never be cleared (void keeps failing / round keeps
    reappearing), the loop must terminate after its bounded retries with
    try_again — never spin forever."""
    open_attempts = {"n": 0}
    void_attempts = {"n": 0}
    leftover = {"id": "stuck-1", "bet": 100, "outcome": {}}

    async def fake_get_seed_pair(tg_id):
        return dict(PAIR)

    async def fake_open_round(tg_id, game, bet, nonce, params, outcome):
        open_attempts["n"] += 1
        raise db.OpenRoundExists("open_round_exists")

    async def fake_get_open_round(tg_id, game):
        return dict(leftover)

    async def fake_settle_round(round_id, tg_id, outcome, payout, status):
        void_attempts["n"] += 1
        raise RuntimeError("db down")  # _void_open_round swallows this

    monkeypatch.setattr(db, "get_seed_pair", fake_get_seed_pair)
    monkeypatch.setattr(db, "open_round", fake_open_round)
    monkeypatch.setattr(db, "get_open_round", fake_get_open_round)
    monkeypatch.setattr(db, "settle_round", fake_settle_round)

    r = _bet(client)
    assert r.status_code == 409
    assert r.json()["error"] == "try_again"
    assert open_attempts["n"] == 4  # bounded: exactly the loop's max attempts
    assert void_attempts["n"] == 4  # one void attempt per retry, no runaway

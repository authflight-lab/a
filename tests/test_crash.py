"""Endpoint/integration tests for the crash game (task 3).

Crash is server-clocked: the bet anchors t0_ms and the round autonomously
crashes once e^(GROWTH * elapsed) reaches the seeded crash point. These tests
pin the settlement semantics at the API boundary:

- /bet never returns the crash point (open-round secrecy contract);
- /cashout claim validation (m < 1, non-numeric, non-finite -> invalid_move);
- claim >= crash point busts; claim below it (while the clock is live) wins;
- a claim ahead of the server clock is clamped to the clock's multiplier;
- once the server clock passes the crash instant, EVERY claim busts;
- /crash/check settles a crashed round autonomously (payout 0, crash point
  revealed) and is idempotent — the round cannot be cashed out afterwards.

Uses the FakeDB from test_seed_endpoints (faithful bt_settle_round semantics).
"""

import pytest
from fastapi.testclient import TestClient

from api import db
from api.auth import require_user
from api.game import crash
from api.main import app, _payout
from .test_seed_endpoints import FakeDB, _sha

SS, CS = "s" * 64, "c" * 64
CP0 = crash.crash_point(SS, CS, 0)  # deterministic: ~5.3448


@pytest.fixture()
def client(monkeypatch):
    from api import main as _main
    _main._ROUND_CACHE.clear()
    fake = FakeDB()
    for name in ("get_user", "upsert_user", "apply_ledger", "get_seed_pair",
                 "create_seed_pair", "open_round", "rotate_seed_pair",
                 "get_open_round", "get_round", "update_round", "update_open_round",
                 "close_round", "settle_round"):
        monkeypatch.setattr(db, name, getattr(fake, name))
    app.dependency_overrides[require_user] = lambda: {
        "tg_id": 1, "user": {"id": 1}, "username": "u", "display_name": "U",
    }
    c = TestClient(app)
    c.fake = fake
    yield c
    app.dependency_overrides.clear()


def _bet(client, bet=100):
    """Open a crash round on the known seed pair; nonce 0 -> crash point CP0."""
    client.get("/bt/api/game/seeds")  # bootstrap user + pair
    client.fake.pairs[1].update({
        "server_seed": SS, "server_hash": _sha(SS), "client_seed": CS, "nonce": 0,
    })
    r = client.post("/bt/api/game/crash/bet", json={"bet": bet, "params": {}})
    body = r.json()
    assert r.status_code == 200 and body.get("round_id"), body
    return body


def _shift_clock(client, rid, elapsed_ms):
    """Rewind the round's t0_ms so the server clock reads ``elapsed_ms`` elapsed.

    The cached round and the fake DB row may share the outcome dict, so the new
    value is ASSIGNED (not decremented) to both to avoid double-shifting.
    """
    from api import main as _main
    import time as _time
    t0 = int(_time.time() * 1000 - elapsed_ms)
    client.fake.rounds[rid]["outcome"]["t0_ms"] = t0
    cached = _main._ROUND_CACHE.get(rid)
    if cached is not None:
        cached["outcome"]["t0_ms"] = t0


def _elapsed_for(mult):
    """Server-clock milliseconds at which the curve reads ``mult``."""
    return crash.crash_ms(mult)


# ---------------------------------------------------------------------------
# Secrecy: the crash point must never leak while the round is open
# ---------------------------------------------------------------------------

def test_bet_and_live_check_never_reveal_crash_point(client):
    b = _bet(client)
    assert "crash_point" not in str(b), b  # /bet: hash + nonce only
    assert "outcome" not in b
    r = client.post("/bt/api/game/crash/check", json={"round_id": b["round_id"]}).json()
    assert r["crashed"] is False
    assert "crash_point" not in str(r), r
    # While alive, check returns only the server-clock multiplier (~1.0 here).
    assert 1.0 <= r["multiplier"] < 1.05


# ---------------------------------------------------------------------------
# Claim validation
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("claim", [0.5, 0.999, -3, "abc", "Infinity", "NaN", None])
def test_invalid_claims_rejected(client, claim):
    b = _bet(client)
    r = client.post("/bt/api/game/crash/cashout",
                    json={"round_id": b["round_id"], "mult_at_cashout": claim})
    assert r.json()["error"] == "invalid_move"
    # The round is still open — an invalid claim must not settle anything.
    assert client.fake.rounds[b["round_id"]]["status"] == "open"


# ---------------------------------------------------------------------------
# Settlement semantics
# ---------------------------------------------------------------------------

def test_claim_at_or_above_crash_point_busts(client):
    bet = 100
    b = _bet(client, bet)
    _shift_clock(client, b["round_id"], _elapsed_for(2.0))  # clock live, below cp
    r = client.post("/bt/api/game/crash/cashout",
                    json={"round_id": b["round_id"], "mult_at_cashout": CP0}).json()
    assert r["payout"] == 0
    assert r["outcome"]["busted"] is True
    assert r["outcome"]["crash_point"] == pytest.approx(CP0)
    assert client.fake.rounds[b["round_id"]]["status"] == "settled"
    assert client.fake.users[1]["balance"] == 1000 - bet


def test_honest_claim_below_crash_point_wins(client):
    bet = 100
    b = _bet(client, bet)
    _shift_clock(client, b["round_id"], _elapsed_for(3.0))  # server curve at ~3x
    r = client.post("/bt/api/game/crash/cashout",
                    json={"round_id": b["round_id"], "mult_at_cashout": 2.5}).json()
    assert r["payout"] == _payout(bet, 2.5)
    assert r["outcome"]["multiplier"] == pytest.approx(2.5)
    assert client.fake.rounds[b["round_id"]]["status"] == "cashed_out"
    assert client.fake.users[1]["balance"] == 1000 - bet + _payout(bet, 2.5)


def test_claim_ahead_of_server_clock_is_clamped(client):
    # An instant claim of 4x (below cp, but way ahead of the ~1.0x server
    # clock) must win only the clock's value — timing fraud pays nothing.
    bet = 100
    b = _bet(client, bet)
    r = client.post("/bt/api/game/crash/cashout",
                    json={"round_id": b["round_id"], "mult_at_cashout": 4.0}).json()
    assert client.fake.rounds[b["round_id"]]["status"] == "cashed_out"
    assert r["outcome"]["multiplier"] < 1.05
    assert r["payout"] <= _payout(bet, 1.05)


def test_cashout_after_crash_instant_busts_any_claim(client):
    bet = 100
    b = _bet(client, bet)
    _shift_clock(client, b["round_id"], _elapsed_for(CP0) + 200)  # crash passed
    r = client.post("/bt/api/game/crash/cashout",
                    json={"round_id": b["round_id"], "mult_at_cashout": 1.5}).json()
    assert r["payout"] == 0
    assert r["outcome"]["busted"] is True
    assert r["outcome"]["crash_point"] == pytest.approx(CP0)
    assert client.fake.users[1]["balance"] == 1000 - bet


# ---------------------------------------------------------------------------
# Autonomous crash via /crash/check
# ---------------------------------------------------------------------------

def test_check_settles_crashed_round_autonomously(client):
    bet = 100
    b = _bet(client, bet)
    rid = b["round_id"]
    _shift_clock(client, rid, _elapsed_for(CP0) + 200)
    r = client.post("/bt/api/game/crash/check", json={"round_id": rid}).json()
    assert r["crashed"] is True
    assert r["payout"] == 0
    assert r["outcome"]["crash_point"] == pytest.approx(CP0)
    assert client.fake.rounds[rid]["status"] == "settled"
    assert client.fake.users[1]["balance"] == 1000 - bet
    # Idempotent: the round is closed for every later actor.
    again = client.post("/bt/api/game/crash/check", json={"round_id": rid}).json()
    assert again["error"] == "round_not_open"
    cash = client.post("/bt/api/game/crash/cashout",
                       json={"round_id": rid, "mult_at_cashout": 1.5}).json()
    assert cash["error"] == "round_not_open"
    assert client.fake.users[1]["balance"] == 1000 - bet  # nothing double-paid


def test_step_and_settle_reject_crash(client):
    b = _bet(client)
    s = client.post("/bt/api/game/crash/step",
                    json={"round_id": b["round_id"], "move": {}}).json()
    assert s["error"] == "invalid_action"
    st = client.post("/bt/api/game/crash/settle",
                     json={"round_id": b["round_id"]}).json()
    assert st["error"] == "invalid_action"

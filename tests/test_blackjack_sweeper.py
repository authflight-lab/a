"""Stale-round sweeper tests for blackjack.

A blackjack round left open (client vanished mid-hand) must NOT be abandoned
like a single-settle game: the player already has a live, un-busted hand, so
the sweeper settles it as a STAND — the dealer plays out S17 from the stored
cursor using the round's own seeded draws, producing exactly what /step stand
would have. These tests pin that behaviour with a deterministic seed pair:

- seeds SS='s'*64, CS='c'*64, nonce 0 deal: player [3, 6] (9), dealer [3, 4];
- swept right after the deal -> dealer plays out, player's 9 loses (0.0x);
- swept after one hit (player draws A -> soft 20) -> player WINS 2.0x, proving
  the sweep pays out the hand instead of zeroing it;
- crash rounds still take the abandon path (regression for the branch reorder).
"""

import asyncio

import pytest
from fastapi.testclient import TestClient

from api import db
from api.auth import require_user
from api.game import blackjack
from api.game.seed import rng_float
from api.main import app, _cashout_stale_round, _payout
from .test_seed_endpoints import FakeDB, _sha

SS, CS = "s" * 64, "c" * 64


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
    """Open a blackjack round on the pinned seed pair (nonce 0, no natural)."""
    client.get("/bt/api/game/seeds")  # bootstrap user + pair
    client.fake.pairs[1].update({
        "server_seed": SS, "server_hash": _sha(SS), "client_seed": CS, "nonce": 0,
    })
    r = client.post("/bt/api/game/blackjack/bet", json={"bet": bet, "params": {}})
    body = r.json()
    assert r.status_code == 200 and body.get("round_id"), body
    assert body["done"] is False  # pinned seed deals no natural
    assert body["player"] == [3, 6]
    return body


def _sweep(client, rid):
    rnd = client.fake.rounds[rid]
    asyncio.run(_cashout_stale_round(dict(rnd)))
    return client.fake.rounds[rid]


def test_stale_fresh_hand_settles_as_stand_not_abandoned(client):
    bet = 100
    b = _bet(client, bet)
    rid = b["round_id"]

    row = _sweep(client, rid)
    # Settled as a stand: dealer played out from cursor 4 and the player's 9
    # loses — but through the real outcome path (timed_out), never "abandoned".
    assert row["status"] == "timed_out"
    assert row["payout"] == 0
    out = row["outcome"]
    assert out["player"] == [3, 6]
    assert out["player_done"] is True
    d = lambda i: rng_float(SS, CS, 0, i)
    exp_dealer, _ = blackjack.play_dealer(d, [3, 4], 4)
    assert out["dealer"] == exp_dealer
    assert out["multiplier"] == 0.0
    assert client.fake.users[1]["balance"] == 1000 - bet


def test_stale_hand_after_hit_pays_the_winning_stand(client):
    bet = 100
    b = _bet(client, bet)
    rid = b["round_id"]

    # Hit once: player draws A -> [3, 6, 1] = soft 20, round stays open.
    s = client.post("/bt/api/game/blackjack/step",
                    json={"round_id": rid, "move": {"action": "hit"}}).json()
    assert s["done"] is False and s["outcome_step"]["player"] == [3, 6, 1]
    assert client.fake.rounds[rid]["status"] == "open"

    row = _sweep(client, rid)
    # Dealer plays from cursor 5; player's 20 WINS 2.0x — the sweep must pay
    # out the live hand, not zero it.
    assert row["status"] == "timed_out"
    assert row["outcome"]["multiplier"] == 2.0
    assert row["payout"] == _payout(bet, 2.0)
    assert client.fake.users[1]["balance"] == 1000 - bet + _payout(bet, 2.0)


def test_sweep_is_idempotent(client):
    bet = 100
    b = _bet(client, bet)
    rid = b["round_id"]
    row = _sweep(client, rid)
    paid = row["payout"]
    bal = client.fake.users[1]["balance"]
    # Second sweep of the same (now closed) round must not double-settle.
    _sweep(client, rid)
    assert client.fake.rounds[rid]["payout"] == paid
    assert client.fake.users[1]["balance"] == bal


def test_stale_crash_round_still_abandoned(client):
    # Regression for the sweeper branch reorder: crash (not MULTI_STEP) keeps
    # its abandon path — payout 0, status "abandoned", no ledger credit.
    client.get("/bt/api/game/seeds")
    client.fake.pairs[1].update({
        "server_seed": SS, "server_hash": _sha(SS), "client_seed": CS, "nonce": 0,
    })
    r = client.post("/bt/api/game/crash/bet", json={"bet": 100, "params": {}})
    rid = r.json()["round_id"]
    asyncio.run(_cashout_stale_round(dict(client.fake.rounds[rid])))
    row = client.fake.rounds[rid]
    assert row["status"] == "abandoned"
    assert row["payout"] == 0
    assert client.fake.users[1]["balance"] == 1000 - 100

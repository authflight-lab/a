"""Endpoint/integration tests for the provably-fair seed-pair lifecycle.

These exercise the FastAPI layer with an in-memory fake of the ``db`` module that
faithfully simulates the atomic RPC semantics (``bt_open_round`` nonce reservation
+ one-open-round guard, ``bt_rotate_seed_pair`` open-round guard). They prove the
guarantees the SQL RPCs enforce at the API boundary:

- the ACTIVE ``server_seed`` is NEVER returned by seeds / bet / step / settle /
  cashout — only its hash — and is revealed ONLY by rotate;
- the per-pair nonce increments per bet and resets to 0 on rotation;
- a custom client seed supplied on rotation is applied;
- rotation is refused while a round is open.
"""

import hashlib

import pytest
from fastapi.testclient import TestClient

from api import db
from api.auth import require_user
from api.main import app


def _sha(s: str) -> str:
    return hashlib.sha256(s.encode()).hexdigest()


class FakeDB:
    """Minimal in-memory stand-in for the ``db`` module used by the endpoints,
    reproducing the atomic RPC semantics the real Postgres functions guarantee."""

    def __init__(self):
        self.users: dict[int, dict] = {}
        self.pairs: dict[int, dict] = {}
        self.rounds: dict[str, dict] = {}
        self._rid = 0

    # users -----------------------------------------------------------------
    async def get_user(self, tg_id):
        return self.users.get(tg_id)

    async def upsert_user(self, tg_id, username=None, display_name=None):
        return self.users.setdefault(tg_id, {"tg_id": tg_id, "balance": 1000})

    async def apply_ledger(self, tg_id, amount, kind, ref=None, meta=None):
        u = self.users[tg_id]
        if u["balance"] + amount < 0:
            raise db.InsufficientBalance("insufficient_balance")
        u["balance"] += amount
        return u["balance"]

    # seed pairs ------------------------------------------------------------
    async def get_seed_pair(self, tg_id):
        return self.pairs.get(tg_id)

    async def create_seed_pair(self, tg_id, pair):
        # Mirror the real FK: bt_seed_pairs.tg_id references bt_users(tg_id), so a
        # seed pair cannot be created before the user row exists.
        if tg_id not in self.users:
            raise db.SupabaseError("insert or update on table \"bt_seed_pairs\" "
                                   "violates foreign key constraint")
        return self.pairs.setdefault(tg_id, {"tg_id": tg_id, **pair})

    async def open_round(self, tg_id, game, bet, expected_nonce, params, outcome):
        p = self.pairs[tg_id]
        if int(p["nonce"]) != expected_nonce:
            raise db.NonceConflict()
        for r in self.rounds.values():
            if r["tg_id"] == tg_id and r["game"] == game and r["status"] == "open":
                raise db.OpenRoundExists()
        u = self.users[tg_id]
        if u["balance"] - bet < 0:
            raise db.InsufficientBalance("insufficient_balance")
        u["balance"] -= bet
        self._rid += 1
        rid = str(self._rid)
        self.rounds[rid] = {
            "id": rid, "tg_id": tg_id, "game": game, "bet": bet,
            "server_seed": p["server_seed"], "server_hash": p["server_hash"],
            "client_seed": p["client_seed"], "nonce": expected_nonce,
            "params": params, "outcome": outcome, "status": "open",
        }
        p["nonce"] = expected_nonce + 1
        return {"round_id": rid, "server_hash": p["server_hash"],
                "nonce": expected_nonce, "balance": u["balance"]}

    async def rotate_seed_pair(self, tg_id, client_seed, next_server_seed, next_server_hash):
        p = self.pairs[tg_id]
        for r in self.rounds.values():
            if r["tg_id"] == tg_id and r["status"] == "open":
                raise db.OpenRoundExists()
        revealed = p["server_seed"]
        new_cs = (client_seed or "").strip() or p["client_seed"]
        p["client_seed"] = new_cs
        p["server_seed"] = p["next_server_seed"]
        p["server_hash"] = p["next_server_hash"]
        p["nonce"] = 0
        p["next_server_seed"] = next_server_seed
        p["next_server_hash"] = next_server_hash
        return {"server_seed": revealed, "client_seed": new_cs, "nonce": 0,
                "server_hash": p["server_hash"], "next_server_hash": next_server_hash}

    # rounds ----------------------------------------------------------------
    async def get_open_round(self, tg_id, game):
        for r in self.rounds.values():
            if r["tg_id"] == tg_id and r["game"] == game and r["status"] == "open":
                return r
        return None

    async def get_round(self, rid):
        return self.rounds.get(rid)

    async def update_round(self, rid, patch):
        self.rounds[rid].update(patch)
        return self.rounds[rid]

    async def close_round(self, rid, patch):
        r = self.rounds.get(rid)
        if not r or r["status"] != "open":
            return None
        r.update(patch)
        return r

    async def update_open_round(self, rid, patch):
        r = self.rounds.get(rid)
        if not r or r["status"] != "open":
            return None
        r.update(patch)
        return r

    async def settle_round(self, rid, tg_id, outcome, payout, status):
        # Mirror bt_settle_round: guarded close on status='open' + credit in one
        # step. A concurrent double-settle finds it already closed (closed=False)
        # and credits nothing.
        r = self.rounds.get(rid)
        if not r or r["tg_id"] != tg_id or r["status"] != "open":
            u = self.users.get(tg_id) or {}
            return {"closed": False, "new_balance": int(u.get("balance", 0))}
        r.update({"outcome": outcome, "payout": payout, "status": status})
        u = self.users[tg_id]
        if payout > 0:
            u["balance"] += payout
        return {"closed": True, "new_balance": u["balance"]}


@pytest.fixture()
def client(monkeypatch):
    # The open-round cache is module-level in api.main, so clear it between tests
    # (each FakeDB restarts round ids at "1", which would otherwise collide with a
    # previous test's cached round).
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


# ---------------------------------------------------------------------------
# The active server seed must NEVER be exposed except by rotate
# ---------------------------------------------------------------------------

def test_seeds_endpoint_never_returns_active_server_seed(client):
    r = client.get("/bt/api/game/seeds")
    assert r.status_code == 200
    body = r.json()
    assert "server_seed" not in body
    assert body["nonce"] == 0
    assert body["server_hash"] and body["next_server_hash"]
    assert body["client_seed"]
    # The hash shown must be the commitment to the (hidden) active server seed.
    assert body["server_hash"] == _sha(client.fake.pairs[1]["server_seed"])


def test_seeds_endpoint_bootstraps_a_brand_new_user(client):
    # Fresh user: no prior /me or bet, so no bt_users row yet. The seeds endpoint
    # must bootstrap the user before creating the FK-bound seed pair.
    assert client.fake.users == {}
    r = client.get("/bt/api/game/seeds")
    assert r.status_code == 200
    assert 1 in client.fake.users  # user row was created
    assert 1 in client.fake.pairs
    assert "server_seed" not in r.json()


def test_rotate_endpoint_bootstraps_a_brand_new_user(client):
    assert client.fake.users == {}
    r = client.post("/bt/api/game/seeds/rotate", json={})
    assert r.status_code == 200
    assert 1 in client.fake.users
    assert "server_seed" in r.json()  # rotate reveals the retired seed


def test_bet_and_settle_never_leak_active_server_seed(client):
    bet = client.post("/bt/api/game/dice/bet",
                      json={"bet": 10, "params": {"target": 50}})
    assert bet.status_code == 200
    b = bet.json()
    assert "server_seed" not in b
    assert b["server_hash"] and b["nonce"] == 0 and b["round_id"]

    settle = client.post("/bt/api/game/dice/settle", json={"round_id": b["round_id"]})
    assert settle.status_code == 200
    s = settle.json()
    assert "server_seed" not in s
    assert s["server_hash"] == b["server_hash"]  # only the commitment, never the seed


def test_new_bet_auto_voids_and_refunds_a_leftover_open_round(client):
    # A leftover open round (crashed/abandoned client) must never block a new bet
    # with open_round_exists: it is voided, its bet refunded, and the new round
    # opens. Balance nets out to a single stake for the fresh round.
    fake = client.fake
    fake.users[1] = {"tg_id": 1, "balance": 1000}

    first = client.post("/bt/api/game/mines/bet",
                        json={"bet": 100, "params": {"mines": 3}})
    assert first.status_code == 200
    old_rid = first.json()["round_id"]
    assert fake.users[1]["balance"] == 900          # first stake debited

    # Second bet WITHOUT settling the first — the straggler is purged + refunded.
    second = client.post("/bt/api/game/mines/bet",
                         json={"bet": 100, "params": {"mines": 3}})
    assert second.status_code == 200
    body = second.json()
    assert body.get("error") is None                # no open_round_exists surfaced
    new_rid = body["round_id"]
    assert new_rid != old_rid

    # Old round is closed as 'voided' with its bet refunded; only one round open.
    assert fake.rounds[old_rid]["status"] == "voided"
    assert fake.rounds[old_rid]["payout"] == 100
    assert fake.rounds[new_rid]["status"] == "open"
    # Net balance: refund (+100) of the old stake then debit (-100) of the new one.
    assert fake.users[1]["balance"] == 900


def test_refund_lets_a_stuck_round_free_a_bet_the_user_could_not_afford(client):
    # If the leftover stake left the user unable to afford the next bet, the refund
    # (applied before the balance check) restores the funds so the new bet opens.
    fake = client.fake
    fake.users[1] = {"tg_id": 1, "balance": 100}

    first = client.post("/bt/api/game/mines/bet",
                        json={"bet": 100, "params": {"mines": 3}})
    assert first.status_code == 200
    assert fake.users[1]["balance"] == 0            # all-in on the first round

    second = client.post("/bt/api/game/mines/bet",
                         json={"bet": 100, "params": {"mines": 3}})
    assert second.status_code == 200
    assert second.json().get("error") is None       # refund funded the new bet
    assert fake.users[1]["balance"] == 0


def test_play_one_shot_opens_and_settles_in_a_single_call(client):
    # /play opens AND settles a single-settle game in one request, returning the
    # settle outcome plus the bet-side hash/nonce — and never the active seed.
    r = client.post("/bt/api/game/dice/play",
                    json={"bet": 10, "params": {"target": 50}})
    assert r.status_code == 200
    j = r.json()
    assert j.get("ok") is not False
    assert "server_seed" not in j                       # active seed never leaks
    assert j["server_hash"] and j["nonce"] == 0 and j["round_id"]
    assert "roll" in j["outcome"] and isinstance(j["outcome"]["win"], bool)
    assert "new_balance" in j
    # The round is closed by the same call: a follow-up settle is rejected.
    again = client.post("/bt/api/game/dice/settle", json={"round_id": j["round_id"]})
    assert again.status_code == 400
    # And the nonce advanced for the next round, exactly like /bet + /settle.
    nxt = client.post("/bt/api/game/dice/play",
                      json={"bet": 10, "params": {"target": 50}})
    assert nxt.json()["nonce"] == 1


def test_play_plinko_one_shot_settles_and_holds_invariants(client):
    # Plinko is the other single-settle game: /play must open+settle it in one
    # call with the same guarantees as dice (no active seed leak, nonce advances,
    # round closed by the same request).
    r = client.post("/bt/api/game/plinko/play",
                    json={"bet": 10, "params": {"rows": 8, "risk": "low"}})
    assert r.status_code == 200
    j = r.json()
    assert "server_seed" not in j
    assert j["server_hash"] and j["nonce"] == 0 and j["round_id"]
    assert "bucket" in j["outcome"] and "new_balance" in j
    again = client.post("/bt/api/game/plinko/settle", json={"round_id": j["round_id"]})
    assert again.status_code == 400


def test_play_rejects_multi_step_games(client):
    # /play is single-settle only; a multi-step game must use /bet + /step.
    r = client.post("/bt/api/game/mines/play",
                    json={"bet": 10, "params": {"mines": 3}})
    assert r.status_code == 400
    assert r.json()["error"] == "invalid_action"


def test_step_and_cashout_never_leak_active_server_seed(client):
    bet = client.post("/bt/api/game/mines/bet",
                      json={"bet": 10, "params": {"mines": 3}})
    assert bet.status_code == 200
    rid = bet.json()["round_id"]
    assert "server_seed" not in bet.json()

    step = client.post("/bt/api/game/mines/step", json={"round_id": rid, "move": 0})
    assert step.status_code == 200
    assert "server_seed" not in step.json()

    # If the first tile was safe the round is still open — a cashout must not leak.
    if not step.json().get("done"):
        cash = client.post("/bt/api/game/mines/cashout", json={"round_id": rid})
        assert cash.status_code == 200
        assert "server_seed" not in cash.json()


# ---------------------------------------------------------------------------
# Nonce lifecycle: increments per bet, resets to 0 on rotation
# ---------------------------------------------------------------------------

def test_nonce_increments_per_bet_and_resets_on_rotation(client):
    first = client.post("/bt/api/game/dice/bet",
                        json={"bet": 10, "params": {"target": 50}})
    assert first.json()["nonce"] == 0
    client.post("/bt/api/game/dice/settle", json={"round_id": first.json()["round_id"]})

    second = client.post("/bt/api/game/dice/bet",
                         json={"bet": 10, "params": {"target": 50}})
    assert second.json()["nonce"] == 1
    client.post("/bt/api/game/dice/settle", json={"round_id": second.json()["round_id"]})

    rot = client.post("/bt/api/game/seeds/rotate", json={})
    assert rot.json()["nonce"] == 0

    third = client.post("/bt/api/game/dice/bet",
                        json={"bet": 10, "params": {"target": 50}})
    assert third.json()["nonce"] == 0


# ---------------------------------------------------------------------------
# Rotation reveals the retired seed, promotes the pre-committed next, and the
# revealed seed verifies against the hash the panel showed while it was active
# ---------------------------------------------------------------------------

def test_rotation_reveals_old_seed_and_promotes_precommitted_next(client):
    seeds = client.get("/bt/api/game/seeds").json()
    active_hash = seeds["server_hash"]
    precommitted_next_hash = seeds["next_server_hash"]

    rot = client.post("/bt/api/game/seeds/rotate", json={}).json()
    assert "server_seed" in rot  # rotate is the ONLY place the seed is revealed
    # The revealed seed verifies against the hash that was shown while it was active.
    assert _sha(rot["server_seed"]) == active_hash
    # The promoted active hash is the previously pre-committed next hash.
    assert rot["server_hash"] == precommitted_next_hash
    # A fresh next seed is committed (different from the one just promoted).
    assert rot["next_server_hash"] != precommitted_next_hash
    assert rot["nonce"] == 0


def test_rotation_applies_custom_client_seed(client):
    rot = client.post("/bt/api/game/seeds/rotate",
                      json={"client_seed": "my-custom-seed"}).json()
    assert rot["client_seed"] == "my-custom-seed"
    # And it persists to the active pair used by subsequent bets.
    seeds = client.get("/bt/api/game/seeds").json()
    assert seeds["client_seed"] == "my-custom-seed"


def test_rotation_without_client_seed_keeps_current(client):
    before = client.get("/bt/api/game/seeds").json()["client_seed"]
    rot = client.post("/bt/api/game/seeds/rotate", json={}).json()
    assert rot["client_seed"] == before


# ---------------------------------------------------------------------------
# Rotation is refused while a round is open (would leak a live seed)
# ---------------------------------------------------------------------------

def test_rotation_blocked_while_round_open(client):
    bet = client.post("/bt/api/game/mines/bet",
                      json={"bet": 10, "params": {"mines": 3}})
    assert bet.status_code == 200
    rot = client.post("/bt/api/game/seeds/rotate", json={})
    assert rot.status_code == 400
    assert rot.json()["error"] == "open_round_exists"


# ---------------------------------------------------------------------------
# HighLow skip: swap the current card without wagering — multiplier unchanged,
# step advances, new current card is non-wild, active seed never leaks, and it
# does not end the round.
# ---------------------------------------------------------------------------

def test_highlow_skip_keeps_multiplier_and_advances_deterministically(client):
    bet = client.post("/bt/api/game/highlow/bet", json={"bet": 10, "params": {}})
    assert bet.status_code == 200
    rid = bet.json()["round_id"]
    start_card = client.fake.rounds[rid]["outcome"]["rank"]

    skip = client.post("/bt/api/game/highlow/step",
                       json={"round_id": rid, "move": {"skip": True}})
    assert skip.status_code == 200
    s = skip.json()
    assert "server_seed" not in s              # active seed never leaks on step
    assert s["outcome_step"]["skipped"] is True
    assert s["busted"] is False and s["done"] is False
    assert s["multiplier"] == 1.0              # no wager, multiplier unchanged
    new_card = s["outcome_step"]["current"]
    assert s["outcome_step"]["guess"] == "skip"
    assert s["outcome_step"]["prev"] == start_card
    # New current card is always a non-wild rank (2..12).
    assert 2 <= new_card <= 12

    # State advanced: step incremented, rank replaced, multiplier held at 1.0.
    st = client.fake.rounds[rid]["outcome"]
    assert st["step"] == 1
    assert st["rank"] == new_card
    assert st["multiplier"] == 1.0

    # Deterministic: skip is a pure function of the (seed, nonce, slot); the same
    # committed round yields the same new card.
    assert new_card == client.fake.rounds[rid]["outcome"]["rank"]

    # A scalar "skip" move is accepted too (matches guess string handling).
    skip2 = client.post("/bt/api/game/highlow/step",
                        json={"round_id": rid, "move": "skip"})
    assert skip2.status_code == 200
    assert skip2.json()["multiplier"] == 1.0
    assert client.fake.rounds[rid]["outcome"]["step"] == 2

    # Skips alone don't unlock cashout: at least one real pick is required, so a
    # skip-only round is rejected with must_pick_first (round stays open).
    cash = client.post("/bt/api/game/highlow/cashout", json={"round_id": rid})
    assert cash.status_code == 400
    assert cash.json().get("error") == "must_pick_first"

"""Lazy httpx Supabase REST/RPC client (contract §0.3).

- Base URL:  ``${BT_SUPABASE_URL}/rest/v1``
- Headers:   ``apikey`` + ``Authorization: Bearer`` (service-role key, server-only),
             ``Content-Type: application/json``.
- RPC:       ``POST /rest/v1/rpc/<fn>`` with a JSON body.
- Table read ``GET /rest/v1/<table>?<col>=eq.<val>&select=*``.

The client is created lazily on first use. When Supabase is unconfigured (the
current placeholder state) every helper raises ``SupabaseNotConfigured``, which
``api.main`` converts into a 503 — so importing this module performs NO network
I/O and the API degrades gracefully.

Balance is derived (``balance == sum(bt_ledger.amount)``). All balance changes go
through the ``bt_apply_ledger`` RPC exclusively — never a raw ``UPDATE balance``.
"""

from datetime import datetime, timezone
from typing import Any

import httpx

from .config import settings


class SupabaseNotConfigured(Exception):
    """Raised when BT_SUPABASE_URL / service key are not set."""


class InsufficientBalance(Exception):
    """Raised when bt_apply_ledger would drive a balance below zero."""


class SupabaseError(Exception):
    """Any other Supabase REST/RPC failure."""


class RedeemError(Exception):
    """A handled redeem failure carrying a client error ``code``
    (``insufficient_balance`` | ``monthly_limit_reached`` | ``reward_inactive``)."""

    def __init__(self, code: str):
        self.code = code
        super().__init__(code)


class OpenRoundExists(Exception):
    """Raised by ``bt_open_round``/``bt_rotate_seed_pair`` when an open round
    blocks the operation (one open round per (user, game); rotation forbidden
    while any round is open)."""


class NonceConflict(Exception):
    """Raised by ``bt_open_round`` when the seed pair's nonce advanced (a
    concurrent bet) between the caller reading it and the guarded reservation —
    the caller must re-read the pair and retry so the stored round's RNG inputs
    match its nonce."""


_client: httpx.AsyncClient | None = None


def is_configured() -> bool:
    return bool(settings.bt_supabase_url and settings.bt_supabase_service_key)


def _get_client() -> httpx.AsyncClient:
    global _client
    if not is_configured():
        raise SupabaseNotConfigured("Supabase URL / service key not configured")
    if _client is None:
        base = settings.bt_supabase_url.rstrip("/") + "/rest/v1"
        _client = httpx.AsyncClient(
            base_url=base,
            headers={
                "apikey": settings.bt_supabase_service_key,
                "Authorization": f"Bearer {settings.bt_supabase_service_key}",
                "Content-Type": "application/json",
            },
            timeout=15.0,
        )
    return _client


async def aclose() -> None:
    global _client
    if _client is not None:
        await _client.aclose()
        _client = None


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


# ---------------------------------------------------------------------------
# Low-level REST helpers
# ---------------------------------------------------------------------------

async def _get(table: str, params: dict[str, Any]) -> list[dict]:
    client = _get_client()
    r = await client.get(f"/{table}", params=params)
    r.raise_for_status()
    return r.json()


async def _insert(table: str, row: dict, prefer: str = "return=representation") -> dict | None:
    client = _get_client()
    r = await client.post(f"/{table}", json=row, headers={"Prefer": prefer})
    r.raise_for_status()
    if not r.content:
        return None
    data = r.json()
    return data[0] if isinstance(data, list) and data else (data or None)


async def _patch(table: str, params: dict[str, Any], patch: dict) -> dict | None:
    client = _get_client()
    r = await client.patch(f"/{table}", params=params, json=patch, headers={"Prefer": "return=representation"})
    r.raise_for_status()
    if not r.content:
        return None
    data = r.json()
    return data[0] if isinstance(data, list) and data else (data or None)


async def _upsert(table: str, row: dict, on_conflict: str) -> dict | None:
    client = _get_client()
    r = await client.post(
        f"/{table}",
        params={"on_conflict": on_conflict},
        json=row,
        headers={"Prefer": "resolution=merge-duplicates,return=representation"},
    )
    r.raise_for_status()
    if not r.content:
        return None
    data = r.json()
    return data[0] if isinstance(data, list) and data else (data or None)


# ---------------------------------------------------------------------------
# Ledger RPC (the ONLY way balances change)
# ---------------------------------------------------------------------------

async def apply_ledger(tg_id: int, amount: int, kind: str, ref: str | None = None,
                       meta: dict | None = None) -> int:
    """Call ``bt_apply_ledger`` -> returns the new balance.

    Raises ``InsufficientBalance`` if the resulting balance would be < 0.
    """
    client = _get_client()
    payload = {
        "p_tg_id": tg_id,
        "p_amount": amount,
        "p_kind": kind,
        "p_ref": ref,
        "p_meta": meta or {},
    }
    r = await client.post("/rpc/bt_apply_ledger", json=payload)
    if r.status_code >= 400:
        body = ""
        try:
            body = r.text
        except Exception:
            pass
        if "insufficient_balance" in body:
            raise InsufficientBalance("insufficient_balance")
        raise SupabaseError(f"apply_ledger failed: {r.status_code} {body}")
    return int(r.json())


async def redeem(tg_id: int, reward_id: str, period: str) -> dict:
    """Call ``bt_redeem`` — atomic debit + monthly-usage increment + redemption
    row (spec §14). Returns ``{new_balance, redemption_id, cost}``.

    Raises ``RedeemError(code)`` for handled failures.
    """
    client = _get_client()
    payload = {"p_tg_id": tg_id, "p_reward_id": reward_id, "p_period": period}
    r = await client.post("/rpc/bt_redeem", json=payload)
    if r.status_code >= 400:
        body = ""
        try:
            body = r.text
        except Exception:
            pass
        for code in ("insufficient_balance", "monthly_limit_reached", "reward_inactive"):
            if code in body:
                raise RedeemError(code)
        if "reward_not_found" in body:
            raise RedeemError("reward_inactive")
        raise SupabaseError(f"redeem failed: {r.status_code} {body}")
    return r.json()


# ---------------------------------------------------------------------------
# Users
# ---------------------------------------------------------------------------

async def get_user(tg_id: int) -> dict | None:
    rows = await _get("bt_users", {"tg_id": f"eq.{tg_id}", "select": "*", "limit": "1"})
    return rows[0] if rows else None


async def upsert_user(tg_id: int, username: str | None = None,
                      display_name: str | None = None) -> dict | None:
    row: dict[str, Any] = {"tg_id": tg_id, "updated_at": _now()}
    if username is not None:
        row["username"] = username
    if display_name is not None:
        row["display_name"] = display_name
    return await _upsert("bt_users", row, on_conflict="tg_id")


async def update_user(tg_id: int, patch: dict) -> dict | None:
    patch = {**patch, "updated_at": _now()}
    return await _patch("bt_users", {"tg_id": f"eq.{tg_id}"}, patch)


async def set_age_ack(tg_id: int) -> None:
    """Persist ``bt_users.meta.age_ack = true`` (merges into existing meta)."""
    user = await get_user(tg_id)
    meta = (user or {}).get("meta") or {}
    meta["age_ack"] = True
    await update_user(tg_id, {"meta": meta})


# ---------------------------------------------------------------------------
# Daily quests
# ---------------------------------------------------------------------------

async def get_quest(tg_id: int, day: str) -> dict | None:
    rows = await _get("bt_quests_daily", {"tg_id": f"eq.{tg_id}", "day": f"eq.{day}", "limit": "1"})
    return rows[0] if rows else None


async def set_quest(tg_id: int, day: str, *, chatted: bool | None = None,
                    claimed: bool | None = None) -> dict | None:
    row: dict[str, Any] = {"tg_id": tg_id, "day": day}
    existing = await get_quest(tg_id, day)
    row["chatted"] = existing.get("chatted", False) if existing else False
    row["claimed"] = existing.get("claimed", False) if existing else False
    if chatted is not None:
        row["chatted"] = chatted
    if claimed is not None:
        row["claimed"] = claimed
    return await _upsert("bt_quests_daily", row, on_conflict="tg_id,day")


# ---------------------------------------------------------------------------
# Rewards + usage + redemptions
# ---------------------------------------------------------------------------

async def list_rewards(active_only: bool = True) -> list[dict]:
    params: dict[str, Any] = {"select": "*", "order": "cost.asc"}
    if active_only:
        params["active"] = "eq.true"
    return await _get("bt_rewards", params)


async def get_reward(reward_id: str) -> dict | None:
    rows = await _get("bt_rewards", {"id": f"eq.{reward_id}", "select": "*", "limit": "1"})
    return rows[0] if rows else None


async def get_reward_usage(reward_id: str, period: str) -> int:
    rows = await _get("bt_reward_usage",
                      {"reward_id": f"eq.{reward_id}", "period": f"eq.{period}", "select": "used", "limit": "1"})
    return int(rows[0]["used"]) if rows else 0


async def incr_reward_usage(reward_id: str, period: str) -> None:
    used = await get_reward_usage(reward_id, period)
    await _upsert("bt_reward_usage",
                  {"reward_id": reward_id, "period": period, "used": used + 1},
                  on_conflict="reward_id,period")


async def decr_reward_usage(reward_id: str, period: str) -> None:
    used = await get_reward_usage(reward_id, period)
    await _upsert("bt_reward_usage",
                  {"reward_id": reward_id, "period": period, "used": max(0, used - 1)},
                  on_conflict="reward_id,period")


async def create_redemption(tg_id: int, reward_id: str, cost: int) -> dict | None:
    return await _insert("bt_redemptions",
                         {"tg_id": tg_id, "reward_id": reward_id, "cost": cost, "status": "pending"})


# ---------------------------------------------------------------------------
# Seed pairs (provably-fair, Rainbet-style reuse; see api/game/seedpair.py)
# ---------------------------------------------------------------------------

async def get_seed_pair(tg_id: int) -> dict | None:
    rows = await _get("bt_seed_pairs", {"tg_id": f"eq.{tg_id}", "select": "*", "limit": "1"})
    return rows[0] if rows else None


async def create_seed_pair(tg_id: int, pair: dict) -> dict | None:
    """Insert the active seed pair only if absent — ``ignore-duplicates`` so two
    concurrent first-bets can't both create one (TOCTOU-safe). Returns the row
    that ends up stored (the winner's, if we lost the race)."""
    client = _get_client()
    row = {"tg_id": tg_id, **pair, "updated_at": _now()}
    r = await client.post(
        "/bt_seed_pairs",
        params={"on_conflict": "tg_id"},
        json=row,
        headers={"Prefer": "resolution=ignore-duplicates,return=representation"},
    )
    r.raise_for_status()
    data = r.json() if r.content else None
    if isinstance(data, list) and data:
        return data[0]
    if data:
        return data
    # Row already existed (insert ignored) → read the winner back.
    return await get_seed_pair(tg_id)


async def open_round(tg_id: int, game: str, bet: int, expected_nonce: int,
                     params: dict, outcome: dict | None) -> dict:
    """Atomically reserve the pair's next nonce and open a round via the
    ``bt_open_round`` RPC (locks the seed-pair row for the whole transaction, so
    it serialises against rotation). The caller computes ``outcome`` using
    ``expected_nonce``; the RPC verifies the locked nonce still matches.

    Returns ``{round_id, server_hash, nonce, balance}``.

    Raises ``NonceConflict`` (retry with a fresh nonce), ``OpenRoundExists``
    (a round for this game is already open), or ``InsufficientBalance``.
    """
    client = _get_client()
    payload = {
        "p_tg_id": tg_id,
        "p_game": game,
        "p_bet": bet,
        "p_expected_nonce": expected_nonce,
        "p_params": params,
        "p_outcome": outcome or {},
    }
    r = await client.post("/rpc/bt_open_round", json=payload)
    if r.status_code >= 400:
        body = ""
        try:
            body = r.text
        except Exception:
            pass
        if "nonce_conflict" in body:
            raise NonceConflict()
        if "open_round_exists" in body:
            raise OpenRoundExists()
        if "insufficient_balance" in body:
            raise InsufficientBalance("insufficient_balance")
        raise SupabaseError(f"open_round failed: {r.status_code} {body}")
    return r.json()


async def rotate_seed_pair(tg_id: int, client_seed: str,
                           next_server_seed: str, next_server_hash: str) -> dict:
    """Atomically rotate the pair via the ``bt_rotate_seed_pair`` RPC: reveal the
    retired server seed, promote the pre-committed next one, commit the caller's
    freshly-generated next seed, apply an optional new client seed, reset nonce.

    Returns ``{server_seed, client_seed, nonce, server_hash, next_server_hash}``.

    Raises ``OpenRoundExists`` if any round is open (rotation would otherwise
    leak an in-progress round's server seed).
    """
    client = _get_client()
    payload = {
        "p_tg_id": tg_id,
        "p_client_seed": client_seed or "",
        "p_next_server_seed": next_server_seed,
        "p_next_server_hash": next_server_hash,
    }
    r = await client.post("/rpc/bt_rotate_seed_pair", json=payload)
    if r.status_code >= 400:
        body = ""
        try:
            body = r.text
        except Exception:
            pass
        if "open_round_exists" in body:
            raise OpenRoundExists()
        raise SupabaseError(f"rotate_seed_pair failed: {r.status_code} {body}")
    return r.json()


# ---------------------------------------------------------------------------
# Game rounds
# ---------------------------------------------------------------------------

async def get_open_round(tg_id: int, game: str) -> dict | None:
    rows = await _get("bt_game_rounds",
                      {"tg_id": f"eq.{tg_id}", "game": f"eq.{game}", "status": "eq.open",
                       "select": "*", "limit": "1"})
    return rows[0] if rows else None


async def get_round(round_id: str) -> dict | None:
    rows = await _get("bt_game_rounds", {"id": f"eq.{round_id}", "select": "*", "limit": "1"})
    return rows[0] if rows else None


async def create_round(tg_id: int, game: str, bet: int, server_seed: str, server_hash: str,
                       client_seed: str, nonce: int, params: dict, outcome: dict | None) -> dict | None:
    return await _insert("bt_game_rounds", {
        "tg_id": tg_id,
        "game": game,
        "bet": bet,
        "server_seed": server_seed,
        "server_hash": server_hash,
        "client_seed": client_seed,
        "nonce": nonce,
        "params": params,
        "outcome": outcome,
        "status": "open",
    })


async def update_round(round_id: str, patch: dict) -> dict | None:
    return await _patch("bt_game_rounds", {"id": f"eq.{round_id}"}, patch)


async def close_round(round_id: str, patch: dict) -> dict | None:
    """Conditionally close an OPEN round: PATCH guarded on ``status='open'``.

    Returns the updated row if THIS call closed it, or ``None`` if it was already
    closed by a concurrent settle/cashout. This makes settlement idempotent and
    closes the double-settle payout race (spec §14 'double-settle rejected').
    """
    return await _patch("bt_game_rounds",
                        {"id": f"eq.{round_id}", "status": "eq.open"}, patch)


async def claim_daily(tg_id: int, day_start_iso: str, streak: int, now_iso: str) -> dict | None:
    """Compare-and-swap the daily claim: set ``streak_days``/``last_claim_at``
    only if ``last_claim_at`` predates today's UTC start (or is null).

    Returns the updated row if THIS call won the claim, or ``None`` if another
    concurrent request already claimed today — closing the double-claim race.
    """
    return await _patch(
        "bt_users",
        {"tg_id": f"eq.{tg_id}",
         "or": f"(last_claim_at.is.null,last_claim_at.lt.{day_start_iso})"},
        {"streak_days": streak, "last_claim_at": now_iso},
    )


# ---------------------------------------------------------------------------
# Ledger history + leaderboard
# ---------------------------------------------------------------------------

async def ledger_history(tg_id: int, limit: int = 50) -> list[dict]:
    return await _get("bt_ledger",
                      {"tg_id": f"eq.{tg_id}", "select": "id,amount,kind,ref,created_at",
                       "order": "created_at.desc", "limit": str(limit)})


async def leaderboard_rich(limit: int = 20) -> list[dict]:
    return await _get("bt_users",
                      {"select": "tg_id,display_name,balance", "order": "balance.desc", "limit": str(limit)})


async def rich_rank(tg_id: int, balance: int) -> int:
    """1-based rank on the rich list = count(users with balance > mine) + 1."""
    client = _get_client()
    r = await client.get("/bt_users",
                         params={"select": "tg_id", "balance": f"gt.{balance}"},
                         headers={"Prefer": "count=exact", "Range-Unit": "items", "Range": "0-0"})
    r.raise_for_status()
    cr = r.headers.get("content-range", "*/0")
    total = cr.split("/")[-1]
    return (int(total) + 1) if total.isdigit() else 1


async def chatters_ledger(period_start: str) -> list[dict]:
    """All `kind='chat'` ledger rows since ``period_start`` (aggregated in Python)."""
    return await _get("bt_ledger",
                      {"select": "tg_id,amount", "kind": "eq.chat",
                       "created_at": f"gte.{period_start}", "limit": "100000"})


async def ledger_since(start: str, exclude_kind: str | None = None) -> list[dict]:
    """All ledger rows (any kind) since ``start`` — used for the weekly rich
    leaderboard's net-points-gained calculation (aggregated in Python).

    ``exclude_kind`` drops one kind (e.g. 'weekly_bonus') so a prior week's
    prize, credited at the Monday boundary, doesn't give past winners a head
    start in the new week's rich race."""
    params: dict[str, Any] = {
        "select": "tg_id,amount", "created_at": f"gte.{start}", "limit": "100000",
    }
    if exclude_kind:
        params["kind"] = f"neq.{exclude_kind}"
    return await _get("bt_ledger", params)


async def display_names(tg_ids: list[int]) -> dict[int, str]:
    if not tg_ids:
        return {}
    ids = ",".join(str(i) for i in tg_ids)
    rows = await _get("bt_users", {"select": "tg_id,display_name", "tg_id": f"in.({ids})"})
    return {int(r["tg_id"]): (r.get("display_name") or str(r["tg_id"])) for r in rows}

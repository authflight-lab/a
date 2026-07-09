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

from datetime import date, datetime, timedelta, timezone
from decimal import Decimal
from typing import Any
from uuid import UUID

import asyncpg
import httpx

from .config import settings
from . import cache, pgpool


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
# asyncpg row normalisation — make a direct-DB row match the PostgREST JSON
# shape so pg-first and REST-fallback paths return identical dicts. asyncpg
# hands back native Python types (UUID, datetime, Decimal); we render them the
# way PostgREST would (uuid/timestamp as strings, integral numeric as int).
# jsonb columns already arrive as dict/list via the pool's type codec.
# ---------------------------------------------------------------------------

def _norm(v: Any) -> Any:
    if isinstance(v, UUID):
        return str(v)
    if isinstance(v, (datetime, date)):
        return v.isoformat()
    if isinstance(v, Decimal):
        i = int(v)
        return i if v == i else float(v)
    return v


def _row(rec: "asyncpg.Record | None") -> dict | None:
    if rec is None:
        return None
    return {k: _norm(v) for k, v in rec.items()}


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
    _invalidate_user(tg_id)
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
        for code in ("insufficient_balance", "monthly_limit_reached", "daily_limit_reached", "reward_inactive", "dev_account"):
            if code in body:
                raise RedeemError(code)
        if "reward_not_found" in body:
            raise RedeemError("reward_inactive")
        raise SupabaseError(f"redeem failed: {r.status_code} {body}")
    _invalidate_user(tg_id)
    return r.json()


# ---------------------------------------------------------------------------
# Users
# ---------------------------------------------------------------------------

# In-process cache TTLs. Kept short for the user profile so a cross-process
# balance change from the bot is never stale for long; the reward catalogue
# changes rarely so it can live longer.
_USER_CACHE_TTL = 5.0
_REWARDS_CACHE_TTL = 60.0
# Home-card stats + rich rank: display-only, so a few seconds of staleness is
# invisible while collapsing the per-/me recomputation (a COUNT over bt_users
# plus the bt_user_stats RPC) into one read per user per TTL window.
_STATS_CACHE_TTL = 10.0
# Leaderboard reads (Rich/Chatters): read-heavy, whole-board data shared by
# every caller; nobody needs it real-time to the second.
_LB_CACHE_TTL = 15.0


def _user_key(tg_id: int) -> str:
    return f"user:{tg_id}"


def _invalidate_user(tg_id: int) -> None:
    cache.invalidate(_user_key(tg_id))
    # Stats ride along with the profile: any balance/activity mutation should
    # also drop the cached home-card stats so /me reflects it promptly.
    cache.invalidate(f"stats:{tg_id}")


async def _get_user_rest(tg_id: int) -> dict | None:
    rows = await _get("bt_users", {"tg_id": f"eq.{tg_id}", "select": "*", "limit": "1"})
    return rows[0] if rows else None


async def get_user(tg_id: int) -> dict | None:
    """Read a user row (direct-DB, REST fallback on any pool/transport failure)."""
    try:
        pool = await pgpool.get_pool()
        rec = await pool.fetchrow(
            "select * from bt_users where tg_id = $1 limit 1", tg_id)
        return _row(rec)
    except Exception as e:  # noqa: BLE001 — classify then fall back or re-raise
        if pgpool.should_fallback(e):
            return await _get_user_rest(tg_id)
        raise


async def get_user_cached(tg_id: int) -> dict | None:
    """Cached read of a user row (short TTL). Use for read-only display paths
    (``/me``, rich-rank). Balance-authoritative paths (claim, bets) must call
    :func:`get_user` directly. The cache is dropped on every same-process
    balance mutation, so a user never sees their own change go stale.
    """
    key = _user_key(tg_id)
    hit = cache.get(key)
    if hit is not None:
        return hit
    row = await get_user(tg_id)
    if row is not None:
        cache.put(key, row, _USER_CACHE_TTL)
    return row


def cache_user(tg_id: int, row: dict | None) -> None:
    """Store a freshly-fetched user row (e.g. after mark_registered in /me)."""
    if row:
        cache.put(_user_key(tg_id), row, _USER_CACHE_TTL)


async def upsert_user(tg_id: int, username: str | None = None,
                      display_name: str | None = None) -> dict | None:
    row: dict[str, Any] = {"tg_id": tg_id, "updated_at": _now()}
    if username is not None:
        row["username"] = username
    if display_name is not None:
        row["display_name"] = display_name
    result = await _upsert("bt_users", row, on_conflict="tg_id")
    _invalidate_user(tg_id)
    return result


async def ensure_identity(tg_id: int, username: str | None = None,
                          display_name: str | None = None,
                          cached_row: dict | None = None) -> dict:
    """Canonical get-or-create + keep-identity-fresh for a user row.

    The single write path authenticated endpoints funnel through so identity
    stays consistent across the bot + api. Writes ONLY when the row is missing
    or a provided non-empty field actually differs from what's stored, so warm
    reads stay write-free (the /me fast-path invariant). Never clobbers a
    stored name with NULL (coalesce semantics, matching ``bt_chat_touch``).
    Returns the current row.
    """
    u = cached_row if cached_row is not None else await get_user_cached(tg_id)
    if u is None:
        u = await upsert_user(tg_id, username, display_name) or {"tg_id": tg_id}
        cache_user(tg_id, u)
        return u
    changed = (
        (bool(display_name) and display_name != u.get("display_name"))
        or (bool(username) and username != u.get("username"))
    )
    if changed:
        u = await upsert_user(
            tg_id, username or u.get("username"),
            display_name or u.get("display_name"),
        ) or u
        cache_user(tg_id, u)
    return u


async def mark_registered(tg_id: int, username: str | None = None,
                          display_name: str | None = None) -> dict | None:
    """Upsert identity and stamp ``started_at`` the first time the user opens
    the Mini App. The ``started_at=is.null`` filter makes the PATCH a no-op on
    repeat calls, so the original registration time is preserved.
    """
    row = await upsert_user(tg_id, username, display_name)
    try:
        await _patch(
            "bt_users",
            {"tg_id": f"eq.{tg_id}", "started_at": "is.null"},
            {"started_at": _now()},
        )
    except Exception:
        pass  # best-effort; never block /me on the registration stamp
    return row


async def update_user(tg_id: int, patch: dict) -> dict | None:
    patch = {**patch, "updated_at": _now()}
    result = await _patch("bt_users", {"tg_id": f"eq.{tg_id}"}, patch)
    _invalidate_user(tg_id)
    return result


async def set_age_ack(tg_id: int) -> None:
    """Persist ``bt_users.meta.age_ack = true`` in a single round trip via the
    ``bt_set_age_ack`` RPC (create-if-absent + jsonb merge)."""
    client = _get_client()
    r = await client.post("/rpc/bt_set_age_ack", json={"p_tg_id": tg_id})
    r.raise_for_status()
    _invalidate_user(tg_id)


# ---------------------------------------------------------------------------
# Daily quests
# ---------------------------------------------------------------------------

async def get_quest(tg_id: int, day: str) -> dict | None:
    rows = await _get("bt_quests_daily", {"tg_id": f"eq.{tg_id}", "day": f"eq.{day}", "limit": "1"})
    return rows[0] if rows else None


async def set_quest(tg_id: int, day: str, *, chatted: bool | None = None,
                    claimed: bool | None = None) -> dict | None:
    # merge-duplicates upsert: only the columns we pass are updated on conflict,
    # so there is no need to read the row first to preserve the other flag. On a
    # first-of-day insert the omitted flag takes its NOT NULL DEFAULT false.
    row: dict[str, Any] = {"tg_id": tg_id, "day": day}
    if chatted is not None:
        row["chatted"] = chatted
    if claimed is not None:
        row["claimed"] = claimed
    return await _upsert("bt_quests_daily", row, on_conflict="tg_id,day")


# ---------------------------------------------------------------------------
# Rewards + usage + redemptions
# ---------------------------------------------------------------------------

async def list_rewards(active_only: bool = True) -> list[dict]:
    # The catalogue changes rarely and /rewards is read-heavy, so cache the rows
    # (usage/remaining counts are still computed live by the caller).
    key = f"rewards:{'active' if active_only else 'all'}"
    hit = cache.get(key)
    if hit is not None:
        return hit
    params: dict[str, Any] = {"select": "*", "order": "cost.asc"}
    if active_only:
        params["active"] = "eq.true"
    rows = await _get("bt_rewards", params)
    cache.put(key, rows, _REWARDS_CACHE_TTL)
    return rows


async def get_reward(reward_id: str) -> dict | None:
    rows = await _get("bt_rewards", {"id": f"eq.{reward_id}", "select": "*", "limit": "1"})
    return rows[0] if rows else None


async def _get_reward_usage_rest(reward_id: str, period: str) -> int:
    rows = await _get("bt_reward_usage",
                      {"reward_id": f"eq.{reward_id}", "period": f"eq.{period}", "select": "used", "limit": "1"})
    return int(rows[0]["used"]) if rows else 0


async def get_reward_usage(reward_id: str, period: str) -> int:
    """Usage counter for one reward in ``period`` (direct-DB, REST fallback)."""
    try:
        pool = await pgpool.get_pool()
        val = await pool.fetchval(
            "select used from bt_reward_usage where reward_id = $1::uuid and period = $2 limit 1",
            str(reward_id), period)
        return int(val) if val is not None else 0
    except Exception as e:  # noqa: BLE001
        if pgpool.should_fallback(e):
            return await _get_reward_usage_rest(reward_id, period)
        raise


async def _get_reward_usages_rest(reward_ids: list[str], period: str) -> dict[str, int]:
    ids = ",".join(str(i) for i in reward_ids)
    rows = await _get("bt_reward_usage",
                      {"reward_id": f"in.({ids})", "period": f"eq.{period}",
                       "select": "reward_id,used"})
    return {str(r["reward_id"]): int(r["used"]) for r in rows}


async def get_reward_usages(reward_ids: list[str], period: str) -> dict[str, int]:
    """Usage counters for many rewards in one round-trip: ``{reward_id: used}``.

    Rewards with no usage row are simply absent (callers default to 0). This is
    the /rewards catalogue path — one query for the whole page instead of one
    round-trip per limited reward (the old N+1). Direct-DB, REST fallback.
    """
    if not reward_ids:
        return {}
    try:
        pool = await pgpool.get_pool()
        recs = await pool.fetch(
            "select reward_id, used from bt_reward_usage "
            "where period = $1 and reward_id = any($2::uuid[])",
            period, [str(i) for i in reward_ids])
        return {str(r["reward_id"]): int(r["used"]) for r in recs}
    except Exception as e:  # noqa: BLE001
        if pgpool.should_fallback(e):
            return await _get_reward_usages_rest(reward_ids, period)
        raise


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


async def has_redeemed_today(tg_id: int, day_start_iso: str) -> bool:
    """True if the user already has a non-rejected redemption since ``day_start_iso``
    (UTC midnight). Rejected redemptions are auto-refunded, so they do not count
    against the one-reward-per-day limit."""
    rows = await _get("bt_redemptions", {
        "tg_id": f"eq.{tg_id}",
        "created_at": f"gte.{day_start_iso}",
        "status": "neq.rejected",
        "select": "id",
        "limit": "1",
    })
    return bool(rows)


# ---------------------------------------------------------------------------
# Seed pairs (provably-fair, Rainbet-style reuse; see api/game/seedpair.py)
# ---------------------------------------------------------------------------

async def _get_seed_pair_rest(tg_id: int) -> dict | None:
    rows = await _get("bt_seed_pairs", {"tg_id": f"eq.{tg_id}", "select": "*", "limit": "1"})
    return rows[0] if rows else None


async def get_seed_pair(tg_id: int) -> dict | None:
    """Read the user's active seed pair (direct-DB, REST fallback)."""
    try:
        pool = await pgpool.get_pool()
        rec = await pool.fetchrow(
            "select * from bt_seed_pairs where tg_id = $1 limit 1", tg_id)
        return _row(rec)
    except Exception as e:  # noqa: BLE001
        if pgpool.should_fallback(e):
            return await _get_seed_pair_rest(tg_id)
        raise


async def _create_seed_pair_rest(tg_id: int, pair: dict) -> dict | None:
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
    return await _get_seed_pair_rest(tg_id)


async def create_seed_pair(tg_id: int, pair: dict) -> dict | None:
    """Insert the active seed pair only if absent — ``on conflict do nothing`` so
    two concurrent first-bets can't both create one (TOCTOU-safe). Returns the row
    that ends up stored (the winner's, if we lost the race). Direct-DB with REST
    fallback on any pool/transport failure."""
    data = {"tg_id": tg_id, **pair, "updated_at": datetime.now(timezone.utc)}
    cols = list(data.keys())
    placeholders = ", ".join(f"${i + 1}" for i in range(len(cols)))
    vals = [data[c] for c in cols]
    sql = (f'insert into bt_seed_pairs ({", ".join(cols)}) '
           f'values ({placeholders}) on conflict (tg_id) do nothing returning *')
    try:
        pool = await pgpool.get_pool()
        async with pool.acquire() as con:
            rec = await con.fetchrow(sql, *vals)
            if rec is None:
                # Insert ignored (row already existed) → read the winner back.
                rec = await con.fetchrow(
                    "select * from bt_seed_pairs where tg_id = $1", tg_id)
            return _row(rec)
    except Exception as e:  # noqa: BLE001
        if pgpool.should_fallback(e):
            return await _create_seed_pair_rest(tg_id, pair)
        raise


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
    try:
        pool = await pgpool.get_pool()
    except Exception as e:  # noqa: BLE001 — pool unavailable → REST
        if pgpool.should_fallback(e):
            return await _open_round_rest(tg_id, game, bet, expected_nonce, params, outcome)
        raise
    try:
        val = await pool.fetchval(
            "select bt_open_round($1, $2, $3, $4, $5, $6)",
            tg_id, game, bet, expected_nonce, params, outcome or {})
    except asyncpg.exceptions.RaiseError as re:
        # The RPC raises plain messages; map them to the same domain exceptions
        # the REST path raised from the response body.
        msg = getattr(re, "message", "") or str(re)
        if "nonce_conflict" in msg:
            raise NonceConflict()
        if "open_round_exists" in msg:
            raise OpenRoundExists()
        if "insufficient_balance" in msg:
            raise InsufficientBalance("insufficient_balance")
        raise SupabaseError(f"open_round failed: {msg}")
    except Exception as e:  # noqa: BLE001 — transport failure → REST
        if pgpool.should_fallback(e):
            return await _open_round_rest(tg_id, game, bet, expected_nonce, params, outcome)
        raise
    _invalidate_user(tg_id)
    return val


async def _open_round_rest(tg_id: int, game: str, bet: int, expected_nonce: int,
                           params: dict, outcome: dict | None) -> dict:
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
    _invalidate_user(tg_id)
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

async def get_stale_open_rounds(minutes: int = 30) -> list[dict]:
    """Return all rounds that have been open for longer than ``minutes``."""
    cutoff = (datetime.now(timezone.utc) - timedelta(minutes=minutes)).isoformat()
    return await _get("bt_game_rounds",
                      {"status": "eq.open", "created_at": f"lt.{cutoff}", "select": "*", "limit": "200"})


async def _get_open_round_rest(tg_id: int, game: str) -> dict | None:
    rows = await _get("bt_game_rounds",
                      {"tg_id": f"eq.{tg_id}", "game": f"eq.{game}", "status": "eq.open",
                       "select": "*", "limit": "1"})
    return rows[0] if rows else None


async def get_open_round(tg_id: int, game: str) -> dict | None:
    """The user's currently-open round for ``game`` (direct-DB, REST fallback)."""
    try:
        pool = await pgpool.get_pool()
        rec = await pool.fetchrow(
            "select * from bt_game_rounds "
            "where tg_id = $1 and game = $2 and status = 'open' limit 1",
            tg_id, game)
        return _row(rec)
    except Exception as e:  # noqa: BLE001
        if pgpool.should_fallback(e):
            return await _get_open_round_rest(tg_id, game)
        raise


async def _get_round_rest(round_id: str) -> dict | None:
    rows = await _get("bt_game_rounds", {"id": f"eq.{round_id}", "select": "*", "limit": "1"})
    return rows[0] if rows else None


async def get_round(round_id: str) -> dict | None:
    """Read a round by id (direct-DB, REST fallback)."""
    try:
        pool = await pgpool.get_pool()
        rec = await pool.fetchrow(
            "select * from bt_game_rounds where id = $1::uuid limit 1", str(round_id))
        return _row(rec)
    except Exception as e:  # noqa: BLE001
        if pgpool.should_fallback(e):
            return await _get_round_rest(round_id)
        raise


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


async def _update_open_round_rest(round_id: str, patch: dict) -> dict | None:
    return await _patch("bt_game_rounds",
                        {"id": f"eq.{round_id}", "status": "eq.open"}, patch)


async def update_open_round(round_id: str, patch: dict) -> dict | None:
    """PATCH a round's mutable state guarded on ``status='open'``.

    Returns the updated row, or ``None`` if the round is no longer open (settled
    by a concurrent cashout/bust, or swept as timed-out). Lets a caller working
    from an in-process cache detect a stale entry and fall back correctly instead
    of reviving a closed round. Direct-DB with REST fallback on pool failure.
    """
    keys = list(patch.keys())
    set_clause = ", ".join(f"{k} = ${i + 2}" for i, k in enumerate(keys))
    vals = [patch[k] for k in keys]
    sql = (f"update bt_game_rounds set {set_clause} "
           f"where id = $1::uuid and status = 'open' returning *")
    try:
        pool = await pgpool.get_pool()
        rec = await pool.fetchrow(sql, str(round_id), *vals)
        return _row(rec)
    except Exception as e:  # noqa: BLE001
        if pgpool.should_fallback(e):
            return await _update_open_round_rest(round_id, patch)
        raise


async def settle_round(round_id: str, tg_id: int, outcome: dict, payout: int,
                       status: str) -> dict:
    """Call ``bt_settle_round`` — atomically close an open round and credit any
    payout in a single transaction (one round trip). Returns
    ``{"closed": bool, "new_balance": int}``.

    Guarded on ``status='open'`` server-side, so a concurrent double-settle
    returns ``closed=False`` and credits nothing (idempotent).
    """
    try:
        pool = await pgpool.get_pool()
    except Exception as e:  # noqa: BLE001 — pool unavailable → REST
        if pgpool.should_fallback(e):
            return await _settle_round_rest(round_id, tg_id, outcome, payout, status)
        raise
    try:
        val = await pool.fetchval(
            "select bt_settle_round($1::uuid, $2, $3, $4, $5)",
            str(round_id), tg_id, outcome, payout, status)
    except asyncpg.PostgresError as pe:
        # A server-reported error is a real failure (never fall back and risk a
        # double credit); surface it the same way the REST path did.
        raise SupabaseError(f"settle_round failed: {pe}")
    except Exception as e:  # noqa: BLE001 — transport failure → REST
        if pgpool.should_fallback(e):
            return await _settle_round_rest(round_id, tg_id, outcome, payout, status)
        raise
    _invalidate_user(tg_id)
    return val


async def _settle_round_rest(round_id: str, tg_id: int, outcome: dict, payout: int,
                             status: str) -> dict:
    client = _get_client()
    payload = {
        "p_round_id": round_id,
        "p_tg_id": tg_id,
        "p_outcome": outcome,
        "p_payout": payout,
        "p_status": status,
    }
    r = await client.post("/rpc/bt_settle_round", json=payload)
    if r.status_code >= 400:
        body = ""
        try:
            body = r.text
        except Exception:
            pass
        raise SupabaseError(f"settle_round failed: {r.status_code} {body}")
    _invalidate_user(tg_id)
    return r.json()


async def _double_round_rest(round_id: str, tg_id: int, extra_bet: int, outcome: dict) -> dict | None:
    client = _get_client()
    payload = {
        "p_round_id": round_id,
        "p_tg_id": tg_id,
        "p_extra_bet": extra_bet,
        "p_outcome": outcome,
    }
    r = await client.post("/rpc/bt_double_round", json=payload)
    if r.status_code >= 400:
        body = ""
        try:
            body = r.text
        except Exception:
            pass
        if "round_not_open" in body:
            return None
        if "insufficient_balance" in body:
            raise InsufficientBalance("insufficient_balance")
        raise SupabaseError(f"double_round failed: {r.status_code} {body}")
    _invalidate_user(tg_id)
    return r.json()


async def _split_round_rest(round_id: str, tg_id: int, extra_bet: int, outcome: dict) -> dict | None:
    client = _get_client()
    payload = {
        "p_round_id": round_id,
        "p_tg_id": tg_id,
        "p_extra_bet": extra_bet,
        "p_outcome": outcome,
    }
    r = await client.post("/rpc/bt_split_round", json=payload)
    if r.status_code >= 400:
        body = ""
        try:
            body = r.text
        except Exception:
            pass
        if "round_not_open" in body:
            return None
        if "insufficient_balance" in body:
            raise InsufficientBalance("insufficient_balance")
        raise SupabaseError(f"split_round failed: {r.status_code} {body}")
    _invalidate_user(tg_id)
    return r.json()


async def split_round(round_id: str, tg_id: int, extra_bet: int, outcome: dict) -> dict | None:
    """Blackjack split: debit ``extra_bet`` (equal to the round's original bet)
    and add it onto the round's bet + write the new two-hand outcome, atomically,
    via the ``bt_split_round`` RPC.

    Returns ``{"balance": int, "bet": int}``, or ``None`` if the round is no
    longer open. Raises ``InsufficientBalance`` if the extra stake can't be
    covered.
    """
    try:
        pool = await pgpool.get_pool()
    except Exception as e:  # noqa: BLE001 — pool unavailable → REST
        if pgpool.should_fallback(e):
            return await _split_round_rest(round_id, tg_id, extra_bet, outcome)
        raise
    try:
        val = await pool.fetchval(
            "select bt_split_round($1::uuid, $2, $3, $4)",
            str(round_id), tg_id, extra_bet, outcome)
    except asyncpg.exceptions.RaiseError as re:
        msg = getattr(re, "message", "") or str(re)
        if "round_not_open" in msg:
            return None
        if "insufficient_balance" in msg:
            raise InsufficientBalance("insufficient_balance")
        raise SupabaseError(f"split_round failed: {msg}")
    except Exception as e:  # noqa: BLE001 — transport failure → REST
        if pgpool.should_fallback(e):
            return await _split_round_rest(round_id, tg_id, extra_bet, outcome)
        raise
    _invalidate_user(tg_id)
    return val


async def double_round(round_id: str, tg_id: int, extra_bet: int, outcome: dict) -> dict | None:
    """Blackjack double-down: debit ``extra_bet`` and add it onto the round's
    bet + write the new outcome, atomically, via the ``bt_double_round`` RPC.

    Returns ``{"balance": int, "bet": int}``, or ``None`` if the round is no
    longer open (settled/expired underneath the caller).

    Raises ``InsufficientBalance`` if the extra stake can't be covered.
    """
    try:
        pool = await pgpool.get_pool()
    except Exception as e:  # noqa: BLE001 — pool unavailable → REST
        if pgpool.should_fallback(e):
            return await _double_round_rest(round_id, tg_id, extra_bet, outcome)
        raise
    try:
        val = await pool.fetchval(
            "select bt_double_round($1::uuid, $2, $3, $4)",
            str(round_id), tg_id, extra_bet, outcome)
    except asyncpg.exceptions.RaiseError as re:
        msg = getattr(re, "message", "") or str(re)
        if "round_not_open" in msg:
            return None
        if "insufficient_balance" in msg:
            raise InsufficientBalance("insufficient_balance")
        raise SupabaseError(f"double_round failed: {msg}")
    except Exception as e:  # noqa: BLE001 — transport failure → REST
        if pgpool.should_fallback(e):
            return await _double_round_rest(round_id, tg_id, extra_bet, outcome)
        raise
    _invalidate_user(tg_id)
    return val


async def claim_daily(tg_id: int, day_start_iso: str, streak: int, now_iso: str) -> dict | None:
    """Compare-and-swap the daily claim: set ``streak_days``/``last_claim_at``
    only if ``last_claim_at`` predates today's UTC start (or is null).

    Returns the updated row if THIS call won the claim, or ``None`` if another
    concurrent request already claimed today — closing the double-claim race.
    """
    result = await _patch(
        "bt_users",
        {"tg_id": f"eq.{tg_id}",
         "or": f"(last_claim_at.is.null,last_claim_at.lt.{day_start_iso})"},
        {"streak_days": streak, "last_claim_at": now_iso},
    )
    _invalidate_user(tg_id)
    return result


# ---------------------------------------------------------------------------
# Ledger history + leaderboard
# ---------------------------------------------------------------------------

async def ledger_history(tg_id: int, limit: int = 50) -> list[dict]:
    return await _get("bt_ledger",
                      {"tg_id": f"eq.{tg_id}", "select": "id,amount,kind,ref,created_at",
                       "order": "created_at.desc", "limit": str(limit)})


async def rounds_history(tg_id: int, limit: int = 50) -> list[dict]:
    """The user's most recent resolved bets, newest first.

    Only true wagering outcomes (settled / cashed_out) — abandoned or voided
    rounds are not bets and would show misleading 0.00x multipliers.
    """
    return await _get("bt_game_rounds",
                      {"tg_id": f"eq.{tg_id}", "status": "in.(settled,cashed_out)",
                       "select": "id,game,bet,payout,status,created_at,settled_at",
                       "order": "created_at.desc", "limit": str(limit)})


async def _leaderboard_rich_rest(limit: int = 20) -> list[dict]:
    return await _get("bt_users",
                      {"select": "tg_id,display_name,balance",
                       "order": "balance.desc", "limit": str(limit),
                       "started_at": "not.is.null", "is_dev": "is.false"})


async def leaderboard_rich(limit: int = 20) -> list[dict]:
    """Top registered, non-dev balances (direct-DB, REST fallback), cached
    briefly: the whole board is shared by every caller and the
    bt_users_balance_idx read is pointless to repeat 20x/minute per user."""
    key = f"lb:rich:alltime:{limit}"
    hit = cache.get(key)
    if hit is not None:
        return hit
    try:
        pool = await pgpool.get_pool()
        recs = await pool.fetch(
            "select tg_id, display_name, balance from bt_users "
            "where started_at is not null and not is_dev "
            "order by balance desc limit $1",
            limit)
        rows = [_row(r) for r in recs]
    except Exception as e:  # noqa: BLE001
        if not pgpool.should_fallback(e):
            raise
        rows = await _leaderboard_rich_rest(limit)
    cache.put(key, rows, _LB_CACHE_TTL)
    return rows


async def _dev_ids_all_rest() -> set[int]:
    rows = await _get("bt_users",
                      {"select": "tg_id", "is_dev": "is.true", "limit": "1000"})
    return {int(r["tg_id"]) for r in rows}


async def dev_ids_all() -> set[int]:
    """Every dev-flagged tg_id (tiny partial-index scan, cached briefly).

    Dev accounts (/dev) are hidden from both leaderboards. The set is shared
    by every leaderboard caller, so one short-lived cache entry covers them
    all; a flag flip is visible within the TTL. Direct-DB, REST fallback —
    failures propagate (the leaderboard should error, not quietly include a
    dev account)."""
    key = "devids:all"
    hit = cache.get(key)
    if hit is not None:
        return hit
    try:
        pool = await pgpool.get_pool()
        recs = await pool.fetch("select tg_id from bt_users where is_dev")
        ids = {int(r["tg_id"]) for r in recs}
    except Exception as e:  # noqa: BLE001
        if not pgpool.should_fallback(e):
            raise
        ids = await _dev_ids_all_rest()
    cache.put(key, ids, _LB_CACHE_TTL)
    return ids


async def _rich_rank_rest(balance: int) -> int:
    client = _get_client()
    r = await client.get("/bt_users",
                         params={"select": "tg_id", "balance": f"gt.{balance}",
                                 "started_at": "not.is.null", "is_dev": "is.false"},
                         headers={"Prefer": "count=exact", "Range-Unit": "items", "Range": "0-0"})
    r.raise_for_status()
    cr = r.headers.get("content-range", "*/0")
    total = cr.split("/")[-1]
    return (int(total) + 1) if total.isdigit() else 1


async def rich_rank(tg_id: int, balance: int) -> int:
    """1-based rank on the rich list = count(registered, non-dev users with
    balance > mine) + 1.

    Direct-DB count (REST fallback), cached briefly per (user, balance): the
    balance is part of the key, so the user's own wins/losses miss the cache
    naturally and only same-balance repeat views (e.g. /me polling) are served
    from it.
    """
    key = f"richrank:{tg_id}:{balance}"
    hit = cache.get(key)
    if hit is not None:
        return hit
    try:
        pool = await pgpool.get_pool()
        n = await pool.fetchval(
            "select count(*) from bt_users "
            "where started_at is not null and not is_dev and balance > $1",
            balance)
        rank = int(n) + 1
    except Exception as e:  # noqa: BLE001
        if not pgpool.should_fallback(e):
            raise
        rank = await _rich_rank_rest(balance)
    cache.put(key, rank, _STATS_CACHE_TTL)
    return rank


async def user_stats(tg_id: int) -> dict:
    """Call ``bt_user_stats`` — single-round-trip home-card stats.

    Returns ``{messages_sent, amount_wagered, messages_rank}``.
    Raises ``SupabaseError`` on failure. Cached briefly (display-only card);
    the cache is dropped alongside the profile on same-process mutations.
    """
    key = f"stats:{tg_id}"
    hit = cache.get(key)
    if hit is not None:
        return hit
    client = _get_client()
    r = await client.post("/rpc/bt_user_stats", json={"p_tg_id": tg_id})
    if r.status_code >= 400:
        raise SupabaseError(f"bt_user_stats failed: {r.status_code} {r.text[:200]}")
    data = r.json()
    out = data if isinstance(data, dict) else {}
    if out:
        cache.put(key, out, _STATS_CACHE_TTL)
    return out


async def claim_backlog(tg_id: int) -> dict:
    """Atomically credit 75% of the user's backlog and clear it.

    Calls ``bt_claim_backlog`` which writes a ``backlog_claim`` ledger row and
    returns ``{awarded, new_balance}``. Invalidates the user cache.
    """
    client = _get_client()
    r = await client.post("/rpc/bt_claim_backlog", json={"p_tg_id": tg_id})
    body = ""
    try:
        body = r.text
    except Exception:
        pass
    if r.status_code >= 400:
        raise SupabaseError(f"bt_claim_backlog failed: {r.status_code} {body}")
    _invalidate_user(tg_id)
    data = r.json()
    return data if isinstance(data, dict) else {"awarded": 0, "new_balance": 0}


async def _chat_counts_since_rest(start_day: str) -> list[dict]:
    return await _get("bt_chat_counts",
                      {"select": "tg_id,count", "day": f"gte.{start_day}",
                       "limit": "100000"})


async def chat_counts_since(start_day: str) -> list[dict]:
    """Per-user message totals on/after ``start_day`` (UTC ISO date), as
    ``[{tg_id, count}]``. Powers the chatters leaderboard (ranks by messages
    sent, not points earned).

    Direct-DB with the SUM/GROUP BY pushed server-side (one small result set
    instead of shipping every per-day row), REST fallback (per-day rows,
    aggregated by the caller — both shapes sum identically). Cached briefly:
    the board is shared by every caller.
    """
    key = f"lb:chat:{start_day}"
    hit = cache.get(key)
    if hit is not None:
        return hit
    try:
        pool = await pgpool.get_pool()
        # asyncpg binds a date param from a datetime.date object, NOT an ISO
        # string (str has no .toordinal → encode error), so parse it first.
        recs = await pool.fetch(
            "select tg_id, sum(count)::bigint as count from bt_chat_counts "
            "where day >= $1 group by tg_id",
            date.fromisoformat(start_day))
        rows = [_row(r) for r in recs]
    except Exception as e:  # noqa: BLE001
        if not pgpool.should_fallback(e):
            raise
        rows = await _chat_counts_since_rest(start_day)
    cache.put(key, rows, _LB_CACHE_TTL)
    return rows


async def ledger_totals_since(
    start: str, exclude_kind: str | None = None
) -> dict[int, int]:
    """Net ledger ``amount`` per ``tg_id`` since ``start`` (UTC ISO), as
    ``{tg_id: net}`` — the weekly rich leaderboard's net-points-gained metric.

    ``exclude_kind`` drops one kind (e.g. 'weekly_bonus') so a prior week's
    prize, credited at the Monday boundary, doesn't give past winners a head
    start in the new week's rich race.

    The aggregation runs DB-side via the direct pool (one round-trip, no row
    cap). This matters: PostgREST enforces a server-side max-rows cap, so a
    single REST fetch of ``bt_ledger`` silently truncates on a busy week
    (thousands of ledger rows) and mis-totals the board — summing some rows'
    bets without their matching wins, which can even net negative. The pool
    avoids that entirely; the REST fallback paginates so it, too, sees every
    row."""
    # pg-first: server-side SUM/GROUP BY, immune to the REST max-rows cap.
    # asyncpg binds a timestamptz param from a datetime, so parse the ISO string.
    start_dt = datetime.fromisoformat(start)
    try:
        pool = await pgpool.get_pool()
        if exclude_kind:
            recs = await pool.fetch(
                "select tg_id, sum(amount)::bigint total from bt_ledger "
                "where created_at >= $1 and kind <> $2 group by tg_id",
                start_dt, exclude_kind)
        else:
            recs = await pool.fetch(
                "select tg_id, sum(amount)::bigint total from bt_ledger "
                "where created_at >= $1 group by tg_id",
                start_dt)
        return {int(r["tg_id"]): int(r["total"]) for r in recs}
    except Exception as e:  # noqa: BLE001 — pool unavailable → REST
        if not pgpool.should_fallback(e):
            raise

    # REST fallback: page through every row so the max-rows cap can never
    # truncate the sum. Cap-agnostic — we advance by the number of rows the
    # server actually returned (which the cap may shrink below ``page``) and
    # stop only on an empty page, so a cap smaller than ``page`` can't end the
    # loop early. ``order=id.asc`` keeps paging stable (no row skipped or
    # double-counted across pages).
    totals: dict[int, int] = {}
    page, offset = 1000, 0
    while True:
        params: dict[str, Any] = {
            "select": "tg_id,amount", "created_at": f"gte.{start}",
            "order": "id.asc", "limit": str(page), "offset": str(offset),
        }
        if exclude_kind:
            params["kind"] = f"neq.{exclude_kind}"
        rows = await _get("bt_ledger", params)
        if not rows:
            break
        for r in rows:
            uid = int(r["tg_id"])
            totals[uid] = totals.get(uid, 0) + int(r["amount"])
        offset += len(rows)
    return totals


async def get_config(key: str) -> str | None:
    """Read a config value from bt_config, or ``None`` if unset/unconfigured."""
    if not is_configured():
        return None
    try:
        rows = await _get("bt_config", {"select": "value", "key": f"eq.{key}", "limit": "1"})
    except Exception:
        return None
    return rows[0]["value"] if rows else None


async def get_log_chat_id() -> int | None:
    raw = await get_config("log_chat_id")
    if raw is None:
        return None
    try:
        return int(raw)
    except (TypeError, ValueError):
        return None


async def _display_names_rest(tg_ids: list[int]) -> dict[int, str]:
    ids = ",".join(str(i) for i in tg_ids)
    rows = await _get("bt_users", {"select": "tg_id,display_name", "tg_id": f"in.({ids})"})
    return {int(r["tg_id"]): (r.get("display_name") or str(r["tg_id"])) for r in rows}


async def display_names(tg_ids: list[int]) -> dict[int, str]:
    """``{tg_id: display_name-or-id}`` for the given users (direct-DB, REST
    fallback)."""
    if not tg_ids:
        return {}
    try:
        pool = await pgpool.get_pool()
        recs = await pool.fetch(
            "select tg_id, display_name from bt_users where tg_id = any($1::bigint[])",
            [int(i) for i in tg_ids])
        return {int(r["tg_id"]): (r["display_name"] or str(r["tg_id"])) for r in recs}
    except Exception as e:  # noqa: BLE001
        if pgpool.should_fallback(e):
            return await _display_names_rest(tg_ids)
        raise


async def _registered_ids_rest(uniq: list[int]) -> set[int]:
    # Batched to stay well under PostgREST URL-length limits on large weeks.
    out: set[int] = set()
    for i in range(0, len(uniq), 150):
        chunk = ",".join(str(x) for x in uniq[i:i + 150])
        rows = await _get("bt_users",
                          {"select": "tg_id", "tg_id": f"in.({chunk})",
                           "started_at": "not.is.null"})
        out.update(int(r["tg_id"]) for r in rows)
    return out


async def registered_ids(tg_ids: list[int]) -> set[int]:
    """Subset of ``tg_ids`` whose users are registered (``started_at`` set).

    The Rich leaderboard is registered-only: unregistered chatters keep their
    points and their spot on Chatters, but never appear on Rich. Direct-DB
    (one ANY() query regardless of week size), REST fallback (chunked)."""
    uniq = list({int(i) for i in tg_ids})
    if not uniq:
        return set()
    try:
        pool = await pgpool.get_pool()
        recs = await pool.fetch(
            "select tg_id from bt_users "
            "where started_at is not null and tg_id = any($1::bigint[])",
            uniq)
        return {int(r["tg_id"]) for r in recs}
    except Exception as e:  # noqa: BLE001
        if pgpool.should_fallback(e):
            return await _registered_ids_rest(uniq)
        raise

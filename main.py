"""Bartender API — FastAPI game engine, shop, and leaderboard (spec §8, contract §4).

Server-authoritative: ALL game math (RNG, multipliers, win/loss, payout) runs
here. The client sends a bet and receives an ``outcome`` to animate — it never
computes odds, outcomes, or balances.

Every endpoint requires a valid ``X-Telegram-Init-Data`` header (auth.require_user).
``tg_id`` is derived ONLY from the validated initData. Balance changes go through
``bt_apply_ledger`` exclusively. CORS is an explicit allowlist (BT_APP_ORIGIN only).

Importing this module performs NO network I/O; Supabase is contacted lazily and,
when unconfigured, endpoints return 503.
"""

import asyncio
import logging
import math
import re
import time
from collections import defaultdict
from datetime import datetime, timedelta, timezone

from fastapi import Body, Depends, FastAPI, Header, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from . import cache, db, notify, pgpool, ratelimit
from .auth import require_user
from .config import settings
from .db import InsufficientBalance, SupabaseNotConfigured
from .game import BET_MAX, BET_MIN, GAMES, MULT_CAP, MULTI_STEP, P_MAX, SINGLE_SETTLE
from .game import blackjack, chicken, crash, dice, flip, highlow, keno, mines, plinko, rps, seedpair, towers
from .game.seed import rng_float

app = FastAPI(title="Bartender API", version="1.0")

# ---------------------------------------------------------------------------
# Stale-round sweeper — background task that auto-cashouts rounds that were
# left open (e.g. user closed Telegram mid-game). Runs every 5 minutes.
# ---------------------------------------------------------------------------
_STALE_MINUTES = 30          # rounds open longer than this are swept
_SWEEP_INTERVAL_SEC = 300    # how often the sweeper checks

logger = logging.getLogger(__name__)


async def _sweep_stale_rounds() -> None:
    """Find open rounds older than _STALE_MINUTES and settle them."""
    try:
        stale = await db.get_stale_open_rounds(_STALE_MINUTES)
    except Exception as exc:
        logger.warning("stale_sweep_db_error: %s", exc)
        return

    for rnd in stale:
        try:
            await _cashout_stale_round(rnd)
        except Exception as exc:
            logger.warning("stale_cashout_failed round=%s: %s", rnd.get("id"), exc)


async def _cashout_stale_round(rnd: dict) -> None:
    name = rnd.get("game", "")
    tg_id = int(rnd["tg_id"])
    params = rnd.get("params") or {}
    state = rnd.get("outcome") or {}
    ss, cs, nonce = rnd["server_seed"], rnd["client_seed"], int(rnd["nonce"])

    try:
        if name == "blackjack" and state.get("split"):
            # A stale split settles as a STAND on every not-yet-done hand: play
            # the dealer once (only if a hand is still live) and resolve each
            # hand independently, exactly what a /step stand chain would have
            # produced. Settled here directly (per-hand payouts can't be a single
            # multiplier) and returns before the generic _finalise below.
            dealer = list(state.get("dealer") or [])
            cursor = int(state.get("next_cursor", 4))
            hands = [dict(h) for h in (state.get("hands") or [])]
            for h in hands:
                h["cards"] = list(h.get("cards") or [])
            draw = lambda i: rng_float(ss, cs, nonce, i)
            if any(not blackjack.is_bust(blackjack.hand_total(h["cards"])[0]) for h in hands):
                dealer, cursor = blackjack.play_dealer(draw, dealer, cursor)
            results, total_payout = [], 0
            for h in hands:
                m = blackjack.outcome_multiplier(h["cards"], dealer)
                results.append(m)
                if m > 0:
                    total_payout += _payout(int(h["bet"]), m)
            outcome = {"dealer": dealer, "next_cursor": cursor, "split": True,
                       "aces": bool(state.get("aces")), "hands": hands,
                       "active": None, "player_done": True, "results": results}
            await _finalise(rnd, tg_id, 0.0, "timed_out", outcome, payout_override=total_payout)
            logger.info("stale_round_swept round=%s game=blackjack(split) tg_id=%s", rnd["id"], tg_id)
            return
        if name == "blackjack":
            # A stale hand settles as a STAND on the player's current cards —
            # the least-punitive deterministic resolution (mirrors how the
            # multi-step games below settle at their current progression
            # instead of being abandoned). The dealer plays out S17 from the
            # stored cursor with the round's own seeded draws, so the result
            # is exactly what /step stand would have produced. Naturals settle
            # at bet time, so an open round always has a live, un-busted hand.
            player = list(state.get("player") or [])
            dealer = list(state.get("dealer") or [])
            cursor = int(state.get("next_cursor", 4))
            if len(player) < 2 or len(dealer) < 2:
                mult, outcome = 0.0, {"multiplier": 0.0}
            else:
                draw = lambda i: rng_float(ss, cs, nonce, i)
                dealer, cursor = blackjack.play_dealer(draw, dealer, cursor)
                mult = blackjack.outcome_multiplier(player, dealer)
                outcome = {"player": player, "dealer": dealer, "next_cursor": cursor,
                           "doubled": bool(state.get("doubled")), "player_done": True,
                           "multiplier": mult}
        elif name not in MULTI_STEP:
            # Single-settle games (dice/plinko) that somehow stayed open, and
            # crash rounds never cashed out (the curve crashed unclaimed): abandon.
            await db.close_round(rnd["id"], {
                "outcome": state, "payout": 0, "status": "abandoned",
                "settled_at": db._now(),
            })
            return

        elif name == "flip":
            streak = int(state.get("streak", 0))
            mult = min(flip.multiplier(streak), MULT_CAP) if streak > 0 else 0.0
            outcome = {"streak": streak, "multiplier": mult}
        elif name == "mines":
            revealed = list(state.get("revealed", []))
            if not revealed:
                mult, outcome = 0.0, {"revealed": [], "multiplier": 0.0}
            else:
                m = int(params.get("mines", 3))
                mult = min(mines.multiplier(len(revealed), m), MULT_CAP)
                mine_set = sorted(mines.mine_positions(ss, cs, nonce, m))
                outcome = {"revealed": revealed, "mines_count": m, "multiplier": mult, "mines": mine_set}
        elif name == "towers":
            floor = int(state.get("floor", 0))
            if floor <= 0:
                mult, outcome = 0.0, {"floor": 0, "multiplier": 0.0}
            else:
                difficulty = str(params.get("difficulty", "medium"))
                mult = min(towers.multiplier(floor, difficulty), MULT_CAP)
                outcome = {"floor": floor, "difficulty": difficulty, "multiplier": mult}
        elif name == "rps":
            wins = int(state.get("wins", 0))
            # Ties advance `step` but not `wins`; a round with no wins has made
            # no winning wager, so it settles at 0 just like the cashout gate.
            if wins < 1:
                mult, outcome = 0.0, {"wins": 0, "multiplier": 0.0}
            else:
                mult = min(rps.multiplier(wins), rps.RPS_MAX_MULT)
                outcome = {"wins": wins, "step": int(state.get("step", 0)), "multiplier": mult}
        elif name == "chicken":
            lane = int(state.get("lane", 0))
            # No lanes crossed means no wager decision was made; settle at 0
            # just like the cashout gate.
            if lane <= 0:
                mult, outcome = 0.0, {"lane": 0, "multiplier": 0.0}
            else:
                difficulty = str(params.get("difficulty", "medium"))
                mult = min(chicken.multiplier(lane, difficulty), chicken.CHICKEN_MAX_MULT)
                outcome = {"lane": lane, "difficulty": difficulty, "multiplier": mult}
        else:  # highlow
            step_n = int(state.get("step", 0))
            # Skips advance `step` but not `picks`; a skip-only round has made no
            # real wager decision, so it settles at 0 just like the cashout gate.
            if int(state.get("picks", 0)) < 1:
                mult, outcome = 0.0, {"step": step_n, "multiplier": 0.0}
            else:
                mult = min(float(state.get("multiplier", 1.0)), MULT_CAP)
                outcome = {"step": step_n, "rank": int(state.get("rank", 0)), "multiplier": mult}
    except Exception:
        mult, outcome = 0.0, {}

    await _finalise(rnd, tg_id, mult, "timed_out", outcome)
    logger.info("stale_round_swept round=%s game=%s tg_id=%s mult=%s", rnd["id"], name, tg_id, mult)


async def _stale_round_sweeper_loop() -> None:
    while True:
        await asyncio.sleep(_SWEEP_INTERVAL_SEC)
        await _sweep_stale_rounds()


@app.on_event("startup")
async def _start_sweeper() -> None:
    asyncio.create_task(_stale_round_sweeper_loop())


@app.on_event("shutdown")
async def _close_pgpool() -> None:
    await pgpool.close_pool()

logger = logging.getLogger("bt.api")
# Ensure request-timing (INFO) lines reach stdout in the Render logs regardless
# of uvicorn's logging config: attach a dedicated handler once and stop
# propagation so lines are not duplicated by a parent handler.
if not logger.handlers:
    _timing_handler = logging.StreamHandler()
    _timing_handler.setFormatter(
        logging.Formatter("%(asctime)s %(levelname)s %(name)s %(message)s")
    )
    logger.addHandler(_timing_handler)
logger.setLevel(logging.INFO)
logger.propagate = False

# CORS: open to all origins (wildcard) per user request. Safe here because auth
# is via the signed X-Telegram-Init-Data header, not cookies — no credentials are
# used, so a wildcard cannot be abused to ride a victim's session. The middleware
# stays (removing it entirely would omit the Access-Control-Allow-Origin header
# and browsers would block every cross-origin request). BT_APP_ORIGIN is ignored.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.middleware("http")
async def _ip_rate_limit_middleware(request: Request, call_next):
    """Pre-auth IP-level guard per IP across all /bt/api/ routes (config bt_rl_ip_*)."""
    if request.url.path.startswith("/bt/api/"):
        xff = request.headers.get("X-Forwarded-For")
        ip = xff.split(",")[0].strip() if xff else (
            request.client.host if request.client else "unknown"
        )
        allowed, retry_after = ratelimit.check(f"ip:{ip}", limit=settings.bt_rl_ip_limit, window_sec=settings.bt_rl_ip_window_sec)
        if not allowed:
            return JSONResponse(
                status_code=429,
                content={"ok": False, "error": "rate_limited"},
                headers={"Retry-After": str(retry_after)},
            )
    return await call_next(request)


@app.middleware("http")
async def _timing_middleware(request: Request, call_next):
    """Measure raw server-side handling time for every request.

    Adds an ``X-Process-Time`` response header (milliseconds) and logs one
    greppable line per request. This is server think-time only — it excludes
    all network latency — so it isolates app/platform cost from the wire.
    Added last so it wraps the rate-limit and CORS middleware, capturing the
    full in-process time.
    """
    start = time.perf_counter()
    try:
        response = await call_next(request)
    except Exception:
        dur_ms = (time.perf_counter() - start) * 1000.0
        logger.info(
            "req method=%s path=%s status=500 dur_ms=%.1f",
            request.method, request.url.path, dur_ms,
        )
        raise
    dur_ms = (time.perf_counter() - start) * 1000.0
    response.headers["X-Process-Time"] = f"{dur_ms:.1f}"
    logger.info(
        "req method=%s path=%s status=%s dur_ms=%.1f",
        request.method, request.url.path, response.status_code, dur_ms,
    )
    return response


@app.exception_handler(SupabaseNotConfigured)
async def _supabase_not_configured(_request: Request, _exc: SupabaseNotConfigured):
    return JSONResponse(status_code=503, content={"ok": False, "error": "supabase_not_configured"})


@app.exception_handler(HTTPException)
async def _http_exception(_request: Request, exc: HTTPException):
    """Flatten dict-shaped ``detail`` into the top-level body.

    FastAPI's default handler nests ``detail`` (e.g. ``raise
    HTTPException(401, detail={"error": "bad_init_data"})``) as
    ``{"detail": {"error": "bad_init_data"}}``. Every other error response in
    this API (rate limits, SupabaseNotConfigured, etc.) is a flat
    ``{"ok": false, "error": ...}`` body, and app/js/api.js's client only ever
    reads a top-level ``error`` field — so a nested detail silently never
    reaches the app's error-code branches. Flatten it here, once, for every
    HTTPException raised anywhere in the API, preserving headers (e.g.
    Retry-After) and any non-dict detail (e.g. FastAPI's own validation 422s).
    """
    if isinstance(exc.detail, dict):
        content = {"ok": False, **exc.detail}
    else:
        content = {"ok": False, "error": exc.detail}
    return JSONResponse(status_code=exc.status_code, content=content, headers=exc.headers)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _today() -> str:
    return datetime.now(timezone.utc).date().isoformat()


def _next_utc_midnight() -> str:
    now = datetime.now(timezone.utc)
    nxt = (now + timedelta(days=1)).replace(hour=0, minute=0, second=0, microsecond=0)
    return nxt.isoformat()


def _utc_day_start() -> str:
    now = datetime.now(timezone.utc)
    return now.replace(hour=0, minute=0, second=0, microsecond=0).isoformat()


# Reward stock is a plain counter (no monthly reset): all usage rows live under
# this single constant period key, so `monthly_limit - used` is the current
# stock and /restock overrides it by setting monthly_limit = used + amount.
STOCK_PERIOD = "ALL"


# Beginning of time — used as an "all-time" lower bound for ledger scans.
_EPOCH = "1970-01-01T00:00:00+00:00"


def _week_reset_at() -> str:
    """ISO timestamp of the NEXT UTC week boundary (next Monday 00:00:00 UTC) —
    when the weekly leaderboard resets. The client uses this to render a
    static "resets in Xd Xh Xm" on load/tab-switch; it never polls."""
    now = datetime.now(timezone.utc)
    monday = now - timedelta(days=now.weekday())
    this_monday = monday.replace(hour=0, minute=0, second=0, microsecond=0)
    return (this_monday + timedelta(days=7)).isoformat()


def _week_start() -> str:
    """Start of the current UTC week (Monday 00:00:00 UTC), ISO-formatted."""
    now = datetime.now(timezone.utc)
    monday = now - timedelta(days=now.weekday())
    return monday.replace(hour=0, minute=0, second=0, microsecond=0).isoformat()


def _daily_claim(streak: int) -> int:
    """D(s) = floor(20 * (1 + 1.5 * (1 - e^(-s/10))))  (spec §5)."""
    return int(20 * (1 + 1.5 * (1 - math.exp(-streak / 10))))


def _payout(bet: int, multiplier: float) -> int:
    """bet * multiplier, rounded to the nearest integer (half rounds up so a
    payout of exactly x.50 always rounds up, not Python's banker's rounding),
    capped at P_MAX. P_MAX is only a last-resort backstop against a bugged
    multiplier — it sits above the largest legitimate single-round win, so real
    wins (towers 7.45x, keno up to ~105x, etc.) are paid in full, never clipped."""
    return min(math.floor(bet * multiplier + 0.5), P_MAX)


def _err(code: str, status: int = 400) -> JSONResponse:
    return JSONResponse(status_code=status, content={"ok": False, "error": code})


def _rl_err(retry_after: int) -> JSONResponse:
    """429 with Retry-After header so clients know when to retry."""
    return JSONResponse(
        status_code=429,
        content={"ok": False, "error": "rate_limited"},
        headers={"Retry-After": str(retry_after)},
    )


# ---------------------------------------------------------------------------
# Health
# ---------------------------------------------------------------------------

@app.get("/bt/api/health")
async def health():
    return {"ok": True, "configured": db.is_configured()}


# ---------------------------------------------------------------------------
# TEMPORARY spike probe (point 2): measure PostgREST vs direct asyncpg latency
# for the SAME hot read, on Render (co-located with Supabase), before deciding
# whether to migrate any hot path off the REST hop. Read-only, returns timings
# only. Gated by a token passed in the X-Probe-Token HEADER (kept out of query
# strings/access logs) and rate-limited to blunt DB-load abuse. Remove once the
# spike question is answered.
# ---------------------------------------------------------------------------
_PROBE_TOKEN = "bt-spike-7719"


@app.get("/bt/api/_pgprobe")
async def pgprobe(n: int = 20, x_probe_token: str = Header("")):
    if x_probe_token != _PROBE_TOKEN:
        raise HTTPException(status_code=404)
    allowed, retry_after = ratelimit.check("pgprobe", limit=20, window_sec=60)
    if not allowed:
        return _rl_err(retry_after)
    n = max(1, min(n, 100))
    sample = 8243604458  # owner tg_id — known to exist

    async def _rest() -> None:
        await db.get_user(sample)

    pool = await pgpool.get_pool()

    async def _pg() -> None:
        async with pool.acquire() as c:
            await c.fetchrow("select * from bt_users where tg_id = $1", sample)

    async def _bench(fn) -> dict:
        for _ in range(3):  # warmup (connection + plan cache)
            await fn()
        xs: list[float] = []
        for _ in range(n):
            s = time.perf_counter()
            await fn()
            xs.append((time.perf_counter() - s) * 1000)
        xs.sort()
        k = len(xs)
        return {
            "min": round(xs[0], 2),
            "median": round(xs[k // 2], 2),
            "avg": round(sum(xs) / k, 2),
            "p95": round(xs[min(k - 1, int(k * 0.95))], 2),
            "max": round(xs[-1], 2),
        }

    return {
        "n": n,
        "sample": sample,
        "postgrest": await _bench(_rest),
        "asyncpg_pool": await _bench(_pg),
    }


# ---------------------------------------------------------------------------
# Account / quests
# ---------------------------------------------------------------------------

@app.get("/bt/api/me")
async def me(user: dict = Depends(require_user)):
    tg_id = user["tg_id"]
    allowed, retry_after = ratelimit.check(f"me:{tg_id}", limit=settings.bt_rl_me_limit, window_sec=settings.bt_rl_me_window_sec)
    if not allowed:
        return _rl_err(retry_after)
    # Opening the Mini App is the definitive "started the app" signal, so stamp
    # started_at here (once). Passive chat may have already created the row (with
    # started_at NULL), so we must still register when it's unstamped — not only
    # when the row is missing. Once started_at is set, warm-cache opens skip both
    # writes (the optimization) since re-registering is a no-op.
    u = await db.get_user_cached(tg_id)
    if u is None or not u.get("started_at"):
        await db.mark_registered(tg_id, user.get("username"), user.get("display_name"))
        u = await db.get_user(tg_id) or {}
        db.cache_user(tg_id, u)
    else:
        # Registered + warm: keep identity canonical via the single
        # ensure_identity write-path — it backfills a missing name and refreshes
        # a changed one, but writes only when something actually differs, so the
        # warm-cache no-write fast path is preserved.
        u = await db.ensure_identity(
            tg_id, user.get("username"), user.get("display_name"), cached_row=u
        )
    day = _today()
    meta = u.get("meta") or {}
    raw_backlog = int(u.get("backlog_pts", 0))
    bal = int(u.get("balance", 0))

    # The quest row and the stats card come from independent reads, so fetch them
    # concurrently to collapse the sequential PostgREST round-trips into a single
    # wall-clock wait. The quest is always needed; the stats card is only computed
    # once the backlog is cleared (registered & claimed).
    need_stats = raw_backlog == 0
    jobs: list = [db.get_quest(tg_id, day)]
    if need_stats:
        jobs += [db.user_stats(tg_id), db.rich_rank(tg_id, bal)]
    results = await asyncio.gather(*jobs, return_exceptions=True)

    q = results[0]
    quest = q if isinstance(q, dict) else {"day": day, "chatted": False, "claimed": False}
    chatted = bool(quest.get("chatted", False))
    claimed = bool(quest.get("claimed", False))

    # Stats card: only computed when backlog is cleared (registered & claimed).
    activity: dict | None = None
    if need_stats:
        us, rr = results[1], results[2]
        if isinstance(us, dict) and not isinstance(rr, Exception):
            activity = {
                "messages_sent": int(us.get("messages_sent", 0)),
                "amount_wagered": int(us.get("amount_wagered", 0)),
                "messages_rank": int(us.get("messages_rank", 1)),
                "rich_rank": int(rr),
            }

    return {
        "tg_id": tg_id,
        "username": u.get("username"),
        "display_name": u.get("display_name"),
        "balance": bal,
        "streak_days": int(u.get("streak_days", 0)),
        "last_claim_at": u.get("last_claim_at"),
        "quest": {"day": day, "chatted": chatted, "claimed": claimed},
        "can_redeem": chatted and claimed,
        "age_ack": bool(meta.get("age_ack", False)),
        # @partygc name-tag perk (drives the gold home-screen badge). The bot
        # owns activation/verification; the app just reflects the stored flag.
        "multiplier_active": bool(u.get("name_multiplier", False)),
        # Backlog from pre-registration chat activity, shown at 75% of raw value.
        "backlog_pts": int(raw_backlog * 0.75) if raw_backlog > 0 else 0,
        # Activity stats — null while backlog is unclaimed (hides the card).
        "stats": activity,
    }


@app.get("/bt/api/stats/series")
async def stats_series(user: dict = Depends(require_user)):
    """7-day daily message + wagered series for the home stats-card charts."""
    tg_id = user["tg_id"]
    try:
        return await db.stats_series_7d(tg_id)
    except Exception:  # noqa: BLE001
        logging.exception("stats_series failed for %s", tg_id)
        return {"days": [], "messages": [], "wagered": []}


@app.post("/bt/api/backlog/claim")
async def backlog_claim(user: dict = Depends(require_user)):
    tg_id = user["tg_id"]
    allowed, retry_after = ratelimit.check(f"backlog:{tg_id}", limit=3, window_sec=60)
    if not allowed:
        return _rl_err(retry_after)
    try:
        result = await db.claim_backlog(tg_id)
    except db.SupabaseNotConfigured:
        return JSONResponse(status_code=503, content={"ok": False, "error": "not_configured"})
    except Exception as e:
        logging.getLogger(__name__).warning("bt_backlog_claim_error", extra={"error": str(e)})
        return {"ok": False, "error": "server_error"}
    return {
        "ok": True,
        "awarded": int(result.get("awarded", 0)),
        "new_balance": int(result.get("new_balance", 0)),
    }


@app.post("/bt/api/claim")
async def claim(user: dict = Depends(require_user)):
    tg_id = user["tg_id"]
    allowed, retry_after = ratelimit.check(f"claim:{tg_id}", limit=5, window_sec=60)
    if not allowed:
        return _rl_err(retry_after)

    u = await db.get_user(tg_id)
    if u is None:
        u = await db.upsert_user(tg_id, user.get("username"), user.get("display_name")) or {}

    today = datetime.now(timezone.utc).date()
    last_claim_at = u.get("last_claim_at")
    last_date = None
    if last_claim_at:
        try:
            last_date = datetime.fromisoformat(str(last_claim_at).replace("Z", "+00:00")).date()
        except ValueError:
            last_date = None

    if last_date == today:
        return {"ok": False, "error": "already_claimed", "next_claim_at": _next_utc_midnight()}

    prev_streak = int(u.get("streak_days", 0))
    if last_date == today - timedelta(days=1):
        streak = prev_streak + 1
    else:
        streak = 1

    # Compare-and-swap: claim the day FIRST, guarded on last_claim_at predating
    # today's UTC start (or null). Only the request that flips it credits, which
    # closes the concurrent double-claim race (spec §14). Credit AFTER the guard
    # wins so a crash forfeits the claim rather than paying it twice.
    won = await db.claim_daily(tg_id, _utc_day_start(), streak, db._now())
    if won is None:
        return {"ok": False, "error": "already_claimed", "next_claim_at": _next_utc_midnight()}

    awarded = _daily_claim(streak)
    new_balance = await db.apply_ledger(tg_id, awarded, "daily", ref=today.isoformat(), meta={"streak": streak})
    await db.set_quest(tg_id, _today(), claimed=True)

    return {
        "ok": True,
        "awarded": awarded,
        "new_balance": new_balance,
        "streak_days": streak,
        "next_claim_at": _next_utc_midnight(),
    }


@app.post("/bt/api/age-ack")
async def age_ack(user: dict = Depends(require_user)):
    tg_id = user["tg_id"]
    allowed, retry_after = ratelimit.check(f"age_ack:{tg_id}", limit=5, window_sec=60)
    if not allowed:
        return _rl_err(retry_after)
    # bt_set_age_ack create-if-absents the row, so no existence check is needed.
    await db.set_age_ack(tg_id)
    return {"ok": True}


# ---------------------------------------------------------------------------
# Shop
# ---------------------------------------------------------------------------

@app.get("/bt/api/rewards")
async def rewards(_user: dict = Depends(require_user)):
    tg_id = _user["tg_id"]
    allowed, retry_after = ratelimit.check(f"rewards:{tg_id}", limit=settings.bt_rl_read_limit, window_sec=settings.bt_rl_read_window_sec)
    if not allowed:
        return _rl_err(retry_after)
    period = STOCK_PERIOD
    items = await db.list_rewards(active_only=True)
    # One batched usage read for every limited reward (was one round-trip per
    # reward — the catalogue N+1). Rewards absent from the map have 0 usage.
    limited_ids = [r["id"] for r in items if int(r.get("monthly_limit", 0)) != 0]
    usages = await db.get_reward_usages(limited_ids, period)
    out = []
    for r in items:
        limit = int(r.get("monthly_limit", 0))
        if limit == 0:
            remaining = None  # unlimited
        else:
            used = usages.get(str(r["id"]), 0)
            remaining = max(0, limit - used)
        out.append({
            "id": r["id"],
            "title": r.get("title"),
            "description": r.get("description"),
            "cost": int(r.get("cost", 0)),
            "monthly_limit": limit,
            "remaining": remaining,
            "active": bool(r.get("active", True)),
        })
    return {"period": period, "rewards": out}


@app.post("/bt/api/redeem")
async def redeem(user: dict = Depends(require_user), body: dict | None = Body(default=None)):
    tg_id = user["tg_id"]
    body = body or {}
    reward_id = body.get("reward_id")
    if not reward_id:
        return _err("invalid_request")
    allowed, retry_after = ratelimit.check(f"redeem:{tg_id}", limit=10, window_sec=60)
    if not allowed:
        return _rl_err(retry_after)

    # Dev accounts (/dev) can play and earn but never claim shop rewards.
    u = await db.get_user_cached(tg_id)
    if u is not None and u.get("is_dev"):
        return {"ok": False, "error": "dev_account"}

    # Activity floor first (spec §5): today's chatted AND claimed (UTC).
    quest = await db.get_quest(tg_id, _today())
    if not (quest and quest.get("chatted") and quest.get("claimed")):
        return {"ok": False, "error": "activity_floor_not_met"}

    # One reward claim per UTC day. Rejected redemptions are auto-refunded and
    # excluded, so a denied claim never burns the user's day.
    if await db.has_redeemed_today(tg_id, _utc_day_start()):
        return {"ok": False, "error": "daily_limit_reached"}

    # Atomic debit + stock-usage increment + redemption row (spec §14) via the
    # bt_redeem RPC — reward/active/stock/balance are all enforced inside one
    # transaction, closing the read-then-write race on the stock counter.
    try:
        result = await db.redeem(tg_id, reward_id, STOCK_PERIOD)
    except db.RedeemError as e:
        return {"ok": False, "error": e.code}

    # Notify the user via Telegram DM that their redemption is pending. Best
    # effort — a failed DM must never break the (already-committed) redemption.
    try:
        reward = await db.get_reward(reward_id)
        title = (reward or {}).get("title") or "your reward"
        cost = int((reward or {}).get("cost", 0))
        await notify.send_dm(
            tg_id,
            f"🍸 <b>Redemption received</b>\n\n"
            f"You redeemed <b>{notify.esc(title)}</b> for {cost} pts.\n"
            f"Status: <b>pending</b> — we'll message you when it's processed.",
        )
    except Exception as e:
        logger.warning("bt_redeem_dm_error", error=str(e))

    # Mirror the redemption request to the logging chat (if one is configured).
    # Best effort — never affects the committed redemption or the response.
    try:
        log_chat = await db.get_log_chat_id()
        redemption_id = str(result.get("redemption_id", ""))
        if log_chat and redemption_id:
            rw = await db.get_reward(reward_id)
            title = (rw or {}).get("title") or "reward"
            limit = int((rw or {}).get("monthly_limit", 0))
            if limit <= 0:
                stock = "Unlimited"
            else:
                # bt_redeem already incremented usage for THIS request, so add
                # it back to show the pre-claim stock the "(-1 after claimed)"
                # wording refers to.
                used = await db.get_reward_usage(reward_id, STOCK_PERIOD)
                stock = str(max(0, limit - used + 1))
            name_parts = (user.get("display_name") or "").split()
            first_name = name_parts[0] if name_parts else str(tg_id)
            link = notify.profile_link(tg_id, user.get("username"), first_name)
            await notify.send_redemption_notification(
                log_chat,
                f"🎟️ <b>Redemption request</b>\n\n"
                f"User: {link}\n"
                f"Prize: <b>{notify.esc(title)}</b>\n"
                f"Stock: {notify.esc(stock)} → {max(0, int(stock) - 1) if stock.isdigit() else stock}\n"
                f"ID: <code>{notify.esc(redemption_id)}</code>",
                redemption_id,
            )
    except Exception as e:
        logger.warning("bt_redeem_log_error", error=str(e))

    return {
        "ok": True,
        "redemption_id": result.get("redemption_id"),
        "new_balance": result.get("new_balance"),
    }


# ---------------------------------------------------------------------------
# VIP tiers
# ---------------------------------------------------------------------------

@app.get("/bt/api/vip")
async def vip(user: dict = Depends(require_user)):
    tg_id = user["tg_id"]
    allowed, retry_after = ratelimit.check(
        f"vip:{tg_id}", limit=settings.bt_rl_read_limit, window_sec=settings.bt_rl_read_window_sec)
    if not allowed:
        return _rl_err(retry_after)
    data = await db.vip_get(tg_id)
    return {"ok": True, "state": data["state"], "tiers": data["tiers"]}


@app.post("/bt/api/vip/claim/{kind}")
async def vip_claim(kind: str, user: dict = Depends(require_user)):
    tg_id = user["tg_id"]
    if kind not in ("rakeback", "weekly", "monthly"):
        return JSONResponse(status_code=404, content={"ok": False, "error": "not_found"})
    allowed, retry_after = ratelimit.check(f"vipclaim:{tg_id}", limit=10, window_sec=60)
    if not allowed:
        return _rl_err(retry_after)
    try:
        res = await db.vip_claim(tg_id, kind)
    except db.SupabaseNotConfigured:
        return JSONResponse(status_code=503, content={"ok": False, "error": "not_configured"})
    except Exception as e:  # noqa: BLE001
        logging.getLogger(__name__).warning("bt_vip_claim_error", extra={"error": str(e), "kind": kind})
        return {"ok": False, "error": "server_error"}
    return res


# ---------------------------------------------------------------------------
# Leaderboard / history
# ---------------------------------------------------------------------------

async def _decorate_vip_levels(
    rows: list[dict], tg_id: int, you: dict | None = None
) -> None:
    """Attach ``vip_level`` to each leaderboard row (and the caller's own row).

    Unregistered users ("not signed up") always render as unranked (level 0),
    even if a stray vip_state row exists — the client shows the unranked badge
    for both level 0 and unregistered."""
    ids = [r["tg_id"] for r in rows if r.get("tg_id") is not None]
    if you is not None:
        ids.append(tg_id)
    if not ids:
        return
    levels = await db.vip_levels(ids)
    reg = await db.registered_ids(ids)
    for r in rows:
        uid = r.get("tg_id")
        r["vip_level"] = levels.get(uid, 0) if uid in reg else 0
    if you is not None:
        you["vip_level"] = levels.get(tg_id, 0) if tg_id in reg else 0


async def _rows_from_totals(
    totals: dict[int, int], tg_id: int, limit: int = 10
) -> tuple[list[dict], dict | None]:
    """Build ranked rows + the caller's own position from a {tg_id: value} map."""
    ordered = sorted(totals.items(), key=lambda kv: kv[1], reverse=True)
    top_ids = [uid for uid, _ in ordered[:limit]]
    names = await db.display_names(top_ids)
    rows = [
        {"rank": i + 1, "tg_id": uid, "display_name": names.get(uid, str(uid)), "value": val}
        for i, (uid, val) in enumerate(ordered[:limit])
    ]
    you = None
    if tg_id in totals:
        rank = 1 + sum(1 for _, v in ordered if v > totals[tg_id])
        you = {"rank": rank, "value": totals[tg_id]}
    return rows, you


@app.get("/bt/api/leaderboard")
async def leaderboard(
    tab: str = "rich", period: str = "weekly", user: dict = Depends(require_user)
):
    tg_id = user["tg_id"]
    allowed, retry_after = ratelimit.check(f"leaderboard:{tg_id}", limit=settings.bt_rl_read_limit, window_sec=settings.bt_rl_read_window_sec)
    if not allowed:
        return _rl_err(retry_after)
    if tab not in ("rich", "chatters"):
        return _err("invalid_tab")
    if period not in ("weekly", "alltime"):
        period = "weekly"

    if tab == "rich":
        if period == "alltime":
            # All-time rich list = current balances.
            top = await db.leaderboard_rich(limit=10)
            rows = [
                {"rank": i + 1, "tg_id": int(r["tg_id"]),
                 "display_name": r.get("display_name") or str(r["tg_id"]),
                 "value": int(r.get("balance", 0))}
                for i, r in enumerate(top)
            ]
            u = await db.get_user_cached(tg_id)
            you = None
            # Rich is registered-only; an unregistered caller has no Rich rank.
            # Dev accounts are hidden from ranks entirely — no "you" row either.
            if u is not None and u.get("started_at") and not u.get("is_dev"):
                bal = int(u.get("balance", 0))
                you = {"rank": await db.rich_rank(tg_id, bal), "value": bal}
            await _decorate_vip_levels(rows, tg_id, you)
            return {"tab": tab, "period": period, "rows": rows, "you": you}

        # Weekly rich = net points gained this week (sum of all ledger amounts),
        # excluding the weekly bonus itself so past winners don't get a head
        # start. Aggregated DB-side so the sum can't be truncated by PostgREST's
        # max-rows cap on a busy week (thousands of ledger rows). The finished
        # registered-only totals map is shared by every caller, so cache it
        # briefly — the per-caller "you" row is still computed fresh from it.
        week_start = _week_start()
        cache_key = f"lb:rich:weekly:{week_start}"
        totals = cache.get(cache_key)
        if totals is None:
            totals = await db.ledger_totals_since(week_start, exclude_kind="weekly_bonus")
            # Rich is registered-only: drop unregistered chatters (they keep
            # their points and their Chatters spot, but never rank on Rich).
            reg = await db.registered_ids(list(totals.keys()))
            totals = {uid: v for uid, v in totals.items() if uid in reg}
            # Dev accounts (/dev) are hidden from ranks; dropping them from
            # the shared totals map also suppresses a dev caller's "you" row.
            devs = await db.dev_ids_all()
            if devs:
                totals = {uid: v for uid, v in totals.items() if uid not in devs}
            cache.put(cache_key, totals, db._LB_CACHE_TTL)
        rows, you = await _rows_from_totals(totals, tg_id)
        await _decorate_vip_levels(rows, tg_id, you)
        return {"tab": tab, "period": period, "rows": rows, "you": you,
                "resets_at": _week_reset_at()}

    # chatters: raw messages sent (ranked by message count, not points earned).
    start_day = _week_start()[:10] if period == "weekly" else _EPOCH[:10]
    counts = await db.chat_counts_since(start_day)
    devs = await db.dev_ids_all()
    totals = defaultdict(int)
    for row in counts:
        uid = int(row["tg_id"])
        if uid in devs:
            # Dev accounts (/dev) are hidden from ranks (and their own "you"
            # row); their messages still count for quests/giveaways elsewhere.
            continue
        totals[uid] += int(row["count"])
    rows, you = await _rows_from_totals(totals, tg_id)
    await _decorate_vip_levels(rows, tg_id, you)
    resp = {"tab": tab, "period": period, "rows": rows, "you": you}
    if period == "weekly":
        resp["resets_at"] = _week_reset_at()
    return resp


@app.get("/bt/api/history")
async def history(user: dict = Depends(require_user)):
    tg_id = user["tg_id"]
    allowed, retry_after = ratelimit.check(f"history:{tg_id}", limit=settings.bt_rl_read_limit, window_sec=settings.bt_rl_read_window_sec)
    if not allowed:
        return _rl_err(retry_after)
    rows = await db.ledger_history(tg_id, limit=50)
    return {"rows": [
        {"id": r.get("id"), "amount": int(r.get("amount", 0)), "kind": r.get("kind"),
         "ref": r.get("ref"), "created_at": r.get("created_at")}
        for r in rows
    ]}


@app.get("/bt/api/bets")
async def bets(user: dict = Depends(require_user)):
    """Last 50 resolved game rounds for the caller (bet-history panel)."""
    tg_id = user["tg_id"]
    allowed, retry_after = ratelimit.check(f"bets:{tg_id}", limit=settings.bt_rl_read_limit, window_sec=settings.bt_rl_read_window_sec)
    if not allowed:
        return _rl_err(retry_after)
    rows = await db.rounds_history(tg_id, limit=50)
    return {"rows": [
        {"id": r.get("id"), "game": r.get("game"), "bet": int(r.get("bet") or 0),
         "payout": int(r.get("payout") or 0), "status": r.get("status"),
         "created_at": r.get("created_at"), "settled_at": r.get("settled_at")}
        for r in rows
    ]}


# ---------------------------------------------------------------------------
# Games
# ---------------------------------------------------------------------------

def _validate_params(name: str, params: dict) -> tuple[bool, dict]:
    """Validate + normalise per-game bet params. Returns (ok, normalised)."""
    p = params or {}
    if name == "dice":
        target = int(p.get("target", 0))
        return (dice.valid_target(target), {"target": target})
    if name == "flip":
        return (True, {})
    if name == "mines":
        m = int(p.get("mines", 0))
        return (1 <= m <= mines.TOTAL - 1, {"mines": m})
    if name == "towers":
        difficulty = str(p.get("difficulty", ""))
        return (towers.valid_difficulty(difficulty), {"difficulty": difficulty})
    if name == "highlow":
        return (True, {})
    if name == "rps":
        return (True, {})
    if name == "chicken":
        difficulty = str(p.get("difficulty", ""))
        return (chicken.valid_difficulty(difficulty), {"difficulty": difficulty})
    if name == "crash":
        return (True, {})
    if name == "plinko":
        rows = int(p.get("rows", 0))
        risk = str(p.get("risk", ""))
        return (plinko.valid_rows(rows) and plinko.valid_risk(risk), {"rows": rows, "risk": risk})
    if name == "blackjack":
        return (True, {})
    if name == "keno":
        raw = p.get("picks", [])
        if not isinstance(raw, (list, tuple)):
            return (False, {})
        # Coerce to ints (JSON may deliver floats like 21.0) before validating.
        picks = []
        for v in raw:
            if isinstance(v, bool):
                return (False, {})
            if isinstance(v, int):
                picks.append(v)
            elif isinstance(v, float) and v.is_integer():
                picks.append(int(v))
            else:
                return (False, {})
        # Store sorted for a stable, canonical params shape.
        return (keno.valid_picks(picks), {"picks": sorted(picks)})
    return (False, {})


def _initial_state(name: str, np: dict, ss: str, cs: str, nonce: int) -> dict:
    """Initial round outcome/state stored on bet (secrets never revealed here)."""
    if name == "flip":
        return {"streak": 0, "multiplier": 1.0}
    if name == "mines":
        return {"revealed": [], "mines": np["mines"]}
    if name == "towers":
        return {"floor": 0, "difficulty": np["difficulty"]}
    if name == "highlow":
        start = highlow.current_card(lambda i: rng_float(ss, cs, nonce, i), 0)
        return {"rank": start, "start_card": start, "step": 0, "multiplier": 1.0}
    if name == "rps":
        return {"step": 0, "wins": 0, "multiplier": 1.0}
    if name == "chicken":
        return {"lane": 0, "difficulty": np["difficulty"]}
    if name == "crash":
        # The crash point is fixed by the bet-time draw and the round clock is
        # anchored HERE: t0_ms is the server bet timestamp, and the round
        # autonomously crashes once e^(GROWTH * elapsed) reaches the crash
        # point (enforced by /cashout and the /crash/check poll). The crash
        # point is stored server-side for auditability; it is NEVER returned
        # to the client while the round is open (the /bet response carries
        # only the hash, and no endpoint exposes an open round's outcome).
        return {"crash_point": crash.crash_point(ss, cs, nonce),
                "t0_ms": int(time.time() * 1000)}
    if name == "blackjack":
        return blackjack.deal_initial(lambda i: rng_float(ss, cs, nonce, i))
    return {}


@app.get("/bt/api/game/seeds")
async def game_seeds(user: dict = Depends(require_user)):
    """The user's active seed pair (public view). Creates one on first access.
    NEVER exposes the active server_seed — only its hash and the next hash."""
    tg_id = user["tg_id"]
    # Bootstrap the user row first: bt_seed_pairs.tg_id has an FK to bt_users, so
    # a brand-new user hitting this endpoint before /me or a bet would otherwise
    # fail on the seed-pair insert. ensure_identity get-or-creates and keeps the
    # name fresh through the one canonical write-path.
    await db.ensure_identity(tg_id, user.get("username"), user.get("display_name"))
    pair = await db.get_seed_pair(tg_id)
    if not pair:
        pair = await db.create_seed_pair(tg_id, seedpair.new_pair())
    return {"ok": True, **seedpair.public_view(pair)}


@app.post("/bt/api/game/seeds/rotate")
async def game_seeds_rotate(user: dict = Depends(require_user), body: dict | None = Body(default=None)):
    """Rotate the active pair: reveal the retired server_seed, promote the
    pre-committed next one, apply an optional new client seed, reset the nonce."""
    tg_id = user["tg_id"]
    allowed, retry_after = ratelimit.check(f"game:{tg_id}", limit=settings.bt_rl_game_limit, window_sec=settings.bt_rl_game_window_sec)
    if not allowed:
        return _rl_err(retry_after)
    body = body or {}
    new_client_seed = str(body.get("client_seed") or "").strip()
    # Ensure a pair exists, then rotate it atomically. The RPC locks the seed-pair
    # row and refuses if ANY round is open (checked under that lock), so revealing
    # the active server_seed can never race a concurrent bet that would still use it.
    # Bootstrap the user row first (bt_seed_pairs.tg_id FK -> bt_users) via the
    # one canonical identity write-path.
    await db.ensure_identity(tg_id, user.get("username"), user.get("display_name"))
    if not await db.get_seed_pair(tg_id):
        await db.create_seed_pair(tg_id, seedpair.new_pair())
    next_seed, next_hash = seedpair.fresh_next()
    try:
        rotated = await db.rotate_seed_pair(tg_id, new_client_seed, next_seed, next_hash)
    except db.OpenRoundExists:
        # Block rotation mid-round: revealing the active server_seed while a round
        # is open would let the player derive that round's outcome before it settles.
        return _err("open_round_exists")
    return {"ok": True, **rotated}


async def _open_bet(name: str, user: dict, body: dict | None):
    """Open a round and reserve its nonce atomically. Returns (rnd, resp, err).

    Shared by /bet (which returns ``resp`` to the client and stops) and /play
    (which uses ``rnd`` to settle a single-settle game in the SAME request).
    ``rnd`` is the full open-round dict — identical to what the open-round cache
    and ``get_round`` hold — so a follow-on settle needs no extra DB read. On any
    handled failure ``err`` is the response to return and ``rnd``/``resp`` are None.
    """
    if name not in GAMES:
        return None, None, _err("unknown_game", 404)
    tg_id = user["tg_id"]
    allowed, retry_after = ratelimit.check(f"game:{tg_id}", limit=settings.bt_rl_game_limit, window_sec=settings.bt_rl_game_window_sec)
    if not allowed:
        return None, None, _rl_err(retry_after)

    body = body or {}
    try:
        bet = int(body.get("bet", 0))
    except (TypeError, ValueError):
        return None, None, _err("invalid_bet")
    ok, np = _validate_params(name, body.get("params") or {})
    if not ok:
        return None, None, _err("invalid_params")

    # Static stake validation needs no DB: distinguish a genuinely invalid
    # stake (below the min or above the table cap) from simply not having
    # enough points — the balance check itself is server-side in bt_open_round
    # (raises insufficient_balance), so no pre-read here on the hot path.
    if bet < BET_MIN or bet > BET_MAX:
        return None, None, {"ok": False, "error": "invalid_bet"}

    # Reuse the user's active seed pair (Rainbet-style): the server & client
    # seeds persist across bets and only the per-pair nonce advances. The active
    # server_seed is NEVER returned here — only its hash — so a reused seed can't
    # be predicted; it is revealed solely on rotation (see /bt/api/game/seeds/rotate).
    pair = await db.get_seed_pair(tg_id)
    if not pair:
        # First bet ever: the seed pair FK requires a bt_users row, and a
        # brand-new user may not have one yet (this used to be covered by a
        # get_user/upsert_user pre-read on EVERY bet — now it only costs a
        # round-trip on this one-time path).
        await db.upsert_user(tg_id, user.get("username"), user.get("display_name"))
        pair = await db.create_seed_pair(tg_id, seedpair.new_pair())

    # Reserve the nonce and open the round atomically (single locked RPC), so two
    # concurrent bets can't share a nonce and a rotation can't reveal the seed a
    # bet is using. We compute the per-game state with the nonce we read; if a
    # concurrent bet advanced it first, the RPC rejects (nonce_conflict) and we
    # retry with the fresh nonce.
    state: dict = {}
    result: dict | None = None
    ss = cs = ""
    nonce = 0
    for _ in range(4):
        ss = pair["server_seed"]
        cs = pair["client_seed"]
        nonce = int(pair["nonce"])
        state = _initial_state(name, np, ss, cs, nonce)
        try:
            result = await db.open_round(tg_id, name, bet, nonce, np, state)
            break
        except InsufficientBalance:
            # A leftover open round (crashed/abandoned client, lost settle) may
            # be holding the very points needed to afford this bet — the RPC
            # debits BEFORE it would hit the one-open-round constraint, so the
            # leftover case can surface here as insufficient_balance. Void +
            # refund the straggler and retry; only report insufficient_balance
            # when there is genuinely no leftover to reclaim.
            leftover = await db.get_open_round(tg_id, name)
            if leftover:
                await _void_open_round(leftover, tg_id)
                continue
            return None, None, {"ok": False, "error": "insufficient_balance"}
        except db.OpenRoundExists:
            # A leftover round for this game is still open (or a concurrent bet
            # raced us). Void + refund the straggler and retry within the loop
            # so the user never sees open_round_exists; falls through to
            # try_again only if we exhaust the retries.
            leftover = await db.get_open_round(tg_id, name)
            if leftover:
                await _void_open_round(leftover, tg_id)
            continue
        except db.NonceConflict:
            refreshed = await db.get_seed_pair(tg_id)
            if not refreshed:
                break
            pair = refreshed
    if result is None:
        return None, None, _err("try_again", 409)

    # The full open-round row. These fields are exactly what get_round would
    # return for this round; bt_open_round has just persisted the identical row.
    rnd = {
        "id": result.get("round_id"),
        "tg_id": tg_id,
        "game": name,
        "bet": bet,
        "server_seed": ss,
        "server_hash": result.get("server_hash"),
        "client_seed": cs,
        "nonce": result.get("nonce", nonce),
        "params": np,
        "outcome": state,
        "status": "open",
    }
    # Warm the open-round cache so the first step (or the /play settle below)
    # skips a get_round read.
    _round_cache_put(rnd)

    resp_params = dict(np)
    if name == "highlow":
        resp_params["start_card"] = state["start_card"]
    resp = {
        "round_id": result.get("round_id"),
        "server_hash": result.get("server_hash"),
        "nonce": result.get("nonce"),
        "balance": result.get("balance"),
        "params": resp_params,
    }
    if name == "blackjack":
        # Only the player's full hand and the dealer's UP card are ever shown
        # at deal time — the dealer's hole card stays hidden until the round
        # ends (stand/bust/double), exactly like a real table.
        resp["player"] = state["player"]
        resp["dealer_up"] = state["dealer"][0]
    if name == "crash":
        # The client's local clock only starts once this response arrives —
        # well after the server actually stamped t0_ms — so without this the
        # client curve always lags the true server clock by the bet round
        # trip, and that fixed offset compounds under the exponential curve
        # into a large visible multiplier gap (never favors the player).
        # Ship t0_ms plus this response's own send time so the client can
        # estimate clock skew and anchor its animation to the server's true
        # elapsed time instead of a local "now" reset on arrival.
        resp["t0_ms"] = state["t0_ms"]
        resp["server_now_ms"] = int(time.time() * 1000)
    return rnd, resp, None


@app.post("/bt/api/game/{name}/bet")
async def game_bet(name: str, user: dict = Depends(require_user), body: dict | None = Body(default=None)):
    rnd, resp, err = await _open_bet(name, user, body)
    if err:
        return err
    if name == "blackjack":
        # A player natural (2-card 21) settles immediately at deal time, per
        # real-table rules — there's no hit/stand/double offered on a natural.
        tg_id = user["tg_id"]
        state = rnd["outcome"]
        player, dealer = state["player"], state["dealer"]
        nat = blackjack.natural_outcome(player, dealer)
        if nat is not None:
            outcome = {**state, "natural": True}
            final = await _finalise(rnd, tg_id, nat, "settled", outcome)
            if final.get("ok") is False:
                return final
            resp["dealer"] = dealer
            resp["multiplier"] = nat
            resp["done"] = True
            resp["outcome"] = final["outcome"]
            resp["payout"] = final["payout"]
            resp["new_balance"] = final["new_balance"]
            return resp
        resp["done"] = False
    return resp


# In-process cache of OPEN rounds, keyed by round_id. It lets a mid-game step
# skip the get_round read: a round's identity fields (seed, nonce, bet, params)
# are immutable for its whole life, and the mutable `outcome` state is written
# through on every step. The DB stays authoritative — every step still writes,
# settlement is still a guarded RPC, and on a cache miss (cold start, a second
# worker/process, or after settle) we fall back to a fresh get_round. Entries
# are dropped on settle. Bounded so a burst of rounds cannot grow it unbounded.
_ROUND_CACHE: dict[str, dict] = {}
_ROUND_CACHE_MAX = 2000


def _round_cache_put(rnd: dict) -> None:
    rid = rnd.get("id")
    if not rid:
        return
    if rid not in _ROUND_CACHE and len(_ROUND_CACHE) >= _ROUND_CACHE_MAX:
        # Crude bound: drop everything and let it refill lazily from the DB.
        # Safe because state is written through, so a miss just costs one read.
        _ROUND_CACHE.clear()
    _ROUND_CACHE[rid] = rnd


def _round_cache_pop(round_id) -> None:
    if round_id:
        _ROUND_CACHE.pop(round_id, None)


async def _load_open_round(name: str, tg_id: int, round_id):
    if not round_id:
        return None, _err("invalid_request")
    # Serve the round from the in-process cache when we already hold it, so a
    # mid-game step does a single write instead of read-then-write. Fall back to
    # a fresh read (the DB is authoritative) on any miss.
    rnd = _ROUND_CACHE.get(round_id)
    from_cache = rnd is not None
    if rnd is None:
        rnd = await db.get_round(round_id)
    if not rnd or int(rnd["tg_id"]) != tg_id or rnd["game"] != name:
        return None, _err("round_not_found", 404)
    if rnd["status"] != "open":
        return None, _err("round_not_open")
    if not from_cache:
        _round_cache_put(rnd)
    return rnd, None


async def _persist_step(rnd: dict, new_state: dict) -> bool:
    """Write a mid-game step's new state in a single guarded round trip.

    Uses a status='open' guarded PATCH so a stale cache entry (e.g. the round was
    settled by a concurrent cashout or swept as timed-out) can never revive a
    closed round: on a miss we drop the cache entry and the caller surfaces
    ``round_not_open``. On success we write the new state through to the cache so
    the next step can skip its read. Returns True if the round was still open.
    """
    updated = await db.update_open_round(rnd["id"], {"outcome": new_state})
    if updated is None:
        _round_cache_pop(rnd["id"])
        return False
    rnd["outcome"] = new_state
    _round_cache_put(rnd)
    return True


async def _finalise(rnd: dict, tg_id: int, multiplier: float, status: str, outcome: dict,
                    payout_override: int | None = None):
    """Atomically close the round (guarded on status='open') AND credit any win.

    Delegates to the ``bt_settle_round`` RPC so the close and the payout credit
    commit in ONE transaction and ONE round trip. This keeps settlement
    idempotent — a concurrent double-settle/cashout finds the round already
    closed, credits nothing, and returns an error envelope (spec §14
    'double-settle rejected') — and, unlike the previous close-then-credit pair,
    a crash can no longer land between the close and the credit (the payout is
    either committed with the close or not at all).
    """
    bet = int(rnd["bet"])
    if payout_override is not None:
        payout = int(payout_override)
    else:
        payout = _payout(bet, multiplier) if multiplier > 0 else 0
    res = await db.settle_round(rnd["id"], tg_id, outcome, payout, status)
    _round_cache_pop(rnd["id"])
    if not res.get("closed"):
        return {"ok": False, "error": "round_not_open"}
    # NB: the active server_seed is intentionally NOT returned here — it stays
    # secret across the pair's reuse and is revealed only on rotation. Rendering
    # relies on the outcome fields (e.g. mines includes the full `mines` layout).
    return {
        "outcome": outcome,
        "payout": payout,
        "new_balance": int(res.get("new_balance", 0)),
        "server_hash": rnd["server_hash"],
    }


def _bj_next_active(hands: list[dict], after: int) -> int | None:
    """Index of the next not-yet-done hand after ``after``, or None if all done."""
    for i in range(after + 1, len(hands)):
        if not hands[i].get("done"):
            return i
    return None


def _bj_split_step_resp(rnd: dict, hands: list[dict], active, aces: bool, **extra) -> dict:
    """Mid-split /step response: hands (cards only), whose turn, ace flag."""
    resp = {
        "outcome_step": {"split": True, "aces": aces, "active": active,
                         "hands": [list(h["cards"]) for h in hands]},
        "multiplier": 0.0, "can_cashout": False, "busted": False,
        "done": False, "bet": int(rnd["bet"]),
    }
    resp["outcome_step"].update(extra)
    return resp


async def _bj_settle_split(rnd: dict, tg_id: int, dealer: list[int], hands: list[dict],
                           cursor: int, ss: str, cs: str, nonce: int, aces: bool) -> dict:
    """All player hands are done — play the dealer once (only if a hand is still
    live) and resolve each hand independently against it. The credited payout is
    the SUM of the per-hand payouts (each hand keeps its own — possibly doubled —
    stake), passed to _finalise as an explicit override since two hands with
    different multipliers can't be one round-level multiplier."""
    draw = lambda i: rng_float(ss, cs, nonce, i)
    if any(not blackjack.is_bust(blackjack.hand_total(h["cards"])[0]) for h in hands):
        dealer, cursor = blackjack.play_dealer(draw, dealer, cursor)
    results, total_payout = [], 0
    for h in hands:
        m = blackjack.outcome_multiplier(h["cards"], dealer)
        results.append(m)
        if m > 0:
            total_payout += _payout(int(h["bet"]), m)
    total_bet = int(rnd["bet"])
    outcome = {"dealer": dealer, "next_cursor": cursor, "split": True, "aces": aces,
               "hands": hands, "active": None, "player_done": True, "results": results}
    final = await _finalise(rnd, tg_id, 0.0, "settled", outcome, payout_override=total_payout)
    if final.get("ok") is False:
        return final
    eff = round(total_payout / total_bet, 4) if total_bet else 0.0
    busted_all = all(blackjack.is_bust(blackjack.hand_total(h["cards"])[0]) for h in hands)
    return {"outcome_step": {"split": True, "aces": aces, "active": None,
                             "hands": [list(h["cards"]) for h in hands],
                             "dealer": dealer, "results": results},
            "multiplier": eff, "can_cashout": False, "busted": busted_all,
            "done": True, "bet": total_bet, **final}


async def _bj_split(rnd: dict, tg_id: int, state: dict, ss: str, cs: str, nonce: int) -> dict:
    """Handle the ``split`` action: divide the opening two identical-rank cards
    into two hands, deal one card to each, and atomically debit a second stake
    equal to the round's bet (bt_split_round). Split aces get one card each and
    auto-stand; a fresh split hand that makes 21 also auto-stands."""
    player = list(state.get("player") or [])
    dealer = list(state.get("dealer") or [])
    cursor = int(state.get("next_cursor", 4))
    if not blackjack.can_split(player) or state.get("doubled") or bool(state.get("player_done")):
        return _err("invalid_move")
    draw = lambda i: rng_float(ss, cs, nonce, i)
    base_bet = int(rnd["bet"])  # the single-hand stake (never doubled before a split)
    aces = blackjack.is_ace_pair(player)
    c0, c1 = player[0], player[1]
    card0 = blackjack.draw_card(draw, cursor); cursor += 1
    card1 = blackjack.draw_card(draw, cursor); cursor += 1
    hands = [
        {"cards": [c0, card0], "bet": base_bet, "doubled": False, "done": False},
        {"cards": [c1, card1], "bet": base_bet, "doubled": False, "done": False},
    ]
    for h in hands:
        t, _ = blackjack.hand_total(h["cards"])
        if aces or t == 21:
            h["done"] = True
    active = _bj_next_active(hands, -1)
    committed = {"dealer": dealer, "next_cursor": cursor, "split": True, "aces": aces,
                 "hands": hands, "active": active, "player_done": active is None}
    try:
        bal = await db.split_round(rnd["id"], tg_id, base_bet, committed)
    except InsufficientBalance:
        return _err("insufficient_balance")
    if bal is None:
        return _err("round_not_open")
    rnd["bet"] = int(bal["bet"])
    rnd["outcome"] = dict(committed)
    if active is None:
        return await _bj_settle_split(rnd, tg_id, dealer, hands, cursor, ss, cs, nonce, aces)
    resp = _bj_split_step_resp(rnd, hands, active, aces)
    resp["balance"] = bal["balance"]
    return resp


async def _bj_step_split(rnd: dict, tg_id: int, state: dict, ss: str, cs: str, nonce: int,
                         action) -> dict:
    """Hit / stand / double on the active hand of a split round, advancing to the
    next hand (and settling once both are done)."""
    if action not in ("hit", "stand", "double"):
        return _err("invalid_move")
    dealer = list(state.get("dealer") or [])
    cursor = int(state.get("next_cursor", 4))
    hands = [dict(h) for h in (state.get("hands") or [])]
    for h in hands:
        h["cards"] = list(h.get("cards") or [])
    aces = bool(state.get("aces"))
    active = state.get("active")
    if active is None or active >= len(hands) or hands[active].get("done"):
        return _err("invalid_move")
    draw = lambda i: rng_float(ss, cs, nonce, i)
    h = hands[active]

    if action == "double":
        # Double-after-split: only on a fresh two-card hand, once. Debits an
        # extra stake equal to this hand's bet and persists the committed hand
        # atomically (bt_double_round), then the hand stands with its one card.
        if len(h["cards"]) != 2 or h.get("doubled"):
            return _err("invalid_move")
        extra = int(h["bet"])
        h["cards"].append(blackjack.draw_card(draw, cursor)); cursor += 1
        h["bet"] = extra * 2
        h["doubled"] = True
        h["done"] = True
        nxt = _bj_next_active(hands, active)
        committed = {"dealer": dealer, "next_cursor": cursor, "split": True, "aces": aces,
                     "hands": hands, "active": nxt, "player_done": nxt is None}
        try:
            bal = await db.double_round(rnd["id"], tg_id, extra, committed)
        except InsufficientBalance:
            # Roll back the optimistic mutation so a retry can double again.
            return _err("insufficient_balance")
        if bal is None:
            return _err("round_not_open")
        rnd["bet"] = int(bal["bet"])
        rnd["outcome"] = dict(committed)
        if nxt is None:
            return await _bj_settle_split(rnd, tg_id, dealer, hands, cursor, ss, cs, nonce, aces)
        resp = _bj_split_step_resp(rnd, hands, nxt, aces)
        resp["balance"] = bal["balance"]
        return resp

    if action == "hit":
        if aces:
            return _err("invalid_move")  # split aces take exactly one card
        h["cards"].append(blackjack.draw_card(draw, cursor)); cursor += 1
        t, _ = blackjack.hand_total(h["cards"])
        if blackjack.is_bust(t) or t == 21:
            h["done"] = True
        nxt = active if not h.get("done") else _bj_next_active(hands, active)
        new_state = {"dealer": dealer, "next_cursor": cursor, "split": True, "aces": aces,
                     "hands": hands, "active": nxt, "player_done": nxt is None}
        if nxt is None:
            rnd["outcome"] = new_state
            return await _bj_settle_split(rnd, tg_id, dealer, hands, cursor, ss, cs, nonce, aces)
        if not await _persist_step(rnd, new_state):
            return _err("round_not_open")
        return _bj_split_step_resp(rnd, hands, nxt, aces, last_bust=blackjack.is_bust(t))

    # stand
    h["done"] = True
    nxt = _bj_next_active(hands, active)
    new_state = {"dealer": dealer, "next_cursor": cursor, "split": True, "aces": aces,
                 "hands": hands, "active": nxt, "player_done": nxt is None}
    if nxt is None:
        rnd["outcome"] = new_state
        return await _bj_settle_split(rnd, tg_id, dealer, hands, cursor, ss, cs, nonce, aces)
    if not await _persist_step(rnd, new_state):
        return _err("round_not_open")
    return _bj_split_step_resp(rnd, hands, nxt, aces)


async def _void_open_round(rnd: dict, tg_id: int) -> None:
    """Purge a leftover open round and refund its full bet — best-effort.

    Reuses the atomic ``bt_settle_round`` RPC (close + credit in one guarded
    transaction) with ``payout = bet`` and ``status = 'voided'``, so the refund
    is idempotent (a concurrent settle finds the round already closed and credits
    nothing) and crash-safe (the credit either commits with the close or not at
    all). Because the credit exactly offsets the round's ``game_bet`` outflow, the
    round nets to zero — no house profit, no phantom loss — which keeps the /stats
    house-edge math correct even though the credit is recorded as a game payout.

    Never raises: a stuck round must never surface ``open_round_exists`` to the
    user, so any failure here is logged and swallowed and the caller proceeds.
    """
    try:
        bet = int(rnd.get("bet", 0))
        await db.settle_round(rnd["id"], tg_id, rnd.get("outcome") or {}, bet, "voided")
        logger.info("open_round_voided round=%s tg_id=%s refund=%s", rnd.get("id"), tg_id, bet)
    except Exception as exc:  # noqa: BLE001 — never block a new bet on a purge failure
        logger.warning("void_open_round_failed round=%s: %s", rnd.get("id"), exc)
    finally:
        _round_cache_pop(rnd.get("id"))


@app.post("/bt/api/game/{name}/settle")
async def game_settle(name: str, user: dict = Depends(require_user), body: dict | None = Body(default=None)):
    if name not in SINGLE_SETTLE:
        return _err("invalid_action")
    tg_id = user["tg_id"]
    allowed, retry_after = ratelimit.check(f"game:{tg_id}", limit=settings.bt_rl_game_limit, window_sec=settings.bt_rl_game_window_sec)
    if not allowed:
        return _rl_err(retry_after)
    body = body or {}
    rnd, err = await _load_open_round(name, tg_id, body.get("round_id"))
    if err:
        return err

    ss, cs, nonce = rnd["server_seed"], rnd["client_seed"], int(rnd["nonce"])
    params = rnd.get("params") or {}
    if name == "dice":
        result = dice.settle(ss, cs, nonce, int(params["target"]))
    elif name == "keno":
        result = keno.play(ss, cs, nonce, list(params["picks"]))
    else:  # plinko
        result = plinko.drop(ss, cs, nonce, int(params["rows"]), str(params["risk"]))
    multiplier = result["multiplier"]
    return await _finalise(rnd, tg_id, multiplier, "settled", result)


@app.post("/bt/api/game/{name}/play")
async def game_play(name: str, user: dict = Depends(require_user), body: dict | None = Body(default=None)):
    """One-shot open+settle for single-settle games (dice, plinko).

    A single-settle game's outcome is fully determined the moment the round opens
    — there is no mid-round interaction — so opening and settling in ONE request
    saves a whole client<->server round trip versus /bet then /settle. Provably
    fair is unchanged: the active server seed stays secret (only its hash is
    returned), the nonce advances via the same atomic open_round RPC, and
    settlement is the same guarded, idempotent settle_round RPC. /bet + /settle
    remain for multi-step games and as a fallback for clients that predate /play.
    """
    if name not in SINGLE_SETTLE:
        return _err("invalid_action")
    tg_id = user["tg_id"]
    rnd, resp, err = await _open_bet(name, user, body)
    if err:
        return err

    ss, cs, nonce = rnd["server_seed"], rnd["client_seed"], int(rnd["nonce"])
    params = rnd.get("params") or {}
    if name == "dice":
        result = dice.settle(ss, cs, nonce, int(params["target"]))
    elif name == "keno":
        result = keno.play(ss, cs, nonce, list(params["picks"]))
    else:  # plinko
        result = plinko.drop(ss, cs, nonce, int(params["rows"]), str(params["risk"]))
    settled = await _finalise(rnd, tg_id, result["multiplier"], "settled", result)
    if settled.get("ok") is False:
        return settled
    # Fold the bet-side identity fields into the settle payload so the client gets
    # everything (hash + nonce for the Provably Fair panel, params, outcome,
    # payout, new_balance) from the single call.
    settled["round_id"] = resp["round_id"]
    settled["nonce"] = resp["nonce"]
    settled["params"] = resp["params"]
    return settled


@app.post("/bt/api/game/{name}/step")
async def game_step(name: str, user: dict = Depends(require_user), body: dict | None = Body(default=None)):
    if name not in MULTI_STEP and name != "blackjack":
        return _err("invalid_action")
    tg_id = user["tg_id"]
    allowed, retry_after = ratelimit.check(f"game:{tg_id}", limit=settings.bt_rl_game_limit, window_sec=settings.bt_rl_game_window_sec)
    if not allowed:
        return _rl_err(retry_after)
    body = body or {}
    rnd, err = await _load_open_round(name, tg_id, body.get("round_id"))
    if err:
        return err

    ss, cs, nonce = rnd["server_seed"], rnd["client_seed"], int(rnd["nonce"])
    params = rnd.get("params") or {}
    state = rnd.get("outcome") or {}
    move = body.get("move")

    if name == "flip":
        streak = int(state.get("streak", 0))
        u = rng_float(ss, cs, nonce, streak)
        coin = "heads" if u < flip.P_WIN else "tails"
        guess = move if move in ("heads", "tails") else "heads"
        win = guess == coin
        if not win:
            outcome = {"streak": streak, "coin": coin, "guess": guess, "busted": True}
            final = await _finalise(rnd, tg_id, 0.0, "settled", outcome)
            return {"outcome_step": outcome, "multiplier": 0.0, "can_cashout": False,
                    "busted": True, "done": True, **final}
        new_streak = streak + 1
        raw = flip.multiplier(new_streak)
        mult = min(raw, MULT_CAP)
        new_state = {"streak": new_streak, "multiplier": mult, "coin": coin, "guess": guess}
        # Auto-cash once the cap is reached: flip has no natural terminal (a
        # streak can grow forever), so without this its 1.96^streak ladder would
        # run past the economy — the same guard mines/towers apply.
        if raw >= MULT_CAP:
            outcome = {**new_state, "cleared": True}
            final = await _finalise(rnd, tg_id, mult, "cashed_out", outcome)
            return {"outcome_step": {"coin": coin, "guess": guess, "streak": new_streak},
                    "multiplier": mult, "can_cashout": True, "busted": False, "done": True, **final}
        if not await _persist_step(rnd, new_state):
            return _err("round_not_open")
        return {"outcome_step": {"coin": coin, "guess": guess, "streak": new_streak},
                "multiplier": mult, "can_cashout": True, "busted": False, "done": False}

    if name == "mines":
        m = int(params["mines"])
        revealed = list(state.get("revealed", []))
        # The client sends the move as {"cell": <idx>} (matching the other games'
        # dict-shaped moves); accept a bare int/str too for robustness.
        raw_cell = move.get("cell") if isinstance(move, dict) else move
        if not isinstance(raw_cell, (int, str)):
            return _err("invalid_move")
        try:
            cell = int(raw_cell)
        except (TypeError, ValueError):
            return _err("invalid_move")
        if cell < 0 or cell >= mines.TOTAL or cell in revealed:
            return _err("invalid_move")
        mine_set = set(mines.mine_positions(ss, cs, nonce, m))
        if cell in mine_set:
            outcome = {"revealed": revealed, "hit": cell, "mines": sorted(mine_set),
                       "mines_count": m, "busted": True}
            final = await _finalise(rnd, tg_id, 0.0, "settled", outcome)
            return {"outcome_step": {"cell": cell, "safe": False}, "multiplier": 0.0,
                    "can_cashout": False, "busted": True, "done": True, **final}
        revealed.append(cell)
        k = len(revealed)
        raw = mines.multiplier(k, m)
        mult = min(raw, MULT_CAP)
        # Auto-cash on a full clear OR once the cap is reached, so a deep-clear
        # tail can't chase a multiplier far beyond the economy.
        done = k == (mines.TOTAL - m) or raw >= MULT_CAP
        new_state = {"revealed": revealed, "mines_count": m, "multiplier": mult}
        if done:
            outcome = {**new_state, "mines": sorted(mine_set), "cleared": True}
            final = await _finalise(rnd, tg_id, mult, "cashed_out", outcome)
            return {"outcome_step": {"cell": cell, "safe": True}, "multiplier": mult,
                    "can_cashout": True, "busted": False, "done": True, **final}
        if not await _persist_step(rnd, new_state):
            return _err("round_not_open")
        return {"outcome_step": {"cell": cell, "safe": True}, "multiplier": mult,
                "can_cashout": True, "busted": False, "done": False}

    if name == "towers":
        difficulty = str(params["difficulty"])
        floor = int(state.get("floor", 0))
        # The client sends {"floor": <f>, "choice": <col>}; the authoritative floor
        # is tracked server-side, so only the chosen column is used. Accept a bare
        # int/str too for robustness.
        raw_col = move.get("choice") if isinstance(move, dict) else move
        if not isinstance(raw_col, (int, str)):
            return _err("invalid_move")
        try:
            col = int(raw_col)
        except (TypeError, ValueError):
            return _err("invalid_move")
        if col < 0 or col >= towers.columns(difficulty):
            return _err("invalid_move")
        traps = set(towers.trap_positions(ss, cs, nonce, floor, difficulty))
        if col in traps:
            outcome = {"floor": floor, "col": col, "traps": sorted(traps),
                       "difficulty": difficulty, "busted": True}
            final = await _finalise(rnd, tg_id, 0.0, "settled", outcome)
            return {"outcome_step": {"floor": floor, "col": col, "safe": False}, "multiplier": 0.0,
                    "can_cashout": False, "busted": True, "done": True, **final}
        new_floor = floor + 1
        raw = towers.multiplier(new_floor, difficulty)
        mult = min(raw, MULT_CAP)
        # Auto-cash on the top floor OR once the cap is reached, so hard's
        # doubling ladder can't run away to ~253x at the top.
        done = new_floor >= towers.FLOORS or raw >= MULT_CAP
        new_state = {"floor": new_floor, "difficulty": difficulty, "multiplier": mult}
        if done:
            outcome = {**new_state, "cleared": True}
            final = await _finalise(rnd, tg_id, mult, "cashed_out", outcome)
            return {"outcome_step": {"floor": floor, "col": col, "safe": True}, "multiplier": mult,
                    "can_cashout": True, "busted": False, "done": True, **final}
        if not await _persist_step(rnd, new_state):
            return _err("round_not_open")
        return {"outcome_step": {"floor": floor, "col": col, "safe": True}, "multiplier": mult,
                "can_cashout": True, "busted": False, "done": False}

    if name == "chicken":
        difficulty = str(params["difficulty"])
        lane = int(state.get("lane", 0))
        # The client sends the crossing zone as {"zone": <idx>} (matching the
        # other games' dict-shaped moves); accept a bare int/str too.
        raw_zone = move.get("zone") if isinstance(move, dict) else move
        if not isinstance(raw_zone, (int, str)):
            return _err("invalid_move")
        try:
            zone = int(raw_zone)
        except (TypeError, ValueError):
            return _err("invalid_move")
        if zone < 0 or zone >= chicken.zones(difficulty, lane):
            return _err("invalid_move")
        cars = set(chicken.car_zones(ss, cs, nonce, lane, difficulty))
        if zone in cars:
            outcome = {"lane": lane, "zone": zone, "cars": sorted(cars),
                       "difficulty": difficulty, "busted": True}
            final = await _finalise(rnd, tg_id, 0.0, "settled", outcome)
            return {"outcome_step": {"lane": lane, "zone": zone, "safe": False}, "multiplier": 0.0,
                    "can_cashout": False, "busted": True, "done": True, **final}
        new_lane = lane + 1
        raw = chicken.multiplier(new_lane, difficulty)
        mult = min(raw, chicken.CHICKEN_MAX_MULT)
        # Auto-cash on the far side of the road OR once the 24x cap is reached
        # (they coincide on the easy ladder; the cap also guards legacy rounds).
        done = new_lane >= chicken.LANES or raw >= chicken.CHICKEN_MAX_MULT
        new_state = {"lane": new_lane, "difficulty": difficulty, "multiplier": mult}
        if done:
            outcome = {**new_state, "cleared": True}
            final = await _finalise(rnd, tg_id, mult, "cashed_out", outcome)
            return {"outcome_step": {"lane": lane, "zone": zone, "safe": True}, "multiplier": mult,
                    "can_cashout": True, "busted": False, "done": True, **final}
        if not await _persist_step(rnd, new_state):
            return _err("round_not_open")
        return {"outcome_step": {"lane": lane, "zone": zone, "safe": True}, "multiplier": mult,
                "can_cashout": True, "busted": False, "done": False}

    if name == "blackjack":
        action = move.get("action") if isinstance(move, dict) else move
        if state.get("split"):
            return await _bj_step_split(rnd, tg_id, state, ss, cs, nonce, action)
        if action == "split":
            return await _bj_split(rnd, tg_id, state, ss, cs, nonce)
        if action not in ("hit", "stand", "double"):
            return _err("invalid_move")
        player = list(state.get("player") or [])
        dealer = list(state.get("dealer") or [])
        cursor = int(state.get("next_cursor", 4))
        doubled = bool(state.get("doubled", False))
        draw = lambda i: rng_float(ss, cs, nonce, i)

        if action == "double":
            # Only allowed as the very first decision (2-card hand, no hits
            # yet, not already doubled) — matches the task's no-splitting,
            # single-double-per-hand v1 scope.
            if len(player) != 2 or doubled:
                return _err("invalid_move")
            # Compute the POST-double committed state (drawn card, advanced
            # cursor, doubled/player_done flags) BEFORE the atomic debit, so
            # bt_double_round persists it together with the bet increase. If
            # the request dies between the RPC and settlement, the open round
            # is replay-safe: a retried double is rejected (doubled=true,
            # 3-card hand), and the stale sweeper / a stand settles exactly
            # the hand this double committed to — no re-charge, no drift.
            player.append(blackjack.draw_card(draw, cursor))
            cursor += 1
            p_total, _ = blackjack.hand_total(player)
            busted_now = blackjack.is_bust(p_total)
            committed = {"player": player, "dealer": dealer, "next_cursor": cursor,
                         "doubled": True, "player_done": True}
            if busted_now:
                committed["busted"] = True
            try:
                bal = await db.double_round(rnd["id"], tg_id, int(rnd["bet"]), committed)
            except InsufficientBalance:
                return _err("insufficient_balance")
            if bal is None:
                return _err("round_not_open")
            rnd["bet"] = int(bal["bet"])
            rnd["outcome"] = dict(committed)  # keep the cached round consistent
            if busted_now:
                final = await _finalise(rnd, tg_id, 0.0, "settled", committed)
                # `bet` = the DOUBLED stake: the client session tracker uses a
                # settle's declared final stake (it only knew the original bet
                # at open time, and the wager doubled mid-round).
                return {"outcome_step": {"player": player, "busted": True}, "multiplier": 0.0,
                        "can_cashout": False, "busted": True, "done": True,
                        "balance": bal["balance"], "bet": int(rnd["bet"]), **final}
            dealer, cursor = blackjack.play_dealer(draw, dealer, cursor)
            mult = blackjack.outcome_multiplier(player, dealer)
            outcome = {"player": player, "dealer": dealer, "next_cursor": cursor,
                       "doubled": True, "player_done": True}
            final = await _finalise(rnd, tg_id, mult, "settled", outcome)
            return {"outcome_step": {"player": player, "dealer": dealer}, "multiplier": mult,
                    "can_cashout": False, "busted": False, "done": True,
                    "balance": bal["balance"], "bet": int(rnd["bet"]), **final}

        if action == "hit":
            # A committed double (or any player_done state) takes no more
            # cards — matters if a double's settlement was interrupted after
            # bt_double_round persisted the doubled hand: the only valid
            # replays are stand (settles the committed hand) or the sweeper.
            if doubled or bool(state.get("player_done")):
                return _err("invalid_move")
            player.append(blackjack.draw_card(draw, cursor))
            cursor += 1
            p_total, _ = blackjack.hand_total(player)
            if blackjack.is_bust(p_total):
                outcome = {"player": player, "dealer": dealer, "next_cursor": cursor,
                           "doubled": doubled, "player_done": True, "busted": True}
                final = await _finalise(rnd, tg_id, 0.0, "settled", outcome)
                return {"outcome_step": {"player": player, "busted": True}, "multiplier": 0.0,
                        "can_cashout": False, "busted": True, "done": True,
                        "bet": int(rnd["bet"]), **final}
            if p_total == 21:
                # Auto-settle: player hit exactly 21 (wins at 2x, dealer plays out).
                dealer, cursor = blackjack.play_dealer(draw, dealer, cursor)
                mult = blackjack.outcome_multiplier(player, dealer)
                outcome = {"player": player, "dealer": dealer, "next_cursor": cursor,
                           "doubled": doubled, "player_done": True}
                final = await _finalise(rnd, tg_id, mult, "settled", outcome)
                return {"outcome_step": {"player": player, "dealer": dealer}, "multiplier": mult,
                        "can_cashout": False, "busted": False, "done": True,
                        "bet": int(rnd["bet"]), **final}
            new_state = {"player": player, "dealer": dealer, "next_cursor": cursor,
                         "doubled": doubled, "player_done": False}
            if not await _persist_step(rnd, new_state):
                return _err("round_not_open")
            return {"outcome_step": {"player": player}, "multiplier": 0.0,
                    "can_cashout": False, "busted": False, "done": False}

        # stand — also the recovery path for a committed-but-unsettled double,
        # so the settle response declares the round's final stake (`bet`).
        dealer, cursor = blackjack.play_dealer(draw, dealer, cursor)
        mult = blackjack.outcome_multiplier(player, dealer)
        outcome = {"player": player, "dealer": dealer, "next_cursor": cursor,
                   "doubled": doubled, "player_done": True}
        final = await _finalise(rnd, tg_id, mult, "settled", outcome)
        return {"outcome_step": {"player": player, "dealer": dealer}, "multiplier": mult,
                "can_cashout": False, "busted": False, "done": True,
                "bet": int(rnd["bet"]), **final}

    if name == "rps":
        # The client sends {"hand": "rock"|"paper"|"scissors"}; accept a bare
        # string too for robustness. One draw per round, cursor = round index
        # (ties consume a draw too, keeping every draw provably deterministic).
        pick = move.get("hand") if isinstance(move, dict) else move
        if pick not in rps.HANDS:
            return _err("invalid_move")
        step_n = int(state.get("step", 0))
        wins = int(state.get("wins", 0))
        player = rps.HANDS.index(pick)
        house = rps.house_hand(rng_float(ss, cs, nonce, step_n))
        house_name = rps.HANDS[house]
        if player == house:
            # Tie — neutral replay: chain multiplier unchanged, fresh draw next.
            cur_mult = min(rps.multiplier(wins), rps.RPS_MAX_MULT)
            new_state = {"step": step_n + 1, "wins": wins, "multiplier": cur_mult}
            if not await _persist_step(rnd, new_state):
                return _err("round_not_open")
            return {"outcome_step": {"house": house_name, "pick": pick, "tie": True,
                                     "win": False, "wins": wins},
                    "multiplier": cur_mult, "tie": True, "can_cashout": wins >= 1,
                    "busted": False, "done": False}
        if not rps.beats(player, house):
            outcome = {"step": step_n, "wins": wins, "house": house_name,
                       "pick": pick, "busted": True}
            final = await _finalise(rnd, tg_id, 0.0, "settled", outcome)
            return {"outcome_step": {"house": house_name, "pick": pick, "tie": False,
                                     "win": False, "wins": wins},
                    "multiplier": 0.0, "can_cashout": False, "busted": True,
                    "done": True, **final}
        new_wins = wins + 1
        raw = rps.multiplier(new_wins)
        mult = min(raw, rps.RPS_MAX_MULT)
        # Auto-cash once the cap is reached (the 5th straight win lands on 20x).
        done = raw >= rps.RPS_MAX_MULT
        new_state = {"step": step_n + 1, "wins": new_wins, "multiplier": mult}
        if done:
            outcome = {**new_state, "capped": True}
            final = await _finalise(rnd, tg_id, mult, "cashed_out", outcome)
            return {"outcome_step": {"house": house_name, "pick": pick, "tie": False,
                                     "win": True, "wins": new_wins},
                    "multiplier": mult, "can_cashout": True, "busted": False,
                    "done": True, **final}
        if not await _persist_step(rnd, new_state):
            return _err("round_not_open")
        return {"outcome_step": {"house": house_name, "pick": pick, "tie": False,
                                 "win": True, "wins": new_wins},
                "multiplier": mult, "can_cashout": True, "busted": False, "done": False}

    # highlow — the client sends the direction as {"guess": ...} (matching the
    # other games' dict-shaped moves); accept a bare string too for robustness.
    guess = move.get("guess") if isinstance(move, dict) else move
    r = int(state.get("rank", 0))
    step_n = int(state.get("step", 0))
    cur_mult = float(state.get("multiplier", 1.0))
    skips = int(state.get("skips", 0))
    picks = int(state.get("picks", 0))
    draw = lambda i: rng_float(ss, cs, nonce, i)
    slot = step_n + 1
    # Skip: swap the current decision card for a fresh one without wagering. The
    # chain multiplier is unchanged and a new slot is consumed so the draw stays
    # provably deterministic. Capped at 5 skips in a row — after that a side must
    # be picked; a real pick resets the allowance.
    if guess == "skip" or (isinstance(move, dict) and move.get("skip")):
        if skips >= 5:
            return _err("skip_limit")
        new_rank = highlow.current_card(draw, slot)
        new_skips = skips + 1
        new_state = {"rank": new_rank, "step": step_n + 1, "multiplier": cur_mult,
                     "skips": new_skips, "picks": picks}
        if not await _persist_step(rnd, new_state):
            return _err("round_not_open")
        return {"outcome_step": {"current": new_rank, "prev": r, "guess": "skip",
                                 "skipped": True, "win": True, "skips": new_skips},
                "multiplier": cur_mult, "can_cashout": picks >= 1, "skips": new_skips,
                "busted": False, "done": False}
    if (guess not in ("higher", "lower") or not highlow.can_pick(guess, r)
            or not highlow.within_cap(cur_mult, guess, r)):
        return _err("invalid_move")
    drawn = highlow.reveal_card(draw, slot)  # revealed card (full 1..13 deck)
    win = highlow.resolve(guess, r, drawn)
    if not win:
        outcome = {"rank": r, "drawn": drawn, "guess": guess, "step": step_n, "busted": True}
        final = await _finalise(rnd, tg_id, 0.0, "settled", outcome)
        return {"outcome_step": {"drawn": drawn, "prev": r, "guess": guess, "win": False},
                "multiplier": 0.0, "can_cashout": False, "busted": True, "done": True, **final}
    new_mult = cur_mult * highlow.step_multiplier(guess, r)
    # Wild reveal (Ace/King) passes through to the next non-wild card, which
    # becomes the new current decision card; a normal card is itself the current.
    new_rank = drawn if not highlow.is_wild(drawn) else highlow.current_card(draw, slot, start_j=1)
    # A real pick resets the skip allowance and counts toward the cashout minimum.
    new_state = {"rank": new_rank, "step": step_n + 1, "multiplier": new_mult,
                 "skips": 0, "picks": picks + 1}
    if not await _persist_step(rnd, new_state):
        return _err("round_not_open")
    return {"outcome_step": {"drawn": drawn, "current": new_rank, "prev": r, "guess": guess, "win": True},
            "multiplier": new_mult, "can_cashout": True, "skips": 0, "busted": False, "done": False}


@app.post("/bt/api/game/{name}/cashout")
async def game_cashout(name: str, user: dict = Depends(require_user), body: dict | None = Body(default=None)):
    # Crash has no /step — /cashout is its ONLY in-round action, so it is
    # allowed here alongside the multi-step games. Blackjack is the opposite:
    # it has /step but NO /cashout — a hand is always played to a stand/bust/
    # double (there's no partial-progress multiplier to bank early).
    if name not in MULTI_STEP and name != "crash":
        return _err("invalid_action")
    tg_id = user["tg_id"]
    allowed, retry_after = ratelimit.check(f"game:{tg_id}", limit=settings.bt_rl_game_limit, window_sec=settings.bt_rl_game_window_sec)
    if not allowed:
        return _rl_err(retry_after)
    body = body or {}
    rnd, err = await _load_open_round(name, tg_id, body.get("round_id"))
    if err:
        return err

    params = rnd.get("params") or {}
    state = rnd.get("outcome") or {}
    ss, cs, nonce = rnd["server_seed"], rnd["client_seed"], int(rnd["nonce"])
    if name == "flip":
        streak = int(state.get("streak", 0))
        mult = min(flip.multiplier(streak), MULT_CAP)
        outcome = {"streak": streak, "multiplier": mult}
    elif name == "mines":
        m = int(params["mines"])
        revealed = list(state.get("revealed", []))
        # Must reveal at least one safe tile before cashing out — no bet-then-
        # immediate-cashout (which would otherwise pay the 1.0x baseline back).
        if not revealed:
            return _err("must_reveal_first")
        mult = min(mines.multiplier(len(revealed), m), MULT_CAP)
        # Include the full mine layout so the UI can reveal the board on cashout.
        # The active server_seed stays secret (revealed only on rotation), so we
        # disclose the computed layout directly rather than the seed.
        mine_set = sorted(mines.mine_positions(ss, cs, nonce, m))
        outcome = {"revealed": revealed, "mines_count": m, "multiplier": mult, "mines": mine_set}
    elif name == "towers":
        difficulty = str(params["difficulty"])
        floor = int(state.get("floor", 0))
        # Must climb at least one floor before cashing out (no bet-then-cashout).
        if floor <= 0:
            return _err("must_climb_first")
        mult = min(towers.multiplier(floor, difficulty), MULT_CAP)
        outcome = {"floor": floor, "difficulty": difficulty, "multiplier": mult}
    elif name == "rps":
        wins = int(state.get("wins", 0))
        # Must win at least one round before cashing out (ties don't count), so
        # a bet can't be cashed straight back at the 1.0x baseline.
        if wins < 1:
            return _err("must_win_first")
        mult = min(rps.multiplier(wins), rps.RPS_MAX_MULT)
        outcome = {"wins": wins, "step": int(state.get("step", 0)), "multiplier": mult}
    elif name == "chicken":
        difficulty = str(params["difficulty"])
        lane = int(state.get("lane", 0))
        # Must cross at least one lane before cashing out (no bet-then-cashout).
        if lane <= 0:
            return _err("must_cross_first")
        mult = min(chicken.multiplier(lane, difficulty), chicken.CHICKEN_MAX_MULT)
        outcome = {"lane": lane, "difficulty": difficulty, "multiplier": mult}
    elif name == "crash":
        # Solo crash, server-clocked: the crash point AND the round clock (t0)
        # were fixed at bet time. The round has already crashed once the
        # server-clock multiplier reaches cp — a claim arriving after that
        # instant busts no matter what it says. A live claim wins at
        # min(claim, server multiplier): the clamp means a client claiming
        # AHEAD of the clock (e.g. an instant 24.9x) just gets the clock's
        # value, so timing fraud is impossible. cp is recomputed from the seed
        # (the stored copy is audit-only); cp is clamped to CRASH_CAP so the
        # curve always crashes at or below the economy ceiling.
        try:
            m = float(body.get("mult_at_cashout", 0))
        except (TypeError, ValueError):
            return _err("invalid_move")
        if not math.isfinite(m) or m < 1.0:
            return _err("invalid_move")
        cp = crash.crash_point(ss, cs, nonce)
        elapsed_ms = time.time() * 1000 - float(state.get("t0_ms") or 0)
        m_server = crash.mult_at(elapsed_ms)
        if m >= cp or m_server >= cp:
            outcome = {"crash_point": cp, "mult_at_cashout": m,
                       "busted": True, "multiplier": 0.0}
            return await _finalise(rnd, tg_id, 0.0, "settled", outcome)
        mult = min(m, m_server, crash.CRASH_CAP)
        outcome = {"crash_point": cp, "multiplier": mult}
    else:  # highlow
        step_n = int(state.get("step", 0))
        # Must make at least one real pick before cashing out (skips don't count),
        # so a bet can't be cashed straight back at the 1.0x baseline.
        if int(state.get("picks", 0)) < 1:
            return _err("must_pick_first")
        mult = min(float(state.get("multiplier", 1.0)), MULT_CAP)
        outcome = {"step": step_n, "rank": int(state.get("rank", 0)), "multiplier": mult}

    return await _finalise(rnd, tg_id, mult, "cashed_out", outcome)


@app.post("/bt/api/game/crash/check")
async def crash_check(user: dict = Depends(require_user), body: dict | None = Body(default=None)):
    """Liveness poll for an open crash round — how the client learns of a crash.

    The round autonomously crashes the instant the server-clock multiplier
    reaches the (secret) crash point. The client polls this while its curve is
    rising: once crashed, the round is settled here at payout 0 with the crash
    point revealed (the same settle a too-late /cashout would produce), so the
    UI can drop the curve at the true crash moment instead of only discovering
    it on a failed cashout. While alive it returns ONLY the server-clock
    multiplier — a value the client already computes itself — never anything
    derived from the crash point (no time-remaining, no hints)."""
    tg_id = user["tg_id"]
    # Separate, looser bucket than game actions: the client polls ~1/s for the
    # life of the curve (up to ~54s to the cap), which would exhaust the shared
    # 60/min game bucket and starve the cashout itself.
    allowed, retry_after = ratelimit.check(f"crashchk:{tg_id}", limit=180, window_sec=60)
    if not allowed:
        return _rl_err(retry_after)
    body = body or {}
    rnd, err = await _load_open_round("crash", tg_id, body.get("round_id"))
    if err:
        return err
    state = rnd.get("outcome") or {}
    ss, cs, nonce = rnd["server_seed"], rnd["client_seed"], int(rnd["nonce"])
    cp = crash.crash_point(ss, cs, nonce)
    elapsed_ms = time.time() * 1000 - float(state.get("t0_ms") or 0)
    m_server = crash.mult_at(elapsed_ms)
    if m_server >= cp:
        outcome = {"crash_point": cp, "busted": True, "multiplier": 0.0}
        final = await _finalise(rnd, tg_id, 0.0, "settled", outcome)
        return {"crashed": True, "multiplier": 0.0, **final}
    return {"ok": True, "crashed": False,
            "multiplier": min(m_server, crash.CRASH_CAP)}

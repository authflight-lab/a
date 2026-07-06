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
from collections import defaultdict
from datetime import datetime, timedelta, timezone

from fastapi import Body, Depends, FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from . import db, notify, ratelimit
from .auth import require_user
from .config import settings
from .db import InsufficientBalance, SupabaseNotConfigured
from .game import BET_MAX, BET_MIN, GAMES, MULT_CAP, MULTI_STEP, P_MAX, SINGLE_SETTLE
from .game import dice, flip, highlow, mines, plinko, seedpair, towers
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
        if name not in MULTI_STEP:
            # Single-settle games (dice/plinko) that somehow stayed open: abandon.
            await db.close_round(rnd["id"], {
                "outcome": state, "payout": 0, "status": "abandoned",
                "settled_at": db._now(),
            })
            return

        if name == "flip":
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
        else:  # highlow
            step_n = int(state.get("step", 0))
            if step_n == 0:
                mult, outcome = 0.0, {"step": 0, "multiplier": 0.0}
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

logger = logging.getLogger("bt.api")

# CORS: explicit allowlist = BT_APP_ORIGIN only. Never a wildcard (spec §8).
_origins = [settings.bt_app_origin] if settings.bt_app_origin else []
app.add_middleware(
    CORSMiddleware,
    allow_origins=_origins,
    allow_credentials=False,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["X-Telegram-Init-Data", "Content-Type"],
)


@app.middleware("http")
async def _ip_rate_limit_middleware(request: Request, call_next):
    """Pre-auth IP-level guard: 120 req / 60 s per IP across all /bt/api/ routes."""
    if request.url.path.startswith("/bt/api/"):
        xff = request.headers.get("X-Forwarded-For")
        ip = xff.split(",")[0].strip() if xff else (
            request.client.host if request.client else "unknown"
        )
        allowed, retry_after = ratelimit.check(f"ip:{ip}", limit=120, window_sec=60)
        if not allowed:
            return JSONResponse(
                status_code=429,
                content={"ok": False, "error": "rate_limited"},
                headers={"Retry-After": str(retry_after)},
            )
    return await call_next(request)


@app.exception_handler(SupabaseNotConfigured)
async def _supabase_not_configured(_request: Request, _exc: SupabaseNotConfigured):
    return JSONResponse(status_code=503, content={"ok": False, "error": "supabase_not_configured"})


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


def _period() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m")


def _period_start() -> str:
    now = datetime.now(timezone.utc)
    return now.replace(day=1, hour=0, minute=0, second=0, microsecond=0).isoformat()


# Beginning of time — used as an "all-time" lower bound for ledger scans.
_EPOCH = "1970-01-01T00:00:00+00:00"


def _week_start() -> str:
    """Start of the current UTC week (Monday 00:00:00 UTC), ISO-formatted."""
    now = datetime.now(timezone.utc)
    monday = now - timedelta(days=now.weekday())
    return monday.replace(hour=0, minute=0, second=0, microsecond=0).isoformat()


def _daily_claim(streak: int) -> int:
    """D(s) = floor(20 * (1 + 1.5 * (1 - e^(-s/10))))  (spec §5)."""
    return int(20 * (1 + 1.5 * (1 - math.exp(-streak / 10))))


def _payout(bet: int, multiplier: float) -> int:
    """bet * multiplier, floored, capped at P_MAX (spec §6)."""
    return min(int(bet * multiplier), P_MAX)


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
# Account / quests
# ---------------------------------------------------------------------------

@app.get("/bt/api/me")
async def me(user: dict = Depends(require_user)):
    tg_id = user["tg_id"]
    allowed, retry_after = ratelimit.check(f"me:{tg_id}", limit=60, window_sec=60)
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
    day = _today()
    quest = await db.get_quest(tg_id, day) or {"day": day, "chatted": False, "claimed": False}
    chatted = bool(quest.get("chatted", False))
    claimed = bool(quest.get("claimed", False))
    meta = u.get("meta") or {}
    raw_backlog = int(u.get("backlog_pts", 0))
    bal = int(u.get("balance", 0))

    # Stats card: only computed when backlog is cleared (registered & claimed).
    activity: dict | None = None
    if raw_backlog == 0:
        try:
            us, rr = await asyncio.gather(
                db.user_stats(tg_id),
                db.rich_rank(tg_id, bal),
                return_exceptions=True,
            )
            if isinstance(us, dict) and not isinstance(rr, Exception):
                activity = {
                    "messages_sent": int(us.get("messages_sent", 0)),
                    "amount_wagered": int(us.get("amount_wagered", 0)),
                    "messages_rank": int(us.get("messages_rank", 1)),
                    "rich_rank": int(rr),
                }
        except Exception:
            pass

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
    allowed, retry_after = ratelimit.check(f"rewards:{tg_id}", limit=20, window_sec=60)
    if not allowed:
        return _rl_err(retry_after)
    period = _period()
    items = await db.list_rewards(active_only=True)
    out = []
    for r in items:
        limit = int(r.get("monthly_limit", 0))
        if limit == 0:
            remaining = None  # unlimited
        else:
            used = await db.get_reward_usage(r["id"], period)
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

    # Activity floor first (spec §5): today's chatted AND claimed (UTC).
    quest = await db.get_quest(tg_id, _today())
    if not (quest and quest.get("chatted") and quest.get("claimed")):
        return {"ok": False, "error": "activity_floor_not_met"}

    # Atomic debit + monthly-usage increment + redemption row (spec §14) via the
    # bt_redeem RPC — reward/active/monthly-limit/balance are all enforced inside
    # one transaction, closing the read-then-write race on monthly_limit.
    try:
        result = await db.redeem(tg_id, reward_id, _period())
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
        if log_chat:
            rw = await db.get_reward(reward_id)
            title = (rw or {}).get("title") or "reward"
            limit = int((rw or {}).get("monthly_limit", 0))
            if limit <= 0:
                stock = "Unlimited"
            else:
                # bt_redeem already incremented usage for THIS request, so add
                # it back to show the pre-claim stock the "(-1 after claimed)"
                # wording refers to.
                used = await db.get_reward_usage(reward_id, _period())
                stock = str(max(0, limit - used + 1))
            name_parts = (user.get("display_name") or "").split()
            first_name = name_parts[0] if name_parts else str(tg_id)
            link = notify.profile_link(tg_id, user.get("username"), first_name)
            await notify.send_dm(
                log_chat,
                f"🎟️ <b>Redemption request</b>\n\n"
                f"User: {link}\n"
                f"Prize: <code>{notify.esc(title)}</code>\n"
                f"Current Stock: {notify.esc(stock)} (-1 after claimed)",
            )
    except Exception as e:
        logger.warning("bt_redeem_log_error", error=str(e))

    return {
        "ok": True,
        "redemption_id": result.get("redemption_id"),
        "new_balance": result.get("new_balance"),
    }


# ---------------------------------------------------------------------------
# Leaderboard / history
# ---------------------------------------------------------------------------

async def _rows_from_totals(
    totals: dict[int, int], tg_id: int, limit: int = 20
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
    allowed, retry_after = ratelimit.check(f"leaderboard:{tg_id}", limit=20, window_sec=60)
    if not allowed:
        return _rl_err(retry_after)
    if tab not in ("rich", "chatters"):
        return _err("invalid_tab")
    if period not in ("weekly", "alltime"):
        period = "weekly"

    if tab == "rich":
        if period == "alltime":
            # All-time rich list = current balances.
            top = await db.leaderboard_rich(limit=20)
            rows = [
                {"rank": i + 1, "tg_id": int(r["tg_id"]),
                 "display_name": r.get("display_name") or str(r["tg_id"]),
                 "value": int(r.get("balance", 0))}
                for i, r in enumerate(top)
            ]
            u = await db.get_user_cached(tg_id)
            you = None
            if u is not None:
                bal = int(u.get("balance", 0))
                you = {"rank": await db.rich_rank(tg_id, bal), "value": bal}
            return {"tab": tab, "period": period, "rows": rows, "you": you}

        # Weekly rich = net points gained this week (sum of all ledger amounts),
        # excluding the weekly bonus itself so past winners don't get a head start.
        ledger = await db.ledger_since(_week_start(), exclude_kind="weekly_bonus")
        totals: dict[int, int] = defaultdict(int)
        for row in ledger:
            totals[int(row["tg_id"])] += int(row["amount"])
        rows, you = await _rows_from_totals(totals, tg_id)
        return {"tab": tab, "period": period, "rows": rows, "you": you}

    # chatters: raw messages sent (ranked by message count, not points earned).
    start_day = _week_start()[:10] if period == "weekly" else _EPOCH[:10]
    counts = await db.chat_counts_since(start_day)
    totals = defaultdict(int)
    for row in counts:
        totals[int(row["tg_id"])] += int(row["count"])
    rows, you = await _rows_from_totals(totals, tg_id)
    return {"tab": tab, "period": period, "rows": rows, "you": you}


@app.get("/bt/api/history")
async def history(user: dict = Depends(require_user)):
    tg_id = user["tg_id"]
    allowed, retry_after = ratelimit.check(f"history:{tg_id}", limit=20, window_sec=60)
    if not allowed:
        return _rl_err(retry_after)
    rows = await db.ledger_history(tg_id, limit=50)
    return {"rows": [
        {"id": r.get("id"), "amount": int(r.get("amount", 0)), "kind": r.get("kind"),
         "ref": r.get("ref"), "created_at": r.get("created_at")}
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
    if name == "plinko":
        rows = int(p.get("rows", 0))
        risk = str(p.get("risk", ""))
        return (plinko.valid_rows(rows) and plinko.valid_risk(risk), {"rows": rows, "risk": risk})
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
    return {}


@app.get("/bt/api/game/seeds")
async def game_seeds(user: dict = Depends(require_user)):
    """The user's active seed pair (public view). Creates one on first access.
    NEVER exposes the active server_seed — only its hash and the next hash."""
    tg_id = user["tg_id"]
    # Bootstrap the user row first: bt_seed_pairs.tg_id has an FK to bt_users, so
    # a brand-new user hitting this endpoint before /me or a bet would otherwise
    # fail on the seed-pair insert.
    if await db.get_user(tg_id) is None:
        await db.upsert_user(tg_id, user.get("username"), user.get("display_name"))
    pair = await db.get_seed_pair(tg_id)
    if not pair:
        pair = await db.create_seed_pair(tg_id, seedpair.new_pair())
    return {"ok": True, **seedpair.public_view(pair)}


@app.post("/bt/api/game/seeds/rotate")
async def game_seeds_rotate(user: dict = Depends(require_user), body: dict | None = Body(default=None)):
    """Rotate the active pair: reveal the retired server_seed, promote the
    pre-committed next one, apply an optional new client seed, reset the nonce."""
    tg_id = user["tg_id"]
    allowed, retry_after = ratelimit.check(f"game:{tg_id}", limit=60, window_sec=60)
    if not allowed:
        return _rl_err(retry_after)
    body = body or {}
    new_client_seed = str(body.get("client_seed") or "").strip()
    # Ensure a pair exists, then rotate it atomically. The RPC locks the seed-pair
    # row and refuses if ANY round is open (checked under that lock), so revealing
    # the active server_seed can never race a concurrent bet that would still use it.
    # Bootstrap the user row first (bt_seed_pairs.tg_id FK -> bt_users).
    if await db.get_user(tg_id) is None:
        await db.upsert_user(tg_id, user.get("username"), user.get("display_name"))
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


@app.post("/bt/api/game/{name}/bet")
async def game_bet(name: str, user: dict = Depends(require_user), body: dict | None = Body(default=None)):
    if name not in GAMES:
        return _err("unknown_game", 404)
    tg_id = user["tg_id"]
    allowed, retry_after = ratelimit.check(f"game:{tg_id}", limit=60, window_sec=60)
    if not allowed:
        return _rl_err(retry_after)

    body = body or {}
    try:
        bet = int(body.get("bet", 0))
    except (TypeError, ValueError):
        return _err("invalid_bet")
    ok, np = _validate_params(name, body.get("params") or {})
    if not ok:
        return _err("invalid_params")

    u = await db.get_user(tg_id)
    if u is None:
        u = await db.upsert_user(tg_id, user.get("username"), user.get("display_name")) or {}
    balance = int(u.get("balance", 0))
    max_bet = min(BET_MAX, balance)
    if bet < BET_MIN or bet > max_bet:
        return {"ok": False, "error": "invalid_bet"}

    # One open round per (user, game) (unique index bt_one_open_round is the backstop).
    if await db.get_open_round(tg_id, name):
        return {"ok": False, "error": "open_round_exists"}

    # Reuse the user's active seed pair (Rainbet-style): the server & client
    # seeds persist across bets and only the per-pair nonce advances. The active
    # server_seed is NEVER returned here — only its hash — so a reused seed can't
    # be predicted; it is revealed solely on rotation (see /bt/api/game/seeds/rotate).
    pair = await db.get_seed_pair(tg_id)
    if not pair:
        pair = await db.create_seed_pair(tg_id, seedpair.new_pair())

    # Reserve the nonce and open the round atomically (single locked RPC), so two
    # concurrent bets can't share a nonce and a rotation can't reveal the seed a
    # bet is using. We compute the per-game state with the nonce we read; if a
    # concurrent bet advanced it first, the RPC rejects (nonce_conflict) and we
    # retry with the fresh nonce.
    state: dict = {}
    result: dict | None = None
    for _ in range(4):
        ss = pair["server_seed"]
        cs = pair["client_seed"]
        nonce = int(pair["nonce"])
        state = _initial_state(name, np, ss, cs, nonce)
        try:
            result = await db.open_round(tg_id, name, bet, nonce, np, state)
            break
        except InsufficientBalance:
            return {"ok": False, "error": "insufficient_balance"}
        except db.OpenRoundExists:
            return {"ok": False, "error": "open_round_exists"}
        except db.NonceConflict:
            refreshed = await db.get_seed_pair(tg_id)
            if not refreshed:
                break
            pair = refreshed
    if result is None:
        return _err("try_again", 409)

    resp_params = dict(np)
    if name == "highlow":
        resp_params["start_card"] = state["start_card"]
    return {
        "round_id": result.get("round_id"),
        "server_hash": result.get("server_hash"),
        "nonce": result.get("nonce"),
        "balance": result.get("balance"),
        "params": resp_params,
    }


async def _load_open_round(name: str, tg_id: int, round_id):
    if not round_id:
        return None, _err("invalid_request")
    rnd = await db.get_round(round_id)
    if not rnd or int(rnd["tg_id"]) != tg_id or rnd["game"] != name:
        return None, _err("round_not_found", 404)
    if rnd["status"] != "open":
        return None, _err("round_not_open")
    return rnd, None


async def _finalise(rnd: dict, tg_id: int, multiplier: float, status: str, outcome: dict):
    """Close the round FIRST (guarded on status='open'), then credit any win.

    Ordering matters: closing before crediting makes settlement idempotent, so
    two concurrent settle/cashout calls for the same round cannot both pay out
    (spec §14 'double-settle rejected'). If another request already closed it we
    return an error envelope and do NOT credit. A crash between close and credit
    forfeits the payout rather than paying it twice.
    """
    bet = int(rnd["bet"])
    payout = _payout(bet, multiplier) if multiplier > 0 else 0
    closed = await db.close_round(rnd["id"], {
        "outcome": outcome, "payout": payout, "status": status, "settled_at": db._now(),
    })
    if closed is None:
        return {"ok": False, "error": "round_not_open"}
    if payout > 0:
        new_balance = await db.apply_ledger(tg_id, payout, "game_win", ref=rnd["game"], meta={"round": rnd["id"]})
    else:
        u = await db.get_user(tg_id)
        new_balance = int((u or {}).get("balance", 0))
    # NB: the active server_seed is intentionally NOT returned here — it stays
    # secret across the pair's reuse and is revealed only on rotation. Rendering
    # relies on the outcome fields (e.g. mines includes the full `mines` layout).
    return {
        "outcome": outcome,
        "payout": payout,
        "new_balance": new_balance,
        "server_hash": rnd["server_hash"],
    }


@app.post("/bt/api/game/{name}/settle")
async def game_settle(name: str, user: dict = Depends(require_user), body: dict | None = Body(default=None)):
    if name not in SINGLE_SETTLE:
        return _err("invalid_action")
    tg_id = user["tg_id"]
    allowed, retry_after = ratelimit.check(f"game:{tg_id}", limit=60, window_sec=60)
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
        multiplier = result["multiplier"]
    else:  # plinko
        result = plinko.drop(ss, cs, nonce, int(params["rows"]), str(params["risk"]))
        multiplier = result["multiplier"]
    return await _finalise(rnd, tg_id, multiplier, "settled", result)


@app.post("/bt/api/game/{name}/step")
async def game_step(name: str, user: dict = Depends(require_user), body: dict | None = Body(default=None)):
    if name not in MULTI_STEP:
        return _err("invalid_action")
    tg_id = user["tg_id"]
    allowed, retry_after = ratelimit.check(f"game:{tg_id}", limit=60, window_sec=60)
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
        mult = flip.multiplier(new_streak)
        new_state = {"streak": new_streak, "multiplier": mult, "coin": coin, "guess": guess}
        await db.update_round(rnd["id"], {"outcome": new_state})
        return {"outcome_step": {"coin": coin, "guess": guess, "streak": new_streak},
                "multiplier": mult, "can_cashout": True, "busted": False, "done": False}

    if name == "mines":
        m = int(params["mines"])
        revealed = list(state.get("revealed", []))
        if not isinstance(move, (int, str)):
            return _err("invalid_move")
        try:
            cell = int(move)
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
        await db.update_round(rnd["id"], {"outcome": new_state})
        return {"outcome_step": {"cell": cell, "safe": True}, "multiplier": mult,
                "can_cashout": True, "busted": False, "done": False}

    if name == "towers":
        difficulty = str(params["difficulty"])
        floor = int(state.get("floor", 0))
        if not isinstance(move, (int, str)):
            return _err("invalid_move")
        try:
            col = int(move)
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
        await db.update_round(rnd["id"], {"outcome": new_state})
        return {"outcome_step": {"floor": floor, "col": col, "safe": True}, "multiplier": mult,
                "can_cashout": True, "busted": False, "done": False}

    # highlow — the client sends the direction as {"guess": ...} (matching the
    # other games' dict-shaped moves); accept a bare string too for robustness.
    guess = move.get("guess") if isinstance(move, dict) else move
    r = int(state.get("rank", 0))
    step_n = int(state.get("step", 0))
    cur_mult = float(state.get("multiplier", 1.0))
    draw = lambda i: rng_float(ss, cs, nonce, i)
    slot = step_n + 1
    # Skip: swap the current decision card for a fresh one without wagering. The
    # chain multiplier is unchanged and a new slot is consumed so the draw stays
    # provably deterministic. This is EV-neutral — every guess pays EV = 1 - HL_EPS
    # regardless of the card — so unlimited skips can't be exploited.
    if guess == "skip" or (isinstance(move, dict) and move.get("skip")):
        new_rank = highlow.current_card(draw, slot)
        new_state = {"rank": new_rank, "step": step_n + 1, "multiplier": cur_mult}
        await db.update_round(rnd["id"], {"outcome": new_state})
        return {"outcome_step": {"current": new_rank, "prev": r, "guess": "skip",
                                 "skipped": True, "win": True},
                "multiplier": cur_mult, "can_cashout": True, "busted": False, "done": False}
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
    new_state = {"rank": new_rank, "step": step_n + 1, "multiplier": new_mult}
    await db.update_round(rnd["id"], {"outcome": new_state})
    return {"outcome_step": {"drawn": drawn, "current": new_rank, "prev": r, "guess": guess, "win": True},
            "multiplier": new_mult, "can_cashout": True, "busted": False, "done": False}


@app.post("/bt/api/game/{name}/cashout")
async def game_cashout(name: str, user: dict = Depends(require_user), body: dict | None = Body(default=None)):
    if name not in MULTI_STEP:
        return _err("invalid_action")
    tg_id = user["tg_id"]
    allowed, retry_after = ratelimit.check(f"game:{tg_id}", limit=60, window_sec=60)
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
        mult = flip.multiplier(streak)
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
    else:  # highlow
        step_n = int(state.get("step", 0))
        mult = min(float(state.get("multiplier", 1.0)), MULT_CAP)
        outcome = {"step": step_n, "rank": int(state.get("rank", 0)), "multiplier": mult}

    return await _finalise(rnd, tg_id, mult, "cashed_out", outcome)

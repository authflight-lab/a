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

import math
import time
from collections import defaultdict
from datetime import datetime, timedelta, timezone

from fastapi import Body, Depends, FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from . import db
from .auth import require_user
from .config import settings
from .db import InsufficientBalance, SupabaseNotConfigured
from .game import BET_MAX, BET_MIN, GAMES, MULTI_STEP, P_MAX, SINGLE_SETTLE
from .game import dice, flip, highlow, mines, plinko, towers
from .game.seed import generate_server_seed, rng_float, server_hash

app = FastAPI(title="Bartender API", version="1.0")

# CORS: explicit allowlist = BT_APP_ORIGIN only. Never a wildcard (spec §8).
_origins = [settings.bt_app_origin] if settings.bt_app_origin else []
app.add_middleware(
    CORSMiddleware,
    allow_origins=_origins,
    allow_credentials=False,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["X-Telegram-Init-Data", "Content-Type"],
)


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


def _daily_claim(streak: int) -> int:
    """D(s) = floor(20 * (1 + 1.5 * (1 - e^(-s/10))))  (spec §5)."""
    return int(20 * (1 + 1.5 * (1 - math.exp(-streak / 10))))


def _payout(bet: int, multiplier: float) -> int:
    """bet * multiplier, floored, capped at P_MAX (spec §6)."""
    return min(int(bet * multiplier), P_MAX)


# --- simple in-memory rate limiting (spec §14: /claim, /game/*, /redeem) ---

_rl_buckets: dict[str, list[float]] = defaultdict(list)


def _rate_limit(key: str, limit: int, window_sec: float) -> bool:
    """Return True if allowed, False if the limit is exceeded."""
    now = time.time()
    bucket = _rl_buckets[key]
    cutoff = now - window_sec
    while bucket and bucket[0] < cutoff:
        bucket.pop(0)
    if len(bucket) >= limit:
        return False
    bucket.append(now)
    return True


def _err(code: str, status: int = 400) -> JSONResponse:
    return JSONResponse(status_code=status, content={"ok": False, "error": code})


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
    u = await db.get_user(tg_id)
    if u is None:
        u = await db.upsert_user(tg_id, user.get("username"), user.get("display_name")) or {}
    day = _today()
    quest = await db.get_quest(tg_id, day) or {"day": day, "chatted": False, "claimed": False}
    chatted = bool(quest.get("chatted", False))
    claimed = bool(quest.get("claimed", False))
    meta = u.get("meta") or {}
    return {
        "tg_id": tg_id,
        "username": u.get("username"),
        "display_name": u.get("display_name"),
        "balance": int(u.get("balance", 0)),
        "streak_days": int(u.get("streak_days", 0)),
        "last_claim_at": u.get("last_claim_at"),
        "quest": {"day": day, "chatted": chatted, "claimed": claimed},
        "can_redeem": chatted and claimed,
        "age_ack": bool(meta.get("age_ack", False)),
    }


@app.post("/bt/api/claim")
async def claim(user: dict = Depends(require_user)):
    tg_id = user["tg_id"]
    if not _rate_limit(f"claim:{tg_id}", limit=5, window_sec=60):
        return _err("rate_limited", 429)

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
    if await db.get_user(tg_id) is None:
        await db.upsert_user(tg_id, user.get("username"), user.get("display_name"))
    await db.set_age_ack(tg_id)
    return {"ok": True}


# ---------------------------------------------------------------------------
# Shop
# ---------------------------------------------------------------------------

@app.get("/bt/api/rewards")
async def rewards(_user: dict = Depends(require_user)):
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
    if not _rate_limit(f"redeem:{tg_id}", limit=10, window_sec=60):
        return _err("rate_limited", 429)

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

    return {
        "ok": True,
        "redemption_id": result.get("redemption_id"),
        "new_balance": result.get("new_balance"),
    }


# ---------------------------------------------------------------------------
# Leaderboard / history
# ---------------------------------------------------------------------------

@app.get("/bt/api/leaderboard")
async def leaderboard(tab: str = "rich", user: dict = Depends(require_user)):
    tg_id = user["tg_id"]
    if tab not in ("rich", "chatters"):
        return _err("invalid_tab")

    if tab == "rich":
        top = await db.leaderboard_rich(limit=20)
        rows = [
            {"rank": i + 1, "tg_id": int(r["tg_id"]),
             "display_name": r.get("display_name") or str(r["tg_id"]),
             "value": int(r.get("balance", 0))}
            for i, r in enumerate(top)
        ]
        u = await db.get_user(tg_id)
        you = None
        if u is not None:
            bal = int(u.get("balance", 0))
            you = {"rank": await db.rich_rank(tg_id, bal), "value": bal}
        return {"tab": tab, "rows": rows, "you": you}

    # chatters: points earned with kind='chat' this period.
    ledger = await db.chatters_ledger(_period_start())
    totals: dict[int, int] = defaultdict(int)
    for row in ledger:
        totals[int(row["tg_id"])] += int(row["amount"])
    ordered = sorted(totals.items(), key=lambda kv: kv[1], reverse=True)
    top_ids = [uid for uid, _ in ordered[:20]]
    names = await db.display_names(top_ids)
    rows = [
        {"rank": i + 1, "tg_id": uid, "display_name": names.get(uid, str(uid)), "value": val}
        for i, (uid, val) in enumerate(ordered[:20])
    ]
    you = None
    if tg_id in totals:
        rank = 1 + sum(1 for _, v in ordered if v > totals[tg_id])
        you = {"rank": rank, "value": totals[tg_id]}
    return {"tab": tab, "rows": rows, "you": you}


@app.get("/bt/api/history")
async def history(user: dict = Depends(require_user)):
    tg_id = user["tg_id"]
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
        start = highlow.draw_card(rng_float(ss, cs, nonce, 0))
        return {"rank": start, "start_card": start, "step": 0, "multiplier": 1.0}
    return {}


@app.post("/bt/api/game/{name}/bet")
async def game_bet(name: str, user: dict = Depends(require_user), body: dict | None = Body(default=None)):
    if name not in GAMES:
        return _err("unknown_game", 404)
    tg_id = user["tg_id"]
    if not _rate_limit(f"game:{tg_id}", limit=60, window_sec=60):
        return _err("rate_limited", 429)

    body = body or {}
    try:
        bet = int(body.get("bet", 0))
    except (TypeError, ValueError):
        return _err("invalid_bet")
    client_seed = str(body.get("client_seed") or "")
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

    ss = generate_server_seed()
    sh = server_hash(ss)
    nonce = await db.next_nonce(tg_id)
    state = _initial_state(name, np, ss, client_seed, nonce)

    try:
        new_balance = await db.apply_ledger(tg_id, -bet, "game_bet", ref=name, meta={"nonce": nonce})
    except InsufficientBalance:
        return {"ok": False, "error": "insufficient_balance"}

    rnd = await db.create_round(tg_id, name, bet, ss, sh, client_seed, nonce, np, state)

    resp_params = dict(np)
    if name == "highlow":
        resp_params["start_card"] = state["start_card"]
    return {
        "round_id": (rnd or {}).get("id"),
        "server_hash": sh,
        "nonce": nonce,
        "balance": new_balance,
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
    return {
        "outcome": outcome,
        "payout": payout,
        "new_balance": new_balance,
        "server_seed": rnd["server_seed"],
        "server_hash": rnd["server_hash"],
    }


@app.post("/bt/api/game/{name}/settle")
async def game_settle(name: str, user: dict = Depends(require_user), body: dict | None = Body(default=None)):
    if name not in SINGLE_SETTLE:
        return _err("invalid_action")
    tg_id = user["tg_id"]
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
        mult = mines.multiplier(k, m)
        done = k == (mines.TOTAL - m)
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
        mult = towers.multiplier(new_floor, difficulty)
        new_state = {"floor": new_floor, "difficulty": difficulty, "multiplier": mult}
        await db.update_round(rnd["id"], {"outcome": new_state})
        return {"outcome_step": {"floor": floor, "col": col, "safe": True}, "multiplier": mult,
                "can_cashout": True, "busted": False, "done": False}

    # highlow
    r = int(state.get("rank", 0))
    step_n = int(state.get("step", 0))
    cur_mult = float(state.get("multiplier", 1.0))
    if move not in ("higher", "lower") or not highlow.can_pick(move, r):
        return _err("invalid_move")
    drawn = highlow.draw_card(rng_float(ss, cs, nonce, step_n + 1))
    win = highlow.resolve(move, r, drawn)
    if not win:
        outcome = {"rank": r, "drawn": drawn, "guess": move, "step": step_n, "busted": True}
        final = await _finalise(rnd, tg_id, 0.0, "settled", outcome)
        return {"outcome_step": {"drawn": drawn, "prev": r, "guess": move, "win": False},
                "multiplier": 0.0, "can_cashout": False, "busted": True, "done": True, **final}
    new_mult = cur_mult * highlow.step_multiplier(move, r)
    new_state = {"rank": drawn, "step": step_n + 1, "multiplier": new_mult}
    await db.update_round(rnd["id"], {"outcome": new_state})
    return {"outcome_step": {"drawn": drawn, "prev": r, "guess": move, "win": True},
            "multiplier": new_mult, "can_cashout": True, "busted": False, "done": False}


@app.post("/bt/api/game/{name}/cashout")
async def game_cashout(name: str, user: dict = Depends(require_user), body: dict | None = Body(default=None)):
    if name not in MULTI_STEP:
        return _err("invalid_action")
    tg_id = user["tg_id"]
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
        mult = mines.multiplier(len(revealed), m)
        # server_seed is revealed to the client on cashout regardless, so the
        # mine layout is already client-derivable — including it here lets the
        # UI reveal the full board (no new information is disclosed).
        mine_set = sorted(mines.mine_positions(ss, cs, nonce, m))
        outcome = {"revealed": revealed, "mines_count": m, "multiplier": mult, "mines": mine_set}
    elif name == "towers":
        difficulty = str(params["difficulty"])
        floor = int(state.get("floor", 0))
        mult = towers.multiplier(floor, difficulty)
        outcome = {"floor": floor, "difficulty": difficulty, "multiplier": mult}
    else:  # highlow
        step_n = int(state.get("step", 0))
        mult = float(state.get("multiplier", 1.0))
        outcome = {"step": step_n, "rank": int(state.get("rank", 0)), "multiplier": mult}

    return await _finalise(rnd, tg_id, mult, "cashed_out", outcome)

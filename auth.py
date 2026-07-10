"""Telegram WebApp initData validation (spec §8, contract §4).

Every endpoint requires the ``X-Telegram-Init-Data`` header. The ``tg_id`` is
derived ONLY from the validated ``initData.user`` JSON — never from the request
body or query. Invalid/stale initData -> 401 ``{"error": "bad_init_data"}``.
"""

import hashlib
import hmac
import json
import time
from urllib.parse import parse_qsl

from fastapi import Header, HTTPException, Request

from . import ratelimit
from .config import settings


class InitDataError(Exception):
    """Raised when initData fails HMAC validation or is stale."""


def _client_ip(request: Request | None) -> str:
    """Same IP-extraction rule as the pre-auth IP middleware (main.py)."""
    if request is None:
        return "unknown"
    xff = request.headers.get("X-Forwarded-For")
    if xff:
        return xff.split(",")[0].strip()
    return request.client.host if request.client else "unknown"


def _reject_unauthenticated(request: Request | None = None):
    """Raise 429 once THIS caller's failed-auth budget is spent, else raise 401.

    A 401 is rejected before any of the app's own UI pacing ever applies (no
    button tap, no poll interval, nothing) — unlike every other status code,
    an external caller hitting the API directly can trigger 401s as fast as
    the network allows. The budget is per-IP (config bt_rl_auth_fail_*): once
    one caller's own failures exhaust it, THEIR further retries get 429
    instead of a fresh 401 — but this must never cap out other users' ability
    to log in, so it is keyed per source IP, not shared globally.
    """
    allowed, retry_after = ratelimit.check(
        f"401:{_client_ip(request)}", limit=settings.bt_rl_auth_fail_limit, window_sec=settings.bt_rl_auth_fail_window_sec
    )
    if not allowed:
        raise HTTPException(
            status_code=429,
            detail={"error": "too_many_failed_auth"},
            headers={"Retry-After": str(retry_after)},
        )
    raise HTTPException(status_code=401, detail={"error": "bad_init_data"})


def verify_init_data(init_data: str, bot_token: str, max_age: int = 3600) -> dict:
    """Validate a Telegram WebApp initData string. Returns the parsed fields.

    Raises ``InitDataError`` on a bad signature or a stale ``auth_date``.
    """
    p = dict(parse_qsl(init_data, keep_blank_values=True))
    h = p.pop("hash", "")
    check = "\n".join(f"{k}={v}" for k, v in sorted(p.items()))
    secret = hmac.new(b"WebAppData", bot_token.encode(), hashlib.sha256).digest()
    calc = hmac.new(secret, check.encode(), hashlib.sha256).hexdigest()
    if not hmac.compare_digest(calc, h):
        raise InitDataError("bad_init_data")
    if time.time() - int(p.get("auth_date", 0)) > max_age:
        raise InitDataError("stale")
    return p  # tg_id comes from p["user"] json only — never from the request body


def resolve_display_name(user: dict, tg_id: int | None = None) -> str:
    """Canonical display-name rule, shared by every API write path.

    first+last name -> @username -> numeric id, always non-empty. This is the
    single source of truth for a user's name on the API side; it mirrors
    aiogram's ``full_name`` semantics so the bot and api never write a
    different name for the same row (the old flicker source).
    """
    if tg_id is None:
        tg_id = user.get("id")
    return (
        " ".join(x for x in (user.get("first_name"), user.get("last_name")) if x)
        or user.get("username")
        or str(tg_id)
    )


async def require_user(
    request: Request,
    x_telegram_init_data: str | None = Header(default=None, alias="X-Telegram-Init-Data"),
) -> dict:
    """FastAPI dependency: validate initData, return ``{"tg_id", "user"}``.

    ``tg_id`` is taken exclusively from the validated ``initData.user`` payload.
    """
    if not x_telegram_init_data:
        _reject_unauthenticated(request)
    if not settings.bot_token:
        # Cannot validate without the bot token — treated as not configured.
        raise HTTPException(status_code=503, detail={"error": "not_configured"})
    try:
        p = verify_init_data(x_telegram_init_data, settings.bot_token)
    except InitDataError:
        _reject_unauthenticated(request)
    try:
        user = json.loads(p["user"])
        tg_id = int(user["id"])
    except (KeyError, ValueError, TypeError, json.JSONDecodeError):
        _reject_unauthenticated(request)
    return {
        "tg_id": tg_id,
        "user": user,
        "username": user.get("username"),
        "display_name": resolve_display_name(user, tg_id),
    }

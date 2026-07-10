





"""Per-credential 401 rate limit (task: retries of a failing login are capped).

A 401 is rejected before any UI pacing applies, so unlike every other status
code it has no natural ceiling. We sit behind Cloudflare Workers, where many
users can share the same edge IP, so the budget is keyed on the literal
failing credential (a hash of the raw initData string) rather than IP or a
single global bucket — either of those would let one bad actor's failures
lock other users out of logging in. This locks:
- non-401 traffic is unaffected;
- retrying the SAME bad initData repeatedly trips 429 `too_many_failed_auth`
  with Retry-After once its budget is spent;
- a DIFFERENT failing attempt (different initData) is judged on its own
  fresh budget, and a valid login always succeeds immediately;
- legitimate authenticated traffic (bypassing require_user) is untouched.
"""

import hashlib
import hmac
import json
import time
from urllib.parse import urlencode

import pytest
from fastapi.testclient import TestClient

from api import ratelimit
from api.config import settings
from api.main import app


def _valid_init_data(bot_token: str, tg_id: int = 999) -> str:
    """Build a real, correctly-signed initData string (mirrors auth.verify_init_data)."""
    fields = {"user": json.dumps({"id": tg_id, "first_name": "T"}), "auth_date": str(int(time.time()))}
    check = "\n".join(f"{k}={v}" for k, v in sorted(fields.items()))
    secret = hmac.new(b"WebAppData", bot_token.encode(), hashlib.sha256).digest()
    fields["hash"] = hmac.new(secret, check.encode(), hashlib.sha256).hexdigest()
    return urlencode(fields)


@pytest.fixture()
def client():
    c = TestClient(app)
    yield c


@pytest.fixture(autouse=True)
def _clean_buckets():
    ratelimit._store.clear()
    yield
    ratelimit._store.clear()


def test_retrying_the_same_bad_creds_is_capped(monkeypatch):
    monkeypatch.setattr(settings, "bt_rl_auth_fail_limit", 3)
    monkeypatch.setattr(settings, "bt_rl_ip_limit", 1000)  # isolate from the general IP guard
    monkeypatch.setattr(settings, "bot_token", "test-token")
    c = TestClient(app)
    bad = {"X-Telegram-Init-Data": "user=%7B%7D&auth_date=1&hash=deadbeef"}

    codes = [c.get("/bt/api/history", headers=bad).status_code for _ in range(4)]
    # First 3 retries of the SAME bad initData are plain 401s; the 4th trips
    # that credential's own budget and gets 429 instead.
    assert codes[:3] == [401, 401, 401]
    assert codes[3] == 429


def test_different_failing_creds_from_shared_ip_are_not_cross_blocked(monkeypatch):
    """We sit behind Cloudflare Workers: many users can share one edge IP, so
    one user's exhausted retry budget must never 429 a different user's
    (also-failing, but distinct) login attempt."""
    monkeypatch.setattr(settings, "bt_rl_auth_fail_limit", 1)
    monkeypatch.setattr(settings, "bt_rl_ip_limit", 1000)
    monkeypatch.setattr(settings, "bot_token", "test-token")
    c = TestClient(app)
    same_ip = {"X-Forwarded-For": "203.0.113.9"}

    r1 = c.get("/bt/api/history", headers={**same_ip, "X-Telegram-Init-Data": "attempt-a"})
    r2 = c.get("/bt/api/history", headers={**same_ip, "X-Telegram-Init-Data": "attempt-a"})
    r3 = c.get("/bt/api/history", headers={**same_ip, "X-Telegram-Init-Data": "attempt-b"})
    assert r1.status_code == 401
    assert r2.status_code == 429  # retry of the SAME bad creds -> capped
    assert r3.status_code == 401  # different creds, same IP -> judged fresh


def test_valid_login_bypasses_an_exhausted_failed_auth_budget(monkeypatch):
    """Exhaust the failed-auth budget with bad creds, then prove a genuinely
    valid login still succeeds immediately — it never touches the limiter."""
    monkeypatch.setattr(settings, "bt_rl_auth_fail_limit", 1)
    monkeypatch.setattr(settings, "bt_rl_ip_limit", 1000)
    monkeypatch.setattr(settings, "bot_token", "test-token")
    c = TestClient(app)
    bad = {"X-Telegram-Init-Data": "same-bad-creds"}

    r1 = c.get("/bt/api/history", headers=bad)
    r2 = c.get("/bt/api/history", headers=bad)
    assert r1.status_code == 401
    assert r2.status_code == 429  # that credential's own budget is spent

    good = {"X-Telegram-Init-Data": _valid_init_data("test-token")}
    r3 = c.get("/bt/api/history", headers=good)
    assert r3.status_code != 429
    assert r3.status_code != 401


def test_valid_login_bypasses_an_exhausted_missing_header_budget(monkeypatch):
    """Same guarantee for the 'missing header' sentinel bucket."""
    monkeypatch.setattr(settings, "bt_rl_auth_fail_limit", 1)
    monkeypatch.setattr(settings, "bt_rl_ip_limit", 1000)
    monkeypatch.setattr(settings, "bot_token", "test-token")
    c = TestClient(app)

    r1 = c.get("/bt/api/history")
    r2 = c.get("/bt/api/history")
    assert r1.status_code == 401
    assert r2.status_code == 429

    good = {"X-Telegram-Init-Data": _valid_init_data("test-token")}
    r3 = c.get("/bt/api/history", headers=good)
    assert r3.status_code != 429
    assert r3.status_code != 401


def test_429_body_and_retry_after(monkeypatch):
    monkeypatch.setattr(settings, "bt_rl_auth_fail_limit", 1)
    monkeypatch.setattr(settings, "bt_rl_ip_limit", 1000)
    monkeypatch.setattr(settings, "bot_token", "test-token")
    c = TestClient(app)
    bad = {"X-Telegram-Init-Data": "same-bad-creds"}

    r1 = c.get("/bt/api/history", headers=bad)
    r2 = c.get("/bt/api/history", headers=bad)
    assert r1.status_code == 401
    assert r2.status_code == 429
    assert r2.json()["error"] == "too_many_failed_auth"
    assert "Retry-After" in r2.headers


def test_default_auth_fail_limit_is_60_per_60s():
    assert settings.bt_rl_auth_fail_limit == 60
    assert settings.bt_rl_auth_fail_window_sec == 60


def test_dict_detail_is_flattened_to_top_level_error():
    """Any HTTPException(detail={"error": ...}) reaches the client flat, not
    nested under "detail" — app/js/api.js only ever reads a top-level `error`."""
    c = TestClient(app)
    r = c.get("/bt/api/history")
    body = r.json()
    assert body["ok"] is False
    assert body["error"] == "bad_init_data"
    assert "detail" not in body


def test_validation_422_shape_is_unchanged():
    """FastAPI's own request-validation errors are handled separately from our
    HTTPException flattening handler and keep their normal `detail` list shape."""
    c = TestClient(app)
    r = c.post("/bt/api/game/dice/bet", data="not-json", headers={"Content-Type": "application/json"})
    assert r.status_code == 422
    assert "detail" in r.json()

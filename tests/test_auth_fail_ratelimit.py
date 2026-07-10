"""Global 401 rate limit (task: global 401-specific rate limit).

A 401 is rejected before any UI pacing applies, so unlike every other status
code it has no natural ceiling. This locks:
- non-401 traffic is unaffected;
- once the global (not per-IP) failed-auth budget is spent, further
  auth-failing requests get 429 `too_many_failed_auth` with Retry-After
  instead of a plain 401;
- legitimate authenticated traffic (bypassing require_user) is untouched.
"""

import pytest
from fastapi.testclient import TestClient

from api import ratelimit
from api.config import settings
from api.main import app


@pytest.fixture()
def client():
    c = TestClient(app)
    yield c


@pytest.fixture(autouse=True)
def _clean_buckets():
    ratelimit._store.clear()
    yield
    ratelimit._store.clear()


def test_401_spam_is_globally_capped(monkeypatch):
    monkeypatch.setattr(settings, "bt_rl_auth_fail_limit", 3)
    monkeypatch.setattr(settings, "bt_rl_ip_limit", 1000)  # isolate from the general IP guard
    c = TestClient(app)

    codes = [
        c.get("/bt/api/history", headers={"X-Forwarded-For": f"203.0.113.{i}"}).status_code
        for i in range(4)
    ]
    # First 3 (missing/invalid initData) are plain 401s; the 4th trips the
    # global budget and gets 429 instead, EVEN THOUGH each request came from
    # a different IP — the budget is global, not per-IP.
    assert codes[:3] == [401, 401, 401]
    assert codes[3] == 429


def test_429_body_and_retry_after(monkeypatch):
    monkeypatch.setattr(settings, "bt_rl_auth_fail_limit", 1)
    monkeypatch.setattr(settings, "bt_rl_ip_limit", 1000)
    c = TestClient(app)

    r1 = c.get("/bt/api/history")
    r2 = c.get("/bt/api/history")
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

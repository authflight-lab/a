"""Rate-limit configuration tests (task: relax API rate limits).

Locks three behaviours:
- the config defaults are the relaxed values (game 60/15s ≈ 240/min sustained,
  ip 600/min, reads 40/min, /me 120/min) and are env-overridable with int
  coercion (malformed overrides fall back to the safe default);
- endpoint call sites read the limits from ``settings`` — tightening or
  loosening is a config change, not a code edit;
- the short game window keeps Retry-After small (seconds, not up to a
  minute) so a tap burst recovers quickly.
"""

import pytest
from fastapi.testclient import TestClient

from api import ratelimit
from api.auth import require_user
from api.config import Settings, settings
from api.main import app


@pytest.fixture()
def client():
    app.dependency_overrides[require_user] = lambda: {
        "tg_id": 424242, "user": {"id": 424242}, "username": "u", "display_name": "U",
    }
    c = TestClient(app)
    yield c
    app.dependency_overrides.clear()


@pytest.fixture(autouse=True)
def _clean_buckets():
    """Isolate the in-memory limiter between tests."""
    ratelimit._store.clear()
    yield
    ratelimit._store.clear()


# ── Config defaults & env overrides ─────────────────────────────────────────

def test_default_limits_are_relaxed():
    s = Settings()
    assert s.bt_rl_ip_limit == 600 and s.bt_rl_ip_window_sec == 60
    assert s.bt_rl_game_limit == 60 and s.bt_rl_game_window_sec == 15
    assert s.bt_rl_me_limit == 120 and s.bt_rl_me_window_sec == 60
    assert s.bt_rl_read_limit == 40 and s.bt_rl_read_window_sec == 60


def test_env_override_coerces_int(monkeypatch):
    monkeypatch.setenv("BT_RL_GAME_LIMIT", "90")
    s = Settings()
    assert s.bt_rl_game_limit == 90
    assert isinstance(s.bt_rl_game_limit, int)


def test_malformed_env_override_keeps_default(monkeypatch):
    monkeypatch.setenv("BT_RL_GAME_LIMIT", "not-a-number")
    s = Settings()
    assert s.bt_rl_game_limit == 60


# ── Short game window ⇒ short Retry-After ───────────────────────────────────

def test_game_window_retry_after_is_seconds_not_a_minute():
    limit, window = settings.bt_rl_game_limit, settings.bt_rl_game_window_sec
    for _ in range(limit):
        allowed, _ = ratelimit.check("game:t", limit=limit, window_sec=window)
        assert allowed
    allowed, retry_after = ratelimit.check("game:t", limit=limit, window_sec=window)
    assert not allowed
    assert 1 <= retry_after <= window  # bounded by the 15 s window, not 60 s


def test_game_sustained_rate_is_about_240_per_minute():
    per_min = settings.bt_rl_game_limit * (60 / settings.bt_rl_game_window_sec)
    assert per_min >= 240


# ── Endpoints honour the configured limits ──────────────────────────────────

def test_ip_middleware_uses_configured_limit(client, monkeypatch):
    monkeypatch.setattr(settings, "bt_rl_ip_limit", 3)
    # Unauthenticated path is fine — the IP guard runs before auth; 4th hits 429.
    codes = [
        client.get("/bt/api/health", headers={"X-Forwarded-For": "203.0.113.9"}).status_code
        for _ in range(4)
    ]
    assert codes[:3].count(429) == 0
    assert codes[3] == 429


def test_read_endpoint_429_after_configured_limit(client, monkeypatch):
    monkeypatch.setattr(settings, "bt_rl_read_limit", 2)

    async def fake_history(tg_id, limit=50):
        return []
    from api import db
    monkeypatch.setattr(db, "ledger_history", fake_history)

    r1 = client.get("/bt/api/history")
    r2 = client.get("/bt/api/history")
    r3 = client.get("/bt/api/history")
    assert r1.status_code == 200 and r2.status_code == 200
    assert r3.status_code == 429
    assert "Retry-After" in r3.headers


def test_nonpositive_env_override_keeps_default(monkeypatch):
    monkeypatch.setenv("BT_RL_GAME_LIMIT", "0")
    monkeypatch.setenv("BT_RL_IP_LIMIT", "-5")
    s = Settings()
    assert s.bt_rl_game_limit == 60
    assert s.bt_rl_ip_limit == 600

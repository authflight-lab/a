"""Tiny in-process TTL cache for read-heavy, rarely-changing data.

Two uses (see :mod:`api.db`): the reward catalogue (changes rarely) and the
``/me`` user profile (short TTL, invalidated on every *same-process* balance
mutation so a user never sees a stale balance after their own action).
Cross-process writes from the bot are bounded by the short TTL. Not shared
across workers — each process keeps its own map, which is fine for these
best-effort reads.
"""

from __future__ import annotations

import time
from typing import Any


_store: dict[str, tuple[float, Any]] = {}


def get(key: str) -> Any | None:
    """Return the cached value for ``key`` if present and unexpired, else None.

    ``None`` is never stored, so a hit is always a real value.
    """
    hit = _store.get(key)
    if hit is None:
        return None
    expires_at, value = hit
    if time.monotonic() >= expires_at:
        _store.pop(key, None)
        return None
    return value


def put(key: str, value: Any, ttl: float) -> None:
    """Cache ``value`` under ``key`` for ``ttl`` seconds. No-op for None."""
    if value is None:
        return
    _store[key] = (time.monotonic() + ttl, value)


def invalidate(key: str) -> None:
    _store.pop(key, None)


def clear() -> None:
    _store.clear()

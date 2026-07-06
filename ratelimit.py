"""Sliding-window in-memory rate limiter (spec §14).

Usage::

    allowed, retry_after = check("game:12345", limit=60, window_sec=60)
    if not allowed:
        return JSONResponse({"error": "rate_limited"}, 429,
                            headers={"Retry-After": str(retry_after)})

``check`` is intentionally synchronous — all access is in a single-threaded
asyncio event loop, so no lock is needed.  The key store is LRU-capped at
``_MAX_KEYS`` entries so it cannot grow unboundedly even with a very large
number of unique users / IPs.
"""

import math
import time
from collections import OrderedDict

_MAX_KEYS = 10_000

_store: OrderedDict[str, list[float]] = OrderedDict()


def _evict_if_full() -> None:
    """Drop the least-recently-used key when the store is at capacity."""
    while len(_store) >= _MAX_KEYS:
        _store.popitem(last=False)


def check(key: str, limit: int, window_sec: float) -> tuple[bool, int]:
    """Sliding-window check.

    Returns ``(allowed, retry_after)`` where *retry_after* is the number of
    seconds the caller should wait before retrying (0 when *allowed* is True).
    """
    now = time.monotonic()
    cutoff = now - window_sec

    if key in _store:
        _store.move_to_end(key)
        bucket = _store[key]
    else:
        _evict_if_full()
        bucket = []
        _store[key] = bucket

    while bucket and bucket[0] < cutoff:
        bucket.pop(0)

    if len(bucket) >= limit:
        retry_after = math.ceil(window_sec - (now - bucket[0]))
        return False, max(1, retry_after)

    bucket.append(now)
    return True, 0

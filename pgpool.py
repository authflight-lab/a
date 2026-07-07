"""Direct asyncpg connection pool to Supabase Postgres (Supavisor session pooler).

This is the low-latency path that bypasses the PostgREST HTTP hop for hot reads.
It runs ALONGSIDE the httpx REST client in ``db.py`` — REST stays the default and
authority; the pool is opt-in per call site (currently only the ``_pgprobe`` spike
endpoint). We connect via the SESSION pooler (port 5432, IPv4) with statement
caching disabled, which is the Supavisor-safe setting for a long-lived pool.

Importing this module performs NO network I/O: the pool is created lazily on first
use and closed on app shutdown.
"""

from __future__ import annotations

import asyncpg

from .config import settings

_pool: asyncpg.Pool | None = None


class PgPoolNotConfigured(Exception):
    """Raised when no ``bt_supabase_db_url`` (DSN) is configured."""


async def get_pool() -> asyncpg.Pool:
    """Return the process-wide asyncpg pool, creating it on first use."""
    global _pool
    if _pool is None:
        dsn = settings.bt_supabase_db_url
        if not dsn:
            raise PgPoolNotConfigured("bt_supabase_db_url is empty")
        _pool = await asyncpg.create_pool(
            dsn=dsn,
            min_size=1,
            max_size=10,
            statement_cache_size=0,  # Supavisor-safe (avoids stale prepared stmts)
            command_timeout=10,
        )
    return _pool


async def close_pool() -> None:
    """Close the pool on shutdown (idempotent)."""
    global _pool
    if _pool is not None:
        await _pool.close()
        _pool = None

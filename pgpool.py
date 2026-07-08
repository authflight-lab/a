"""Direct asyncpg connection pool to Supabase Postgres (Supavisor session pooler).

This is the low-latency path that bypasses the PostgREST HTTP hop for hot reads.
It runs ALONGSIDE the httpx REST client in ``db.py`` — REST stays the default and
authority; the pool is opt-in per call site (currently only the ``_pgprobe`` spike
endpoint). We connect via the SESSION pooler (port 5432, IPv4); prepared-statement
caching is enabled there (each pooled connection is pinned to one backend) and
disabled automatically if the DSN points at the transaction pooler (:6543).

Importing this module performs NO network I/O: the pool is created lazily on first
use and closed on app shutdown.
"""

from __future__ import annotations

import asyncio
import json

import asyncpg

from .config import settings

_pool: asyncpg.Pool | None = None


class PgPoolNotConfigured(Exception):
    """Raised when no ``bt_supabase_db_url`` (DSN) is configured."""


async def _init_conn(con: asyncpg.Connection) -> None:
    """Per-connection setup: decode ``json``/``jsonb`` to/from Python objects so
    callers pass and receive ``dict``/``list`` (matching the PostgREST client's
    shape) instead of raw JSON text."""
    await con.set_type_codec(
        "jsonb", encoder=json.dumps, decoder=json.loads, schema="pg_catalog")
    await con.set_type_codec(
        "json", encoder=json.dumps, decoder=json.loads, schema="pg_catalog")


async def get_pool() -> asyncpg.Pool:
    """Return the process-wide asyncpg pool, creating it on first use."""
    global _pool
    if _pool is None:
        dsn = settings.bt_supabase_db_url
        if not dsn:
            raise PgPoolNotConfigured("bt_supabase_db_url is empty")
        # Prepared-statement caching: safe on the SESSION pooler (:5432 — each
        # pool connection is pinned to one backend for its lifetime, so cached
        # statements stay valid) and it saves a parse round-trip per query. The
        # TRANSACTION pooler (:6543) multiplexes backends between transactions,
        # which invalidates cached statements ("prepared statement does not
        # exist"), so only there is the cache disabled. If those errors ever
        # appear on :5432, revert to statement_cache_size=0 unconditionally.
        stmt_cache = 0 if ":6543" in dsn else 100
        _pool = await asyncpg.create_pool(
            dsn=dsn,
            min_size=5,   # warm connections for bet bursts (no connect latency)
            max_size=10,
            statement_cache_size=stmt_cache,
            command_timeout=10,
            init=_init_conn,
        )
    return _pool


def should_fallback(exc: BaseException) -> bool:
    """Return True when ``exc`` is a pool/transport failure — i.e. the query
    almost certainly never reached the server (or the pool is unconfigured), so a
    caller may safely retry the operation over the REST client without risking a
    double-execute. Returns False for server-reported data errors (``RaiseError``
    and other ``PostgresError`` subclasses), which MUST be mapped/propagated, not
    silently re-run."""
    if isinstance(exc, PgPoolNotConfigured):
        return True
    if isinstance(exc, (
        asyncpg.InterfaceError,               # pool closed / client-side misuse
        asyncpg.PostgresConnectionError,      # 08xxx connection-class errors
        asyncpg.exceptions.CannotConnectNowError,     # 57P03 (server starting up)
        asyncpg.exceptions.TooManyConnectionsError,   # 53300 (pooler saturated)
    )):
        return True
    if isinstance(exc, (OSError, ConnectionError, asyncio.TimeoutError, TimeoutError)):
        return True
    return False


async def close_pool() -> None:
    """Close the pool on shutdown (idempotent)."""
    global _pool
    if _pool is not None:
        await _pool.close()
        _pool = None

"""Provably-fair seed-pair lifecycle (Rainbet-style commit-reveal).

One active pair per user (``client_seed`` + secret ``server_seed``) is reused
across many bets by incrementing a per-pair ``nonce``; it rotates only on demand.
A ``next`` server seed is committed ahead of time — its hash is shown BEFORE
rotation — so rotating reveals the retired ``server_seed`` and promotes the
pre-committed one (whose hash the player already saw).

The active ``server_seed`` is NEVER exposed while in use (only its hash), so a
reused seed can't be predicted. It becomes verifiable only after rotation.

These transitions are pure and DB-agnostic: they operate on plain dicts so any
store can persist them and they can be unit-tested without I/O.
"""

import os

from .seed import generate_server_seed, server_hash


def random_client_seed() -> str:
    """A default client seed (16 bytes hex) — same shape the old client minted."""
    return os.urandom(16).hex()


def new_pair(client_seed: str | None = None) -> dict:
    """A fresh active pair: random server seed + a pre-committed next server seed."""
    cs = (client_seed or "").strip() or random_client_seed()
    ss = generate_server_seed()
    nx = generate_server_seed()
    return {
        "client_seed": cs,
        "server_seed": ss,
        "server_hash": server_hash(ss),
        "nonce": 0,
        "next_server_seed": nx,
        "next_server_hash": server_hash(nx),
    }


def public_view(pair: dict) -> dict:
    """Client-safe view. NEVER includes the active ``server_seed`` — only its
    hash and the pre-committed next hash."""
    return {
        "client_seed": pair["client_seed"],
        "nonce": int(pair["nonce"]),
        "server_hash": pair["server_hash"],
        "next_server_hash": pair["next_server_hash"],
    }


def rotate(pair: dict, client_seed: str | None = None) -> tuple[dict, str]:
    """Retire+reveal the active server seed, promote the pre-committed next one,
    commit a fresh next seed, apply the new client seed (or keep the current one
    when none is given), and reset the nonce to 0.

    Returns ``(new_pair, revealed_server_seed)``.
    """
    revealed = pair["server_seed"]
    promoted = pair["next_server_seed"]
    promoted_hash = pair["next_server_hash"]
    nx = generate_server_seed()
    cs = (client_seed or "").strip() or pair["client_seed"]
    new = {
        "client_seed": cs,
        "server_seed": promoted,
        "server_hash": promoted_hash,
        "nonce": 0,
        "next_server_seed": nx,
        "next_server_hash": server_hash(nx),
    }
    return new, revealed

"""Provably-fair seeding helpers (spec §6).

Commit-reveal (Rainbet-style seed-pair reuse — see ``seedpair.py``):
1. One active pair per user (``server_seed`` + ``client_seed``) is committed via
   ``server_hash`` and REUSED across bets; each bet advances ``nonce``.
2. The active ``server_seed`` is revealed only when the pair is ROTATED (not per
   bet). The client then verifies ``sha256(server_seed) == server_hash`` and can
   recompute every past bet from ``(server_seed, client_seed, nonce, cursor)``.

RNG derivation:
    h = HMAC_SHA256(server_seed, f"{client_seed}:{nonce}:{cursor}")
    u = sum(h[i] / 256**(i+1) for i in range(4))  in [0, 1)

``cursor`` increments per draw within a round; ``nonce`` increments per bet on the
active pair and resets to 0 on rotation.
"""

import hashlib
import hmac
import os


def generate_server_seed() -> str:
    return os.urandom(32).hex()


def server_hash(seed: str) -> str:
    return hashlib.sha256(seed.encode()).hexdigest()


def rng_float(server_seed: str, client_seed: str, nonce: int, cursor: int) -> float:
    msg = f"{client_seed}:{nonce}:{cursor}".encode()
    h = hmac.new(server_seed.encode(), msg, hashlib.sha256).digest()
    return sum(h[i] / 256 ** (i + 1) for i in range(4))


def rng_int(server_seed: str, client_seed: str, nonce: int, cursor: int, n: int) -> int:
    return int(rng_float(server_seed, client_seed, nonce, cursor) * n)

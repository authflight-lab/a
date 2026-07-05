"""Provably-fair seeding helpers (spec §6).

Commit-reveal:
1. ``bet``     -> server generates ``server_seed``, returns ``server_hash``.
2. ``settle``  -> outcome derived from the seeded RNG, ``server_seed`` revealed.
   The client can then verify ``sha256(server_seed) == server_hash``.

RNG derivation:
    h = HMAC_SHA256(server_seed, f"{client_seed}:{nonce}:{cursor}")
    u = sum(h[i] / 256**(i+1) for i in range(4))  in [0, 1)

``cursor`` increments per draw within a round; ``nonce`` per round per user.
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

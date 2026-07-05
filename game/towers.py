"""Towers — multi-step (spec §7.4).

    M_tower(L) = (1 - EPS) * (C / (C - t))^L

| Difficulty | C | t | per-floor |
|-----------|---|---|-----------|
| easy      | 4 | 1 | 1.3175x   |
| medium    | 3 | 1 | 1.485x    |
| hard      | 2 | 1 | 1.98x     |

One trap position per floor from the seeded RNG. Client calls ``/step`` per floor
pick, ``/cashout`` to lock. Trap hit -> settle with payout 0.
"""

from . import EPS
from .seed import rng_int

DIFFICULTIES = {
    "easy": {"C": 4, "t": 1},
    "medium": {"C": 3, "t": 1},
    "hard": {"C": 2, "t": 1},
}


def valid_difficulty(difficulty: str) -> bool:
    return difficulty in DIFFICULTIES


def columns(difficulty: str) -> int:
    return DIFFICULTIES[difficulty]["C"]


def multiplier(level: int, difficulty: str) -> float:
    d = DIFFICULTIES[difficulty]
    C, t = d["C"], d["t"]
    return (1 - EPS) * (C / (C - t)) ** level


def trap_positions(server_seed: str, client_seed: str, nonce: int, floor: int, difficulty: str) -> list[int]:
    """The ``t`` trap columns on a given floor (t == 1 for all difficulties)."""
    d = DIFFICULTIES[difficulty]
    C, t = d["C"], d["t"]
    cols = list(range(C))
    cursor = floor
    traps = []
    for _ in range(t):
        j = rng_int(server_seed, client_seed, nonce, cursor, len(cols))
        traps.append(cols.pop(j))
        cursor += 1000  # keep per-floor trap draws well separated
    return sorted(traps)


def rtp_distribution(level: int, difficulty: str) -> list[tuple[float, float]]:
    """[(P(survive L floors), M(L)), (P(hit a trap), 0)]."""
    d = DIFFICULTIES[difficulty]
    C, t = d["C"], d["t"]
    p_floor = (C - t) / C
    p = p_floor ** level
    return [(p, multiplier(level, difficulty)), (1.0 - p, 0.0)]

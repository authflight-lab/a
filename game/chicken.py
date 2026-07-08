"""Chicken Cross — multi-step road crossing (towers-style, horizontal).

    M_chicken(L) = MULT_SCALE * (1 - EPS) * (C / (C - T))^L

Each lane is a road with ``C`` crossing zones, ``T`` of which are occupied by
cars (seeded RNG, one distinct zone per draw). The player picks the zone to
cross through; hitting a car zone busts the run, a safe cross compounds the
multiplier by ``C / (C - T)`` (scaled by the flat 2% edge). All client-side
vehicle animation is cosmetic — the outcome is only this zone comparison.

| Difficulty | C | T | growth/lane |
|-----------|---|---|-------------|
| easy      | 3 | 1 | ~1.47x      |
| medium    | 2 | 1 | ~1.96x      |
| hard      | 3 | 2 | ~2.94x      |
| daredevil | 4 | 3 | ~3.92x      |

``CHICKEN_MAX_MULT`` (20) caps the run — a crossing that reaches it auto-cashes
out — and ``LANES`` (8) bounds the road depth (easy is the only difficulty that
gets that far before capping).
"""

from . import EPS
from .seed import rng_int

# Maximum road depth. Reaching the far side (or the cap, whichever first)
# auto-cashes out.
LANES = 8

# Flat scale on the ladder (1.0 = the plain flat-EPS identity; kept as an
# explicit knob to match towers/mines convention). The RTP test derives its
# target from this, so tuning it here keeps everything in lockstep.
MULT_SCALE = 1.0

# Hard ceiling on the chicken multiplier (economy guard, below the global
# MULT_CAP on purpose). A safe cross that reaches it auto-cashes out.
CHICKEN_MAX_MULT = 20.0

DIFFICULTIES = {
    "easy": {"C": 3, "T": 1},
    "medium": {"C": 2, "T": 1},
    "hard": {"C": 3, "T": 2},
    "daredevil": {"C": 4, "T": 3},
}


def valid_difficulty(difficulty: str) -> bool:
    return difficulty in DIFFICULTIES


def zones(difficulty: str) -> int:
    return DIFFICULTIES[difficulty]["C"]


def multiplier(lanes_crossed: int, difficulty: str) -> float:
    d = DIFFICULTIES[difficulty]
    C, T = d["C"], d["T"]
    return MULT_SCALE * (1 - EPS) * (C / (C - T)) ** lanes_crossed


def car_zones(server_seed: str, client_seed: str, nonce: int, lane: int, difficulty: str) -> list[int]:
    """The ``T`` distinct car-occupied zones on a given lane (seeded draw per
    car; cursor = lane index, subsequent draws separated by a 1000 stride so
    per-lane draws never collide — same scheme as towers)."""
    d = DIFFICULTIES[difficulty]
    C, T = d["C"], d["T"]
    free = list(range(C))
    cursor = lane
    cars = []
    for _ in range(T):
        j = rng_int(server_seed, client_seed, nonce, cursor, len(free))
        cars.append(free.pop(j))
        cursor += 1000
    return sorted(cars)


def rtp_distribution(lanes_crossed: int, difficulty: str) -> list[tuple[float, float]]:
    """[(P(survive L lanes), M(L)), (P(hit a car), 0)]."""
    d = DIFFICULTIES[difficulty]
    C, T = d["C"], d["T"]
    p_lane = (C - T) / C
    p = p_lane ** lanes_crossed
    return [(p, multiplier(lanes_crossed, difficulty)), (1.0 - p, 0.0)]

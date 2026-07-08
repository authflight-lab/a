"""Chicken Cross — multi-step road crossing (Rainbet-style easy ladder).

    M_chicken(n) = (1 - CHICKEN_EDGE) * TOTAL / (TOTAL - n)

The road is a deck of ``TOTAL`` (25) zones with ONE car, drawn without
replacement: lane ``k`` (0-based) has ``TOTAL - k`` zones left and the seeded
draw picks the car among them, so per-lane survival is (24-k)/(25-k) and the
chance of surviving ``n`` lanes is exactly (25-n)/25. The fair ladder is
25/(25-n); scaled by the flat 4% edge that yields the published progression
1.00, 1.04, 1.09, 1.14, 1.20, 1.26, 1.33, 1.41, 1.50, ... 24.00 — a flat 96%
RTP at every decision point. The zone the client sends is cosmetic (every
zone has identical odds); the seeded car draw settles the outcome.

Only "easy" is selectable. The legacy geometric difficulties are kept ONLY so
rounds opened before the easy-only remodel can still settle/cash out.
"""

from . import EPS
from .seed import rng_int

# Zone deck for the easy road (one car, drawn without replacement per lane).
TOTAL = 25

# Road depth: crossing the final lane pays TOTAL/(TOTAL-24) * 0.96 = 24.00x.
LANES = 24

# Chicken uses its own flat edge (4%) so the ladder lands on the published
# values exactly (lane 1 pays 1.00x). The RTP test derives its target from
# this, so tuning it here keeps everything in lockstep.
CHICKEN_EDGE = 0.04

# Hard ceiling on the chicken multiplier — the exact value of the final lane,
# so the cap and the far side of the road coincide.
CHICKEN_MAX_MULT = 24.0

# Legacy geometric difficulties (C zones, T cars) — settle-only, NOT valid
# for new bets.
LEGACY = {
    "medium": (2, 1),
    "hard": (3, 2),
    "daredevil": (4, 3),
}


def valid_difficulty(difficulty: str) -> bool:
    return difficulty == "easy"


def zones(difficulty: str, lane: int = 0) -> int:
    """Zone count on a given lane (easy shrinks the deck one zone per lane)."""
    if difficulty == "easy":
        return max(TOTAL - lane, 1)
    return LEGACY[difficulty][0]


def multiplier(lanes_crossed: int, difficulty: str) -> float:
    if difficulty == "easy":
        return (1 - CHICKEN_EDGE) * TOTAL / (TOTAL - min(lanes_crossed, LANES))
    C, T = LEGACY[difficulty]
    return (1 - EPS) * (C / (C - T)) ** lanes_crossed


def car_zones(server_seed: str, client_seed: str, nonce: int, lane: int, difficulty: str) -> list[int]:
    """The car-occupied zone(s) on a given lane (seeded draw per car; cursor =
    lane index, subsequent draws separated by a 1000 stride so per-lane draws
    never collide — same scheme as towers)."""
    if difficulty == "easy":
        return [rng_int(server_seed, client_seed, nonce, lane, zones("easy", lane))]
    C, T = LEGACY[difficulty]
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
    if difficulty == "easy":
        p = (TOTAL - lanes_crossed) / TOTAL
    else:
        C, T = LEGACY[difficulty]
        p = ((C - T) / C) ** lanes_crossed
    return [(p, multiplier(lanes_crossed, difficulty)), (1.0 - p, 0.0)]

"""HighLow — multi-step (spec §7.5).

    p_hi(r) = (13 - r) / 13,   p_lo(r) = (r - 1) / 13
    M_chain = (1 - EPS)^n * prod_{j=1}^{n} 1 / p_{d_j}(r_j)

Ranks are 1..13. A tie (next card equals current rank) counts as a loss, so the
win probability for a direction is exactly ``p_dir(r)`` and each step's RTP is
exactly ``1 - EPS``. A direction whose probability is 0 must be disabled
server-side (``higher`` at r=13, ``lower`` at r=1).
"""

from . import EPS

RANKS = 13


def p_higher(r: int) -> float:
    return (RANKS - r) / RANKS


def p_lower(r: int) -> float:
    return (r - 1) / RANKS


def prob(direction: str, r: int) -> float:
    return p_higher(r) if direction == "higher" else p_lower(r)


def can_pick(direction: str, r: int) -> bool:
    return prob(direction, r) > 0.0


def step_multiplier(direction: str, r: int) -> float:
    """Per-step multiplier factor (1 - EPS) / p_dir(r)."""
    p = prob(direction, r)
    if p <= 0.0:
        return 0.0
    return (1 - EPS) / p


def draw_card(u: float) -> int:
    """Map an rng_float draw to a card rank in 1..13."""
    return int(u * RANKS) + 1


def resolve(direction: str, current: int, drawn: int) -> bool:
    """True == win. Ties count as a loss (matches the p_hi/p_lo formulas)."""
    if direction == "higher":
        return drawn > current
    return drawn < current


def rtp_distribution(direction: str, r: int) -> list[tuple[float, float]]:
    """Single pick: [(P(win), step_multiplier), (P(lose), 0)]."""
    p = prob(direction, r)
    return [(p, step_multiplier(direction, r)), (1.0 - p, 0.0)]

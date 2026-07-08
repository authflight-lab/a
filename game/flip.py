"""Flip — chainable multi-step (spec §7.2).

    M_flip(k) = 1.96^k     (per flip factor = (1 - EPS) / 0.5 = 1.96)

Each flip is one RNG draw; win with probability 0.5. Client calls ``/step`` per
flip and ``/cashout`` to lock the accumulated multiplier.
"""

from . import EPS

# (1 - EPS) / p_win with p_win = 0.5 -> 1.96
P_WIN = 0.5
FACTOR = (1 - EPS) / P_WIN  # 1.96


def multiplier(streak: int) -> float:
    return FACTOR ** streak


def flip_once(u: float) -> bool:
    """True == win. ``u`` is a single rng_float draw in [0, 1)."""
    return u < P_WIN


def rtp_distribution() -> list[tuple[float, float]]:
    """Single flip: [(P(win), FACTOR), (P(lose), 0)]."""
    return [(P_WIN, FACTOR), (1.0 - P_WIN, 0.0)]

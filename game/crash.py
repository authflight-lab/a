"""Crash — solo curve, cashout-only (spec §7 addendum, task 3).

One RNG draw fixes the whole round at bet time:

    u  = rng_float(server_seed, client_seed, nonce, cursor=0)
    cp = min(max(1.0, (1 - EPS) / u), CRASH_CAP)

The player cashes out at a multiplier m of their choosing; they win m iff
m < cp, else they bust. For any m in (1, CRASH_CAP):

    P(win at m) = P(cp > m) = P(u < (1 - EPS)/m) = (1 - EPS)/m
    EV          = m * (1 - EPS)/m = 1 - EPS

so the house edge is exactly EPS at every cashout target — no timing skill
involved. The client's rising-curve animation is cosmetic; the server never
tracks elapsed time, only the multiplier the client claims at cashout.

P(instant bust, cp == 1.0) = P(u >= 1 - EPS) = EPS.
"""

from . import EPS, MULT_CAP
from .seed import rng_float

# Crash shares the global economy ceiling: a cashout can never exceed it, and
# the crash point itself is clamped to it so m == CRASH_CAP always busts.
CRASH_CAP = MULT_CAP


def crash_point(server_seed: str, client_seed: str, nonce: int) -> float:
    """The round's predetermined crash multiplier (server-side secret until settle)."""
    u = rng_float(server_seed, client_seed, nonce, 0)
    if u <= 0.0:  # measure-zero guard: rng_float is in [0, 1)
        return CRASH_CAP
    return min(max(1.0, (1.0 - EPS) / u), CRASH_CAP)


def sample_crash_points(n: int, server_seed: str = "s" * 64,
                        client_seed: str = "c" * 64) -> list[float]:
    """n seeded crash-point draws (nonce = 0..n-1) for Monte-Carlo RTP checks."""
    return [crash_point(server_seed, client_seed, i) for i in range(n)]


def simulate_ev(cps: list[float], target: float) -> float:
    """Monte-Carlo EV of the strategy "cash out at ``target``" over a sample.

    Payoff per round is ``target`` if the crash point exceeds it, else 0, so
    the empirical mean must land on 1 - EPS for ANY fixed target in
    (1, CRASH_CAP). (Note E[cp] itself is NOT the RTP — cashing out exactly at
    the crash point busts; only strictly below it wins.)
    """
    return sum(target for cp in cps if cp > target) / len(cps)

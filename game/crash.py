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


def simulate_ev(n: int, server_seed: str = "s" * 64, client_seed: str = "c" * 64) -> float:
    """Monte-Carlo E[min(cp, CRASH_CAP)] over n seeded draws (RTP test helper).

    A player who could cash out at the last instant before the crash would
    collect exactly min(cp, CRASH_CAP); its expectation is the game's RTP
    ceiling and must come out ~= 1 - EPS.
    """
    total = 0.0
    for i in range(n):
        total += crash_point(server_seed, client_seed, i)
    return total / n

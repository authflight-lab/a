"""Crash — solo curve, cashout-only (spec §7 addendum, task 3).

One RNG draw fixes the whole round at bet time:

    u  = rng_float(server_seed, client_seed, nonce, cursor=0)
    cp = min(max(1.0, (1 - EPS) / u), CRASH_CAP)

The player cashes out at a multiplier m of their choosing; they win m iff
m < cp, else they bust. For any m in (1, CRASH_CAP):

    P(win at m) = P(cp > m) = P(u < (1 - EPS)/m) = (1 - EPS)/m
    EV          = m * (1 - EPS)/m = 1 - EPS

so the house edge is exactly EPS at every cashout target — no timing skill
involved.

The round is TIME-based and server-anchored: the bet stores a server t0, the
live multiplier is mult_at(now - t0) = e^(GROWTH * elapsed_ms), and the round
autonomously crashes the moment that curve reaches cp (enforced by /cashout
and the /crash/check poll — a claim arriving after that instant busts, and a
win is clamped to the server-clock multiplier so a client cannot claim ahead
of time). The client animates the same formula; the server clock is the truth.

P(instant bust, cp == 1.0) = P(u >= 1 - EPS) = EPS.
"""

import math

from . import EPS, MULT_CAP
from .seed import rng_float

# Crash shares the global economy ceiling: a cashout can never exceed it, and
# the crash point itself is clamped to it so m == CRASH_CAP always busts.
CRASH_CAP = MULT_CAP

# Curve growth per elapsed millisecond: mult(t) = e^(GROWTH * t). Tuned so the
# curve reaches ~2x in ~11.5s, ~5x in ~27s, ~25x in ~54s. MUST match the
# client animation in app/js/games/crash.js (GROWTH there) — the server clock
# settles the round, the client only draws it.
GROWTH = 0.00006


def mult_at(elapsed_ms: float) -> float:
    """The live curve multiplier after ``elapsed_ms`` on the server clock."""
    if elapsed_ms <= 0:
        return 1.0
    return math.exp(GROWTH * elapsed_ms)


def crash_ms(cp: float) -> float:
    """Milliseconds from bet until the curve reaches the crash point ``cp``."""
    return math.log(max(cp, 1.0)) / GROWTH


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

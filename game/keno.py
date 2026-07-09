"""Keno — single-settle (Stake model).

Grid of 40 numbers. The player picks 1..10; the server draws 10 DISTINCT
numbers. Payout multiplier = ``PAYTABLE[picks][hits]`` where
``hits = |picks ∩ drawn|``. One pick → frequent small wins; ten picks → rare
huge ones. It is a single-action bet (pick → draw → settle), so it rides the
same one-request ``/play`` path as dice/plinko — no ``/step``, no chaining.

Math (fixed by the format, not a design choice). The hit count is
hypergeometric:

    P(h | k) = C(10, h) * C(30, k - h) / C(40, k)

``SHAPE`` is the art-directed *relative* payout profile per pick-count. The
``PAYTABLE`` is derived from it AT IMPORT: each row is multiplied by a single
scale ``factor = (1 - EPS) / raw`` that forces the edge identity

    sum_h  P(h | k) * PAYTABLE[k][h]  ==  1 - EPS

to hold EXACTLY for every k (verified by ``rtp_distribution`` in the RTP test).
Tune ``SHAPE`` freely for volatility — the factor always re-normalises the row
back to the house edge. ``EPS`` is the same global house edge as every other
game (see ``game/__init__.py``).

Draw fairness: pulling 10 distinct numbers walks a shrinking pool of sizes
40, 39, 38, ... — all non-powers-of-two, exactly where the naïve
``int(u * n)`` truncation in ``seed.rng_int`` would bias the low indices. Keno
is the most audit-visible game for that bias, so the draw here uses a
REJECTION-SAMPLED index (``_draw_index``) built from the same HMAC bytes.
"""

import math

from . import EPS
from .seed import rng_float

GRID, DRAW, MAX_PICKS = 40, 10, 10

# Relative payout weights per (picks k, hits h). Missing hit counts pay 0. This
# is the Stake/Rainbet "Classic" profile: a deliberately FLAT middle (modest
# wins at the hit counts players actually reach) that only spikes near the top
# (10-pick tops at ~100x). It is used as SHAPE — the per-row factor rescales each
# row to the exact edge, and since the raw curve already sits at ~0.99 RTP the
# factor is ~0.99, so the numbers land right on the Classic table. An earlier
# top-heavy SHAPE was mathematically fair (edge still exact) but felt unplayable:
# it paid nothing until near-impossible hit counts, then paid lottery numbers.
SHAPE = {
    1:  {1: 3.96},
    2:  {1: 1.9, 2: 4.5},
    3:  {1: 1, 2: 3.1, 3: 10.4},
    4:  {1: 0.8, 2: 1.8, 3: 5, 4: 22.5},
    5:  {1: 0.25, 2: 1.4, 3: 4.1, 4: 16.5, 5: 36},
    6:  {2: 1, 3: 3.68, 4: 7, 5: 16.5, 6: 40},
    7:  {2: 0.47, 3: 3, 4: 4.5, 5: 14, 6: 31, 7: 60},
    8:  {3: 2.2, 4: 4, 5: 13, 6: 22, 7: 55, 8: 70},
    9:  {3: 1.55, 4: 3, 5: 8, 6: 15, 7: 44, 8: 60, 9: 85},
    10: {3: 1.4, 4: 2.25, 5: 4.5, 6: 8, 7: 17, 8: 50, 9: 80, 10: 100},
}


def p_hit(k: int, h: int) -> float:
    """Hypergeometric P(h hits | k picks). 0 outside the feasible range."""
    if h < 0 or h > k or (k - h) > (GRID - DRAW):
        return 0.0
    return math.comb(DRAW, h) * math.comb(GRID - DRAW, k - h) / math.comb(GRID, k)


def _build_paytable() -> dict[int, dict[int, float]]:
    """Rescale each SHAPE row so its RTP is exactly ``1 - EPS`` (unrounded)."""
    table: dict[int, dict[int, float]] = {}
    for k, shape in SHAPE.items():
        raw = sum(p_hit(k, h) * m for h, m in shape.items())
        factor = (1 - EPS) / raw
        table[k] = {h: shape.get(h, 0.0) * factor for h in range(k + 1)}
    return table


PAYTABLE = _build_paytable()


def valid_picks(picks) -> bool:
    """1..10 numbers, all distinct, all in [1, 40]. Rejects bools/floats."""
    if not isinstance(picks, (list, tuple)):
        return False
    if not (1 <= len(picks) <= MAX_PICKS):
        return False
    seen: set[int] = set()
    for p in picks:
        # bool is an int subclass — reject it explicitly so True/False can't
        # masquerade as 1/0.
        if isinstance(p, bool) or not isinstance(p, int):
            return False
        if p < 1 or p > GRID or p in seen:
            return False
        seen.add(p)
    return True


def _draw_index(server_seed: str, client_seed: str, nonce: int, base_cursor: int, n: int) -> int:
    """Unbiased index in [0, n) via rejection sampling on a 32-bit HMAC draw.

    ``rng_float`` packs the first 4 HMAC bytes into a value in [0, 1) with exact
    32-bit resolution; ``int(u * 2**32)`` recovers that 32-bit integer. We keep
    only draws below the largest multiple of ``n`` that fits in 32 bits, so every
    residue class 0..n-1 is equally likely — no ``int(u*n)`` low-index bias.
    Rejection probability is < n / 2**32 (~1e-8 for n<=40), so the 64-attempt
    budget is astronomically sufficient; the final fallback never runs in
    practice but keeps the function total.
    """
    span = 1 << 32
    limit = span - (span % n)
    r = 0
    for attempt in range(64):
        u = rng_float(server_seed, client_seed, nonce, base_cursor + attempt)
        r = int(u * span)
        if r < limit:
            return r % n
    return r % n


def draw(server_seed: str, client_seed: str, nonce: int) -> list[int]:
    """10 DISTINCT numbers from 1..40, provably fair and unbiased."""
    pool = list(range(1, GRID + 1))
    drawn: list[int] = []
    for i in range(DRAW):
        # Each draw gets its own cursor window (64 slots) so rejections in one
        # draw can never collide with the next draw's bytes.
        j = _draw_index(server_seed, client_seed, nonce, i * 64, len(pool))
        drawn.append(pool.pop(j))
    return drawn


def settle(picks, drawn: list[int]) -> dict:
    hits = len(set(picks) & set(drawn))
    mult = PAYTABLE[len(picks)][hits]
    return {"picks": list(picks), "drawn": drawn, "hits": hits, "multiplier": mult}


def play(server_seed: str, client_seed: str, nonce: int, picks) -> dict:
    """Draw + settle in one call (the single-settle entry point)."""
    return settle(picks, draw(server_seed, client_seed, nonce))


def rtp_distribution(k: int) -> list[tuple[float, float]]:
    """[(P(h|k), PAYTABLE[k][h]) for h in 0..k] — the RTP-identity input."""
    return [(p_hit(k, h), PAYTABLE[k][h]) for h in range(k + 1)]

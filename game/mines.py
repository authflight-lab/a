"""Mines — multi-step (spec §7.3).

    M_mines(k, m) = (1 - edge(k)) * prod_{i=0}^{k-1} (25 - i) / (25 - m - i)

Mine positions come from a Fisher-Yates shuffle of the 25 cells (seeded RNG),
taking the first ``m``. Client calls ``/step`` per reveal, ``/cashout`` to lock.

Unlike the other games (which carry a flat ``EPS`` edge), mines FRONT-LOADS its
house edge: a flat edge still leaves the very first safe reveal paying above 1x
even at low mine counts, so a player could reveal a single tile and cash out for
a near-guaranteed small profit. ``edge(k)`` is heaviest on the first reveal and
decays geometrically back to the base ``EPS``, so early cash-outs — especially at
low mine counts — start below 1x and only turn a profit after a few reveals or
once enough mines are on the board.
"""

from . import EPS
from .seed import rng_int

TOTAL = 25

# Front-loaded edge: edge(k) = EPS + EDGE_RAMP * EDGE_DECAY^(k-1). Tuned so the
# first safe reveal is ~0.90x at 1 mine, ~1x around 3-4 mines, and >1x for higher
# mine counts. Adjust these two constants to reshape (or, EDGE_RAMP=0 to disable)
# the ramp; the RTP identity test derives its target from edge() so it follows.
EDGE_RAMP = 0.13
EDGE_DECAY = 0.6

# Flat scale on every mines multiplier (the progression was scaling too fast).
# 0.90 => all multipliers 10% lower; set to 1.0 to remove the reduction.
MULT_SCALE = 0.90


def edge(k: int) -> float:
    """Effective house edge after ``k`` safe reveals (>= EPS, decays to EPS)."""
    if k <= 0:
        return 0.0
    return EPS + EDGE_RAMP * (EDGE_DECAY ** (k - 1))


def multiplier(k: int, m: int) -> float:
    """Multiplier after ``k`` safe reveals with ``m`` mines on the board."""
    prod = 1.0
    for i in range(k):
        prod *= (TOTAL - i) / (TOTAL - m - i)
    return MULT_SCALE * (1 - edge(k)) * prod


def mine_positions(server_seed: str, client_seed: str, nonce: int, m: int) -> list[int]:
    """Fisher-Yates shuffle of the 25 cells; the first ``m`` are mines."""
    cells = list(range(TOTAL))
    cursor = 0
    for i in range(TOTAL - 1, 0, -1):
        j = rng_int(server_seed, client_seed, nonce, cursor, i + 1)
        cells[i], cells[j] = cells[j], cells[i]
        cursor += 1
    return sorted(cells[:m])


def rtp_distribution(k: int, m: int) -> list[tuple[float, float]]:
    """[(P(survive k), M(k, m)), (P(hit a mine), 0)]."""
    p = 1.0
    for i in range(k):
        p *= (TOTAL - m - i) / (TOTAL - i)
    return [(p, multiplier(k, m)), (1.0 - p, 0.0)]

"""HighLow — multi-step (spec §7.5).

    p_hi(r) = (14 - r) / 13,   p_lo(r) = r / 13
    M_chain = (1 - HL_EPS)^n * prod_{j=1}^{n} 1 / p_{d_j}(r_j)

Deck is ranks 1..13. Uses the "Rainbet" rule: a tie (next card equals current
rank) counts as a WIN for whichever direction was picked ("higher or same" /
"lower or same"). HighLow carries its OWN house edge ``HL_EPS`` (larger than the
global ``EPS``) because its multipliers chain and can compound quickly.

Two guards keep payouts sane for a points economy:

* Aces (1) and Kings (13) are WILD — they never sit as the current decision card
  (that would make one side a guaranteed 100% win that can only *shrink* the
  chain). When one is dealt it passes through to the next non-wild card. The
  *revealed* next card stays on the full 1..13 deck, so the current card is
  always 2..12, every side's win chance stays in [1/13, 12/13], and each step's
  factor stays in (~1.03x, ~6.18x] — no single card can hand out a huge jump.
* ``HL_MAX_MULT`` caps the chain multiplier so a lucky run can't balloon into an
  absurd payout; a step that would exceed it is not offered (cash out instead).
"""

# HighLow-specific house edge (5%). Larger than the global EPS (1%) on purpose;
# other games are unaffected.
HL_EPS = 0.05

RANKS = 13

# rng sub-indices reserved per slot so wild cards can be skipped deterministically
# without colliding with the next slot's draws. P(needing all 64) ~ (2/13)^64.
STRIDE = 64

# Hard ceiling on the chain multiplier (economy guard). A winning step that would
# push the chain past this is not offered — the player cashes out instead.
HL_MAX_MULT = 25.0


def p_higher(r: int) -> float:
    """P(next >= r): ranks r..13 win (includes the tie)."""
    return (RANKS - r + 1) / RANKS


def p_lower(r: int) -> float:
    """P(next <= r): ranks 1..r win (includes the tie)."""
    return r / RANKS


def prob(direction: str, r: int) -> float:
    return p_higher(r) if direction == "higher" else p_lower(r)


def step_multiplier(direction: str, r: int) -> float:
    """Per-step multiplier factor (1 - HL_EPS) / p_dir(r)."""
    p = prob(direction, r)
    if p <= 0.0:
        return 0.0
    return (1 - HL_EPS) / p


def can_pick(direction: str, r: int) -> bool:
    """Offered only if the pick can grow the chain (step factor > 1x)."""
    return step_multiplier(direction, r) > 1.0


def within_cap(cur_mult: float, direction: str, r: int) -> bool:
    """False once a winning step would push the chain past ``HL_MAX_MULT``."""
    return cur_mult * step_multiplier(direction, r) <= HL_MAX_MULT + 1e-9


def draw_card(u: float) -> int:
    """Map an rng_float draw to a card rank in 1..13."""
    return int(u * RANKS) + 1


def is_wild(r: int) -> bool:
    """Aces (1) and Kings (13) are wild — they never sit as the current card."""
    return r <= 1 or r >= RANKS


def reveal_card(draw_float, slot: int) -> int:
    """The revealed next card (full 1..13 deck) for a slot. ``draw_float(i)`` maps
    an rng index to a uniform float in [0, 1)."""
    return draw_card(draw_float(slot * STRIDE))


def current_card(draw_float, slot: int, start_j: int = 0) -> int:
    """First non-wild card (2..12) at ``slot``, skipping wilds from ``start_j``."""
    for j in range(start_j, STRIDE):
        r = draw_card(draw_float(slot * STRIDE + j))
        if not is_wild(r):
            return r
    return (RANKS + 1) // 2  # fallback; probability ~ (2/13)^STRIDE


def resolve(direction: str, current: int, drawn: int) -> bool:
    """True == win. Ties count as a win for the picked direction."""
    if direction == "higher":
        return drawn >= current
    return drawn <= current


def rtp_distribution(direction: str, r: int) -> list[tuple[float, float]]:
    """Single pick: [(P(win), step_multiplier), (P(lose), 0)]."""
    p = prob(direction, r)
    return [(p, step_multiplier(direction, r)), (1.0 - p, 0.0)]

"""HighLow — multi-step (spec §7.5).

    p_hi(r) = (14 - r) / 13,   p_lo(r) = r / 13
    M_chain = (1 - HL_EPS)^n * prod_{j=1}^{n} 1 / p_{d_j}(r_j)

Ranks are 1..13. Uses the "Rainbet" rule: a tie (next card equals current rank)
counts as a WIN for whichever direction was picked ("higher or same" / "lower
or same"). HighLow carries its OWN house edge ``HL_EPS`` (larger than the global
``EPS``) because its multipliers chain and can compound quickly — a tighter edge
here protects the points economy from high-variance runaway payouts.

A direction is only offered when it can actually grow the multiplier, i.e. when
its step factor ``(1 - HL_EPS) / p_dir(r) > 1``  <=>  ``p_dir(r) < 1 - HL_EPS``.
This disables the degenerate "guaranteed win" side (``lower`` at r=13, ``higher``
at r=1), which would otherwise pay < 1x and *shrink* the chain. At least one
side is always available for every rank.
"""

# HighLow-specific house edge (5%). Larger than the global EPS (1%) on purpose;
# other games are unaffected.
HL_EPS = 0.05

RANKS = 13


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


def draw_card(u: float) -> int:
    """Map an rng_float draw to a card rank in 1..13."""
    return int(u * RANKS) + 1


def resolve(direction: str, current: int, drawn: int) -> bool:
    """True == win. Ties count as a win for the picked direction."""
    if direction == "higher":
        return drawn >= current
    return drawn <= current


def rtp_distribution(direction: str, r: int) -> list[tuple[float, float]]:
    """Single pick: [(P(win), step_multiplier), (P(lose), 0)]."""
    p = prob(direction, r)
    return [(p, step_multiplier(direction, r)), (1.0 - p, 0.0)]

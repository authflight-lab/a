"""RTP identity test for all 6 games (spec §6).

For every game and every valid configuration/decision point:

    sum_over_outcomes  P(outcome) * M(outcome)  ==  RTP target   (+/- 1e-9)

The target is ``1 - EPS`` for dice, flip and plinko. Three games deviate on
purpose: mines front-loads its edge (target ``MULT_SCALE * (1 - edge(k))``),
towers damps its whole ladder (target ``MULT_SCALE * (1 - EPS)``), and highlow
uses its own ``HL_EPS``. Multi-step games satisfy their identity at each
decision: P(reaching the state) * M(state) + P(busting) * 0 == target.
"""

import math

from api.game import EPS, dice, flip, highlow, mines, plinko, towers

TARGET = 1 - EPS
TOL = 1e-9


def _rtp(dist) -> float:
    return math.fsum(p * m for p, m in dist)


def test_dice_rtp():
    for target in range(dice.T_MIN, dice.T_MAX + 1):
        assert abs(_rtp(dice.rtp_distribution(target)) - TARGET) < TOL


def test_flip_rtp():
    assert abs(_rtp(flip.rtp_distribution()) - TARGET) < TOL


def test_mines_rtp():
    # Mines front-loads its house edge (edge decays from a heavier first-reveal
    # edge back to the base EPS), so the per-reveal RTP is 1 - edge(k) — the same
    # for every mine count m at a given reveal depth k, but no longer a flat
    # TARGET across depths.
    for m in range(1, mines.TOTAL):  # 1..24 mines
        for k in range(0, mines.TOTAL - m + 1):  # 0..(safe cells) reveals
            assert abs(_rtp(mines.rtp_distribution(k, m)) - mines.MULT_SCALE * (1 - mines.edge(k))) < TOL


def test_towers_rtp():
    # Towers is uniformly damped 15% below the flat-EPS identity (operator
    # decision): RTP == MULT_SCALE * (1 - EPS) at every decision point.
    towers_target = towers.MULT_SCALE * TARGET
    for difficulty in towers.DIFFICULTIES:
        for level in range(0, 12):
            assert abs(_rtp(towers.rtp_distribution(level, difficulty)) - towers_target) < TOL


def test_highlow_rtp():
    # HighLow carries its own (larger) house edge, so its RTP target differs.
    hl_target = 1 - highlow.HL_EPS
    for r in range(1, highlow.RANKS + 1):
        for direction in ("higher", "lower"):
            if not highlow.can_pick(direction, r):
                continue
            assert abs(_rtp(highlow.rtp_distribution(direction, r)) - hl_target) < TOL


def test_plinko_rtp():
    for n in plinko.ROWS:
        for risk in plinko.RISK:
            assert abs(_rtp(plinko.rtp_distribution(n, risk)) - TARGET) < TOL

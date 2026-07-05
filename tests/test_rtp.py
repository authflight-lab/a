"""RTP identity test for all 6 games (spec §6).

For every game and every valid configuration/decision point:

    sum_over_outcomes  P(outcome) * M(outcome)  ==  1 - EPS   (+/- 1e-9)

Multi-step games (flip, mines, towers, highlow) satisfy the identity at each
decision: P(reaching the state) * M(state) + P(busting) * 0 == 1 - EPS.
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
    for m in range(1, mines.TOTAL):  # 1..24 mines
        for k in range(0, mines.TOTAL - m + 1):  # 0..(safe cells) reveals
            assert abs(_rtp(mines.rtp_distribution(k, m)) - TARGET) < TOL


def test_towers_rtp():
    for difficulty in towers.DIFFICULTIES:
        for level in range(0, 12):
            assert abs(_rtp(towers.rtp_distribution(level, difficulty)) - TARGET) < TOL


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

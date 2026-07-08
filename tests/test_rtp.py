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

from api.game import EPS, chicken, crash, dice, flip, highlow, mines, plinko, rps, towers

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


def test_rps_rtp():
    # Ties are EV-neutral replays (multiplier unchanged, fresh draw), so the
    # identity holds per RESOLVED round: P(win)*FACTOR + P(lose)*0 == 1 - EPS.
    assert abs(_rtp(rps.rtp_distribution()) - TARGET) < TOL
    # The ladder: 1.96^4 = 14.76x is the last full rung; the 5th win exceeds
    # the 20x cap (auto-cashout territory).
    assert rps.multiplier(4) < rps.RPS_MAX_MULT < rps.multiplier(5)


def test_rps_rules():
    # rock(0) beats scissors(2), paper(1) beats rock(0), scissors(2) beats paper(1)
    assert rps.beats(0, 2) and rps.beats(1, 0) and rps.beats(2, 1)
    assert not (rps.beats(2, 0) or rps.beats(0, 1) or rps.beats(1, 2))
    assert not any(rps.beats(h, h) for h in range(3))
    # house_hand maps [0,1) uniformly onto 0..2 and never overflows.
    assert rps.house_hand(0.0) == 0
    assert rps.house_hand(0.5) == 1
    assert rps.house_hand(0.999999999) == 2


def test_chicken_rtp():
    # Flat-EPS identity at every decision point, for every difficulty:
    # P(surviving L lanes) * M(L) + P(hit) * 0 == MULT_SCALE * (1 - EPS).
    chicken_target = chicken.MULT_SCALE * TARGET
    for difficulty in chicken.DIFFICULTIES:
        for lanes in range(0, chicken.LANES + 1):
            assert abs(_rtp(chicken.rtp_distribution(lanes, difficulty)) - chicken_target) < TOL


def test_chicken_ladder():
    # Per-lane growth matches the spec (~1.47 / 1.96 / 2.94 / 3.92) and every
    # difficulty crosses the 20x cap within the 8-lane road, so a run always
    # terminates via auto-cashout or the far side.
    growth = {"easy": 1.47, "medium": 1.96, "hard": 2.94, "daredevil": 3.92}
    for difficulty, g in growth.items():
        assert abs(chicken.multiplier(1, difficulty) - g) < 1e-9
    # easy is the slowest ladder: it exceeds the cap only on the final lane.
    assert chicken.multiplier(chicken.LANES - 1, "easy") < chicken.CHICKEN_MAX_MULT
    assert chicken.multiplier(chicken.LANES, "easy") > chicken.CHICKEN_MAX_MULT


def test_chicken_car_zones():
    # car_zones returns exactly T distinct in-range zones for every lane.
    ss, cs = "s" * 64, "c" * 64
    for difficulty, d in chicken.DIFFICULTIES.items():
        for lane in range(chicken.LANES):
            cars = chicken.car_zones(ss, cs, 0, lane, difficulty)
            assert len(cars) == d["T"] == len(set(cars))
            assert all(0 <= z < d["C"] for z in cars)


def test_crash_rtp():
    # Crash's outcome is continuous, so the identity is checked by Monte Carlo:
    # for a fixed cashout target m the payoff is m·1{cp > m}, whose expectation
    # is exactly (1 - EPS) for any m in (1, CRASH_CAP). The 100k HMAC draws are
    # deterministic (fixed seeds), so this is a stable check, not a flaky one.
    cps = crash.sample_crash_points(400_000)
    for target in (1.5, 2.0, 5.0):
        ev = crash.simulate_ev(cps, target)
        assert abs(ev - TARGET) < 0.005, f"target={target} ev={ev}"

    # Formula bounds: cp is clamped to [1.0, CRASH_CAP] for every draw, and the
    # cap matches the global economy ceiling.
    assert crash.CRASH_CAP == 25.0
    for i in range(200):
        cp = crash.crash_point("s" * 64, "c" * 64, i)
        assert 1.0 <= cp <= crash.CRASH_CAP


def test_plinko_rtp():
    for n in plinko.ROWS:
        for risk in plinko.RISK:
            assert abs(_rtp(plinko.rtp_distribution(n, risk)) - TARGET) < TOL

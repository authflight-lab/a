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

from api.game import EPS, blackjack, chicken, crash, dice, flip, highlow, mines, plinko, rps, towers

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
    # Flat identity at every decision point:
    # P(surviving L lanes) * M(L) + P(hit) * 0 == (1 - edge), where easy uses
    # its own 4% edge (Rainbet-style ladder) and legacy difficulties keep EPS.
    easy_target = 1 - chicken.CHICKEN_EDGE
    for lanes in range(0, chicken.LANES + 1):
        assert abs(_rtp(chicken.rtp_distribution(lanes, "easy")) - easy_target) < TOL
    for difficulty in chicken.LEGACY:
        for lanes in range(0, 9):
            assert abs(_rtp(chicken.rtp_distribution(lanes, difficulty)) - TARGET) < TOL


def test_chicken_ladder():
    # The easy ladder matches the published Rainbet-style progression exactly
    # (0.96 * 25/(25-n)) and terminates at the 24x cap on the final lane.
    published = [1.00, 1.04, 1.09, 1.14, 1.20, 1.26, 1.33, 1.41, 1.50]
    for n, want in enumerate(published, start=1):
        assert abs(round(chicken.multiplier(n, "easy"), 2) - want) < 1e-9
    assert chicken.multiplier(chicken.LANES - 1, "easy") < chicken.CHICKEN_MAX_MULT
    assert abs(chicken.multiplier(chicken.LANES, "easy") - chicken.CHICKEN_MAX_MULT) < 1e-9
    # Only "easy" is selectable; legacy difficulties settle but can't be bet.
    assert chicken.valid_difficulty("easy")
    for difficulty in chicken.LEGACY:
        assert not chicken.valid_difficulty(difficulty)


def test_chicken_car_zones():
    # car_zones returns exactly one in-range zone per easy lane (deck shrinks
    # by one zone per lane), and T distinct zones for legacy difficulties.
    ss, cs = "s" * 64, "c" * 64
    for lane in range(chicken.LANES):
        cars = chicken.car_zones(ss, cs, 0, lane, "easy")
        assert len(cars) == 1
        assert all(0 <= z < chicken.zones("easy", lane) for z in cars)
    for difficulty, (dc, dt) in chicken.LEGACY.items():
        for lane in range(8):
            cars = chicken.car_zones(ss, cs, 0, lane, difficulty)
            assert len(cars) == dt == len(set(cars))
            assert all(0 <= z < dc for z in cars)


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


def test_blackjack_rtp():
    # Blackjack pays REAL fixed table odds (2x/2.5x/1x/0x), not an EPS-scaled
    # multiplier, so there's no single closed-form target to assert exact
    # equality against (unlike every other game here). Instead: simulate a
    # simple mimic-the-dealer player strategy and assert the resulting RTP
    # lands in blackjack's well-known real-world band (~99.5% with basic
    # strategy, lower with a naive one) — a sanity check on the payout table
    # and natural/bust/push logic, not a fairness proof.
    dist = blackjack.rtp_distribution(200_000)
    rtp = _rtp(dist)
    assert 0.85 < rtp < 1.02, f"blackjack rtp out of band: {rtp}"
    # Every outcome must be one of the four real fixed multipliers.
    assert all(m in (0.0, 1.0, 2.0, 2.5) for _, m in dist)


def test_blackjack_hand_logic():
    # Ace reduction: A + 6 is a soft 17 (counts as 11+6), not a bust.
    total, soft = blackjack.hand_total([1, 6])
    assert total == 17 and soft
    # A + K is a natural 21 (2 cards) — is_blackjack only fires on exactly 2.
    assert blackjack.is_blackjack([1, 13])
    assert not blackjack.is_blackjack([7, 7, 7])  # 21 via 3 cards is NOT a natural
    # Two aces reduce to 12, not bust.
    total, soft = blackjack.hand_total([1, 1])
    assert total == 12 and soft
    # Bust detection and dealer S17 (stands on soft 17 too).
    assert blackjack.is_bust(22)
    assert not blackjack.is_bust(21)
    assert not blackjack.dealer_should_hit([1, 6])  # soft 17 -> stand (S17)
    assert blackjack.dealer_should_hit([10, 6])     # hard 16 -> hit
    # outcome_multiplier: dealer bust pays 2x even if player is low but not bust.
    assert blackjack.outcome_multiplier([10, 5], [10, 6, 10]) == 2.0
    # push (equal, no bust) pays 1x.
    assert blackjack.outcome_multiplier([10, 9], [10, 9]) == 1.0
    # player bust always loses regardless of dealer hand.
    assert blackjack.outcome_multiplier([10, 9, 5], [2, 2]) == 0.0
    # natural_outcome: player natural vs non-natural dealer pays 2.5x; both
    # natural is a push; no player natural returns None (round continues).
    assert blackjack.natural_outcome([1, 13], [10, 9]) == 2.5
    assert blackjack.natural_outcome([1, 13], [1, 12]) == 1.0
    assert blackjack.natural_outcome([10, 9], [1, 12]) is None

"""Keno — RTP identity, draw fairness, and pick validation."""

import math

from api.game import EPS, keno

TARGET = 1 - EPS
TOL = 1e-9


def _rtp(dist) -> float:
    return math.fsum(p * m for p, m in dist)


def test_keno_rtp():
    # Every pick-count's paytable row must return exactly (1 - EPS) — the whole
    # point of the per-row scale factor. A single fat-fingered SHAPE cell would
    # silently shift the edge, so this locks it.
    for k in range(1, keno.MAX_PICKS + 1):
        assert abs(_rtp(keno.rtp_distribution(k)) - TARGET) < TOL, f"k={k}"


def test_keno_hit_distribution_sums_to_one():
    # Hypergeometric sanity: P(h|k) over all feasible h is a probability dist.
    for k in range(1, keno.MAX_PICKS + 1):
        total = math.fsum(keno.p_hit(k, h) for h in range(k + 1))
        assert abs(total - 1.0) < TOL, f"k={k} sum={total}"


def test_keno_draw_distinct_and_in_range():
    # 10 distinct numbers in 1..40 for many seeds/nonces.
    for nonce in range(200):
        drawn = keno.draw("s" * 64, "c" * 64, nonce)
        assert len(drawn) == keno.DRAW
        assert len(set(drawn)) == keno.DRAW
        assert all(1 <= n <= keno.GRID for n in drawn)


def test_keno_draw_is_deterministic():
    a = keno.draw("s" * 64, "c" * 64, 7)
    b = keno.draw("s" * 64, "c" * 64, 7)
    assert a == b


def test_keno_draw_unbiased():
    # Rejection-sampled indices must be ~uniform over 1..40. Chi-square-lite:
    # every number should appear roughly DRAW/GRID of the time across N draws.
    N = 20000
    counts = {n: 0 for n in range(1, keno.GRID + 1)}
    for nonce in range(N):
        for n in keno.draw("seed" * 16, "cli" * 21 + "x", nonce):
            counts[n] += 1
    expected = N * keno.DRAW / keno.GRID
    for n, c in counts.items():
        # Within 8% of expectation — a biased int(u*n) truncation would skew the
        # low numbers well past this band.
        assert abs(c - expected) / expected < 0.08, f"n={n} c={c} exp={expected}"


def test_keno_settle_hits_and_multiplier():
    drawn = list(range(1, 11))  # drew 1..10
    # 3 of 4 picks hit (1,2,3 in draw; 40 not).
    res = keno.settle([1, 2, 3, 40], drawn)
    assert res["hits"] == 3
    assert res["drawn"] == drawn
    assert res["multiplier"] == keno.PAYTABLE[4][3]
    # Zero hits pays 0.
    assert keno.settle([40, 39, 38], drawn)["hits"] == 0
    assert keno.settle([40, 39, 38], drawn)["multiplier"] == 0.0


def test_keno_valid_picks():
    assert keno.valid_picks([1])
    assert keno.valid_picks(list(range(1, 11)))  # 10 picks ok
    assert not keno.valid_picks([])              # empty
    assert not keno.valid_picks(list(range(1, 12)))  # 11 picks
    assert not keno.valid_picks([1, 1, 2])       # dupes
    assert not keno.valid_picks([0])             # below range
    assert not keno.valid_picks([41])            # above range
    assert not keno.valid_picks([True])          # bool masquerading as 1
    assert not keno.valid_picks("5")             # not a list

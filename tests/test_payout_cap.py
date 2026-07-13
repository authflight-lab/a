"""Regression: the P_MAX payout ceiling must never clip a LEGITIMATE win.

A prior bug set ``P_MAX = 2000`` which, at ``BET_MAX = 500``, silently clamped
every win to 4x — towers showing 7.45x or keno showing 23.09x both settled to
2000 points. P_MAX is only a last-resort backstop against a mispriced multiplier;
it must sit above the largest legitimate single-round payout so real wins pay in
full. The real economy guard is each game's own multiplier cap.
"""

from api.game import BET_MAX, MULT_CAP, P_MAX, keno
from api.main import _payout


def _keno_max_mult() -> float:
    return max(m for row in keno.PAYTABLE.values() for m in row.values())


def test_pmax_above_largest_legit_win():
    # The biggest legitimate single-round payout is keno's top paytable row at
    # the max bet (keno intentionally exceeds MULT_CAP via its paytable curve).
    largest_legit = BET_MAX * _keno_max_mult()
    assert P_MAX > largest_legit, (
        f"P_MAX={P_MAX} clips a legitimate keno win of {largest_legit:.0f}"
    )


def test_payout_not_clipped_for_real_multipliers():
    # The exact multipliers from the bug report must pay their full amount.
    assert _payout(500, 7.45) == 3725   # towers example
    assert _payout(500, 23.09) == 11545  # keno example
    # A within-MULT_CAP win at max bet is paid in full, not clamped.
    assert _payout(BET_MAX, MULT_CAP) == BET_MAX * MULT_CAP


def test_pmax_still_backstops_a_bugged_multiplier():
    # An absurd (bugged) multiplier is still capped, so the ceiling remains a
    # real safety net rather than being removed entirely.
    assert _payout(BET_MAX, 10_000.0) == P_MAX

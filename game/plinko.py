"""Plinko — single-settle (spec §7.6).

    M(j) = lambda * b^(|j - n/2|^alpha)
    lambda = (1 - EPS) / sum_{j=0}^{n} P(j) * b^(|j - n/2|^alpha)

P(j) is the binomial landing distribution C(n, j) / 2^n (each peg is a fair
50/50 bounce). ``lambda`` is computed once per (n, b, alpha) AT IMPORT and
cached — never recalculated per round. The full path is returned in ``outcome``.

| Risk | b   | alpha |
|------|-----|-------|
| low  | 1.3 | 1.0   |
| high | 2.2 | 1.1   |

Rows n in {8, 12, 16}.
"""

import math

from . import EPS
from .seed import rng_float

RISK = {
    "low": {"b": 1.3, "alpha": 1.0},
    "high": {"b": 2.2, "alpha": 1.1},
}
ROWS = (8, 12, 16)


def _binom_pmf(n: int, j: int) -> float:
    return math.comb(n, j) / (2 ** n)


def _bucket_weight(j: int, n: int, b: float, alpha: float) -> float:
    return b ** (abs(j - n / 2) ** alpha)


def _compute_lambda(n: int, b: float, alpha: float) -> float:
    denom = sum(_binom_pmf(n, j) * _bucket_weight(j, n, b, alpha) for j in range(n + 1))
    return (1 - EPS) / denom


# Precompute + cache lambda for every supported (n, risk) at import time.
_LAMBDA_CACHE: dict[tuple[int, str], float] = {}
for _n in ROWS:
    for _risk, _cfg in RISK.items():
        _LAMBDA_CACHE[(_n, _risk)] = _compute_lambda(_n, _cfg["b"], _cfg["alpha"])


def valid_rows(n: int) -> bool:
    return n in ROWS


def valid_risk(risk: str) -> bool:
    return risk in RISK


def multiplier(j: int, n: int, risk: str) -> float:
    cfg = RISK[risk]
    lam = _LAMBDA_CACHE[(n, risk)]
    return lam * _bucket_weight(j, n, cfg["b"], cfg["alpha"])


def drop(server_seed: str, client_seed: str, nonce: int, n: int, risk: str) -> dict:
    path = []
    j = 0
    for cursor in range(n):
        u = rng_float(server_seed, client_seed, nonce, cursor)
        right = u < 0.5
        path.append("R" if right else "L")
        j += 1 if right else 0
    return {"bucket": j, "path": path, "rows": n, "risk": risk, "multiplier": multiplier(j, n, risk)}


def rtp_distribution(n: int, risk: str) -> list[tuple[float, float]]:
    """[(P(j), M(j)) for each bucket j in 0..n]."""
    return [(_binom_pmf(n, j), multiplier(j, n, risk)) for j in range(n + 1)]

"""Dice — single-settle (spec §7.1).

    M_dice(T) = (1 - EPS) * 100 / T,   T in [2, 98]
    Roll R = rng_float * 100. Win if R < T.
"""

from . import EPS
from .seed import rng_float

T_MIN = 2
T_MAX = 98


def multiplier(target: int) -> float:
    return (1 - EPS) * 100.0 / target


def valid_target(target: int) -> bool:
    return T_MIN <= target <= T_MAX


def settle(server_seed: str, client_seed: str, nonce: int, target: int) -> dict:
    roll = rng_float(server_seed, client_seed, nonce, 0) * 100.0
    win = roll < target
    return {
        "target": target,
        "roll": roll,
        "win": win,
        "multiplier": multiplier(target) if win else 0.0,
    }


def rtp_distribution(target: int) -> list[tuple[float, float]]:
    """[(P(win), M), (P(lose), 0)] — R uniform on [0, 100) so P(win) = T/100."""
    p_win = target / 100.0
    return [(p_win, multiplier(target)), (1.0 - p_win, 0.0)]

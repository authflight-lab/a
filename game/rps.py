"""RPS — chainable multi-step (Rock Paper Scissors streak ladder).

    M_rps(w) = FACTOR^w,   FACTOR = (1 - EPS) / 0.5 = 1.96

Each round is ONE rng draw mapping to the house hand (0=rock, 1=paper,
2=scissors); the cursor advances per round (ties consume a draw too, keeping
every draw provably deterministic). Ties are NEUTRAL: the chain multiplier is
unchanged and the round replays. Conditional on a resolution the step is a
fair 50/50, so the per-win factor matches Flip's 1.96 and carries the same
flat 2% edge. ``RPS_MAX_MULT`` caps the chain (a capped win auto-cashes) —
1.96^4 = 14.76x is the last full rung; the 5th win lands on the 20x cap.
"""

from . import EPS

HANDS = ("rock", "paper", "scissors")

P_WIN = 0.5  # conditional on a non-tie resolution
FACTOR = (1 - EPS) / P_WIN  # 1.96

# Hard ceiling on the RPS chain multiplier (economy guard, below the global
# MULT_CAP on purpose). A winning step that reaches it auto-cashes out.
RPS_MAX_MULT = 20.0


def multiplier(wins: int) -> float:
    """Chain multiplier after ``wins`` straight wins (uncapped)."""
    return FACTOR ** wins


def house_hand(u: float) -> int:
    """Map one rng_float draw in [0,1) to the house hand index (0..2)."""
    h = int(u * 3)
    return 2 if h > 2 else h


def beats(player: int, house: int) -> bool:
    """True if the player's hand beats the house's (rock>scissors, etc.)."""
    return (player - house) % 3 == 1


def rtp_distribution() -> list[tuple[float, float]]:
    """One RESOLVED (non-tie) round: [(P(win), FACTOR), (P(lose), 0)].
    Ties are EV-neutral replays, so the identity holds per resolution."""
    return [(P_WIN, FACTOR), (1.0 - P_WIN, 0.0)]

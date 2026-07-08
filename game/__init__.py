"""Game engines package.

Every game's outcome is computed here, server-side, from a seeded RNG. Each
engine module exposes pure functions that compute an outcome + multiplier from
`(server_seed, client_seed, nonce, params)`, plus an `rtp_distribution(...)`
helper used by the RTP identity test.

Global constants (spec §6):
- ``EPS``   house edge (2%). Every game derives its RTP from this single value.
- ``P_MAX`` max payout per round (points).
- ``MULT_CAP`` global ceiling on any round's win multiplier (economy guard).
- ``BET_MIN`` / ``BET_MAX`` bet range is ``[BET_MIN, min(BET_MAX, balance)]``.
"""

EPS = 0.02
P_MAX = 2000
# Hard ceiling on any game's win multiplier, so an open-ended progression
# (towers doubling, mines deep-clears, highlow chaining) can't balloon a single
# round far beyond the point-earning economy. A capped step auto-cashes out.
MULT_CAP = 25.0
BET_MIN = 1
BET_MAX = 350

SINGLE_SETTLE = ("dice", "plinko")
MULTI_STEP = ("flip", "mines", "towers", "highlow")
GAMES = ("dice", "flip", "mines", "towers", "highlow", "plinko")

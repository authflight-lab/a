"""Game engines package.

Every game's outcome is computed here, server-side, from a seeded RNG. Each
engine module exposes pure functions that compute an outcome + multiplier from
`(server_seed, client_seed, nonce, params)`, plus an `rtp_distribution(...)`
helper used by the RTP identity test.

Global constants (spec §6):
- ``EPS``   house edge (1%). Every game derives its RTP from this single value.
- ``P_MAX`` max payout per round (points).
- ``BET_MIN`` / ``BET_MAX`` bet range is ``[BET_MIN, min(BET_MAX, balance)]``.
"""

EPS = 0.01
P_MAX = 5000
BET_MIN = 1
BET_MAX = 350

SINGLE_SETTLE = ("dice", "plinko")
MULTI_STEP = ("flip", "mines", "towers", "highlow")
GAMES = ("dice", "flip", "mines", "towers", "highlow", "plinko")

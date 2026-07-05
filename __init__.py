"""Bartender API package — FastAPI game engine, shop, and leaderboard.

All game math (RNG, multipliers, win/loss, payout) lives in this package only.
The bot never calls this API; they coordinate exclusively through Supabase.

Importing this package (or `api.main`) performs NO network I/O — the Supabase
client is created lazily on first use and degrades gracefully when unconfigured.
"""

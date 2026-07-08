"""API settings module (per the locked config-in-code decision, contract §0.2).

Every value is sourced from an environment variable of the SAME NAME, upper-cased
(e.g. ``BT_SUPABASE_URL``, ``BT_SUPABASE_SERVICE_KEY``, ``BOT_TOKEN``,
``BT_APP_ORIGIN``). Defaults are intentionally EMPTY so no secret ever ships in
source — the deploy host MUST supply them. The bot token is duplicated here (also
present in bot/config.py) on purpose (spec §13) because this is a separate
deployable; it is used ONLY for Telegram initData validation.

Required in production:
  - BOT_TOKEN                — Telegram bot token (for initData HMAC validation)
  - BT_SUPABASE_URL          — Supabase project URL
  - BT_SUPABASE_SERVICE_KEY  — Supabase service-role key (server-side authority)
  - BT_APP_ORIGIN            — Mini App origin(s) for the CORS allowlist. One or
                               more EXACT origins, comma-separated, e.g.
                               https://app.partygc.online,https://foo.pages.dev

Reading settings performs NO network I/O.
"""

import os
from dataclasses import dataclass, fields


@dataclass
class Settings:
    # Mirror of the bot token in bot/config.py — used ONLY for initData validation.
    bot_token: str = ""
    bt_supabase_url: str = ""
    bt_supabase_service_key: str = ""
    # HARDCODED (user request): direct Postgres DSN for the asyncpg pool, used to
    # bypass the PostgREST HTTP hop on hot reads. This is the Supavisor SESSION
    # pooler (port 5432, IPv4, statement caching disabled pool-side) for the same
    # Supabase project as bt_supabase_url. The '*' in the password is %2A-encoded
    # so the DSN parses. An env var of the same name (BT_SUPABASE_DB_URL) overrides
    # it when present. NOTE: putting a DB password in source makes this repo as
    # sensitive as the database itself.
    bt_supabase_db_url: str = (
        "postgresql://postgres.jbtakenwtinlipbgrujm:zetro552020%2A"
        "@aws-0-us-east-1.pooler.supabase.com:5432/postgres?sslmode=require"
    )
    # Exact origin of the deployed Mini App; the CORS allowlist is built from this.
    bt_app_origin: str = ""

    # ── Rate limits (env-overridable, e.g. BT_RL_GAME_LIMIT=90) ──────────────
    # Pre-auth per-IP guard across all /bt/api/ routes. Generous because mobile
    # carriers (CGNAT) pool many legit users behind one IP.
    bt_rl_ip_limit: int = 600
    bt_rl_ip_window_sec: int = 60
    # Per-user game bucket, shared by bet + every step + cashout. Short window
    # so tap bursts recover in seconds and Retry-After stays small.
    # 60 / 15 s ≈ 240 req/min sustained — above the fastest human tapping.
    bt_rl_game_limit: int = 60
    bt_rl_game_window_sec: int = 15
    # Per-user /me profile reads.
    bt_rl_me_limit: int = 120
    bt_rl_me_window_sec: int = 60
    # Per-user read endpoints (leaderboard, history, bets, rewards).
    bt_rl_read_limit: int = 40
    bt_rl_read_window_sec: int = 60

    def __post_init__(self) -> None:
        # Environment overrides (same field name, upper-cased) win when present.
        # Values are coerced to the field's declared type (int fields via int()).
        for f in fields(self):
            env_val = os.environ.get(f.name.upper())
            if env_val is not None and env_val != "":
                if f.type is int or f.type == "int":
                    try:
                        parsed = int(env_val)
                    except ValueError:
                        continue  # keep the safe default on a malformed override
                    if parsed > 0:  # zero/negative limits would break the limiter
                        setattr(self, f.name, parsed)
                else:
                    setattr(self, f.name, env_val)


settings = Settings()

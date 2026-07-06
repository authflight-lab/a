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
  - BT_APP_ORIGIN            — exact Mini App origin for the CORS allowlist,
                               e.g. https://app.partygc.online

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
    # Exact origin of the deployed Mini App; the CORS allowlist is built from this.
    bt_app_origin: str = ""

    def __post_init__(self) -> None:
        # Environment overrides (same field name, upper-cased) win when present.
        for f in fields(self):
            env_val = os.environ.get(f.name.upper())
            if env_val is not None and env_val != "":
                setattr(self, f.name, env_val)


settings = Settings()

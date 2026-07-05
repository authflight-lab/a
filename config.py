"""API settings module (per the locked config-in-code decision, contract §0.2).

This is a separate deployable from the bot, so the bot token + Supabase keys are
duplicated here on purpose (spec §13 acknowledges this). Values are placeholders
now (empty strings); each field is overridable by an environment variable of the
SAME NAME (upper-cased) when present, e.g. ``BT_SUPABASE_URL``.

Reading settings performs NO network I/O.
"""

import os
from dataclasses import dataclass, fields


@dataclass
class Settings:
    # Mirror of the bot token in bot/config.py — used ONLY for initData validation.
    bot_token: str = "8685205697:AAHMw11dVTkN0gW-hnvt5Igku5au6D5BlmQ"
    bt_supabase_url: str = ""
    bt_supabase_service_key: str = ""
    bt_app_origin: str = ""

    def __post_init__(self) -> None:
        # Environment overrides (same field name, upper-cased) win when present.
        for f in fields(self):
            env_val = os.environ.get(f.name.upper())
            if env_val is not None and env_val != "":
                setattr(self, f.name, env_val)


settings = Settings()

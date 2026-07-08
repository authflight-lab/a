"""Best-effort Telegram DM sender for the API process.

The API is a separate deployable from the bot and does NOT instantiate an
aiogram ``Bot``. When it needs to notify a user directly (e.g. a redemption was
accepted and is pending), it calls the Telegram Bot API ``sendMessage`` endpoint
over HTTPS using the same bot token it already holds for initData validation.

Every send is best-effort: a user who never started a chat with the bot, blocked
it, or any transient network error must never break the calling request. Callers
should still wrap ``send_dm`` in their own try/except and log failures.
"""

import html
import logging

import httpx

from .config import settings

logger = logging.getLogger("bt.api.notify")

_TELEGRAM_API = "https://api.telegram.org"


def esc(text: object) -> str:
    """HTML-escape a value for safe inclusion in a parse_mode=HTML message."""
    return html.escape(str(text if text is not None else ""))


def profile_link(tg_id: int, username: str | None, name: str | None) -> str:
    """HTML anchor to a user's Telegram profile.

    Prefers a public ``https://t.me/<username>`` link; falls back to a
    ``tg://user?id=`` deep link when the user has no username.
    """
    label = esc((name or "").strip() or (f"@{username}" if username else str(tg_id)))
    if username:
        href = f"https://t.me/{username.lstrip('@')}"
    else:
        href = f"tg://user?id={tg_id}"
    return f'<a href="{href}">{label}</a>'


async def _send(tg_id: int, payload: dict) -> bool:
    """POST a sendMessage payload. Returns True on success, never raises."""
    token = settings.bot_token
    if not token:
        logger.warning("bt_dm_no_token")
        return False
    url = f"{_TELEGRAM_API}/bot{token}/sendMessage"
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            resp = await client.post(url, json=payload)
        if resp.status_code == 200 and resp.json().get("ok"):
            return True
        logger.warning("bt_dm_failed", extra={"tg_id": tg_id, "status": resp.status_code})
        return False
    except Exception as e:
        logger.warning("bt_dm_error tg_id=%s error=%s", tg_id, e)
        return False


async def send_dm(tg_id: int, text: str) -> bool:
    """Send an HTML DM to ``tg_id``. Returns True on success, False otherwise.

    Never raises — swallows and logs all errors so the caller's request can
    proceed regardless of delivery outcome.
    """
    return await _send(tg_id, {
        "chat_id": tg_id,
        "text": text,
        "parse_mode": "HTML",
        "disable_web_page_preview": True,
    })


async def send_redemption_notification(
    log_chat: int,
    text: str,
    redemption_id: str,
) -> bool:
    """Send a redemption alert to the log chat with Fulfil / Reject buttons.

    The callback_data format ``redeem_fulfil:<uuid>`` / ``redeem_reject:<uuid>``
    is handled by the bot's callback query handler, so admins can act without
    ever typing or copy-pasting the UUID.
    """
    payload = {
        "chat_id": log_chat,
        "text": text,
        "parse_mode": "HTML",
        "disable_web_page_preview": True,
        "reply_markup": {
            "inline_keyboard": [[
                {"text": "✅ Fulfil", "callback_data": f"redeem_fulfil:{redemption_id}"},
                {"text": "❌ Reject", "callback_data": f"redeem_reject:{redemption_id}"},
            ]]
        },
    }
    return await _send(log_chat, payload)

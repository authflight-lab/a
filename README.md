# Bartender API (`/api/`)

Server-authoritative FastAPI service — **all** game math (RNG, multipliers,
win/loss, payout), the shop, and the leaderboard. Deploys to **Render** as a Web
Service. The bot never calls this API; they coordinate only through Supabase.

## Layout

```
api/
  config.py        # settings (config-in-code); env of same NAME overrides
  auth.py          # verify_init_data + require_user dependency (spec §8)
  db.py            # lazy httpx Supabase REST/RPC client (graceful when unset)
  main.py          # FastAPI app + all /bt/api/* endpoints (contract §4)
  game/
    seed.py        # generate_server_seed, server_hash, rng_float, rng_int (§6)
    dice.py flip.py mines.py towers.py highlow.py plinko.py   # engines (§7)
  tests/
    test_rtp.py    # RTP identity: Σ P·M = 1−ε ± 1e-9 for all 6 games
  requirements.txt
```

## Config

All config lives in `config.py` (placeholders now). Each field is overridable by
an environment variable of the **same name, upper-cased**:

| Field | Env var | Purpose |
|-------|---------|---------|
| `bot_token` | `BOT_TOKEN` | initData HMAC validation only (mirrors the bot) |
| `bt_supabase_url` | `BT_SUPABASE_URL` | Supabase project URL |
| `bt_supabase_service_key` | `BT_SUPABASE_SERVICE_KEY` | service-role key (server-only) |
| `bt_app_origin` | `BT_APP_ORIGIN` | the single CORS allowlist origin (no wildcard) |

When Supabase is unconfigured the app still imports and starts; endpoints that
need the DB return `503 {"error":"supabase_not_configured"}`.

## Run locally

```bash
pip install -r api/requirements.txt
# from the repo root:
uvicorn api.main:app --host 0.0.0.0 --port 8000
```

## Tests

```bash
python -m pytest api/tests -q
```

The RTP identity test must pass before deploy (spec §6/§14).

## Deploy to Render

- **Type:** Web Service (Starter — no spin-down).
- **Root Directory:** repo root (so `api` is importable as a package).
- **Build Command:** `pip install -r api/requirements.txt`
- **Start Command:** `uvicorn api.main:app --host 0.0.0.0 --port $PORT`
- **Environment:** set `BOT_TOKEN`, `BT_SUPABASE_URL`, `BT_SUPABASE_SERVICE_KEY`,
  `BT_APP_ORIGIN`. Keep `BOT_TOKEN` + Supabase keys in sync with the bot on
  rotation (spec §13).

## Security notes (spec §14)

- `tg_id` is derived **only** from validated `initData` — never from the body/query.
- All balance changes go through the `bt_apply_ledger` RPC — no raw `UPDATE balance`.
- CORS is an explicit allowlist (`BT_APP_ORIGIN`) — no `*`.
- The service-role key never leaves the server; it is not part of `/app/`.

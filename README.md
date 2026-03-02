# Numatix — Options Strategy Dashboard & Bot

Options trading dashboard and automated strategy runner. The app is protected by **username + password + 2FA (TOTP)**. The trading bot runs as separate processes started from the dashboard.

---

## Table of contents

- [How the project works](#how-the-project-works)
- [Rules and capabilities](#rules-and-capabilities)
- [Adding a user (auth)](#adding-a-user-auth)
- [Running the app](#running-the-app)
- [Config and accounts](#config-and-accounts)
- [Customization and automation](#customization-and-automation)

---

## How the project works

### High-level flow

1. **Dashboard (Streamlit)**  
   You run `streamlit run app.py`. You log in with **username**, **password**, and **6-digit 2FA code**. After that you see the Options Strategy Dashboard (tabs: Config Editor, Positions Viewer, Historical Data) and can start/stop the bot.

2. **Trading bot (main.py)**  
   From the dashboard you choose an **account** (nickname), **symbol** (e.g. SPX, XSP), and **strike range**. Clicking “Start Strategy” launches a **separate process** that runs `main.py` with those arguments. That process connects to IBKR, runs the strategy logic, and writes positions to a per-account, per-symbol SQLite DB.

3. **Auth**  
   Stored in `auth.json`. Passwords are **bcrypt-hashed**; you never type or paste the hash. You only use the **plain password** you set when adding the user. 2FA uses **TOTP** (e.g. Google Authenticator) with a secret stored per user in `auth.json`.

4. **State and data**  
   - **strategy_state.json** — Pause/stop flags per account+symbol (used by the dashboard and the bot).  
   - **bot.{account}.{symbol}.pid** / **bot.{account}.{symbol}.status** — Process tracking.  
   - **positions_{account}_{symbol}.db** — SQLite DB of positions for that account+symbol.  
   - **config.json** / **config_{nickname}.json** — Broker, underlying, trade parameters, time windows, etc.

### Main components

| Component | Role |
|-----------|------|
| **app.py** | Entry point. Login (username + password + 2FA), then loads the dashboard. **Always run this**, not `frontend.py` directly. |
| **frontend.py** | Dashboard UI: run config, strike range, start/stop bot, config editor, positions, historical data. Protected by `st.session_state["authenticated"]`. |
| **auth.py** | Loads `auth.json`, verifies password (bcrypt) and TOTP (pyotp). Used by `app.py`. |
| **main.py** | Bot process: parses `--account`, `--symbol`, `--config`, `--range`, `--exclude`; runs `StrategyManager` and per-offset `Strategy` threads. |
| **strategy/strategy.py** | Single-strike strategy logic (entry/exit, VWAP, take-profit, stop-loss, hedging). |
| **broker/strategy_broker.py** | Wrapper around IBKR; shared by all strategy threads. |
| **db/position_db.py** | Per-account, per-symbol SQLite position storage. |
| **db/multi_account_db.py** | Queries across all `positions_*.db` files for the dashboard. |
| **helpers/state_manager.py** | Read/write pause/stop state in `strategy_state.json`. |

---

## Rules and capabilities

### Authentication

- **Login:** Username + password (plain text at login) + 6-digit TOTP code.  
- **Password:** Stored in `auth.json` as a **bcrypt hash**. You must never put a plain password in `auth.json`; use `setup_auth.py` to add/update users (it hashes for you).  
- **2FA:** TOTP only. Each user has a `totp_secret` in `auth.json`; users add it to an authenticator app.  
- **Session:** `st.session_state["authenticated"]` is set to `True` **only after** successful OTP verification.  
- **Protection:** Max **5 OTP attempts** per login flow; after that the session is locked (user must refresh and try again).  
- **Dashboard and pages:** If not authenticated, the main dashboard and the Premium VWAP Graph page show a message and stop; they do not show data.

### Bot and dashboard

- One **bot process** per **(account, symbol)**. Starting the same account+symbol again while it’s running is blocked.  
- **Account** in the UI is a **nickname** from `accounts.json`; it is resolved to `ibkr_account_id` for the broker.  
- **Config** is per-profile: `config_{nickname}.json`. The dashboard uses the selected account’s config when starting the bot.  
- **Strike range** is a list of offsets from ATM (e.g. 4–10 → offsets 4,5,…,10). Optional **exclude** list can drop specific offsets.

### Data and state

- **Positions** live in `positions_{account}_{symbol}.db`. The dashboard uses `MultiAccountDB` to aggregate across DBs.  
- **Pause/Stop** are stored in `strategy_state.json` under keys `{account}_{symbol}`. The bot checks this file periodically.

---

## Adding a user (auth)

You do **not** edit `auth.json` by hand for passwords (they must be bcrypt-hashed). Use the setup script once per user.

### Step 1: Run the setup script

From the project root:

```bash
python setup_auth.py
```

You will be prompted for:

1. **Username** — e.g. `vedansh`, `aryaman`. This is what they type on the login screen.  
2. **Password** — their actual password (typed once; the script hashes it and stores the hash).  
3. **TOTP secret** — either paste an existing base32 secret, or press **Enter** to have one generated (the script prints it).

### Step 2: Configure the authenticator app

- If you **generated** a secret: add it to Google Authenticator / Authy / etc. (manual entry, “enter key”).  
- If you **pasted** a secret: ensure the same secret is in the user’s authenticator app.

### Step 3: Log in

User goes to the app, enters **username**, **password** (the one you set in the script), then the **6-digit code** from the app. No hash is ever typed in the UI.

### auth.json format (reference only — use setup_auth.py to populate)

```json
{
  "users": {
    "aryaman": {
      "password": "$2b$12$...",
      "totp_secret": "JBSWY3DPEHPK3PXP"
    }
  }
}
```

- **Never** put a plain-text password in `"password"`.  
- To **add** another user or **change** a password: run `python setup_auth.py` again; it merges into existing `auth.json`.

---

## Running the app

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Start the dashboard (with login)

```bash
streamlit run app.py
```

- Open the URL (e.g. http://localhost:8501).  
- Log in with username, password, and 2FA code.  
- Use the dashboard to edit config, view positions/history, and start/stop the bot.

### 3. Bot process (started from the dashboard)

The dashboard runs `main.py` in a subprocess with arguments like:

- `--account` — nickname from `accounts.json`  
- `--symbol` — e.g. SPX, XSP  
- `--config` — path to `config_{nickname}.json`  
- `--range` — e.g. `4-10` (strike offsets)  
- `--exclude` — optional, e.g. `-1,5,7`

You can also run the bot manually for testing:

```bash
python main.py --account Vedansh --symbol SPX --config config_Vedansh.json --range=4-10
```

---

## Config and accounts

### accounts.json

Maps **nicknames** (used in the UI and for config/file names) to **IBKR account IDs**:

```json
{
  "accounts": [
    { "nickname": "Vedansh", "ibkr_account_id": "DUH300582" }
  ]
}
```

- Add a new row to add a new “account” in the dashboard.  
- Each nickname typically has its own `config_{nickname}.json` (create from Config Editor or copy from `config.json`).

### config.json / config_{nickname}.json

- **broker** — host, port, client_id (TWS/Gateway).  
- **underlying** — symbol, exchange, currency, trading_class, multiplier.  
- **expiry** — option expiry date.  
- **trade_parameters** — quantities, VWAP multipliers, take-profit levels, stop-loss, strike_step, drawdown_limit, profit_limit, etc.  
- **time_controls** — entry_start, entry_end, force_exit_time, timezone.  
- **hedging** — enable_hedges, offsets, quantity.  
- **paused** / **stopped** — legacy flags; runtime control is via `strategy_state.json`.

Editing is done in the dashboard’s **Config Editor** tab (per-profile).

---

## Customization and automation

### Adding or changing a user

- **Add user:** Run `python setup_auth.py`, enter username, password, and optionally TOTP secret (or generate).  
- **Change password:** Run `python setup_auth.py` again with the same username; it overwrites that user’s entry in `auth.json`.  
- **Change TOTP:** Edit `auth.json` and replace that user’s `totp_secret`, or run a small script that updates only `totp_secret` (keep password hash unchanged).

### Adding a new dashboard page

- Add a file under `pages/`, e.g. `pages/3_My_Page.py`.  
- At the top of the page’s main logic, enforce auth:

  ```python
  if not st.session_state.get("authenticated", False):
      st.warning("Please log in via the main app to access this page.")
      st.stop()
  ```

- Streamlit will show it in the sidebar. Session state is shared with `app.py`, so login state applies.

### Changing or extending the strategy

- **Entry/exit logic:** Edit `strategy/strategy.py` (e.g. `run()`, order placement, VWAP logic, take-profit/stop-loss).  
- **New parameters:** Add fields to the config JSON and read them in `Strategy.load_config()`, then use them in your logic.  
- **New strategy “flavor”:** You can add a new class in `strategy/` (e.g. `strategy_alt.py`) and, in `main.py`, choose between `Strategy` and the new class when constructing instances (same manager/broker/account/config pattern).

### Changing broker or API

- **IBKR connection and orders:** Implement or modify adapters in `broker/` (e.g. `ib_broker.py`, `strategy_broker.py`).  
- **StrategyBroker** is the single point used by all strategy threads; keep it thread-safe and use it for any new strategy class.

### Automating bot start/stop (e.g. cron or systemd)

- Start: run `main.py` with the desired `--account`, `--symbol`, `--config`, `--range` (and optional `--exclude`). Use the same working directory as the project so paths to config and DBs are correct.  
- Stop: either kill the process (PID in `bot.{account}.{symbol}.pid`) or set that account+symbol to stopped in `strategy_state.json` (the bot checks every few seconds and exits).  
- For multiple account+symbol pairs, start one process per pair (same as the dashboard does).

### Adding a new config profile or account

- **Account:** Add an entry to `accounts.json` with `nickname` and `ibkr_account_id`.  
- **Config:** Create `config_{nickname}.json` (copy from `config.json` and edit, or use the dashboard Config Editor and “Initialize from config.json” for that profile).

### Database and positions

- **Position schema:** Defined in `db/position_db.py` (`PositionDB`). Migrations are done with `ALTER TABLE` in `_init_db()`. For new columns, add a similar `try/except` block.  
- **Cross-account queries:** Use `MultiAccountDB` (already used by the dashboard). Filter by `account` and/or `symbol` if needed.

### Auth and security

- **Stricter rate limits:** In `app.py`, `_render_otp_step(max_attempts=5)` and the attempt counter can be adjusted; you can also add rate limiting per username (e.g. store failed attempts in a file or cache).  
- **Password policy:** Enforced only in `setup_auth.py`; you can add length/complexity checks there before hashing.  
- **TOTP window:** In `auth.py`, `verify_totp(..., valid_window=1)` allows one step before/after; change `valid_window` if you need different tolerance.

---

## File layout (summary)

```
├── app.py                 # Entry: login + dashboard
├── frontend.py             # Dashboard UI (do not run directly)
├── main.py                 # Bot process entry
├── auth.py                 # Auth logic (auth.json, bcrypt, TOTP)
├── setup_auth.py           # One-time user setup (run to add users)
├── auth.json               # Users, hashed passwords, TOTP secrets
├── accounts.json           # Nickname → ibkr_account_id
├── config.json             # Default config
├── config_*.json           # Per-profile configs
├── strategy_state.json     # Pause/stop per account+symbol
├── positions_*.db          # SQLite per account+symbol
├── bot.*.pid / bot.*.status
├── broker/                 # IBKR and strategy broker
├── db/                     # Position DB, multi-account wrapper
├── helpers/                # State, positions, graph, etc.
├── strategy/               # Strategy logic
└── pages/                  # Streamlit subpages (e.g. Premium VWAP Graph)
```

Use this README as the single place for **rules**, **how the project works**, **how to add users**, and **what to change for automation or new classes**.

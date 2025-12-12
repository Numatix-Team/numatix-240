import json
import os
import threading
import random
import time
from datetime import datetime
import pytz
import pandas as pd

# =========================================================
# THREAD-SAFE JSON FILE HANDLING
# =========================================================

json_lock = threading.Lock()

def load_positions_file(path="positions.json"):
    """Read positions.json safely."""
    if not os.path.exists(path):
        return {"positions": [], "active_positions": {"position_open": False}}

    try:
        with open(path, "r") as f:
            return json.load(f)
    except:
        return {"positions": [], "active_positions": {"position_open": False}}


def save_positions_file(data, path="positions.json"):
    """Atomic write to positions.json."""
    with json_lock:
        tmp = f"{path}.tmp"
        with open(tmp, "w") as f:
            json.dump(data, f, indent=4)
        os.replace(tmp, path)


# =========================================================
# TIME HELPERS
# =========================================================

def parse_time_string(s):
    """Convert 'HH:MM' into time object."""
    from datetime import time
    h, m = s.split(":")
    return time(int(h), int(m))


def parse_ib_datetime(ts, target_tz):
    """Convert IBKR (Central) → Eastern Time."""
    try:
        date_part, time_part = ts.split(" ")
        naive = datetime.strptime(date_part + " " + time_part, "%Y%m%d %H:%M:%S")
        central = pytz.timezone("US/Central").localize(naive)
        return central.astimezone(target_tz)
    except:
        return pd.to_datetime(ts)


# =========================================================
# POSITION HELPERS
# =========================================================

def generate_position_id(symbol, strike, right, tz):
    time_str = datetime.now(tz).strftime("%Y-%m-%dT%H-%M-%S")
    epoch = int(time.time())
    return f"{epoch}_{symbol}_{strike}{right}_{time_str}"


def create_position_entry(
    symbol, expiry, strike, right, side, qty,
    entry_price, order_id, position_type, tz, bid=0, ask=0
):
    data = load_positions_file()
    pos_id = generate_position_id(symbol, strike, right, tz)
    now = datetime.now(tz).isoformat()

    pos = {
        "id": pos_id,
        "symbol": symbol,
        "expiry": expiry,
        "right": right,
        "position_type": position_type,
        "side": side,

        "strike": strike,
        "qty": qty,

        "active": True,
        "entry_time": now,
        "exit_time": None,

        "entry_price": entry_price,
        "bid": bid,
        "ask": ask,
        "last_price": entry_price,
        "close_price": None,
        "pnl_pct": 0.0,

        "order_id_entry": order_id,
        "order_id_exit": None,

        "last_update": now
    }

    data.setdefault("positions", []).append(pos)
    save_positions_file(data)
    return pos_id


def get_position_by_id(pos_id):
    if pos_id is None:
        return None
    data = load_positions_file()
    for pos in data.get("positions", []):
        if pos.get("id") == pos_id:
            return pos
    return None


def simulated_bid_ask(base_price):
    """Used only for _update_live_leg simulation."""
    bid = base_price + random.uniform(-2, 2)
    ask = bid + random.uniform(0.1, 0.5)
    return bid, ask


# =========================================================
# ACTIVE POSITION ID MANAGEMENT
# =========================================================

def load_active_ids():
    data = load_positions_file()
    return data.get("active_positions", {})


def save_active_ids(position_open, atm_call_id, atm_put_id, otm_call_id, otm_put_id):
    data = load_positions_file()
    data["active_positions"] = {
        "position_open": position_open,
        "atm_call_id": atm_call_id,
        "atm_put_id": atm_put_id,
        "otm_call_id": otm_call_id,
        "otm_put_id": otm_put_id
    }
    save_positions_file(data)

def update_position_in_json(updated_pos):
    if updated_pos is None:
        return

    data = load_positions_file()
    positions = data.get("positions", [])

    updated_id = str(updated_pos["id"])
    replaced = False

    # ---------------------------------------
    # Replace matching position by ID
    # ---------------------------------------
    for i, pos in enumerate(positions):
        if str(pos["id"]) == updated_id:
            positions[i] = updated_pos
            replaced = True
            break

    if not replaced:
        positions.append(updated_pos)

    data["positions"] = positions

    # ---------------------------------------
    # Compute TOTAL REALIZED + UNREALIZED PnL
    # ---------------------------------------
    total_realized = 0.0
    total_unrealized = 0.0

    for pos in positions:
        pnl_val = pos.get("pnl_value", 0) or 0

        if pos.get("active"):
            total_unrealized += pnl_val
        else:
            total_realized += pnl_val

    total_combined = total_realized + total_unrealized

    # ---------------------------------------
    # Store totals in JSON
    # ---------------------------------------
    data["summary"] = {
        "total_realized_pnl": round(total_realized, 2),
        "total_unrealized_pnl": round(total_unrealized, 2),
        "total_combined_pnl": round(total_combined, 2),
        "last_update": datetime.now().isoformat()
    }

    save_positions_file(data)


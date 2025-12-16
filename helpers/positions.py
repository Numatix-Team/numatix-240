import json
import os
import threading
import random
import time
from datetime import datetime
import pytz
import pandas as pd
from db.position_db import PositionDB

# =========================================================
# DATABASE INSTANCE
# =========================================================

_db = PositionDB()


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
    account, symbol, expiry, strike, right, side, qty,
    entry_price, order_id, position_type, tz, bid=0, ask=0
):
    pos_id = generate_position_id(symbol, strike, right, tz)
    now = datetime.now(tz).isoformat()

    pos = {
        "id": pos_id,
        "account": account,
        "symbol": symbol,
        "expiry": expiry,
        "right": right,
        "position_type": position_type,
        "side": side,
        "strike": strike,
        "initial_qty": qty,
        "qty": qty,
        "active": True,
        "entry_time": now,
        "exit_time": None,
        "entry_price": entry_price,
        "close_price": None,
        "bid": bid,
        "ask": ask,
        "last_price": entry_price,
        "order_id_entry": order_id,
        "order_id_exit": None,
        "last_update": now,
        "realized_pnl": 0.0,
        "unrealized_pnl": 0.0
    }

    _db.insert_position(pos)
    return pos_id


def get_position_by_id(pos_id):
    if pos_id is None:
        return None
    return _db.get_position(pos_id)


def simulated_bid_ask(base_price):
    """Used only for _update_live_leg simulation."""
    bid = base_price + random.uniform(-2, 2)
    ask = bid + random.uniform(0.1, 0.5)
    return bid, ask


# =========================================================
# ACTIVE POSITION ID MANAGEMENT
# =========================================================

def load_active_ids(account=None):
    """Load active position IDs. Uses database active field."""
    return _db.get_active_position_ids(account)


def save_active_ids(position_open, atm_call_id, atm_put_id, otm_call_id, otm_put_id, account=None):
    """Save active position IDs by updating active field in database."""
    # Set all positions to inactive if position_open is False
    if not position_open:
        active_positions = _db.get_active_positions(account)
        for pos in active_positions:
            _db.update_position(pos["id"], {"active": 0})
        return
    
    # Update active status for specific positions
    if atm_call_id:
        _db.update_position(atm_call_id, {"active": 1})
    if atm_put_id:
        _db.update_position(atm_put_id, {"active": 1})
    if otm_call_id:
        _db.update_position(otm_call_id, {"active": 1})
    if otm_put_id:
        _db.update_position(otm_put_id, {"active": 1})

def update_position_in_json(updated_pos):
    """Update position in database. Name kept for backward compatibility."""
    if updated_pos is None:
        return
    
    # Remove pnl_pct if present (we don't store it anymore)
    if "pnl_pct" in updated_pos:
        del updated_pos["pnl_pct"]
    
    pos_id = updated_pos.get("id")
    if not pos_id:
        return
    
    # Check if position exists
    existing = _db.get_position(pos_id)
    if existing:
        # Update existing position
        _db.update_position(pos_id, updated_pos)
    else:
        # Insert new position
        _db.insert_position(updated_pos)


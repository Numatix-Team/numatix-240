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

# Global database instance - will be initialized per account+symbol
_db_cache = {}
_db_lock = threading.Lock()

def get_db(account, symbol):
    """Get or create database instance for account (nickname) and symbol. File: positions_{account}_{symbol}.db"""
    key = f"{account}_{symbol}"
    with _db_lock:
        if key not in _db_cache:
            _db_cache[key] = PositionDB(account=account, symbol=symbol)
        return _db_cache[key]


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

def generate_position_id(symbol, strike, right, tz, position_type=None, strike_offset=None):
    """Generate a unique position ID.
    
    For hedges (OTM positions), includes strike_offset to ensure uniqueness across strategies.
    For ATM positions, includes microseconds for sub-second precision.
    """
    now = datetime.now(tz)
    time_str = now.strftime("%Y-%m-%dT%H-%M-%S")
    epoch = int(time.time())
    
    # For hedges (OTM), include strike_offset to differentiate between strategies
    # that share the same hedge strike but have different offsets
    if position_type == "OTM" and strike_offset is not None:
        return f"{epoch}_{symbol}_{strike}{right}_offset{strike_offset}_{time_str}"
    else:
        # For ATM positions, use microseconds for sub-second precision
        microseconds = now.microsecond
        return f"{epoch}_{microseconds}_{symbol}_{strike}{right}_{time_str}"


def create_position_entry(
    account, symbol, expiry, strike, right, side, qty,
    entry_price, order_id, position_type, tz, bid=0, ask=0, strike_offset=None
):
    pos_id = generate_position_id(symbol, strike, right, tz, position_type=position_type, strike_offset=strike_offset)
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
        "order_id_entry": int(order_id) if order_id is not None else None,
        "order_id_exit": None,
        "last_update": now,
        "realized_pnl": 0.0,
        "unrealized_pnl": 0.0
    }

    # Use account+symbol specific database
    db = get_db(account, symbol)
    db.insert_position(pos)
    return pos_id


def get_position_by_id(pos_id, account=None, symbol=None):
    """Get position by ID from the database for the given account (nickname) and symbol."""
    if pos_id is None or not account or not symbol:
        return None
    db = get_db(account, symbol)
    return db.get_position(pos_id)


def simulated_bid_ask(base_price):
    """Used only for _update_live_leg simulation."""
    bid = base_price + random.uniform(-2, 2)
    ask = bid + random.uniform(0.1, 0.5)
    return bid, ask


# =========================================================
# ACTIVE POSITION ID MANAGEMENT
# =========================================================

def load_active_ids(account=None, symbol=None):
    """Load active position IDs from the database for the given account (nickname) and symbol."""
    if not account or not symbol:
        return {
            "position_open": False,
            "atm_call_id": None,
            "atm_put_id": None,
            "otm_call_id": None,
            "otm_put_id": None,
        }
    db = get_db(account, symbol)
    return db.get_active_position_ids(account=None, symbol=None)


def save_active_ids(position_open, atm_call_id, atm_put_id, otm_call_id, otm_put_id, account=None, symbol=None):
    """Save active position IDs by updating active field in database based on quantity.
    
    IMPORTANT: Only updates the specific position IDs passed in. Does NOT touch other positions.
    This is critical for multi-threaded scenarios where multiple strategies run simultaneously.
    """
    if not account or not symbol:
        raise ValueError("account and symbol are required for save_active_ids")
    
    db = get_db(account, symbol)
    
    # Update active status for specific positions based on their quantity
    # Active should be True if qty > 0, False if qty <= 0
    # We only update the positions that are passed in, never touch other positions
    def update_active_if_needed(pos_id):
        if pos_id:
            pos = get_position_by_id(pos_id, account, symbol)
            if pos:
                # IBKR reports SELL/short as negative; treat as active if |qty| > 0
                qty = pos.get("qty", 0)
                try:
                    qty_magnitude = abs(int(float(qty)))
                except (TypeError, ValueError):
                    qty_magnitude = 0
                should_be_active = (qty_magnitude > 0)
                current_active = pos.get("active", False)
                # Only update if it's incorrect
                if current_active != should_be_active:
                    db.update_position(pos_id, {"active": 1 if should_be_active else 0})
                    print(f"[SAVE_ACTIVE] Updated {pos_id[:20]}... active={should_be_active} (qty={qty})")
    
    # Always update based on quantity, regardless of position_open flag
    # If position_open is False, we still check each position's quantity
    update_active_if_needed(atm_call_id)
    update_active_if_needed(atm_put_id)
    update_active_if_needed(otm_call_id)
    update_active_if_needed(otm_put_id)

def update_position_in_db(updated_pos):
    """Update position in database."""
    if updated_pos is None:
        return
    
    # Remove pnl_pct if present (we don't store it anymore)
    if "pnl_pct" in updated_pos:
        del updated_pos["pnl_pct"]
    
    pos_id = updated_pos.get("id")
    if not pos_id:
        return
    
    # Get account and symbol from position
    account = updated_pos.get("account")
    symbol = updated_pos.get("symbol")
    pos_type = updated_pos.get("position_type", "UNKNOWN")
    right = updated_pos.get("right", "")
    strike = updated_pos.get("strike", 0)
    
    if not account or not symbol:
        return
    db = get_db(account, symbol)
    existing = db.get_position(pos_id)
    if existing:
        last_price = updated_pos.get("last_price")
        unrealized = updated_pos.get("unrealized_pnl", 0) or 0
        realized = updated_pos.get("realized_pnl", 0) or 0
        qty = updated_pos.get("qty", 0)
        last_price_str = f"${last_price:.2f}" if last_price is not None else "N/A"
        print(f"[DB] Updating {pos_type} {right} @ Strike {strike} in {account}/{symbol} DB:")
        print(f"[DB]   Last Price: {last_price_str}, Qty: {qty}")
        print(f"[DB]   Realized PnL: ${realized:.2f}, Unrealized PnL: ${unrealized:.2f}")
        db.update_position(pos_id, updated_pos)
        print(f"[DB] Update complete")
    else:
        print(f"[DB] Inserting new {pos_type} {right} @ Strike {strike} in {account}/{symbol} DB")
        db.insert_position(updated_pos)
        print(f"[DB] Insert complete")


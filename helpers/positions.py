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
    """Get or create database instance for account+symbol combination."""
    key = f"{account}_{symbol}"
    
    with _db_lock:
        if key not in _db_cache:
            _db_cache[key] = PositionDB(account=account, symbol=symbol)
        return _db_cache[key]

# For backward compatibility - will be replaced when account/symbol are known
_db = None


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

    # Use account+symbol specific database
    db = get_db(account, symbol)
    db.insert_position(pos)
    return pos_id


def get_position_by_id(pos_id, account=None, symbol=None):
    """Get position by ID. If account and symbol are provided, uses specific DB, otherwise searches all."""
    if pos_id is None:
        return None
    
    # If account and symbol are known, use specific database
    if account and symbol:
        db = get_db(account, symbol)
        return db.get_position(pos_id)
    
    # Otherwise, search across all databases (for backward compatibility)
    # This is less efficient but maintains compatibility
    import glob
    import os
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    pattern = os.path.join(base_dir, "positions_*.db")
    
    for db_file in glob.glob(pattern):
        db = PositionDB(db_path=db_file)
        pos = db.get_position(pos_id)
        if pos:
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

def load_active_ids(account=None, symbol=None):
    """Load active position IDs. Uses database active field."""
    if account and symbol:
        db = get_db(account, symbol)
        return db.get_active_position_ids(account=None, symbol=None)  # No filter needed, DB is already filtered
    else:
        # For backward compatibility, search all databases
        from db.multi_account_db import MultiAccountDB
        multi_db = MultiAccountDB()
        active_positions = multi_db.get_active_positions(account, symbol)
        
        # Build active IDs dict
        atm_call_id = None
        atm_put_id = None
        otm_call_id = None
        otm_put_id = None
        
        for pos in active_positions:
            pos_type = pos.get("position_type", "")
            if pos_type == "ATM":
                if pos.get("right") == "C":
                    atm_call_id = pos["id"]
                elif pos.get("right") == "P":
                    atm_put_id = pos["id"]
            elif pos_type == "OTM":
                if pos.get("right") == "C":
                    otm_call_id = pos["id"]
                elif pos.get("right") == "P":
                    otm_put_id = pos["id"]
        
        return {
            "position_open": len(active_positions) > 0,
            "atm_call_id": atm_call_id,
            "atm_put_id": atm_put_id,
            "otm_call_id": otm_call_id,
            "otm_put_id": otm_put_id
        }


def save_active_ids(position_open, atm_call_id, atm_put_id, otm_call_id, otm_put_id, account=None, symbol=None):
    """Save active position IDs by updating active field in database."""
    if not account or not symbol:
        raise ValueError("account and symbol are required for save_active_ids")
    
    db = get_db(account, symbol)
    
    # Set all positions to inactive if position_open is False
    if not position_open:
        active_positions = db.get_active_positions(account=None, symbol=None)
        for pos in active_positions:
            db.update_position(pos["id"], {"active": 0})
        return
    
    # Update active status for specific positions
    if atm_call_id:
        db.update_position(atm_call_id, {"active": 1})
    if atm_put_id:
        db.update_position(atm_put_id, {"active": 1})
    if otm_call_id:
        db.update_position(otm_call_id, {"active": 1})
    if otm_put_id:
        db.update_position(otm_put_id, {"active": 1})

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
    
    # Get account and symbol from position
    account = updated_pos.get("account")
    symbol = updated_pos.get("symbol")
    
    if account and symbol:
        # Use account+symbol specific database
        db = get_db(account, symbol)
        existing = db.get_position(pos_id)
        if existing:
            db.update_position(pos_id, updated_pos)
        else:
            db.insert_position(updated_pos)
    else:
        # Fallback: search all databases (less efficient)
        import glob
        import os
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        pattern = os.path.join(base_dir, "positions_*.db")
        
        found = False
        for db_file in glob.glob(pattern):
            db = PositionDB(db_path=db_file)
            existing = db.get_position(pos_id)
            if existing:
                db.update_position(pos_id, updated_pos)
                found = True
                break
        
        if not found:
            # If not found and we have account/symbol, create in the right DB
            if account and symbol:
                db = get_db(account, symbol)
                db.insert_position(updated_pos)


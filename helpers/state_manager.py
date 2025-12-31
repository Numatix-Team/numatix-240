"""
State manager for per-account pause/stop control
"""
import json
import os
import threading
from pathlib import Path

STATE_FILE = "strategy_state.json"
_lock = threading.Lock()

def get_state_file_path():
    """Get the full path to the state file"""
    return os.path.join(os.path.dirname(os.path.dirname(__file__)), STATE_FILE)

def load_state():
    """Load state from file"""
    state_file = get_state_file_path()
    if not os.path.exists(state_file):
        return {}
    
    with _lock:
        try:
            with open(state_file, "r") as f:
                return json.load(f)
        except:
            return {}

def save_state(state):
    """Save state to file"""
    state_file = get_state_file_path()
    with _lock:
        try:
            with open(state_file, "w") as f:
                json.dump(state, f, indent=2)
        except Exception as e:
            print(f"[StateManager] Error saving state: {e}")

def _get_key(account, symbol=None):
    """Generate state key from account and symbol"""
    if symbol:
        return f"{account}_{symbol}"
    return account

def get_account_state(account, symbol=None):
    """Get state for a specific account+symbol combination"""
    state = load_state()
    key = _get_key(account, symbol)
    return state.get(key, {
        "paused": False,
        "stopped": False
    })

def set_account_paused(account, paused=True, symbol=None):
    """Set pause state for an account+symbol combination"""
    state = load_state()
    key = _get_key(account, symbol)
    if key not in state:
        state[key] = {"paused": False, "stopped": False}
    state[key]["paused"] = paused
    save_state(state)

def set_account_stopped(account, stopped=True, symbol=None):
    """Set stop state for an account+symbol combination"""
    state = load_state()
    key = _get_key(account, symbol)
    if key not in state:
        state[key] = {"paused": False, "stopped": False}
    state[key]["stopped"] = stopped
    if stopped:
        state[key]["paused"] = False  # Can't be paused if stopped
    save_state(state)

def is_account_paused(account, symbol=None):
    """Check if account+symbol is paused"""
    account_state = get_account_state(account, symbol)
    return account_state.get("paused", False)

def is_account_stopped(account, symbol=None):
    """Check if account+symbol is stopped"""
    account_state = get_account_state(account, symbol)
    return account_state.get("stopped", False)

def clear_account_state(account, symbol=None):
    """Clear state for an account+symbol (when process exits)"""
    state = load_state()
    key = _get_key(account, symbol)
    if key in state:
        del state[key]
        save_state(state)


import streamlit as st
import json
import os
import pandas as pd
from streamlit_autorefresh import st_autorefresh
import subprocess
import sys
from datetime import datetime
import pytz
import time
from db.position_db import PositionDB
from db.multi_account_db import MultiAccountDB
from helpers.state_manager import (
    set_account_paused, set_account_stopped, 
    is_account_paused, is_account_stopped,
    get_account_state
)
python_path = sys.executable

# Database instance - use MultiAccountDB to query across all account+symbol databases
_db = MultiAccountDB()

# Eastern timezone
EASTERN_TZ = pytz.timezone("US/Eastern")

CONFIG_PATH = "config.json"
POSITIONS_PATH = "positions.json"


def get_config_path_for_nickname(nickname):
    """Return config file path for a profile: config_{nickname}.json (safe filename)."""
    safe = "".join(c if c.isalnum() or c in "._-" else "_" for c in (nickname or "default"))
    return f"config_{safe}.json"


# Account+symbol-specific PID/status file paths
def get_pid_path(account, symbol=None):
    if symbol:
        return f"bot.{account}.{symbol}.pid"
    return f"bot.{account}.pid"

def get_status_path(account, symbol=None):
    if symbol:
        return f"bot.{account}.{symbol}.status"
    return f"bot.{account}.status"

# -------------------------------------
# Load JSON helpers
# -------------------------------------
def save_pid(pid, account, symbol=None):
    pid_path = get_pid_path(account, symbol)
    with open(pid_path, "w") as f:
        f.write(str(pid))

def load_pid(account, symbol=None):
    pid_path = get_pid_path(account, symbol)
    if not os.path.exists(pid_path):
        return None
    try:
        return int(open(pid_path).read().strip())
    except:
        return None

def get_all_running_accounts():
    """Get list of all account+symbol combinations that have running processes by scanning for PID files"""
    running_accounts = []
    
    # Scan for all bot.*.pid files (supports both bot.account.pid and bot.account.symbol.pid)
    base_dir = os.path.dirname(os.path.abspath(__file__))
    for filename in os.listdir(base_dir):
        if filename.startswith("bot.") and filename.endswith(".pid"):
            # Extract account (nickname) and symbol from bot.{account}.{symbol}.pid
            parts = filename[4:-4].split(".")  # Remove "bot." prefix and ".pid" suffix
            account = parts[0]
            symbol = parts[1] if len(parts) > 1 else None
            
            pid = load_pid(account, symbol)
            if pid and pid_running(pid, account, symbol):
                running_accounts.append((account, symbol))
    
    return running_accounts

def pid_running(pid, account, symbol=None):
    """Check if process is running by PID and status file for specific account+symbol"""
    if pid is None:
        return False
    
    # First check if status file exists and is recent (within 30 seconds)
    status_file = get_status_path(account, symbol)
    if os.path.exists(status_file):
        try:
            with open(status_file, "r") as f:
                content = f.read().strip()
                if content.startswith("running|"):
                    # Check if status was updated recently (within 30 seconds)
                    parts = content.split("|")
                    status_time_str = parts[1]
                    status_time = datetime.fromisoformat(status_time_str)
                    time_diff = (datetime.now() - status_time.replace(tzinfo=None)).total_seconds()
                    if time_diff < 30:  # Status updated within last 30 seconds
                        return True
        except:
            pass
    
    # Fallback: Check if process exists
    try:
        if os.name == 'nt':  # Windows
            # Use tasklist to check if process exists
            result = subprocess.run(
                ['tasklist', '/FI', f'PID eq {pid}'],
                capture_output=True,
                text=True,
                timeout=2
            )
            return str(pid) in result.stdout
        else:
            # Unix/Linux
            os.kill(pid, 0)
            return True
    except (OSError, subprocess.TimeoutExpired, subprocess.SubprocessError):
        return False

def sync_flags_with_pid():
    """Ensure config.json matches the actual process state for all accounts."""
    # Get all accounts by scanning for PID files
    base_dir = os.path.dirname(os.path.abspath(__file__))
    all_accounts = []
    
    # Find all account+symbol combinations with PID files
    try:
        for filename in os.listdir(base_dir):
            if filename.startswith("bot.") and filename.endswith(".pid"):
                parts = filename[4:-4].split(".")
                if len(parts) == 1:
                    all_accounts.append((parts[0], None))
                else:
                    all_accounts.append((parts[0], parts[1]))
    except:
        pass
    
    # Clean up stale PID files for accounts that are not running
    for acc, sym in all_accounts:
        pid = load_pid(acc, sym)
        if pid and not pid_running(pid, acc, sym):
            # Process is dead, clean up files
            pid_path = get_pid_path(acc, sym)
            status_path = get_status_path(acc, sym)
            try:
                if os.path.exists(pid_path):
                    os.remove(pid_path)
                if os.path.exists(status_path):
                    os.remove(status_path)
            except:
                pass


def update_flags(paused=None, stopped=None):
    config = load_json(CONFIG_PATH)
    if config is None:
        return

    if paused is not None:
        config["paused"] = paused

    if stopped is not None:
        config["stopped"] = stopped

    save_json(CONFIG_PATH, config)



def load_json(path):
    if not os.path.exists(path):
        st.error(f"{path} not found!")
        return None

    with open(path, "r") as f:
        return json.load(f)


def load_accounts():
    """Load accounts from accounts.json. Returns list of {nickname, ibkr_account_id}. Used for display (nickname) and passing ibkr_account_id to main/broker."""
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "accounts.json")
    if not os.path.exists(path):
        return []
    try:
        with open(path, "r") as f:
            data = json.load(f)
        return data.get("accounts") or []
    except Exception:
        return []


def save_json(path, data):
    with open(path, "w") as f:
        json.dump(data, f, indent=4)
    st.success(f"Saved changes to {path}")


# -------------------------------------
# CONFIG EDITOR PAGE
# -------------------------------------
def edit_config_page():
    st.header("Configuration Editor")

    accounts = load_accounts()
    account_nicknames = [a["nickname"] for a in accounts] if accounts else ["default"]
    selected_nickname = st.selectbox(
        "Profile (Account)",
        account_nicknames,
        index=0,
        help="Configuration is stored in config_{nickname}.json for this profile.",
        key="config_editor_profile"
    )
    config_path = get_config_path_for_nickname(selected_nickname)

    if not os.path.exists(config_path):
        st.info(f"**{config_path}** not found. Create it manually, or initialize from the default config below.")
        if os.path.exists(CONFIG_PATH) and st.button("Initialize from config.json", key="init_config_from_default"):
            try:
                with open(CONFIG_PATH, "r") as f:
                    default_config = json.load(f)
                with open(config_path, "w") as f:
                    json.dump(default_config, f, indent=4)
                st.success(f"Created {config_path} from config.json. You can edit and save below.")
                st.rerun()
            except Exception as e:
                st.error(f"Failed to create config: {e}")
        return

    config = load_json(config_path)
    if config is None:
        return

    st.caption(f"Editing: **{config_path}**")

    # -------------------------
    # Broker Section
    # -------------------------
    st.subheader("Broker Settings")
    broker = config["broker"]

    col1, col2, col3 = st.columns(3)
    with col1:
        broker["host"] = st.text_input("Host", broker["host"])
    with col2:
        broker["port"] = st.number_input("Port", value=broker["port"])
    with col3:
        broker["client_id"] = st.number_input("Client ID", value=broker["client_id"], step=1)

    st.markdown("---")

    # -------------------------
    # Underlying Section
    # -------------------------
    st.subheader("Underlying Settings")
    underlying = config["underlying"]

    col1, col2, col3 = st.columns(3)
    with col1:
        underlying["symbol"] = st.text_input("Symbol", underlying["symbol"])
    with col2:
        underlying["exchange"] = st.text_input("Exchange", underlying["exchange"])
    with col3:
        underlying["currency"] = st.text_input("Currency", underlying["currency"])
    
    col4, col5, col6 = st.columns(3)
    with col4:
        underlying["trading_class"] = st.text_input("Trading Class", underlying.get("trading_class", ""))
    with col5:
        underlying["multiplier"] = st.number_input("Multiplier", value=int(underlying.get("multiplier", 100)), min_value=1, step=1, help="Contract multiplier (e.g. 100 for SPX, 10 for XSP)")
    with col6:
        st.write("")

    st.markdown("---")

    # -------------------------
    # Expiry Section
    # -------------------------
    st.subheader("Expiry Settings")
    expiry = config["expiry"]

    col1, col2, col3 = st.columns(3)
    with col1:
        expiry["date"] = st.text_input("Expiry Date (YYYYMMDD)", expiry["date"])
    with col2:
        st.write("")
    with col3:
        st.write("")

    st.markdown("---")

    # -------------------------
    # Trade Parameters
    # -------------------------
    st.subheader("Trade Parameters")
    tp = config["trade_parameters"]

    col1, col2, col3 = st.columns(3)
    with col1:
        tp["call_quantity"] = st.number_input("Call Quantity", value=tp["call_quantity"])
    with col2:
        tp["put_quantity"] = st.number_input("Put Quantity", value=tp["put_quantity"])
    with col3:
        tp["strike_step"] = st.number_input("Strike Step", value=tp["strike_step"])

    col1, col2 = st.columns(2)
    with col1:
        tp["max_bid_ask_spread"] = st.number_input("Max Bid/Ask Spread", value=tp["max_bid_ask_spread"])
    with col2:
        st.write("")  # Empty column for layout

    col1, col2, col3 = st.columns(3)
    with col1:
        tp["entry_vwap_multiplier"] = st.number_input("Entry VWAP Multiplier", value=tp["entry_vwap_multiplier"])
    with col2:
        tp["exit_vwap_multiplier"] = st.number_input("Exit VWAP Multiplier", value=tp["exit_vwap_multiplier"])
    st.markdown("### Take Profit Levels")

    # Ensure structure exists
    if "take_profit_levels" not in tp:
        tp["take_profit_levels"] = [
            {"pnl_percent": 0.30, "exit_percent": 0.50},
            {"pnl_percent": 0.40, "exit_percent": 0.30},
            {"pnl_percent": 0.50, "exit_percent": 0.20},
        ]

    tp_levels = tp["take_profit_levels"]

    for i, level in enumerate(tp_levels):
        col1, col2, col3 = st.columns([3, 3, 1])

        with col1:
            level["pnl_percent"] = st.number_input(
                f"TP {i+1} PnL %",
                min_value=0.0,
                max_value=5.0,
                step=0.01,
                value=float(level["pnl_percent"]),
                key=f"tp_pnl_{i}"
            )

        with col2:
            level["exit_percent"] = st.number_input(
                f"TP {i+1} Exit %",
                min_value=0.0,
                max_value=1.0,
                step=0.05,
                value=float(level["exit_percent"]),
                key=f"tp_exit_{i}"
            )

    st.markdown("### Stop Loss, Drawdown & Profit Limit")
    col1, col2, col3 = st.columns(3)
    with col1:
        tp["stop_loss"] = st.number_input("Stop Loss (%)", value=tp["stop_loss"])
    with col2:
        tp["drawdown_limit"] = st.number_input(
            "Drawdown Limit ($)",
            value=int(tp.get("drawdown_limit", 5000)),
            min_value=0,
            step=500,
            help="Exit all and stop when total PnL is at or below this loss. Checked every 30s."
        )
    with col3:
        tp["profit_limit"] = st.number_input(
            "Profit Limit ($)",
            value=int(tp.get("profit_limit", 5000)),
            min_value=0,
            step=500,
            help="Exit all and stop when total PnL reaches or exceeds this profit. Checked every 30s."
        )

    st.markdown("---")

    # -------------------------
    # Time Controls
    # -------------------------
    st.subheader("Time Controls")
    tc = config["time_controls"]

    col1, col2, col3 = st.columns(3)
    with col1:
        tc["entry_start"] = st.text_input("Entry Start Time", tc["entry_start"])
    with col2:
        tc["entry_end"] = st.text_input("Entry End Time", tc["entry_end"])
    with col3:
        tc["force_exit_time"] = st.text_input("Force Exit Time", tc["force_exit_time"])

    col1, col2, col3 = st.columns(3)
    with col1:
        tc["timezone"] = st.text_input("Timezone", tc["timezone"])
    with col2:
        st.write("")
    with col3:
        st.write("")

    st.markdown("---")

    # -------------------------
    # Hedging
    # -------------------------
    st.subheader("Hedging Settings")
    hedging = config["hedging"]

    col1, col2, col3 = st.columns(3)
    with col1:
        hedging["enable_hedges"] = st.checkbox("Enable Hedges", hedging["enable_hedges"])
    with col2:
        hedging["hedge_call_offset"] = st.number_input("Hedge Call Offset", value=hedging["hedge_call_offset"])
    with col3:
        hedging["hedge_put_offset"] = st.number_input("Hedge Put Offset", value=hedging["hedge_put_offset"])

    col1, col2, col3 = st.columns(3)
    with col1:
        hedging["hedge_quantity"] = st.number_input("Hedge Quantity", value=hedging["hedge_quantity"])
    with col2:
        st.write("")
    with col3:
        st.write("")

    st.markdown("---")

    total_exit = sum(l["exit_percent"] for l in tp["take_profit_levels"])
    if total_exit > 1.0:
        st.error("❌ Total TP exit percentage exceeds 100%")
        return


    if st.button("Save Configuration"):
        save_json(config_path, config)



# -------------------------------------
# POSITIONS VIEWER PAGE
# -------------------------------------
def positions_page():
    st.header("Positions Dashboard")

    accounts = load_accounts()
    account_options = [None] + [a["nickname"] for a in accounts] if accounts else [None]

    # Get filters (account = nickname; DB is keyed by nickname)
    col1, col2 = st.columns(2)
    
    with col1:
        account = st.selectbox(
            "Filter by Account",
            account_options,
            format_func=lambda x: "All Accounts" if x is None else x,
            index=0
        )
    
    with col2:
        symbol = st.selectbox(
            "Filter by Instrument",
            [None, "SPX", "XSP", "QQQ"],
            format_func=lambda x: "All Instruments" if x is None else x,
            index=0
        )

    # Get today's date in Eastern time
    today_eastern = datetime.now(EASTERN_TZ).date()
    today_str = today_eastern.strftime("%Y-%m-%d")

    # Get all positions from database
    all_positions = _db.get_all_positions(account)
    
    # Filter to only show positions that were opened today AND (if closed) closed today in Eastern time
    # Also filter by symbol if selected
    positions = []
    for p in all_positions:
        entry_time = p.get("entry_time", "")
        exit_time = p.get("exit_time")
        pos_symbol = p.get("symbol")
        
        # Filter by symbol if selected
        if symbol and pos_symbol != symbol:
            continue
        
        # Must be opened today
        if not entry_time.startswith(today_str):
            continue
        
        # If closed, must be closed today
        if exit_time and not exit_time.startswith(today_str):
            continue
        
        positions.append(p)
    
    # Removed active positions tracking since we're running multiple threads with different strikes
    # Each thread manages its own positions independently

    # ------------------------------------
    # Compute PnL from database fields (today only)
    # ------------------------------------
    # Use realized_pnl and unrealized_pnl fields from database
    realized = sum(p.get("realized_pnl", 0) or 0 for p in positions)
    unrealized = sum(p.get("unrealized_pnl", 0) or 0 for p in positions)
    combined = realized + unrealized

    # Closed trade stats (from today's positions only)
    closed_positions = [p for p in positions if not p.get("active", False)]
    realized_list = [p.get("realized_pnl", 0) or 0 for p in closed_positions]

    wins = [p for p in realized_list if p > 0]
    losses = [p for p in realized_list if p < 0]

    win_rate = (len(wins) / len(realized_list) * 100) if realized_list else 0
    avg_win = (sum(wins) / len(wins)) if wins else 0
    avg_loss = (sum(losses) / len(losses)) if losses else 0

    # ------------------------------------
    # Display PnL Summary Cards
    # ------------------------------------
    filter_text = f"Today: {today_str}"
    if account:
        filter_text += f" | Account: {account}"
    if symbol:
        filter_text += f" | Instrument: {symbol}"
    st.subheader(f"PnL Summary ({filter_text})")

    # ------- Row 1 -------
    row1 = st.columns(3)

    with row1[0]:
        st.metric("Realized PnL", f"${realized:.2f}")

    with row1[1]:
        st.metric("Unrealized PnL", f"${unrealized:.2f}")

    with row1[2]:
        st.metric("Total Combined PnL", f"${combined:.2f}")

    # ------- Row 2 -------
    row2 = st.columns(3)

    with row2[0]:
        st.metric("Win Rate", f"{win_rate:.2f}%")

    with row2[1]:
        st.metric("Avg Win", f"${avg_win:.2f}")

    with row2[2]:
        st.metric("Avg Loss", f"${avg_loss:.2f}")





    st.markdown("---")

    # ------------------------------------
    # ALL POSITIONS TABLE
    # ------------------------------------
    table_filter_text = f"Today's Positions (Eastern Time: {today_str})"
    if account:
        table_filter_text += f" | Account: {account}"
    if symbol:
        table_filter_text += f" | Instrument: {symbol}"
    st.subheader(table_filter_text)

    if not positions:
        st.warning(f"No positions found for today ({today_str}).")
        return

    df = pd.DataFrame(positions)

    # Note: Using realized_pnl and unrealized_pnl fields from database

    # Remove fields that shouldn't be displayed in frontend (but keep in database)
    columns_to_hide = ['id', 'order_id_entry', 'order_id_exit', 'last_update', 'bid', 'ask', 'last', 'expiry', 'side']
    df = df.drop(columns=[col for col in columns_to_hide if col in df.columns])

    # Safe datetime conversion
    df["entry_time"] = pd.to_datetime(df["entry_time"], errors="coerce")
    df = df.sort_values("entry_time", ascending=False)

    # Highlight active rows
    def highlight_active(row):
        if row["active"]:
            return ['background-color: #0B5A0B; color: white; font-weight: 600;' for _ in row]
        return ['' for _ in row]

    st.dataframe(
        df.style.apply(highlight_active, axis=1),
        width="stretch"
    )

def update_flags(paused=None, stopped=None):
    config = load_json(CONFIG_PATH)
    if not config:
        return

    if paused is not None:
        config["paused"] = paused
    if stopped is not None:
        config["stopped"] = stopped

    save_json(CONFIG_PATH, config)

# -------------------------------------
# HELPER: Display historical data
# -------------------------------------
def _display_historical_data(positions, filters=None):
    """Helper function to display historical data results"""
    # Display summary
    st.markdown("---")
    st.subheader("Summary Statistics")
    
    # Calculate realized PnL (only for closed positions)
    closed_positions = [p for p in positions if not p.get("active", False)]
    realized_pnl = sum(p.get("realized_pnl", 0) or 0 for p in closed_positions)
    
    # Additional stats
    total_positions = len(positions)
    closed_count = len(closed_positions)
    active_count = total_positions - closed_count
    
    wins = [p.get("realized_pnl", 0) or 0 for p in closed_positions if (p.get("realized_pnl", 0) or 0) > 0]
    losses = [p.get("realized_pnl", 0) or 0 for p in closed_positions if (p.get("realized_pnl", 0) or 0) < 0]
    
    win_rate = (len(wins) / closed_count * 100) if closed_count > 0 else 0
    avg_win = (sum(wins) / len(wins)) if wins else 0
    avg_loss = (sum(losses) / len(losses)) if losses else 0
    total_wins = sum(wins)
    total_losses = sum(losses)
    
    # Display metrics
    row1 = st.columns(4)
    with row1[0]:
        st.metric("Total Positions", total_positions)
    with row1[1]:
        st.metric("Closed Positions", closed_count)
    with row1[2]:
        st.metric("Active Positions", active_count)
    with row1[3]:
        st.metric("Total Realized PnL", f"${realized_pnl:.2f}")
    
    row2 = st.columns(4)
    with row2[0]:
        st.metric("Win Rate", f"{win_rate:.2f}%")
    with row2[1]:
        st.metric("Total Wins", f"${total_wins:.2f}")
    with row2[2]:
        st.metric("Total Losses", f"${total_losses:.2f}")
    with row2[3]:
        st.metric("Avg Win", f"${avg_win:.2f}")
    
    row3 = st.columns(4)
    with row3[0]:
        st.metric("Avg Loss", f"${avg_loss:.2f}")
    with row3[1]:
        st.metric("Winning Trades", len(wins))
    with row3[2]:
        st.metric("Losing Trades", len(losses))
    with row3[3]:
        profit_factor = abs(total_wins / total_losses) if total_losses != 0 else float('inf') if total_wins > 0 else 0
        st.metric("Profit Factor", f"{profit_factor:.2f}" if profit_factor != float('inf') else "∞")
    
    # Display detailed table
    st.markdown("---")
    st.subheader("Position Details")
    
    # Prepare data for display (excluding hidden fields)
    display_data = []
    for p in positions:
        display_data.append({
            "Account": p.get("account", ""),
            "Symbol": p.get("symbol", ""),
            "Type": p.get("position_type", ""),
            "Right": p.get("right", ""),
            "Strike": p.get("strike", 0),
            "Qty": p.get("qty", 0),
            "Entry Time": p.get("entry_time", ""),
            "Exit Time": p.get("exit_time", ""),
            "Entry Price": f"${p.get('entry_price', 0):.2f}" if p.get("entry_price") else "N/A",
            "Close Price": f"${p.get('close_price', 0):.2f}" if p.get("close_price") else "N/A",
            "Realized PnL": f"${p.get('realized_pnl', 0):.2f}" if p.get("realized_pnl") else "$0.00",
            "Unrealized PnL": f"${p.get('unrealized_pnl', 0):.2f}" if p.get("unrealized_pnl") else "$0.00",
            "Active": "Yes" if p.get("active") else "No"
        })
    
    df = pd.DataFrame(display_data)
    
    # Display dataframe
    st.dataframe(
        df,
        use_container_width=True,
        hide_index=True
    )
    
    # Download button
    csv = df.to_csv(index=False)
    if filters:
        account_display = filters.get("account") or "all"
        symbol_display = filters.get("symbol") or "all"
        start_display = filters.get("start_date") or "all"
        end_display = filters.get("end_date") or "all"
    else:
        account_display = "all"
        symbol_display = "all"
        start_display = "all"
        end_display = "all"
    
    st.download_button(
        label="Download as CSV",
        data=csv,
        file_name=f"historical_data_{account_display}_{symbol_display}_{start_display}_{end_display}.csv",
        mime="text/csv"
    )

# -------------------------------------
# HISTORICAL DATA PAGE
# -------------------------------------
def historical_data_page():
    st.header("Historical Data Analysis")
    
    # Initialize session state for historical data
    if "historical_data" not in st.session_state:
        st.session_state.historical_data = None
    if "historical_filters" not in st.session_state:
        st.session_state.historical_filters = None
    
    # Filters section
    st.subheader("Filters")
    
    accounts = load_accounts()
    account_options = [None] + [a["nickname"] for a in accounts] if accounts else [None]

    col1, col2 = st.columns(2)
    
    with col1:
        account = st.selectbox(
            "Filter by Account",
            account_options,
            format_func=lambda x: "All Accounts" if x is None else x,
            index=0,
            key="hist_account"
        )
    
    with col2:
        symbol = st.selectbox(
            "Filter by Instrument",
            [None, "SPX", "XSP", "QQQ"],
            format_func=lambda x: "All Instruments" if x is None else x,
            index=0,
            key="hist_symbol"
        )
    
    # Date range
    st.markdown("### Date Range")
    col3, col4 = st.columns(2)
    
    with col3:
        start_date = st.date_input(
            "Start Date",
            value=None,
            key="hist_start_date"
        )
    
    with col4:
        end_date = st.date_input(
            "End Date",
            value=None,
            key="hist_end_date"
        )
    
    # Validate date range
    if start_date and end_date and start_date > end_date:
        st.error("Start date must be before or equal to end date.")
        return
    
    # Clear data button
    if st.session_state.historical_data is not None:
        if st.button("Clear Results", type="secondary"):
            st.session_state.historical_data = None
            st.session_state.historical_filters = None
            st.rerun()
    
    # Query button
    if st.button("Query Historical Data", type="primary"):
        # Format dates for database query
        start_date_str = start_date.strftime("%Y-%m-%d") if start_date else None
        end_date_str = end_date.strftime("%Y-%m-%d") if end_date else None
        
        # Query database
        positions = _db.get_positions_with_filters(
            account=account,
            symbol=symbol,
            start_date=start_date_str,
            end_date=end_date_str
        )
        
        # Store in session state
        st.session_state.historical_data = positions
        st.session_state.historical_filters = {
            "account": account,
            "symbol": symbol,
            "start_date": start_date_str,
            "end_date": end_date_str
        }
        
        if not positions:
            st.warning("No positions found matching the selected criteria.")
            st.session_state.historical_data = None
            st.session_state.historical_filters = None
            return
        
        # Display the data using helper function
        _display_historical_data(positions, st.session_state.historical_filters)
    
    # Display stored data if available (when page refreshes but data is in session state)
    elif st.session_state.historical_data is not None:
        positions = st.session_state.historical_data
        filters = st.session_state.historical_filters
        
        st.info("📊 Displaying previously queried data. Auto-refresh is disabled. Click 'Clear Results' to enable auto-refresh again.")
        
        # Display the data using helper function
        _display_historical_data(positions, filters)



# -------------------------------------
# STATUS INDICATOR
# -------------------------------------
def status_indicator():
    """Show status for all running accounts with their controls"""
    running_accounts = get_all_running_accounts()
    accounts = load_accounts()
    id_to_nickname = {a["ibkr_account_id"]: a["nickname"] for a in accounts} if accounts else {}

    if running_accounts:
        st.markdown("### Running Accounts")
        for acc, sym in running_accounts:
            status_path = get_status_path(acc, sym)
            symbol_info = sym or "N/A"
            paused = False
            try:
                if os.path.exists(status_path):
                    with open(status_path, "r") as f:
                        parts = f.read().strip().split("|")
                        if len(parts) >= 4:
                            symbol_info = parts[3]
                            paused = parts[0] == "paused"
            except:
                pass

            account_display = id_to_nickname.get(acc, acc)

            # Get state from state manager (using account+symbol)
            account_state = get_account_state(acc, sym)
            is_paused = account_state.get("paused", False)

            col1, col2, col3, col4 = st.columns([2, 1, 1, 1])
            with col1:
                status_color = "#ca8a04" if is_paused else "#10b981"
                status_text = "PAUSED" if is_paused else "RUNNING"
                st.markdown(
                    f"""
                    <div style="padding:8px 15px;border-radius:6px;
                                background:{status_color}20;border:1px solid {status_color};display:inline-block;">
                        <span style="color:{status_color};font-size:14px;font-weight:600;">
                            ● {account_display} ({symbol_info}) - {status_text}
                        </span>
                    </div>
                    """,
                    unsafe_allow_html=True
                )
            with col2:
                if st.button("Pause", key=f"pause_{acc}_{sym}", disabled=is_paused):
                    set_account_paused(acc, True, sym)
                    st.rerun()
            with col3:
                if st.button("Resume", key=f"resume_{acc}_{sym}", disabled=not is_paused):
                    set_account_paused(acc, False, sym)
                    st.rerun()
            with col4:
                if st.button("Stop", key=f"stop_{acc}_{sym}"):
                    set_account_stopped(acc, True, sym)
                    # Also kill the process
                    pid = load_pid(acc, sym)
                    if pid and pid_running(pid, acc, sym):
                        try:
                            if os.name == 'nt':
                                subprocess.run(
                                    ['taskkill', '/F', '/PID', str(pid), '/T'],
                                    capture_output=True,
                                    timeout=5
                                )
                            else:
                                os.kill(pid, 9)
                        except:
                            pass
                    # Clean up files
                    pid_path = get_pid_path(acc, sym)
                    status_path = get_status_path(acc, sym)
                    try:
                        if os.path.exists(pid_path):
                            os.remove(pid_path)
                        if os.path.exists(status_path):
                            os.remove(status_path)
                    except:
                        pass
                    st.rerun()
    else:
        st.markdown(
            f"""
            <div style="padding:10px 20px;border-radius:8px;
                        background:#6b728020;border:1px solid #6b7280;display:inline-block;">
                <span style="color:#6b7280;font-size:18px;font-weight:600;">
                    ● IDLE (No accounts running)
                </span>
            </div>
            """,
            unsafe_allow_html=True
        )


# -------------------------------------
# MAIN APP — TABS
# -------------------------------------
def main():
    st.title("Options Strategy Dashboard")

    sync_flags_with_pid()
    status_indicator()

    # ===============================
    # NEW: ACCOUNT + SYMBOL SELECTION
    # ===============================
    st.subheader("Run Configuration")

    accounts = load_accounts()
    account_nicknames = [a["nickname"] for a in accounts] if accounts else ["default"]

    col1, col2 = st.columns(2)

    with col1:
        selected_nickname = st.selectbox(
            "Select Account",
            account_nicknames,
            index=0
        )
        # Use nickname for PID, state, DB, and --account (main.py resolves to ibkr_account_id for broker)
        account = selected_nickname

    with col2:
        symbol = st.selectbox(
            "Select Symbol",
            ["SPX", "XSP", "QQQ"],
            index=0
        )

    st.markdown("---")

    # Strike Range Configuration
    st.subheader("Strike Range Configuration")
    
    col3, col4 = st.columns(2)
    with col3:
        range_start = st.number_input(
            "Range Start (offset)",
            min_value=-50,
            max_value=50,
            value=4,
            step=1,
            help="Starting offset value (e.g., -2 means ATM-2)"
        )
    with col4:
        range_end = st.number_input(
            "Range End (offset)",
            min_value=-50,
            max_value=50,
            value=10,
            step=1,
            help="Ending offset value (e.g., 2 means ATM+2)"
        )
    
    exclude_offsets = st.text_input(
        "Exclude offsets (optional)",
        value="",
        placeholder="e.g. -1, 5, 7 or -1;5;7",
        help="Offsets to exclude from the range above (can be negative). Use comma, semicolon, or space as separator."
    )
    
    if range_start > range_end:
        st.error("Range start must be less than or equal to range end")
        strike_range_str = None
        strike_range_arg = None
    else:
        strike_range_str = f"{range_start}:{range_end}"
        # Use hyphen for --range so Windows doesn't drop the value (colon can be parsed as drive)
        strike_range_arg = f"{range_start}-{range_end}"

    st.markdown("---")

    st.subheader("Start New Strategy")

    # ---------------- START ---------------- 
    if st.button("Start Strategy", type="primary"):
        if strike_range_str is None or strike_range_arg is None:
            st.error("Please fix the range configuration before starting")
        else:
            base_dir = os.path.dirname(os.path.abspath(__file__))
            main_path = os.path.join(base_dir, "main.py")

            # Check if account+symbol is already running
            existing_pid = load_pid(account, symbol)
            if existing_pid and pid_running(existing_pid, account, symbol):
                st.warning(f"Account {account} with symbol {symbol} is already running! Stop it first before starting again.")
            else:
                try:
                    # Clear any previous state (using account+symbol)
                    set_account_paused(account, False, symbol)
                    set_account_stopped(account, False, symbol)
                    
                    # Build command: use profile-specific config config_{nickname}.json
                    profile_config_path = get_config_path_for_nickname(selected_nickname)
                    if not os.path.exists(profile_config_path):
                        st.error(f"Config for this profile not found: {profile_config_path}. Create it manually or use Config Editor to initialize from default.")
                    else:
                        cmd_args = [python_path, main_path, '--account', selected_nickname, '--symbol', symbol, '--config', profile_config_path, f'--range={strike_range_arg}']
                        if exclude_offsets and exclude_offsets.strip():
                            cmd_args.append(f'--exclude={exclude_offsets.strip()}')
                        
                        if os.name == 'nt':  # Windows
                            # Use CREATE_NEW_CONSOLE to open in new window
                            # Note: The PID will be written by main.py itself when it starts
                            p = subprocess.Popen(
                                cmd_args,
                                cwd=base_dir,
                                creationflags=subprocess.CREATE_NEW_CONSOLE
                            )
                            # Save the PID (main.py will also write it, but this is a backup)
                            save_pid(p.pid, account, symbol)
                        else:
                            p = subprocess.Popen(cmd_args, cwd=base_dir)
                            save_pid(p.pid, account, symbol)
                        
                        range_display = strike_range_str + (f" (exclude: {exclude_offsets.strip()})" if exclude_offsets and exclude_offsets.strip() else "")
                        st.success(f"Started {selected_nickname} | {symbol} | Range: {range_display}")
                        time.sleep(1)  # Give it a moment to start
                        st.rerun()
                except Exception as e:
                    st.error(f"Error: {e}")
    
    st.info("💡 Use the controls in the 'Running Accounts' section above to pause, resume, or stop individual strategies.")

    # ===============================
    # TABS
    # ===============================
    tab1, tab2, tab3 = st.tabs(["Config Editor", "Positions Viewer", "Historical Data"])
    
    with tab1:
        edit_config_page()

    with tab2:
        positions_page()

    with tab3:
        historical_data_page()
    
    # Only auto-refresh if there's no historical data in session state
    # This prevents losing queried data on the historical data page
    if "historical_data" not in st.session_state or st.session_state.historical_data is None:
        st_autorefresh(interval=5000, limit=None, key="auto_refresh")

# -------------------------------------
# ENTRY POINT
# -------------------------------------
if __name__ == "__main__":
    st.set_page_config(
        page_title="Options Strategy Dashboard",
        page_icon="📈",
        layout="wide"
    )
    main()
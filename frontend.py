import streamlit as st
import json
import os
import pandas as pd
from streamlit_autorefresh import st_autorefresh
import subprocess
import sys
from datetime import datetime
import pytz
from db.position_db import PositionDB
python_path = sys.executable
st_autorefresh(interval=5000, limit=None, key="auto_refresh")

# Database instance
_db = PositionDB()

# Eastern timezone
EASTERN_TZ = pytz.timezone("US/Eastern")

CONFIG_PATH = "config.json"
POSITIONS_PATH = "positions.json"
PID_PATH = "bot.pid"


# -------------------------------------
# Load JSON helpers
# -------------------------------------
def save_pid(pid):
    with open(PID_PATH, "w") as f:
        f.write(str(pid))

def load_pid():
    if not os.path.exists(PID_PATH):
        return None
    try:
        return int(open(PID_PATH).read().strip())
    except:
        return None

def pid_running(pid):
    if pid is None:
        return False
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
    """Ensure config.json matches the actual process state."""
    pid = load_pid()
    alive = pid_running(pid)

    config = load_json(CONFIG_PATH)
    if config is None:
        return

    paused = config.get("paused", False)
    stopped = config.get("stopped", False)

    # If PID is dead but the bot is marked running → fix it
    if (not alive) and (not stopped):
        config["stopped"] = True
        config["paused"] = False   # can't be paused if dead
        save_json(CONFIG_PATH, config)

        # Remove PID file if stale
        if os.path.exists(PID_PATH):
            os.remove(PID_PATH)


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


def save_json(path, data):
    with open(path, "w") as f:
        json.dump(data, f, indent=4)
    st.success(f"Saved changes to {path}")


# -------------------------------------
# CONFIG EDITOR PAGE
# -------------------------------------
def edit_config_page():
    st.header("Configuration Editor")

    config = load_json(CONFIG_PATH)
    if config is None:
        return

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

    col1, col2, col3 = st.columns(3)
    with col1:
        tp["atm_call_offset"] = st.number_input("ATM Call Offset", value=tp["atm_call_offset"])
    with col2:
        tp["atm_put_offset"] = st.number_input("ATM Put Offset", value=tp["atm_put_offset"])
    with col3:
        tp["max_bid_ask_spread"] = st.number_input("Max Bid/Ask Spread", value=tp["max_bid_ask_spread"])

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

    st.markdown("### Stop Loss")
    col1, col2, col3 = st.columns(3)
    with col1:
        tp["stop_loss"] = st.number_input("Stop Loss (%)", value=tp["stop_loss"])
    with col2:
        st.write("")
    with col3:
        st.write("")

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
        save_json(CONFIG_PATH, config)



# -------------------------------------
# POSITIONS VIEWER PAGE
# -------------------------------------
def positions_page():
    st.header("Positions Dashboard")

    # Get account filter (optional - can add UI for this later)
    account = st.selectbox(
        "Filter by Account (optional)",
        [None, "acc1", "acc2", "acc3"],
        format_func=lambda x: "All Accounts" if x is None else x,
        index=0
    )

    # Get today's date in Eastern time
    today_eastern = datetime.now(EASTERN_TZ).date()
    today_str = today_eastern.strftime("%Y-%m-%d")

    # Get all positions from database
    all_positions = _db.get_all_positions(account)
    
    # Filter to only show positions that were opened today AND (if closed) closed today in Eastern time
    positions = []
    for p in all_positions:
        entry_time = p.get("entry_time", "")
        exit_time = p.get("exit_time")
        
        # Must be opened today
        if not entry_time.startswith(today_str):
            continue
        
        # If closed, must be closed today
        if exit_time and not exit_time.startswith(today_str):
            continue
        
        positions.append(p)
    
    active_positions = [p for p in positions if p.get("active", False)]
    
    # Build active IDs from filtered active positions (today only)
    active_ids = {
        "position_open": len(active_positions) > 0,
        "atm_call_id": next((p["id"] for p in active_positions if p.get("position_type") == "ATM" and p.get("right") == "C"), None),
        "atm_put_id": next((p["id"] for p in active_positions if p.get("position_type") == "ATM" and p.get("right") == "P"), None),
        "otm_call_id": next((p["id"] for p in active_positions if p.get("position_type") == "OTM" and p.get("right") == "C"), None),
        "otm_put_id": next((p["id"] for p in active_positions if p.get("position_type") == "OTM" and p.get("right") == "P"), None),
    }

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
    st.subheader(f"PnL Summary (Today: {today_str})")

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
    # ACTIVE POSITIONS
    # ------------------------------------
    st.subheader("Active Positions")

    if not active_ids["position_open"]:
        st.info("No active positions.")
    else:
        col1, col2 = st.columns(2)

        with col1:
            st.write("### ATM Positions")
            st.write(f"**ATM Call ID:** {active_ids.get('atm_call_id')}")
            st.write(f"**ATM Put ID:** {active_ids.get('atm_put_id')}")

        with col2:
            st.write("### OTM Positions")
            st.write(f"**OTM Call ID:** {active_ids.get('otm_call_id')}")
            st.write(f"**OTM Put ID:** {active_ids.get('otm_put_id')}")

    st.markdown("---")

    # ------------------------------------
    # ALL POSITIONS TABLE
    # ------------------------------------
    st.subheader(f"Today's Positions (Eastern Time: {today_str})")

    if not positions:
        st.warning(f"No positions found for today ({today_str}).")
        return

    df = pd.DataFrame(positions)

    # Note: Using realized_pnl and unrealized_pnl fields from database

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
# STATUS INDICATOR
# -------------------------------------
def status_indicator():
    config = load_json(CONFIG_PATH)
    if not config:
        return

    paused = config.get("paused", False)
    stopped = config.get("stopped", False)

    pid = load_pid()
    alive = pid_running(pid)

    if alive and not paused and not stopped:
        text, color = "RUNNING", "#16a34a"
    elif alive and paused:
        text, color = "PAUSED", "#ca8a04"
    elif not alive and stopped:
        text, color = "STOPPED", "#dc2626"
    else:
        text, color = "IDLE", "#6b7280"

    st.markdown(
        f"""
        <div style="padding:10px 20px;border-radius:8px;
                    background:{color}20;border:1px solid {color};display:inline-block;">
            <span style="color:{color};font-size:18px;font-weight:600;">
                ● {text}
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

    col1, col2 = st.columns(2)

    with col1:
        account = st.selectbox(
            "Select Account",
            ["acc1", "acc2", "acc3"],
            index=0
        )

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
    
    if range_start > range_end:
        st.error("Range start must be less than or equal to range end")
        strike_range_str = None
    else:
        strike_range_str = f"{range_start}:{range_end}"

    st.markdown("---")

    st.subheader("Strategy Controls")

    colA, colB, colC = st.columns(3)

    # ---------------- START ---------------- 
    with colA:
        if st.button("Start Strategy"):
            if strike_range_str is None:
                st.error("Please fix the range configuration before starting")
            else:
                update_flags(paused=False, stopped=False)

                base_dir = os.path.dirname(os.path.abspath(__file__))
                main_path = os.path.join(base_dir, "main.py")

                try:
                    # Simple: just pass the 4 arguments
                    cmd_args = [python_path, main_path, '--account', account, '--symbol', symbol, '--range', strike_range_str]
                    
                    if os.name == 'nt':  # Windows
                        # Use CREATE_NEW_CONSOLE to open in new window
                        subprocess.Popen(
                            cmd_args,
                            cwd=base_dir,
                            creationflags=subprocess.CREATE_NEW_CONSOLE
                        )
                    else:
                        subprocess.Popen(cmd_args, cwd=base_dir)
                    
                    st.success(f"Started {account} | {symbol} | Range: {strike_range_str}")
                except Exception as e:
                    st.error(f"Error: {e}")

    # ---------------- PAUSE ----------------
    with colB:
        if st.button("Pause Strategy"):
            update_flags(paused=True)
            st.warning("Strategy paused.")

    # ---------------- STOP ----------------
    with colC:
        if st.button("Stop Strategy"):
            pid = load_pid()
            if pid and pid_running(pid):
                try:
                    if os.name == 'nt':  # Windows
                        subprocess.run(['taskkill', '/F', '/PID', str(pid)], capture_output=True)
                    else:
                        os.kill(pid, 9)
                except:
                    pass

            update_flags(stopped=True, paused=False)

            if os.path.exists(PID_PATH):
                os.remove(PID_PATH)

            st.error("Strategy stopped.")

    # ===============================
    # TABS
    # ===============================
    tab1, tab2 = st.tabs(["Config Editor", "Positions Viewer"])

    with tab1:
        edit_config_page()

    with tab2:
        positions_page()

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
import streamlit as st
import json
import os
import pandas as pd
from streamlit_autorefresh import st_autorefresh

# Refresh page every 5 seconds
st_autorefresh(interval=5000, limit=None, key="auto_refresh")

CONFIG_PATH = "config.json"
POSITIONS_PATH = "positions.json"


# -------------------------------------
# Load JSON helpers
# -------------------------------------
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
    st.header("⚙️ Configuration Editor")

    config = load_json(CONFIG_PATH)
    if config is None:
        return

    # ----- TABS FOR CONFIG SECTIONS -----
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "Broker",
        "Underlying",
        "Expiry",
        "Trade Parameters",
        "Time Controls",
        "Hedging"
    ])

    with tab1:
        broker = config["broker"]
        broker["host"] = st.text_input("Host", broker["host"])
        broker["port"] = st.number_input("Port", value=broker["port"])
        broker["client_id"] = st.number_input("Client ID", value=broker["client_id"], step=1)

    with tab2:
        underlying = config["underlying"]
        underlying["symbol"] = st.text_input("Symbol", underlying["symbol"])
        underlying["exchange"] = st.text_input("Exchange", underlying["exchange"])
        underlying["currency"] = st.text_input("Currency", underlying["currency"])

    with tab3:
        expiry = config["expiry"]
        expiry["date"] = st.text_input("Expiry Date (YYYYMMDD)", expiry["date"])

    with tab4:
        tp = config["trade_parameters"]
        tp["call_quantity"] = st.number_input("Call Quantity", value=tp["call_quantity"])
        tp["put_quantity"] = st.number_input("Put Quantity", value=tp["put_quantity"])
        tp["atm_call_offset"] = st.number_input("ATM Call Offset", value=tp["atm_call_offset"])
        tp["atm_put_offset"] = st.number_input("ATM Put Offset", value=tp["atm_put_offset"])
        tp["entry_vwap_multiplier"] = st.number_input("Entry VWAP Multiplier", value=tp["entry_vwap_multiplier"])
        tp["exit_vwap_multiplier"] = st.number_input(
            "Exit VWAP Multiplier (Exit if combined > VWAP x this)",
            value=tp["exit_vwap_multiplier"]
        )
        tp["take_profit"] = st.number_input("Take Profit (%)", value=tp["take_profit"])
        tp["stop_loss"] = st.number_input("Stop Loss (%)", value=tp["stop_loss"])
        tp["max_bid_ask_spread"] = st.number_input("Max Bid/Ask Spread", value=tp["max_bid_ask_spread"])
        tp["strike_step"] = st.number_input("Strike Step", value=tp["strike_step"])

    with tab5:
        tc = config["time_controls"]
        tc["entry_start"] = st.text_input("Entry Start Time", tc["entry_start"])
        tc["entry_end"] = st.text_input("Entry End Time", tc["entry_end"])
        tc["force_exit_time"] = st.text_input("Force Exit Time", tc["force_exit_time"])
        tc["timezone"] = st.text_input("Timezone", tc["timezone"])

    with tab6:
        hedging = config["hedging"]
        hedging["enable_hedges"] = st.checkbox("Enable Hedges", hedging["enable_hedges"])
        hedging["hedge_call_offset"] = st.number_input("Hedge Call Offset", value=hedging["hedge_call_offset"])
        hedging["hedge_put_offset"] = st.number_input("Hedge Put Offset", value=hedging["hedge_put_offset"])
        hedging["hedge_quantity"] = st.number_input("Hedge Quantity", value=hedging["hedge_quantity"])

    if st.button("💾 Save Configuration"):
        save_json(CONFIG_PATH, config)


# -------------------------------------
# POSITIONS VIEWER PAGE
# -------------------------------------
def positions_page():
    st.header("📊 Positions Dashboard")

    data = load_json(POSITIONS_PATH)
    if data is None:
        return

    positions = data["positions"]
    active = data["active_positions"]

    # ------------------------------------
    # Compute Combined PnL if positions active
    # ------------------------------------
    combined_atm_pnl = None
    combined_otm_pnl = None
    total_pnl = None

    if active["position_open"]:
        atm_call = next((p for p in positions if p["id"] == active["atm_call_id"]), None)
        atm_put  = next((p for p in positions if p["id"] == active["atm_put_id"]), None)

        if atm_call and atm_put:
            combined_atm_pnl = atm_call["pnl_pct"] + atm_put["pnl_pct"]

        otm_call = next((p for p in positions if p["id"] == active["otm_call_id"]), None)
        otm_put  = next((p for p in positions if p["id"] == active["otm_put_id"]), None)

        if otm_call and otm_put:
            combined_otm_pnl = otm_call["pnl_pct"] + otm_put["pnl_pct"]

        # Total
        total_pnl = 0
        for pid in ["atm_call_id", "atm_put_id", "otm_call_id", "otm_put_id"]:
            if active.get(pid):
                pos = next((p for p in positions if p["id"] == active[pid]), None)
                if pos:
                    total_pnl += pos["pnl_pct"]

    # ------------------------------------
    # Display PnL Summary Cards
    # ------------------------------------
    st.subheader("💰 PnL Summary")

    if not active["position_open"]:
        st.info("No active positions for PnL calculation.")
    else:
        col1, col2, col3 = st.columns(3)

        with col1:
            if combined_atm_pnl is not None:
                st.metric("ATM Combined PnL", f"{combined_atm_pnl*100:.2f}%")
            else:
                st.metric("ATM Combined PnL", "N/A")

        with col2:
            if combined_otm_pnl is not None:
                st.metric("OTM Combined PnL", f"{combined_otm_pnl*100:.2f}%")
            else:
                st.metric("OTM Combined PnL", "N/A")

        with col3:
            if total_pnl is not None:
                st.metric("TOTAL Strategy PnL", f"{total_pnl*100:.2f}%")
            else:
                st.metric("TOTAL Strategy PnL", "N/A")

    st.markdown("---")

    # ------------------------------------
    # ACTIVE POSITIONS
    # ------------------------------------
    st.subheader("🔥 Active Positions")

    if not active["position_open"]:
        st.info("No active positions.")
    else:
        col1, col2 = st.columns(2)

        with col1:
            st.write("### ATM Positions")
            st.write(f"**ATM Call ID:** {active.get('atm_call_id')}")
            st.write(f"**ATM Put ID:** {active.get('atm_put_id')}")

        with col2:
            st.write("### OTM Positions")
            st.write(f"**OTM Call ID:** {active.get('otm_call_id')}")
            st.write(f"**OTM Put ID:** {active.get('otm_put_id')}")

    st.markdown("---")

    # ------------------------------------
    # ALL POSITIONS TABLE
    # ------------------------------------
    st.subheader("📘 Historical & Active Positions")

    if not positions:
        st.warning("No positions found.")
        return

    df = pd.DataFrame(positions)

    # Convert PnL %
    if "pnl_pct" in df.columns:
        df["pnl_pct"] = (df["pnl_pct"] * 100).round(2).astype(str) + "%"

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



# -------------------------------------
# MAIN APP — TABS
# -------------------------------------
def main():
    st.title("📈 SPX Options Strategy Dashboard")

    tab1, tab2 = st.tabs(["⚙️ Config Editor", "📊 Positions Viewer"])

    with tab1:
        edit_config_page()

    with tab2:
        positions_page()


if __name__ == "__main__":
    main()

"""
Premium vs VWAP Graph Generator Page
Separate page to avoid auto-refresh interruptions
"""
import streamlit as st
import json
import os
import pandas as pd
import sys
from datetime import datetime
import pytz

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from helpers.graph_generator import get_current_price, get_today_expiry, generate_graph

# Page config
st.set_page_config(
    page_title="Premium vs VWAP Graph",
    page_icon="📊",
    layout="wide"
)

CONFIG_PATH = "config.json"

def load_json(path):
    if not os.path.exists(path):
        st.error(f"{path} not found!")
        return None
    with open(path, "r") as f:
        return json.load(f)

def main():
    st.title("Premium vs VWAP Graph Generator")
    
    # Initialize session state
    if "graph_spot_price" not in st.session_state:
        st.session_state.graph_spot_price = None
    if "graph_atm_strike" not in st.session_state:
        st.session_state.graph_atm_strike = None
    if "graph_strike_step" not in st.session_state:
        st.session_state.graph_strike_step = None
    if "price_fetch_success" not in st.session_state:
        st.session_state.price_fetch_success = False
    if "graph_generation_success" not in st.session_state:
        st.session_state.graph_generation_success = False
    
    # Load config
    config = load_json(CONFIG_PATH)
    if config is None:
        st.error("Config file not found!")
        return
    
    # Get symbol and expiry
    col1, col2 = st.columns(2)
    
    with col1:
        symbol = st.selectbox(
            "Symbol",
            ["SPX", "XSP"],
            index=0,
            key="graph_symbol"
        )
    
    with col2:
        tz_str = config.get("time_controls", {}).get("timezone", "US/Eastern")
        today_expiry = get_today_expiry(tz_str)
        expiry = st.text_input(
            "Expiry (YYYYMMDD)",
            value=today_expiry,
            key="graph_expiry",
            help="Enter expiry date in YYYYMMDD format"
        )
    
    st.markdown("---")
    
    # Get current price button
    if st.button("Get Current Price", type="primary", key="get_price_btn"):
        with st.spinner("Fetching current price..."):
            try:
                print(f"[GRAPH PAGE] Get Current Price button clicked for symbol={symbol}")
                spot_price, atm_strike, strike_step = get_current_price(symbol, CONFIG_PATH)
                print(f"[GRAPH PAGE] Received: spot_price={spot_price}, atm_strike={atm_strike}, strike_step={strike_step}")
                if spot_price is not None:
                    st.session_state.graph_spot_price = spot_price
                    st.session_state.graph_atm_strike = atm_strike
                    st.session_state.graph_strike_step = strike_step
                    st.session_state.price_fetch_success = True
                    print(f"[GRAPH PAGE] Session state updated - spot_price={spot_price}, atm_strike={atm_strike}")
                    st.rerun()
                else:
                    st.session_state.price_fetch_success = False
                    st.error("❌ Could not fetch current price. Please check your broker connection.")
            except Exception as e:
                print(f"[GRAPH PAGE] ERROR: {str(e)}")
                st.session_state.price_fetch_success = False
                st.error(f"Error fetching price: {str(e)}")
                import traceback
                st.code(traceback.format_exc())
    
    # Display current price if available (always show if in session state)
    if st.session_state.graph_spot_price is not None and st.session_state.graph_atm_strike is not None:
        
        if st.session_state.get("price_fetch_success", False):
            st.info(f"**Current Spot Price:** ${st.session_state.graph_spot_price:.2f} | **ATM Strike:** {st.session_state.graph_atm_strike}")
        else:
            st.error("❌ Could not fetch current price. Please check your broker connection.")
    
    st.markdown("---")
    
    # Strike input
    strike_input = st.number_input(
        "Strike Price",
        min_value=0.0,
        value=float(st.session_state.graph_atm_strike) if st.session_state.graph_atm_strike else 0.0,
        step=1.0,
        key="graph_strike_input",
        help="Enter the strike price to analyze"
    )
    
    # Duration and bar size
    col3, col4 = st.columns(2)
    with col3:
        duration = st.text_input(
            "Duration",
            value="1 D",
            key="graph_duration",
            help="Duration for historical data (e.g., '1 D', '2 D')"
        )
    with col4:
        bar_size = st.selectbox(
            "Bar Size",
            options=["1 min", "5 mins"],
            index=0,  # Default to "1 min"
            key="graph_bar_size",
            help="Bar size for historical data"
        )
    
    st.markdown("---")
    
    # Generate graph button
    if st.button("Generate Graph", type="primary", key="generate_graph_btn"):
        print(f"[GRAPH PAGE] Generate Graph button clicked")
        if not expiry or len(expiry) != 8 or not expiry.isdigit():
            st.error("Please enter a valid expiry date in YYYYMMDD format")
        elif strike_input <= 0:
            st.error("Please enter a valid strike price")
        else:
            with st.spinner("Generating graph... This may take a moment."):
                try:
                    # Round strike to nearest strike step if available
                    if st.session_state.graph_strike_step:
                        strike = round(strike_input / st.session_state.graph_strike_step) * st.session_state.graph_strike_step
                    else:
                        strike = strike_input
                    
                    print(f"[GRAPH PAGE] Calling generate_graph with: symbol={symbol}, expiry={expiry}, strike={strike}")
                    
                    # Generate the graph data
                    try:
                        plot_df = generate_graph(
                            symbol=symbol,
                            expiry=expiry,
                            strike=strike,
                            config_path=CONFIG_PATH,
                            duration=duration,
                            bar_size=bar_size
                        )
                        print(f"[GRAPH PAGE] Graph data received. Shape: {plot_df.shape if plot_df is not None else 'None'}")
                        
                        if plot_df is None or plot_df.empty:
                            print(f"[GRAPH PAGE] ERROR: No data or empty dataframe")
                            st.error("❌ No data returned from graph generation")
                            st.session_state.graph_generation_success = False
                        else:
                            print(f"[GRAPH PAGE] Storing graph data in session state...")
                            # Store in session state for persistence
                            st.session_state.last_graph_df = plot_df.copy()
                            st.session_state.last_graph_params = {
                                'symbol': symbol,
                                'expiry': expiry,
                                'strike': strike
                            }
                            st.session_state.graph_generation_success = True
                            print(f"[GRAPH PAGE] Graph data stored in session state. Shape: {plot_df.shape}")
                            st.rerun()
                    except ValueError as ve:
                        print(f"[GRAPH PAGE] ValueError from generate_graph: {str(ve)}")
                        st.error(f"❌ {str(ve)}")
                        import traceback
                        st.code(traceback.format_exc())
                    except Exception as e:
                        print(f"[GRAPH PAGE] Unexpected ERROR generating graph: {str(e)}")
                        st.error(f"Error generating graph: {str(e)}")
                        import traceback
                        st.code(traceback.format_exc())
                    
                except Exception as e:
                    print(f"[GRAPH PAGE] Outer exception: {str(e)}")
                    st.error(f"Error: {str(e)}")
                    import traceback
                    st.code(traceback.format_exc())
    
    # Display graph from session state if available
    if "last_graph_df" in st.session_state:
        print(f"[GRAPH PAGE] Found last_graph_df in session state! Shape: {st.session_state.last_graph_df.shape}")
        if not st.session_state.last_graph_df.empty:
            st.markdown("---")
            params = st.session_state.get("last_graph_params", {})
            if params:
                st.subheader(f"{params.get('symbol', '')} {params.get('expiry', '')} {params.get('strike', '')} - Combined Premium vs VWAP")
            else:
                st.subheader("Generated Graph - Combined Premium vs VWAP")
            
            print(f"[GRAPH PAGE] Displaying graph. Shape: {st.session_state.last_graph_df.shape}")
            print(f"[GRAPH PAGE] Time range: {st.session_state.last_graph_df.index.min()} to {st.session_state.last_graph_df.index.max()}")
            
            # Create a display-friendly version with time formatted as HH:MM
            display_df = st.session_state.last_graph_df.copy()
            display_df = display_df.reset_index()  # Reset index to make time a column
            if 'Time' in display_df.columns:
                # Convert to string format HH:MM for better display
                display_df['Time'] = pd.to_datetime(display_df['Time']).dt.strftime('%H:%M')
            display_df = display_df.set_index('Time')
            
            st.line_chart(display_df, use_container_width=True)
            
            # if len(st.session_state.last_graph_df) > 0:
            #     final_premium = st.session_state.last_graph_df['Combined Premium'].iloc[-1]
            #     final_vwap = st.session_state.last_graph_df['VWAP'].iloc[-1]
            #     col1, col2 = st.columns(2)
            #     with col1:
            #         st.metric("Final Premium", f"${final_premium:.2f}")
            #     with col2:
            #         st.metric("Final VWAP", f"${final_vwap:.2f}")
            

if __name__ == "__main__":
    main()


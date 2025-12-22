"""
Graph generation module for premium vs VWAP visualization.
Can be used both from command line and from frontend.
"""
import json
import pandas as pd
import numpy as np
from datetime import datetime, time as dt_time
import pytz
import sys
import os
import time as time_mod
import threading

# Add parent directory to path to import from broker
# Since we're now in helpers/, we need to go up one level to access broker/
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, parent_dir)

from broker.ib_broker import IBBroker

# Global counter for unique client IDs
_client_id_counter = 100  # Start from 100 to avoid conflicts
_client_id_lock = threading.Lock()

def get_unique_client_id():
    """Get a unique client ID for IBKR connection"""
    global _client_id_counter
    with _client_id_lock:
        _client_id_counter += 1
        return _client_id_counter


def get_today_expiry(tz_str="US/Eastern"):
    """Get today's date in YYYYMMDD format for 0DTE options"""
    tz = pytz.timezone(tz_str)
    today = datetime.now(tz)
    return today.strftime("%Y%m%d")


def filter_market_hours(df, tz_str="US/Eastern"):
    """Filter dataframe to only include market hours (9:30 AM - 4:00 PM ET)"""
    tz = pytz.timezone(tz_str)
    
    # Convert time column to datetime if it's not already
    if not pd.api.types.is_datetime64_any_dtype(df['time']):
        df['time'] = pd.to_datetime(df['time'])
    
    # Ensure timezone-aware
    if df['time'].dt.tz is None:
        df['time'] = df['time'].dt.tz_localize(tz)
    else:
        df['time'] = df['time'].dt.tz_convert(tz)
    
    # Filter to market hours (9:30 AM - 4:00 PM ET)
    market_start = dt_time(9, 30)
    market_end = dt_time(16, 0)  # 4:00 PM ET (market close)
    
    df['time_only'] = df['time'].dt.time
    df_filtered = df[(df['time_only'] >= market_start) & (df['time_only'] <= market_end)].copy()
    df_filtered = df_filtered.drop('time_only', axis=1)
    
    print(f"[DEBUG] Market hours filter: {market_start} to {market_end}")
    if len(df_filtered) > 0:
        print(f"[DEBUG] Time range in filtered data: {df_filtered['time'].min()} to {df_filtered['time'].max()}")
    
    return df_filtered


def calculate_vwap_over_time(df):
    """Calculate cumulative VWAP over time"""
    df = df.copy()
    df['turnover'] = df['combined_premium'] * df['combined_volume']
    df['cumulative_turnover'] = df['turnover'].cumsum()
    df['cumulative_volume'] = df['combined_volume'].cumsum()
    
    # Calculate VWAP at each point in time
    df['vwap'] = df['cumulative_turnover'] / df['cumulative_volume']
    
    # Replace any NaN or inf values with previous valid value
    df['vwap'] = df['vwap'].replace([np.inf, -np.inf], np.nan).ffill()
    
    return df


def get_atm_strike(spot_price, strike_step):
    """Round spot price to nearest strike"""
    return round(spot_price / strike_step) * strike_step


def get_current_price(symbol, config_path="config.json"):
    """Get current spot price and calculate ATM strike using IBBroker directly"""
    print(f"[DEBUG] get_current_price called with symbol={symbol}, config_path={config_path}")
    try:
        with open(config_path, 'r') as f:
            config = json.load(f)
        print(f"[DEBUG] Config loaded successfully")
    except FileNotFoundError:
        print(f"[DEBUG] ERROR: Config file not found")
        return None, None, None
    
    host = config.get("broker", {}).get("host", "127.0.0.1")
    port = config.get("broker", {}).get("port", 7497)
    exchange = config.get("underlying", {}).get("exchange", "SMART")
    strike_step = config.get("trade_parameters", {}).get("strike_step", 1)
    
    print(f"[DEBUG] Broker settings: host={host}, port={port}, exchange={exchange}, strike_step={strike_step}")
    
    # Use unique client ID
    client_id = get_unique_client_id()
    print(f"[DEBUG] Using client_id={client_id}")
    broker = IBBroker()
    
    try:
        # Connect to IBKR
        print(f"[DEBUG] Connecting to IBKR...")
        broker.connect_to_ibkr(host, port, client_id)
        time_mod.sleep(1)  # Give it time to connect
        print(f"[DEBUG] Connected to IBKR")
        
        # Get request ID
        req_id = 10000 + client_id  # Unique request ID
        print(f"[DEBUG] Requesting spot price with req_id={req_id}")
        
        # Get spot price
        spot_price = broker.get_index_spot(symbol, req_id, exchange)
        print(f"[DEBUG] Spot price received: {spot_price}")
        
        if spot_price is None:
            print(f"[DEBUG] ERROR: Spot price is None")
            return None, None, None
        
        atm_strike = get_atm_strike(spot_price, strike_step)
        print(f"[DEBUG] Success! spot_price={spot_price}, atm_strike={atm_strike}, strike_step={strike_step}")
        return spot_price, atm_strike, strike_step
    except Exception as e:
        print(f"[DEBUG] ERROR getting current price: {e}")
        import traceback
        print(f"[DEBUG] Traceback: {traceback.format_exc()}")
        return None, None, None
    finally:
        # Disconnect
        try:
            print(f"[DEBUG] Disconnecting broker...")
            broker.disconnect()
            print(f"[DEBUG] Disconnected")
        except Exception as e:
            print(f"[DEBUG] Error disconnecting: {e}")
            pass


def generate_graph(symbol, expiry, strike, config_path="config.json", duration="1 D", bar_size="1 min"):
    """
    Generate graph data for combined premium vs VWAP.
    Returns DataFrame with time, combined_premium, and vwap columns.
    """
    print(f"[DEBUG] generate_graph called: symbol={symbol}, expiry={expiry}, strike={strike}, duration={duration}, bar_size={bar_size}")
    # Load config
    try:
        with open(config_path, 'r') as f:
            config = json.load(f)
        print(f"[DEBUG] Config loaded successfully")
    except FileNotFoundError:
        print(f"[DEBUG] ERROR: Config file not found")
        raise ValueError(f"Config file '{config_path}' not found")
    
    # Get broker settings
    host = config.get("broker", {}).get("host", "127.0.0.1")
    port = config.get("broker", {}).get("port", 7497)
    exchange = config.get("underlying", {}).get("exchange", "SMART")
    strike_step = config.get("trade_parameters", {}).get("strike_step", 1)
    tz_str = config.get("time_controls", {}).get("timezone", "US/Eastern")
    
    print(f"[DEBUG] Broker settings: host={host}, port={port}, exchange={exchange}, strike_step={strike_step}")
    
    # Round strike to nearest strike step
    strike = round(strike / strike_step) * strike_step
    print(f"[DEBUG] Rounded strike: {strike}")
    
    # Use unique client ID
    client_id = get_unique_client_id()
    print(f"[DEBUG] Using client_id={client_id}")
    broker = IBBroker()
    
    try:
        # Connect to IBKR
        print(f"[DEBUG] Connecting to IBKR...")
        broker.connect_to_ibkr(host, port, client_id)
        time_mod.sleep(1)  # Give it time to connect
        print(f"[DEBUG] Connected to IBKR")
        
        # Get request IDs for OHLC data
        call_req_id = 20000 + client_id
        put_req_id = 30000 + client_id
        print(f"[DEBUG] Request IDs: call_req_id={call_req_id}, put_req_id={put_req_id}")
        
        # Fetch OHLC data for call and put at the specified strike
        print(f"[DEBUG] Fetching call OHLC data...")
        call_ohlc_data = broker.get_option_ohlc(
            symbol, expiry, strike, "C",
            duration=duration, bar_size=bar_size, req_id=call_req_id
        )
        print(f"[DEBUG] Call OHLC data received: {len(call_ohlc_data)} bars")
        
        print(f"[DEBUG] Fetching put OHLC data...")
        put_ohlc_data = broker.get_option_ohlc(
            symbol, expiry, strike, "P",
            duration=duration, bar_size=bar_size, req_id=put_req_id
        )
        print(f"[DEBUG] Put OHLC data received: {len(put_ohlc_data)} bars")
        
        # Convert to DataFrame
        call_ohlc = pd.DataFrame(call_ohlc_data)
        put_ohlc = pd.DataFrame(put_ohlc_data)
        print(f"[DEBUG] DataFrames created: call_ohlc shape={call_ohlc.shape}, put_ohlc shape={put_ohlc.shape}")
        
        if call_ohlc.empty or put_ohlc.empty:
            print(f"[DEBUG] ERROR: No OHLC data available")
            raise ValueError("No OHLC data available")
        
        # Merge call and put data on timestamp
        print(f"[DEBUG] Merging call and put data...")
        df = call_ohlc.merge(put_ohlc, on="time", suffixes=("_call", "_put"))
        print(f"[DEBUG] Merged dataframe shape: {df.shape}")
        
        if df.empty:
            print(f"[DEBUG] ERROR: No overlapping timestamps")
            raise ValueError("No overlapping timestamps between call and put data")
        
        # Filter to market hours
        print(f"[DEBUG] Filtering to market hours...")
        df = filter_market_hours(df, tz_str)
        print(f"[DEBUG] After market hours filter: {df.shape}")
        
        if df.empty:
            print(f"[DEBUG] ERROR: No data in market hours")
            raise ValueError("No data in market hours")
        
        # Calculate combined premium and volume
        # Combined premium = call close price + put close price (both at the same strike)
        print(f"[DEBUG] Calculating combined premium and volume...")
        df['combined_premium'] = df['close_call'] + df['close_put']
        df['combined_volume'] = df['volume_call'] + df['volume_put']
        
        # Filter out zero volume bars
        print(f"[DEBUG] Filtering zero volume bars...")
        df = df[df['combined_volume'] > 0]
        print(f"[DEBUG] After zero volume filter: {df.shape}")
        
        if df.empty:
            print(f"[DEBUG] ERROR: All bars have zero volume")
            raise ValueError("All bars have zero volume")
        
        # Sort by time
        print(f"[DEBUG] Sorting by time...")
        df = df.sort_values('time').reset_index(drop=True)
        
        # Calculate VWAP over time
        print(f"[DEBUG] Calculating VWAP over time...")
        df = calculate_vwap_over_time(df)
        print(f"[DEBUG] VWAP calculated. Final dataframe shape: {df.shape}")
        
        # Prepare data for plotting - return a simplified DataFrame
        print(f"[DEBUG] Preparing data for plotting...")
        
        # Convert time to a simpler format for display (just time, not full datetime)
        # Keep the full datetime for proper sorting, but create a display-friendly time column
        plot_df = pd.DataFrame({
            'time': df['time'],
            'Combined Premium': df['combined_premium'],
            'VWAP': df['vwap']
        })
        
        # Set time as index for streamlit plotting (using full datetime for proper ordering)
        plot_df = plot_df.set_index('time')
        
        # Rename index to 'Time' for better display
        plot_df.index.name = 'Time'
        
        # Ensure time is properly formatted for display
        print(f"[DEBUG] Time index type: {type(plot_df.index)}")
        print(f"[DEBUG] Time range: {plot_df.index.min()} to {plot_df.index.max()}")
        print(f"[DEBUG] First few time values: {plot_df.index[:5]}")
        
        print(f"[DEBUG] Plot dataframe prepared. Shape: {plot_df.shape}")
        print(f"[DEBUG] Plot dataframe columns: {plot_df.columns.tolist()}")
        print(f"[DEBUG] Time range: {plot_df.index.min()} to {plot_df.index.max()}")
        print(f"[DEBUG] Plot dataframe head:\n{plot_df.head()}")
        print(f"[DEBUG] Returning plot_df successfully")
        
        return plot_df
        
    except ValueError as ve:
        # Re-raise ValueError as-is (these are our expected errors)
        print(f"[DEBUG] ValueError in generate_graph: {str(ve)}")
        raise
    except Exception as e:
        print(f"[DEBUG] Unexpected ERROR in generate_graph: {str(e)}")
        import traceback
        print(f"[DEBUG] Traceback: {traceback.format_exc()}")
        raise ValueError(f"Error generating graph: {str(e)}")
    finally:
        # Disconnect - but don't raise any errors here!
        try:
            print(f"[DEBUG] Disconnecting broker...")
            broker.disconnect()
            print(f"[DEBUG] Disconnected")
        except Exception as e:
            print(f"[DEBUG] Error disconnecting: {e}")
            pass


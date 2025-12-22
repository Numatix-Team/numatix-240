import pandas as pd
import numpy as np
import matplotlib
import os
import sys

# Set backend based on whether running as standalone or imported
if __name__ == "__main__":
    # Running as standalone - use interactive backend
    matplotlib.use('TkAgg')  # Use TkAgg for Windows/Linux, works well for standalone
else:
    # Being imported - use non-interactive backend
    matplotlib.use('Agg')

import matplotlib.pyplot as plt
from pathlib import Path
from datetime import datetime
import pytz


def calculate_indicators_and_plot(hist_df, account="default", symbol="XSP", timezone_str="US/Eastern"):
    """
    Calculate indicators from hist_df and create a plot of combined_premium vs VWAP.
    
    Args:
        hist_df: DataFrame with OHLC data (must have combined_premium and combined_volume columns)
        account: Account name for filename
        symbol: Symbol name for filename
        timezone_str: Timezone string for timestamp
    """
    # Create a copy to avoid modifying original
    df = hist_df.copy()
    
    # Ensure numeric (very important with live feeds)
    df["combined_premium"] = pd.to_numeric(df["combined_premium"], errors="coerce")
    df["combined_volume"] = pd.to_numeric(df["combined_volume"], errors="coerce")
    
    # Turnover
    df["turnover"] = df["combined_premium"] * df["combined_volume"]
    
    # Cumulative values
    df["cum_turnover"] = df["turnover"].cumsum()
    df["cum_volume"] = df["combined_volume"].cumsum()
    
    # VWAP column (safe division)
    df["vwap"] = df["cum_turnover"] / df["cum_volume"].replace(0, pd.NA)
    
    # Total VWAP for session
    tot_vol = df["combined_volume"].sum()
    if tot_vol == 0:
        print("Error: Total volume is zero, cannot calculate VWAP")
        return None, None
    
    vwap_total = float(df["turnover"].sum() / tot_vol)
    print(f"[Aryama] Total VWAP={vwap_total:.2f}")
    
    # Save dataframe with calculated indicators to signals directory
    Path("signals").mkdir(exist_ok=True)
    tz = pytz.timezone(timezone_str)
    timestamp_str = datetime.now(tz).strftime("%Y%m%d_%H%M%S")
    indicators_file_path = f"signals/indicators_{account}_{symbol}_{timestamp_str}.csv"
    df.to_csv(indicators_file_path, index=False)
    print(f"[Aryama] Saved indicators CSV to: {indicators_file_path}")
    
    # Create plot: combined_premium vs VWAP
    plt.figure(figsize=(12, 6))
    
    # Plot combined_premium
    if "time" in df.columns:
        # If time column exists, use it as x-axis
        time_col = pd.to_datetime(df["time"], errors="coerce")
        plt.plot(time_col, df["combined_premium"], label="Combined Premium", linewidth=2, alpha=0.7)
        plt.plot(time_col, df["vwap"], label="VWAP", linewidth=2, alpha=0.7)
        plt.xlabel("Time")
    else:
        # Otherwise use index
        plt.plot(df.index, df["combined_premium"], label="Combined Premium", linewidth=2, alpha=0.7)
        plt.plot(df.index, df["vwap"], label="VWAP", linewidth=2, alpha=0.7)
        plt.xlabel("Index")
    
    plt.ylabel("Price")
    plt.title(f"Combined Premium vs VWAP - {account} | {symbol}")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    # Save plot to signals directory
    plot_file_path = f"signals/plot_{account}_{symbol}_{timestamp_str}.png"
    plt.savefig(plot_file_path, dpi=150, bbox_inches='tight')
    print(f"[Aryama] Saved plot to: {plot_file_path}")
    
    # Show plot - blocking if running standalone, non-blocking if imported
    if __name__ == "__main__":
        # Running standalone - show plot and keep it open
        plt.show(block=True)
    else:
        # Being imported - show non-blocking or just save
        try:
            plt.show(block=False)
            # Give it a moment to display, then close
            import time
            time.sleep(0.5)
            plt.close()
        except:
            plt.close()
    
    return df, vwap_total


if __name__ == "__main__":
    # Standalone script - load hist_df from CSV and plot
    import sys
    
    # Default file to use if none provided
    default_file = "hist_df.csv"
    
    if len(sys.argv) > 1:
        csv_path = sys.argv[1]
    else:
        # Try default file
        if os.path.exists(default_file):
            csv_path = default_file
            print(f"[Aryama] No file specified, using default: {default_file}")
        else:
            print("Usage: python aryama.py <csv_file_path> [account] [symbol]")
            print(f"Example: python aryama.py {default_file} default XSP")
            print(f"\nOr just run: python aryama.py (will use {default_file} if it exists)")
            sys.exit(1)
    
    if not os.path.exists(csv_path):
        print(f"Error: File not found: {csv_path}")
        sys.exit(1)
    
    print(f"[Aryama] Loading data from: {csv_path}")
    try:
        hist_df = pd.read_csv(csv_path)
        print(f"[Aryama] Loaded {len(hist_df)} rows")
    except Exception as e:
        print(f"Error loading CSV: {e}")
        sys.exit(1)
    
    # Check required columns
    required_cols = ["combined_premium", "combined_volume"]
    missing_cols = [col for col in required_cols if col not in hist_df.columns]
    if missing_cols:
        print(f"Error: Missing required columns: {missing_cols}")
        print(f"Available columns: {list(hist_df.columns)}")
        sys.exit(1)
    
    # Extract account and symbol from filename if possible, otherwise use defaults
    account = "default"
    symbol = "XSP"
    if len(sys.argv) > 2:
        account = sys.argv[2]
    if len(sys.argv) > 3:
        symbol = sys.argv[3]
    
    print(f"[Aryama] Processing with account={account}, symbol={symbol}")
    df_result, vwap = calculate_indicators_and_plot(hist_df, account=account, symbol=symbol)
    
    if df_result is not None:
        print(f"\n[Aryama] Summary:")
        print(f"  Total VWAP: {vwap:.2f}")
        print(f"  Data points: {len(df_result)}")
        print(f"  Combined Premium range: {df_result['combined_premium'].min():.2f} - {df_result['combined_premium'].max():.2f}")
        print(f"  VWAP range: {df_result['vwap'].min():.2f} - {df_result['vwap'].max():.2f}")
        print(f"\n[Aryama] Plot window should be displayed. Close it when done viewing.")
    else:
        print("[Aryama] Failed to calculate indicators")
        sys.exit(1)


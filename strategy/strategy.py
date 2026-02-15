import os
import sys
from pathlib import Path

_root = Path(__file__).resolve().parent.parent
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

import math
import json
import csv
import time as time_mod
from datetime import datetime
import pandas as pd
import pytz
from helpers.positions import (
    parse_time_string,
    get_db,
    get_position_by_id,
    create_position_entry,
    update_position_in_db,
    save_active_ids,
    load_active_ids,
)

class Strategy:

    def __init__(self, manager, broker, account, config_path="config.json", strike_offset=0, base_atm_price=None, symbol=None, ibkr_account_id=None):
        self.manager = manager
        self.broker = broker
        self.config_path = config_path
        self.account = account
        self.ibkr_account_id = ibkr_account_id if ibkr_account_id is not None else account
        self.strike_offset = strike_offset
        self.base_atm_price = base_atm_price
        self.symbol_override = symbol

        # runtime active IDs
        self.atm_call_id = None
        self.atm_put_id = None
        self.otm_call_id = None
        self.otm_put_id = None

        self.position_open = False
        self.position = None

        self.load_config(config_path)
        
        # Override symbol if provided (from frontend)
        if self.symbol_override:
            self.symbol = self.symbol_override

        self.atm = None
        self.call_strike = None
        self.put_strike = None
        self.hedge_call_strike = None
        self.hedge_put_strike = None
        self.vwap = None
        self.hist_df = pd.DataFrame()

        # Database is initialized automatically, no need to create JSON file

        self._load_active_ids()

    # CONFIG LOADING
    def load_config(self, path):
        with open(path, "r") as f:
            cfg = json.load(f)

        self.symbol = cfg["underlying"]["symbol"]
        self.exchange = cfg["underlying"]["exchange"]
        self.currency = cfg["underlying"]["currency"]
        self.trading_class = cfg["underlying"].get("trading_class", "")
        self.multiplier = int(cfg["underlying"].get("multiplier", 100))

        self.expiry = cfg["expiry"]["date"]

        tp = cfg["trade_parameters"]
        self.call_qty = tp["call_quantity"]
        self.put_qty = tp["put_quantity"]
        self.entry_vwap_mult = tp["entry_vwap_multiplier"]
        self.exit_vwap_mult = tp["exit_vwap_multiplier"]
        self.tp_levels = []
        for level in tp.get("take_profit_levels", []):
            self.tp_levels.append({
                "pnl": float(level["pnl_percent"]),
                "exit_frac": float(level["exit_percent"]),
                "done": False
            })
        self.sl_pct = tp["stop_loss"]
        self.max_spread = tp["max_bid_ask_spread"]
        self.strike_step = tp["strike_step"]

        tc = cfg["time_controls"]
        self.entry_start = parse_time_string(tc["entry_start"])
        self.entry_end = parse_time_string(tc["entry_end"])
        self.force_exit_time = parse_time_string(tc["force_exit_time"])
        self.tz = pytz.timezone(tc["timezone"])

        h = cfg["hedging"]
        self.enable_hedges = h["enable_hedges"]
        self.hedge_call_offset = h["hedge_call_offset"]
        self.hedge_put_offset = h["hedge_put_offset"]
        self.hedge_qty = h["hedge_quantity"]

        print("[Strategy] Config reloaded.")

        self.init_call_qty = self.call_qty
        self.init_put_qty  = self.put_qty
        self.init_hedge_qty = self.hedge_qty if self.enable_hedges else 0

        self.curr_call_qty = self.call_qty
        self.curr_put_qty  = self.put_qty
        self.curr_hedge_qty = self.hedge_qty if self.enable_hedges else 0   

    # RESTORE ACTIVE POSITIONS
    def _load_active_ids(self):
        """Load active positions that match the current strategy's strike prices."""
        # Calculate strikes using base_atm_price, strike_offset, and strike_step
        if self.base_atm_price is not None:
            # base_atm_price is already rounded to nearest strike_step
            atm_with_offset = self.base_atm_price + self.strike_offset * self.strike_step
            # Call and put strikes are the same as ATM (no offset)
            self.call_strike = atm_with_offset
            self.put_strike = atm_with_offset
            
            if self.enable_hedges:
                # Hedges use current ATM price (base_atm_price) with hedge offset
                self.hedge_call_strike = self.base_atm_price + self.hedge_call_offset * self.strike_step
                self.hedge_put_strike = self.base_atm_price + self.hedge_put_offset * self.strike_step
            else:
                self.hedge_call_strike = None
                self.hedge_put_strike = None
            
            print(f"[LOAD] Calculated strikes: Call={self.call_strike}, Put={self.put_strike}")
            if self.enable_hedges:
                print(f"[LOAD] Hedge strikes: Call={self.hedge_call_strike}, Put={self.hedge_put_strike}")
        else:
            # If base_atm_price is not set, we can't calculate strikes yet
            # Will be calculated in fetch_data() later
            print(f"[LOAD] base_atm_price not set, skipping strike-based position loading")
            self.position_open = False
            self.atm_call_id = None
            self.atm_put_id = None
            self.otm_call_id = None
            self.otm_put_id = None
            return
        
        db = get_db(self.account, self.symbol)
        all_active = db.get_active_positions(self.account, self.symbol)
        
        # Filter positions by matching strikes
        atm_call_id = None
        atm_put_id = None
        otm_call_id = None
        otm_put_id = None
        
        for pos in all_active:
            pos_strike = pos.get("strike")
            pos_type = pos.get("position_type", "")
            pos_right = pos.get("right", "")
            
            # Check ATM call
            if pos_type == "ATM" and pos_right == "C" and pos_strike == self.call_strike:
                atm_call_id = pos["id"]
                print(f"[LOAD] Found matching ATM Call position: ID={atm_call_id}, Strike={pos_strike}")
            
            # Check ATM put
            elif pos_type == "ATM" and pos_right == "P" and pos_strike == self.put_strike:
                atm_put_id = pos["id"]
                print(f"[LOAD] Found matching ATM Put position: ID={atm_put_id}, Strike={pos_strike}")
            
            # Check hedge call (if enabled)
            elif self.enable_hedges and pos_type == "OTM" and pos_right == "C" and pos_strike == self.hedge_call_strike:
                otm_call_id = pos["id"]
                print(f"[LOAD] Found matching Hedge Call position: ID={otm_call_id}, Strike={pos_strike}")
            
            # Check hedge put (if enabled)
            elif self.enable_hedges and pos_type == "OTM" and pos_right == "P" and pos_strike == self.hedge_put_strike:
                otm_put_id = pos["id"]
                print(f"[LOAD] Found matching Hedge Put position: ID={otm_put_id}, Strike={pos_strike}")
        
        # Set the loaded IDs
        self.atm_call_id = atm_call_id
        self.atm_put_id = atm_put_id
        self.otm_call_id = otm_call_id
        self.otm_put_id = otm_put_id
        
        # Position is open if we have at least one ATM position
        self.position_open = (atm_call_id is not None) or (atm_put_id is not None)
        
        if self.position_open:
            self.position = {
                "atm_call_id": self.atm_call_id,
                "atm_put_id": self.atm_put_id,
                "otm_call_id": self.otm_call_id,
                "otm_put_id": self.otm_put_id
            }
            
            # Restore quantities from database positions
            if self.atm_call_id:
                call_pos = get_position_by_id(self.atm_call_id, self.account, self.symbol)
                if call_pos:
                    self.curr_call_qty = call_pos.get("qty", 0)
                    self.init_call_qty = self.curr_call_qty  # Use current as initial if restoring
            
            if self.atm_put_id:
                put_pos = get_position_by_id(self.atm_put_id, self.account, self.symbol)
                if put_pos:
                    self.curr_put_qty = put_pos.get("qty", 0)
                    self.init_put_qty = self.curr_put_qty  # Use current as initial if restoring
            
            if self.enable_hedges:
                if self.otm_call_id:
                    hedge_call_pos = get_position_by_id(self.otm_call_id, self.account, self.symbol)
                    if hedge_call_pos:
                        self.curr_hedge_qty = hedge_call_pos.get("qty", 0)
                        self.init_hedge_qty = self.curr_hedge_qty  # Use current as initial if restoring
                elif self.otm_put_id:
                    # If only one hedge exists, use its quantity
                    hedge_put_pos = get_position_by_id(self.otm_put_id, self.account, self.symbol)
                    if hedge_put_pos:
                        self.curr_hedge_qty = hedge_put_pos.get("qty", 0)
                        self.init_hedge_qty = self.curr_hedge_qty
            
            print(f"[LOAD] Active positions loaded: ATM_C={atm_call_id}, ATM_P={atm_put_id}, OTM_C={otm_call_id}, OTM_P={otm_put_id}")
            print(f"[LOAD] Restored quantities: Call={self.curr_call_qty}, Put={self.curr_put_qty}, Hedge={self.curr_hedge_qty if self.enable_hedges else 0}")
        else:
            print(f"[LOAD] No matching active positions found for strikes: Call={self.call_strike}, Put={self.put_strike}")
            self.position = None
    

    # FETCH MARKET DATA
   
    def fetch_data(self):
        # -------------------------------
        # 1) Use provided base ATM price (already rounded in StrategyManager)
        # -------------------------------
        if self.base_atm_price is not None:
            # base_atm_price is already rounded to nearest strike_step in StrategyManager
            # Use it directly without re-rounding
            atm = self.base_atm_price
        else:
            # Fallback: fetch from broker (for backward compatibility only)
            spot = self.broker.current_price(self.symbol, self.exchange)
            if spot is None:
                print("[Strategy] Spot unavailable.")
                return False
            # Round to nearest strike step
            atm = round(spot / self.strike_step) * self.strike_step

        # -------------------------------
        # 2) Compute strikes with offset
        # -------------------------------
        # Apply strike offset: strike = ATM + strike_offset
        atm_with_offset = atm + self.strike_offset * self.strike_step
        self.atm = atm_with_offset
        # Call and put strikes are the same as ATM (no offset)
        self.call_strike = atm_with_offset
        self.put_strike = atm_with_offset

        if self.enable_hedges:
            # Hedges use current ATM price (atm) with hedge offset
            self.hedge_call_strike = atm + self.hedge_call_offset * self.strike_step
            self.hedge_put_strike = atm + self.hedge_put_offset * self.strike_step

        print(f"[Strategy] Offset={self.strike_offset}, Base_ATM={atm}, ATM={atm_with_offset}, CALL={self.call_strike}, PUT={self.put_strike}")

        # -------------------------------
        # 3) Fetch OHLC data for call & put
        # -------------------------------
        c_ohlc = self.broker.get_option_ohlc(self.symbol, self.expiry, self.call_strike, "C")
        p_ohlc = self.broker.get_option_ohlc(self.symbol, self.expiry, self.put_strike, "P")

        if c_ohlc.empty or p_ohlc.empty:
            print("[Strategy] Missing OHLC")
            return False

        # -------------------------------
        # 5) Merge call & put on timestamp
        # -------------------------------
        df = c_ohlc.merge(p_ohlc, on="time", suffixes=("_call", "_put"))

        if df.empty:
            print("[Strategy] No OHLC data inside market hours.")
            return False

        # -------------------------------
        # 8) Compute combined premium + combined volume
        # -------------------------------
        df["combined_premium"] = df["close_call"] + df["close_put"]
        df["combined_volume"] = df["volume_call"] + df["volume_put"]

        df = df[df["combined_volume"] > 0]

        if df.empty:
            print("[Strategy] All OHLC bars had zero volume.")
            return False

        # -------------------------------
        # 9) Save final dataframe
        # -------------------------------
        self.hist_df = df
       
        return True

    # VWAP
    # def calculate_indicators(self):
    #     df = self.hist_df
    #     df.to_csv("hist_df.csv")
    #     print(df["combined_volume"].sum())
    #     df["turnover"] = df["combined_premium"] * df["combined_volume"]
    #     tot_vol = df["combined_volume"].sum()
    #     if tot_vol == 0:
    #         return False
    #     self.vwap = float(df["turnover"].sum() / tot_vol)
    #     print(f"[Strategy] VWAP={self.vwap:.2f}")
        
    #     return True
    def calculate_indicators(self):
        df = self.hist_df.copy()
        # Save raw data (debugging / plotting)
        df.to_csv("hist_df.csv", index=False)
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
            return False

        self.vwap = float(df["turnover"].sum() / tot_vol)

        print(f"[Strategy] VWAP={self.vwap:.2f}")

        # # Push back to class
        # self.hist_df = df
        # self.hist_df.to_csv("check_Arya.csv", index=False)

        return True
    # Utility
    def _extract_price(self, d):
        if not d:
            return None
        if d.get("mid") is not None:
            return d["mid"]
        bid, ask = d.get("bid"), d.get("ask")
        if bid is not None and ask is not None:
            return (bid + ask) / 2
        return d.get("last") or bid or ask

    def _spread(self, d):
        if not d or d.get("bid") is None or d.get("ask") is None:
            return None
        return d["ask"] - d["bid"]

    #generates signals 
    def generate_signals(self):
        now = datetime.now(self.tz).time()
        if not (self.entry_start <= now <= self.entry_end):
            # Log signal even when outside entry window
            self.log_signal("OUTSIDE_ENTRY_WINDOW", None)
            return {"action": "NONE"}
        print(f"[SIGNAL] Fetching OHLC data for signal generation...")
        print(f"[SIGNAL]   CALL: {self.symbol} {self.expiry} {self.call_strike}C")
        call_ohlc = self.broker.get_option_ohlc(self.symbol, self.expiry, self.call_strike, "C")
        print(f"[SIGNAL]   PUT: {self.symbol} {self.expiry} {self.put_strike}P")
        put_ohlc = self.broker.get_option_ohlc(self.symbol, self.expiry, self.put_strike, "P")
        
        # Get latest data from OHLC (data is already sorted chronologically from IBKR)
        if call_ohlc.empty or put_ohlc.empty:
            self.log_signal("NONE", None)
            return {"action": "NONE"}
        
        # Get the latest bar (most recent - last row)
        latest_call = call_ohlc.iloc[-1]
        latest_put = put_ohlc.iloc[-1]
        
        c_price = latest_call['close']
        p_price = latest_put['close']
        print(f"[SIGNAL] Latest call close: {c_price}, Latest put close: {p_price}")

        if c_price is None or p_price is None or pd.isna(c_price) or pd.isna(p_price):
            self.log_signal("NONE", None)
            return {"action": "NONE"}

        combined = c_price + p_price
        print(f"[SIGNAL] Combined premium (from OHLC): {combined}")

        # Fetch latest bid/ask to check spread
        print(f"[SIGNAL] Fetching latest bid/ask for spread check...")
        call_premium = self.broker.get_option_premium(self.symbol, self.expiry, self.call_strike, "C")
        put_premium = self.broker.get_option_premium(self.symbol, self.expiry, self.put_strike, "P")
        
        if call_premium is None or put_premium is None:
            print(f"[SIGNAL] Could not fetch bid/ask data for spread check")
            self.log_signal("NONE", combined)
            return {"action": "NONE"}
        
        call_bid = call_premium.get("bid")
        call_ask = call_premium.get("ask")
        put_bid = put_premium.get("bid")
        put_ask = put_premium.get("ask")
        
        # Check if we have valid bid/ask data
        if call_bid is None or call_ask is None or put_bid is None or put_ask is None:
            print(f"[SIGNAL] Missing bid/ask data for spread check")
            self.log_signal("NONE", combined)
            return {"action": "NONE"}
        
        # Calculate spreads separately
        call_spread = call_ask - call_bid
        put_spread = put_ask - put_bid
        
        print(f"[SIGNAL] Call spread: {call_spread:.4f}, Put spread: {put_spread:.4f}, Max allowed: {self.max_spread:.4f}")
        
        # Check if either spread is too wide (check separately)
        if call_spread > self.max_spread:
            print(f"[SIGNAL] Call spread too wide: {call_spread:.4f} > {self.max_spread:.4f}")
            self.log_signal("NONE", combined)
            return {"action": "NONE"}
        
        if put_spread > self.max_spread:
            print(f"[SIGNAL] Put spread too wide: {put_spread:.4f} > {self.max_spread:.4f}")
            self.log_signal("NONE", combined)
            return {"action": "NONE"}
        
        # Get latest premium from bid/ask (mid price) for logging
        call_mid = (call_bid + call_ask) / 2
        put_mid = (put_bid + put_ask) / 2
        latest_combined_premium = call_mid + put_mid
        print(f"[SIGNAL] Latest premium (bid/ask mid): {latest_combined_premium:.4f}")

        threshold = self.vwap * self.entry_vwap_mult

        if not self.position_open and combined < threshold:
            # Log signal when SELL_STRADDLE is triggered
            self.log_signal("SELL_STRADDLE", combined)
            return {"action": "SELL_STRADDLE", "combined": combined}

        # Log signal when no action is taken
        self.log_signal("NONE", combined)
        return {"action": "NONE"}

    # ORDER EXECUTION
    def execute_trade(self, signal):
        combined = signal["combined"]
        print(f"[Strategy] SELL STRADDLE @ {combined}")
        
        # Track filled legs for rollback if needed
        filled_legs = []
        
        try:
            # Hedges
            if self.enable_hedges:
                hc = self.broker.place_option_market_order(
                    self.symbol, self.expiry, self.hedge_call_strike, "C", self.hedge_qty, "BUY", self.exchange
                )
                fill_hc = hc.get("fill_price")
                if fill_hc is None:
                    print("[EXECUTE] Call hedge failed to fill - closing any filled legs")
                    self._close_filled_legs(filled_legs)
                    return

                self.otm_call_id = create_position_entry(
                    self.account, self.symbol, self.expiry, self.hedge_call_strike, "C",
                    "BUY", self.hedge_qty,
                    fill_hc, hc["order_id"], "OTM", self.tz,
                    bid=0, ask=0, strike_offset=self.strike_offset
                )
                filled_legs.append((self.otm_call_id, {
                    "strike": self.hedge_call_strike, "right": "C", "qty": self.hedge_qty, "side": "BUY"
                }, self.hedge_qty, "BUY"))

                hp = self.broker.place_option_market_order(
                    self.symbol, self.expiry, self.hedge_put_strike, "P", self.hedge_qty, "BUY", self.exchange
                )
                fill_hp = hp.get("fill_price")
                if fill_hp is None:
                    print("[EXECUTE] Put hedge failed to fill - closing any filled legs")
                    self._close_filled_legs(filled_legs)
                    return

                self.otm_put_id = create_position_entry(
                    self.account, self.symbol, self.expiry, self.hedge_put_strike, "P",
                    "BUY", self.hedge_qty,
                    fill_hp, hp["order_id"], "OTM", self.tz,
                    bid=0, ask=0, strike_offset=self.strike_offset
                )
                filled_legs.append((self.otm_put_id, {
                    "strike": self.hedge_put_strike, "right": "P", "qty": self.hedge_qty, "side": "BUY"
                }, self.hedge_qty, "BUY"))

            # ATM CALL
            c = self.broker.place_option_market_order(
                self.symbol, self.expiry, self.call_strike, "C", self.call_qty, "SELL", self.exchange
            )
            fill_call = c.get("fill_price")
            print(f"[EXECUTE] ATM Call fill price: {fill_call}")
            if fill_call is None:
                print("[EXECUTE] ATM Call failed to fill - closing any filled legs")
                self._close_filled_legs(filled_legs)
                return

            self.atm_call_id = create_position_entry(
                self.account, self.symbol, self.expiry, self.call_strike, "C",
                "SELL", self.call_qty,
                fill_call, c["order_id"], "ATM", self.tz,
                bid=0, ask=0
            )
            filled_legs.append((self.atm_call_id, {
                "strike": self.call_strike, "right": "C", "qty": self.call_qty, "side": "SELL"
            }, self.call_qty, "SELL"))

            # ATM PUT
            p = self.broker.place_option_market_order(
                self.symbol, self.expiry, self.put_strike, "P", self.put_qty, "SELL", self.exchange
            )
            fill_put = p.get("fill_price")
            print(f"[EXECUTE] ATM Put fill price: {fill_put}")
            if fill_put is None:
                print("[EXECUTE] ATM Put failed to fill - closing any filled legs")
                self._close_filled_legs(filled_legs)
                return

            self.atm_put_id = create_position_entry(
                self.account, self.symbol, self.expiry, self.put_strike, "P",
                "SELL", self.put_qty,
                fill_put, p["order_id"], "ATM", self.tz,
                bid=0, ask=0
            )
            filled_legs.append((self.atm_put_id, {
                "strike": self.put_strike, "right": "P", "qty": self.put_qty, "side": "SELL"
            }, self.put_qty, "SELL"))

            # All legs filled successfully
            self.position_open = True
            save_active_ids(
                True, self.atm_call_id, self.atm_put_id, self.otm_call_id, self.otm_put_id, self.account, self.symbol
            )

            self.init_call_qty = self.call_qty
            self.init_put_qty  = self.put_qty
            self.init_hedge_qty = self.hedge_qty if self.enable_hedges else 0

            self.curr_call_qty = self.call_qty
            self.curr_put_qty  = self.put_qty
            self.curr_hedge_qty = self.hedge_qty if self.enable_hedges else 0
            
        except Exception as e:
            print(f"[EXECUTE] Error during trade execution: {e} - closing any filled legs")
            self._close_filled_legs(filled_legs)
            raise

    def _close_filled_legs(self, filled_legs):
        """Close all legs that were successfully filled if entry failed"""
        if not filled_legs:
            return
        
        print(f"[EXECUTE] Closing {len(filled_legs)} filled leg(s) due to entry failure")
        
        for pos_id, contract_info, qty, original_side in filled_legs:
            try:
                # Determine opposite action to close
                # If we originally SELL, we need to BUY to close
                # If we originally BUY, we need to SELL to close
                close_action = "BUY" if original_side == "SELL" else "SELL"
                
                print(f"[EXECUTE] Closing leg: {close_action} {qty}x {contract_info['strike']}{contract_info['right']}")
                
                resp = self.broker.place_option_market_order(
                    self.symbol, self.expiry, contract_info["strike"], contract_info["right"],
                    qty, close_action, self.exchange
                )
                
                # Close the position in database
                if resp and resp.get("fill_price"):
                    self._close_leg(pos_id, resp)
                else:
                    print(f"[EXECUTE] Warning: Could not close leg {pos_id} - order may not have filled")
                    
            except Exception as e:
                print(f"[EXECUTE] Error closing leg {pos_id}: {e}")
        
        # Reset position tracking
        self.atm_call_id = None
        self.atm_put_id = None
        self.otm_call_id = None
        self.otm_put_id = None
        self.position_open = False
        self.curr_call_qty = 0
        self.curr_put_qty = 0
        self.curr_hedge_qty = 0
        
        for tp in self.tp_levels:
            tp["done"] = False

    # LIVE UPDATES
    def _update_live_leg(self, pos_id, test_mode=False, pnl_list=None):
        pos = get_position_by_id(pos_id, self.account, self.symbol)
        if pos is None:
            print(f"[UPDATE] Position {pos_id} not found in database")
            return

        qty = pos.get("qty", 0)
        if qty <= 0:
            print(f"[UPDATE] Position {pos_id} has quantity 0, skipping update")
            return

        pos_type = pos.get("position_type", "UNKNOWN")
        right = pos.get("right", "")
        strike = pos.get("strike", 0)
        side = pos.get("side", "")
        symbol = pos.get("symbol", "")
        expiry = pos.get("expiry", "")
        print(f"[UPDATE] Updating {pos_type} {right} @ Strike {strike} ({side}) - ID: {pos_id[:20]}...")

        def _normalize_expiry(e):
            if not e:
                return ""
            return str(e).strip().replace("-", "").replace(" ", "")[:8]

        def _normalize_symbol(s):
            return (s or "").strip().upper()

        def _normalize_right(r):
            r = (r or "").strip().upper()
            if r in ("C", "CALL"):
                return "C"
            if r in ("P", "PUT"):
                return "P"
            return r

        # If we have PnL from IBKR, a match tells us this position is live at the broker.
        # We always compute unrealized PnL from *our* qty and entry_price (same contract may exist
        # before the bot, so broker's total PnL is for full size; we use our size only).
        match_found = False
        if pnl_list:
            pos_expiry_n = _normalize_expiry(expiry)
            pos_strike = float(strike) if strike is not None else 0
            pos_symbol_n = _normalize_symbol(symbol)
            pos_right_n = _normalize_right(right)
            for row in pnl_list:
                row_expiry = _normalize_expiry(row.get("expiry"))
                row_strike = float(row.get("strike", 0))
                row_symbol = _normalize_symbol(row.get("symbol"))
                row_right = _normalize_right(row.get("right"))
                if (pos_symbol_n == row_symbol
                    and pos_expiry_n == row_expiry
                    and abs(pos_strike - row_strike) < 0.01
                    and pos_right_n == row_right):
                    match_found = True
                    print(f"[UPDATE] Matched IBKR position (broker qty={row.get('pos')}); will use our qty and entry_price for unrealized PnL")
                    break

        # Fetch bid/ask for last_price (required to compute unrealized PnL from our qty and entry_price)
        try:
            data = self.broker.get_option_premium(
                pos["symbol"], pos["expiry"], pos["strike"], pos["right"], test_mode=test_mode
            )
        except Exception as e:
            print(f"[UPDATE] Broker error (will retry next cycle): {e}")
            return

        if data is None and not match_found:
            print(f"[UPDATE] SKIPPED: No data from IBKR and no PnL match for {pos_type} {right} @ Strike {strike}")
            return

        bid = data.get("bid") if data else None
        ask = data.get("ask") if data else None
        last = data.get("last") if data else None
        if data:
            print(f"[UPDATE] Received market data: bid={bid}, ask={ask}, last={last}")

        # Always compute unrealized PnL from our position's qty and entry_price (so pre-bot size is not included)
        entry = pos.get("entry_price")
        qty = pos.get("qty")
        if entry is None or qty is None:
            return
        if pos["side"] == "SELL":
            last_price = bid
        else:
            last_price = ask
        if last_price is None:
            print(f"[UPDATE] SKIPPED: No price for {pos_type} {right} @ Strike {strike}")
            return
        if pos["side"] == "SELL":
            unrealized_pnl = (entry - last_price) * qty * self.multiplier
        else:
            unrealized_pnl = (last_price - entry) * qty * self.multiplier

        # Only update unrealized PnL here; keep existing realized PnL from DB (do not overwrite from broker)
        existing_realized_pnl = pos.get("realized_pnl", 0.0) or 0.0

        pos["bid"] = bid
        pos["ask"] = ask
        pos["last_price"] = last
        pos["unrealized_pnl"] = unrealized_pnl
        pos["realized_pnl"] = existing_realized_pnl
        pos["last_update"] = datetime.now(self.tz).isoformat()
        
        # Update active flag based on quantity (always True here since we skip if qty <= 0)
        pos["active"] = True
        
        # Clear exit_time if it was set (position is active)
        if pos.get("exit_time"):
            pos["exit_time"] = None

        old_unrealized = pos.get("unrealized_pnl", 0.0) or 0.0
        bid_str = f"${bid:.2f}" if bid is not None else "N/A"
        ask_str = f"${ask:.2f}" if ask is not None else "N/A"
        last_str = f"${last:.2f}" if last is not None else "N/A"
        print(f"[UPDATE] Price Data: Bid={bid_str}, Ask={ask_str}, Last={last_str}")
        print(f"[UPDATE] PnL Summary: Realized=${existing_realized_pnl:.2f}, Unrealized=${unrealized_pnl:.2f} (was ${old_unrealized:.2f})")

        print(f"[UPDATE] Saving to database...")
        update_position_in_db(pos)
        print(f"[UPDATE] Database updated for {pos_type} {right} @ Strike {strike}")

    def _partial_exit_atm(self, exit_fraction, reason):
        print(f"[TP] Partial exit {exit_fraction*100:.0f}% → {reason}")

        # Get actual available quantities from database positions
        call_available_qty = 0
        put_available_qty = 0
        hedge_call_available_qty = 0
        hedge_put_available_qty = 0
        
        if self.atm_call_id:
            call_pos = get_position_by_id(self.atm_call_id, self.account, self.symbol)
            if call_pos:
                call_available_qty = call_pos.get("qty", 0)
        
        if self.atm_put_id:
            put_pos = get_position_by_id(self.atm_put_id, self.account, self.symbol)
            if put_pos:
                put_available_qty = put_pos.get("qty", 0)
        
        if self.enable_hedges:
            if self.otm_call_id:
                hedge_call_pos = get_position_by_id(self.otm_call_id, self.account, self.symbol)
                if hedge_call_pos:
                    hedge_call_available_qty = hedge_call_pos.get("qty", 0)
            
            if self.otm_put_id:
                hedge_put_pos = get_position_by_id(self.otm_put_id, self.account, self.symbol)
                if hedge_put_pos:
                    hedge_put_available_qty = hedge_put_pos.get("qty", 0)

        # Calculate exit qty from ORIGINAL qty
        call_exit_qty = int(math.ceil(self.init_call_qty * exit_fraction))
        put_exit_qty  = int(math.ceil(self.init_put_qty * exit_fraction))
        hedge_call_exit_qty = int(math.ceil(self.init_hedge_qty * exit_fraction)) if self.enable_hedges else 0
        hedge_put_exit_qty  = int(math.ceil(self.init_hedge_qty * exit_fraction)) if self.enable_hedges else 0

        # Cap by remaining qty (use both in-memory and database quantities for safety)
        call_exit_qty = min(call_exit_qty, self.curr_call_qty, call_available_qty)
        put_exit_qty  = min(put_exit_qty, self.curr_put_qty, put_available_qty)
        if self.enable_hedges:
            hedge_call_exit_qty = min(hedge_call_exit_qty, self.curr_hedge_qty, hedge_call_available_qty)
            hedge_put_exit_qty  = min(hedge_put_exit_qty, self.curr_hedge_qty, hedge_put_available_qty)

        if call_exit_qty <= 0 or put_exit_qty <= 0:
            print("[TP] Nothing left to exit")
            return
        
        # Validate that exit quantities don't exceed available quantities
        if call_exit_qty > call_available_qty:
            print(f"[TP] WARNING: Call exit qty ({call_exit_qty}) exceeds available qty ({call_available_qty}). Capping to available.")
            call_exit_qty = call_available_qty
        
        if put_exit_qty > put_available_qty:
            print(f"[TP] WARNING: Put exit qty ({put_exit_qty}) exceeds available qty ({put_available_qty}). Capping to available.")
            put_exit_qty = put_available_qty
        
        if self.enable_hedges:
            if hedge_call_exit_qty > hedge_call_available_qty:
                print(f"[TP] WARNING: Hedge Call exit qty ({hedge_call_exit_qty}) exceeds available qty ({hedge_call_available_qty}). Capping to available.")
                hedge_call_exit_qty = hedge_call_available_qty
            
            if hedge_put_exit_qty > hedge_put_available_qty:
                print(f"[TP] WARNING: Hedge Put exit qty ({hedge_put_exit_qty}) exceeds available qty ({hedge_put_available_qty}). Capping to available.")
                hedge_put_exit_qty = hedge_put_available_qty
        
        if call_exit_qty <= 0 or put_exit_qty <= 0:
            print("[TP] Nothing left to exit after validation")
            return

        # ---- CALL ----
        call_resp = self.broker.place_option_market_order(
            self.symbol, self.expiry, self.call_strike,
            "C", call_exit_qty, "BUY", self.exchange, True
        )

        # ---- PUT ----
        put_resp = self.broker.place_option_market_order(
            self.symbol, self.expiry, self.put_strike,
            "P", put_exit_qty, "BUY", self.exchange, True
        )

        # ---- HEDGE CALL (if enabled) ----
        hedge_call_resp = None
        if self.enable_hedges and hedge_call_exit_qty > 0:
            hedge_call_resp = self.broker.place_option_market_order(
                self.symbol, self.expiry, self.hedge_call_strike,
                "C", hedge_call_exit_qty, "SELL", self.exchange, True
            )

        # ---- HEDGE PUT (if enabled) ----
        hedge_put_resp = None
        if self.enable_hedges and hedge_put_exit_qty > 0:
            hedge_put_resp = self.broker.place_option_market_order(
                self.symbol, self.expiry, self.hedge_put_strike,
                "P", hedge_put_exit_qty, "SELL", self.exchange, True
            )

        # Update CURRENT quantities
        self.curr_call_qty -= call_exit_qty
        self.curr_put_qty  -= put_exit_qty
        if self.enable_hedges:
            self.curr_hedge_qty -= hedge_call_exit_qty  # Both hedges use same qty

        # Update database positions with new quantities and realized PnL
        if self.atm_call_id:
            call_pos = get_position_by_id(self.atm_call_id, self.account, self.symbol)
            if call_pos:
                entry = call_pos.get("entry_price")
                exit_price = call_resp.get("fill_price") if call_resp else None
                old_qty = call_pos.get("qty", 0)
                old_realized = call_pos.get("realized_pnl", 0) or 0
                
                print(f"[TP] CALL Position Update:")
                print(f"[TP]   Old Qty: {old_qty}, Exit Qty: {call_exit_qty}, New Qty: {self.curr_call_qty}")
                print(f"[TP]   Entry Price: ${entry:.2f}, Exit Price: ${exit_price:.2f}")
                
                if entry is not None and exit_price is not None:
                    # Calculate realized PnL for partial exit (SELL side): fill_price * multiplier * quantity
                    partial_realized_pnl = (entry - exit_price) * self.multiplier * call_exit_qty
                    new_realized = old_realized + partial_realized_pnl
                    call_pos["realized_pnl"] = new_realized
                    print(f"[TP]   Realized PnL: ${old_realized:.2f} + ${partial_realized_pnl:.2f} = ${new_realized:.2f}")
                    
                    # Calculate weighted average close price for multiple partial exits
                    previous_exit_qty = self.init_call_qty - old_qty  # Total quantity exited before this exit
                    new_exit_qty = call_exit_qty
                    existing_close_price = call_pos.get("close_price")
                    
                    if existing_close_price is not None and previous_exit_qty > 0:
                        # Weighted average: (old_avg * old_qty + new_price * new_qty) / total_exit_qty
                        total_exit_qty = previous_exit_qty + new_exit_qty
                        weighted_avg_close = (existing_close_price * previous_exit_qty + exit_price * new_exit_qty) / total_exit_qty
                        call_pos["close_price"] = weighted_avg_close
                        print(f"[TP]   Average Close Price: ${weighted_avg_close:.4f} (from ${previous_exit_qty} @ ${existing_close_price:.4f} + {new_exit_qty} @ ${exit_price:.4f})")
                    else:
                        # First partial exit, use current exit price
                        call_pos["close_price"] = exit_price
                        print(f"[TP]   Close Price: ${exit_price:.4f} (first partial exit)")
                    
                    # Update order_id_exit with latest order ID
                    call_order_id = call_resp.get("order_id") if call_resp else None
                    if call_order_id is not None:
                        call_pos["order_id_exit"] = int(call_order_id)
                        print(f"[TP]   Exit Order ID: {call_pos['order_id_exit']}")
                
                call_pos["qty"] = self.curr_call_qty
                call_pos["last_update"] = datetime.now(self.tz).isoformat()
                
                # Update active flag based on quantity (always set it, don't just check for <= 0)
                call_pos["active"] = (self.curr_call_qty > 0)
                
                if self.curr_call_qty <= 0:
                    call_pos["exit_time"] = datetime.now(self.tz).isoformat()
                    call_pos["unrealized_pnl"] = 0.0
                    print(f"[TP] CALL position fully closed (qty=0), marking as inactive")
                else:
                    # Clear exit_time if position is still active
                    if call_pos.get("exit_time"):
                        call_pos["exit_time"] = None
                    print(f"[TP] CALL position still active (qty={self.curr_call_qty})")
                
                # Recalculate unrealized PnL immediately with new quantity
                # Get current market price for recalculation
                if self.curr_call_qty > 0:
                    print(f"[TP]   Fetching current bid/ask prices for remaining CALL position...")
                    data = self.broker.get_option_premium(
                        call_pos["symbol"], call_pos["expiry"], call_pos["strike"], call_pos["right"], test_mode=False
                    )
                    if data:
                        bid = data.get("bid")
                        ask = data.get("ask")
                        last = data.get("last")
                        
                        # Use BID for SELL positions (call_pos is SELL side)
                        last_price = bid
                        
                        if last_price is not None:
                            # Recalculate unrealized PnL with remaining quantity (SELL side), with multiplier
                            old_unrealized = call_pos.get("unrealized_pnl", 0) or 0
                            call_pos["unrealized_pnl"] = (entry - last_price) * self.curr_call_qty * self.multiplier
                            call_pos["last_price"] = last  # Store actual last price in database
                            call_pos["bid"] = bid
                            call_pos["ask"] = ask
                            last_str = f"${last:.2f}" if last is not None else "N/A"
                            print(f"[TP]   Unrealized PnL: ${old_unrealized:.2f} → ${call_pos['unrealized_pnl']:.2f} (based on new qty {self.curr_call_qty})")
                            print(f"[TP]   Using Price: ${last_price:.2f} (bid) for PnL, Storing Last: {last_str} in DB")
                        else:
                            print(f"[TP] Could not get current price for unrealized PnL recalculation")
                    else:
                        print(f"[TP] No market data returned for unrealized PnL recalculation")
                print(f"[TP] Updating CALL position in database...")
                update_position_in_db(call_pos)
                print(f"[TP] CALL position updated in database")

        if self.atm_put_id:
            put_pos = get_position_by_id(self.atm_put_id, self.account, self.symbol)
            if put_pos:
                entry = put_pos.get("entry_price")
                exit_price = put_resp.get("fill_price") if put_resp else None
                old_qty = put_pos.get("qty", 0)
                old_realized = put_pos.get("realized_pnl", 0) or 0
                
                print(f"[TP] PUT Position Update:")
                print(f"[TP]   Old Qty: {old_qty}, Exit Qty: {put_exit_qty}, New Qty: {self.curr_put_qty}")
                print(f"[TP]   Entry Price: ${entry:.2f}, Exit Price: ${exit_price:.2f}")
                
                if entry is not None and exit_price is not None:
                    # Calculate realized PnL for partial exit (SELL side): fill_price * multiplier * quantity
                    partial_realized_pnl = (entry - exit_price) * self.multiplier * put_exit_qty
                    new_realized = old_realized + partial_realized_pnl
                    put_pos["realized_pnl"] = new_realized
                    print(f"[TP]   Realized PnL: ${old_realized:.2f} + ${partial_realized_pnl:.2f} = ${new_realized:.2f}")
                    
                    # Calculate weighted average close price for multiple partial exits
                    previous_exit_qty = self.init_put_qty - old_qty  # Total quantity exited before this exit
                    new_exit_qty = put_exit_qty
                    existing_close_price = put_pos.get("close_price")
                    
                    if existing_close_price is not None and previous_exit_qty > 0:
                        # Weighted average: (old_avg * old_qty + new_price * new_qty) / total_exit_qty
                        total_exit_qty = previous_exit_qty + new_exit_qty
                        weighted_avg_close = (existing_close_price * previous_exit_qty + exit_price * new_exit_qty) / total_exit_qty
                        put_pos["close_price"] = weighted_avg_close
                        print(f"[TP]   Average Close Price: ${weighted_avg_close:.4f} (from ${previous_exit_qty} @ ${existing_close_price:.4f} + {new_exit_qty} @ ${exit_price:.4f})")
                    else:
                        # First partial exit, use current exit price
                        put_pos["close_price"] = exit_price
                        print(f"[TP]   Close Price: ${exit_price:.4f} (first partial exit)")
                    
                    # Update order_id_exit with latest order ID
                    put_order_id = put_resp.get("order_id") if put_resp else None
                    if put_order_id is not None:
                        put_pos["order_id_exit"] = int(put_order_id)
                        print(f"[TP]   Exit Order ID: {put_pos['order_id_exit']}")
                
                put_pos["qty"] = self.curr_put_qty
                put_pos["last_update"] = datetime.now(self.tz).isoformat()
                
                # Update active flag based on quantity (always set it, don't just check for <= 0)
                put_pos["active"] = (self.curr_put_qty > 0)
                
                if self.curr_put_qty <= 0:
                    put_pos["exit_time"] = datetime.now(self.tz).isoformat()
                    put_pos["unrealized_pnl"] = 0.0
                    print(f"[TP] PUT position fully closed (qty=0), marking as inactive")
                else:
                    # Clear exit_time if position is still active
                    if put_pos.get("exit_time"):
                        put_pos["exit_time"] = None
                    print(f"[TP] PUT position still active (qty={self.curr_put_qty})")
                
                # Recalculate unrealized PnL immediately with new quantity
                # Get current market price for recalculation
                if self.curr_put_qty > 0:
                    print(f"[TP]   Fetching current bid/ask prices for remaining PUT position...")
                    data = self.broker.get_option_premium(
                        put_pos["symbol"], put_pos["expiry"], put_pos["strike"], put_pos["right"], test_mode=False
                    )
                    if data:
                        bid = data.get("bid")
                        ask = data.get("ask")
                        last = data.get("last")
                        
                        # Use BID for SELL positions (put_pos is SELL side)
                        last_price = bid
                        
                        if last_price is not None:
                            # Recalculate unrealized PnL with remaining quantity (SELL side), with multiplier
                            old_unrealized = put_pos.get("unrealized_pnl", 0) or 0
                            put_pos["unrealized_pnl"] = (entry - last_price) * self.curr_put_qty * self.multiplier
                            put_pos["last_price"] = last  # Store actual last price in database
                            put_pos["bid"] = bid
                            put_pos["ask"] = ask
                            last_str = f"${last:.2f}" if last is not None else "N/A"
                            print(f"[TP]   Unrealized PnL: ${old_unrealized:.2f} → ${put_pos['unrealized_pnl']:.2f} (based on new qty {self.curr_put_qty})")
                            print(f"[TP]   Using Price: ${last_price:.2f} (bid) for PnL, Storing Last: {last_str} in DB")
                        else:
                            print(f"[TP] Could not get current price for unrealized PnL recalculation")
                    else:
                        print(f"[TP] No market data returned for unrealized PnL recalculation")
                print(f"[TP] Updating PUT position in database...")
                update_position_in_db(put_pos)
                print(f"[TP] PUT position updated in database")

        # Update hedge positions with new quantities and realized PnL
        if self.enable_hedges:
            if self.otm_call_id and hedge_call_exit_qty > 0:
                hedge_call_pos = get_position_by_id(self.otm_call_id, self.account, self.symbol)
                if hedge_call_pos:
                    entry = hedge_call_pos.get("entry_price")
                    exit_price = hedge_call_resp.get("fill_price") if hedge_call_resp else None
                    old_qty = hedge_call_pos.get("qty", 0)
                    old_realized = hedge_call_pos.get("realized_pnl", 0) or 0
                    
                    print(f"[TP] HEDGE CALL Position Update:")
                    print(f"[TP]   Old Qty: {old_qty}, Exit Qty: {hedge_call_exit_qty}, New Qty: {self.curr_hedge_qty}")
                    print(f"[TP]   Entry Price: ${entry:.2f}, Exit Price: ${exit_price:.2f}")
                    
                    if entry is not None and exit_price is not None:
                        # Calculate realized PnL for partial exit (BUY side): fill_price * multiplier * quantity
                        partial_realized_pnl = (exit_price - entry) * self.multiplier * hedge_call_exit_qty
                        new_realized = old_realized + partial_realized_pnl
                        hedge_call_pos["realized_pnl"] = new_realized
                        print(f"[TP]   Realized PnL: ${old_realized:.2f} + ${partial_realized_pnl:.2f} = ${new_realized:.2f}")
                        
                        # Calculate weighted average close price for multiple partial exits
                        previous_exit_qty = self.init_hedge_qty - old_qty  # Total quantity exited before this exit
                        new_exit_qty = hedge_call_exit_qty
                        existing_close_price = hedge_call_pos.get("close_price")
                        
                        if existing_close_price is not None and previous_exit_qty > 0:
                            # Weighted average: (old_avg * old_qty + new_price * new_qty) / total_exit_qty
                            total_exit_qty = previous_exit_qty + new_exit_qty
                            weighted_avg_close = (existing_close_price * previous_exit_qty + exit_price * new_exit_qty) / total_exit_qty
                            hedge_call_pos["close_price"] = weighted_avg_close
                            print(f"[TP]   Average Close Price: ${weighted_avg_close:.4f} (from ${previous_exit_qty} @ ${existing_close_price:.4f} + {new_exit_qty} @ ${exit_price:.4f})")
                        else:
                            # First partial exit, use current exit price
                            hedge_call_pos["close_price"] = exit_price
                            print(f"[TP]   Close Price: ${exit_price:.4f} (first partial exit)")
                        
                        # Update order_id_exit with latest order ID
                        hedge_call_order_id = hedge_call_resp.get("order_id") if hedge_call_resp else None
                        if hedge_call_order_id is not None:
                            hedge_call_pos["order_id_exit"] = int(hedge_call_order_id)
                            print(f"[TP]   Exit Order ID: {hedge_call_pos['order_id_exit']}")
                    
                    hedge_call_pos["qty"] = self.curr_hedge_qty
                    hedge_call_pos["last_update"] = datetime.now(self.tz).isoformat()
                    
                    # Update active flag based on quantity (always set it, don't just check for <= 0)
                    hedge_call_pos["active"] = (self.curr_hedge_qty > 0)
                    
                    if self.curr_hedge_qty <= 0:
                        hedge_call_pos["exit_time"] = datetime.now(self.tz).isoformat()
                        hedge_call_pos["unrealized_pnl"] = 0.0
                        print(f"[TP] HEDGE CALL position fully closed (qty=0), marking as inactive")
                    else:
                        # Clear exit_time if position is still active
                        if hedge_call_pos.get("exit_time"):
                            hedge_call_pos["exit_time"] = None
                        print(f"[TP] HEDGE CALL position still active (qty={self.curr_hedge_qty})")
                    
                    # Recalculate unrealized PnL immediately with new quantity
                    if self.curr_hedge_qty > 0:
                        print(f"[TP]   Fetching current bid/ask prices for remaining HEDGE CALL position...")
                        data = self.broker.get_option_premium(
                            hedge_call_pos["symbol"], hedge_call_pos["expiry"], hedge_call_pos["strike"], hedge_call_pos["right"], test_mode=False
                        )
                        if data:
                            bid = data.get("bid")
                            ask = data.get("ask")
                            last = data.get("last")
                            
                            # Use ASK for BUY positions (hedge_call_pos is BUY side)
                            last_price = ask
                            
                            if last_price is not None:
                                # Recalculate unrealized PnL with remaining quantity (BUY side), with multiplier
                                old_unrealized = hedge_call_pos.get("unrealized_pnl", 0) or 0
                                hedge_call_pos["unrealized_pnl"] = (last_price - entry) * self.curr_hedge_qty * self.multiplier
                                hedge_call_pos["last_price"] = last
                                hedge_call_pos["bid"] = bid
                                hedge_call_pos["ask"] = ask
                                last_str = f"${last:.2f}" if last is not None else "N/A"
                                print(f"[TP]   Unrealized PnL: ${old_unrealized:.2f} → ${hedge_call_pos['unrealized_pnl']:.2f} (based on new qty {self.curr_hedge_qty})")
                                print(f"[TP]   Using Price: ${last_price:.2f} (ask) for PnL, Storing Last: {last_str} in DB")
                            else:
                                print(f"[TP] Could not get current price for unrealized PnL recalculation")
                        else:
                            print(f"[TP] No market data returned for unrealized PnL recalculation")
                    print(f"[TP] Updating HEDGE CALL position in database...")
                    update_position_in_db(hedge_call_pos)
                    print(f"[TP] HEDGE CALL position updated in database")

            if self.otm_put_id and hedge_put_exit_qty > 0:
                hedge_put_pos = get_position_by_id(self.otm_put_id, self.account, self.symbol)
                if hedge_put_pos:
                    entry = hedge_put_pos.get("entry_price")
                    exit_price = hedge_put_resp.get("fill_price") if hedge_put_resp else None
                    old_qty = hedge_put_pos.get("qty", 0)
                    old_realized = hedge_put_pos.get("realized_pnl", 0) or 0
                    
                    print(f"[TP] HEDGE PUT Position Update:")
                    print(f"[TP]   Old Qty: {old_qty}, Exit Qty: {hedge_put_exit_qty}, New Qty: {self.curr_hedge_qty}")
                    print(f"[TP]   Entry Price: ${entry:.2f}, Exit Price: ${exit_price:.2f}")
                    
                    if entry is not None and exit_price is not None:
                        # Calculate realized PnL for partial exit (BUY side): fill_price * multiplier * quantity
                        partial_realized_pnl = (exit_price - entry) * self.multiplier * hedge_put_exit_qty
                        new_realized = old_realized + partial_realized_pnl
                        hedge_put_pos["realized_pnl"] = new_realized
                        print(f"[TP]   Realized PnL: ${old_realized:.2f} + ${partial_realized_pnl:.2f} = ${new_realized:.2f}")
                        
                        # Calculate weighted average close price for multiple partial exits
                        previous_exit_qty = self.init_hedge_qty - old_qty  # Total quantity exited before this exit
                        new_exit_qty = hedge_put_exit_qty
                        existing_close_price = hedge_put_pos.get("close_price")
                        
                        if existing_close_price is not None and previous_exit_qty > 0:
                            # Weighted average: (old_avg * old_qty + new_price * new_qty) / total_exit_qty
                            total_exit_qty = previous_exit_qty + new_exit_qty
                            weighted_avg_close = (existing_close_price * previous_exit_qty + exit_price * new_exit_qty) / total_exit_qty
                            hedge_put_pos["close_price"] = weighted_avg_close
                            print(f"[TP]   Average Close Price: ${weighted_avg_close:.4f} (from ${previous_exit_qty} @ ${existing_close_price:.4f} + {new_exit_qty} @ ${exit_price:.4f})")
                        else:
                            # First partial exit, use current exit price
                            hedge_put_pos["close_price"] = exit_price
                            print(f"[TP]   Close Price: ${exit_price:.4f} (first partial exit)")
                        
                        # Update order_id_exit with latest order ID
                        hedge_put_order_id = hedge_put_resp.get("order_id") if hedge_put_resp else None
                        if hedge_put_order_id is not None:
                            hedge_put_pos["order_id_exit"] = int(hedge_put_order_id)
                            print(f"[TP]   Exit Order ID: {hedge_put_pos['order_id_exit']}")
                    
                    hedge_put_pos["qty"] = self.curr_hedge_qty
                    hedge_put_pos["last_update"] = datetime.now(self.tz).isoformat()
                    
                    # Update active flag based on quantity (always set it, don't just check for <= 0)
                    hedge_put_pos["active"] = (self.curr_hedge_qty > 0)
                    
                    if self.curr_hedge_qty <= 0:
                        hedge_put_pos["exit_time"] = datetime.now(self.tz).isoformat()
                        hedge_put_pos["unrealized_pnl"] = 0.0
                        print(f"[TP] HEDGE PUT position fully closed (qty=0), marking as inactive")
                    else:
                        # Clear exit_time if position is still active
                        if hedge_put_pos.get("exit_time"):
                            hedge_put_pos["exit_time"] = None
                        print(f"[TP] HEDGE PUT position still active (qty={self.curr_hedge_qty})")
                    
                    # Recalculate unrealized PnL immediately with new quantity
                    if self.curr_hedge_qty > 0:
                        print(f"[TP]   Fetching current bid/ask prices for remaining HEDGE PUT position...")
                        data = self.broker.get_option_premium(
                            hedge_put_pos["symbol"], hedge_put_pos["expiry"], hedge_put_pos["strike"], hedge_put_pos["right"], test_mode=False
                        )
                        if data:
                            bid = data.get("bid")
                            ask = data.get("ask")
                            last = data.get("last")
                            
                            # Use ASK for BUY positions (hedge_put_pos is BUY side)
                            last_price = ask
                            
                            if last_price is not None:
                                # Recalculate unrealized PnL with remaining quantity (BUY side), with multiplier
                                old_unrealized = hedge_put_pos.get("unrealized_pnl", 0) or 0
                                hedge_put_pos["unrealized_pnl"] = (last_price - entry) * self.curr_hedge_qty * self.multiplier
                                hedge_put_pos["last_price"] = last
                                hedge_put_pos["bid"] = bid
                                hedge_put_pos["ask"] = ask
                                last_str = f"${last:.2f}" if last is not None else "N/A"
                                print(f"[TP]   Unrealized PnL: ${old_unrealized:.2f} → ${hedge_put_pos['unrealized_pnl']:.2f} (based on new qty {self.curr_hedge_qty})")
                                print(f"[TP]   Using Price: ${last_price:.2f} (ask) for PnL, Storing Last: {last_str} in DB")
                            else:
                                print(f"[TP] Could not get current price for unrealized PnL recalculation")
                        else:
                            print(f"[TP] No market data returned for unrealized PnL recalculation")
                    print(f"[TP] Updating HEDGE PUT position in database...")
                    update_position_in_db(hedge_put_pos)
                    print(f"[TP] HEDGE PUT position updated in database")

        print(
            f"[TP] Remaining qty → CALL {self.curr_call_qty}, "
            f"PUT {self.curr_put_qty}"
        )
        if self.enable_hedges:
            print(f"[TP] Remaining hedge qty → {self.curr_hedge_qty}")
        
        # Check if both positions are fully closed and mark position as closed
        if self.curr_call_qty <= 0 and self.curr_put_qty <= 0:
            print("[TP] All quantities exited via partial close - marking position as closed")
            self.position_open = False
            # Update only THIS strategy's positions (not all positions in database)
            save_active_ids(False, self.atm_call_id, self.atm_put_id, self.otm_call_id, self.otm_put_id, self.account, self.symbol)


    # POSITION MANAGEMENT
    def manage_positions(self, poll_interval):
        while True:
            if self.manager.paused:
                print(f"[Strategy-{self.strike_offset}] Paused - waiting...")
                time_mod.sleep(5)
                continue

            print("\n---------------------------")
            print("[MANAGE] Managing Position")
            print("---------------------------")

            if not self.position_open:
                print("[MANAGE] No position open → return")
                return None

            # Request PnL for all open positions once per cycle (IBKR), then update each leg
            pnl_list = []
            try:
                pnl_list = self.broker.get_all_open_positions_pnl(account=self.ibkr_account_id)
            except Exception as e:
                print(f"[MANAGE] Could not fetch positions PnL list: {e}")
            # Update all legs (pass pnl_list so we can set PnL from IBKR when we find a match)
            print(f"[MANAGE] Updating all position legs...")
            if self.atm_call_id:
                print(f"[MANAGE] → Updating ATM CALL leg...")
                self._update_live_leg(self.atm_call_id, pnl_list=pnl_list)
            if self.atm_put_id:
                print(f"[MANAGE] → Updating ATM PUT leg...")
                self._update_live_leg(self.atm_put_id, pnl_list=pnl_list)
            if self.enable_hedges and self.otm_call_id:
                print(f"[MANAGE] → Updating OTM CALL hedge...")
                self._update_live_leg(self.otm_call_id, pnl_list=pnl_list)
            if self.enable_hedges and self.otm_put_id:
                print(f"[MANAGE] → Updating OTM PUT hedge...")
                self._update_live_leg(self.otm_put_id, pnl_list=pnl_list)
            print(f"[MANAGE] All legs updated")

            # Fetch updated ATM legs
            ac = get_position_by_id(self.atm_call_id, self.account, self.symbol)
            ap = get_position_by_id(self.atm_put_id, self.account, self.symbol)

            if not ac or not ap:
                print("[MANAGE] ATM legs not ready yet → waiting")
                time_mod.sleep(poll_interval)
                continue

            # Calculate total unrealized PnL (including hedges if enabled)
            unrealized_pnl_total = (ac.get("unrealized_pnl", 0) or 0) + (ap.get("unrealized_pnl", 0) or 0)
            
            # Calculate net entry value (total capital at risk)
            # For SELL positions: we received entry_price * qty
            # For BUY positions: we paid entry_price * qty
            # Net entry = what we received from SELL + what we paid for BUY (total capital at risk)
            sell_received = 0
            buy_paid = 0
            
            if ac.get("side") == "SELL":
                sell_received += ac["entry_price"] * ac.get("qty", 0)
            else:
                buy_paid += ac["entry_price"] * ac.get("qty", 0)
            
            if ap.get("side") == "SELL":
                sell_received += ap["entry_price"] * ap.get("qty", 0)
            else:
                buy_paid += ap["entry_price"] * ap.get("qty", 0)

            # Include hedges if enabled
            if self.enable_hedges:
                hc = get_position_by_id(self.otm_call_id, self.account, self.symbol) if self.otm_call_id else None
                hp = get_position_by_id(self.otm_put_id, self.account, self.symbol) if self.otm_put_id else None
                
                if hc:
                    unrealized_pnl_total += (hc.get("unrealized_pnl", 0) or 0)
                    if hc.get("side") == "SELL":
                        sell_received += hc["entry_price"] * hc.get("qty", 0)
                    else:
                        buy_paid += hc["entry_price"] * hc.get("qty", 0)
                    print(f"[MANAGE] Including hedge call: Entry=${hc['entry_price']:.2f}, Unrealized=${hc.get('unrealized_pnl', 0) or 0:.2f}")
                
                if hp:
                    unrealized_pnl_total += (hp.get("unrealized_pnl", 0) or 0)
                    if hp.get("side") == "SELL":
                        sell_received += hp["entry_price"] * hp.get("qty", 0)
                    else:
                        buy_paid += hp["entry_price"] * hp.get("qty", 0)
                    print(f"[MANAGE] Including hedge put: Entry=${hp['entry_price']:.2f}, Unrealized=${hp.get('unrealized_pnl', 0) or 0:.2f}")

            # Net entry in dollars = total capital at risk (unrealized_pnl_total is stored in dollars)
            net_entry_points = sell_received + buy_paid
            net_entry = (net_entry_points * self.multiplier) if net_entry_points > 0 else 1

            # PnL percentage: unrealized PnL (dollars) / net entry (dollars)
            pnl_pct = unrealized_pnl_total / net_entry if net_entry > 0 else 0

            print(f"[MANAGE] SELL Received: ${sell_received * self.multiplier:.2f}, BUY Paid: ${buy_paid * self.multiplier:.2f}")
            print(f"[MANAGE] Net Entry Value: ${net_entry:.2f} (total capital at risk, dollars)")
            print(f"[MANAGE] Combined Unrealized PnL: ${unrealized_pnl_total:.2f} USD "
                f"({pnl_pct*100:.2f}%)")

            now = datetime.now(self.tz).time()

            # --------------------
            # VWAP EXIT CHECK
            # --------------------
            if self.vwap is not None:
                vwap_exit_level = self.vwap * self.exit_vwap_mult

                # Calculate current combined premium for VWAP exit check (using bid for SELL positions)
                current_combined = ac.get("bid", ac.get("last_price", 0)) + ap.get("bid", ap.get("last_price", 0))
                
                print(f"[MANAGE] VWAP EXIT → current: {current_combined:.4f}, "
                    f"required: > {vwap_exit_level:.4f} "
                    f"(vwap={self.vwap:.4f}, mult={self.exit_vwap_mult})")

                if current_combined > vwap_exit_level:
                    print("[MANAGE] VWAP EXIT Triggered")
                    return self.exit_position("VWAP EXIT")
            else:
                print("[MANAGE] VWAP not calculated yet - skipping VWAP exit check")


            # --------------------
            # TAKE PROFIT CHECK
            # --------------------
            for tp in self.tp_levels:
                if tp["done"]:
                    continue

                print(
                    f"[MANAGE] TP CHECK → pnl: {pnl_pct*100:.2f}% "
                    f"required: {tp['pnl']*100:.0f}%"
                )

                if pnl_pct >= tp["pnl"]:
                    self._partial_exit_atm(tp["exit_frac"], f"TP {int(tp['pnl']*100)}%")
                    tp["done"] = True
                    break   # only one TP per cycle
            
            if self.curr_call_qty <= 0 and self.curr_put_qty <= 0:
                print("[MANAGE] All quantities exited via TP")
                self.position_open = False
                # Update only THIS strategy's positions (not all positions in database)
                save_active_ids(False, self.atm_call_id, self.atm_put_id, self.otm_call_id, self.otm_put_id, self.account, self.symbol)
                # Reset TP levels since all positions are closed
                for tp in self.tp_levels:
                    tp["done"] = False
                print("[MANAGE] TP levels reset - all positions closed")
                return {"exit_reason": "ALL TP COMPLETED"}

            # --------------------
            # STOP LOSS CHECK
            # --------------------
            print(f"[MANAGE] STOP LOSS → current pnl_pct: {pnl_pct*100:.2f}%, "
                f"required: ≤ {-self.sl_pct*100:.2f}%")

            if pnl_pct <= -self.sl_pct:
                print("[MANAGE] STOP LOSS Triggered")
                return self.exit_position("STOP LOSS")


            # --------------------
            # TIME EXIT
            # --------------------
            print(f"[MANAGE] TIME EXIT → now: {now}, cutoff: {self.force_exit_time}")

            if now >= self.force_exit_time:
                print("[MANAGE] FORCE EXIT Triggered")
                return self.exit_position("FORCED EXIT")

            print(f"[MANAGE] Sleeping {poll_interval} seconds...\n")
            time_mod.sleep(poll_interval)


    # EXIT
    def exit_position(self, reason):
        print(f"[Strategy] EXIT — {reason}")

        if self.atm_call_id and self.curr_call_qty > 0:
            resp = self.broker.place_option_market_order(
                self.symbol, self.expiry, self.call_strike, "C", self.curr_call_qty, "BUY", self.exchange, True
            )
            self._close_leg(self.atm_call_id, resp)

        if self.atm_put_id and self.curr_put_qty > 0:
            resp = self.broker.place_option_market_order(
                self.symbol, self.expiry, self.put_strike, "P", self.curr_put_qty, "BUY", self.exchange, True
            )
            self._close_leg(self.atm_put_id, resp)

        if self.enable_hedges:

            if self.otm_call_id and self.curr_hedge_qty > 0:
                resp = self.broker.place_option_market_order(
                    self.symbol, self.expiry, self.hedge_call_strike, "C", self.curr_hedge_qty, "SELL", self.exchange, True
                )
                self._close_leg(self.otm_call_id, resp)

            if self.otm_put_id and self.curr_hedge_qty > 0:
                resp = self.broker.place_option_market_order(
                    self.symbol, self.expiry, self.hedge_put_strike, "P", self.curr_hedge_qty, "SELL", self.exchange, True
                )
                self._close_leg(self.otm_put_id, resp)

        self.init_call_qty = 0
        self.init_put_qty = 0
        self.init_hedge_qty = 0
        self.curr_call_qty = 0
        self.curr_put_qty = 0
        self.curr_hedge_qty = 0

        # Update only THIS strategy's positions (not all positions in database)
        # Save the IDs before clearing them
        atm_call = self.atm_call_id
        atm_put = self.atm_put_id
        otm_call = self.otm_call_id
        otm_put = self.otm_put_id
        save_active_ids(False, atm_call, atm_put, otm_call, otm_put, self.account, self.symbol)

        self.atm_call_id = None
        self.atm_put_id = None
        self.otm_call_id = None
        self.otm_put_id = None
        self.position_open = False

        # Reset TP levels if stop loss is triggered
        if reason == "STOP LOSS":
            for tp in self.tp_levels:
                tp["done"] = False
            print("[EXIT] TP levels reset - stop loss triggered")

        return {"exit_reason": reason}

    # CLOSE LEG
    def _close_leg(self, pos_id, resp):
        """Close position using fill price from resp. Updates realized_pnl with (entry - fill_price) * qty * multiplier."""
        pos = get_position_by_id(pos_id, self.account, self.symbol)
        if pos is None:
            return

        close_price = resp.get("fill_price")  # Use fill price for realized PnL
        now = datetime.now(self.tz).isoformat()

        entry = pos["entry_price"]
        qty = pos["qty"]
        existing_realized_pnl = pos.get("realized_pnl", 0.0) or 0.0  # From partial exits (also used fill price)

        if entry is not None and close_price is not None:
            if pos["side"] == "SELL":
                remaining_realized_pnl = (entry - close_price) * self.multiplier * qty
            else:
                remaining_realized_pnl = (close_price - entry) * self.multiplier * qty
        else:
            remaining_realized_pnl = 0

        # Total realized PnL = existing (from partial exits) + remaining (from full close)
        total_realized_pnl = existing_realized_pnl + remaining_realized_pnl

        pos["active"] = False
        pos["exit_time"] = now
        pos["close_price"] = close_price
        order_id_exit = resp.get("order_id")
        pos["order_id_exit"] = int(order_id_exit) if order_id_exit is not None else None
        pos["qty"] = 0  # Set quantity to 0 when completely closed
        pos["realized_pnl"] = total_realized_pnl  # Total realized PnL including partial exits
        pos["unrealized_pnl"] = 0.0  # No unrealized PnL for closed positions
        pos["last_update"] = now
        
        print(f"[CLOSE] Closing {pos.get('position_type', '')} {pos.get('right', '')} @ Strike {pos.get('strike', 0)}")
        print(f"[CLOSE]   Entry: ${entry:.2f}, Close: ${close_price:.2f}, Qty: {qty}")
        print(f"[CLOSE]   Existing Realized: ${existing_realized_pnl:.2f}, Remaining Realized: ${remaining_realized_pnl:.2f}")
        print(f"[CLOSE]   Total Realized PnL: ${total_realized_pnl:.2f}")

        update_position_in_db(pos)

    def run(self, poll_interval=60):
        print("[Strategy] RUN LOOP STARTED")
        
        # Initialize signal log file by logging strategy start
        # This ensures the CSV file is created for all strategies, even if they fail early
        self.log_signal("STRATEGY_STARTED", None)

        while self.manager.keep_running:
            if self.manager.paused:
                print(f"[Strategy-{self.strike_offset}] Paused - waiting...")
                time_mod.sleep(5)
                continue

            try:
                # If a position is open → manage it (PnL updates; broker may block in ensure_connected if disconnected)
                if self.position_open:
                    if not self.fetch_data():
                        time_mod.sleep(poll_interval)
                        continue
                    if not self.calculate_indicators():
                        time_mod.sleep(poll_interval)
                        continue
                    self.manage_positions(10)
                    time_mod.sleep(poll_interval)
                    continue

                # No position → fetch data, indicators, look for signal
                if not self.fetch_data():
                    time_mod.sleep(poll_interval)
                    continue
                if not self.calculate_indicators():
                    time_mod.sleep(poll_interval)
                    continue
                signal = self.generate_signals()
                if signal["action"] == "SELL_STRADDLE":
                    self.execute_trade(signal)
                    continue
                time_mod.sleep(poll_interval)

            except Exception as e:
                print(f"[Strategy-{self.strike_offset}] Cycle error (will retry): {e}")
                time_mod.sleep(poll_interval)

    def log_signal(self, action, combined):
        """Save each signal to a CSV file identified by account, symbol, and offset in a date folder."""

        # Create signals folder if missing
        Path("signals").mkdir(exist_ok=True)

        # Use date-based folder
        date_str = datetime.now(self.tz).strftime("%Y-%m-%d")
        date_folder = Path("signals") / date_str
        date_folder.mkdir(exist_ok=True)

        # Create filename with account, symbol, and offset
        filename = f"{self.account}_{self.symbol}_{self.strike_offset}_signals.csv"
        file_path = date_folder / filename

        # Check if file exists (to write header only once)
        file_exists = os.path.isfile(file_path)

        with open(file_path, "a", newline="") as f:
            writer = csv.writer(f)

            # Write header if new file
            if not file_exists:
                writer.writerow(["timestamp", "action", "combined", "call_strike",
                                "put_strike", "strike_offset", "account", "symbol", "vwap"])

            writer.writerow([
                datetime.now(self.tz).isoformat(),
                action,
                combined if combined is not None else "",
                self.call_strike,
                self.put_strike,
                self.strike_offset, 
                self.account,  
                self.symbol,  
                self.vwap
            ])

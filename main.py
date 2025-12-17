import os
import json
import time
import random
import threading
from datetime import datetime, time as dt_time
from typing import Any, Dict, Optional
import pandas as pd
import pytz
import numpy as np
import time as time_mod
import csv
from pathlib import Path
import argparse
from broker.ib_broker import IBBroker
from log import setup_logger
from helpers.positions import *
from db.position_db import PositionDB
from helpers.state_manager import is_account_paused, is_account_stopped, set_account_stopped

setup_logger()

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--account",
        type=str,
        default="default",
        help="Account identifier"
    )

    parser.add_argument(
        "--symbol",
        type=str,
        default=None,
        help="Override trading symbol"
    )

    parser.add_argument(
        "--config",
        type=str,
        default="config.json",
        help="Path to config file"
    )

    parser.add_argument(
        "--range",
        type=str,
        default=None,
        help="Strike range as 'start:end' (e.g., '-2:2' for offsets -2 to 2)"
    )

    parser.add_argument(
        "--test",
        action="store_true",
        help="Run in test mode (executes test function in main thread only)"
    )

    return parser.parse_args()


class Strategy:

    def __init__(self, manager, broker, account, config_path="config.json", strike_offset=0, base_atm_price=None):
        self.manager = manager
        self.broker = broker
        self.config_path = config_path
        self.account = account
        self.strike_offset = strike_offset  # n value: strike = ATM + n
        self.base_atm_price = base_atm_price  # Base ATM price from broker

        # runtime active IDs
        self.atm_call_id = None
        self.atm_put_id = None
        self.otm_call_id = None
        self.otm_put_id = None

        self.position_open = False
        self.position = None

        self.load_config(config_path)

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

        self.expiry = cfg["expiry"]["date"]

        tp = cfg["trade_parameters"]
        self.call_qty = tp["call_quantity"]
        self.put_qty = tp["put_quantity"]
        self.atm_call_offset = tp["atm_call_offset"]
        self.atm_put_offset = tp["atm_put_offset"]
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

        self.curr_call_qty = self.call_qty
        self.curr_put_qty  = self.put_qty   

    # RESTORE ACTIVE POSITIONS
    def _load_active_ids(self):
        active = load_active_ids(self.account, self.symbol)

        self.position_open = active.get("position_open", False)
        self.atm_call_id = active.get("atm_call_id")
        self.atm_put_id = active.get("atm_put_id")
        self.otm_call_id = active.get("otm_call_id")
        self.otm_put_id = active.get("otm_put_id")

        if self.position_open:
            self.position = {
                "atm_call_id": self.atm_call_id,
                "atm_put_id": self.atm_put_id,
                "otm_call_id": self.otm_call_id,
                "otm_put_id": self.otm_put_id
            }

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
        self.call_strike = atm_with_offset + self.atm_call_offset * self.strike_step
        self.put_strike = atm_with_offset - self.atm_put_offset * self.strike_step

        if self.enable_hedges:
            self.hedge_call_strike = atm_with_offset + self.hedge_call_offset * self.strike_step
            self.hedge_put_strike = atm_with_offset - self.hedge_put_offset * self.strike_step

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
    def calculate_indicators(self):
        df = self.hist_df
        df["turnover"] = df["combined_premium"] * df["combined_volume"]
        tot_vol = df["combined_volume"].sum()
        if tot_vol == 0:
            return False
        self.vwap = float(df["turnover"].sum() / tot_vol)
        print(f"[Strategy] VWAP={self.vwap:.2f}")
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
            return {"action": "NONE"}
        cp = self.broker.get_option_premium(self.symbol, self.expiry, self.call_strike, "C")
        pp = self.broker.get_option_premium(self.symbol, self.expiry, self.put_strike, "P")
        c_price = self._extract_price(cp)
        p_price = self._extract_price(pp)
        print(c_price,p_price)

        if c_price is None or p_price is None:
            self.log_signal("NONE", None)
            return {"action": "NONE"}

        combined = c_price + p_price
        print(combined)

        if any([
            self._spread(cp) is None,
            self._spread(pp) is None,
            self._spread(cp) > self.max_spread,
            self._spread(pp) > self.max_spread
        ]):
            return {"action": "NONE"}

        threshold = self.vwap * self.entry_vwap_mult

        if not self.position_open and combined < threshold:
            return {"action": "SELL_STRADDLE", "combined": combined}

        self.log_signal("NONE", combined)
        return {"action": "NONE"}

    # ORDER EXECUTION
    def execute_trade(self, signal):
        combined = signal["combined"]
        print(f"[Strategy] SELL STRADDLE @ {combined}")
        
        # Track filled legs for rollback if needed
        filled_legs = []  # List of (position_id, contract_info, qty, side) tuples
        
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
                    bid=0, ask=0
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
                    bid=0, ask=0
                )
                filled_legs.append((self.otm_put_id, {
                    "strike": self.hedge_put_strike, "right": "P", "qty": self.hedge_qty, "side": "BUY"
                }, self.hedge_qty, "BUY"))

            # ATM CALL
            c = self.broker.place_option_market_order(
                self.symbol, self.expiry, self.call_strike, "C", self.call_qty, "SELL", self.exchange
            )
            fill_call = c.get("fill_price")
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

            self.curr_call_qty = self.call_qty
            self.curr_put_qty  = self.put_qty
            
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
        
        for tp in self.tp_levels:
            tp["done"] = False

    # LIVE UPDATES
    def _update_live_leg(self, pos_id, test_mode=False):
        pos = get_position_by_id(pos_id)
        if pos is None or not pos.get("active", False):
            return

        data = self.broker.get_option_premium(
            pos["symbol"], pos["expiry"], pos["strike"], pos["right"], test_mode=test_mode
        )

        if data is None:
            print(f"[UPDATE] No data returned for position {pos_id}, skipping update")
            return

        bid = data.get("bid")
        ask = data.get("ask")

        # Use last price by default, fallback to mid price if last is not available
        last_price = data.get("last")
        if last_price is None:
            # If last is not available, try mid price (calculated from bid/ask if available)
            if bid is not None and ask is not None:
                last_price = (bid + ask) / 2  # Mid price
            else:
                last_price = data.get("mid")  # Pre-calculated mid if available

        # Check if we have a valid price before proceeding
        if last_price is None:
            print(f"[UPDATE] No price data available for position {pos_id}, skipping update")
            return

        entry = pos.get("entry_price")
        qty = pos.get("qty")

        # Additional None checks for entry and qty
        if entry is None:
            print(f"[UPDATE] Entry price is None for position {pos_id}, skipping update")
            return
        
        if qty is None:
            print(f"[UPDATE] Quantity is None for position {pos_id}, skipping update")
            return

        # -----------------------------
        # Calculate unrealized PnL (only for remaining quantity)
        # -----------------------------
        if pos["side"] == "SELL":
            unrealized_pnl = (entry - last_price) * qty
        else:
            unrealized_pnl = (last_price - entry) * qty

        existing_realized_pnl = pos.get("realized_pnl", 0.0) or 0.0

        pos["bid"] = bid
        pos["ask"] = ask
        pos["last_price"] = last_price
        pos["unrealized_pnl"] = unrealized_pnl
        pos["realized_pnl"] = existing_realized_pnl  # Keep existing realized PnL from partial exits
        pos["last_update"] = datetime.now(self.tz).isoformat()

        update_position_in_json(pos)

    def _partial_exit_atm(self, exit_fraction, reason):
        print(f"[TP] Partial exit {exit_fraction*100:.0f}% → {reason}")

        # Calculate exit qty from ORIGINAL qty
        call_exit_qty = int(round(self.init_call_qty * exit_fraction))
        put_exit_qty  = int(round(self.init_put_qty * exit_fraction))

        # Cap by remaining qty
        call_exit_qty = min(call_exit_qty, self.curr_call_qty)
        put_exit_qty  = min(put_exit_qty, self.curr_put_qty)

        if call_exit_qty <= 0 or put_exit_qty <= 0:
            print("[TP] Nothing left to exit")
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

        # Update CURRENT quantities
        self.curr_call_qty -= call_exit_qty
        self.curr_put_qty  -= put_exit_qty

        # Update database positions with new quantities and realized PnL
        if self.atm_call_id:
            call_pos = get_position_by_id(self.atm_call_id)
            if call_pos:
                entry = call_pos.get("entry_price")
                exit_price = call_resp.get("fill_price") if call_resp else None
                
                if entry is not None and exit_price is not None:
                    # Calculate realized PnL for partial exit (SELL side)
                    partial_realized_pnl = (entry - exit_price) * call_exit_qty
                    call_pos["realized_pnl"] = (call_pos.get("realized_pnl", 0) or 0) + partial_realized_pnl
                
                call_pos["qty"] = self.curr_call_qty
                call_pos["last_update"] = datetime.now(self.tz).isoformat()
                update_position_in_json(call_pos)

        if self.atm_put_id:
            put_pos = get_position_by_id(self.atm_put_id)
            if put_pos:
                entry = put_pos.get("entry_price")
                exit_price = put_resp.get("fill_price") if put_resp else None
                
                if entry is not None and exit_price is not None:
                    # Calculate realized PnL for partial exit (SELL side)
                    partial_realized_pnl = (entry - exit_price) * put_exit_qty
                    put_pos["realized_pnl"] = (put_pos.get("realized_pnl", 0) or 0) + partial_realized_pnl
                
                put_pos["qty"] = self.curr_put_qty
                put_pos["last_update"] = datetime.now(self.tz).isoformat()
                update_position_in_json(put_pos)

        print(
            f"[TP] Remaining qty → CALL {self.curr_call_qty}, "
            f"PUT {self.curr_put_qty}"
        )


    # POSITION MANAGEMENT
    def manage_positions(self, poll_interval):
        while True:
            print("\n---------------------------")
            print("[MANAGE] Managing Position")
            print("---------------------------")

            if not self.position_open:
                print("[MANAGE] No position open → return")
                return None

            # Update all legs
            if self.atm_call_id: self._update_live_leg(self.atm_call_id)
            if self.atm_put_id: self._update_live_leg(self.atm_put_id)
            if self.enable_hedges and self.otm_call_id: self._update_live_leg(self.otm_call_id)
            if self.enable_hedges and self.otm_put_id: self._update_live_leg(self.otm_put_id)

            # Fetch updated ATM legs
            ac = get_position_by_id(self.atm_call_id)
            ap = get_position_by_id(self.atm_put_id)

            if not ac or not ap:
                print("[MANAGE] ATM legs not ready yet → waiting")
                time_mod.sleep(poll_interval)
                continue

            # Combined valuations
            current = ac["last_price"] + ap["last_price"]
            entry_total = ac["entry_price"] + ap["entry_price"]

            pnl_pct = (entry_total - current) / entry_total

            unrealized_pnl_total = (ac.get("unrealized_pnl", 0) or 0) + (ap.get("unrealized_pnl", 0) or 0)

            print(f"[MANAGE] ATM Combined Unrealized PnL: ${unrealized_pnl_total:.2f} USD "
                f"({pnl_pct*100:.2f}%)")

            now = datetime.now(self.tz).time()

            # --------------------
            # VWAP EXIT CHECK
            # --------------------
            vwap_exit_level = self.vwap * self.exit_vwap_mult

            print(f"[MANAGE] VWAP EXIT → current: {current:.4f}, "
                f"required: > {vwap_exit_level:.4f} "
                f"(vwap={self.vwap:.4f}, mult={self.exit_vwap_mult})")

            if current > vwap_exit_level:
                print("[MANAGE] VWAP EXIT Triggered")
                return self.exit_position("VWAP EXIT")


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
                save_active_ids(False, None, None, None, None, self.account, self.symbol)
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

            if self.otm_call_id:
                resp = self.broker.place_option_market_order(
                    self.symbol, self.expiry, self.hedge_call_strike, "C", self.hedge_qty, "SELL", self.exchange, True
                )
                self._close_leg(self.otm_call_id, resp)

            if self.otm_put_id:
                resp = self.broker.place_option_market_order(
                    self.symbol, self.expiry, self.hedge_put_strike, "P", self.hedge_qty, "SELL", self.exchange, True
                )
                self._close_leg(self.otm_put_id, resp)

        self.init_call_qty = 0
        self.init_put_qty = 0
        self.curr_call_qty = 0
        self.curr_put_qty = 0


        save_active_ids(False, None, None, None, None, self.account)

        self.atm_call_id = None
        self.atm_put_id = None
        self.otm_call_id = None
        self.otm_put_id = None
        self.position_open = False

        return {"exit_reason": reason}

    # CLOSE LEG
    def _close_leg(self, pos_id, resp):
        pos = get_position_by_id(pos_id)
        if pos is None:
            return

        fill = resp.get("fill_price")
        close_price = fill
        now = datetime.now(self.tz).isoformat()

        entry = pos["entry_price"]
        qty = pos["qty"]

        if entry is not None and close_price is not None:
            if pos["side"] == "SELL":
                realized_pnl = (entry - close_price) * qty
            else:
                realized_pnl = (close_price - entry) * qty
        else:
            realized_pnl = 0

        pos["active"] = False
        pos["exit_time"] = now
        pos["close_price"] = close_price
        pos["order_id_exit"] = resp.get("order_id")
        pos["qty"] = 0  # Set quantity to 0 when completely closed
        pos["realized_pnl"] = realized_pnl
        pos["unrealized_pnl"] = 0.0  # No unrealized PnL for closed positions
        pos["last_update"] = now

        update_position_in_json(pos)


    # TEST MODE
    def test_place_and_exit_only_atm(self):
        print("\n========== TEST MODE ==========\n")
        print("Testing: Create Position → Update Position → Partial Close → Full Close\n")

        # ==========================================
        # STEP 1: CREATE POSITIONS (ATM only, no hedges)
        # ==========================================
        print("\n[TEST STEP 1] Creating ATM positions...")
        test_strike = 684
        test_qty = 10

        call = self.broker.place_option_market_order(
            self.symbol, self.expiry, test_strike, "C", test_qty, "SELL", self.exchange, test_mode=True
        )
        put = self.broker.place_option_market_order(
            self.symbol, self.expiry, test_strike, "P", test_qty, "SELL", self.exchange, test_mode=True
        )

        entry_call = call["fill_price"]
        entry_put = put["fill_price"]

        print(f"[TEST] Call entry: ${entry_call:.2f}, Put entry: ${entry_put:.2f}")

        self.atm_call_id = create_position_entry(
            self.account, self.symbol, self.expiry, test_strike, "C",
            "SELL", test_qty, entry_call, call["order_id"], "ATM", self.tz
        )

        self.atm_put_id = create_position_entry(
            self.account, self.symbol, self.expiry, test_strike, "P",
            "SELL", test_qty, entry_put, put["order_id"], "ATM", self.tz
        )

        self.position_open = True
        self.init_call_qty = test_qty
        self.init_put_qty = test_qty
        self.curr_call_qty = test_qty
        self.curr_put_qty = test_qty
        save_active_ids(True, self.atm_call_id, self.atm_put_id, None, None, self.account, self.symbol)

        print(f"[TEST] Positions created: Call ID={self.atm_call_id}, Put ID={self.atm_put_id}")
        print(f"[TEST] Initial quantities: Call={self.curr_call_qty}, Put={self.curr_put_qty}")

        # ==========================================
        # STEP 2: UPDATE POSITIONS (Simulate live updates)
        # ==========================================
        print("\n[TEST STEP 2] Updating positions (simulating live market data)...")
        for i in range(5):
            self._update_live_leg(self.atm_call_id, test_mode=True)
            self._update_live_leg(self.atm_put_id, test_mode=True)
            
            # Get updated positions to show PnL
            call_pos = get_position_by_id(self.atm_call_id)
            put_pos = get_position_by_id(self.atm_put_id)
            if call_pos and put_pos:
                total_pnl = (call_pos.get("unrealized_pnl", 0) or 0) + (put_pos.get("unrealized_pnl", 0) or 0)
                print(f"[TEST] Update {i+1}/5 → Total Unrealized PnL: ${total_pnl:.2f}")
            
            time_mod.sleep(2)

        # ==========================================
        # STEP 3: PARTIAL CLOSE (Close 50% of position)
        # ==========================================
        print("\n[TEST STEP 3] Partially closing positions (50%)...")
        partial_exit_qty = int(test_qty * 0.5)  # Close 50%
        
        call_partial = self.broker.place_option_market_order(
            self.symbol, self.expiry, test_strike, "C", partial_exit_qty, "BUY", self.exchange, test_mode=True
        )
        put_partial = self.broker.place_option_market_order(
            self.symbol, self.expiry, test_strike, "P", partial_exit_qty, "BUY", self.exchange, test_mode=True
        )

        print(f"[TEST] Partial close: {partial_exit_qty} contracts each")
        print(f"[TEST] Call partial exit price: ${call_partial['fill_price']:.2f}")
        print(f"[TEST] Put partial exit price: ${put_partial['fill_price']:.2f}")

        # Update remaining quantities
        self.curr_call_qty -= partial_exit_qty
        self.curr_put_qty -= partial_exit_qty

        # Update position in database with new quantity and PnL
        call_pos = get_position_by_id(self.atm_call_id)
        put_pos = get_position_by_id(self.atm_put_id)
        
        if call_pos:
            # Calculate realized PnL for partial exit
            entry = call_pos["entry_price"]
            exit_price = call_partial["fill_price"]
            partial_realized_pnl = (entry - exit_price) * partial_exit_qty
            call_pos["qty"] = self.curr_call_qty
            call_pos["realized_pnl"] = (call_pos.get("realized_pnl", 0) or 0) + partial_realized_pnl
            update_position_in_json(call_pos)
        
        if put_pos:
            entry = put_pos["entry_price"]
            exit_price = put_partial["fill_price"]
            partial_realized_pnl = (entry - exit_price) * partial_exit_qty
            put_pos["qty"] = self.curr_put_qty
            put_pos["realized_pnl"] = (put_pos.get("realized_pnl", 0) or 0) + partial_realized_pnl
            update_position_in_json(put_pos)

        print(f"[TEST] Remaining quantities: Call={self.curr_call_qty}, Put={self.curr_put_qty}")
        print(f"[TEST] Positions still active: {self.position_open}")

        # Update positions one more time to show current PnL
        print("\n[TEST] Updating positions after partial close...")
        self._update_live_leg(self.atm_call_id, test_mode=True)
        self._update_live_leg(self.atm_put_id, test_mode=True)
        call_pos = get_position_by_id(self.atm_call_id)
        put_pos = get_position_by_id(self.atm_put_id)
        if call_pos and put_pos:
            total_unrealized = (call_pos.get("unrealized_pnl", 0) or 0) + (put_pos.get("unrealized_pnl", 0) or 0)
            total_realized = (call_pos.get("realized_pnl", 0) or 0) + (put_pos.get("realized_pnl", 0) or 0)
            print(f"[TEST] Current Total Unrealized PnL: ${total_unrealized:.2f}")
            print(f"[TEST] Current Total Realized PnL: ${total_realized:.2f}")

        # ==========================================
        # STEP 4: FULL CLOSE (Close remaining position)
        # ==========================================
        print("\n[TEST STEP 4] Fully closing remaining positions...")
        
        call_exit = self.broker.place_option_market_order(
            self.symbol, self.expiry, test_strike, "C", self.curr_call_qty, "BUY", self.exchange, test_mode=True
        )
        put_exit = self.broker.place_option_market_order(
            self.symbol, self.expiry, test_strike, "P", self.curr_put_qty, "BUY", self.exchange, test_mode=True
        )

        print(f"[TEST] Full close: {self.curr_call_qty} call contracts, {self.curr_put_qty} put contracts")
        print(f"[TEST] Call exit price: ${call_exit['fill_price']:.2f}")
        print(f"[TEST] Put exit price: ${put_exit['fill_price']:.2f}")

        self._close_leg(self.atm_call_id, call_exit)
        self._close_leg(self.atm_put_id, put_exit)

        # Mark position as closed
        self.position_open = False
        self.curr_call_qty = 0
        self.curr_put_qty = 0
        save_active_ids(False, None, None, None, None, self.account)

        # Get final positions to show final PnL
        call_pos = get_position_by_id(self.atm_call_id)
        put_pos = get_position_by_id(self.atm_put_id)
        if call_pos and put_pos:
            final_realized = (call_pos.get("realized_pnl", 0) or 0) + (put_pos.get("realized_pnl", 0) or 0)
            print(f"[TEST] Final Total Realized PnL: ${final_realized:.2f}")
            print(f"[TEST] Call final Realized PnL: ${call_pos.get('realized_pnl', 0) or 0:.2f}")
            print(f"[TEST] Put final Realized PnL: ${put_pos.get('realized_pnl', 0) or 0:.2f}")

        print("\n========== TEST COMPLETE ==========\n")

    def run(self, poll_interval=60):
        print("[Strategy] RUN LOOP STARTED")

        while self.manager.keep_running:
            # Check if paused - if so, just wait and check again
            if self.manager.paused:
                print(f"[Strategy-{self.strike_offset}] Paused - waiting...")
                time_mod.sleep(5)  # Check every 5 seconds when paused
                continue

            # If a position is open → manage it
            if self.position_open:
                # Reload config and fetch data before managing positions
                self.load_config(self.config_path)
                if not self.fetch_data():
                    time_mod.sleep(poll_interval)
                    continue
                if not self.calculate_indicators():
                    time_mod.sleep(poll_interval)
                    continue
                self.manage_positions(10)
                time_mod.sleep(poll_interval)
                continue
            print("No position open")

            # No position → fetch market data
            # Config is already loaded in __init__, no need to reload
            if not self.fetch_data():
                time_mod.sleep(poll_interval)
                continue

            print("Data fetched")

            # Calculate VWAP
            if not self.calculate_indicators():
                time_mod.sleep(poll_interval)
                continue
            print("Indicators Calculated")
            # Generate signal
            signal = self.generate_signals()
            
            print(f"Sigal generated: {signal}")

            if signal["action"] == "SELL_STRADDLE":
                self.execute_trade(signal)
                continue
            print(f"Sleeping for {poll_interval}")
            time_mod.sleep(poll_interval)

    def log_signal(self, action, combined):
        """Save each signal to a daily CSV file."""

        # Create signals folder if missing
        Path("signals").mkdir(exist_ok=True)

        # Use date-based filename
        date_str = datetime.now(self.tz).strftime("%Y-%m-%d")
        file_path = f"signals/{date_str}_signals.csv"

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


class StrategyBroker:
    def __init__(self, config_path="config.json", test_mode=False):
        with open(config_path, "r") as f:
            self.config = json.load(f)
        self.test_mode = test_mode
        # Skip IBKR connection in test mode
        if not test_mode:
            host = self.config["broker"]["host"]
            port = self.config["broker"]["port"]
            client_id = self.config["broker"]["client_id"]

            self.ib_broker = IBBroker()
            self.ib_broker.connect_to_ibkr(host, port, client_id)
        else:
            self.ib_broker = None
            print("[StrategyBroker] Test mode - skipping IBKR connection")

        self.request_id_counter = 1
        self.counter_lock = threading.Lock()

    def get_next_available_order_id(self):
        if self.test_mode:
            return 1000  # Mock order ID for test mode
        return self.ib_broker.get_next_order_id_from_ibkr()

    def reset_order_counter_to_next_available(self):
        next_id = self.get_next_available_order_id()
        if next_id:
            with self.counter_lock:
                self.request_id_counter = next_id - 2000
            print(f"Reset order counter to start from IBKR ID: {next_id}")

    def current_price(self, symbol, exchange, test_mode=False):
        if test_mode or self.test_mode:
            # Return mock price around 680-690 for testing
            return round(random.uniform(680, 690), 2)
        
        with self.counter_lock:
            req_id = self.request_id_counter + 2000
            self.request_id_counter += 1
        return self.ib_broker.get_index_spot(symbol, req_id, exchange)

    def get_option_premium(self, symbol, expiry, strike, right, test_mode=False):
        # TEST MODE: Return mock premium data
        if test_mode:
            return {
                "bid": round(random.uniform(0.3, 0.8), 2),
                "ask": round(random.uniform(0.4, 0.9), 2),
                "last": round(random.uniform(0.35, 0.85), 2),
                "mid": round(random.uniform(0.35, 0.85), 2)
            }

        with self.counter_lock:
            req_id = self.request_id_counter + 3000
            self.request_id_counter += 1
        return self.ib_broker.get_option_premium(symbol, expiry, strike, right, req_id)

    def get_option_tick(self, symbol, expiry, strike, right):
        with self.counter_lock:
            req_id = self.request_id_counter + 5000
            self.request_id_counter += 1
        return self.ib_broker.get_option_tick(symbol, expiry, strike, right, req_id)

    def get_option_ohlc(self, symbol, expiry, strike, right, duration="1 D", bar_size="1 min"):
        with self.counter_lock:
            req_id = self.request_id_counter + 6000
            self.request_id_counter += 1
        x = self.ib_broker.get_option_ohlc(symbol, expiry, strike, right, duration, bar_size, req_id)
        return pd.DataFrame(x)

    def place_option_market_order(self, symbol, expiry, strike, right, qty, action, exchange, wait_until_filled=False, test_mode=False):
        print(f"[BROKER] MARKET ORDER → {action} {qty}x {symbol} {expiry} {strike}{right}")

        # TEST MODE: Return mock data instead of placing actual orders
        if test_mode:   
            # Mock fill price based on action (SELL = higher price, BUY = lower price)
            if action == "SELL":
                base_price = random.uniform(0.5, 1.5)
            else:
                base_price = random.uniform(0.3, 1.2)
            
            mock_order_id = random.randint(1000, 9999)
            mock_fill_price = round(base_price, 2)
            
            print(f"[BROKER TEST MODE] Mock order filled → OrderID: {mock_order_id}, Fill: ${mock_fill_price}")
            return {
                "order_id": mock_order_id,
                "fill_price": mock_fill_price
            }

        # Get orderId directly from IBKR
        with self.counter_lock:
            req_id = self.get_next_available_order_id()

        # Place the order
        order_id, fill_price = self.ib_broker.place_market_option_order(
            symbol, exchange, expiry, strike, right, action, qty, req_id, wait_until_filled
        )

        return {
            "order_id": order_id,
            "fill_price": fill_price
        }

        # return {
        #     "order_id": 1,
        #     "fill_price": 2
        # }



# STRATEGY MANAGER
class StrategyManager:
    def __init__(self, config_path="config.json", account="default", symbol_override=None, strike_range=None, current_atm_price=None, test_mode=False):
        self.account = account
        self.config_path = config_path
        self.strike_range = strike_range if strike_range is not None else [0]  # Default to [0] for backward compatibility
        self.current_atm_price = current_atm_price
        self.test_mode = test_mode
        self.broker = StrategyBroker(config_path=config_path, test_mode=test_mode)

        # Load config to get symbol and exchange for ATM price fetch
        with open(config_path, "r") as f:
            cfg = json.load(f)
        self.symbol = symbol_override if symbol_override else cfg["underlying"]["symbol"]
        self.exchange = cfg["underlying"]["exchange"]

        # Get current ATM price from broker if not provided
        if self.current_atm_price is None:
            if test_mode:
                # Use mock price in test mode
                print(f"[Manager] Test mode - using mock ATM price")
                spot = self.broker.current_price(self.symbol, self.exchange, test_mode=True)
            else:
                print(f"[Manager] Fetching current ATM price for {self.symbol}...")
                spot = self.broker.current_price(self.symbol, self.exchange)
            
            if spot is None:
                raise ValueError(f"Could not fetch current price for {self.symbol}")
            # Round to nearest strike step
            strike_step = cfg["trade_parameters"]["strike_step"]
            self.current_atm_price = round(spot / strike_step) * strike_step
            print(f"[Manager] Current ATM price: {self.current_atm_price}")

        # Create multiple Strategy instances, one for each offset in range
        self.strategies = []
        self.strategy_threads = []
        
        for offset in self.strike_range:
            strategy = Strategy(
                manager=self,
                broker=self.broker,
                account=account,
                config_path=config_path,
                strike_offset=offset,
                base_atm_price=self.current_atm_price
            )
            # Override symbol if provided
            if symbol_override:
                strategy.symbol = symbol_override
            self.strategies.append(strategy)

        self.keep_running = True
        self.paused = False
        print(f"[Manager] Created {len(self.strategies)} strategy instances for offsets: {self.strike_range}")

    def run(self):
        # TEST MODE: Run test function in main thread only
        if self.test_mode:
            print(f"[Manager] TEST MODE - Running test function in main thread only")
            if not self.strategies:
                print("[Manager] No strategies available for testing")
                return
            
            # Use first strategy for testing
            test_strategy = self.strategies[0]
            print(f"[Manager] Running test on strategy with offset={test_strategy.strike_offset}")
            test_strategy.test_place_and_exit_only_atm()
            return
        
        print(f"[Manager] Starting {len(self.strategies)} strategies | account={self.account} | ATM={self.current_atm_price}")
        
        # Start each strategy in its own thread, with a few seconds delay between each
        for i, strategy in enumerate(self.strategies):
            thread = threading.Thread(
                target=strategy.run,
                name=f"Strategy-{strategy.strike_offset}",
                daemon=False
            )
            thread.start()
            self.strategy_threads.append(thread)
            print(f"[Manager] Started thread for strategy with offset={strategy.strike_offset}")
            
            # Wait a few seconds before starting the next thread (except for the last one)
            if i < len(self.strategies) - 1:
                time.sleep(3)  # 3 second delay between thread spawns
        
        # Monitor state file and control strategies
        try:
            while self.keep_running:
                # Check if stopped (using account+symbol key)
                if is_account_stopped(self.account, self.symbol):
                    print(f"[Manager] Stop signal received for account {self.account} symbol {self.symbol}")
                    self.keep_running = False
                    break
                
                # Check if paused (using account+symbol key)
                self.paused = is_account_paused(self.account, self.symbol)
                
                time.sleep(2)  # Check state every 2 seconds
        except KeyboardInterrupt:
            print(f"[Manager] Keyboard interrupt received")
            self.keep_running = False
        
        # Wait for all threads to complete
        for thread in self.strategy_threads:
            thread.join()
        
        print(f"[Manager] All strategies stopped for account {self.account}") 
        


if __name__ == "__main__":
    import os
    import atexit
    
    args = parse_args()
    
    # Write PID to file for frontend to track (account+symbol-specific)
    account = args.account
    symbol = args.symbol or "default"  # Use symbol from args or default
    pid_file = f"bot.{account}.{symbol}.pid"
    status_file = f"bot.{account}.{symbol}.status"
    
    def cleanup_files():
        """Clean up PID and status files on exit"""
        try:
            if os.path.exists(pid_file):
                os.remove(pid_file)
            if os.path.exists(status_file):
                os.remove(status_file)
            print(f"[Main] Cleaned up tracking files for account {account} on exit")
        except:
            pass
    
    # Register cleanup on normal exit
    atexit.register(cleanup_files)
    
    # Handle signals for graceful shutdown (Unix/Linux)
    if os.name != 'nt':
        import signal
        signal.signal(signal.SIGTERM, lambda s, f: (cleanup_files(), exit(0)))
        signal.signal(signal.SIGINT, lambda s, f: (cleanup_files(), exit(0)))
    
    try:
        with open(pid_file, "w") as f:
            f.write(str(os.getpid()))
        print(f"[Main] PID written to {pid_file}: {os.getpid()} (account: {account})")
        
        # Write initial status
        with open(status_file, "w") as f:
            f.write(f"running|{datetime.now().isoformat()}")
    except Exception as e:
        print(f"[Main] Warning: Could not write tracking files: {e}")

    # Parse strike range if provided
    strike_range = [0]  # Default
    if args.range:
        try:
            start, end = map(int, args.range.split(":"))
            strike_range = list(range(start, end + 1))
            print(f"[Main] Parsed strike range: {strike_range}")
        except ValueError:
            print(f"[Main] Invalid range format '{args.range}'. Using default [0]")
            strike_range = [0]

    # In test mode, use single strategy (offset 0) regardless of range
    if args.test:
        print("[Main] Test mode enabled - using single strategy (offset=0)")
        strike_range = [0]

    try:
        manager = StrategyManager(
            config_path=args.config,
            account=args.account,
            symbol_override=args.symbol,
            strike_range=strike_range,
            test_mode=False
        )
        
        # Start a background thread to update status file periodically
        def update_status():
            while manager.keep_running:
                try:
                    paused_state = "paused" if manager.paused else "running"
                    with open(status_file, "w") as f:
                        f.write(f"{paused_state}|{datetime.now().isoformat()}|{account}|{args.symbol or manager.symbol}")
                except:
                    pass
                time_mod.sleep(5)  # Update every 5 seconds
        
        status_thread = threading.Thread(target=update_status, daemon=True)
        status_thread.start()
        
        manager.run()
    finally:
        # Clear stop state when exiting
        from helpers.state_manager import clear_account_state
        clear_account_state(account, symbol)
        cleanup_files()

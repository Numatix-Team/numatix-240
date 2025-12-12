import os
import json
import time
import random
import threading
from datetime import datetime, time as dt_time
from zoneinfo import ZoneInfo
from typing import Any, Dict, Optional
import pandas as pd
import pytz
import numpy as np
import time as time_mod
import csv
from pathlib import Path

from broker.ib_broker import IBBroker
from log import setup_logger
from helpers.positions import *

setup_logger()


class Strategy:

    def __init__(self, manager, broker, config_path="config.json"):
        self.manager = manager
        self.broker = broker
        self.config_path = config_path

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

        if not os.path.exists("positions.json"):
            save_positions_file({"positions": [], "active_positions": {"position_open": False}})

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
        self.tp_pct = tp["take_profit"]
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

    # RESTORE ACTIVE POSITIONS
    def _load_active_ids(self):
        active = load_active_ids()

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
        # 1) Fetch spot price
        # -------------------------------
        spot = self.broker.current_price(self.symbol, self.exchange)
        if spot is None:
            print("[Strategy] Spot unavailable.")
            return False

        # -------------------------------
        # 2) Compute strikes
        # -------------------------------
        atm = round(spot / self.strike_step) * self.strike_step
        self.atm = atm
        self.call_strike = atm + self.atm_call_offset * self.strike_step
        self.put_strike = atm - self.atm_put_offset * self.strike_step

        if self.enable_hedges:
            self.hedge_call_strike = atm + self.hedge_call_offset * self.strike_step
            self.hedge_put_strike = atm - self.hedge_put_offset * self.strike_step

        print(f"[Strategy] ATM={atm}, CALL={self.call_strike}, PUT={self.put_strike}")

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

        # Block entry before market open
        if now < self.entry_start:
            return {"action": "NONE"}

        # Block entry after entry window
        if now > self.entry_end:
            return {"action": "BLOCKED_TIME"}

        cp = self.broker.get_option_premium(self.symbol, self.expiry, self.call_strike, "C")
        pp = self.broker.get_option_premium(self.symbol, self.expiry, self.put_strike, "P")

        c_price = self._extract_price(cp)
        p_price = self._extract_price(pp)

        if c_price is None or p_price is None:
            return {"action": "NONE"}

        combined = c_price + p_price

        # Log every signal check
        self.log_signal("CHECK", combined)

        # Spread check
        if self._spread(cp) is None or self._spread(pp) is None:
            return {"action": "NONE"}

        if self._spread(cp) > self.max_spread or self._spread(pp) > self.max_spread:
            return {"action": "NONE"}

        threshold = self.vwap * self.entry_vwap_mult

        if not self.position_open and combined < threshold:
            self.log_signal("SELL_STRADDLE", combined)
            return {"action": "SELL_STRADDLE", "combined": combined}

        return {"action": "NONE"}

    # ORDER EXECUTION
    def execute_trade(self, signal):
        combined = signal["combined"]
        print(f"[Strategy] SELL STRADDLE @ {combined}")
        
        # Hedges
        if self.enable_hedges:

            hc = self.broker.place_option_market_order(
                self.symbol, self.expiry, self.hedge_call_strike, "C", self.hedge_qty, "BUY", self.exchange
            )
            fill_hc = hc["fill_price"]
            if fill_hc is None:
                print("Call hedge returned None")
                return

            self.otm_call_id = create_position_entry(
                self.symbol, self.expiry, self.hedge_call_strike, "C",
                "BUY", self.hedge_qty,
                fill_hc, hc["order_id"], "OTM", self.tz,
                bid=0, ask=0
            )

            hp = self.broker.place_option_market_order(
                self.symbol, self.expiry, self.hedge_put_strike, "P", self.hedge_qty, "BUY", self.exchange
            )
            fill_hp = hp["fill_price"]
            if fill_hp is None:
                print("Put hedge returned None")
                return

            self.otm_put_id = create_position_entry(
                self.symbol, self.expiry, self.hedge_put_strike, "P",
                "BUY", self.hedge_qty,
                fill_hp, hp["order_id"], "OTM", self.tz,
                bid=0, ask=0
            )

        # ATM CALL
        c = self.broker.place_option_market_order(
            self.symbol, self.expiry, self.call_strike, "C", self.call_qty, "SELL", self.exchange
        )
        fill_call = c["fill_price"]
        if fill_call is None:
            print("Call strike returned None")
            return

        self.atm_call_id = create_position_entry(
            self.symbol, self.expiry, self.call_strike, "C",
            "SELL", self.call_qty,
            fill_call, c["order_id"], "ATM", self.tz,
            bid=0, ask=0
        )

        # ATM PUT
        p = self.broker.place_option_market_order(
            self.symbol, self.expiry, self.put_strike, "P", self.put_qty, "SELL", self.exchange
        )
        fill_put = p["fill_price"]
        if fill_put is None:
            print("Put strike returned None")
            return

        self.atm_put_id = create_position_entry(
            self.symbol, self.expiry, self.put_strike, "P",
            "SELL", self.put_qty,
            fill_put, p["order_id"], "ATM", self.tz,
            bid=0, ask=0
        )

        self.position_open = True
        save_active_ids(
            True, self.atm_call_id, self.atm_put_id, self.otm_call_id, self.otm_put_id
        )

    # LIVE UPDATES
    def _update_live_leg(self, pos_id):
        pos = get_position_by_id(pos_id)
        if pos is None or not pos.get("active", False):
            return

        data = self.broker.get_option_premium(
            pos["symbol"], pos["expiry"], pos["strike"], pos["right"]
        )

        bid = data.get("bid")
        ask = data.get("ask")

        if bid is not None and ask is not None:
            last_price = (bid + ask) / 2
        else:
            last_price = data.get("last") or data.get("mid")

        entry = pos["entry_price"]
        qty = pos["qty"]

        # -----------------------------
        # NEW: PNL VALUE INSTEAD OF %
        # -----------------------------
        if pos["side"] == "SELL":
            pnl_value = (entry - last_price) * qty
            pnl_pct = (entry - last_price) / entry
        else:
            pnl_value = (last_price - entry) * qty
            pnl_pct = (last_price - entry) / entry

        pos["bid"] = bid
        pos["ask"] = ask
        pos["last_price"] = last_price
        pos["pnl_value"] = pnl_value
        pos["pnl_pct"] = pnl_pct  # still stored for reporting
        pos["last_update"] = datetime.now(self.tz).isoformat()

        update_position_in_json(pos)



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

            pnl_value_total = ac["pnl_value"] + ap["pnl_value"]

            print(f"[MANAGE] ATM Combined PnL Value: {pnl_value_total:.2f} USD "
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
            print(f"[MANAGE] TAKE PROFIT → current pnl_pct: {pnl_pct*100:.2f}%, "
                f"required: ≥ {self.tp_pct*100:.2f}%")

            if pnl_pct >= self.tp_pct:
                print("[MANAGE] TAKE PROFIT Triggered")
                return self.exit_position("TAKE PROFIT")


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

        if self.atm_call_id:
            resp = self.broker.place_option_market_order(
                self.symbol, self.expiry, self.call_strike, "C", self.call_qty, "BUY", self.exchange, True
            )
            self._close_leg(self.atm_call_id, resp)

        if self.atm_put_id:
            resp = self.broker.place_option_market_order(
                self.symbol, self.expiry, self.put_strike, "P", self.put_qty, "BUY", self.exchange, True
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

        save_active_ids(False, None, None, None, None)

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
                pnl_value = (entry - close_price) * qty
                pnl_pct = (entry - close_price) / entry
            else:
                pnl_value = (close_price - entry) * qty
                pnl_pct = (close_price - entry) / entry
        else:
            pnl_value = 0
            pnl_pct = 0

        pos["active"] = False
        pos["exit_time"] = now
        pos["close_price"] = close_price
        pos["order_id_exit"] = resp.get("order_id")
        pos["pnl_value"] = pnl_value
        pos["pnl_pct"] = pnl_pct
        pos["last_update"] = now

        update_position_in_json(pos)


    # TEST MODE
    def test_place_and_exit_only_atm(self):
        print("\n========== TEST MODE ==========\n")

        call = self.broker.place_option_market_order(self.symbol, self.expiry, 684, "C", 1, "SELL", self.exchange)
        put = self.broker.place_option_market_order(self.symbol, self.expiry, 684, "P", 1, "SELL", self.exchange)

        entry_call = call["fill_price"]
        entry_put = put["fill_price"]

        self.atm_call_id = create_position_entry(
            self.symbol, self.expiry, 684, "C",
            "SELL", 10, entry_call, call["order_id"], "ATM", self.tz
        )

        self.atm_put_id = create_position_entry(
            self.symbol, self.expiry, 684, "P",
            "SELL", 10, entry_put, put["order_id"], "ATM", self.tz
        )

        self.position_open = True
        save_active_ids(True, self.atm_call_id, self.atm_put_id, None, None)

        for i in range(10):
            self._update_live_leg(self.atm_call_id)
            self._update_live_leg(self.atm_put_id)
            print(f"Updated {i+1}/10")
            time_mod.sleep(1)

        call_exit = self.broker.place_option_market_order(self.symbol, self.expiry, 684, "C", 10, "BUY", self.exchange)
        put_exit = self.broker.place_option_market_order(self.symbol, self.expiry, 684, "P", 10, "BUY", self.exchange)

        self._close_leg(self.atm_call_id, call_exit)
        self._close_leg(self.atm_put_id, put_exit)

        print("\n========== TEST COMPLETE ==========\n")

    def run(self, poll_interval=60):
        print("[Strategy] RUN LOOP STARTED")

        while self.manager.keep_running:

            # If a position is open → manage it
            if self.position_open:
                self.manage_positions(10)
                time_mod.sleep(poll_interval)
                continue
            print("No position open")

            self.load_config(self.config_path)
            # No position → fetch market data
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

            # if signal["action"] == "SELL_STRADDLE":
            if True:
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
                                "put_strike", "vwap"])

            writer.writerow([
                datetime.now(self.tz).isoformat(),
                action,
                combined if combined is not None else "",
                self.call_strike,
                self.put_strike,
                self.vwap
            ])


class StrategyBroker:
    def __init__(self, config_path="config.json"):
        with open(config_path, "r") as f:
            self.config = json.load(f)

        host = self.config["broker"]["host"]
        port = self.config["broker"]["port"]
        client_id = self.config["broker"]["client_id"]

        self.ib_broker = IBBroker()
        self.ib_broker.connect_to_ibkr(host, port, client_id)

        self.request_id_counter = 1
        self.counter_lock = threading.Lock()

    def get_next_available_order_id(self):
        return self.ib_broker.get_next_order_id_from_ibkr()

    def reset_order_counter_to_next_available(self):
        next_id = self.get_next_available_order_id()
        if next_id:
            with self.counter_lock:
                self.request_id_counter = next_id - 2000
            print(f"Reset order counter to start from IBKR ID: {next_id}")

    def current_price(self, symbol, exchange):
        with self.counter_lock:
            req_id = self.request_id_counter + 2000
            self.request_id_counter += 1
        return self.ib_broker.get_index_spot(symbol, req_id, exchange)

    def get_option_premium(self, symbol, expiry, strike, right):
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

    def place_option_market_order(self, symbol, expiry, strike, right, qty, action, exchange, wait_until_filled=False):
        print(f"[BROKER] MARKET ORDER → {action} {qty}x {symbol} {expiry} {strike}{right}")

        # Get orderId directly from IBKR
        with self.counter_lock:
            req_id = req_id = self.get_next_available_order_id()

        # place the order
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
    def __init__(self):
        self.broker = StrategyBroker()
        self.strategy = Strategy(self, self.broker)
        self.keep_running = True

    def run(self):
        self.strategy.run()
        


if __name__ == "__main__":
    manager = StrategyManager()
    manager.run()

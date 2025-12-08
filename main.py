import json
import threading
from broker.ib_broker import IBBroker
import pandas as pd
import numpy as np
from db.db_logger import OptionDBLogger
import time
from datetime import datetime, time
import pytz
from log import setup_logger
setup_logger()

import json
import time as time_mod
import pandas as pd
import pytz
from datetime import datetime, time as time_obj
from typing import Any, Dict, Optional

import json
import time as time_mod
import pandas as pd
import pytz
from datetime import datetime, time as time_obj
from typing import Any, Dict, Optional


class Strategy:
    SPX_STRIKE_STEP = 5   # SPX always uses 5-point strikes

    def __init__(self, manager, broker, config_path="config.json"):
        self.manager = manager
        self.broker = broker

        with open(config_path, "r") as f:
            cfg = json.load(f)

        # =============== Underlying ===============
        self.symbol = cfg["underlying"]["symbol"]
        self.exchange = cfg["underlying"]["exchange"]
        self.currency = cfg["underlying"]["currency"]

        # =============== Expiry ===============
        self.expiry = cfg["expiry"]["date"]

        # =============== Trade Parameters ===============
        tp = cfg["trade_parameters"]
        self.call_qty = tp["call_quantity"]
        self.put_qty = tp["put_quantity"]
        self.atm_call_offset = tp["atm_call_offset"]
        self.atm_put_offset = tp["atm_put_offset"]
        self.entry_vwap_mult = tp["entry_vwap_multiplier"]
        self.tp_pct = tp["take_profit"]
        self.sl_pct = tp["stop_loss"]
        self.max_spread = tp["max_bid_ask_spread"]

        # =============== Time Controls ===============
        tc = cfg["time_controls"]
        self.entry_start = self._parse_time(tc["entry_start"])
        self.entry_end = self._parse_time(tc["entry_end"])
        self.force_exit_time = self._parse_time(tc["force_exit_time"])
        self.tz = pytz.timezone(tc["timezone"])

        # =============== Hedging ===============
        h = cfg["hedging"]
        self.enable_hedges = h["enable_hedges"]
        self.hedge_call_offset = h["hedge_call_offset"]
        self.hedge_put_offset = h["hedge_put_offset"]
        self.hedge_qty = h["hedge_quantity"]

        # Internal State
        self.atm = None
        self.call_strike = None
        self.put_strike = None
        self.hedge_call_strike = None
        self.hedge_put_strike = None

        self.vwap = None
        self.hist_df = pd.DataFrame()
        self.position_open = False
        self.position = None

    # -----------------------------------------------------------------
    def _parse_time(self, s):
        h, m = s.split(":")
        return time_obj(int(h), int(m))

    # -----------------------------------------------------------------
    # CENTRAL → EASTERN TIMESTAMP CONVERSION
    # -----------------------------------------------------------------
    def _parse_ib_datetime(self, s):
        """
        IBKR historical OHLC timestamps come in US/Central.
        Convert them to US/Eastern.
        Example: '20251208 10:48:00 US/Central'
        """
        try:
            parts = s.split(" ")
            date_part = parts[0]  # 20251208
            time_part = parts[1]  # 10:48:00
            tz_part = parts[2] if len(parts) > 2 else "US/Central"

            # Parse naive datetime
            naive = datetime.strptime(date_part + " " + time_part, "%Y%m%d %H:%M:%S")

            # Localize to IBKR's timezone (US/Central)
            central = pytz.timezone("US/Central").localize(naive)

            # Convert to Eastern
            eastern = central.astimezone(self.tz)
            return eastern

        except Exception:
            return pd.to_datetime(s)

    # -----------------------------------------------------------------
    def _extract_price(self, d):
        if not d:
            return None

        # 1. If mid provided, use it
        if "mid" in d and d["mid"] is not None:
            return d["mid"]

        bid = d.get("bid")
        ask = d.get("ask")

        # 2. Compute mid manually if both bid and ask exist
        if bid is not None and ask is not None:
            return (bid + ask) / 2

        # 3. Fallback to last traded price
        if "last" in d and d["last"] is not None:
            return d["last"]

        # 4. Final fallback to bid or ask if only one exists
        if bid is not None:
            return bid

        if ask is not None:
            return ask

        return None

    def _spread(self, d):
        if d.get("bid") is None or d.get("ask") is None:
            return None
        return d["ask"] - d["bid"]

    # -----------------------------------------------------------------
    # 1. Fetch Historical Data for VWAP
    # -----------------------------------------------------------------
    def fetch_data(self):
        # -------- Spot price --------
        spot = self.broker.current_price(self.symbol, self.exchange)
        if spot is None:
            print("[Strategy] Spot unavailable.")
            return False

        # -------- ATM Calculation --------
        atm = round(spot / self.SPX_STRIKE_STEP) * self.SPX_STRIKE_STEP
        self.atm = atm

        self.call_strike = atm + (self.atm_call_offset * self.SPX_STRIKE_STEP)
        self.put_strike = atm - (self.atm_put_offset * self.SPX_STRIKE_STEP)

        if self.enable_hedges:
            self.hedge_call_strike = atm + (self.hedge_call_offset * self.SPX_STRIKE_STEP)
            self.hedge_put_strike = atm - (self.hedge_put_offset * self.SPX_STRIKE_STEP)

        print(f"[Strategy] ATM={atm}, Call={self.call_strike}, Put={self.put_strike}")

        # -------- Fetch OHLC for VWAP --------
        c_ohlc = self.broker.get_option_ohlc(self.symbol, self.expiry, self.call_strike, "C")
        p_ohlc = self.broker.get_option_ohlc(self.symbol, self.expiry, self.put_strike, "P")

        if c_ohlc.empty or p_ohlc.empty:
            print("[Strategy] OHLC missing.")
            return False

        # ---- Convert IBKR timestamps from Central → Eastern ----
        c_ohlc["time"] = c_ohlc["time"].apply(self._parse_ib_datetime)
        p_ohlc["time"] = p_ohlc["time"].apply(self._parse_ib_datetime)

        # Merge
        df = c_ohlc.merge(p_ohlc, on="time", suffixes=("_call", "_put"))

        # ---- FILTER TO TODAY’S RTH (9:30 ET → now) ----
        today = datetime.now(self.tz).date()
        market_open = datetime.now(self.tz).replace(hour=9, minute=30, second=0, microsecond=0)

        df["time"] = df["time"].dt.tz_localize(self.tz)
        df = df[df["time"].dt.date == today]
        df = df[df["time"] >= market_open]
        if df.empty:
            print("[Strategy] No RTH bars available for VWAP.")
            return False

        df["combined_premium"] = df["close_call"] + df["close_put"]
        df["combined_volume"] = df["volume_call"] + df["volume_put"]

        df = df[df["combined_volume"] > 0]
        if df.empty:
            print("[Strategy] No valid VWAP bars.")
            return False

        self.hist_df = df
        return True

    # -----------------------------------------------------------------
    # 2. VWAP Calculation
    # -----------------------------------------------------------------
    def calculate_indicators(self):
        df = self.hist_df
        print(df)
        df["turnover"] = df["combined_premium"] * df["combined_volume"].astype(float)

        tot_vol = df["combined_volume"].sum()
        if tot_vol == 0:
            return False

        self.vwap = float(df["turnover"].sum()) / float(tot_vol)
        print(f"[Strategy] VWAP={self.vwap:.2f}")
        return True

    # -----------------------------------------------------------------
    # 3. Signal Generation
    # -----------------------------------------------------------------
    def generate_signals(self):
        now = datetime.now(self.tz).time()

        if not (self.entry_start <= now <= self.entry_end):
            return {"action": "NONE"}

        cp = self.broker.get_option_premium(self.symbol, self.expiry, self.call_strike, "C")
        pp = self.broker.get_option_premium(self.symbol, self.expiry, self.put_strike, "P")

        c_price = self._extract_price(cp)
        p_price = self._extract_price(pp)

        if c_price is None or p_price is None:
            return {"action": "NONE", "reason": "No price"}

        combined = c_price + p_price

        if (
            self._spread(cp) is None or
            self._spread(pp) is None or
            self._spread(cp) > self.max_spread or
            self._spread(pp) > self.max_spread
        ):
            return {"action": "NONE", "reason": "Spread too wide"}

        entry_threshold = self.vwap * self.entry_vwap_mult

        if not self.position_open and combined < entry_threshold:
            return {"action": "SELL_STRADDLE", "combined": combined}

        return {"action": "NONE"}

    # -----------------------------------------------------------------
    # 4. Dummy Execution
    # -----------------------------------------------------------------
    def execute_trade(self, signal):
        combined = signal["combined"]

        print(f"[Strategy] SELL STRADDLE @ combined={combined}")

        self.broker.place_option_market_order(self.symbol, self.expiry, self.call_strike, "C", self.call_qty, "SELL")
        self.broker.place_option_market_order(self.symbol, self.expiry, self.put_strike, "P", self.put_qty, "SELL")

        if self.enable_hedges:
            print("[Strategy] Placing hedges...")
            self.broker.place_option_market_order(self.symbol, self.expiry, self.hedge_call_strike, "C", self.hedge_qty, "BUY")
            self.broker.place_option_market_order(self.symbol, self.expiry, self.hedge_put_strike, "P", self.hedge_qty, "BUY")

        self.position_open = True
        self.position = {
            "entry_combined": combined,
            "strike_call": self.call_strike,
            "strike_put": self.put_strike
        }

    # -----------------------------------------------------------------
    def manage_positions(self, poll_interval):
        while True:
            if not self.position_open:
                return None

            now = datetime.now(self.tz).time()
            if now >= self.force_exit_time:
                return self.exit_position("FORCED EXIT")

            cp = self.broker.get_option_premium(self.symbol, self.expiry, self.call_strike, "C")
            pp = self.broker.get_option_premium(self.symbol, self.expiry, self.put_strike, "P")

            c_price = self._extract_price(cp)
            p_price = self._extract_price(pp)

            if not c_price or not p_price:
                return None

            combined = c_price + p_price
            entry = self.position["entry_combined"]
            pnl_pct = (entry - combined) / entry

            if pnl_pct >= self.tp_pct:
                return self.exit_position("TAKE PROFIT")

            if pnl_pct <= -self.sl_pct:
                return self.exit_position("STOP LOSS")

            time.sleep(poll_interval)


    # -----------------------------------------------------------------
    def exit_position(self, reason):
        print(f"[Strategy] EXIT — {reason}")

        self.broker.place_option_market_order(self.symbol, self.expiry, self.call_strike, "C", self.call_qty, "BUY")
        self.broker.place_option_market_order(self.symbol, self.expiry, self.put_strike, "P", self.put_qty, "BUY")

        self.position_open = False
        self.position = None
        return {"exit_reason": reason}

    # -----------------------------------------------------------------
    def run(self, poll_interval=1.0):
        print("[Strategy] Initializing...")

        while not (self.fetch_data() and self.calculate_indicators()):
            time_mod.sleep(1)

        print("[Strategy] Running...")

        while self.manager.keep_running:
            signal = self.generate_signals()

            if signal.get("action") == "SELL_STRADDLE":
                self.execute_trade(signal)

            if self.position_open:
                res = self.manage_positions(poll_interval)
                print(res)


            time_mod.sleep(poll_interval)

class StrategyBroker:
    def __init__(self, config_path="config.json"):
        """Initialize StrategyBroker with IBBroker instance"""
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
        """Get the next available order ID directly from IBKR"""
        return self.ib_broker.get_next_order_id_from_ibkr()

    def reset_order_counter_to_next_available(self):
        """Reset the counter to start from the next available order ID from IBKR"""
        next_id = self.get_next_available_order_id()
        if next_id:
            with self.counter_lock:
                self.request_id_counter = next_id - 2000
            print(f"Reset order counter to start from IBKR ID: {next_id}")
        else:
            print("Could not get next order ID from IBKR")

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
    
    def place_option_market_order(self, symbol, expiry, strike, right, qty, action):
        print(
            f"[BROKER] MARKET ORDER → {action} {qty}x {symbol} {expiry} {strike}{right}"
        )
        # return fake order id
        return f"FAKE_ORDER_{symbol}_{expiry}_{strike}_{right}_{action}"


class StrategyManager:
    def __init__(self):
        self.broker = StrategyBroker()
        self.strategy = Strategy(self, self.broker)
        self.db = OptionDBLogger()
        self.keep_running = True

    def run(self):
        # self.collection_thread = threading.Thread(target=self.collect_option_ticks)
        # self.collection_thread.start()
        self.strategy.run()
    
    # def collect_option_ticks(self):
    #     # To be added to creds.json
    #     symbol = "SPX"
    #     expiry = "20251212"

    #     call_strike = 6800
    #     put_strike = 6800

    #     while self.keep_running:
    #         # GET CALL DATA
    #         try:
    #             call_data = self.broker.get_option_tick(symbol, expiry, call_strike, "C")
    #             put_data  = self.broker.get_option_tick(symbol, expiry, put_strike, "P")
                
    #             if call_data is None or put_data is None:
    #                 print("Option tick returned None — retrying...")
    #                 continue

    #             if call_data.get("bid") is None or call_data.get("ask") is None:
    #                 print("Call data incomplete — retrying...")
    #                 continue

    #             if put_data.get("bid") is None or put_data.get("ask") is None:
    #                 print("Put data incomplete — retrying...")
    #                 continue
    #         except Exception as e:
    #             print(f"Error getting option tick: {e}")
    #             continue


    #         timestamp = datetime.now(pytz.timezone('US/Eastern')).isoformat()

    #         # store call
    #         self.db.insert_tick({
    #             "timestamp": timestamp,
    #             "symbol": symbol,
    #             "expiry": expiry,
    #             "strike": call_strike,
    #             "right": "C",
    #             **call_data,
    #             "mid": None if call_data["bid"] is None else (call_data["bid"] + call_data["ask"]) / 2
    #         })

    #         # store put
    #         self.db.insert_tick({
    #             "timestamp": timestamp,
    #             "symbol": symbol,
    #             "expiry": expiry,
    #             "strike": put_strike,
    #             "right": "P",
    #             **put_data,
    #             "mid": None if put_data["bid"] is None else (put_data["bid"] + put_data["ask"]) / 2
    #         })


if __name__ == "__main__":
    manager = StrategyManager()
    manager.run()

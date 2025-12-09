import os
import json
import time
import random
import threading
from datetime import datetime, time
from typing import Any, Dict, Optional
import pandas as pd
import pytz
import numpy as np
import time as time_mod

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
        spot = self.broker.current_price(self.symbol, self.exchange)
        if spot is None:
            print("[Strategy] Spot unavailable.")
            return False

        atm = round(spot / self.strike_step) * self.strike_step
        self.atm = atm
        self.call_strike = atm + self.atm_call_offset * self.strike_step
        self.put_strike = atm - self.atm_put_offset * self.strike_step

        if self.enable_hedges:
            self.hedge_call_strike = atm + self.hedge_call_offset * self.strike_step
            self.hedge_put_strike = atm - self.hedge_put_offset * self.strike_step

        print(f"[Strategy] ATM={atm}, CALL={self.call_strike}, PUT={self.put_strike}")

        c_ohlc = self.broker.get_option_ohlc(self.symbol, self.expiry, self.call_strike, "C")
        p_ohlc = self.broker.get_option_ohlc(self.symbol, self.expiry, self.put_strike, "P")

        if c_ohlc.empty or p_ohlc.empty:
            print("[Strategy] Missing OHLC")
            return False

        c_ohlc["time"] = c_ohlc["time"].apply(lambda x: parse_ib_datetime(x, self.tz))
        p_ohlc["time"] = p_ohlc["time"].apply(lambda x: parse_ib_datetime(x, self.tz))

        df = c_ohlc.merge(p_ohlc, on="time", suffixes=("_call", "_put"))

        today = datetime.now(self.tz).date()
        market_open = datetime.now(self.tz).replace(hour=9, minute=30, second=0, microsecond=0)

        df = df[df["time"].dt.date == today]
        df = df[df["time"] >= market_open]

        if df.empty:
            return False

        df["combined_premium"] = df["close_call"] + df["close_put"]
        df["combined_volume"] = df["volume_call"] + df["volume_put"]
        df = df[df["combined_volume"] > 0]

        if df.empty:
            return False

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

    # SIGNAL GENERATION
    def generate_signals(self):
        now = datetime.now(self.tz).time()
        if not (self.entry_start <= now <= self.entry_end):
            return {"action": "NONE"}

        cp = self.broker.get_option_premium(self.symbol, self.expiry, self.call_strike, "C")
        pp = self.broker.get_option_premium(self.symbol, self.expiry, self.put_strike, "P")

        c_price = self._extract_price(cp)
        p_price = self._extract_price(pp)

        if c_price is None or p_price is None:
            return {"action": "NONE"}

        combined = c_price + p_price

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

        return {"action": "NONE"}

    # ORDER EXECUTION
    def execute_trade(self, signal):
        combined = signal["combined"]
        print(f"[Strategy] SELL STRADDLE @ {combined}")

        # ATM CALL
        c = self.broker.place_option_market_order(
            self.symbol, self.expiry, self.call_strike, "C", self.call_qty, "SELL"
        )
        fill_call = c["fill_price"]

        self.atm_call_id = create_position_entry(
            self.symbol, self.expiry, self.call_strike, "C",
            "SELL", self.call_qty,
            fill_call, c["order_id"], "ATM", self.tz,
            bid=0, ask=0
        )

        # ATM PUT
        p = self.broker.place_option_market_order(
            self.symbol, self.expiry, self.put_strike, "P", self.put_qty, "SELL"
        )
        fill_put = p["fill_price"]

        self.atm_put_id = create_position_entry(
            self.symbol, self.expiry, self.put_strike, "P",
            "SELL", self.put_qty,
            fill_put, p["order_id"], "ATM", self.tz,
            bid=0, ask=0
        )

        # Hedges
        if self.enable_hedges:

            hc = self.broker.place_option_market_order(
                self.symbol, self.expiry, self.hedge_call_strike, "C", self.hedge_qty, "BUY"
            )
            fill_hc = hc["fill_price"]

            self.otm_call_id = create_position_entry(
                self.symbol, self.expiry, self.hedge_call_strike, "C",
                "BUY", self.hedge_qty,
                fill_hc, hc["order_id"], "OTM", self.tz,
                bid=0, ask=0
            )

            hp = self.broker.place_option_market_order(
                self.symbol, self.expiry, self.hedge_put_strike, "P", self.hedge_qty, "BUY"
            )
            fill_hp = hp["fill_price"]

            self.otm_put_id = create_position_entry(
                self.symbol, self.expiry, self.hedge_put_strike, "P",
                "BUY", self.hedge_qty,
                fill_hp, hp["order_id"], "OTM", self.tz,
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

        # ================================
        # REAL IBKR PRICE UPDATE (COMMENTED)
        # Uncomment this block when using real IBKR premiums
        # ================================
        #
        # data = self.broker.get_option_premium(
        #     pos["symbol"],
        #     pos["expiry"],
        #     pos["strike"],
        #     pos["right"]
        # )
        #
        # bid = data.get("bid")
        # ask = data.get("ask")
        #
        # # Compute last_price like IBKR does
        # if bid is not None and ask is not None:
        #     last_price = (bid + ask) / 2
        # else:
        #     last_price = data.get("last") or data.get("mid")
        #
        # ================================
        # SIMULATION MODE (CURRENT MODE)
        # ================================
        bid, ask = simulated_bid_ask(pos["entry_price"])
        last_price = (bid + ask) / 2

        entry = pos["entry_price"]

        if pos["side"] == "SELL":
            pnl_pct = (entry - last_price) / entry
        else:
            pnl_pct = (last_price - entry) / entry

        pos["bid"] = bid
        pos["ask"] = ask
        pos["last_price"] = last_price
        pos["pnl_pct"] = pnl_pct
        pos["last_update"] = datetime.now(self.tz).isoformat()

        update_position_in_json(pos)


    # POSITION MANAGEMENT
    def manage_positions(self, poll_interval):
        while True:
            if not self.position_open:
                return None

            if self.atm_call_id: self._update_live_leg(self.atm_call_id)
            if self.atm_put_id: self._update_live_leg(self.atm_put_id)
            if self.enable_hedges and self.otm_call_id: self._update_live_leg(self.otm_call_id)
            if self.enable_hedges and self.otm_put_id: self._update_live_leg(self.otm_put_id)

            ac = get_position_by_id(self.atm_call_id)
            ap = get_position_by_id(self.atm_put_id)

            if not ac or not ap:
                time_mod.sleep(poll_interval)
                continue

            current = ac["last_price"] + ap["last_price"]
            entry_total = ac["entry_price"] + ap["entry_price"]

            pnl_pct = (entry_total - current) / entry_total

            now = datetime.now(self.tz).time()

            if now >= self.force_exit_time:
                return self.exit_position("FORCED EXIT")

            if pnl_pct >= self.tp_pct:
                return self.exit_position("TAKE PROFIT")

            if pnl_pct <= -self.sl_pct:
                return self.exit_position("STOP LOSS")

            time_mod.sleep(poll_interval)

    # EXIT
    def exit_position(self, reason):
        print(f"[Strategy] EXIT — {reason}")

        if self.atm_call_id:
            resp = self.broker.place_option_market_order(
                self.symbol, self.expiry, self.call_strike, "C", self.call_qty, "BUY"
            )
            self._close_leg(self.atm_call_id, resp)

        if self.atm_put_id:
            resp = self.broker.place_option_market_order(
                self.symbol, self.expiry, self.put_strike, "P", self.put_qty, "BUY"
            )
            self._close_leg(self.atm_put_id, resp)

        if self.enable_hedges:

            if self.otm_call_id:
                resp = self.broker.place_option_market_order(
                    self.symbol, self.expiry, self.hedge_call_strike, "C", self.hedge_qty, "SELL"
                )
                self._close_leg(self.otm_call_id, resp)

            if self.otm_put_id:
                resp = self.broker.place_option_market_order(
                    self.symbol, self.expiry, self.hedge_put_strike, "P", self.hedge_qty, "SELL"
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

        if entry is not None and close_price is not None:
            if pos["side"] == "SELL":
                pnl = (entry - close_price) / entry
            else:
                pnl = (close_price - entry) / entry
        else:
            pnl = 0

        pos["active"] = False
        pos["exit_time"] = now
        pos["close_price"] = close_price
        pos["order_id_exit"] = resp.get("order_id")
        pos["pnl_pct"] = pnl
        pos["last_update"] = now

        update_position_in_json(pos)

    # TEST MODE
    def test_place_and_exit_only_atm(self):
        print("\n========== TEST MODE ==========\n")

        call = self.broker.place_option_market_order(self.symbol, self.expiry, 6800, "C", 10, "SELL")
        put = self.broker.place_option_market_order(self.symbol, self.expiry, 6800, "P", 10, "SELL")

        entry_call = call["fill_price"]
        entry_put = put["fill_price"]

        self.atm_call_id = create_position_entry(
            self.symbol, self.expiry, 6800, "C",
            "SELL", 10, entry_call, call["order_id"], "ATM", self.tz
        )

        self.atm_put_id = create_position_entry(
            self.symbol, self.expiry, 6800, "P",
            "SELL", 10, entry_put, put["order_id"], "ATM", self.tz
        )

        self.position_open = True
        save_active_ids(True, self.atm_call_id, self.atm_put_id, None, None)

        for i in range(10):
            self._update_live_leg(self.atm_call_id)
            self._update_live_leg(self.atm_put_id)
            print(f"Updated {i+1}/10")
            time_mod.sleep(1)

        call_exit = self.broker.place_option_market_order(self.symbol, self.expiry, 6800, "C", 10, "BUY")
        put_exit = self.broker.place_option_market_order(self.symbol, self.expiry, 6800, "P", 10, "BUY")

        self._close_leg(self.atm_call_id, call_exit)
        self._close_leg(self.atm_put_id, put_exit)

        print("\n========== TEST COMPLETE ==========\n")

    def run(self, poll_interval=1):
        print("[Strategy] RUN LOOP STARTED")

        while self.manager.keep_running:

            # If a position is open → manage it
            if self.position_open:
                self.manage_positions(poll_interval)
                time_mod.sleep(poll_interval)
                continue
            
            self.load_config(self.config_path)
            # No position → fetch market data
            if not self.fetch_data():
                time_mod.sleep(poll_interval)
                continue

            # Calculate VWAP
            if not self.calculate_indicators():
                time_mod.sleep(poll_interval)
                continue

            # Generate signal
            signal = self.generate_signals()

            if signal["action"] == "SELL_STRADDLE":
                self.execute_trade(signal)
                continue

            time_mod.sleep(poll_interval)


# BROKER WRAPPER
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

    def place_option_market_order(self, symbol, expiry, strike, right, qty, action):
        print(f"[BROKER] MARKET ORDER → {action} {qty}x {symbol} {expiry} {strike}{right}")

        # REAL LOGIC WILL COME FROM IBKR FILLS
        fill_price = np.round(np.random.uniform(1.0, 50.0), 2)
        order_id = f"SIM_{int(time.time()*1000)}"

        return {"fill_price": fill_price, "order_id": order_id}


# STRATEGY MANAGER
class StrategyManager:
    def __init__(self):
        self.broker = StrategyBroker()
        self.strategy = Strategy(self, self.broker)
        self.keep_running = True

    def run(self):
        self.strategy.test_place_and_exit_only_atm()


if __name__ == "__main__":
    manager = StrategyManager()
    manager.run()

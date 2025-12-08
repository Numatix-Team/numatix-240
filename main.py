import json
import threading
from broker.ib_broker import IBBroker
import pandas as pd
import numpy as np
from db.db_logger import OptionDBLogger
import time
from datetime import datetime
import time as time_mod
import pytz
from log import setup_logger
setup_logger()
import os
from datetime import time as time_obj
from typing import Any, Dict, Optional
import random
import time as time_mod
import os
from datetime import datetime, time as time_obj
from typing import Any, Dict, Optional


class Strategy:
    SPX_STRIKE_STEP = 5   # SPX always uses 5-point strikes

    def __init__(self, manager, broker, config_path="config.json"):
        self.manager = manager
        self.broker = broker
        self.config_path = config_path

        # runtime active ids (kept in memory)
        self.otm_call_id: Optional[str] = None
        self.otm_put_id: Optional[str] = None
        self.atm_call_id: Optional[str] = None
        self.atm_put_id: Optional[str] = None

        # json write lock
        self.json_lock = threading.Lock()

        # Load config once
        self.load_config(config_path)

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

        # Ensure positions.json exists
        if not os.path.exists("positions.json"):
            with open("positions.json", "w") as f:
                json.dump({"positions": [], "active_positions": {"position_open": False}}, f, indent=4)

        # Load any active IDs from previous session
        self._load_positions()
        self._load_active_ids()


    # ======================================================
    # CONFIG LOADING
    # ======================================================
    def load_config(self, path=None):
        if path is None:
            path = self.config_path
        try:
            with open(path, "r") as f:
                cfg = json.load(f)
        except Exception as e:
            print("[Strategy] ERROR reading config.json:", e)
            return

        # Underlying
        self.symbol = cfg["underlying"]["symbol"]
        self.exchange = cfg["underlying"]["exchange"]
        self.currency = cfg["underlying"]["currency"]

        # Expiry
        self.expiry = cfg["expiry"]["date"]

        # Trade parameters
        tp = cfg["trade_parameters"]
        self.call_qty = tp["call_quantity"]
        self.put_qty = tp["put_quantity"]
        self.atm_call_offset = tp["atm_call_offset"]
        self.atm_put_offset = tp["atm_put_offset"]
        self.entry_vwap_mult = tp["entry_vwap_multiplier"]
        self.tp_pct = tp["take_profit"]
        self.sl_pct = tp["stop_loss"]
        self.max_spread = tp["max_bid_ask_spread"]

        # Time controls
        tc = cfg["time_controls"]
        self.entry_start = self._parse_time(tc["entry_start"])
        self.entry_end = self._parse_time(tc["entry_end"])
        self.force_exit_time = self._parse_time(tc["force_exit_time"])
        self.tz = pytz.timezone(tc["timezone"])

        # Hedging
        h = cfg["hedging"]
        self.enable_hedges = h["enable_hedges"]
        self.hedge_call_offset = h["hedge_call_offset"]
        self.hedge_put_offset = h["hedge_put_offset"]
        self.hedge_qty = h["hedge_quantity"]

        print("[Strategy] Config reloaded from config.json")


    # ======================================================
    # POSITIONS.JSON HELPERS
    # ======================================================
    def _load_positions_file(self):
        """Read positions.json safely."""
        if not os.path.exists("positions.json"):
            return {"positions": [], "active_positions": {"position_open": False}}

        try:
            with open("positions.json", "r") as f:
                return json.load(f)
        except:
            return {"positions": [], "active_positions": {"position_open": False}}

    def _save_positions_file(self, data):
        """Atomic write to positions.json."""
        with self.json_lock:
            tmp = "positions.json.tmp"
            with open(tmp, "w") as f:
                json.dump(data, f, indent=4)
            os.replace(tmp, "positions.json")


    # ======================================================
    # UTILITY METHODS
    # ======================================================
    def _parse_time(self, s):
        h, m = s.split(":")
        return time_obj(int(h), int(m))

    def _parse_ib_datetime(self, s):
        """Convert IBKR Central timestamp → Eastern."""
        try:
            parts = s.split(" ")
            date_part = parts[0]
            time_part = parts[1]
            naive = datetime.strptime(date_part + " " + time_part, "%Y%m%d %H:%M:%S")
            central = pytz.timezone("US/Central").localize(naive)
            return central.astimezone(self.tz)
        except:
            return pd.to_datetime(s)

    def _derive_fill_price(self, bid, ask, side):
        if side == "BUY":
            return ask if ask is not None else bid
        return bid if bid is not None else ask


    # ======================================================
    # POSITION CREATION / UPDATE
    # ======================================================
    def _generate_position_id(self, strike, right):
        time_str = datetime.now(self.tz).strftime("%Y-%m-%dT%H-%M-%S")
        epoch = int(time.time())
        return f"{epoch}_{self.symbol}_{strike}{right}_{time_str}"

    def _create_position_entry(self, strike, right, side, qty, bid, ask, order_id, position_type):
        data = self._load_positions_file()
        pos_id = self._generate_position_id(strike, right)
        now = datetime.now(self.tz).isoformat()
        entry_price = self._derive_fill_price(bid, ask, side)

        position = {
            "id": pos_id,
            "symbol": self.symbol,
            "expiry": self.expiry,
            "right": right,
            "position_type": position_type,
            "side": side,

            "strike": strike,
            "qty": qty,

            "active": True,
            "entry_time": now,
            "exit_time": None,

            "entry_price": entry_price,
            "entry_bid": bid,
            "entry_ask": ask,

            "last_price": entry_price,
            "close_price": None,

            "pnl_pct": 0.0,

            "order_id_entry": order_id,
            "order_id_exit": None,

            "last_update": now
        }

        data.setdefault("positions", []).append(position)
        self._save_positions_file(data)
        return pos_id

    def _get_position_by_id(self, pos_id):
        if pos_id is None:
            return None
        data = self._load_positions_file()
        for pos in data.get("positions", []):
            if pos.get("id") == pos_id:
                return pos
        return None

    def _update_position_in_json(self, updated_pos):
        if updated_pos is None:
            return
        data = self._load_positions_file()
        positions = data.get("positions", [])
        for i, pos in enumerate(positions):
            if pos["id"] == updated_pos["id"]:
                positions[i] = updated_pos
                data["positions"] = positions
                self._save_positions_file(data)
                return
        positions.append(updated_pos)
        data["positions"] = positions
        self._save_positions_file(data)


    # ======================================================
    # RANDOM TEST PRICE UPDATER
    # ======================================================
    def _update_live_leg(self, pos_id):
        pos = self._get_position_by_id(pos_id)
        if pos is None or not pos.get("active", False):
            return

        # Random simulated bid/ask
        base = pos.get("entry_price", 10)
        bid = base + random.uniform(-2, 2)
        ask = bid + random.uniform(0.1, 0.5)

        # mid
        last_price = (bid + ask) / 2

        entry_price = pos["entry_price"]

        if pos["side"] == "SELL":
            pnl_pct = (entry_price - last_price) / entry_price
        else:
            pnl_pct = (last_price - entry_price) / entry_price

        pos["last_price"] = last_price
        pos["pnl_pct"] = pnl_pct
        pos["last_update"] = datetime.now(self.tz).isoformat()
        pos["bid"] = bid
        pos["ask"] = ask

        self._update_position_in_json(pos)


    # ======================================================
    # POSITION EXITING
    # ======================================================
    def _close_position(self, pos_id, resp):
        pos = self._get_position_by_id(pos_id)
        if pos is None:
            return

        bid = resp.get("bid")
        ask = resp.get("ask")
        order_id = resp.get("order_id")

        if pos["side"] == "SELL":
            close_price = ask if ask is not None else bid
        else:
            close_price = bid if bid is not None else ask

        now = datetime.now(self.tz).isoformat()

        pos["active"] = False
        pos["exit_time"] = now
        pos["close_price"] = close_price
        pos["order_id_exit"] = order_id
        pos["last_update"] = now

        entry = pos["entry_price"]
        if entry is not None and close_price is not None:
            if pos["side"] == "SELL":
                pos["pnl_pct"] = (entry - close_price) / entry
            else:
                pos["pnl_pct"] = (close_price - entry) / entry

        self._update_position_in_json(pos)


    # ======================================================
    # ACTIVE ID MANAGEMENT
    # ======================================================
    def _save_active_ids(self):
        data = self._load_positions_file()
        data["active_positions"] = {
            "position_open": self.position_open,
            "atm_call_id": self.atm_call_id,
            "atm_put_id": self.atm_put_id,
            "otm_call_id": self.otm_call_id,
            "otm_put_id": self.otm_put_id
        }
        self._save_positions_file(data)

    def _load_active_ids(self):
        data = self._load_positions_file()
        active = data.get("active_positions", {})
        self.atm_call_id = active.get("atm_call_id")
        self.atm_put_id = active.get("atm_put_id")
        self.otm_call_id = active.get("otm_call_id")
        self.otm_put_id = active.get("otm_put_id")
        self.position_open = active.get("position_open", False)

        if self.position_open:
            self.position = {
                "atm_call_id": self.atm_call_id,
                "atm_put_id": self.atm_put_id,
                "otm_call_id": self.otm_call_id,
                "otm_put_id": self.otm_put_id
            }


    # ======================================================
    # DATA FETCHING / VWAP
    # ======================================================
    def fetch_data(self):
        spot = self.broker.current_price(self.symbol, self.exchange)
        if spot is None:
            print("[Strategy] Spot unavailable.")
            return False

        atm = round(spot / self.SPX_STRIKE_STEP) * self.SPX_STRIKE_STEP
        self.atm = atm

        self.call_strike = atm + (self.atm_call_offset * self.SPX_STRIKE_STEP)
        self.put_strike = atm - (self.atm_put_offset * self.SPX_STRIKE_STEP)

        if self.enable_hedges:
            self.hedge_call_strike = atm + (self.hedge_call_offset * self.SPX_STRIKE_STEP)
            self.hedge_put_strike = atm - (self.hedge_put_offset * self.SPX_STRIKE_STEP)

        print(f"[Strategy] ATM={atm}, Call={self.call_strike}, Put={self.put_strike}")

        c_ohlc = self.broker.get_option_ohlc(self.symbol, self.expiry, self.call_strike, "C")
        p_ohlc = self.broker.get_option_ohlc(self.symbol, self.expiry, self.put_strike, "P")

        if c_ohlc.empty or p_ohlc.empty:
            print("[Strategy] OHLC missing.")
            return False

        c_ohlc["time"] = c_ohlc["time"].apply(self._parse_ib_datetime)
        p_ohlc["time"] = p_ohlc["time"].apply(self._parse_ib_datetime)

        df = c_ohlc.merge(p_ohlc, on="time", suffixes=("_call", "_put"))

        today = datetime.now(self.tz).date()
        market_open = datetime.now(self.tz).replace(hour=9, minute=30, second=0, microsecond=0)

        df["time"] = df["time"].dt.tz_localize(self.tz)
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


    def calculate_indicators(self):
        df = self.hist_df
        df["turnover"] = df["combined_premium"] * df["combined_volume"].astype(float)

        tot_vol = df["combined_volume"].sum()
        if tot_vol == 0:
            return False

        self.vwap = float(df["turnover"].sum()) / float(tot_vol)
        print(f"[Strategy] VWAP={self.vwap:.2f}")

        return True


    # ======================================================
    # SIGNAL GENERATION
    # ======================================================
    def _extract_price(self, d):
        if not d:
            return None
        if d.get("mid") is not None:
            return d["mid"]
        bid = d.get("bid")
        ask = d.get("ask")
        if bid is not None and ask is not None:
            return (bid + ask) / 2
        if d.get("last") is not None:
            return d["last"]
        if bid is not None:
            return bid
        if ask is not None:
            return ask
        return None

    def _spread(self, d):
        if not d:
            return None
        if d.get("bid") is None or d.get("ask") is None:
            return None
        return d["ask"] - d["bid"]

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

        if (
            self._spread(cp) is None or self._spread(pp) is None or
            self._spread(cp) > self.max_spread or
            self._spread(pp) > self.max_spread
        ):
            return {"action": "NONE"}

        entry_threshold = self.vwap * self.entry_vwap_mult

        if not self.position_open and combined < entry_threshold:
            return {"action": "SELL_STRADDLE", "combined": combined}

        return {"action": "NONE"}


    # ======================================================
    # TRADE EXECUTION
    # ======================================================
    def execute_trade(self, signal):
        combined = signal["combined"]
        print(f"[Strategy] SELL STRADDLE @ combined={combined}")

        # ATM CALL
        call_resp = self.broker.place_option_market_order(
            self.symbol, self.expiry, self.call_strike, "C", self.call_qty, "SELL"
        )
        self.atm_call_id = self._create_position_entry(
            self.call_strike, "C", "SELL", self.call_qty,
            call_resp["bid"], call_resp["ask"], call_resp["order_id"], "ATM"
        )

        # ATM PUT
        put_resp = self.broker.place_option_market_order(
            self.symbol, self.expiry, self.put_strike, "P", self.put_qty, "SELL"
        )
        self.atm_put_id = self._create_position_entry(
            self.put_strike, "P", "SELL", self.put_qty,
            put_resp["bid"], put_resp["ask"], put_resp["order_id"], "ATM"
        )

        # HEDGES
        if self.enable_hedges:
            hc = self.broker.place_option_market_order(
                self.symbol, self.expiry, self.hedge_call_strike, "C", self.hedge_qty, "BUY"
            )
            self.otm_call_id = self._create_position_entry(
                self.hedge_call_strike, "C", "BUY", self.hedge_qty,
                hc["bid"], hc["ask"], hc["order_id"], "OTM"
            )

            hp = self.broker.place_option_market_order(
                self.symbol, self.expiry, self.hedge_put_strike, "P", self.hedge_qty, "BUY"
            )
            self.otm_put_id = self._create_position_entry(
                self.hedge_put_strike, "P", "BUY", self.hedge_qty,
                hp["bid"], hp["ask"], hp["order_id"], "OTM"
            )

        # Save active IDs
        self.position_open = True
        self.position = {
            "atm_call_id": self.atm_call_id,
            "atm_put_id": self.atm_put_id,
            "otm_call_id": self.otm_call_id,
            "otm_put_id": self.otm_put_id
        }
        self._save_active_ids()
        self._save_positions()


    # ======================================================
    # POSITION MONITORING
    # ======================================================
    def manage_positions(self, poll_interval):
        while True:
            if not self.position_open:
                return None

            # update active legs
            if self.atm_call_id:
                self._update_live_leg(self.atm_call_id)
            if self.atm_put_id:
                self._update_live_leg(self.atm_put_id)
            if self.enable_hedges and self.otm_call_id:
                self._update_live_leg(self.otm_call_id)
            if self.enable_hedges and self.otm_put_id:
                self._update_live_leg(self.otm_put_id)

            ac = self._get_position_by_id(self.atm_call_id)
            ap = self._get_position_by_id(self.atm_put_id)
            if not ac or not ap:
                time_mod.sleep(poll_interval)
                continue

            c_price = ac.get("last_price")
            p_price = ap.get("last_price")

            entry_combined = ac["entry_price"] + ap["entry_price"]
            current_combined = c_price + p_price

            pnl_pct = (entry_combined - current_combined) / entry_combined

            now = datetime.now(self.tz).time()

            if now >= self.force_exit_time:
                return self.exit_position("FORCED EXIT")

            if pnl_pct >= self.tp_pct:
                return self.exit_position("TAKE PROFIT")

            if pnl_pct <= -self.sl_pct:
                return self.exit_position("STOP LOSS")

            time_mod.sleep(poll_interval)


    # ======================================================
    # POSITION EXIT
    # ======================================================
    def exit_position(self, reason):
        print(f"[Strategy] EXIT — {reason}")

        if self.atm_call_id:
            resp = self.broker.place_option_market_order(
                self.symbol, self.expiry, self.call_strike, "C", self.call_qty, "BUY"
            )
            self._close_position(self.atm_call_id, resp)

        if self.atm_put_id:
            resp = self.broker.place_option_market_order(
                self.symbol, self.expiry, self.put_strike, "P", self.put_qty, "BUY"
            )
            self._close_position(self.atm_put_id, resp)

        if self.enable_hedges:
            if self.otm_call_id:
                resp = self.broker.place_option_market_order(
                    self.symbol, self.expiry, self.hedge_call_strike, "C", self.hedge_qty, "SELL"
                )
                self._close_position(self.otm_call_id, resp)
            if self.otm_put_id:
                resp = self.broker.place_option_market_order(
                    self.symbol, self.expiry, self.hedge_put_strike, "P", self.hedge_qty, "SELL"
                )
                self._close_position(self.otm_put_id, resp)

        self.atm_call_id = None
        self.atm_put_id = None
        self.otm_call_id = None
        self.otm_put_id = None

        self.position_open = False
        self.position = None

        self._save_active_ids()
        self._save_positions()

        return {"exit_reason": reason}


    # ======================================================
    # SAVE / LOAD POSITIONS SNAPSHOTS
    # ======================================================
    def _load_positions(self):
        if not os.path.exists("positions.json"):
            return
        try:
            data = self._load_positions_file()
            active = data.get("active_positions", {})
            if active.get("position_open"):
                self.position_open = True
                self.position = {
                    "atm_call_id": active.get("atm_call_id"),
                    "atm_put_id": active.get("atm_put_id"),
                    "otm_call_id": active.get("otm_call_id"),
                    "otm_put_id": active.get("otm_put_id")
                }
        except:
            pass

    def _save_positions(self):
        data = self._load_positions_file()
        data["active_positions"] = {
            "position_open": self.position_open,
            "atm_call_id": self.atm_call_id,
            "atm_put_id": self.atm_put_id,
            "otm_call_id": self.otm_call_id,
            "otm_put_id": self.otm_put_id
        }
        self._save_positions_file(data)


    # ======================================================
    # TEST MODE
    # ======================================================
    def test_place_and_exit_only_atm(self):
        print("\n========== TEST MODE ==========\n")

        call_resp = self.broker.place_option_market_order(self.symbol, self.expiry, 6800, "C", 10, "SELL")
        put_resp = self.broker.place_option_market_order(self.symbol, self.expiry, 6800, "P", 10, "SELL")
        hc_resp = self.broker.place_option_market_order(self.symbol, self.expiry, 6810, "C", 5, "BUY")
        hp_resp = self.broker.place_option_market_order(self.symbol, self.expiry, 6790, "P", 5, "BUY")

        self.atm_call_id = self._create_position_entry(6800, "C", "SELL", 10, call_resp["bid"], call_resp["ask"], call_resp["order_id"], "ATM")
        self.atm_put_id = self._create_position_entry(6800, "P", "SELL", 10, put_resp["bid"], put_resp["ask"], put_resp["order_id"], "ATM")

        self.otm_call_id = self._create_position_entry(6810, "C", "BUY", 5, hc_resp["bid"], hc_resp["ask"], hc_resp["order_id"], "OTM")
        self.otm_put_id = self._create_position_entry(6790, "P", "BUY", 5, hp_resp["bid"], hp_resp["ask"], hp_resp["order_id"], "OTM")

        self.position_open = True
        self._save_active_ids()

        print("Placed ATM & Hedge legs.")

        for i in range(10):
            self._update_live_leg(self.atm_call_id)
            self._update_live_leg(self.atm_put_id)
            self._update_live_leg(self.otm_call_id)
            self._update_live_leg(self.otm_put_id)
            print(f"Updated {i+1}/10")
            time_mod.sleep(1)

        print("Closing ATM legs...")

        call_exit = self.broker.place_option_market_order(self.symbol, self.expiry, 6800, "C", 10, "BUY")
        put_exit = self.broker.place_option_market_order(self.symbol, self.expiry, 6800, "P", 10, "BUY")

        self._close_position(self.atm_call_id, call_exit)
        self._close_position(self.atm_put_id, put_exit)

        self.atm_call_id = None
        self.atm_put_id = None
        self._save_active_ids()

        print("ATM legs closed. Hedging legs still open.")

        time_mod.sleep(10)

        print("Re-buying ATM legs...")

        new_c = self.broker.place_option_market_order(self.symbol, self.expiry, 6800, "C", 10, "BUY")
        new_p = self.broker.place_option_market_order(self.symbol, self.expiry, 6800, "P", 10, "BUY")

        self.atm_call_id = self._create_position_entry(6800, "C", "BUY", 10, new_c["bid"], new_c["ask"], new_c["order_id"], "ATM")
        self.atm_put_id = self._create_position_entry(6800, "P", "BUY", 10, new_p["bid"], new_p["ask"], new_p["order_id"], "ATM")

        self._save_active_ids()

        print("New ATM legs placed.")
        print("\n========== TEST COMPLETE ==========\n")

# ----------------------------
# StrategyBroker (stub / adapter)
# ----------------------------
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
        """
        Placeholder / stub. Real broker should return an object with:
            { "bid": float_or_none, "ask": float_or_none, "order_id": str }
        For now return a fake response for testing.
        """
        print(f"[BROKER] MARKET ORDER → {action} {qty}x {symbol} {expiry} {strike}{right}")
        # In real IBKR integration, use the fill price returned by IBKR. This stub
        # returns bid/ask approximately around a dummy mid using random jitter.
        # Replace this with your broker's actual return.
        mid = np.round(np.random.uniform(1.0, 50.0), 2)
        spread = 0.05 if mid > 10 else 0.02
        bid = np.round(mid - spread/2, 2)
        ask = np.round(mid + spread/2, 2)
        order_id = f"SIM_{int(time.time()*1000)}"
        return {"bid": bid, "ask": ask, "order_id": order_id}


# ----------------------------
# StrategyManager (wrap)
# ----------------------------
class StrategyManager:
    def __init__(self):
        self.broker = StrategyBroker()
        self.strategy = Strategy(self, self.broker)
        self.db = OptionDBLogger()
        self.keep_running = True

    def run(self):
        self.strategy.test_place_and_exit_only_atm()


if __name__ == "__main__":
    manager = StrategyManager()
    manager.run()

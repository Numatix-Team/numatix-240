import json
import time as time_mod
import pytz
import pandas as pd
import threading
from datetime import datetime
from ibapi.order import Order

from broker.ib_broker import IBBroker
from db.db_logger import OptionDBLogger
from log import setup_logger


# =========================================================
# STRATEGY
# =========================================================
import json
from datetime import datetime, time as time_obj
from typing import Any, Dict, List, Optional

import pandas as pd
import pytz


class Strategy:
    """
    VWAP-based short straddle strategy.

    Expected broker interface (StrategyBroker):
        - current_price(symbol: str, exchange: str) -> float | dict | None
        - get_option_tick(symbol: str, expiry: str, strike: float, right: str) -> dict | None
        - get_option_ohlc(symbol: str, expiry: str, strike: float, right: str,
                          duration: str = "1 D", bar: str = "1 min") -> pd.DataFrame
        - place_option_market_order(symbol: str, expiry: str, strike: float, right: str,
                                    qty: int, action: str) -> Any
    """

    def __init__(self,manager,broker, config_path: str = "config.json") -> None:
        self.broker = broker
        self.manager= manager

        # -------- Load config safely --------
        with open(config_path, "r") as f:
            cfg = json.load(f)

        underlying = cfg.get("underlying", {})
        expiry_cfg = cfg.get("expiry", {})
        tp_cfg = cfg.get("trade_parameters", {})
        vwap_cfg = cfg.get("vwap", {})

        self.symbol: str = underlying.get("symbol", "SPX")
        self.exchange: str = underlying.get("exchange", "SMART")
        self.currency: str = underlying.get("currency", "USD")
        self.expiry: str = expiry_cfg.get("date", "")
        self.strike_step: float = underlying.get("strike_step", 5.0)

        # Trade parameters
        self.call_quantity: int = tp_cfg.get("call_quantity", 1)
        self.put_quantity: int = tp_cfg.get("put_quantity", 1)
        self.lot_multiplier: int = tp_cfg.get("option_multiplier", 100)

        self.entry_vwap_factor: float = tp_cfg.get("entry_vwap_factor", 0.99)
        self.exit_vwap_factor: float = tp_cfg.get("exit_vwap_factor", 1.01)

        self.take_profit_points: float = tp_cfg.get("take_profit_points", 20.0)
        self.stop_loss_points: float = tp_cfg.get("stop_loss_points", 30.0)

        self.max_bid_ask_spread: float = tp_cfg.get("max_bid_ask_spread", 1.0)

        # Time controls
        self.tz = pytz.timezone(tp_cfg.get("timezone", "US/Eastern"))
        self.entry_start_time: time_obj = self._parse_time(
            tp_cfg.get("entry_start_time", "09:45")
        )
        self.entry_end_time: time_obj = self._parse_time(
            tp_cfg.get("entry_end_time", "14:30")
        )
        self.final_exit_time: time_obj = self._parse_time(
            tp_cfg.get("final_exit_time", "15:10")
        )

        # VWAP calc window
        self.hist_duration: str = vwap_cfg.get("hist_duration", "1 D")
        self.bar_size: str = vwap_cfg.get("bar_size", "5 mins")

        # Internal state
        self.atm_strike: Optional[float] = None
        self.hist_df: pd.DataFrame = pd.DataFrame()
        self.vwap: Optional[float] = None

        self.position_open: bool = False
        self.position: Optional[Dict[str, Any]] = None  # holds current position info

        # Results log (simple list of dicts; can be saved to CSV)
        self.results: List[Dict[str, Any]] = []


    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _parse_time(self, s: str) -> time_obj:
        """Parse 'HH:MM' string into time object."""
        hour, minute = s.split(":")
        return time_obj(int(hour), int(minute))

    def _round_to_step(self, price: float, step: float) -> float:
        """Round underlying spot to nearest strike step for ATM."""
        return round(price / step) * step

    def _extract_price_from_tick(self, tick: Any) -> Optional[float]:
        """
        Extract a usable price from a tick dict or raw float.
        Priority: mid(bid/ask) -> last -> close -> price.
        """
        if tick is None:
            return None
        if isinstance(tick, (int, float)):
            return float(tick)
        if not isinstance(tick, dict):
            return None

        bid = tick.get("bid")
        ask = tick.get("ask")
        if bid is not None and ask is not None:
            try:
                return (float(bid) + float(ask)) / 2.0
            except Exception:
                pass

        for key in ("last", "close", "price"):
            if key in tick and tick[key] is not None:
                try:
                    return float(tick[key])
                except Exception:
                    continue
        return None

    def _bid_ask_spread(self, tick: Any) -> Optional[float]:
        """Return bid-ask spread from tick dict, or None if not available."""
        if not isinstance(tick, dict):
            return None
        bid = tick.get("bid")
        ask = tick.get("ask")
        if bid is None or ask is None:
            return None
        try:
            return float(ask) - float(bid)
        except Exception:
            return None

    # ------------------------------------------------------------------
    # 1) Fetch data
    # ------------------------------------------------------------------
    def fetch_data(self) -> bool:
        """
        Fetch spot, identify ATM strike, and get historical OHLC
        for ATM call & put to build combined straddle series.
        """
        # --- Spot & ATM strike ---
        spot_tick = self.broker.current_price(self.symbol, self.exchange)
        spot = self._extract_price_from_tick(spot_tick)
        if spot is None:
            print("[Strategy] No valid spot price yet.")
            return False

        self.atm_strike = self._round_to_step(spot, self.strike_step)
        print(f"[Strategy] Spot={spot:.2f}, ATM strike={self.atm_strike}")

        # --- Historical OHLC for ATM call & put ---
        call_df = self.broker.get_option_ohlc(
            self.symbol,
            self.expiry,
            self.atm_strike,
            "C",
            duration=self.hist_duration,
            bar=self.bar_size,
        )
        put_df = self.broker.get_option_ohlc(
            self.symbol,
            self.expiry,
            self.atm_strike,
            "P",
            duration=self.hist_duration,
            bar=self.bar_size,
        )

        if call_df is None or put_df is None or call_df.empty or put_df.empty:
            print("[Strategy] Empty historical data for ATM options.")
            return False

        # Normalize time column
        if "time" in call_df.columns:
            call_df["time"] = pd.to_datetime(call_df["time"])
        if "time" in put_df.columns:
            put_df["time"] = pd.to_datetime(put_df["time"])

        # After merge we expect: close_call, close_put, volume_call, volume_put
        data = call_df.merge(put_df, on="time", suffixes=("_call", "_put"))

        required_cols = {"close_call", "close_put", "volume_call", "volume_put"}
        if not required_cols.issubset(set(data.columns)):
            raise ValueError(
                f"Historical data missing required columns: "
                f"{required_cols - set(data.columns)}"
            )

        data["combined_premium"] = data["close_call"] + data["close_put"]
        data["combined_volume"] = data["volume_call"] + data["volume_put"]
        data = data[data["combined_volume"] > 0].copy()

        if data.empty:
            print("[Strategy] No bars with positive combined volume.")
            return False

        self.hist_df = data
        return True

    # ------------------------------------------------------------------
    # 2) Calculate indicators (VWAP)
    # ------------------------------------------------------------------
    def calculate_indicators(self) -> bool:
        """
        Calculate VWAP of the combined straddle premium.
        VWAP = sum(price * volume) / sum(volume) over the historical window.
        """
        if self.hist_df is None or self.hist_df.empty:
            print("[Strategy] No historical data loaded. Call fetch_data() first.")
            return False

        df = self.hist_df.copy()
        df["turnover"] = df["combined_premium"] * df["combined_volume"]
        total_vol = df["combined_volume"].sum()
        if total_vol <= 0:
            print("[Strategy] Total volume is zero; cannot compute VWAP.")
            return False

        self.vwap = float(df["turnover"].sum() / total_vol)
        print(f"[Strategy] Computed VWAP={self.vwap:.2f}")
        return True

    # ------------------------------------------------------------------
    # 3) Generate signals
    # ------------------------------------------------------------------
    def generate_signals(self) -> Dict[str, Any]:
        """
        Use live combined premium vs VWAP to generate entry/exit intent.
        Currently only generates entry signals (short straddle) when no position is open.
        """
        now = datetime.now(self.tz)
        now_t = now.time()

        if not (self.entry_start_time <= now_t <= self.entry_end_time):
            return {"action": "NONE", "reason": "Outside entry window", "timestamp": now}

        if self.atm_strike is None or self.vwap is None:
            return {
                "action": "NONE",
                "reason": "ATM or VWAP not ready; run fetch_data() + calculate_indicators()",
                "timestamp": now,
            }

        # Live ticks for ATM call & put
        call_tick = self.broker.get_option_tick(
            self.symbol, self.expiry, self.atm_strike, "C"
        )
        put_tick = self.broker.get_option_tick(
            self.symbol, self.expiry, self.atm_strike, "P"
        )

        call_price = self._extract_price_from_tick(call_tick)
        put_price = self._extract_price_from_tick(put_tick)

        if call_price is None or put_price is None:
            return {"action": "NONE", "reason": "No live prices", "timestamp": now}

        combined_premium = call_price + put_price

        # Liquidity / spread check
        call_spread = self._bid_ask_spread(call_tick)
        put_spread = self._bid_ask_spread(put_tick)

        if (
            call_spread is None
            or put_spread is None
            or call_spread > self.max_bid_ask_spread
            or put_spread > self.max_bid_ask_spread
        ):
            return {
                "action": "NONE",
                "reason": "Bid-ask spread too wide / unavailable",
                "timestamp": now,
            }

        # If position already open, we don't generate new entry here
        if self.position_open:
            return {
                "action": "HOLD",
                "reason": "Position already open; manage_positions() will handle exits",
                "timestamp": now,
            }

        # Entry condition: combined premium < entry_vwap_factor * VWAP
        entry_threshold = self.entry_vwap_factor * self.vwap
        if combined_premium < entry_threshold:
            return {
                "action": "SELL_STRADDLE",
                "timestamp": now,
                "atm_strike": self.atm_strike,
                "combined_premium": combined_premium,
                "call_price": call_price,
                "put_price": put_price,
                "call_spread": call_spread,
                "put_spread": put_spread,
            }

        return {
            "action": "NONE",
            "reason": "Entry condition not satisfied",
            "timestamp": now,
        }

    # ------------------------------------------------------------------
    # 4) Execute trade
    # ------------------------------------------------------------------
    def execute_trade(self, signal: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Execute market orders based on generated signal.
        For SELL_STRADDLE: sell 1x ATM call + 1x ATM put (quantities configurable).
        """
        if signal.get("action") != "SELL_STRADDLE":
            return None

        if self.position_open:
            print("[Strategy] Position already open; ignoring entry signal.")
            return None

        atm_strike = signal["atm_strike"]
        entry_premium = signal["combined_premium"]
        now = signal["timestamp"]

        # Place sell orders for ATM call & put
        call_order_id = self.broker.place_option_market_order(
            self.symbol, self.expiry, atm_strike, "C", self.call_quantity, "SELL"
        )
        put_order_id = self.broker.place_option_market_order(
            self.symbol, self.expiry, atm_strike, "P", self.put_quantity, "SELL"
        )

        self.position_open = True
        self.position = {
            "atm_strike": atm_strike,
            "call_quantity": self.call_quantity,
            "put_quantity": self.put_quantity,
            "entry_premium": entry_premium,
            "vwap_at_entry": self.vwap,
            "entry_time": now,
            "call_order_id": call_order_id,
            "put_order_id": put_order_id,
        }

        record = {
            "timestamp": now.isoformat(),
            "event": "ENTRY",
            "atm_strike": atm_strike,
            "entry_premium": entry_premium,
            "vwap_at_entry": self.vwap,
            "call_order_id": call_order_id,
            "put_order_id": put_order_id,
        }
        self.results.append(record)

        print(
            f"[Strategy] Entered short straddle at strike {atm_strike}, "
            f"combined premium {entry_premium:.2f}"
        )
        return record

    # ------------------------------------------------------------------
    # 5) Manage positions (exits / monitoring)
    # ------------------------------------------------------------------
    def manage_positions(self) -> Optional[Dict[str, Any]]:
        """
            Monitor open position and apply exit logic:
                - Per-strike VWAP reversion: combined premium > exit_vwap_factor * VWAP
                - Overall TP/SL in points from entry premium
                - Time-based exit before final_exit_time
        """
        if not self.position_open or self.position is None or self.atm_strike is None:
            return None

        now = datetime.now(self.tz)
        now_t = now.time()

        # Live ticks
        call_tick = self.broker.get_option_tick(
            self.symbol, self.expiry, self.atm_strike, "C"
        )
        put_tick = self.broker.get_option_tick(
            self.symbol, self.expiry, self.atm_strike, "P"
        )

        call_price = self._extract_price_from_tick(call_tick)
        put_price = self._extract_price_from_tick(put_tick)

        if call_price is None or put_price is None:
            print("[Strategy] manage_positions: live prices not available.")
            return None

        combined_premium = call_price + put_price
        entry_premium = self.position["entry_premium"]
        pnl_points = entry_premium - combined_premium  # short straddle: lower premium = profit

        exit_reason = None

        # 1) Time-based exit – square off before cut-off
        if now_t >= self.final_exit_time:
            exit_reason = "TIME_EXIT"

        # 2) Hard stop-loss in points
        if exit_reason is None and pnl_points <= -self.stop_loss_points:
            exit_reason = "STOP_LOSS"

        # 3) Take profit in points
        if exit_reason is None and pnl_points >= self.take_profit_points:
            exit_reason = "TAKE_PROFIT"

        # 4) VWAP-based exit: combined premium > exit_vwap_factor * VWAP
        if exit_reason is None and self.vwap is not None:
            exit_threshold = self.exit_vwap_factor * self.vwap
            if combined_premium > exit_threshold:
                exit_reason = "VWAP_REVERSION"

        # If no exit condition met, just log and return
        if exit_reason is None:
            print(
                f"[Strategy] Holding position. PnL (pts)={pnl_points:.2f}, "
                f"combined premium={combined_premium:.2f}"
            )
            # Optionally update in-memory position metrics
            self.position["last_premium"] = combined_premium
            self.position["last_pnl_points"] = pnl_points
            self.position["last_update"] = now
            return None

        # --- Exit: buy back straddle with market orders ---
        qty_call = self.position["call_quantity"]
        qty_put = self.position["put_quantity"]

        call_close_id = self.broker.place_option_market_order(
            self.symbol, self.expiry, self.atm_strike, "C", qty_call, "BUY"
        )
        put_close_id = self.broker.place_option_market_order(
            self.symbol, self.expiry, self.atm_strike, "P", qty_put, "BUY"
        )

        record = {
            "timestamp": now.isoformat(),
            "event": "EXIT",
            "reason": exit_reason,
            "atm_strike": self.atm_strike,
            "entry_premium": entry_premium,
            "exit_premium": combined_premium,
            "pnl_points": pnl_points,
            "call_close_order_id": call_close_id,
            "put_close_order_id": put_close_id,
        }
        self.results.append(record)

        self.position_open = False
        self.position = None

        print(
            f"[Strategy] Exited position ({exit_reason}). "
            f"PnL (pts)={pnl_points:.2f}, exit premium={combined_premium:.2f}"
        )
        return record

    # ------------------------------------------------------------------
    # 6) Utility: save results to CSV
    # ------------------------------------------------------------------
    def save_results(self, path: str) -> None:
        """Save self.results into a CSV file."""
        if not self.results:
            print("[Strategy] No results to save.")
            return
        df = pd.DataFrame(self.results)
        df.to_csv(path, index=False)
        print(f"[Strategy] Results saved to {path}")

    
    def run(self, poll_interval: float = 1.0):
        print("[Strategy] Starting main run loop.")
        # 1) Ensure we have historical data + VWAP
        while self.manager.keep_running:
            try:
                ok = self.fetch_data()
                if not ok:
                    print("[Strategy] fetch_data failed — retrying in 2s")
                    time_mod.sleep(2)
                    continue

                ok2 = self.calculate_indicators()
                if not ok2:
                    print("[Strategy] calculate_indicators failed — retrying fetch")
                    time_mod.sleep(2)
                    continue

                # got VWAP and ATM strike
                break
            except Exception as e:
                print(f"[Strategy] Error during initial data load: {e}")
                time_mod.sleep(2)

        if not self.manager.keep_running:
            print("[Strategy] Manager requested stop during initialization.")
            return

        print("[Strategy] Initialization complete. Entering polling loop.")
        # 2) Poll loop for entry signals and position management
        try:
            while self.manager.keep_running:
                try:
                    signal = self.generate_signals()

                    if signal.get("action") == "SELL_STRADDLE":
                        print("[Strategy] Entry signal detected:", signal)
                        entry_record = self.execute_trade(signal)
                        # if execute_trade returns None, treat as failed entry
                        if entry_record is None:
                            print("[Strategy] execute_trade returned None — will continue polling.")
                        else:
                            # We have an open position — monitor it until closed
                            print("[Strategy] Position opened. Entering monitoring loop.")
                            while self.manager.keep_running and self.position_open:
                                try:
                                    exit_rec = self.manage_positions()
                                    # manage_positions returns a record when it exits
                                    if exit_rec:
                                        print("[Strategy] Position closed:", exit_rec)
                                        break
                                except Exception as me:
                                    print(f"[Strategy] Error in manage_positions: {me}")
                                time_mod.sleep(poll_interval)
                            # after closure, go back to collecting fresh VWAP/window if needed
                            # Optionally re-fetch historical data to refresh VWAP
                            try:
                                # small delay before recalculating
                                time_mod.sleep(1.0)
                                if self.fetch_data():
                                    self.calculate_indicators()
                            except Exception:
                                pass

                    else:
                        # No entry — just wait and poll again
                        # Optionally we can refresh VWAP at a lower frequency
                        # If you want to update VWAP continuously, uncomment below:
                        # if time_to_refresh_vwap(): self.fetch_data(); self.calculate_indicators()
                        pass

                except Exception as e:
                    print(f"[Strategy] Error in main poll loop: {e}")

                time_mod.sleep(poll_interval)

        except KeyboardInterrupt:
            print("[Strategy] KeyboardInterrupt received — stopping loop.")

        finally:
            print("[Strategy] Exiting run loop. Saving results.")
            try:
                # Save results with timestamped filename (optional)
                self.save_results("strategy_results.csv")
            except Exception:
                pass




# =========================================================
# BROKER
# =========================================================
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

# =========================================================
# MANAGER
# =========================================================
class StrategyManager:
    def __init__(self):
        setup_logger()
        self.keep_running = True
        self.broker = StrategyBroker()
        self.strategy = Strategy(self, self.broker, "config.json")



    def run(self):
        try:
            print("[Manager] Starting strategy.")
            self.strategy.run(poll_interval=1.0)   # adjust poll_interval as needed
        finally:
            print("[Manager] Strategy finished. Shutting down.")
            self.keep_running = False


if __name__ == "__main__":
    StrategyManager().run()
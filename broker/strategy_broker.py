import json
import random
import threading
import time
import pandas as pd
from .ib_broker import IBBroker


class StrategyBroker:
    """
    Thread-safe broker wrapper for IBKR API.
    Designed to be shared across multiple Strategy instances running in separate threads.
    Uses locks to ensure thread-safe access to shared resources.
    Handles disconnect/reconnect: blocks until connected, retries every 30s on connection loss.
    """
    RECONNECT_INTERVAL = 30
    PNL_CACHE_INTERVAL = 10  # seconds between global PnL list refreshes

    def __init__(self, config_path="config.json", test_mode=False):
        with open(config_path, "r") as f:
            self.config = json.load(f)
        self.test_mode = test_mode
        self._broker_lock = threading.Lock()
        # Shared PnL list: one background thread updates for all strategy threads
        self._pnl_cache = []
        self._pnl_cache_lock = threading.Lock()
        self._pnl_cache_account = None
        self._pnl_updater_stop = threading.Event()
        self._pnl_updater_thread = None
        # Skip IBKR connection in test mode
        if not test_mode:
            self._host = self.config["broker"]["host"]
            self._port = self.config["broker"]["port"]
            self._client_id = self.config["broker"]["client_id"]
            self.ib_broker = IBBroker(config_path=config_path)
            self.ib_broker.connect_to_ibkr(self._host, self._port, self._client_id)
        else:
            self._host = self._port = self._client_id = None
            self.ib_broker = None
            print("[StrategyBroker] Test mode - skipping IBKR connection")

        self.request_id_counter = 1
        self.counter_lock = threading.Lock()  # Thread-safe lock for request ID counter

    def ensure_connected(self):
        """Block until connection is alive, reconnecting every RECONNECT_INTERVAL if needed."""
        if self.test_mode or self.ib_broker is None:
            return
        while True:
            with self._broker_lock:
                if self.ib_broker.is_connection_alive():
                    return
                print("[StrategyBroker] Connection down, reconnecting in {}s...".format(self.RECONNECT_INTERVAL))
            time.sleep(self.RECONNECT_INTERVAL)
            with self._broker_lock:
                if self.ib_broker.is_connection_alive():
                    return
                try:
                    print("[StrategyBroker] Attempting reconnect...")
                    self.ib_broker.reconnect(self._host, self._port, self._client_id)
                    print("[StrategyBroker] Reconnected.")
                except Exception as e:
                    print(f"[StrategyBroker] Reconnect failed: {e}")

    def get_next_available_order_id(self):
        if self.test_mode:
            return 1000  # Mock order ID for test mode
        self.ensure_connected()
        try:
            return self.ib_broker.get_next_order_id_from_ibkr()
        except Exception as e:
            self.ib_broker.connected = False
            raise

    def reset_order_counter_to_next_available(self):
        next_id = self.get_next_available_order_id()
        if next_id:
            with self.counter_lock:
                self.request_id_counter = next_id - 2000
            print(f"Reset order counter to start from IBKR ID: {next_id}")

    def current_price(self, symbol, exchange, test_mode=False):
        if test_mode or self.test_mode:
            return round(random.uniform(680, 690), 2)
        self.ensure_connected()
        with self.counter_lock:
            req_id = self.request_id_counter + 2000
            self.request_id_counter += 1
            try:
                return self.ib_broker.get_index_spot(symbol, req_id, exchange)
            except Exception as e:
                self.ib_broker.connected = False
                raise

    def get_option_premium(self, symbol, expiry, strike, right, test_mode=False):
        if test_mode:
            return {
                "bid": round(random.uniform(0.3, 0.8), 2),
                "ask": round(random.uniform(0.4, 0.9), 2),
                "last": round(random.uniform(0.35, 0.85), 2),
                "mid": round(random.uniform(0.35, 0.85), 2)
            }
        self.ensure_connected()
        with self.counter_lock:
            req_id = self.request_id_counter + 3000
            self.request_id_counter += 1
            try:
                return self.ib_broker.get_option_premium(symbol, expiry, strike, right, req_id)
            except Exception as e:
                self.ib_broker.connected = False
                raise

    def get_option_tick(self, symbol, expiry, strike, right):
        self.ensure_connected()
        with self.counter_lock:
            req_id = self.request_id_counter + 5000
            self.request_id_counter += 1
            try:
                return self.ib_broker.get_option_tick(symbol, expiry, strike, right, req_id)
            except Exception as e:
                self.ib_broker.connected = False
                raise

    def get_option_ohlc(self, symbol, expiry, strike, right, duration="2 D", bar_size="1 min"):
        self.ensure_connected()
        with self.counter_lock:
            req_id = self.request_id_counter + 6000
            self.request_id_counter += 1
            try:
                x = self.ib_broker.get_option_ohlc(symbol, expiry, strike, right, duration, bar_size, req_id)
                return pd.DataFrame(x)
            except Exception as e:
                self.ib_broker.connected = False
                raise

    def get_all_open_positions_pnl(self, account=None):
        """Request PnL for all open positions from IBKR. Returns list of {symbol, expiry, strike, right, unrealized_pnl, realized_pnl, ...}."""
        if self.test_mode or self.ib_broker is None:
            return []
        self.ensure_connected()
        try:
            return self.ib_broker.get_all_open_positions_pnl(account=account)
        except Exception as e:
            self.ib_broker.connected = False
            raise

    def start_pnl_cache_updater(self, account):
        """Start a single background thread that refreshes the shared PnL list every PNL_CACHE_INTERVAL seconds.
        All strategy threads should use get_cached_pnl_list() instead of calling get_all_open_positions_pnl."""
        if self.test_mode or self.ib_broker is None:
            return
        self._pnl_cache_account = account
        # One immediate fetch so cache is populated right away
        try:
            with self._pnl_cache_lock:
                self._pnl_cache = list(self.get_all_open_positions_pnl(account=account))
            print(f"[StrategyBroker] PnL cache initial fetch: {len(self._pnl_cache)} positions")
        except Exception as e:
            print(f"[StrategyBroker] PnL cache initial fetch failed: {e}")
        if self._pnl_updater_thread is not None and self._pnl_updater_thread.is_alive():
            return
        self._pnl_updater_stop.clear()

        def _updater():
            while not self._pnl_updater_stop.wait(timeout=self.PNL_CACHE_INTERVAL):
                try:
                    fresh = self.get_all_open_positions_pnl(account=self._pnl_cache_account)
                    print(f"[StrategyBroker] PnL cache refreshed: {len(fresh)} positions: {fresh}")
                    with self._pnl_cache_lock:
                        self._pnl_cache = list(fresh)
                except Exception as e:
                    print(f"[StrategyBroker] PnL cache refresh failed: {e}")

        self._pnl_updater_thread = threading.Thread(target=_updater, name="PnLCacheUpdater", daemon=True)
        self._pnl_updater_thread.start()
        print(f"[StrategyBroker] PnL cache updater started (every {self.PNL_CACHE_INTERVAL}s)")

    def get_cached_pnl_list(self):
        """Return a snapshot of the shared PnL list (used by all strategy threads). Updated every PNL_CACHE_INTERVAL by background thread."""
        if self.test_mode:
            return []
        with self._pnl_cache_lock:
            return list(self._pnl_cache)

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

        self.ensure_connected()
        with self.counter_lock:
            req_id = self.get_next_available_order_id()

            try:
                order_id, fill_price = self.ib_broker.place_market_option_order(
                    symbol, exchange, expiry, strike, right, action, qty, req_id, wait_until_filled
                )
                return {"order_id": order_id, "fill_price": fill_price}
            except Exception as e:
                self.ib_broker.connected = False
                raise

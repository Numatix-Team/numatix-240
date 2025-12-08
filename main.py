import json
import threading
from broker.ib_broker import IBBroker
import pandas as pd
import numpy as np
from db.db_logger import OptionDBLogger
import time
from datetime import datetime
import pytz

class Strategy:
    def __init__(self,manager, broker, config_path="config.json"):
        self.broker = broker
        self.manager = manager
        
        with open(config_path, "r") as f:
            self.config = json.load(f)
        self.symbol = self.config["underlying"]["symbol"]
        self.exchange = self.config["underlying"]["exchange"]
        self.currency = self.config["underlying"]["currency"]
        self.expiry = self.config["expiry"]["date"]
        self.call_quantity = self.config["trade_parameters"]["call_quantity"]
        self.put_quantity = self.config["trade_parameters"]["put_quantity"]
        self.atm_call_offset = self.config["trade_parameters"]["atm_call_offset"]
        self.atm_put_offset = self.config["trade_parameters"]["atm_put_offset"]
        self.entry_vwap_multiplier = self.config["trade_parameters"]["entry_vwap_multiplier"]
        self.take_profit = self.config["trade_parameters"]["take_profit"]
        self.stop_loss = self.config["trade_parameters"]["stop_loss"]
        self.max_bid_ask_spread = self.config["trade_parameters"]["max_bid_ask_spread"]

    def run(self):
        price = self.broker.get_option_premium("SPX", "20251212", 6800, "C")
        print(price)

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
                self.request_id_counter = next_id - 2000  # Adjust counter to match order ID range
            print(f"Reset order counter to start from IBKR ID: {next_id}")
        else:
            print("Could not get next order ID from IBKR")

    def current_price(self, symbol, exchange):
        return self.ib_broker.get_index_spot(symbol, exchange)

    def get_option_premium(self, symbol, expiry, strike, right):
        with self.counter_lock:
            req_id = self.request_id_counter + 3000  # Offset for index data
            self.request_id_counter += 1    
        return self.ib_broker.get_option_premium(symbol, expiry, strike, right, req_id)

    def get_option_tick(self, symbol, expiry, strike, right):
        with self.counter_lock:
            req_id = self.request_id_counter + 5000
            self.request_id_counter += 1

        return self.ib_broker.get_option_tick(symbol, expiry, strike, right, req_id)

class StrategyManager:
    def __init__(self):
        self.broker = StrategyBroker()
        self.strategy = Strategy(self, self.broker)
        self.db = OptionDBLogger()
        self.keep_running = True

    def run(self):
        self.collection_thread = threading.Thread(target=self.collect_option_ticks)
        self.collection_thread.start()
        self.strategy.run()
    
    def collect_option_ticks(self):
        symbol = "SPX"
        expiry = "20251212"

        call_strike = 6800
        put_strike = 6800

        while self.keep_running:
            # GET CALL DATA
            try:
                call_data = self.broker.get_option_tick(symbol, expiry, call_strike, "C")
                put_data  = self.broker.get_option_tick(symbol, expiry, put_strike, "P")
                
                if call_data is None or put_data is None:
                    print("Option tick returned None — retrying...")
                    continue

                if call_data.get("bid") is None or call_data.get("ask") is None:
                    print("Call data incomplete — retrying...")
                    continue

                if put_data.get("bid") is None or put_data.get("ask") is None:
                    print("Put data incomplete — retrying...")
                    continue
            except Exception as e:
                print(f"Error getting option tick: {e}")
                continue


            timestamp = datetime.now(pytz.timezone('US/Eastern')).isoformat()

            # store call
            self.db.insert_tick({
                "timestamp": timestamp,
                "symbol": symbol,
                "expiry": expiry,
                "strike": call_strike,
                "right": "C",
                **call_data,
                "mid": None if call_data["bid"] is None else (call_data["bid"] + call_data["ask"]) / 2
            })

            # store put
            self.db.insert_tick({
                "timestamp": timestamp,
                "symbol": symbol,
                "expiry": expiry,
                "strike": put_strike,
                "right": "P",
                **put_data,
                "mid": None if put_data["bid"] is None else (put_data["bid"] + put_data["ask"]) / 2
            })

            time.sleep(1)   # 0.5 second collection interval

if __name__ == "__main__":
    
    manager = StrategyManager()
    manager.run()

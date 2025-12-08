
import time
import math

import pytz
import datetime as dt

from ib_insync import *
import pandas as pd

config_path="config.json"
with open(config_path, "r") as f:
    config = json.load(f)
host = config["broker"]["host"]
port = config["broker"]["port"]
client_id = config["broker"]["client_id"]

class IBTWSAPI:

    def __init__(self, host=host, port=port, client_id=client_id, 
                 currency=config["underlying"]["currency"], exchange=config["underlying"]["exchange"]):

        self.client = None
        self.host = host
        self.port = port
        self.client_id = client_id
        self.currency = currency
        self.exchange = exchange

    def connect(self) -> bool:
        """
        Connect the system with TWS account\n
        """
        # try:
        self.client = IB()
        self.ib = self.client
        self.client.connect(host=self.host, port=self.port, clientId=self.client_id, timeout=60)
        print("Connected")

    def is_connected(self) -> bool:
        """
        Get the connection status\n
        """
        return self.client.isConnected()

    def get_account_info(self):
        """
        Returns connected account info\n
        """
        account_info = self.client.accountSummary()
        return account_info

    def get_account_balance(self) -> float:
        """
        Returns account balance\n
        """
        for acc in self.get_account_info():
            if acc.tag == "AvailableFunds":
                return float(acc.value)

    def get_positions(self):
        return self.client.positions()

    def get_open_orders(self):
        x = self.client.reqOpenOrders()
        self.client.sleep(2)
        return x

    def close_all_open_orders(self):
        open_orders = self.client.reqOpenOrders()
        print(open_orders)
        for order in open_orders:
            self.client.cancelOrder(order=order.orderStatus)

    def place_market_order(self, contract, qty, side):
        buy_order = MarketOrder(side, qty)
        buy_trade = self.client.placeOrder(contract, buy_order)
        print("waiting for order to be placed")
        n = 1
        while True:
            if buy_trade.isDone():
                print("Order placed successfully")
                fill_price = buy_trade.orderStatus.avgFillPrice
                order_id = buy_trade.order.orderId
                print("Fill price:", fill_price)
                return buy_trade, fill_price, order_id
            else:
                print(f"Waiting...{contract.right}... {n} seconds")
                n += 1
                if n == 10:
                    return 0, 0, buy_trade.order.orderId
                time.sleep(1)

    def current_price(self, symbol, exchange='CBOE'):
        spx_contract = Index(symbol, exchange)

        market_data = self.client.reqMktData(spx_contract)
        self.ib.sleep(2)

        while util.isNan(market_data.last):
            self.ib.sleep(3)
        if market_data.last > 0:
            return market_data.last
        else:
            print("Market data is not subscribed or unavailable for", symbol)
            return None
    def get_stock_price(self, symbol, exchange="SMART"):
        stock_contract = Stock(symbol, exchange, self.currency)
        self.client.qualifyContracts(stock_contract)

        ticker = self.client.reqMktData(stock_contract, "", snapshot=True)

        while util.isNan(ticker.last) and util.isNan(ticker.marketPrice()):
            self.client.sleep(0.1)

        price = ticker.last if not util.isNan(ticker.last) else ticker.marketPrice()

        if price and price > 0:
            return price
        else:
            print(f"Market data is not subscribed or unavailable for {symbol}.")
            return None


    def get_historical_data(self, stock_symbol: str, num: str = '1 D', bar_size: str = '5 mins', exchange: str = 'SMART'):
            """
            Fetch historical OHLC data for a stock.

            :param stock_symbol: Stock ticker (e.g., 'AAPL')
            :param num: Duration string (e.g., '1 D' for 1 day, '1 W' for 1 week)
            :param bar_size: Bar size (e.g., '1 min', '5 mins', '1 day')
            :param exchange: Exchange to use (default 'SMART')
            :return: pandas DataFrame with columns: ['date', 'open', 'high', 'low', 'close', 'volume']
            """
            stock_contract = Stock(stock_symbol, exchange, self.currency)
            self.client.qualifyContracts(stock_contract)

            # Fetch historical data
            bars = self.client.reqHistoricalData(
                stock_contract,
                endDateTime='',
                durationStr=num,
                barSizeSetting=bar_size,
                whatToShow='TRADES',
                useRTH=True,
                formatDate=1
            )

            # Convert to DataFrame
            data = pd.DataFrame([{
                'date': bar.date,
                'open': bar.open,
                'high': bar.high,
                'low': bar.low,
                'close': bar.close,
                'volume': bar.volume
            } for bar in bars])

            return data

    def place_order(
            self,
            symbol: str,
            side: str,
            quantity: int,
            order_type: str = "MARKET",
            price: float = None,
            exchange: str = "SMART",
    ):
        # Build contract
        stock_contract = Stock(symbol, exchange, self.currency)
        self.client.qualifyContracts(stock_contract)

        # Choose order type
        if order_type.upper() == "MARKET":
            order = MarketOrder(side.upper(), quantity)
        elif order_type.upper() == "LIMIT":
            if price is None:
                raise ValueError("Price must be provided for LIMIT orders")
            order = LimitOrder(side.upper(), quantity, price)
        elif order_type.upper() == "STOP":
            if price is None:
                raise ValueError("Price must be provided for STOP orders")
            order = StopOrder(side.upper(), quantity, price)
        else:
            raise ValueError(f"Invalid order type: {order_type}")

        # Place order
        trade = self.client.placeOrder(stock_contract, order)

        for i in range(10):
            if trade.isDone():
                break
            print(f"Waiting for order to be filled... {i} seconds")
            self.client.sleep(1)
            if i == 9:
                self.cancel_order(trade.order.orderId)
        # Return only requested fields
        return {
            "orderId": trade.order.orderId,
            "status": trade.orderStatus.status,
            "avgFillPrice": trade.orderStatus.avgFillPrice
        }

    def get_quote(self, symbol: str, currency: str = 'USD', exchange: str = 'SMART'):
        contract = Stock(symbol, exchange, currency)
        self.client.qualifyContracts(contract)

        # Request snapshot data
        ticker = self.client.reqMktData(contract, snapshot=True)

        # Wait for both price and volume data to populate, sleep 0.1 while NaN (max 5 seconds)
        max_wait_time = 5.0  # Maximum wait time in seconds
        wait_time = 0.0
        while (util.isNan(ticker.marketPrice()) and util.isNan(ticker.last) and 
               util.isNan(ticker.volume) and wait_time < max_wait_time):
            self.client.sleep(0.1)
            wait_time += 0.1

        price = None
        if ticker.marketPrice() and ticker.marketPrice() > 0:
            price = ticker.marketPrice()
        elif ticker.last and ticker.last > 0:
            price = ticker.last

        volume = ticker.volume
        
        # Additional check: if volume is still NaN, try to get it again with a longer wait
        import math
        if math.isnan(volume):
            print(f"Volume still NaN for {symbol}, waiting longer...")
            additional_wait = 0.0
            while util.isNan(ticker.volume) and additional_wait < 2.0:
                self.client.sleep(0.2)
                additional_wait += 0.2
            volume = ticker.volume

        return {
            "symbol": symbol,
            "price": price,
            "volume": volume
        }

    def get_slo_bulk_quotes(self, symbols: list, max_retries: int = 3, retry_delay: float = 2.0):
        result = {}
        for i in symbols:
            quote = self.get_quote(i)
            print(quote)
            result[i] = quote
        return result

    def get_volume(self, symbol: str, exchange: str = "SMART", currency: str = "USD", duration: str = "2 D", bar_size: str = "1 day"):
        try:
            contract = Stock(symbol, exchange, currency)

            bars = self.client.reqHistoricalData(
                contract,
                endDateTime='',
                durationStr=duration,          # last X days
                barSizeSetting=bar_size,     # daily data
                whatToShow='TRADES',
                useRTH=True,                # regular trading hours only
                formatDate=1
            )

            if len(bars) < 1:
                print(f"Not enough data returned for {symbol}")
                return pd.DataFrame()  # Return empty DataFrame

            # Convert bars to DataFrame with date and volume
            data = []
            for bar in bars:
                data.append({
                    'date': bar.date,
                    'volume': bar.volume,
                    'open': bar.open,
                    'high': bar.high,
                    'low': bar.low,
                    'close': bar.close
                })
            
            df = pd.DataFrame(data)
            df.set_index('date', inplace=True)
            return df

        except Exception as e:
            print(f"Error fetching volume for {symbol}: {e}")
            return pd.DataFrame()  # Return empty DataFrame on error


    def get_bulk_quotes(self, symbols: list, currency: str = 'USD', batch_size: int = 50, exchange: str = 'SMART'):
        
        results = {}

        for i in range(0, len(symbols), batch_size):
            batch = symbols[i:i + batch_size]
            try:
                contracts = [Stock(sym, exchange, currency) for sym in batch]
                self.client.qualifyContracts(*contracts)

                tickers = [self.client.reqMktData(c, snapshot=True) for c in contracts]

                self.client.sleep(1)

                for ticker in tickers:
                    price = None
                    if ticker.marketPrice() and ticker.marketPrice() > 0:
                        price = ticker.marketPrice()
                    elif ticker.last and ticker.last > 0:
                        price = ticker.last

                    volume = ticker.volume

                    if price is not None:
                        results[ticker.contract.symbol] = {
                            "price": price,
                            "volume": volume
                        }
            except Exception as e:
                print(f"Skipping batch {batch}: {e}")

        return results

    def cancel_order(self, order_id: int) -> None:
        """
        Cancel open order\n
        """
        orders = self.client.reqOpenOrders()
        for order in orders:
            if order.orderStatus.orderId == order_id:
                self.client.cancelOrder(order=order.orderStatus)

    def place_bracket_order(
            self,
            symbol: str,
            quantity: int,
            price: float = ...,
            stoploss: float = None,
            targetprofit: float = None,
            expiry: str = None,
            strike: float = None,
            right: str = None,
            trailingpercent: float = False,
            convert_to_mkt_order_in: int = 0
    ) -> dict:
        get_exit_side = "BUY"
        c = self._create_contract(contract="options", symbol=symbol, exchange="EUREX", expiry=expiry, strike=strike,
                                  right=right)

        entry_order_info, stoploss_order_info, targetprofit_order_info = None, None, None
        parent_id = self.client.client.getReqId()

        en_order = LimitOrder(action="SELL", totalQuantity=quantity, lmtPrice=price)
        time.sleep(5)
        en_order.orderId = parent_id
        en_order.transmit = False

        def create_trailing_stop(quantity, parent_id=None):
            sl_order = Order()
            sl_order.action = get_exit_side
            sl_order.totalQuantity = quantity
            sl_order.orderType = "TRAIL"
            sl_order.trailingPercent = trailingpercent
            if parent_id:
                sl_order.parentId = parent_id
            sl_order.transmit = True
            return sl_order

        if trailingpercent:
            sl_order = create_trailing_stop(quantity, en_order.orderId)
        elif stoploss:
            sl_order = StopOrder(action=get_exit_side, totalQuantity=quantity, stopPrice=stoploss)
            sl_order.transmit = True

        entry_order_info = self.client.placeOrder(contract=c, order=en_order)
        self.client.sleep(1)
        if stoploss or trailingpercent:
            stoploss_order_info = self.client.placeOrder(contract=c, order=sl_order)
            print("waiting for order to be placed")
            n = 0
            while True:
                if entry_order_info.isDone():
                    print("Order placed successfully")
                    fill_price = entry_order_info.orderStatus.avgFillPrice
                    print("Fill price:", fill_price)
                    return {
                        "parent_id": parent_id,
                        "entry": entry_order_info,
                        "stoploss": stoploss_order_info,
                        "targetprofit": targetprofit_order_info,
                        "contract": c,
                        "order": sl_order,
                        "avgFill": fill_price,
                        "order_info": entry_order_info
                    }
                elif convert_to_mkt_order_in > 0 and n >= convert_to_mkt_order_in:  # Modified condition
                    print(f"Limit order not filled after {n} seconds, converting to market order")
                    market_order = MarketOrder(action="SELL", totalQuantity=quantity)
                    market_order.orderId = self.client.client.getReqId()
                    market_order.transmit = True

                    self.cancel_order(parent_id)
                    self.client.sleep(5)

                    entry_order_info = self.client.placeOrder(contract=c, order=market_order)
                    self.client.sleep(5)

                    if entry_order_info.isDone():
                        fill_price = entry_order_info.orderStatus.avgFillPrice
                        print("Market order filled at:", fill_price)

                        # Place trailing stop after market order fills
                        trailing_stop = create_trailing_stop(quantity)
                        stoploss_order_info = self.client.placeOrder(contract=c, order=trailing_stop)

                        return {
                            "parent_id": parent_id,
                            "entry": entry_order_info,
                            "stoploss": stoploss_order_info,
                            "targetprofit": targetprofit_order_info,
                            "contract": c,
                            "order": trailing_stop,
                            "avgFill": fill_price,
                            "order_info": entry_order_info
                        }
                else:
                    print(f"Waiting...{right}... {n + 1} seconds")
                    n += 1
                    time.sleep(1)
        else:
            print("Give Stoploss as one of the parameters")

    def cancel_order(self, order_id: int) -> None:
        """
        Cancel open order\n
        """
        orders = self.client.reqOpenOrders()
        for order in orders:
            if order.orderStatus.orderId == order_id:
                self.client.cancelOrder(order=order.orderStatus)

    def check_positions(self):
        x = self.get_positions()
        return x

    def cancel_positions(self):
        orders = self.get_open_orders()
        for order in orders:
            self.client.cancelOrder(order=order.orderStatus)
        positions = self.get_positions()
        for position in positions:
            if position.position < 0:
                print(position)
                action = "BUY"
                quantity = abs(position.position)
                contract = Option(
                    symbol=position.contract.symbol,
                    lastTradeDateOrContractMonth=position.contract.lastTradeDateOrContractMonth,
                    strike=position.contract.strike,
                    right=position.contract.right,
                    exchange="SMART",
                    currency=self.currency,
                    multiplier='100',
                )
                buy_order = MarketOrder(action, quantity)
                buy_trade = self.client.placeOrder(contract, buy_order)
                self.client.sleep(1)
                # self.place_market_order(contract=contract, qty=quantity, side=action)
                print(f"Closing position: {action} {quantity} {position.contract.localSymbol} at market")

    def cancel_all(self):
        orders = self.get_open_orders()
        for order in orders:
            self.client.cancelOrder(order=order.orderStatus)
        positions = self.get_positions()
        for position in positions:
            print(position)
            action = "SELL" if position.position > 0 else "BUY"
            quantity = abs(position.position)
            contract = Option(
                symbol=position.contract.symbol,
                lastTradeDateOrContractMonth=position.contract.lastTradeDateOrContractMonth,
                strike=position.contract.strike,
                right=position.contract.right,
                exchange="SMART",
                currency=self.currency,
                multiplier='100',
            )
            buy_order = MarketOrder(action, quantity)
            buy_trade = self.client.placeOrder(contract, buy_order)
            self.client.sleep(1)
            # self.place_market_order(contract=contract, qty=quantity, side=action)
            print(f"Closing position: {action} {quantity} {position.contract.localSymbol} at market")

    def close_all_positions(self, reason="manual_close"):
        """Close all open positions"""
        print(f"\n=== CLOSING ALL POSITIONS ({reason}) ===")
        
        # First cancel all open orders
        print("Cancelling all open orders...")
        self.close_all_open_orders()
        
        # Get all positions
        positions = self.get_positions()
        if not positions:
            print("No open positions found")
            return 0, 0.0
        
        closed_count = 0
        total_pnl = 0.0
        
        print(f"Found {len(positions)} open positions")
        
        for position in positions:
            try:
                symbol = position.contract.symbol
                current_position = position.position
                
                if current_position == 0:
                    continue
                
                print(f"Closing position for {symbol}: {current_position} shares")
                
                # Determine action based on position
                if current_position > 0:
                    # Long position - sell to close
                    action = "SELL"
                    quantity = current_position
                else:
                    # Short position - buy to close
                    action = "BUY"
                    quantity = abs(current_position)
                
                # Create market order to close position
                if hasattr(position.contract, 'symbol'):
                    # Stock position
                    contract = Stock(symbol, self.exchange, self.currency)
                else:
                    # Use existing contract
                    contract = position.contract
                
                # Place market order
                order = MarketOrder(action, quantity)
                trade = self.client.placeOrder(contract, order)
                
                print(f"Order placed: {action} {quantity} shares of {symbol}")
                
                # Wait for fill
                self.client.sleep(2)
                
                closed_count += 1
                
            except Exception as e:
                print(f"Error closing position for {symbol}: {e}")
                continue
        
        print(f"\n=== CLOSURE SUMMARY ===")
        print(f"Positions closed: {closed_count}")
        print(f"Total P&L: ${total_pnl:.2f}")
        print("=" * 50)
        
        return closed_count, total_pnl



if __name__ == "__main__":
    broker = IBTWSAPI()
    broker.connect()
    broker.close_all_positions()



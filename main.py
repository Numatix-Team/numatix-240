import os
import json
import time
import threading
import time as time_mod
from datetime import datetime
import argparse
from broker.strategy_broker import StrategyBroker
from strategy.strategy import Strategy
from log import setup_logger
from helpers.state_manager import is_account_paused, is_account_stopped, set_account_stopped
from helpers.positions import get_db

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
        nargs="?",
        default=None,
        metavar="START:END",
        help="Strike range as 'start:end' or 'start-end'; start/end can be negative (e.g. '-2:3')"
    )

    parser.add_argument(
        "--exclude",
        type=str,
        nargs="?",
        default="",
        metavar="LIST",
        help="Offsets to exclude from range (can be negative); e.g. -1,5,7 or -1;5;7"
    )

    return parser.parse_args()


def _resolve_nickname_to_ibkr(nickname):
    """Resolve account nickname to IBKR account ID from accounts.json. Returns nickname if not found (fallback)."""
    base_dir = os.path.dirname(os.path.abspath(__file__))
    path = os.path.join(base_dir, "accounts.json")
    if not os.path.exists(path):
        return nickname
    try:
        with open(path, "r") as f:
            data = json.load(f)
        accounts = data.get("accounts") or []
        for a in accounts:
            if a.get("nickname") == nickname:
                return a.get("ibkr_account_id", nickname)
    except Exception:
        pass
    return nickname


# STRATEGY MANAGER
class StrategyManager:
    def __init__(self, config_path="config.json", account="default", symbol_override=None, strike_range=None, current_atm_price=None, test_mode=False, ibkr_account_id=None):
        self.account = account 
        self.ibkr_account_id = ibkr_account_id if ibkr_account_id is not None else account
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
        # PnL limits from config (defaults 5000): drawdown when total PnL <= -limit, profit when total PnL >= limit
        tp_cfg = cfg.get("trade_parameters", {})
        self.drawdown_limit = float(tp_cfg.get("drawdown_limit", 5000))
        self.profit_limit = float(tp_cfg.get("profit_limit", 5000))
        self.drawdown_triggered = False
        self.profit_limit_triggered = False

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
                base_atm_price=self.current_atm_price,
                symbol=symbol_override,
                ibkr_account_id=self.ibkr_account_id
            )
            self.strategies.append(strategy)

        self.keep_running = True
        self.paused = False
        # Single background thread updates shared PnL list every 10s; all strategies use it
        if not self.test_mode:
            self.broker.start_pnl_cache_updater(self.ibkr_account_id)
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

        # PnL limit checker: every 30s query DB for total PnL; trigger drawdown or profit limit and stop
        def _pnl_limit_checker():
            while self.keep_running:
                time.sleep(30)
                if not self.keep_running:
                    break
                try:
                    db = get_db(self.account, self.symbol)
                    total_pnl = db.get_total_combined_pnl()
                    if total_pnl <= -self.drawdown_limit:
                        self.drawdown_triggered = True
                        set_account_stopped(self.account, self.symbol)
                        print(f"[Manager] DRAWDOWN LIMIT reached: total_pnl=${total_pnl:.2f} <= -${self.drawdown_limit}; stopping all strategies.")
                        break
                    if total_pnl >= self.profit_limit:
                        self.profit_limit_triggered = True
                        set_account_stopped(self.account, self.symbol)
                        print(f"[Manager] PROFIT LIMIT reached: total_pnl=${total_pnl:.2f} >= ${self.profit_limit}; exiting all positions and stopping.")
                        break
                except Exception as e:
                    print(f"[Manager] PnL limit check error: {e}")

        pnl_limit_thread = threading.Thread(target=_pnl_limit_checker, name="PnLLimitChecker", daemon=True)
        pnl_limit_thread.start()
        print(f"[Manager] PnL limit checker started (drawdown<=-${self.drawdown_limit}, profit>=${self.profit_limit}, check every 30s)")

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
    
    # args.account is nickname; use it for DB, PID, state. Resolve to ibkr_account_id for broker.
    account = args.account
    ibkr_account_id = _resolve_nickname_to_ibkr(account)
    symbol = args.symbol or "default"
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

    # Print full config once at startup
    try:
        with open(args.config, "r") as f:
            startup_config = json.load(f)
        print(f"[Main] Config file: {args.config}")
        print("[Main] Full config:")
        print(json.dumps(startup_config, indent=2))
    except Exception as e:
        print(f"[Main] Warning: Could not load/print config: {e}")

    # Parse strike range if provided (start/end can be negative, e.g. -2:3)
    # Support both ':' and '-' as separator (Windows may drop ':' in some launch scenarios)
    strike_range = [0]  # Default
    if args.range and args.range.strip():
        try:
            range_str = args.range.strip()
            if ":" in range_str:
                start, end = map(int, range_str.split(":", 1))
            elif "-" in range_str:
                # Support "4-10" or "-2-3" (negative start)
                parts = range_str.split("-", 2)
                if len(parts) == 2 and parts[0].strip() != "":
                    start, end = int(parts[0].strip()), int(parts[1].strip())
                elif len(parts) == 3 and parts[0].strip() == "":
                    start, end = -int(parts[1].strip()), int(parts[2].strip())
                else:
                    raise ValueError(f"Range format unclear: {range_str!r}")
            else:
                raise ValueError(f"Range must contain ':' or '-'")
            strike_range = list(range(start, end + 1))
            # Parse exclude list (comma, semicolon, or space separated; values can be negative)
            exclude_set = set()
            exclude_str = (args.exclude or "").strip()
            if exclude_str:
                for part in exclude_str.replace(";", ",").split(","):
                    for num in part.split():
                        token = num.strip()
                        if not token:
                            continue
                        try:
                            exclude_set.add(int(token))
                        except ValueError:
                            pass
                if exclude_set:
                    strike_range = [x for x in strike_range if x not in exclude_set]
                    strike_range.sort()
                    print(f"[Main] Excluded offsets: {sorted(exclude_set)} → strike range: {strike_range}")
            if not strike_range:
                print(f"[Main] Range empty after exclusions. Using default [0]")
                strike_range = [0]
            else:
                print(f"[Main] Parsed strike range: {strike_range}")
        except ValueError:
            print(f"[Main] Invalid range format '{args.range}'. Using default [0]")
            strike_range = [0]

    try:
        manager = StrategyManager(
            config_path=args.config,
            account=account,
            symbol_override=args.symbol,
            strike_range=strike_range,
            test_mode=False,
            ibkr_account_id=ibkr_account_id
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

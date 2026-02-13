"""
Standalone script to connect to IBKR and print PnL for all open (option) positions.
Usage:
  python print_pnl_list.py [--config config.json] [--account ACCOUNT]
"""
import argparse
import json
import os
import sys

# Ensure project root is on path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from broker.ib_broker import IBBroker


def main():
    parser = argparse.ArgumentParser(description="Print PnL list for all open positions from IBKR")
    parser.add_argument("--config", default="config.json", help="Path to config file")
    parser.add_argument("--account", default=None, help="Filter by account (optional)")
    args = parser.parse_args()

    if not os.path.exists(args.config):
        print(f"Config not found: {args.config}")
        return 1

    with open(args.config, "r") as f:
        config = json.load(f)

    host = config["broker"]["host"]
    port = config["broker"]["port"]
    client_id = 5

    print("Connecting to IBKR...")
    broker = IBBroker(config_path=args.config)
    broker.connect_to_ibkr(host, port, client_id)

    try:
        pnl_list = broker.get_all_open_positions_pnl(account=args.account)
    finally:
        try:
            broker.disconnect_from_ibkr()
        except Exception:
            pass
        print("Disconnected.")

    print("\n--- PnL list (open option positions) ---")
    if not pnl_list:
        print("(no option positions)")
        return 0

    for i, row in enumerate(pnl_list, 1):
        print(f"  {i}. {row.get('symbol')} {row.get('expiry')} {row.get('strike')}{row.get('right')}  "
              f"pos={row.get('pos')}  account={row.get('account')}  "
              f"unrealized_pnl={row.get('unrealized_pnl')}  realized_pnl={row.get('realized_pnl')}")
    print(f"\nTotal rows: {len(pnl_list)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

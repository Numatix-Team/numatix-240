import sqlite3
import threading
from typing import Optional, List, Dict


class PositionDB:
    def __init__(self, db_path=None, account=None, symbol=None):
        """
        Initialize PositionDB with account and symbol.
        If account and symbol are provided, uses account+symbol specific database.
        Otherwise uses the provided db_path (for backward compatibility).
        """
        if account and symbol:
            # Use account+symbol specific database
            self.db_path = f"positions_{account}_{symbol}.db"
        elif db_path:
            # Use provided path (backward compatibility)
            self.db_path = db_path
        else:
            # Default to old format for backward compatibility
            self.db_path = "positions.db"
        
        self.account = account
        self.symbol = symbol
        self.lock = threading.Lock()
        self._init_db()

    # ---------------------------
    # DB INIT
    # ---------------------------
    def _init_db(self):
        with self._connect() as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS positions (
                    id TEXT PRIMARY KEY,
                    account TEXT,
                    symbol TEXT,
                    expiry TEXT,
                    right TEXT,
                    position_type TEXT,
                    side TEXT,
                    strike REAL,
                    initial_qty INTEGER,
                    qty INTEGER,
                    active INTEGER,
                    entry_time TEXT,
                    exit_time TEXT,
                    entry_price REAL,
                    close_price REAL,
                    bid REAL,
                    ask REAL,
                    last_price REAL,
                    order_id_entry INTEGER,
                    order_id_exit INTEGER,
                    last_update TEXT,
                    realized_pnl REAL,
                    unrealized_pnl REAL
                )
            """)
            # Migrate: add new columns if they don't exist
            try:
                conn.execute("ALTER TABLE positions ADD COLUMN initial_qty INTEGER DEFAULT 0")
            except:
                pass
            try:
                conn.execute("ALTER TABLE positions ADD COLUMN realized_pnl REAL DEFAULT 0")
            except:
                pass
            try:
                conn.execute("ALTER TABLE positions ADD COLUMN unrealized_pnl REAL DEFAULT 0")
            except:
                pass
            # Migrate: remove old columns if they exist
            try:
                conn.execute("ALTER TABLE positions DROP COLUMN pnl_pct")
            except:
                pass
            try:
                conn.execute("ALTER TABLE positions DROP COLUMN pnl_value")
            except:
                pass
            conn.commit()

    def _connect(self):
        return sqlite3.connect(self.db_path, check_same_thread=False)

    # ---------------------------
    # INSERT
    # ---------------------------
    def insert_position(self, pos: Dict):
        with self.lock, self._connect() as conn:
            conn.execute("""
                INSERT INTO positions (
                    id, account, symbol, expiry, right, position_type, side,
                    strike, initial_qty, qty, active, entry_time, exit_time,
                    entry_price, close_price, bid, ask, last_price,
                    order_id_entry, order_id_exit, last_update,
                    realized_pnl, unrealized_pnl
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                pos["id"],
                pos["account"],
                pos["symbol"],
                pos["expiry"],
                pos["right"],
                pos["position_type"],
                pos["side"],
                pos["strike"],
                pos.get("initial_qty", pos.get("qty", 0)),
                pos["qty"],
                int(pos["active"]),
                pos["entry_time"],
                pos.get("exit_time"),
                pos["entry_price"],
                pos.get("close_price"),
                pos.get("bid"),
                pos.get("ask"),
                pos.get("last_price"),
                int(pos["order_id_entry"]) if pos.get("order_id_entry") is not None else None,
                int(pos.get("order_id_exit")) if pos.get("order_id_exit") is not None else None,
                pos["last_update"],
                pos.get("realized_pnl", 0),
                pos.get("unrealized_pnl", 0),
            ))
            conn.commit()

    # ---------------------------
    # UPDATE (GENERIC)
    # ---------------------------
    def update_position(self, pos_id: str, updates: Dict):
        if not updates:
            return

        fields = ", ".join(f"{k}=?" for k in updates.keys())
        values = list(updates.values()) + [pos_id]

        with self.lock, self._connect() as conn:
            conn.execute(
                f"UPDATE positions SET {fields} WHERE id=?",
                values
            )
            conn.commit()

    # ---------------------------
    # GET BY ID
    # ---------------------------
    def get_position(self, pos_id: str) -> Optional[Dict]:
        with self._connect() as conn:
            cur = conn.execute(
                "SELECT * FROM positions WHERE id=?",
                (pos_id,)
            )
            row = cur.fetchone()
            return self._row_to_dict(cur, row)

    # ---------------------------
    # GET ACTIVE POSITIONS
    # ---------------------------
    def get_active_positions(self, account: Optional[str] = None, symbol: Optional[str] = None) -> List[Dict]:
        query = "SELECT * FROM positions WHERE active=1"
        params = []

        if account:
            query += " AND account=?"
            params.append(account)
        
        if symbol:
            query += " AND symbol=?"
            params.append(symbol)

        with self._connect() as conn:
            cur = conn.execute(query, params)
            rows = cur.fetchall()
            return [self._row_to_dict(cur, r) for r in rows]

    # ---------------------------
    # GET ALL POSITIONS
    # ---------------------------
    def get_all_positions(self, account: Optional[str] = None) -> List[Dict]:
        query = "SELECT * FROM positions"
        params = []

        if account:
            query += " WHERE account=?"
            params.append(account)

        with self._connect() as conn:
            cur = conn.execute(query, params)
            rows = cur.fetchall()
            return [self._row_to_dict(cur, r) for r in rows]

    # ---------------------------
    # GET POSITIONS WITH FILTERS (for historical data)
    # ---------------------------
    def get_positions_with_filters(
        self, 
        account: Optional[str] = None,
        symbol: Optional[str] = None,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None
    ) -> List[Dict]:
        """Get positions with optional filters for account, symbol, and date range."""
        query = "SELECT * FROM positions WHERE 1=1"
        params = []

        if account:
            query += " AND account=?"
            params.append(account)

        if symbol:
            query += " AND symbol=?"
            params.append(symbol)

        if start_date:
            query += " AND entry_time >= ?"
            params.append(start_date)

        if end_date:
            # Include positions that were entered on or before end_date
            # end_date should be in format YYYY-MM-DD, we want to include the entire day
            query += " AND entry_time <= ?"
            params.append(f"{end_date} 23:59:59")

        with self._connect() as conn:
            cur = conn.execute(query, params)
            rows = cur.fetchall()
            return [self._row_to_dict(cur, r) for r in rows]

    # ---------------------------
    # CLOSE POSITION
    # ---------------------------
    def close_position(self, pos_id: str, close_data: Dict):
        close_data["active"] = 0
        self.update_position(pos_id, close_data)

    # ---------------------------
    # DELETE (optional)
    # ---------------------------
    def delete_position(self, pos_id: str):
        with self.lock, self._connect() as conn:
            conn.execute("DELETE FROM positions WHERE id=?", (pos_id,))
            conn.commit()

    # ---------------------------
    # PnL CALCULATIONS
    # ---------------------------
    def get_total_realized_pnl(self, account: Optional[str] = None, date_filter: Optional[str] = None) -> float:
        """Calculate total realized PnL (closed positions)."""
        query = "SELECT COALESCE(SUM(realized_pnl), 0) FROM positions WHERE active=0"
        params = []
        if account:
            query += " AND account=?"
            params.append(account)
        if date_filter:
            query += " AND DATE(exit_time) = ?"
            params.append(date_filter)
        
        with self._connect() as conn:
            cur = conn.execute(query, params)
            result = cur.fetchone()
            return float(result[0]) if result else 0.0

    def get_total_unrealized_pnl(self, account: Optional[str] = None, date_filter: Optional[str] = None) -> float:
        """Calculate total unrealized PnL (active positions)."""
        query = "SELECT COALESCE(SUM(unrealized_pnl), 0) FROM positions WHERE active=1"
        params = []
        if account:
            query += " AND account=?"
            params.append(account)
        if date_filter:
            query += " AND DATE(entry_time) = ?"
            params.append(date_filter)
        
        with self._connect() as conn:
            cur = conn.execute(query, params)
            result = cur.fetchone()
            return float(result[0]) if result else 0.0

    def get_total_combined_pnl(self, account: Optional[str] = None) -> float:
        """Calculate total combined PnL (realized + unrealized)."""
        realized = self.get_total_realized_pnl(account)
        unrealized = self.get_total_unrealized_pnl(account)
        return realized + unrealized

    # ---------------------------
    # ACTIVE POSITION IDS (for backward compatibility)
    # ---------------------------
    def get_active_position_ids(self, account: Optional[str] = None, symbol: Optional[str] = None) -> Dict:
        """Get active position IDs in the old format for compatibility."""
        active_positions = self.get_active_positions(account, symbol)
        
        # Find ATM and OTM positions
        atm_call_id = None
        atm_put_id = None
        otm_call_id = None
        otm_put_id = None
        
        for pos in active_positions:
            pos_type = pos.get("position_type", "")
            if pos_type == "ATM":
                if pos.get("right") == "C":
                    atm_call_id = pos["id"]
                elif pos.get("right") == "P":
                    atm_put_id = pos["id"]
            elif pos_type == "OTM":
                if pos.get("right") == "C":
                    otm_call_id = pos["id"]
                elif pos.get("right") == "P":
                    otm_put_id = pos["id"]
        
        return {
            "position_open": len(active_positions) > 0,
            "atm_call_id": atm_call_id,
            "atm_put_id": atm_put_id,
            "otm_call_id": otm_call_id,
            "otm_put_id": otm_put_id
        }

    # ---------------------------
    # UTILITY
    # ---------------------------
    def _row_to_dict(self, cursor, row):
        if row is None:
            return None
        return {d[0]: row[i] for i, d in enumerate(cursor.description)}

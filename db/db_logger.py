import sqlite3
import threading
import time
from datetime import datetime

class OptionDBLogger:
    def __init__(self, db_path="option_data.db"):
        self.db_path = db_path
        self._setup_db()

    def _setup_db(self):
        conn = sqlite3.connect(self.db_path)
        conn.execute("PRAGMA journal_mode=WAL;")
        
        conn.execute("""
            CREATE TABLE IF NOT EXISTS option_ticks (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT,
                symbol TEXT,
                expiry TEXT,
                strike REAL,
                right TEXT,
                bid REAL,
                ask REAL,
                last REAL,
                mid REAL,
                iv REAL,
                delta REAL,
                volume REAL,
                open_interest REAL
            )
        """)
        conn.commit()
        conn.close()

    def insert_tick(self, data: dict):
        """Each thread opens its own connection."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute("""
            INSERT INTO option_ticks 
            (timestamp, symbol, expiry, strike, right, bid, ask, last, mid, iv, delta, volume, open_interest)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            data["timestamp"], data["symbol"], data["expiry"], data["strike"], data["right"],
            data["bid"], data["ask"], data["last"], data["mid"], data["iv"],
            data["delta"], data["volume"], data["open_interest"]
        ))

        conn.commit()
        conn.close()

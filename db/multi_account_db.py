"""
MultiAccountDB - Wrapper to query across all account+symbol databases
"""
import os
import glob
from typing import Optional, List, Dict
from db.position_db import PositionDB


class MultiAccountDB:
    """
    Wrapper class that queries across all account+symbol specific databases.
    Automatically discovers all position database files.
    """
    
    def __init__(self):
        self.base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    
    def _discover_databases(self, account=None, symbol=None):
        """Discover all position database files matching the criteria."""
        pattern = os.path.join(self.base_dir, "positions_*.db")
        db_files = glob.glob(pattern)
        
        matching_dbs = []
        for db_file in db_files:
            # Extract account and symbol from filename: positions_acc1_SPX.db
            filename = os.path.basename(db_file)
            if not filename.startswith("positions_") or not filename.endswith(".db"):
                continue
            
            # Remove "positions_" prefix and ".db" suffix
            parts = filename[10:-3].split("_", 1)  # Split into [account, symbol]
            if len(parts) != 2:
                continue  # Skip old format files
            
            db_account = parts[0]
            db_symbol = parts[1]
            
            # Filter by account if specified
            if account and db_account != account:
                continue
            
            # Filter by symbol if specified
            if symbol and db_symbol != symbol:
                continue
            
            matching_dbs.append((db_account, db_symbol, db_file))
        
        return matching_dbs
    
    def get_all_positions(self, account: Optional[str] = None, symbol: Optional[str] = None) -> List[Dict]:
        """Get all positions from all matching databases."""
        all_positions = []
        matching_dbs = self._discover_databases(account, symbol)
        
        for db_account, db_symbol, db_file in matching_dbs:
            db = PositionDB(db_path=db_file)
            positions = db.get_all_positions()
            all_positions.extend(positions)
        
        return all_positions
    
    def get_active_positions(self, account: Optional[str] = None, symbol: Optional[str] = None) -> List[Dict]:
        """Get active positions from all matching databases."""
        all_positions = []
        matching_dbs = self._discover_databases(account, symbol)
        
        for db_account, db_symbol, db_file in matching_dbs:
            db = PositionDB(db_path=db_file)
            positions = db.get_active_positions()
            all_positions.extend(positions)
        
        return all_positions
    
    def get_positions_with_filters(
        self,
        account: Optional[str] = None,
        symbol: Optional[str] = None,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None
    ) -> List[Dict]:
        """Get positions with filters from all matching databases."""
        all_positions = []
        matching_dbs = self._discover_databases(account, symbol)
        
        for db_account, db_symbol, db_file in matching_dbs:
            db = PositionDB(db_path=db_file)
            # Get all positions from this database
            positions = db.get_all_positions()
            
            # Apply date filters
            if start_date or end_date:
                filtered = []
                for p in positions:
                    entry_time = p.get("entry_time", "")
                    
                    if start_date and entry_time < start_date:
                        continue
                    
                    if end_date:
                        # end_date should be in format YYYY-MM-DD, we want to include the entire day
                        if entry_time > f"{end_date} 23:59:59":
                            continue
                    
                    filtered.append(p)
                positions = filtered
            
            all_positions.extend(positions)
        
        return all_positions
    
    def get_total_realized_pnl(self, account: Optional[str] = None, date_filter: Optional[str] = None) -> float:
        """Calculate total realized PnL across all matching databases."""
        total = 0.0
        matching_dbs = self._discover_databases(account, None)  # symbol not used in this calculation
        
        for db_account, db_symbol, db_file in matching_dbs:
            db = PositionDB(db_path=db_file)
            total += db.get_total_realized_pnl(account=None, date_filter=date_filter)
        
        return total
    
    def get_total_unrealized_pnl(self, account: Optional[str] = None, date_filter: Optional[str] = None) -> float:
        """Calculate total unrealized PnL across all matching databases."""
        total = 0.0
        matching_dbs = self._discover_databases(account, None)  # symbol not used in this calculation
        
        for db_account, db_symbol, db_file in matching_dbs:
            db = PositionDB(db_path=db_file)
            total += db.get_total_unrealized_pnl(account=None, date_filter=date_filter)
        
        return total


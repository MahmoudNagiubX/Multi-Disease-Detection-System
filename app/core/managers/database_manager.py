import sqlite3
from typing import Any, Iterable, Optional

class DatabaseManager: # Encapsulation
    def __init__(self, db_path: str = "instance/app.db") -> None: 
        self.db_path = db_path          # Path to the SQLite database file
        
    def get_connection(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)    # Create and return a new SQLite connection
        conn.row_factory = sqlite3.Row
        return conn
    
    def execute(self, query: str, params: Iterable[Any] = ()) -> None:
        with self.get_connection() as conn:     # Execute an INSERT/UPDATE/DELETE query
            conn.execute(query, tuple(params))
            conn.commit()
    
    def execute_and_get_id(self, query: str, params: Iterable[Any] = ()) -> Optional[int]:
        """
        Execute an INSERT query and return the last_insert_rowid() from the same connection.
        This ensures reliable retrieval of the inserted row ID in SQLite.
        Returns None if the insert fails or no row ID is available.
        """
        with self.get_connection() as conn:
            cursor = conn.execute(query, tuple(params))
            row_id = cursor.lastrowid
            conn.commit()
            return row_id if row_id else None
            
    def fetch_one(self, query: str, params: Iterable[Any] = ()) -> Optional[sqlite3.Row]:
        with self.get_connection() as conn:     # Execute a SELECT query and return a single row
            cur = conn.execute(query, tuple(params))
            row = cur.fetchone()
        return row
    
    def fetch_all(self, query: str, params: Iterable[Any] = ()) -> list[sqlite3.Row]:
        with self.get_connection() as conn:     # Execute a SELECT query and return all rows as a list
            cur = conn.execute(query, tuple(params))
            rows = cur.fetchall()
        return rows
            
    def init_db(self) -> None:
        with self.get_connection() as conn:     # Initialize database tables if they do not exist
            cursor = conn.cursor()
            
            cursor.execute(     # USERS TABLE
            """ 
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                username TEXT NOT NULL UNIQUE,
                email TEXT NOT NULL UNIQUE,
                password_hash TEXT NOT NULL,
                created_at TEXT NOT NULL,
                updated_at TEXT,
                is_active INTEGER NOT NULL DEFAULT 1
            );
            """
        )
            cursor.execute(     # PREDICTION LOGS TABLE
            """
            CREATE TABLE IF NOT EXISTS prediction_logs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER NOT NULL,
                model_type TEXT NOT NULL,
                input_summary TEXT,
                prediction_result TEXT NOT NULL,
                probability REAL,
                created_at TEXT NOT NULL,
                FOREIGN KEY (user_id) REFERENCES users(id)
            );
            """
        )     
            cursor.execute(     # CHAT LOGS TABLE
            """
            CREATE TABLE IF NOT EXISTS chat_logs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER NOT NULL,
                role TEXT NOT NULL,
                message TEXT NOT NULL,
                created_at TEXT NOT NULL,
                FOREIGN KEY (user_id) REFERENCES users(id)
            );
            """
        )    
            conn.commit()    
            
db_manager = DatabaseManager()  # global instance rest of the app can use
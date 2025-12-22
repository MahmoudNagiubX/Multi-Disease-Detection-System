from dataclasses import dataclass
from typing import Optional, Any
from werkzeug.security import generate_password_hash, check_password_hash
from datetime import datetime, timezone

@dataclass
class User:     # Encapsulation
    id: Optional[int]
    username: str
    email: str
    password_hash: str
    created_at: str
    updated_at: Optional[str] = None
    is_active: int = 1

    @staticmethod
    def now_iso() -> str: # Return current time in ISO format (YYYY-MM-DDTHH:MM:SS)
        return datetime.now(timezone.utc).isoformat(timespec="seconds")
    
    def set_password(self, plain_password: str) -> None: # Hash and set the user's password hash
        self.password_hash = generate_password_hash(plain_password)
        
    def check_password(self, plain_password: str) -> bool: # Check if provided password matches the stored
        return check_password_hash(self.password_hash, plain_password)
    
    def to_dict(self) -> dict[str, Any]: # Safe dictionary representation
        return {
            "id": self.id,
            "username": self.username,
            "email": self.email,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "is_active": self.is_active,
        }
        
    @classmethod
    def from_row(cls, row: Any) -> "User": # Create a User instance from a sqlite3.Row object
        return cls(
            id = row["id"],
            username = row["username"],
            email = row["email"],
            password_hash = row["password_hash"],
            created_at = row["created_at"],
            updated_at = row["updated_at"],
            is_active = row["is_active"],
        )
from typing import Tuple, Union
from app.core.managers.database_manager import db_manager, DatabaseManager
from app.models.user.user import User
from app.services.base_service import BaseService


class AuthService(BaseService):  # Handles user registration, login, and password changes
    def __init__(self, db: DatabaseManager = db_manager) -> None:
        # Initialize shared BaseService database attribute
        super().__init__(db)
        # Preserve existing public attribute name for backward compatibility
        self.db = self._db
    
    def register(self, username: str, email: str, password: str) -> Tuple[bool, str]: # Register a new user
        
        # Basic validation
        if not username or len(username.strip()) == 0:
            return False, "Username cannot be empty."
        
        if not email or len(email.strip()) == 0:
            return False, "Email cannot be empty."
        
        if not password or len(password) < 6:
            return False, "Password must be at least 6 characters long."
        
        # Email format validation (basic check)
        email = email.strip().lower()
        if "@" not in email or "." not in email.split("@")[-1]:
            return False, "Please enter a valid email address."
        
        # Check if username already exists
        row = self.db.fetch_one(
            "SELECT * FROM users WHERE username = ?",
            (username,),
        )
        if row is not None:
            return False, "Username is already taken."

        # Check if email already exists
        row = self.db.fetch_one(
            "SELECT * FROM users WHERE email = ?",
            (email,),
        )
        if row is not None:
            return False, "Email is already registered."

        # Create new User instance
        created_at = User.now_iso()
        user = User(
            id = None,
            username = username,
            email = email,
            password_hash = "",
            created_at = created_at,
            updated_at = None,
            is_active = 1,
        )
        user.set_password(password)

        # Insert into database
        self.db.execute(
            """
            INSERT INTO users (username, email, password_hash, created_at, updated_at, is_active)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                user.username,
                user.email,
                user.password_hash,
                user.created_at,
                user.updated_at,
                user.is_active,
            ),
        )

        return True, "Registration successful. You can now log in."
    
    def login(
        self,
        identifier: str, # Username/Email
        password: str,
    ) -> Tuple[bool, Union[str, User]]:
       
        # Search by Email/Username
        row = self.db.fetch_one(
            "SELECT * FROM users WHERE email = ? OR username = ?",
            (identifier, identifier),
        )
        if row is None:
            return False, "User not found."

        user = User.from_row(row)

        if not user.check_password(password):
            return False, "Incorrect password."

        if not user.is_active:
            return False, "Account is deactivated."

        return True, user
    
auth_service = AuthService() # Global service instance
    

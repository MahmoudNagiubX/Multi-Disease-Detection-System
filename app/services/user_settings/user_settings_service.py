from __future__ import annotations
from typing import Optional, Tuple, Dict, Any
from app.core.managers.database_manager import db_manager, DatabaseManager
from app.models.user.user import User 
from app.services.base_service import BaseService


class UserSettingsService(BaseService): 
    #Handles:(Fetch basic profile/Change password/Clear prediction history/Delete account)
    def __init__(self, db: DatabaseManager = db_manager) -> None:
        super().__init__(db)
        # Provide a public alias consistent with other services if needed
        self.db = self._db

    def _fetch_user_instance(self, user_id: int) -> Optional[User]:
        row = self.db.fetch_one(
            """
            SELECT id, username, email, password_hash,
                   created_at, updated_at,
                   is_active
            FROM users
            WHERE id = ?
            """,
            (user_id,),
        )

        if row is None:
            return None

        # Use User.from_row() to create object
        return User.from_row(row)

    def get_profile(self, user_id: int) -> Optional[Dict[str, Any]]:
        user = self._fetch_user_instance(user_id)
        if user is None:
            return None

        return {
            "id": user.id,
            "username": user.username,
            "email": user.email,
            "created_at": user.created_at,
            "updated_at": user.updated_at,
        }

    def change_password(
        self,
        user_id: int,
        old_password: str,
        new_password: str,
        confirm_password: str,
    ) -> Tuple[bool, str]:

        if not old_password or not new_password or not confirm_password:
            return False, "All password fields are required."

        if new_password != confirm_password:
            return False, "New password and confirmation do not match."

        user = self._fetch_user_instance(user_id)
        if user is None:
            return False, "User not found."

        # Check old password using User.check_password
        if not user.check_password(old_password):
            return False, "Old password is incorrect."

        # Update password using User.set_password
        user.set_password(new_password)

        # Save updated hash
        self.db.execute(
            "UPDATE users SET password_hash = ?, updated_at = ? WHERE id = ?",
            (user.password_hash, User.now_iso(), user_id),
        )

        return True, "Password updated successfully."

    def clear_prediction_history(self, user_id: int) -> Tuple[bool, str]:
        self.db.execute(
            "DELETE FROM prediction_logs WHERE user_id = ?",
            (user_id,),
        )
        return True, "Prediction history cleared."

    def delete_account(self, user_id: int) -> Tuple[bool, str]:
        self.db.execute(
            "DELETE FROM users WHERE id = ?",
            (user_id,),
        )
        return True, "Your account has been deleted."

user_settings_service = UserSettingsService()

       
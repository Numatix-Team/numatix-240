import json
import os
from dataclasses import dataclass
from typing import Dict, Optional, Any

import bcrypt
import pyotp


AUTH_FILE_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "auth.json")


class AuthError(Exception):
    """Base authentication error."""


class InvalidAuthConfigError(AuthError):
    """Raised when auth.json has an unexpected structure."""


@dataclass
class AuthUser:
    username: str
    password_hash: str
    totp_secret: str


class AuthManager:
    """
    Thin authentication layer around auth.json that handles:
    - Loading user records
    - Verifying bcrypt-hashed passwords
    - Verifying TOTP codes

    Expected auth.json structure:

    {
      "users": {
        "aryaman": {
          "password": "$2b$12$hashedpasswordstring",
          "totp_secret": "JBSWY3DPEHPK3PXP"
        }
      }
    }
    """

    def __init__(self, auth_file: str = AUTH_FILE_PATH) -> None:
        self.auth_file = auth_file
        self._data = self._load_auth_file()

    def _load_auth_file(self) -> Dict[str, Any]:
        if not os.path.exists(self.auth_file):
            # For production, prefer provisioning auth.json out-of-band.
            # We intentionally do NOT auto-create users here.
            return {"users": {}}

        with open(self.auth_file, "r", encoding="utf-8") as f:
            data = json.load(f)

        if "users" not in data or not isinstance(data["users"], dict):
            raise InvalidAuthConfigError("auth.json must contain a 'users' object mapping usernames to credentials.")

        return data

    def reload(self) -> None:
        """Reload auth.json from disk (useful if credentials are rotated while the app is running)."""
        self._data = self._load_auth_file()

    def get_user(self, username: str) -> Optional[AuthUser]:
        users = self._data.get("users") or {}
        raw = users.get(username)
        if not isinstance(raw, dict):
            return None

        password_hash = raw.get("password")
        totp_secret = raw.get("totp_secret")

        if not password_hash or not totp_secret:
            return None

        return AuthUser(username=username, password_hash=password_hash, totp_secret=totp_secret)

    # ------------------------------------------------------------------
    # Public verification helpers
    # ------------------------------------------------------------------
    def verify_password(self, username: str, password: str) -> bool:
        """
        Verify a plaintext password against the bcrypt hash stored in auth.json.
        Returns False if the user does not exist or the password is invalid.
        """
        user = self.get_user(username)
        if not user:
            # Do not leak existence of usernames
            return False

        try:
            return bcrypt.checkpw(
                password.encode("utf-8"),
                user.password_hash.encode("utf-8"),
            )
        except ValueError:
            # In case the stored hash is malformed
            return False

    def verify_totp(self, username: str, otp_code: str, valid_window: int = 1) -> bool:
        """
        Verify a 6-digit TOTP code for a given user.
        valid_window allows for a small time drift (in 30s steps).
        """
        user = self.get_user(username)
        if not user:
            return False

        if not otp_code or not otp_code.isdigit() or len(otp_code) != 6:
            return False

        try:
            totp = pyotp.TOTP(user.totp_secret)
            return bool(totp.verify(otp_code, valid_window=valid_window))
        except Exception:
            return False

    # ------------------------------------------------------------------
    # Utility helpers (for provisioning / admin scripts)
    # ------------------------------------------------------------------
    @staticmethod
    def hash_password(plaintext_password: str) -> str:
        """
        Hash a plaintext password using bcrypt.
        Intended for use in out-of-band provisioning scripts or an admin console.
        """
        salt = bcrypt.gensalt()
        hashed = bcrypt.hashpw(plaintext_password.encode("utf-8"), salt)
        return hashed.decode("utf-8")


# A module-level singleton is convenient for Streamlit apps.
auth_manager = AuthManager()


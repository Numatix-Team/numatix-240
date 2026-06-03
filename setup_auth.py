#!/usr/bin/env python3
"""
One-time setup: create or update auth.json with a user (username, password, 2FA secret).
Password is hashed with bcrypt before saving. Run once per user you want to add.
"""
import json
import os
import sys

import bcrypt
import pyotp

AUTH_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "auth.json")


def main():
    print("Setup auth.json (username + password + 2FA secret)\n")
    username = input("Username: ").strip()
    if not username:
        print("Username required.")
        sys.exit(1)
    password = input("Password: ").strip()
    if not password:
        print("Password required.")
        sys.exit(1)
    totp_secret = input("TOTP secret (or press Enter to generate): ").strip()
    if not totp_secret:
        totp_secret = pyotp.random_base32()
        print(f"Generated TOTP secret: {totp_secret}")
        print("Add this in your authenticator app (Google Authenticator, Authy, etc.).\n")

    hashed = bcrypt.hashpw(password.encode("utf-8"), bcrypt.gensalt()).decode("utf-8")
    data = {"users": {}}
    if os.path.exists(AUTH_FILE):
        with open(AUTH_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
        if "users" not in data:
            data["users"] = {}
    data["users"][username] = {"password": hashed, "totp_secret": totp_secret}
    with open(AUTH_FILE, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
    print(f"Saved. User '{username}' can log in with this password and 2FA code.")


if __name__ == "__main__":
    main()

from typing import Optional

import streamlit as st

from auth import auth_manager
from frontend import main as dashboard_main


def _init_session_state() -> None:
    """Ensure all authentication-related keys exist in session_state."""
    defaults = {
        "authenticated": False,
        "auth_username": "",
        "auth_stage": "credentials",  # "credentials" -> "otp" -> "authenticated"
        "otp_attempts": 0,
        "otp_locked": False,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def _reset_auth_state() -> None:
    """Reset login state (used for logout or on hard failures)."""
    st.session_state.authenticated = False
    st.session_state.auth_username = ""
    st.session_state.auth_stage = "credentials"
    st.session_state.otp_attempts = 0
    st.session_state.otp_locked = False


def _render_logout() -> None:
    """Render a logout control in the sidebar when authenticated."""
    with st.sidebar:
        st.markdown("---")
        st.caption("Authentication")
        if st.button("Logout", type="secondary", use_container_width=True):
            _reset_auth_state()
            st.success("Logged out.")
            st.rerun()


def _render_credentials_step() -> None:
    """
    Step 1: Username + password verification.
    On success, advance to OTP stage.
    """
    st.subheader("Sign in")

    username = st.text_input("Username", key="auth_username_input")
    password = st.text_input("Password", type="password", key="auth_password_input")
    submit = st.button("Continue", type="primary")

    if submit:
        # Always reload auth.json in case credentials changed on disk.
        auth_manager.reload()

        if not username or not password:
            st.error("Please enter both username and password.")
            return

        if not auth_manager.verify_password(username, password):
            # Do not reveal which field was incorrect to avoid leaking valid usernames.
            st.error("Invalid username or password.")
            return

        # Credentials are correct -> proceed to OTP stage.
        st.session_state.auth_username = username
        st.session_state.auth_stage = "otp"
        st.session_state.otp_attempts = 0
        st.session_state.otp_locked = False
        st.success("Enter your 6-digit code.")
        st.rerun()


def _render_otp_step(max_attempts: int = 5) -> None:
    """
    Step 2: OTP verification using pyotp.TOTP.
    Enforces a maximum number of attempts per session to mitigate brute-force attacks.
    """
    st.subheader("2FA code")

    username: Optional[str] = st.session_state.get("auth_username") or None
    if not username:
        # Defensive reset if state is inconsistent.
        _reset_auth_state()
        st.warning("Your session expired. Please sign in again.")
        st.rerun()

    attempts = int(st.session_state.get("otp_attempts", 0))
    locked = bool(st.session_state.get("otp_locked", False))

    if locked or attempts >= max_attempts:
        st.session_state.otp_locked = True
        st.error(
            f"Too many invalid OTP attempts (limit: {max_attempts}). "
            "Please restart your session or contact an administrator."
        )
        return

    otp_code = st.text_input("6-digit code", max_chars=6, key="auth_otp_input")
    st.caption(f"Attempts left: {max_attempts - attempts}")

    if st.button("Verify OTP", type="primary"):
        if not otp_code:
            st.error("Please enter your 6-digit OTP code.")
            return

        if auth_manager.verify_totp(username, otp_code):
            # Only now do we mark the session as authenticated.
            st.session_state.authenticated = True
            st.session_state.auth_stage = "authenticated"
            st.session_state.otp_attempts = 0
            st.session_state.otp_locked = False
            st.success("Successfully authenticated.")
            st.rerun()
            return

        # Failed verification -> increment attempts and potentially lock.
        st.session_state.otp_attempts = attempts + 1
        attempts = st.session_state.otp_attempts

        if attempts >= max_attempts:
            st.session_state.otp_locked = True
            st.error(
                f"Invalid OTP code. You have exceeded the maximum number of attempts ({max_attempts}). "
                "This session has been locked."
            )
        else:
            remaining_after = max_attempts - attempts
            st.error(f"Invalid OTP code. Attempts remaining: {remaining_after}")


def render_login_flow() -> None:
    """Top-level login controller that orchestrates the two-step authentication."""
    _init_session_state()

    st.title("Login")

    stage = st.session_state.get("auth_stage", "credentials")

    if stage == "credentials":
        _render_credentials_step()
    elif stage == "otp":
        _render_otp_step()
    elif stage == "authenticated" and st.session_state.get("authenticated", False):
        # Edge case: user is already authenticated but landed here (e.g., via direct URL).
        st.success("You are already signed in.")
    else:
        # Any inconsistent state -> reset for safety.
        _reset_auth_state()
        _render_credentials_step()


def main() -> None:
    st.set_page_config(
        page_title="Numatix Dashboard",
        page_icon="🔐",
        layout="wide",
    )

    _init_session_state()

    if not st.session_state.get("authenticated", False):
        render_login_flow()
        return

    # Authenticated: show logout control + main dashboard.
    _render_logout()
    dashboard_main()


if __name__ == "__main__":
    main()


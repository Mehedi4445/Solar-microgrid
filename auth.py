"""
utils/auth.py
-------------
Authentication helpers that wrap Streamlit session state.
"""

import streamlit as st
from database import verify_user


def login(username: str, password: str) -> bool:
    """
    Attempt login.
    On success stores user info in st.session_state and returns True.
    """
    user = verify_user(username, password)
    if user:
        st.session_state["logged_in"]  = True
        st.session_state["username"]   = user["username"]
        st.session_state["role"]       = user["role"]
        st.session_state["user_id"]    = user["id"]
        return True
    return False


def logout():
    """Clear authentication state."""
    for key in ["logged_in", "username", "role", "user_id"]:
        st.session_state.pop(key, None)


def is_logged_in() -> bool:
    return st.session_state.get("logged_in", False)


def current_role() -> str:
    return st.session_state.get("role", "")


def current_user() -> str:
    return st.session_state.get("username", "")


def require_login():
    """Redirect to login if not authenticated (call at top of each page)."""
    if not is_logged_in():
        st.warning("🔒 Please log in first.")
        st.stop()


def require_admin():
    """Stop execution if the current user is not an admin."""
    require_login()
    if current_role() != "admin":
        st.error("⛔ Admin access required.")
        st.stop()

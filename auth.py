"""Authentication and role-based access.

* Supabase mode: email/password via Supabase. The role comes from the user's
  `app_metadata.role`, which only an admin/service key can set (users cannot edit it).
* Demo mode (no Supabase configured): no login; a sidebar switch lets you showcase each role.
"""
from __future__ import annotations

from dataclasses import dataclass

import streamlit as st

import config

PERMISSIONS: dict[str, set[str]] = {
    "viewer": {"dashboard", "statistics"},
    "manager": {"dashboard", "ai_analyst", "rag", "statistics", "reports"},
    "analyst": {"dashboard", "ai_analyst", "rag", "evaluation", "sql", "cleaning", "forecast",
                "root_cause", "statistics", "copilot", "reports"},
    "admin": {"dashboard", "ai_analyst", "rag", "evaluation", "sql", "cleaning", "forecast",
              "root_cause", "relationships", "statistics", "copilot", "reports"},
}


@dataclass(frozen=True)
class User:
    id: str
    email: str
    role: str


def can(role: str, feature: str) -> bool:
    return feature in PERMISSIONS.get(role, set())


def mode() -> str:
    if config.SUPABASE_URL and config.SUPABASE_KEY:
        try:
            import supabase  # noqa: F401
            return "supabase"
        except ImportError:
            pass
    return "demo"


def _client():
    # New client per call on purpose: supabase-py keeps the signed-in session inside the
    # client object, so sharing one across Streamlit sessions would mix users up.
    from supabase import create_client
    return create_client(config.SUPABASE_URL, config.SUPABASE_KEY)


def _friendly(exc: Exception) -> str:
    msg = str(exc)
    low = msg.lower()
    if "invalid login" in low:
        return "Incorrect email or password."
    if "not confirmed" in low:
        return "Please confirm your email first (check your inbox)."
    if "already registered" in low or "already been registered" in low:
        return "An account with this email already exists."
    if "getaddrinfo" in low or "connect" in low or "timed out" in low:
        return "Could not reach Supabase. Check your internet connection and SUPABASE_URL."
    return f"Authentication failed: {msg[:200]}"


def _role_from(user) -> str:
    role = str((getattr(user, "app_metadata", None) or {}).get("role", "")).lower()
    return role if role in config.ROLES else config.DEFAULT_ROLE


def sign_in(email: str, password: str) -> User:
    res = _client().auth.sign_in_with_password({"email": email, "password": password})
    return User(id=str(res.user.id), email=res.user.email or email, role=_role_from(res.user))


def sign_up(email: str, password: str) -> None:
    _client().auth.sign_up({"email": email, "password": password})


def require_login() -> User:
    """Return the signed-in user, or render the login screen and stop the script."""
    if mode() == "demo":
        role = st.sidebar.selectbox("Demo role", config.ROLES, index=config.ROLES.index("admin"),
                                    help="Demo mode has no login. Switch roles to show what each one sees.")
        st.sidebar.caption("⚠️ Demo mode: Supabase is not configured, so authentication is off.")
        return User("demo", "demo@local", role)

    user = st.session_state.get("user")
    if user:
        return user

    _, mid, _ = st.columns([1, 2, 1])
    with mid:
        st.title("📊 InsightForge - AI Data Analyst")
        st.caption("Sign in to continue")
        tab_in, tab_up = st.tabs(["Log in", "Sign up"])
        with tab_in:
            with st.form("login_form"):
                email = st.text_input("Email")
                password = st.text_input("Password", type="password")
                if st.form_submit_button("Log in", type="primary"):
                    if not email or not password:
                        st.error("Email and password are required.")
                    else:
                        try:
                            st.session_state.user = sign_in(email.strip(), password)
                            st.rerun()
                        except Exception as exc:
                            st.error(_friendly(exc))
        with tab_up:
            with st.form("signup_form"):
                email = st.text_input("Email", key="su_email")
                password = st.text_input("Password (min 8 characters)", type="password", key="su_pw")
                if st.form_submit_button("Create account"):
                    if not email or len(password) < 8:
                        st.error("Enter an email and a password of at least 8 characters.")
                    else:
                        try:
                            sign_up(email.strip(), password)
                            st.success("Account created. Confirm your email if required, then log in.")
                        except Exception as exc:
                            st.error(_friendly(exc))
    st.stop()


def logout() -> None:
    for key in list(st.session_state.keys()):
        del st.session_state[key]
    st.rerun()

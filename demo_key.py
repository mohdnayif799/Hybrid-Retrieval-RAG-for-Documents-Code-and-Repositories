"""
demo_key.py — API key selection and shared-key rate limiting for the
public Streamlit demo.

Reads a shared Gemini key from the GOOGLE_API_KEY environment variable
(set this in your host's secret/environment config — e.g. Hugging Face
Space Secrets, or `export GOOGLE_API_KEY=...` locally).

A visitor can optionally supply their own key. Their key always wins over
the shared one and is never subject to the query limit. It is passed in
as a plain function argument on every call — this module never writes it
to disk, never logs it, and never stores it anywhere itself (not even in
st.session_state). Only the shared-key USAGE COUNT lives in
st.session_state, scoped to that browser session.

Usage in app.py:

    from demo_key import get_api_key, limit_reached, increment_query_count

    user_key = st.sidebar.text_input(
        "Your own Gemini API key (optional)", type="password"
    )

    if limit_reached(user_key):
        st.sidebar.error("Shared demo limit reached — enter your own key above to continue.")
        st.stop()

    api_key = get_api_key(user_key)
    if not api_key:
        st.error("No API key available.")
        st.stop()

    try:
        response = model.generate_content(prompt)  # your existing Gemini call
        increment_query_count(user_key)             # only counts on success
    except Exception as e:
        st.error(f"Gemini request failed: {e}")
"""

import os
import streamlit as st

SHARED_QUERY_LIMIT = 5
_COUNT_KEY = "_shared_key_query_count"


def _has_own_key(user_provided_key):
    """True when the visitor supplied a real (non-blank) key of their own."""
    return bool(user_provided_key and user_provided_key.strip())


def _shared_key():
    """The shared key from the environment, or None if it isn't configured."""
    return os.environ.get("GOOGLE_API_KEY") or None


def _count():
    return st.session_state.get(_COUNT_KEY, 0)


def limit_reached(user_provided_key=None):
    """
    Whether the shared-key quota is used up for this session.
    Always False when the visitor supplied their own key — the limit
    only applies to the shared key.
    """
    if _has_own_key(user_provided_key):
        return False
    return _count() >= SHARED_QUERY_LIMIT


def get_api_key(user_provided_key=None):
    """
    The key app.py should use for the next request.

    The visitor's own key, if given, always wins and is never limited.
    Otherwise the shared key is returned, unless its quota is used up or
    it isn't configured — either case returns None so app.py can stop
    cleanly instead of calling Gemini with a missing key.
    """
    if _has_own_key(user_provided_key):
        return user_provided_key.strip()
    if limit_reached(user_provided_key):
        return None
    return _shared_key()


def increment_query_count(user_provided_key=None):
    """
    Call once, right after a successful Gemini call. No-ops when the
    visitor's own key was used, since the shared quota doesn't apply to it
    — so a failed request never burns a query, and a BYO-key request
    never counts against the shared pool.
    """
    if not _has_own_key(user_provided_key):
        st.session_state[_COUNT_KEY] = _count() + 1


def queries_remaining(user_provided_key=None):
    """
    Optional convenience, not one of the three required functions — safe
    to ignore or delete. Shared-key queries left this session, for a UI
    hint like "3 of 5 free queries left". Returns None when the visitor
    is on their own key, since no limit applies to show.
    """
    if _has_own_key(user_provided_key):
        return None
    return max(0, SHARED_QUERY_LIMIT - _count())

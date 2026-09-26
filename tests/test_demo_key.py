"""demo_key.shared_key_configured: whether the server offers a shared key."""

import pytest

pytest.importorskip("streamlit")

from demo_key import shared_key_configured  # noqa: E402


def test_shared_key_configured_when_env_var_is_set(monkeypatch):
    monkeypatch.setenv("GOOGLE_API_KEY", "fake-key-for-tests")
    assert shared_key_configured() is True


@pytest.mark.parametrize("value", [None, ""])
def test_shared_key_not_configured_when_env_var_is_missing_or_blank(monkeypatch, value):
    if value is None:
        monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    else:
        monkeypatch.setenv("GOOGLE_API_KEY", value)
    assert shared_key_configured() is False

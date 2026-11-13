"""Pure current-tool default precedence, with no path or loading policy."""
from __future__ import annotations

import pytest

from lib.tooling_defaults import resolve_tool_default


@pytest.mark.parametrize("explicit,profile,legacy,expected", [
    ("cli", "profile", "legacy", "cli"),
    ("", "profile", "legacy", ""),
    ("  ./relative/../value  ", "profile", "legacy", "  ./relative/../value  "),
    ("cli", None, "", "cli"),
    (None, "profile", "legacy", "profile"),
    (None, "../profile/path", "legacy", "../profile/path"),
    (None, "", "legacy", "legacy"),
    (None, None, "legacy", "legacy"),
    (None, "profile", "", "profile"),
    (None, "", "", ""),
    (None, None, "", ""),
])
def test_precedence_preserves_explicit_values(explicit, profile, legacy, expected):
    assert resolve_tool_default(explicit, profile, legacy) == expected


def test_calls_do_not_retain_prior_defaults():
    assert resolve_tool_default(None, "first-profile", "first-fallback") == "first-profile"
    assert resolve_tool_default(None, None, "second-fallback") == "second-fallback"
    assert resolve_tool_default("", "third-profile", "third-fallback") == ""
    assert resolve_tool_default(None, "", "") == ""

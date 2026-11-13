"""Pure precedence for explicitly supplied current-tool defaults."""
from __future__ import annotations


def resolve_tool_default(explicit: str | None, profile_default: str | None,
                         legacy_default: str) -> str:
    """Choose CLI > nonempty supplied profile value > caller fallback.

    Explicit values, including the empty string, are returned verbatim.
    """
    if explicit is not None:
        return explicit
    return profile_default or legacy_default

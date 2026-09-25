"""Project-neutral parsing for serialized record keys."""
from __future__ import annotations


def parse_record_key(key: str) -> dict[str, str]:
    """Parse a semicolon-delimited ``name=value`` record key into a dict.

    Empty input and non-record sentinel references return an empty mapping. The
    parser intentionally preserves the legacy spelling and whitespace of names
    and values.
    """
    if not key or key.startswith(("config:", "adapter:", "provenance:")):
        return {}
    out: dict[str, str] = {}
    for part in key.split(";"):
        if "=" not in part:
            return {}
        k, v = part.split("=", 1)
        out[k] = v
    return out

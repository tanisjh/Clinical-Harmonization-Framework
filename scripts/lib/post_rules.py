"""post_rules.py — vetted implementations of artifact-level adjustments.

Phase 2c, ABTC1603-B. A post-rule reads values the harmonization has already
produced and changes some of them, for named artifacts only. It cannot run
before the table exists, so it is not a mapping and does not live with them.

The split is the same one the field-level registry keeps: a function here
implements *how* an adjustment is carried out and knows nothing about what it
means. Every threshold, every delivered value and every field name arrives as a
declared parameter from the specification. A test asserts this module contains
no clinical term.

Each implementation returns a list of *events* rather than mutating anything.
The caller decides what to do with them, which is what makes the change
observable: a stage that quietly rewrote a table would leave a reader unable to
say what happened or why.
"""
from __future__ import annotations

from typing import Any

import pandas as pd


def landmark_split(frame: pd.DataFrame, *, input_field: str,
                   landmark_days: float, at_or_above: dict[str, Any],
                   below: dict[str, Any]) -> list[dict]:
    """Split rows on a numeric threshold and assign a value set to each side.

    Rows whose `input_field` is at or above the threshold take one set of field
    values; rows below take the other. **A row whose input is missing is left
    alone** — an absent measurement is not evidence for either side, and
    assigning one would invent a finding.

    Each side is declared as `{name: <branch label>, values: {field: value}}`.
    The label is the specification's word for what that side means; this
    function only carries it through to the event, so the delivered provenance
    can say which branch was taken without this module knowing what it means.

    Returns one event per (row, field) actually changed, carrying the value it
    supersedes, so the caller can record what the adjustment did.
    """
    if input_field not in frame.columns:
        raise KeyError(f"post rule reads {input_field!r}, which the assembled "
                       f"table does not have")
    numeric = pd.to_numeric(frame[input_field], errors="coerce")
    threshold = float(landmark_days)

    events: list[dict] = []
    for mask, side, predicate in (
            (numeric >= threshold, at_or_above, f"{input_field} >= {threshold:g}"),
            (numeric < threshold, below, f"{input_field} < {threshold:g}")):
        label = side.get("name", "")
        for field, value in (side.get("values") or {}).items():
            for idx in frame.index[mask]:
                events.append({
                    "row": idx,
                    "field": field,
                    "previous_value": (frame.at[idx, field]
                                       if field in frame.columns else None),
                    "value": value,
                    "branch": label,
                    "predicate": predicate,
                    "input_field": input_field,
                    "input_value": frame.at[idx, input_field],
                    "threshold": threshold,
                })
    return events

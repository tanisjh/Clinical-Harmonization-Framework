"""contributions.py — Structured provenance for multi-input derivations.

A single `raw_value` is the right representation for a value copied or mapped
from one source cell. It is the wrong representation for a value chosen from
several candidates: cramming them into one string
("measurement_a=missing; measurement_b=alpha")
produces something a human can read and a machine cannot use, and it loses which
candidate won and why the others did not.

This module adds the smallest backward-compatible alternative from the Phase 1
plan §7: a JSON `contributions` payload carried on the provenance row itself. No
new output file, no new join, no change to any existing column.

Each contribution answers: which file, which record, which column, what did it
say, what part did it play, and was it used?

Status vocabulary
  selected       this candidate supplied the final value
  unavailable    examined, but null/absent in the source
  rejected       examined and non-null, but the rule declined it
  not_evaluated  never examined (an earlier candidate already won)
  condition      informed which branch was taken, but supplied no value
  input          positional harmonized input passed to a registered field rule

`not_evaluated` is deliberately distinct from `rejected`: a first-non-null chain
short-circuits, so later candidates were not consulted at all. Recording them as
"rejected" would assert a judgement the pipeline never made.

`condition` is distinct from all four: ABTC1603's `os_time` is
`Days to Death if Vital Status == "DEAD" else Days to Last Contact`. Vital Status
is a genuine input — without it the choice cannot be explained — but it is not a
candidate for the value. Marking it `selected` would both break "at most one
selected" and claim a value it never supplied. Exactly one contribution may be
`selected`, and it must equal the final value.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from typing import Any, Iterable

from .replay_contract import encode_typed_raw
from .anchor_evidence import ANCHOR_RECORD_KINDS

STATUS_SELECTED = "selected"
STATUS_UNAVAILABLE = "unavailable"
STATUS_REJECTED = "rejected"
STATUS_NOT_EVALUATED = "not_evaluated"
STATUS_CONDITION = "condition"
STATUS_INPUT = "input"

VALID_STATUSES = frozenset({
    STATUS_SELECTED, STATUS_UNAVAILABLE, STATUS_REJECTED, STATUS_NOT_EVALUATED,
    STATUS_CONDITION, STATUS_INPUT,
})

#: Statuses whose contribution was read from the source and can be re-verified.
#: `not_evaluated` is excluded: the pipeline never looked at it.
VERIFIABLE_STATUSES = frozenset({
    STATUS_SELECTED, STATUS_UNAVAILABLE, STATUS_REJECTED, STATUS_CONDITION,
    STATUS_INPUT,
})


@dataclass(frozen=True)
class Contribution:
    """One candidate input to a multi-input derivation."""

    role: str                      # part played, e.g. "pfs_time_candidate"
    status: str                    # one of VALID_STATUSES
    source_column: str
    raw_value: Any = ""
    rank: int = -1                 # position in the candidate order (0-based)
    source_logical_label: str = ""
    source_resolved_path: str = ""
    source_sha256: str = ""
    source_sheet: str = ""
    source_record_key: str = ""
    source_row_number: Any = ""
    rule_id: str = ""              # selection/fallback rule that evaluated it
    #: For inputs that are themselves harmonized values rather than source cells
    #: (e.g. the pfs_time a post-pivot override reads), the anchor+field key of
    #: the provenance row that produced them. A contribution must identify its
    #: origin by EITHER a file (resolved_path + sha256) OR this reference.
    provenance_ref: str = ""
    source_ref_id: str = ""
    raw_value_json: str = ""
    evidence_refs: str = ""
    #: Route by which a grouping or external anchor supplied this record.
    #: Blank retains the legacy Contribution shape for non-anchor evidence.
    anchor_record_kind: str = ""
    #: Additional-record declaration ID for a synthesized anchor record.
    anchor_record_origin: str = ""
    #: Zero-based position after external-anchor filtering and reset_index.
    anchor_filtered_position: Any = ""

    def __post_init__(self) -> None:
        if self.status not in VALID_STATUSES:
            raise ValueError(
                f"invalid contribution status {self.status!r}; "
                f"expected one of {sorted(VALID_STATUSES)}"
            )
        if self.anchor_record_kind not in ANCHOR_RECORD_KINDS | {""}:
            raise ValueError(
                f"invalid anchor_record_kind {self.anchor_record_kind!r}")
        if self.anchor_record_origin and self.anchor_record_kind != "synthesized":
            raise ValueError(
                "anchor_record_origin requires anchor_record_kind='synthesized'")
        if self.anchor_record_kind == "synthesized" and not self.anchor_record_origin:
            raise ValueError(
                "synthesized anchor contribution requires anchor_record_origin")
        if self.anchor_record_kind == "external":
            if (isinstance(self.anchor_filtered_position, bool)
                    or not isinstance(self.anchor_filtered_position, int)
                    or self.anchor_filtered_position < 0):
                raise ValueError(
                    "external anchor contribution requires a non-negative "
                    "anchor_filtered_position")
        elif self.anchor_filtered_position not in ("", None):
            raise ValueError(
                "anchor_filtered_position requires anchor_record_kind='external'")

    def as_record(self) -> dict[str, Any]:
        d = asdict(self)
        for optional in ("source_ref_id", "raw_value_json", "evidence_refs",
                         "anchor_record_kind", "anchor_record_origin",
                         "anchor_filtered_position"):
            if d[optional] in ("", None):
                d.pop(optional)
        # Values come from pandas; make them JSON-safe without losing the text.
        v = d.get("raw_value")
        if v is None:
            d["raw_value"] = ""
        elif not isinstance(v, (str, int, float, bool)):
            d["raw_value"] = str(v)
        elif isinstance(v, float) and v != v:      # NaN
            d["raw_value"] = ""
        return d


def from_source_ref(ref, **kwargs) -> Contribution:
    """Build a Contribution, filling the source identity from a SourceRef."""
    if "raw_value" in kwargs and "raw_value_json" not in kwargs:
        kwargs["raw_value_json"] = encode_typed_raw(kwargs["raw_value"]).to_json()
    return Contribution(
        source_logical_label=ref.logical_label,
        source_resolved_path=ref.resolved_path,
        source_sha256=ref.sha256,
        source_sheet=ref.sheet or "",
        source_ref_id=ref.source_ref_id,
        **kwargs,
    )


def to_json(contributions: Iterable[Contribution]) -> str:
    """Serialize contributions for the provenance `contributions` column.

    Compact separators keep provenance_long.csv from bloating; sort_keys makes
    the output deterministic so golden comparisons stay stable.
    """
    items = [c.as_record() for c in contributions]
    if not items:
        return ""
    return json.dumps(items, separators=(",", ":"), sort_keys=True)


def from_json(payload: str) -> list[dict[str, Any]]:
    """Parse a `contributions` payload. Returns [] for an empty cell."""
    if not payload:
        return []
    return json.loads(payload)


def selected_of(payload: str) -> dict[str, Any] | None:
    """The contribution that supplied the final value, if any."""
    for c in from_json(payload):
        if c.get("status") == STATUS_SELECTED:
            return c
    return None

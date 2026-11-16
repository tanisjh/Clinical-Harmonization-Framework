"""
provenance.py — Data structures for tracking provenance of every extracted cell.

A `Cell` is one harmonized (trial, cimac_part_id, Cimac.id, Collection_Event, field) value
with full source attribution. Extractors emit a stream of Cells; the orchestrator
pivots them into the wide harmonized output and writes the long form as
provenance_long.csv.
"""
from __future__ import annotations

import hashlib
import json
import re
from dataclasses import asdict, dataclass, field
from typing import Any

PROVENANCE_SCHEMA_VERSION = 1
PROVENANCE_EVENT_REF_VERSION = 1
_PROVENANCE_EVENT_REF_RE = re.compile(r"^pev1-[0-9a-f]{64}$")


def canonical_row_identity(row_key, row) -> dict[str, str]:
    """Return the schema-derived, JSON-safe identity of one delivered row.

    Provenance CSV currently stores its compatibility anchors as text.  Render
    row-key values the same way so execution-time references and replay-time
    references agree without putting any domain-specific field names into the
    ``pev1`` contract.
    """
    out: dict[str, str] = {}
    for name in row_key:
        if name not in row:
            raise ValueError(f"row identity field {name!r} is absent")
        value = row[name]
        if value is None or (isinstance(value, float) and value != value):
            out[str(name)] = ""
        else:
            out[str(name)] = str(value)
    if not out:
        raise ValueError("row identity is empty")
    return out


def provenance_event_ref(*, run_id: str, spec_ref: str, mapping_id: str,
                         row_identity: dict[str, str],
                         harmonized_field: str) -> str:
    """Derive the closed ``pev1`` identity for one mapping provenance event."""
    if not all(isinstance(value, str) and value for value in (
            run_id, spec_ref, mapping_id, harmonized_field)):
        raise ValueError("pev1 identity requires run/spec/mapping/field")
    if not isinstance(row_identity, dict) or not row_identity:
        raise ValueError("pev1 identity requires a non-empty row_identity")
    if any(not isinstance(key, str) or not isinstance(value, str)
           for key, value in row_identity.items()):
        raise ValueError("pev1 row_identity keys and values must be strings")
    payload = {
        "harmonized_field": harmonized_field,
        "identity_version": PROVENANCE_EVENT_REF_VERSION,
        "mapping_id": mapping_id,
        "row_identity": row_identity,
        "run_id": run_id,
        "spec_ref": spec_ref,
    }
    encoded = json.dumps(
        payload, ensure_ascii=False, separators=(",", ":"), sort_keys=True,
    ).encode("utf-8")
    return f"pev1-{hashlib.sha256(encoded).hexdigest()}"


def is_provenance_event_ref(value: str) -> bool:
    """Whether *value* is exactly one closed version-1 event reference."""
    return bool(_PROVENANCE_EVENT_REF_RE.fullmatch(value or ""))


@dataclass
class Cell:
    trial:              str            # template `trial` value, e.g. "EAY131_Z1D"
    cimac_part_id:      str            # participant ID anchor
    Cimac_id:           str            # sample ID anchor (template's Cimac.id)
    Collection_Event:   str            # event anchor (template's Collection_Event)
    harmonized_field:   str            # one of the 20 template columns
    value:              Any            # extracted value (may be None)
    confidence:         float          # [0,1]
    source_file:        str            # path relative to trial dir
    source_column:      str            # column name in source
    # DEPRECATED (Phase 1e): a positional offset into a FILTERED, deduplicated
    # frame, so it is a reproduction key rather than a source locator. Retained
    # for compatibility; use source_row_number (row in the frame as read) plus
    # source_record_key. Do not add new consumers.
    source_row_idx:     int | str      # row index or row description
    extraction_method:  str            # "direct" | "value_map" | "derived" | ...
    notes:              str = ""       # free-text, optional

    # ── Phase 1a structured provenance (additive) ────────────────────────────
    # All default to "" so unmigrated extractors construct Cells unchanged and
    # simply leave these blank. Populated by CellFactory when it is
    # handed a SourceRef instead of a bare filename string.
    #
    # `source_file` above is retained verbatim for compatibility; it is a
    # LOGICAL LABEL (config column_map key), not a path claim. The authoritative
    # source location is source_resolved_path + source_sha256.
    source_logical_label: str = ""      # == source_file; named for clarity
    source_resolved_path: str = ""      # repo-relative path ACTUALLY opened
    source_sha256:        str = ""      # content hash of that file at read time
    source_sheet:         str = ""      # Excel sheet; "" for CSV
    source_row_number:    Any = ""      # row number in the frame AS READ (stable-ish)
    source_record_key:    str = ""      # business key, e.g. participant_key=SYNTH-001
    raw_value:            Any = ""      # verbatim source value, pre-transformation
    rule_id:              str = ""      # stable rule identifier, e.g. "direct_copy.v1"
    rule_version:         str = ""      # version of that rule
    # Multi-input derivations only (Phase 1b). JSON list of candidate inputs,
    # each with its own source locator, raw value, role and selected/rejected/
    # unavailable status. Empty for single-source values, which use raw_value.
    # See lib/contributions.py.
    contributions:        str = ""
    # Which output artifacts this row is authoritative for (semicolon-joined).
    # Empty = all artifacts in which the anchor is delivered, which is the
    # default and describes every row produced before Phase 1c.
    # Set explicitly when artifacts disagree — e.g. the ABTC1603 gold-only
    # clinical-benefit override. See lib/artifacts.py.
    authoritative_for:    str = ""
    # Rule parameters for derived/override events: threshold, predicate,
    # applicability and approval reference, as JSON. Distinct from
    # `contributions`, which describes the INPUTS rather than the rule.
    derivation:           str = ""
    # What happened to this value: delivered / transformed / provisional /
    # suppressed / row_excluded / source_missing / not_applicable.
    # Empty means delivered. See lib/value_states.py.
    value_state:          str = ""

    # Stage 0 strict-replay evidence bindings. Empty defaults preserve every
    # historical constructor and deliberately do not make old rows replayable.
    run_id:               str = ""
    spec_ref:             str = ""
    mapping_id:           str = ""
    source_ref_id:        str = ""
    raw_value_json:       str = ""
    evidence_refs:        str = ""

    def as_record(self) -> dict[str, Any]:
        d = asdict(self)
        # rename Cimac_id -> Cimac.id for output consistency with the template
        d["Cimac.id"] = d.pop("Cimac_id")
        return d


@dataclass
class FlagRow:
    """A row added to flagged_for_review.csv when confidence < threshold."""
    trial:                       str
    cimac_part_id:               str
    Cimac_id:                    str
    Collection_Event:            str
    harmonized_field:            str
    source_files:                str       # semicolon-joined
    candidate_source_variables:  str       # semicolon-joined
    observed_source_values:      str       # semicolon-joined
    proposed_mapping:            str
    confidence_score:            float
    reason_low_confidence:       str
    question_for_reviewer:       str

    def as_record(self) -> dict[str, Any]:
        d = asdict(self)
        d["Cimac.id"] = d.pop("Cimac_id")
        return d

"""suppression.py — provenance for a candidate the delivery threshold withheld.

A harmonized value can be absent for several different reasons, and they are not
the same fact. The source may have held nothing. The field may have no meaning
for this trial. The whole record may not belong. This module covers a fourth
case: the harmonization DID produce a value, and it was withheld because the
confidence it carries is below the threshold configured for that field.

Without a record of that decision the canonical table and the cell stream
disagree in a way a reader cannot resolve. The table shows nothing; the cell
that proposed a value is still sitting in provenance carrying it; and nothing
says which of "never found" and "found and deliberately withheld" happened.
Those two readings support opposite conclusions about the trial's data.

Suppression is a RUNTIME event, not a declaration
-------------------------------------------------
A specification cannot honestly say "this value will be suppressed". Doing so
would assert in advance what the source data contains. The decision needs three
things — a candidate value, the confidence attached to it, and the threshold in
force for that field — and all three exist only once the harmonization has run.
So the engine emits the event at the moment the decision is actually taken, and
no trial specification mentions suppression at all.

The decision itself belongs to the shared pivot, which already reports every
value it withheld. This module does not repeat that judgement. It turns what the
pivot reported into provenance, and it identifies the candidate cell by the
facts the pivot recorded about it rather than by re-running the choice — a
second implementation of the choice would be a second definition of it.

No artifact vocabulary
----------------------
The delivery layer records which published files a suppression applies to. The
canonical path has one table, so the question does not arise, and the artifact
list is deliberately absent rather than merely unimplemented. What is kept is
the part that is scientifically meaningful: what was proposed, where it came
from, how confident it was, which threshold applied, and that the threshold is
the reason it is not there.
"""
from __future__ import annotations

import json
from typing import Any

from lib import artifacts as artifact_names
from lib import rule_ids, value_states
from lib.contributions import STATUS_REJECTED, Contribution
from lib.contributions import to_json as contributions_json
from lib.provenance import Cell

#: The source a suppression cell cites. The decision IS its own source: no file
#: was read to make it, and borrowing the candidate's filename here would claim
#: the source said something it never said.
DECISION_SOURCE = "DERIVED:confidence_threshold"

#: The `extraction_method` and derivation `kind` the delivery layer established.
#: Reused rather than renamed, so one semantic identity covers both paths.
EXTRACTION_METHOD = "value_suppressed"
KIND = "threshold_suppression"

RATIONALE = ("value withheld because its extraction confidence is "
             "below the configured per-field threshold")


def threshold_config_path(field: str, thresholds: dict | None) -> str:
    """Where the effective threshold was read from, as it was actually read.

    A field with its own entry cites that entry; a field without one falls
    through to the default and says so. Naming a per-field key that does not
    exist would be a provenance claim a reader could not check.
    """
    per_field = (thresholds or {}).get("per_field") or {}
    if field in per_field:
        return f"confidence_thresholds.per_field.{field}"
    return "confidence_thresholds.default"


def _same_value(a: Any, b: Any) -> bool:
    """Equality that also holds for a value that is not equal to itself."""
    return a is b or a == b


def _candidate(cells: list[Cell], decision: dict) -> Cell | None:
    """The cell whose value the pivot withheld.

    Identified by what the pivot reported about it — the rule that produced it,
    the value it proposed, and the confidence it carried. That is recognition,
    not re-adjudication: which candidate wins has already been decided, and
    deciding it again here would let the two answers drift apart.

    The fallback exists only so that a cell stream this module cannot match
    still yields a suppression event with whatever source identity is available,
    rather than no event at all.
    """
    for c in cells:
        if (c.rule_id == decision["input_rule_id"]
                and _same_value(c.value, decision["proposed"])
                and float(c.confidence) == float(decision["confidence"])):
            return c
    return next((c for c in cells if c.source_resolved_path), None)


def record_canonical_suppression(suppressed: list[dict], cells: list[Cell], *,
                                 thresholds: dict | None = None,
                                 run_id: str = "", spec_ref: str = ""
                                 ) -> list[Cell]:
    """Write every threshold suppression the pivot reported into provenance.

    Two things happen, and both are what makes the absence legible:

      1. the decision gains a cell of its own, stating the candidate, its
         confidence, the threshold that applied and the rule that withheld it;
      2. the cells that proposed the value are **demoted** — they explain
         nothing the canonical table carries, and a cell still claiming a value
         the table does not hold is not merely incomplete, it is wrong.

    `suppressed` is the pivot's own report, one entry per withheld value.
    Returns the new cells; mutates the superseded ones in place.
    """
    if not suppressed:
        return []

    by_key: dict[tuple, list[Cell]] = {}
    for c in cells:
        by_key.setdefault((c.trial, c.cimac_part_id, c.Cimac_id,
                           c.Collection_Event, c.harmonized_field), []).append(c)

    new: list[Cell] = []
    for decision in suppressed:
        key = (decision["trial"], decision["cimac_part_id"], decision["Cimac.id"],
               decision["Collection_Event"], decision["harmonized_field"])
        field = key[4]
        candidates = by_key.get(key, [])
        for c in candidates:
            c.authoritative_for = artifact_names.NONE

        src = _candidate(candidates, decision)
        # The rejected candidate rides along as a contribution rather than as
        # the cell's own raw_value: `contributions` is the structure built to
        # say which input was examined, what it read and what became of it, and
        # "examined, non-null, declined" is exactly `rejected`.
        contributions = []
        if src is not None:
            contributions.append(Contribution(
                role=f"{field}_candidate", status=STATUS_REJECTED, rank=0,
                source_column=src.source_column, raw_value=src.raw_value,
                source_logical_label=src.source_logical_label,
                source_resolved_path=src.source_resolved_path,
                source_sha256=src.source_sha256,
                source_sheet=src.source_sheet,
                source_record_key=src.source_record_key,
                source_row_number=src.source_row_number,
                rule_id=src.rule_id))

        proposed = decision["proposed"]
        new.append(Cell(
            trial=key[0], cimac_part_id=key[1], Cimac_id=key[2],
            Collection_Event=key[3], harmonized_field=field,
            value=None, confidence=1.0,
            source_file=DECISION_SOURCE,
            source_column=field,
            source_row_idx=-1,
            extraction_method=EXTRACTION_METHOD,
            notes=(f"withheld: confidence {decision['confidence']:.2f} < "
                   f"threshold {decision['threshold']:.2f}; "
                   f"proposed value {proposed!r}"),
            source_logical_label=DECISION_SOURCE,
            source_resolved_path=(src.source_resolved_path if src else ""),
            source_sha256=(src.source_sha256 if src else ""),
            source_sheet=(src.source_sheet if src else ""),
            source_row_number=(src.source_row_number if src else ""),
            source_record_key=(src.source_record_key if src else ""),
            rule_id=rule_ids.VALUE_SUPPRESSED, rule_version="1",
            contributions=contributions_json(contributions),
            # One canonical table, so there is no artifact list to narrow. The
            # cells this decision supersedes carry the explicit "(none)".
            authoritative_for="",
            value_state=value_states.SUPPRESSED,
            run_id=run_id, spec_ref=spec_ref,
            derivation=json.dumps({
                "kind": KIND,
                "rule_id": rule_ids.VALUE_SUPPRESSED,
                "threshold": decision["threshold"],
                "confidence": decision["confidence"],
                "predicate": f"confidence < {decision['threshold']:.2f}",
                "withheld_value": "" if proposed is None else str(proposed),
                "input_rule_id": decision["input_rule_id"],
                "input_rule_version": (src.rule_version if src else ""),
                "final_value": None,
                "config_path": threshold_config_path(field, thresholds),
                "rationale": RATIONALE,
            }, separators=(",", ":"), sort_keys=True),
        ))
    return new

"""artifact_stage.py — decisions that apply to a finished table, not a record.

Phase 2c, ABTC1603-B. The field-level executor turns source data into a
harmonized table. Some decisions cannot be made there, because they are not
about a value at all:

  * an **exclusion** says a participant does not belong in a particular
    delivered artifact — but belongs in another, so it cannot be a filter on the
    data and cannot be a mapping;
  * a **post-rule** reads values the harmonization has already produced and
    adjusts some of them, again for named artifacts only.

Both are declared in the same approved specification as the mappings. What makes
them different is *when* they can run: only once a table exists, and separately
for each artifact, because the same underlying value can be delivered in one
artifact and superseded in another.

This module is that stage. It is deliberately small and deliberately not a
workflow engine: given one assembled table and the name of one artifact, it
applies what the specification says applies there, and returns the resulting
table together with **events** describing every change. Nothing is mutated
silently — a stage that quietly rewrote a table would leave a reader unable to
say what happened or why.

Recording those decisions back into provenance is a second, separate step
(`record_provenance`), because writing history is not the same act as making a
decision, and the caller may want the events without the rewriting.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

import pandas as pd

from lib import artifacts as artifact_names
from lib import rule_registry
from lib import value_states
from lib.contributions import STATUS_CONDITION, Contribution
from lib.contributions import to_json as contributions_json
from lib.provenance import Cell
from lib.spec_model import HarmonizationSpec

#: The pseudo-field an exclusion event is recorded under. It is not a target
#: field: the event is about the ABSENCE of a whole record, not about a value.
ROW_EVENT_FIELD = "__row_exclusion__"


class ArtifactStageError(RuntimeError):
    """An artifact-level decision cannot be carried out as declared."""


@dataclass(frozen=True)
class ExclusionEvent:
    """One participant dropped from one artifact, and why."""

    artifact: str
    exclusion_id: str
    trial: str
    participant: str
    reason: str
    rule_id: str
    rule_version: str
    n_rows: int

    def as_record(self) -> dict:
        return {"artifact": self.artifact, "exclusion_id": self.exclusion_id,
                "trial": self.trial, "participant": self.participant,
                "reason": self.reason, "rule_id": self.rule_id,
                "rule_version": self.rule_version, "n_rows": self.n_rows}


@dataclass(frozen=True)
class OverrideEvent:
    """One value changed by a post-rule, in one artifact."""

    artifact: str
    post_rule_id: str
    trial: str
    participant: str
    sample: str
    collection_event: str
    field: str
    previous_value: Any
    value: Any
    branch: str
    predicate: str
    input_field: str
    input_value: Any
    threshold: float
    rule_id: str
    rule_version: str
    reason: str

    @property
    def anchor(self) -> tuple[str, str, str, str]:
        return (self.trial, self.participant, self.sample, self.collection_event)

    def as_record(self) -> dict:
        prev = self.previous_value
        if prev is not None and isinstance(prev, float) and prev != prev:
            prev = None
        return {"artifact": self.artifact, "post_rule_id": self.post_rule_id,
                "anchor": list(self.anchor), "field": self.field,
                "previous_value": None if prev is None else str(prev),
                "value": self.value, "branch": self.branch,
                "predicate": self.predicate, "input_field": self.input_field,
                "rule_id": self.rule_id, "rule_version": self.rule_version,
                "reason": self.reason}


@dataclass
class ArtifactResult:
    """What one artifact looks like after its declared decisions were applied."""

    artifact: str
    frame: pd.DataFrame
    exclusions: tuple[ExclusionEvent, ...] = ()
    overrides: tuple[OverrideEvent, ...] = ()
    #: exclusions and post-rules declared but not applicable to this artifact.
    not_in_scope: dict[str, tuple[str, ...]] = field(default_factory=dict)

    def summary(self) -> str:
        return (f"{self.artifact}: {len(self.frame)} rows, "
                f"{len(self.exclusions)} exclusion(s), "
                f"{len(self.overrides)} override(s)")


# ---------------------------------------------------------------------------
# Applying the decisions
# ---------------------------------------------------------------------------

def _excluded_participants(exclusion) -> tuple[str, ...]:
    """The participants an exclusion names. The specification says who."""
    match = exclusion.match or {}
    predicate = match.get("predicate")
    value = match.get("value")
    if predicate == "in":
        return tuple(str(v) for v in (value or []))
    if predicate == "equals":
        return (str(value),)
    raise ArtifactStageError(
        f"exclusion {exclusion.id!r} matches with predicate {predicate!r}, "
        f"which this stage does not implement (expected 'in' or 'equals')")


def apply_canonical_rules(spec: HarmonizationSpec, frame: pd.DataFrame, *,
                          participant_key: str = "") -> ArtifactResult:
    """Finish the harmonization: apply the rules the canonical result depends on.

    **This entry point takes no artifact name, and that is the point.** A trial
    is harmonized correctly by applying the decisions its approved specification
    records; which delivered files a project happens to publish is a separate
    concern, and a new trial must not have to know any of their names.

    Exclusions first, then table-level rules — so an adjustment never reads a
    row the harmonization has decided does not belong.
    """
    return _apply_rules(spec, frame, spec.canonical_exclusions(),
                        spec.canonical_post_rules(), artifact="",
                        participant_key=participant_key)


def apply_artifact_rules(spec: HarmonizationSpec, frame: pd.DataFrame,
                         artifact: str, *,
                         participant_key: str = "") -> ArtifactResult:
    """Apply everything the specification declares for one named artifact.

    The delivery-layer entry point, kept because the current CIMAC pipeline
    publishes three tables that deliberately disagree, and ABTC1603-B proves
    this reproduces them exactly. The canonical path above does not use it.
    """
    return _apply_rules(spec, frame, spec.exclusions_for(artifact),
                        spec.post_rules_for(artifact), artifact=artifact,
                        participant_key=participant_key,
                        report_out_of_scope=True)


def _apply_rules(spec: HarmonizationSpec, frame: pd.DataFrame,
                 exclusions, post_rules, *, artifact: str,
                 participant_key: str = "",
                 report_out_of_scope: bool = False) -> ArtifactResult:
    """Apply a selected set of exclusions and table-level rules to one table.

    `frame` is not modified; a copy is returned.
    """
    key = participant_key or spec.record.participant_key
    if not key:
        raise ArtifactStageError(
            f"{spec.ref}: no participant key; the record declares none")
    out = frame.copy()

    exclusion_events: list[ExclusionEvent] = []
    for exclusion in exclusions:
        if exclusion.scope != "participant":
            raise ArtifactStageError(
                f"{spec.ref}: exclusion {exclusion.id!r} has scope "
                f"{exclusion.scope!r}, which this stage does not implement")
        field_name = (exclusion.match or {}).get("field") or key
        for participant in _excluded_participants(exclusion):
            mask = out[field_name].astype(str) == participant
            n = int(mask.sum())
            exclusion_events.append(ExclusionEvent(
                artifact=artifact, exclusion_id=exclusion.id, trial=spec.trial,
                participant=participant, reason=exclusion.reason,
                rule_id=exclusion.rule.id, rule_version=exclusion.rule.version,
                n_rows=n))
            if n:
                out = out[~mask]
        out = out.reset_index(drop=True)

    override_events: list[OverrideEvent] = []
    for rule in post_rules:
        registered = rule_registry.resolve(rule.rule.id, rule.rule.version,
                                           kind=rule_registry.KIND_POST)
        rule_registry.check_contract(registered, rule.inputs, rule.params)
        events = registered.func(out, **rule.params)
        for e in events:
            idx = e["row"]
            override_events.append(OverrideEvent(
                artifact=artifact, post_rule_id=rule.id, trial=spec.trial,
                participant=str(out.at[idx, key]),
                sample=str(out.at[idx, spec.record.sample_key]),
                collection_event=str(out.at[idx, spec.record.collection_event_key]),
                field=e["field"], previous_value=e["previous_value"],
                value=e["value"], branch=e["branch"], predicate=e["predicate"],
                input_field=e["input_field"], input_value=e["input_value"],
                threshold=e["threshold"], rule_id=rule.rule.id,
                rule_version=rule.rule.version,
                reason=rule.rule.rationale or rule.description))
        for e in events:
            out.at[e["row"], e["field"]] = e["value"]

    out_of_scope: dict[str, tuple[str, ...]] = {}
    if report_out_of_scope:
        out_of_scope = {
            "exclusions": tuple(e.id for e in spec.exclusions
                                if artifact not in e.applies_to),
            "post_rules": tuple(r.id for r in spec.post_rules
                                if artifact not in r.artifacts),
        }
    else:
        # Nothing is "out of scope" for the canonical result — a declared rule
        # either finishes the harmonization or is marked as not canonical, and
        # the second kind is named so it is visibly withheld rather than missed.
        out_of_scope = {
            "exclusions": tuple(e.id for e in spec.exclusions if not e.canonical),
            "post_rules": tuple(r.id for r in spec.post_rules if not r.canonical),
        }
    return ArtifactResult(
        artifact=artifact, frame=out,
        exclusions=tuple(exclusion_events), overrides=tuple(override_events),
        not_in_scope=out_of_scope)


# ---------------------------------------------------------------------------
# Recording the decisions in provenance
# ---------------------------------------------------------------------------

def _clean(value: Any) -> Any:
    if value is None or (isinstance(value, float) and value != value):
        return ""
    return value


def record_provenance(spec: HarmonizationSpec, results: dict[str, ArtifactResult],
                      cells: list[Cell], *, sources: dict | None = None,
                      run_id: str = ""
                      ) -> list[Cell]:
    """Write the artifact decisions back into the cell stream.

    Two things happen, and both are what makes the audit hold:

      1. each decision gains a cell of its own, authoritative for exactly the
         artifacts it applies to;
      2. the cells it supersedes are **demoted** to the artifacts where their
         value is still what gets delivered. A cell that keeps claiming an
         artifact whose delivered value it no longer explains is not merely
         incomplete — it is wrong.

    Returns the new cells; mutates the superseded ones in place.
    """
    return (_record_overrides(spec, results, cells, run_id=run_id)
            + _record_exclusions(spec, results, cells, sources or {},
                                 run_id=run_id))


def record_canonical_provenance(spec: HarmonizationSpec, result: ArtifactResult,
                                cells: list[Cell], *,
                                sources: dict | None = None,
                                run_id: str = "") -> list[Cell]:
    """Record the canonical decisions, without any artifact vocabulary.

    There is one canonical table, so "which output does this row speak for" is
    not a question. What still matters scientifically is the same two facts the
    delivery layer records:

      * an approved rule replaced a value — and which value it replaced;
      * a participant does not belong in the harmonized result — and why.

    The cells those decisions supersede are marked as explaining no delivered
    value (`artifacts.NONE`), which is simply true: the canonical table carries
    something else. That is a deliberate simplification of the delivery layer's
    per-artifact authority, and it loses nothing the canonical result depends on.
    """
    overrides = _record_overrides(
        spec, {"": result}, cells, canonical=True, run_id=run_id)
    exclusions = _record_exclusions(spec, {"": result}, cells, sources or {},
                                    canonical=True, run_id=run_id)
    return overrides + exclusions


def _record_overrides(spec, results, cells, *, canonical: bool = False,
                      run_id: str = "") -> list[Cell]:
    by_rule: dict[str, Any] = {r.id: r for r in spec.post_rules}
    merged: dict[tuple, dict] = {}
    for result in results.values():
        for e in result.overrides:
            key = (*e.anchor, e.field)
            rec = merged.setdefault(key, {"event": e, "artifacts": set()})
            rec["artifacts"].add(e.artifact)

    by_key: dict[tuple, list[Cell]] = {}
    for c in cells:
        by_key.setdefault((c.trial, c.cimac_part_id, c.Cimac_id,
                           c.Collection_Event, c.harmonized_field), []).append(c)

    new: list[Cell] = []
    for key, rec in merged.items():
        e: OverrideEvent = rec["event"]
        rule = by_rule[e.post_rule_id]
        if canonical:
            # One canonical table: the override explains the delivered value
            # everywhere, and the cell it replaced explains nothing delivered.
            applies, not_applies = "", artifact_names.NONE
        else:
            applies = artifact_names.join(rec["artifacts"])
            not_applies = artifact_names.join(
                set(artifact_names.ALL_ARTIFACTS) - rec["artifacts"])

        superseded = by_key.get(key, [])
        for c in superseded:
            c.authoritative_for = not_applies
        prev_rule = (rule.supersedes_rule
                     or (superseded[0].rule_id if superseded else ""))
        prev = e.previous_value
        prev_txt = "" if prev is None or (isinstance(prev, float) and prev != prev) \
            else str(prev)

        # The input gates which branch the rule takes; it does not itself become
        # the delivered value, so it is a CONDITION rather than a candidate. It
        # is a harmonized value rather than a source cell, so it identifies
        # itself by provenance reference.
        input_ref = "|".join([*e.anchor, e.input_field])
        contribs = [Contribution(
            role=f"{e.input_field}_input", status=STATUS_CONDITION,
            source_column=e.input_field, raw_value=_clean(e.input_value), rank=0,
            source_record_key=f"{spec.record.participant_key}={e.participant}",
            rule_id=e.rule_id,
            source_logical_label=f"provenance:{e.input_field}",
            provenance_ref=input_ref)]

        derivation = json.dumps({
            "kind": "post_pivot_override",
            "rule_id": e.rule_id,
            "stage": rule.stage,
            "landmark_days": e.threshold,
            "predicate": e.predicate,
            "branch": e.branch,
            "input": {"field": e.input_field,
                      "value": (None if _clean(e.input_value) == ""
                                else float(e.input_value)),
                      "provenance_ref": input_ref},
            "final_value": e.value,
            **({} if canonical else {
                "applies_to": list(artifact_names.parse(applies)),
                "not_applies_to": (list(artifact_names.parse(not_applies))
                                   if not_applies else [])}),
            "supersedes": {"value": prev_txt, "rule_id": prev_rule},
            "approval_ref": rule.rule.approval_ref,
            "rationale": e.reason,
        }, separators=(",", ":"), sort_keys=True)

        new.append(Cell(
            trial=e.trial, cimac_part_id=e.participant, Cimac_id=e.sample,
            Collection_Event=e.collection_event, harmonized_field=e.field,
            value=e.value, confidence=rule.confidence,
            source_file=rule.source_label, source_column=e.input_field,
            source_row_idx=-1, extraction_method="post_pivot_override",
            notes=(f"{e.branch}: {e.input_field}={e.input_value!r} vs "
                   f"{e.threshold:g}-day landmark -> {e.field}={e.value!r}; "
                   f"supersedes {prev_txt!r}"
                   + ("" if canonical else f" for {applies}")),
            source_logical_label=rule.source_label,
            source_record_key=f"{spec.record.participant_key}={e.participant}",
            raw_value=_clean(e.input_value),
            rule_id=e.rule_id, rule_version=e.rule_version,
            contributions=contributions_json(contribs),
            authoritative_for=applies, derivation=derivation,
            value_state=rule.value_state,
            run_id=run_id, spec_ref=spec.ref,
        ))
    return new


def _record_exclusions(spec, results, cells, sources, *,
                       canonical: bool = False, run_id: str = "") -> list[Cell]:
    by_participant: dict[str, list[ExclusionEvent]] = {}
    for result in results.values():
        for e in result.exclusions:
            by_participant.setdefault(e.participant, []).append(e)
    if not by_participant:
        return []

    # Which artifacts actually carry each participant, read from the frames that
    # were produced rather than assumed from what was intended. The canonical
    # path has one table, so the question does not arise.
    present_in: dict[str, list[str]] = {p: [] for p in by_participant}
    if not canonical:
        for name in artifact_names.ALL_ARTIFACTS:
            result = results.get(name)
            if result is None:
                continue
            held = set(result.frame[spec.record.participant_key].astype(str))
            for participant in by_participant:
                if participant in held:
                    present_in[participant].append(name)

    cells_by_participant: dict[str, list[Cell]] = {}
    for c in cells:
        cells_by_participant.setdefault(c.cimac_part_id, []).append(c)

    new: list[Cell] = []
    for participant in sorted(by_participant):
        events = by_participant[participant]
        present = present_in.get(participant, [])
        absent = [a for a in artifact_names.ALL_ARTIFACTS
                  if a in results and a not in present]
        own = cells_by_participant.get(participant, [])
        for c in own:
            # Excluded from the canonical result, so the participant's values
            # explain nothing that is delivered.
            c.authoritative_for = (artifact_names.NONE if canonical
                                   else artifact_names.join(present))

        first = events[0]
        exclusion = next(e for e in spec.exclusions if e.id == first.exclusion_id)
        # The file the exclusion is *about* — the one the specification says the
        # participant should have been in. Taken from the declaration rather
        # than from whichever of the participant's cells happens to come first,
        # so the recorded evidence does not depend on emission order.
        src = sources.get(exclusion.source)
        resolved = src.resolved_path if src else ""
        sha = src.sha256 if src else ""
        if not resolved:
            fallback = next((c for c in own if c.source_resolved_path), None)
            resolved = fallback.source_resolved_path if fallback else ""
            sha = fallback.source_sha256 if fallback else ""
        new.append(Cell(
            trial=first.trial, cimac_part_id=participant,
            Cimac_id=(own[0].Cimac_id if own else ""),
            Collection_Event=(own[0].Collection_Event if own else ""),
            harmonized_field=ROW_EVENT_FIELD,
            value=None, confidence=1.0,
            source_file=exclusion.source_label or "",
            source_column=(exclusion.match or {}).get("field", ""),
            source_row_idx=-1, extraction_method="row_excluded",
            notes=(f"{participant} excluded: {first.reason}" if canonical else
                   f"{participant} excluded from "
                   f"{', '.join(absent) or 'no artifact'}: {first.reason}"),
            source_logical_label=exclusion.source_label or "",
            source_resolved_path=resolved,
            source_sha256=sha,
            source_record_key=f"{spec.record.participant_key}={participant}",
            rule_id=first.rule_id, rule_version=first.rule_version,
            authoritative_for=("" if canonical
                               else artifact_names.join(absent)),
            value_state=value_states.ROW_EXCLUDED,
            run_id=run_id, spec_ref=spec.ref,
            derivation=json.dumps({
                "kind": "row_exclusion",
                "rule_id": first.rule_id,
                "config_path": exclusion.evidence_path,
                "reason": first.reason,
                **({} if canonical else {"absent_from": absent,
                                         "present_in": present}),
                # A declared exclusion whose participant the harmonization never
                # produced is VACUOUS — say so rather than claiming a row was
                # removed.
                "existed_in_intermediate": bool(own),
                "vacuous": not own,
                "n_provenance_rows_retained": len(own),
                "approval_ref": exclusion.rule.approval_ref,
            }, separators=(",", ":"), sort_keys=True),
        ))
    return new

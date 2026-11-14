"""canonical.py — one trial in, the canonical harmonized table out.

This is the endpoint the framework exists to produce:

    source data -> field harmonization -> table-level harmonization
                -> CANONICAL HARMONIZED TABLE

`harmonize_trial` runs the whole path for a single trial and returns the table
its approved specification says is correct. `assemble` puts several of those
together. Neither takes an artifact name, and that is deliberate: a new trial
must be harmonizable knowing only its own specification and the target schema.
Which files a project happens to publish, and which historical tables it keeps
for comparison, are a delivery concern that lives outside this module.

Everything here is mechanism. The clinical decisions — which fields come from
where, which participants do not belong, which approved rule finishes a value —
are all in the specification, and a test asserts this module names no trial and
no clinical term.

Canonical values follow the **target schema's declared types**, not whatever
Python type a legacy extractor happened to produce. The same indicator arrives
as `"1"` from one trial, `1` from another and `1.0` from a third; that is
representation drift between eleven independently written extractors, not a
clinical distinction, and letting it through would make the canonical dataset
inherit an inconsistency none of its specifications intends:

    source / legacy representation -> harmonization -> TARGET SCHEMA TYPE

Only the canonical value is normalized. Provenance keeps the raw source value
exactly as it was read, because that is a record of what the file said.

Assembly is deliberately boring: check the contract matches, concatenate,
refuse duplicate record keys, keep the schema's column order. If it ever needs
to be interesting, something has been put in the wrong layer.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

import pandas as pd

from lib import artifact_stage, suppression
from lib.anchor_evidence import RuntimeExternalAnchor
from lib.provenance import Cell
from lib.schema_model import T_INTEGER, T_NUMBER, contract_for
from lib.source_ref import SourceRef
from lib.spec_executor import execute
from lib.spec_model import HarmonizationSpec


class CanonicalError(RuntimeError):
    """A canonical harmonization cannot be produced as specified."""


@dataclass
class CanonicalTrial:
    """One trial, harmonized to the point where it is scientifically finished."""

    spec_ref: str
    trial: str
    frame: pd.DataFrame
    #: The full cell stream, including the decisions the table-level stage made
    #: and the demotions they caused.
    cells: list[Cell] = field(default_factory=list)
    exclusions: tuple = ()
    overrides: tuple = ()
    #: Declared rules deliberately withheld from the canonical result.
    withheld: dict[str, tuple[str, ...]] = field(default_factory=dict)
    #: Rows the field harmonization produced, before table-level rules.
    intermediate_rows: int = 0
    #: Reader-produced source facts from this exact execution.  Additive so
    #: existing callers constructing CanonicalTrial directly remain valid.
    sources: dict[str, SourceRef] = field(default_factory=dict)

    def summary(self) -> str:
        return (f"{self.spec_ref}: {self.intermediate_rows} intermediate rows -> "
                f"{len(self.frame)} canonical rows "
                f"({len(self.exclusions)} exclusion(s), "
                f"{len(self.overrides)} table-level change(s))")


def _is_absent(value: Any) -> bool:
    try:
        if value is None or pd.isna(value):
            return True
    except (TypeError, ValueError):
        pass
    return isinstance(value, str) and value.strip() == ""


def coerce_to_type(value: Any, declared: str, field: str) -> Any:
    """One value, rendered as the target schema declares it.

    Mirrors the rules the schema validator already enforces, so a canonical
    table cannot be built that the validator would then reject: a numeric field
    must read as a number, and an integer field's values must be integral.
    Anything that cannot be read that way is an error rather than a silent
    coercion — a value nobody can interpret is a finding, not a null.

    A field the schema does not declare is left alone; the contract check in
    `assemble` is what refuses columns that do not belong.
    """
    # Every field, whatever its type: an absence is one thing, spelled one way.
    # A NaN left in a string column is a float pretending to be a missing value.
    if _is_absent(value):
        return None
    if declared not in (T_INTEGER, T_NUMBER):
        return value
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise CanonicalError(
            f"{field}: {value!r} cannot be read as {declared}, which the target "
            f"schema declares") from exc
    if declared == T_NUMBER:
        return number
    if number % 1 != 0:
        raise CanonicalError(
            f"{field}: {value!r} is not integral, but the target schema "
            f"declares {field} as an integer")
    return int(number)


def normalize_to_schema(frame: pd.DataFrame, schema) -> pd.DataFrame:
    """Render every canonical value as its declared target type.

    The schema is the authority. No field is named here: the types come from
    the contract, so a target schema with different fields normalizes without
    this function changing.
    """
    declared = {f.name: f.type for f in schema.fields}
    out = frame.copy()
    for column in out.columns:
        kind = declared.get(column)
        values = [coerce_to_type(v, kind, column) for v in out[column]]
        # Assigned as an object Series on purpose. Handing pandas a plain list
        # lets it infer a dtype again — a column of integers and absences comes
        # back as float64, which is the very rewriting this function exists to
        # undo, and it would silently do it on the way out.
        out[column] = pd.Series(values, dtype=object, index=out.index)
    return out


def _pivot():
    """The shared pivot, imported lazily so there is exactly one of it.

    Choosing among competing values and withholding those below their threshold
    is the same operation whichever path assembles the table; duplicating it
    would be a second definition of what the harmonization decides.
    """
    from lib.pivot import pivot_cells_to_wide
    return pivot_cells_to_wide


def harmonize_trial(spec: HarmonizationSpec, project_root: Path,
                    anchor: pd.DataFrame | RuntimeExternalAnchor | None = None,
                    schema=None, *,
                    thresholds: dict | None = None, run_id: str = "") -> CanonicalTrial:
    """Harmonize one trial all the way to its canonical table.

    Six steps, none of which needs to know which trial this is:

      1. execute the specification's field mappings;
      2. pivot to a table, choosing among competing values and withholding
         those whose confidence is below their field's threshold;
      3. record the withholdings, so a value that was found and declined is
         distinguishable from one that was never there;
      4. apply the table-level rules the canonical result depends on;
      5. record what those decisions were and what they superseded;
      6. render every value as the target schema declares it.

    Takes the schema rather than a contract because it needs both what the
    columns are and what they mean.

    `anchor` is omitted when the specification constructs its own records — a
    trial whose records exist only as a grouping of source rows has no anchor a
    caller could supply, and asking for one would invite a caller to override
    what the specification decided.
    """
    if schema is None:
        raise CanonicalError("harmonize_trial requires a target schema")
    contract = contract_for(schema)
    cells, diagnostics = execute(spec, project_root, anchor, contract=contract, run_id=run_id)
    if diagnostics.deferred:
        raise CanonicalError(
            f"{spec.ref}: mapping(s) {sorted(diagnostics.deferred)} were not "
            f"executed, so this table is not the canonical result: "
            f"{diagnostics.deferred}")

    wide, _flags, withheld = _pivot()(cells, thresholds or {}, contract)
    intermediate_rows = len(wide)

    # The pivot decides what is delivered; it also reports what it declined and
    # why, and that report is the only place the reason survives. Recording it
    # here keeps suppression a runtime fact rather than something a
    # specification would have to predict about its own data.
    suppressions = suppression.record_canonical_suppression(
        withheld, cells, thresholds=thresholds, run_id=run_id,
        spec_ref=spec.ref)

    result = artifact_stage.apply_canonical_rules(spec, wide.reset_index(drop=True))
    extra = artifact_stage.record_canonical_provenance(
        spec, result, cells, sources=diagnostics.sources, run_id=run_id)

    # Normalizing here, after the table-level rules, means the schema has the
    # last word on representation and pandas' dtype inference during the pivot
    # has none. Provenance is untouched: the cells keep the raw source values.
    canonical_frame = normalize_to_schema(result.frame, schema)

    return CanonicalTrial(
        spec_ref=spec.ref, trial=spec.trial, frame=canonical_frame,
        cells=cells + suppressions + extra, exclusions=result.exclusions,
        overrides=result.overrides, withheld=result.not_in_scope,
        intermediate_rows=intermediate_rows,
        sources=diagnostics.sources.copy())


def assemble(trials: Iterable[CanonicalTrial], schema) -> pd.DataFrame:
    """Combine canonical trial tables into one canonical harmonized table.

    Row order is the order the trials were given, and within a trial the order
    its rows already had — which is the order its row anchor supplied. Stated
    rather than incidental, and deliberately not borrowed from any historical
    table: a comparison artifact must not decide what the canonical result looks
    like.
    """
    trials = list(trials)
    if not trials:
        raise CanonicalError("no trials to assemble")

    contract = contract_for(schema)
    columns = list(contract.columns)
    for t in trials:
        got = list(t.frame.columns)
        if got != columns:
            missing = [c for c in columns if c not in got]
            extra = [c for c in got if c not in columns]
            raise CanonicalError(
                f"{t.spec_ref}: table does not match the target contract "
                f"{contract.schema_ref}; missing {missing}, unexpected {extra}")

    frames = [t.frame.astype(object) for t in trials]
    combined = pd.concat(frames, ignore_index=True)[columns].astype(object)

    key = _record_key_columns(trials[0], contract)
    duplicated = combined.duplicated(subset=key, keep=False)
    if duplicated.any():
        offenders = (combined.loc[duplicated, key]
                     .astype(str).agg(" | ".join, axis=1).unique()[:5])
        raise CanonicalError(
            f"assembled table has {int(duplicated.sum())} rows sharing a record "
            f"key {key}; the same record cannot be harmonized twice. "
            f"First: {list(offenders)}")

    expected = sum(len(t.frame) for t in trials)
    if len(combined) != expected:
        raise CanonicalError(
            f"assembled {len(combined)} rows from trial tables totalling "
            f"{expected}")
    return combined


def _record_key_columns(trial: CanonicalTrial, contract) -> list[str]:
    """The columns that identify one harmonized record.

    Taken first from the target schema's explicit row key, so the key is a
    property of the contract rather than a list written here.  The older role
    and CIMAC-name fallbacks remain for contracts created by legacy callers.
    """
    if contract.row_key:
        return list(contract.row_key)
    declared = [f.name for f in getattr(contract, "identifier_fields", [])]
    if declared:
        return declared
    return [c for c in ("trial", "cimac_part_id", "Cimac.id", "Collection_Event")
            if c in trial.frame.columns]


def validate(combined: pd.DataFrame, trials: Iterable[CanonicalTrial],
             schema) -> dict[str, Any]:
    """Structural checks on an assembled canonical table.

    Not a substitute for target-schema validation, which the caller runs
    separately; these are the invariants assembly itself is responsible for.
    """
    trials = list(trials)
    contract = contract_for(schema)
    per_trial = {t.trial: len(t.frame) for t in trials}
    counts = combined["trial"].value_counts().to_dict() if "trial" in combined else {}
    problems: list[str] = []

    if list(combined.columns) != list(contract.columns):
        problems.append("column order does not follow the target contract")
    if counts != per_trial:
        problems.append(f"row counts do not balance: assembled {counts}, "
                        f"trial tables {per_trial}")
    if sum(per_trial.values()) != len(combined):
        problems.append("assembled row count does not equal the sum of trials")
    unexpected = sorted(set(counts) - set(per_trial))
    if unexpected:
        problems.append(f"rows from trial(s) {unexpected} that were not assembled")

    return {"rows": len(combined), "per_trial": per_trial,
            "columns": len(combined.columns), "problems": problems}

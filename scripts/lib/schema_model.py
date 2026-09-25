"""schema_model.py — A generic, versioned target-schema for harmonized tables.

This module is **project-agnostic**. It knows about types, vocabularies, bounds,
nullability and cross-field predicates. It does not know that `clinical_benefit`
exists or that `race` may be `white`; those belong in a schema instance such as
`schemas/cimac_clinical_v1.yaml`. A second harmonization project supplies its own
instance and reuses this loader unchanged.

What a schema document is for
-----------------------------
Before Phase 2a the CIMAC target was a Python list of 19 column names
(`TEMPLATE_COLUMNS`). A list cannot say what a column means, what it may
contain, whether a blank is a missing measurement or an inapplicable one, or
which artifacts a rule applies to. Everything a validator would need was either
implicit in the extractors or written only in prose. This module gives that
knowledge one machine-readable home, versioned as a whole so a validation result
can name the exact contract it was produced against.

Design limits (deliberate)
--------------------------
This is not an ontology, a standards server, or a general rule language. It is
the smallest description that supports structural, type, vocabulary, bound and
simple cross-field validation. Two cross-field predicate kinds are implemented
(`requires_non_null`, `less_or_equal`); adding a third should require evidence
that a real schema needs it, not that one could exist.

Strictness
----------
Loading is strict: unknown keys, duplicate field names, malformed severities and
unparsable versions are load errors, not warnings. A schema that silently ignores
a misspelled key is worse than no schema, because it reports success for a
constraint nobody is enforcing.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field as dc_field
from pathlib import Path
from typing import Any

import yaml

# ── severities ──────────────────────────────────────────────────────────────
ERROR = "error"
WARNING = "warning"
INFO = "info"
SEVERITIES = (ERROR, WARNING, INFO)

#: Sort rank, most severe first. Used for deterministic report ordering.
SEVERITY_RANK = {ERROR: 0, WARNING: 1, INFO: 2}

# ── field types ─────────────────────────────────────────────────────────────
T_STRING = "string"
T_NUMBER = "number"
T_INTEGER = "integer"
FIELD_TYPES = (T_STRING, T_NUMBER, T_INTEGER)

#: Semantic roles. Descriptive: the validator treats `identifier` and
#: `partition` as participating in the row key only if the schema's `row_key`
#: names them, so a role is documentation plus a hook, never a hidden rule.
ROLES = ("identifier", "partition", "category", "measure", "derived", "annotation")

CROSS_FIELD_KINDS = ("requires_non_null", "less_or_equal")

_VERSION_RE = re.compile(r"^\d+\.\d+\.\d+$")

# Keys accepted at each level. Anything else is a load error.
_DOC_KEYS = {
    "schema_id", "schema_version", "title", "description", "evidence",
    "artifacts", "row_key", "column_order", "column_order_severity",
    "unexpected_column_severity", "null_encoding", "fields", "constraints",
}
_FIELD_KEYS = {
    "name", "description", "role", "type", "unit", "required_column",
    "nullable", "vocabulary", "bounds", "pattern", "pattern_severity",
    "value_states", "artifacts", "open_questions", "evidence",
    "deprecated", "deprecation_note",
}
_VOCAB_KEYS = {"values", "canonical", "unknown_severity", "non_canonical_severity"}
_BOUNDS_KEYS = {"min", "max", "severity"}
_CONSTRAINT_KEYS = {"id", "kind", "field", "other", "severity", "description", "evidence"}


class SchemaError(ValueError):
    """The schema document itself is invalid. Raised only at load time."""


def _require(cond: bool, msg: str) -> None:
    if not cond:
        raise SchemaError(msg)


def _check_keys(got: dict, allowed: set[str], where: str) -> None:
    unknown = sorted(set(got) - allowed)
    _require(not unknown,
             f"{where}: unknown propert{'y' if len(unknown) == 1 else 'ies'} "
             f"{unknown} (allowed: {sorted(allowed)})")


def _severity(value: Any, where: str, default: str) -> str:
    if value is None:
        return default
    _require(value in SEVERITIES,
             f"{where}: severity must be one of {list(SEVERITIES)}, got {value!r}")
    return value


@dataclass(frozen=True)
class Vocabulary:
    """Permitted values, and the subset considered canonical.

    Two tiers because harmonized data legitimately carries values that are
    permitted but not normalized — a source vocabulary preserved verbatim by an
    approved decision. Reporting those separately from genuinely unknown values
    is the difference between a normalization backlog and a data defect.
    """

    values: tuple[str, ...]
    canonical: tuple[str, ...] = ()
    unknown_severity: str = WARNING
    non_canonical_severity: str = INFO

    @property
    def non_canonical(self) -> tuple[str, ...]:
        if not self.canonical:
            return ()
        return tuple(v for v in self.values if v not in self.canonical)


@dataclass(frozen=True)
class Bounds:
    min: float | None = None
    max: float | None = None
    severity: str = ERROR


@dataclass(frozen=True)
class Field:
    name: str
    type: str = T_STRING
    role: str = "annotation"
    description: str = ""
    unit: str = ""
    required_column: bool = True
    nullable: bool = True
    vocabulary: Vocabulary | None = None
    bounds: Bounds | None = None
    pattern: str = ""
    pattern_severity: str = WARNING
    #: Non-default provenance value_states this field may legitimately carry.
    #: Covers both absence-explaining states (`source_missing`, `not_applicable`,
    #: `suppressed`, `row_excluded`) and presence-qualifying ones (`provisional`,
    #: `transformed`). An ordinary null is simply `nullable: true` with no state.
    value_states: tuple[str, ...] = ()
    #: Artifacts this field applies to; empty means every artifact.
    artifacts: tuple[str, ...] = ()
    #: References to open review questions touching this field. Documentation
    #: only — an open question never changes a severity or fails a run.
    open_questions: tuple[str, ...] = ()
    evidence: str = ""
    deprecated: bool = False
    deprecation_note: str = ""

    def applies_to(self, artifact: str) -> bool:
        return not self.artifacts or artifact in self.artifacts


@dataclass(frozen=True)
class TargetContract:
    """The output shape a writer must produce, derived from a schema.

    Exists so a writer never carries its own column list. Before this, the CIMAC
    orchestrator held `TEMPLATE_COLUMNS` — a second definition of the target that
    could drift from the schema silently, because nothing compared them.
    """

    schema_ref: str
    artifact: str
    columns: tuple[str, ...]
    required: tuple[str, ...]
    row_key: tuple[str, ...] = ()


def contract_for(schema: "TargetSchema", artifact: str = "") -> TargetContract:
    """Columns and required columns for one artifact, in delivery order.

    `column_order` wins when the schema declares one, because that is the
    delivery contract; otherwise declaration order is used, which is the only
    other ordering a schema states.
    """
    applicable = schema.fields_for(artifact)
    names = {f.name for f in applicable}
    if schema.column_order:
        columns = tuple(c for c in schema.column_order if c in names)
    else:
        columns = tuple(f.name for f in applicable)
    required = tuple(c for c in columns
                     if (fl := schema.field(c)) is not None and fl.required_column)
    return TargetContract(schema_ref=schema.ref, artifact=artifact,
                          columns=columns, required=required,
                          row_key=schema.row_key)


@dataclass(frozen=True)
class CrossFieldConstraint:
    id: str
    kind: str
    field: str
    other: str
    severity: str = WARNING
    description: str = ""
    evidence: str = ""


@dataclass(frozen=True)
class TargetSchema:
    schema_id: str
    schema_version: str
    title: str = ""
    description: str = ""
    evidence: str = ""
    #: Artifact names this schema describes. Empty means "any artifact".
    artifacts: tuple[str, ...] = ()
    #: Columns that together identify a row uniquely.
    row_key: tuple[str, ...] = ()
    #: Exact delivered column order, when the schema is a delivery contract.
    column_order: tuple[str, ...] = ()
    column_order_severity: str = ERROR
    unexpected_column_severity: str = ERROR
    #: The only string that means "no value". Anything else null-looking is a
    #: finding rather than a silently accepted null.
    null_encoding: str = ""
    fields: tuple[Field, ...] = dc_field(default_factory=tuple)
    constraints: tuple[CrossFieldConstraint, ...] = dc_field(default_factory=tuple)
    #: Content hash of the file this was loaded from; "" when built in memory.
    sha256: str = ""
    source_path: str = ""

    @property
    def ref(self) -> str:
        """`schema_id@version` — how a validation result names its contract."""
        return f"{self.schema_id}@{self.schema_version}"

    def field(self, name: str) -> Field | None:
        for f in self.fields:
            if f.name == name:
                return f
        return None

    def fields_for(self, artifact: str) -> tuple[Field, ...]:
        return tuple(f for f in self.fields if f.applies_to(artifact))

    def version_tuple(self) -> tuple[int, int, int]:
        major, minor, patch = self.schema_version.split(".")
        return int(major), int(minor), int(patch)


# ── parsing ─────────────────────────────────────────────────────────────────

def _parse_vocabulary(raw: Any, where: str) -> Vocabulary:
    _require(isinstance(raw, dict), f"{where}: vocabulary must be a mapping")
    _check_keys(raw, _VOCAB_KEYS, where)
    values = raw.get("values")
    _require(isinstance(values, list) and values,
             f"{where}: vocabulary.values must be a non-empty list")
    vals = [str(v) for v in values]
    _require(len(set(vals)) == len(vals),
             f"{where}: vocabulary.values contains duplicates")
    canonical = [str(v) for v in (raw.get("canonical") or [])]
    missing = sorted(set(canonical) - set(vals))
    _require(not missing,
             f"{where}: vocabulary.canonical entries {missing} are not in values")
    return Vocabulary(
        values=tuple(vals),
        canonical=tuple(canonical),
        unknown_severity=_severity(raw.get("unknown_severity"),
                                   f"{where}.unknown_severity", WARNING),
        non_canonical_severity=_severity(raw.get("non_canonical_severity"),
                                         f"{where}.non_canonical_severity", INFO),
    )


def _parse_bounds(raw: Any, where: str) -> Bounds:
    _require(isinstance(raw, dict), f"{where}: bounds must be a mapping")
    _check_keys(raw, _BOUNDS_KEYS, where)
    lo, hi = raw.get("min"), raw.get("max")
    _require(lo is not None or hi is not None,
             f"{where}: bounds needs at least one of min/max")
    for label, v in (("min", lo), ("max", hi)):
        _require(v is None or isinstance(v, (int, float)),
                 f"{where}: bounds.{label} must be numeric, got {v!r}")
    if lo is not None and hi is not None:
        _require(float(lo) <= float(hi), f"{where}: bounds.min exceeds bounds.max")
    return Bounds(min=None if lo is None else float(lo),
                  max=None if hi is None else float(hi),
                  severity=_severity(raw.get("severity"), f"{where}.severity", ERROR))


def _parse_field(raw: Any, index: int) -> Field:
    where = f"fields[{index}]"
    _require(isinstance(raw, dict), f"{where}: must be a mapping")
    _check_keys(raw, _FIELD_KEYS, where)
    name = raw.get("name")
    _require(isinstance(name, str) and name.strip(),
             f"{where}: name is required and must be a non-empty string")
    where = f"field {name!r}"

    ftype = raw.get("type", T_STRING)
    _require(ftype in FIELD_TYPES,
             f"{where}: type must be one of {list(FIELD_TYPES)}, got {ftype!r}")
    role = raw.get("role", "annotation")
    _require(role in ROLES, f"{where}: role must be one of {list(ROLES)}, got {role!r}")

    vocab = raw.get("vocabulary")
    if vocab is not None:
        _require(ftype == T_STRING,
                 f"{where}: vocabulary is only meaningful on a {T_STRING} field")
        vocab = _parse_vocabulary(vocab, where)
    bounds = raw.get("bounds")
    if bounds is not None:
        _require(ftype in (T_NUMBER, T_INTEGER),
                 f"{where}: bounds are only meaningful on a numeric field")
        bounds = _parse_bounds(bounds, where)

    pattern = raw.get("pattern", "") or ""
    if pattern:
        _require(ftype == T_STRING,
                 f"{where}: pattern is only meaningful on a {T_STRING} field")
        try:
            re.compile(pattern)
        except re.error as exc:
            raise SchemaError(f"{where}: pattern is not a valid regex ({exc})") from exc

    for flag in ("required_column", "nullable", "deprecated"):
        v = raw.get(flag)
        _require(v is None or isinstance(v, bool), f"{where}: {flag} must be a boolean")

    for seq in ("value_states", "artifacts", "open_questions"):
        v = raw.get(seq)
        _require(v is None or isinstance(v, list), f"{where}: {seq} must be a list")

    return Field(
        name=name,
        type=ftype,
        role=role,
        description=raw.get("description", "") or "",
        unit=raw.get("unit", "") or "",
        required_column=bool(raw.get("required_column", True)),
        nullable=bool(raw.get("nullable", True)),
        vocabulary=vocab,
        bounds=bounds,
        pattern=pattern,
        pattern_severity=_severity(raw.get("pattern_severity"),
                                   f"{where}.pattern_severity", WARNING),
        value_states=tuple(str(s) for s in (raw.get("value_states") or [])),
        artifacts=tuple(str(a) for a in (raw.get("artifacts") or [])),
        open_questions=tuple(str(q) for q in (raw.get("open_questions") or [])),
        evidence=raw.get("evidence", "") or "",
        deprecated=bool(raw.get("deprecated", False)),
        deprecation_note=raw.get("deprecation_note", "") or "",
    )


def _parse_constraint(raw: Any, index: int, field_names: set[str]) -> CrossFieldConstraint:
    where = f"constraints[{index}]"
    _require(isinstance(raw, dict), f"{where}: must be a mapping")
    _check_keys(raw, _CONSTRAINT_KEYS, where)
    cid = raw.get("id")
    _require(isinstance(cid, str) and cid.strip(), f"{where}: id is required")
    where = f"constraint {cid!r}"
    kind = raw.get("kind")
    _require(kind in CROSS_FIELD_KINDS,
             f"{where}: kind must be one of {list(CROSS_FIELD_KINDS)}, got {kind!r}")
    for key in ("field", "other"):
        val = raw.get(key)
        _require(isinstance(val, str) and val, f"{where}: {key} is required")
        _require(val in field_names, f"{where}: {key} {val!r} is not a declared field")
    return CrossFieldConstraint(
        id=cid,
        kind=kind,
        field=raw["field"],
        other=raw["other"],
        severity=_severity(raw.get("severity"), f"{where}.severity", WARNING),
        description=raw.get("description", "") or "",
        evidence=raw.get("evidence", "") or "",
    )


def parse(doc: Any, *, sha256: str = "", source_path: str = "") -> TargetSchema:
    """Build a `TargetSchema` from an already-parsed YAML/JSON document."""
    _require(isinstance(doc, dict), "schema document must be a mapping")
    _check_keys(doc, _DOC_KEYS, "schema")

    schema_id = doc.get("schema_id")
    _require(isinstance(schema_id, str) and schema_id.strip(),
             "schema: schema_id is required")
    version = doc.get("schema_version")
    _require(isinstance(version, str) and _VERSION_RE.match(version),
             f"schema: schema_version must be MAJOR.MINOR.PATCH, got {version!r}")

    raw_fields = doc.get("fields")
    _require(isinstance(raw_fields, list) and raw_fields,
             "schema: fields must be a non-empty list")
    fields = tuple(_parse_field(f, i) for i, f in enumerate(raw_fields))

    names = [f.name for f in fields]
    dupes = sorted({n for n in names if names.count(n) > 1})
    _require(not dupes, f"schema: duplicate field definitions for {dupes}")
    name_set = set(names)

    for key in ("row_key", "column_order", "artifacts"):
        v = doc.get(key)
        _require(v is None or isinstance(v, list), f"schema: {key} must be a list")

    row_key = tuple(str(c) for c in (doc.get("row_key") or []))
    unknown_key = sorted(set(row_key) - name_set)
    _require(not unknown_key, f"schema: row_key names undeclared fields {unknown_key}")

    column_order = tuple(str(c) for c in (doc.get("column_order") or []))
    if column_order:
        unknown_order = sorted(set(column_order) - name_set)
        _require(not unknown_order,
                 f"schema: column_order names undeclared fields {unknown_order}")
        missing_order = sorted(name_set - set(column_order))
        _require(not missing_order,
                 f"schema: column_order omits declared fields {missing_order}")

    raw_constraints = doc.get("constraints") or []
    _require(isinstance(raw_constraints, list), "schema: constraints must be a list")
    constraints = tuple(_parse_constraint(c, i, name_set)
                        for i, c in enumerate(raw_constraints))
    cids = [c.id for c in constraints]
    dup_c = sorted({c for c in cids if cids.count(c) > 1})
    _require(not dup_c, f"schema: duplicate constraint ids {dup_c}")

    null_encoding = doc.get("null_encoding", "")
    _require(isinstance(null_encoding, str),
             "schema: null_encoding must be a string")

    return TargetSchema(
        schema_id=schema_id,
        schema_version=version,
        title=doc.get("title", "") or "",
        description=doc.get("description", "") or "",
        evidence=doc.get("evidence", "") or "",
        artifacts=tuple(str(a) for a in (doc.get("artifacts") or [])),
        row_key=row_key,
        column_order=column_order,
        column_order_severity=_severity(doc.get("column_order_severity"),
                                        "schema.column_order_severity", ERROR),
        unexpected_column_severity=_severity(doc.get("unexpected_column_severity"),
                                             "schema.unexpected_column_severity", ERROR),
        null_encoding=null_encoding,
        fields=fields,
        constraints=constraints,
        sha256=sha256,
        source_path=source_path,
    )


def load(path: str | Path) -> TargetSchema:
    """Read and validate a schema file. Raises `SchemaError` on any problem."""
    from .source_ref import sha256_file  # local import: keeps this module standalone

    p = Path(path)
    if not p.is_file():
        raise SchemaError(f"schema file not found: {p}")
    try:
        doc = yaml.safe_load(p.read_text())
    except yaml.YAMLError as exc:
        raise SchemaError(f"schema file is not valid YAML: {exc}") from exc
    return parse(doc, sha256=sha256_file(p), source_path=str(p))

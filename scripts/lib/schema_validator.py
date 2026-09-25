"""schema_validator.py — Intrinsic validation of a harmonized table against a schema.

Project-agnostic, like `schema_model`. Every rule below is driven entirely by the
schema instance; nothing here names a CIMAC field, trial or clinical value.

Intrinsic versus extrinsic
--------------------------
The pre-existing `validate_extractions.py` is *extrinsic*: it compares one
artifact to a reference table. That can only speak about rows the reference
contains, so the two trials absent from it (10013, 14C0059G in the CIMAC case)
received no automated value checking at all. This validator is *intrinsic*: it
asks whether a table is internally consistent with its declared contract, so it
covers every row of every artifact regardless of whether a reference exists.

No silent coercion
------------------
The artifact is read as raw text (`dtype=str`, `keep_default_na=False`) and this
module does its own typing. Letting pandas infer would make several defects
invisible: `"NA"` would become a legitimate-looking null, a stray non-numeric
token would turn a whole column to `object` without complaint, and a value the
schema forbids could be silently rewritten into one it allows. A value that
cannot be interpreted as its declared type is reported, never converted.

Findings
--------
A `Finding` is one rule firing against one field, carrying a count and up to
`MAX_EXAMPLES` located examples rather than one row per violation — a column of
1,700 bad values is one problem, not 1,700. Ordering is fully deterministic
(severity, rule, field, message) so two runs over the same input produce
byte-identical reports.
"""
from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass, field as dc_field
from pathlib import Path
from typing import Any, Iterable

import pandas as pd

from .schema_model import (
    ERROR,
    INFO,
    SEVERITY_RANK,
    T_INTEGER,
    T_NUMBER,
    T_STRING,
    WARNING,
    Field,
    TargetSchema,
)

MAX_EXAMPLES = 5

#: Strings that look like a null but are not the schema's declared encoding.
#: Kept here rather than in the schema: recognising that `"NA"` is null-ish is a
#: property of tabular data, not of any one project's vocabulary.
NULLISH = ("na", "n/a", "nan", "none", "null", "<na>", "nil", ".")


class ArtifactError(ValueError):
    """The artifact could not be read at all. Distinct from a validation failure."""


@dataclass(frozen=True)
class Finding:
    severity: str
    rule: str
    field: str
    message: str
    count: int = 1
    examples: tuple[str, ...] = ()

    def sort_key(self) -> tuple:
        return (SEVERITY_RANK.get(self.severity, 9), self.rule, self.field, self.message)


@dataclass
class ValidationResult:
    schema_ref: str
    schema_id: str
    schema_version: str
    schema_sha256: str
    artifact: str
    artifact_sha256: str
    n_rows: int
    n_cols: int
    findings: list[Finding] = dc_field(default_factory=list)

    @property
    def n_errors(self) -> int:
        return sum(1 for f in self.findings if f.severity == ERROR)

    @property
    def n_warnings(self) -> int:
        return sum(1 for f in self.findings if f.severity == WARNING)

    @property
    def n_info(self) -> int:
        return sum(1 for f in self.findings if f.severity == INFO)

    @property
    def verdict(self) -> str:
        return "fail" if self.n_errors else "pass"

    def sorted_findings(self) -> list[Finding]:
        return sorted(self.findings, key=lambda f: f.sort_key())

    def to_dict(self) -> dict:
        return {
            "schema_ref": self.schema_ref,
            "schema_id": self.schema_id,
            "schema_version": self.schema_version,
            "schema_sha256": self.schema_sha256,
            "artifact": self.artifact,
            "artifact_sha256": self.artifact_sha256,
            "n_rows": self.n_rows,
            "n_cols": self.n_cols,
            "verdict": self.verdict,
            "counts": {"error": self.n_errors, "warning": self.n_warnings,
                       "info": self.n_info},
            "findings": [asdict(f) for f in self.sorted_findings()],
        }

    def to_json(self, *, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, sort_keys=True)


# ── loading ─────────────────────────────────────────────────────────────────

def load_artifact(path: str | Path) -> pd.DataFrame:
    """Read a delivered CSV as raw text so this module can do its own typing."""
    p = Path(path)
    if not p.is_file():
        raise ArtifactError(f"artifact not found: {p}")
    try:
        df = pd.read_csv(p, dtype=str, keep_default_na=False, na_filter=False)
    except (OSError, UnicodeDecodeError, pd.errors.ParserError,
            pd.errors.EmptyDataError) as exc:
        raise ArtifactError(f"artifact could not be read: {p} ({exc})") from exc
    return df


def _examples(values: Iterable[Any], locations: Iterable[Any] | None = None
              ) -> tuple[str, ...]:
    """Up to MAX_EXAMPLES rendered examples, deduplicated and order-stable."""
    out: list[str] = []
    seen: set[str] = set()
    locs = list(locations) if locations is not None else None
    for i, v in enumerate(values):
        text = f"{v!r}" if locs is None else f"{v!r} (row {locs[i]})"
        if text in seen:
            continue
        seen.add(text)
        out.append(text)
        if len(out) >= MAX_EXAMPLES:
            break
    return tuple(out)


def _is_null(series: pd.Series, null_encoding: str) -> pd.Series:
    return series == null_encoding


# ── individual rule groups ──────────────────────────────────────────────────

def _check_structure(schema: TargetSchema, df: pd.DataFrame, artifact: str,
                     out: list[Finding]) -> None:
    cols = list(df.columns)

    dupes = sorted({c for c in cols if cols.count(c) > 1})
    for c in dupes:
        out.append(Finding(ERROR, "structure.duplicate_column", c,
                           f"column {c!r} appears {cols.count(c)} times"))

    expected = [f.name for f in schema.fields_for(artifact) if f.required_column]
    for name in expected:
        if name not in cols:
            out.append(Finding(ERROR, "structure.missing_column", name,
                               f"required column {name!r} is absent"))

    declared = {f.name for f in schema.fields}
    for c in sorted(set(cols) - declared):
        out.append(Finding(schema.unexpected_column_severity,
                           "structure.unexpected_column", c,
                           f"column {c!r} is not declared in the schema"))

    if schema.column_order:
        applicable = [c for c in schema.column_order
                      if (fl := schema.field(c)) and fl.applies_to(artifact)]
        if cols != applicable and not (set(applicable) - set(cols)):
            out.append(Finding(schema.column_order_severity,
                               "structure.column_order", "",
                               "delivered column order differs from the schema "
                               "column_order contract",
                               examples=(f"expected {applicable}", f"actual {cols}")))

    if schema.row_key and all(k in cols for k in schema.row_key):
        key = df[list(schema.row_key)].agg("||".join, axis=1)
        dup_mask = key.duplicated(keep=False)
        n_dup = int(dup_mask.sum())
        if n_dup:
            first = df.loc[dup_mask, list(schema.row_key)].head(MAX_EXAMPLES)
            out.append(Finding(ERROR, "structure.duplicate_row_key", "",
                               f"{n_dup} rows share a row key "
                               f"{list(schema.row_key)} that must be unique",
                               count=n_dup,
                               examples=_examples(first.to_dict("records"))))


def _check_nulls(fl: Field, series: pd.Series, schema: TargetSchema,
                 out: list[Finding]) -> pd.Series:
    """Report null problems; return the mask of values to type-check."""
    is_null = _is_null(series, schema.null_encoding)

    nullish = (~is_null) & series.str.strip().str.lower().isin(NULLISH)
    n = int(nullish.sum())
    if n:
        out.append(Finding(ERROR, "type.unsupported_null_encoding", fl.name,
                           f"{n} values use a null-like token instead of the "
                           f"schema null encoding {schema.null_encoding!r}",
                           count=n,
                           examples=_examples(series[nullish].head(MAX_EXAMPLES),
                                              series[nullish].head(MAX_EXAMPLES).index)))

    if not fl.nullable:
        n_null = int(is_null.sum())
        if n_null:
            out.append(Finding(ERROR, "value.unexpected_null", fl.name,
                               f"{n_null} nulls in a field declared nullable: false",
                               count=n_null,
                               examples=_examples(is_null[is_null].head(MAX_EXAMPLES).index)))

    padded = (~is_null) & (series != series.str.strip())
    n_pad = int(padded.sum())
    if n_pad:
        out.append(Finding(WARNING, "value.whitespace_padding", fl.name,
                           f"{n_pad} values carry leading or trailing whitespace",
                           count=n_pad,
                           examples=_examples(series[padded].head(MAX_EXAMPLES))))

    return ~is_null & ~nullish


def _check_type(fl: Field, series: pd.Series, present: pd.Series,
                out: list[Finding]) -> pd.Series | None:
    """Validate the declared type. Returns numeric values for numeric fields."""
    values = series[present]
    if fl.type == T_STRING:
        return None

    numeric = pd.to_numeric(values, errors="coerce")
    bad = numeric.isna()
    n_bad = int(bad.sum())
    if n_bad:
        out.append(Finding(ERROR, "type.not_numeric", fl.name,
                           f"{n_bad} values cannot be read as {fl.type}",
                           count=n_bad,
                           examples=_examples(values[bad].head(MAX_EXAMPLES),
                                              values[bad].head(MAX_EXAMPLES).index)))
    good = numeric.dropna()
    if fl.type == T_INTEGER and not good.empty:
        frac = good[(good % 1) != 0]
        if not frac.empty:
            out.append(Finding(ERROR, "type.not_integer", fl.name,
                               f"{len(frac)} values are not integral",
                               count=len(frac),
                               examples=_examples(frac.head(MAX_EXAMPLES),
                                                  frac.head(MAX_EXAMPLES).index)))
    return good


def _check_vocabulary(fl: Field, series: pd.Series, present: pd.Series,
                      out: list[Finding]) -> None:
    if fl.vocabulary is None:
        return
    vocab = fl.vocabulary
    values = series[present]
    permitted = set(vocab.values)

    unknown_mask = ~values.isin(permitted)
    n_unknown = int(unknown_mask.sum())
    if n_unknown:
        distinct = sorted(set(values[unknown_mask]))
        out.append(Finding(vocab.unknown_severity, "vocabulary.unknown_value", fl.name,
                           f"{n_unknown} values ({len(distinct)} distinct) are not in "
                           f"the declared vocabulary",
                           count=n_unknown, examples=_examples(distinct)))

    if vocab.canonical:
        non_canon = set(vocab.non_canonical)
        mask = values.isin(non_canon)
        n = int(mask.sum())
        if n:
            distinct = sorted(set(values[mask]))
            out.append(Finding(vocab.non_canonical_severity,
                               "vocabulary.non_canonical_value", fl.name,
                               f"{n} values are permitted but not canonical "
                               f"(normalization inconsistency)",
                               count=n, examples=_examples(distinct)))


def _check_bounds(fl: Field, numeric: pd.Series | None, out: list[Finding]) -> None:
    if fl.bounds is None or numeric is None or numeric.empty:
        return
    b = fl.bounds
    if b.min is not None:
        below = numeric[numeric < b.min]
        if not below.empty:
            out.append(Finding(b.severity, "bounds.below_minimum", fl.name,
                               f"{len(below)} values below the declared minimum {b.min}",
                               count=len(below),
                               examples=_examples(below.head(MAX_EXAMPLES),
                                                  below.head(MAX_EXAMPLES).index)))
    if b.max is not None:
        above = numeric[numeric > b.max]
        if not above.empty:
            out.append(Finding(b.severity, "bounds.above_maximum", fl.name,
                               f"{len(above)} values above the declared maximum {b.max}",
                               count=len(above),
                               examples=_examples(above.head(MAX_EXAMPLES),
                                                  above.head(MAX_EXAMPLES).index)))


def _check_pattern(fl: Field, series: pd.Series, present: pd.Series,
                   out: list[Finding]) -> None:
    if not fl.pattern:
        return
    rx = re.compile(fl.pattern)
    values = series[present]
    bad = values[~values.map(lambda v: bool(rx.fullmatch(v)))]
    if not bad.empty:
        out.append(Finding(fl.pattern_severity, "value.pattern_mismatch", fl.name,
                           f"{len(bad)} values do not match the declared pattern "
                           f"{fl.pattern!r}",
                           count=len(bad),
                           examples=_examples(sorted(set(bad)))))


def _check_deprecated(fl: Field, df: pd.DataFrame, out: list[Finding]) -> None:
    if fl.deprecated and fl.name in df.columns:
        note = f" — {fl.deprecation_note}" if fl.deprecation_note else ""
        out.append(Finding(INFO, "schema.deprecated_field", fl.name,
                           f"field is marked deprecated in the schema{note}"))


def _check_open_questions(fl: Field, df: pd.DataFrame, out: list[Finding]) -> None:
    """Surface, never enforce.

    A field under an unresolved review question is reported at INFO so the
    question stays visible in every validation report. Turning an open question
    into an error would make a mechanism nobody has agreed to block a delivery.
    """
    if fl.open_questions and fl.name in df.columns:
        out.append(Finding(INFO, "schema.open_question", fl.name,
                           f"field is subject to {len(fl.open_questions)} open review "
                           f"question(s); the schema records them without enforcing them",
                           count=len(fl.open_questions),
                           examples=tuple(fl.open_questions[:MAX_EXAMPLES])))


def _check_constraints(schema: TargetSchema, df: pd.DataFrame, artifact: str,
                       out: list[Finding]) -> None:
    for c in schema.constraints:
        fa, fb = schema.field(c.field), schema.field(c.other)
        if not (fa and fb and fa.applies_to(artifact) and fb.applies_to(artifact)):
            continue
        if c.field not in df.columns or c.other not in df.columns:
            continue
        left, right = df[c.field], df[c.other]
        ln = _is_null(left, schema.null_encoding)
        rn = _is_null(right, schema.null_encoding)

        if c.kind == "requires_non_null":
            mask = (~ln) & rn
            n = int(mask.sum())
            if n:
                out.append(Finding(c.severity, f"constraint.{c.id}", c.field,
                                   f"{n} rows have {c.field!r} present while "
                                   f"{c.other!r} is null",
                                   count=n,
                                   examples=_examples(mask[mask].head(MAX_EXAMPLES).index)))
        elif c.kind == "less_or_equal":
            both = (~ln) & (~rn)
            lv = pd.to_numeric(left[both], errors="coerce")
            rv = pd.to_numeric(right[both], errors="coerce")
            comparable = lv.notna() & rv.notna()
            viol = comparable & (lv > rv)
            n = int(viol.sum())
            if n:
                idx = viol[viol].head(MAX_EXAMPLES).index
                out.append(Finding(c.severity, f"constraint.{c.id}", c.field,
                                   f"{n} rows have {c.field!r} greater than {c.other!r}",
                                   count=n,
                                   examples=_examples(
                                       [f"{c.field}={lv[i]}, {c.other}={rv[i]}" for i in idx],
                                       idx)))


def _check_coverage(fl: Field, series: pd.Series, schema: TargetSchema,
                    n_rows: int, out: list[Finding]) -> None:
    """Descriptive: a field that is entirely null in an artifact it applies to."""
    if n_rows and int(_is_null(series, schema.null_encoding).sum()) == n_rows:
        out.append(Finding(INFO, "coverage.all_null", fl.name,
                           f"field is null in all {n_rows} rows of this artifact",
                           count=n_rows))


# ── provenance-aware missing-state rule (optional) ──────────────────────────

def check_value_states(schema: TargetSchema, provenance: pd.DataFrame,
                       *, artifact: str = "") -> list[Finding]:
    """Compare observed value_states against those each field declares.

    Optional and generic: it needs a long-form provenance frame with
    `harmonized_field` and `value_state` columns, and reports states a field did
    not declare. Runs only when such a frame is supplied, so the core validator
    stays artifact-only.

    This is the rule that gives missing-state semantics teeth. Without it a
    schema could claim a blank `age` means `not_applicable` while the pipeline
    recorded `source_missing`, and nothing would notice.
    """
    out: list[Finding] = []
    needed = {"harmonized_field", "value_state"}
    if not needed.issubset(provenance.columns):
        return out
    prov = provenance.copy()
    prov["value_state"] = prov["value_state"].fillna("").astype(str)
    # An empty value_state means "delivered" (the pre-Phase-1 default) and is
    # never a missing state, so it is not compared against the declaration.
    prov = prov[prov["value_state"] != ""]
    for fl in schema.fields:
        if artifact and not fl.applies_to(artifact):
            continue
        seen = sorted(set(prov.loc[prov["harmonized_field"] == fl.name, "value_state"]))
        undeclared = [s for s in seen if s not in fl.value_states]
        if undeclared:
            out.append(Finding(WARNING, "provenance.undeclared_value_state", fl.name,
                               f"provenance records value_state(s) the schema does "
                               f"not declare for this field",
                               count=len(undeclared),
                               examples=_examples(undeclared)))
    return out


# ── entry point ─────────────────────────────────────────────────────────────

def validate(schema: TargetSchema, df: pd.DataFrame, *, artifact: str,
             artifact_sha256: str = "") -> ValidationResult:
    """Validate one loaded artifact against one schema. Pure: no I/O."""
    findings: list[Finding] = []
    _check_structure(schema, df, artifact, findings)

    for fl in schema.fields_for(artifact):
        _check_deprecated(fl, df, findings)
        _check_open_questions(fl, df, findings)
        if fl.name not in df.columns:
            continue
        col = df[fl.name]
        if isinstance(col, pd.DataFrame):      # duplicate column name; already reported
            continue
        series = col.astype(str)
        present = _check_nulls(fl, series, schema, findings)
        numeric = _check_type(fl, series, present, findings)
        _check_vocabulary(fl, series, present, findings)
        _check_bounds(fl, numeric, findings)
        _check_pattern(fl, series, present, findings)
        _check_coverage(fl, series, schema, len(df), findings)

    _check_constraints(schema, df, artifact, findings)

    return ValidationResult(
        schema_ref=schema.ref,
        schema_id=schema.schema_id,
        schema_version=schema.schema_version,
        schema_sha256=schema.sha256,
        artifact=artifact,
        artifact_sha256=artifact_sha256,
        n_rows=len(df),
        n_cols=len(df.columns),
        findings=findings,
    )


def render(result: ValidationResult, *, show_examples: bool = True) -> str:
    """Human-readable report. Deterministic for a given result."""
    L: list[str] = []
    L.append("─" * 74)
    L.append(f"SCHEMA VALIDATION  {result.artifact}")
    L.append("─" * 74)
    L.append(f"schema      {result.schema_ref}")
    if result.schema_sha256:
        L.append(f"            sha256 {result.schema_sha256[:16]}…")
    if result.artifact_sha256:
        L.append(f"artifact    sha256 {result.artifact_sha256[:16]}…")
    L.append(f"shape       {result.n_rows} rows × {result.n_cols} cols")
    L.append(f"VERDICT     {result.verdict.upper()}  "
             f"({result.n_errors} error, {result.n_warnings} warning, "
             f"{result.n_info} info)")
    L.append("")
    if not result.findings:
        L.append("No findings.")
        return "\n".join(L)

    for sev in (ERROR, WARNING, INFO):
        group = [f for f in result.sorted_findings() if f.severity == sev]
        if not group:
            continue
        L.append(f"{sev.upper()} ({len(group)})")
        for f in group:
            where = f.field or "(table)"
            L.append(f"  [{f.rule}] {where}")
            L.append(f"      {f.message}")
            if show_examples and f.examples:
                for ex in f.examples:
                    L.append(f"        · {ex}")
        L.append("")
    return "\n".join(L).rstrip() + "\n"

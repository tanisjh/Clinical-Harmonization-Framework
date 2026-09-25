"""spec_preflight.py — deterministic checks a specification can fail BEFORE it runs.

A specification is a document, and a document can be wrong in two very different
ways. It can be malformed — the parser already refuses that — or it can be
well-formed and describe something the data does not contain. The second kind is
the dangerous one, because nothing about it looks wrong.

The case that motivated this module: a mapping naming a source column that does
not exist. `pandas.Series.get` returns `None` for a key it does not hold, so a
misspelled column produced a well-formed cell carrying no value, at whatever
confidence the mapping declared for absence. The run succeeded. The table was
full of nulls that looked like missing data rather than a typo. An author — and
especially an agent proposing a specification from a data dictionary — has no
way to tell those apart after the fact.

So this module answers one question: **does every name this document uses
actually exist where it says it does?**

WHAT IT IS NOT
--------------
It does not check that a mapping is clinically right, that a confidence is
justified, or that a value map is complete. Those are review questions and no
amount of determinism settles them. Calling this "validation" without that
caveat would overstate it.

It also does not recompute anything. Provenance and trace answer "where did this
value come from"; this answers "can this document run at all".

THE EXECUTABLE SUBSET
---------------------
The grammar deliberately parses more than the runtime executes — a construct can
be declared, reviewed and staged before the engine drives it. That is useful for
design and fatal for an author who assumes parsing means running. The runtime's
own constants are the authority here, imported rather than restated, so this
module cannot drift from what the executor actually supports.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

from lib import rule_registry, vocabularies
from lib.spec_model import ANCHOR_SOURCE_ID, HarmonizationSpec


@dataclass(frozen=True)
class ColumnRef:
    """One place a specification names a column of one source."""

    source: str
    column: str
    #: Where the name was written, for a diagnostic that can be acted on.
    where: str
    #: A column that legitimately may be absent — a member-specific column in a
    #: source group, where each member file carries its own spelling.
    optional: bool = False


@dataclass
class Finding:
    kind: str
    message: str

    def __str__(self) -> str:                      # pragma: no cover - trivial
        return f"[{self.kind}] {self.message}"


@dataclass
class PreflightReport:
    spec_ref: str
    findings: list[Finding] = field(default_factory=list)
    summary: dict[str, Any] = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        return not self.findings

    def fail(self, kind: str, message: str) -> None:
        self.findings.append(Finding(kind, message))


# ── what the runtime can actually drive ─────────────────────────────────────

def executable_subset() -> dict[str, tuple[str, ...]]:
    """The runtime's own view of what it supports.

    Read from the executor's constants rather than restated, so a construct
    cannot appear supported here after being removed there.
    """
    from lib import spec_executor as ex

    return {
        "ops": tuple(ex.SUPPORTED_OPS),
        "readers": tuple(ex._READER_LOADERS),
        "cardinalities": tuple(ex.SUPPORTED_CARDINALITIES),
        # The executor implements these three and refuses the rest by name.
        "duplicate_policies": ("first", "last", "error"),
    }


def _file_types_for_readers(readers: Iterable[str]) -> tuple[str, ...]:
    """File types the supported readers can open."""
    from lib import spec_executor as ex

    kinds = {ex._READER_LOADERS[r] for r in readers if r in ex._READER_LOADERS}
    out: list[str] = []
    if "csv" in kinds:
        out += ["csv"]
    if "xlsx" in kinds:
        out += ["xlsx", "xls"]
    return tuple(out)


# ── every place a document names a source column ────────────────────────────

def column_references(spec: HarmonizationSpec) -> list[ColumnRef]:
    """Collect every source column this specification says it will read.

    Deliberately conservative about what counts. A `custom_rule`'s `column` is a
    LABEL describing its inputs (`"BOR.binary,pfs_time"`), not a column; so is
    `not_applicable`'s. Anchor columns are the row anchor's, not a file's, and
    the executor already refuses a missing one by name. Neither is collected
    here, because a false positive would block a correct specification.
    """
    from lib.spec_model import _COLUMN_OPS

    refs: list[ColumnRef] = []

    def add(source: str, column: str, where: str, optional: bool = False) -> None:
        if not source or not column or source == ANCHOR_SOURCE_ID:
            return
        if _is_group_ref(spec, source):
            # A group role resolves per record to one member's source; each
            # member is checked through `column_by_member` instead.
            return
        refs.append(ColumnRef(source, column, where, optional))

    for m in spec.mappings:
        where = f"mapping {m.target!r}"
        if m.op in _COLUMN_OPS and m.column:
            add(m.source, m.column, where)
        for member_id, column in (m.column_by_member or {}).items():
            for sid in _member_sources(spec, m.group or m.source, member_id):
                add(sid, column, f"{where}.column_by_member[{member_id!r}]", True)
        for i, cand in enumerate(m.candidates or ()):
            add(cand.source, cand.column, f"{where}.candidates[{i}]")
            if cand.when is not None:
                add(getattr(cand.when, "source", ""), getattr(cand.when, "column", ""),
                    f"{where}.candidates[{i}].when")
        for i, clause in enumerate(m.clauses or ()):
            add(clause.source, clause.column, f"{where}.clauses[{i}]")
        if m.when and getattr(m.when, "column", ""):
            add(getattr(m.when, "source", ""), m.when.column, f"{where}.when")

    for r in spec.reads:
        where = f"read {r.id!r}"
        add(r.source, r.column, where)
        for member_id, column in (r.column_by_member or {}).items():
            for sid in _member_sources(spec, r.source, member_id):
                add(sid, column, f"{where}.column_by_member[{member_id!r}]", True)

    rec = spec.record
    if rec.grouping and rec.anchor_source:
        for c in rec.grouping.key_columns:
            add(rec.anchor_source, c, "record.grouping.key_columns")
        for target_field, column in (rec.grouping.anchor_fields or {}).items():
            add(rec.anchor_source, column,
                f"record.grouping.anchor_fields[{target_field!r}]")

    return refs


def _is_group_ref(spec: HarmonizationSpec, ref: str) -> bool:
    return "." in ref and any(g.id == ref.split(".", 1)[0] for g in spec.source_groups)


def _member_sources(spec: HarmonizationSpec, ref: str, member_id: str) -> list[str]:
    """The source ids a named group member supplies."""
    gid = ref.split(".", 1)[0] if ref else ""
    for group in spec.source_groups:
        if group.id != gid:
            continue
        for member in group.members:
            if member.id == member_id:
                return sorted(set(member.sources.values()))
    return []


# ── participant keys ────────────────────────────────────────────────────────

def participant_column_for(spec: HarmonizationSpec, source_id: str) -> str:
    """The column this source uses for the record's participant identity."""
    src = spec.source(source_id)
    if src is not None and src.participant_column:
        return src.participant_column
    return spec.record.participant_key


def sources_needing_participant_index(spec: HarmonizationSpec) -> list[str]:
    """Sources the executor will index by participant.

    The grouping anchor is excluded for the same reason the executor excludes
    it: it holds several rows per record by design, and collapsing it to one
    would discard the group.
    """
    grouping_source = (spec.record.anchor_source
                       if spec.record.anchor == "grouping" else "")
    needed = {r.source for r in column_references(spec)}
    for decl in spec.record.additional_records:
        if decl.participants_from:
            needed.add(decl.participants_from)
    return sorted(sid for sid in needed if sid and sid != grouping_source)


# ── reading a source's header without running the harmonization ─────────────

def source_columns(spec: HarmonizationSpec, source_id: str,
                   project_root: Path) -> tuple[str, ...]:
    """The columns a source actually has, read through the declared reader.

    Uses the same readers the executor uses, so a header that only appears after
    a preamble is skipped is the header this sees. Raises the same errors the
    executor would raise for a missing or uncontained file.
    """
    from lib.cidc_io import read_cidc_csv
    from lib.source_ref import assert_within_roots

    src = spec.source(source_id)
    if src is None:
        raise KeyError(f"no source {source_id!r} is declared")
    path = assert_within_roots(project_root / src.path, project_root)
    if not path.is_file():
        raise FileNotFoundError(str(path))
    kind = executable_subset()["readers"]
    if src.reader not in kind:
        raise ValueError(f"reader {src.reader!r} is not executable")
    if src.reader == "xlsx_openpyxl@v1":
        from lib.xlsx_reader import parse_xlsx_bytes
        frame = parse_xlsx_bytes(path.read_bytes(), sheet_name=src.sheet)
    elif src.reader == "excel@v1":
        import pandas as pd
        frame = pd.read_excel(path, sheet_name=src.sheet or 0, nrows=0)
    else:
        frame = read_cidc_csv(path).df
    return tuple(str(c) for c in frame.columns)


def check_columns(spec: HarmonizationSpec, project_root: Path,
                  report: PreflightReport) -> dict[str, tuple[str, ...]]:
    """Every declared column name must exist in the file it names.

    This is the check that closes the silent-null hole. A column the file does
    not have is reported with the source, the missing name and where the
    specification asked for it — the three things needed to fix it — rather than
    becoming a null nobody can distinguish from missing data.
    """
    refs = column_references(spec)
    headers: dict[str, tuple[str, ...]] = {}
    wanted = {r.source for r in refs} | set(sources_needing_participant_index(spec))
    for sid in sorted(wanted):
        try:
            headers[sid] = source_columns(spec, sid, project_root)
        except FileNotFoundError as e:
            report.fail("source.missing_file",
                        f"source {sid!r}: file does not exist: {e}")
        except Exception as e:                       # containment, reader, sheet
            report.fail("source.unreadable",
                        f"source {sid!r}: {type(e).__name__}: {e}")

    for ref in refs:
        cols = headers.get(ref.source)
        if cols is None or ref.column in cols:
            continue
        if ref.optional:
            continue
        near = _suggest(ref.column, cols)
        hint = f"; did you mean {near!r}?" if near else ""
        report.fail(
            "column.missing",
            f"{ref.where}: source {ref.source!r} has no column "
            f"{ref.column!r}{hint}")

    for sid in sources_needing_participant_index(spec):
        cols = headers.get(sid)
        if cols is None:
            continue
        key = participant_column_for(spec, sid)
        if key not in cols:
            near = _suggest(key, cols)
            hint = f"; did you mean {near!r}?" if near else ""
            report.fail(
                "participant_key.missing",
                f"source {sid!r} has no participant key column {key!r}"
                f"{hint} (declare Source.participant_column if this file "
                f"spells it differently)")
    return headers


def _suggest(name: str, options: Iterable[str] | None) -> str:
    """The closest available column name, when one is close enough to help."""
    import difflib

    if not options:
        return ""
    hit = difflib.get_close_matches(name, list(options), n=1, cutoff=0.75)
    return hit[0] if hit else ""


# ── the whole pre-flight ────────────────────────────────────────────────────

def preflight(spec: HarmonizationSpec, project_root: Path, *,
              check_sources: bool = True) -> PreflightReport:
    """Everything deterministic that can be known before the harmonization runs.

    Grammar and schema conformance are already settled by the time a spec object
    exists — the parser refuses a malformed document. What is added here is
    everything the parser deliberately does not do: whether the constructs are
    ones the runtime can actually drive, whether the registered rules resolve,
    and whether the files and columns the document names are really there.
    """
    report = PreflightReport(spec_ref=spec.ref)
    subset = executable_subset()

    # 1. executable subset — parsing is not running
    for src in spec.sources:
        if src.reader not in subset["readers"]:
            report.fail("reader.not_executable",
                        f"source {src.id!r}: reader {src.reader!r} parses but "
                        f"cannot be executed (executable: "
                        f"{list(subset['readers'])})")
        allowed_types = _file_types_for_readers(subset["readers"])
        if src.file_type not in allowed_types:
            report.fail("file_type.not_executable",
                        f"source {src.id!r}: file_type {src.file_type!r} parses "
                        f"but no executable reader opens it (executable: "
                        f"{list(allowed_types)})")
        if not src.path:
            report.fail("source.path_pattern",
                        f"source {src.id!r}: declares path_pattern, which the "
                        f"executor does not resolve; declare an exact path")
    for m in spec.mappings:
        if m.op not in subset["ops"]:
            report.fail("op.not_executable",
                        f"mapping {m.target!r}: op {m.op!r} parses but is not "
                        f"implemented (executable: {list(subset['ops'])})")
    policy = spec.record.duplicates.policy
    if policy not in subset["duplicate_policies"]:
        report.fail("duplicates.not_executable",
                    f"record.duplicates.policy {policy!r} parses but is not "
                    f"implemented (executable: "
                    f"{list(subset['duplicate_policies'])})")
    for join in spec.record.joins:
        if join.cardinality not in subset["cardinalities"]:
            report.fail("join.not_executable",
                        f"join {join.id!r}: cardinality {join.cardinality!r} "
                        f"parses but is not implemented")
    if spec.record.anchor not in ("external", "grouping"):
        report.fail("anchor.not_executable",
                    f"record.anchor {spec.record.anchor!r} parses but this "
                    f"executor sources records from an external anchor or by "
                    f"grouping a source")

    # 2. registered rules resolve by id, version AND kind
    for m in spec.mappings:
        if m.op == "custom_rule":
            _check_rule(report, m.rule, "field", f"mapping {m.target!r}")
    for pr in spec.post_rules:
        _check_rule(report, pr.rule, "post", f"post_rule {pr.id!r}")

    # 3. vocabularies resolve
    for ref in spec.vocabularies:
        try:
            vocabularies.resolve(ref)
        except Exception as e:
            report.fail("vocabulary.unresolved", f"{ref}: {e}")

    # 4. the files and columns really exist
    headers: dict[str, tuple[str, ...]] = {}
    if check_sources:
        headers = check_columns(spec, project_root, report)

    report.summary = _summarize(spec, headers)
    return report


def _check_rule(report: PreflightReport, rule, kind: str, where: str) -> None:
    if rule is None or not rule.id:
        report.fail("rule.missing", f"{where}: names no rule")
        return
    try:
        rule_registry.resolve(rule.id, rule.version or "1", kind=kind)
    except Exception as e:
        report.fail("rule.unresolved", f"{where}: {type(e).__name__}: {e}")


def _summarize(spec: HarmonizationSpec, headers: dict) -> dict[str, Any]:
    """A structural picture a reviewer can read without running anything."""
    open_questions = sorted({
        m.rule.open_question_ref for m in spec.mappings
        if m.rule is not None and getattr(m.rule, "open_question_ref", "")})
    return {
        "spec_ref": spec.ref,
        "trial": spec.trial,
        "status": spec.status,
        "target_schema": spec.target_schema_ref,
        "anchor": spec.record.anchor,
        "participant_key": spec.record.participant_key,
        "record_key_column": spec.record.record_key_column or spec.record.participant_key,
        "sources": [
            {"id": s.id, "path": s.path, "reader": s.reader,
             "sheet": s.sheet, "participant_column": s.participant_column
             or spec.record.participant_key,
             "columns_in_file": len(headers.get(s.id, ())) or None}
            for s in spec.sources],
        "target_fields": sorted({m.target for m in spec.mappings}),
        "mappings_by_op": _counted(m.op for m in spec.mappings),
        "vocabularies": list(spec.vocabularies),
        "registered_rules": sorted({m.rule.id for m in spec.mappings
                                    if m.op == "custom_rule" and m.rule}),
        "exclusions": [e.id for e in spec.exclusions],
        "post_rules": [p.id for p in spec.post_rules],
        "additional_records": [d.id for d in spec.record.additional_records],
        "open_questions": open_questions,
    }


def _counted(values: Iterable[str]) -> dict[str, int]:
    out: dict[str, int] = {}
    for v in values:
        out[v] = out.get(v, 0) + 1
    return dict(sorted(out.items()))

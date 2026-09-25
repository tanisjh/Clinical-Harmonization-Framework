"""spec_model.py — The contract between mapping proposals and the deterministic engine.

Project-agnostic, like `schema_model`. This module knows about sources, record
construction, mapping operations, joins and exclusions. It does not know that
`clinical_benefit` exists, that `CIMAC-s1400i` is a trial, or that `nivo` is a
treatment; those live in a specification instance under `specs/`.

What a specification is
-----------------------
The target schema says *what the harmonized output means*. A harmonization
specification says *how one particular source dataset maps into it*. Together
they are everything the deterministic engine needs, which is the point: the
onboarding layer (eventually AI-assisted) produces a specification, humans
approve the clinical judgements inside it, and the engine executes it with no
model in the path.

What a specification is NOT
---------------------------
It carries no runtime facts. No resolved paths, no content hashes, no row counts,
no timestamps. A `Source` declares where a file is *expected*; what was actually
opened is a runtime `SourceRef`, recorded in provenance and the run manifest.
Merging those layers would make a static document that cannot be reviewed before
a run and cannot be compared across runs.

Why there is no expression language
-----------------------------------
A mapping selects one of a fixed set of typed operations. There is deliberately
no embedded Python, no eval, no lambda: a specification is data to be reviewed,
and a reviewer cannot approve arbitrary code. When current logic needs something
the operation set cannot express, it is declared as `custom_rule` — a named
deterministic implementation with an explicit input/output contract — rather
than pretending the rule is declarative. That escape hatch is what makes gradual
migration honest.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field as dc_field, replace as dc_replace
from pathlib import Path
from typing import Any

import yaml

from .conditions import (
    NULLARY_PREDICATES as _NULLARY_PREDICATES,
    NUMERIC_PREDICATES as _NUMERIC_PREDICATES,
    PREDICATES,
    SET_PREDICATES as _SET_PREDICATES,
    VALUE_PREDICATES as _VALUE_PREDICATES,
    PredicateDefinitionError,
    validate_predicate_definition,
)

# ── operations ──────────────────────────────────────────────────────────────
OP_DIRECT = "direct"
OP_CONSTANT = "constant"
OP_VALUE_MAP = "value_map"
OP_FALLBACK = "fallback"
OP_CONDITIONAL = "conditional"
OP_UNIT_CONVERSION = "unit_conversion"
OP_FILE_MEMBERSHIP = "file_membership"
OP_JOIN_LOOKUP = "join_lookup"
OP_NOT_APPLICABLE = "not_applicable"
#: The target has NO APPROVED MAPPING YET. Distinct from every other absence:
#: `not_applicable` says the field does not apply to this trial, and a
#: `value_map` miss says a map was consulted and did not cover the value. This
#: says the clinical decision has not been made, so the source value is read and
#: carried as the raw value, no value is delivered, and the low confidence and
#: the rule id are what flag it for review. Deleting the mapping instead would
#: lose the flag; inventing a map would answer a question that is still open.
OP_NO_MAPPING_CONFIGURED = "no_mapping_configured"
OP_CUSTOM_RULE = "custom_rule"

OPERATIONS = (
    OP_DIRECT, OP_CONSTANT, OP_VALUE_MAP, OP_FALLBACK, OP_CONDITIONAL,
    OP_UNIT_CONVERSION, OP_FILE_MEMBERSHIP, OP_JOIN_LOOKUP, OP_NOT_APPLICABLE,
    OP_NO_MAPPING_CONFIGURED, OP_CUSTOM_RULE,
)

#: Operations that read a named column from a named source.
_COLUMN_OPS = (OP_DIRECT, OP_VALUE_MAP, OP_UNIT_CONVERSION,
               OP_NO_MAPPING_CONFIGURED)

#: Reserved source id for the row anchor. A mapping may read a column from the
#: anchor rows themselves rather than from a declared file — CIMAC's sample
#: identifiers live only there. Permitted when `record.anchor` is `external`
#: (the anchor was supplied) or `grouping` (the engine constructed it from the
#: group key). Spelled with a `$` so it can never collide with a real source id.
ANCHOR_SOURCE_ID = "$anchor"

#: How a source group decides which of its members holds a record. Closed: the
#: only rule implemented is "the first member that contains it", which is the
#: only one the cohort-split pattern needs.
GROUP_SELECTIONS = ("first_containing_record",)

#: Separator in a group-qualified source reference, `<group>.<role>`.
GROUP_REF_SEP = "."

#: Readers a source may declare. A reader is *how* a file is opened, which
#: `file_type` alone does not settle: the CIDC exports carry a preamble that a
#: plain `pandas.read_csv` cannot parse. Closed on purpose — a specification
#: never names Python.
READERS = ("cidc_csv@v1", "plain_csv@v1", "excel@v1",
           "xlsx_openpyxl@v1")

#: `confidence: from_rule` — the value's confidence is computed by the
#: registered rule that produces it, not declared here. Legitimate only for
#: operations whose implementation returns a confidence (`custom_rule`); an
#: ordinary operation must state a number, so that clinical confidence policy
#: is visible in the reviewed document rather than defaulted in Python.
CONFIDENCE_FROM_RULE = "from_rule"

#: How a `direct` mapping coerces the source value. `none` copies verbatim;
#: `numeric` parses a number; `text` collapses whitespace and yields nothing for
#: an absent value, so a blank cell becomes an absence rather than a NaN.
COERCIONS = ("none", "numeric", "text")

#: Whether the verbatim source value is recorded alongside the result. `none`
#: exists only to reproduce an existing provenance stream that omits it.
#: `value` records the verbatim source value; `coerced` records the value after
#: the mapping's own transformation; `none` records nothing. All three occur in
#: the existing provenance stream.
RAW_VALUE_MODES = ("value", "none", "coerced")

#: What happens when a source holds no row for the record being built. `error`
#: is the default and the safe one: a record that should be there and is not is
#: a finding, never a silent null.
MISSING_RECORD_MODES = ("error", "lookup_miss", "omit")

#: Where the condition contribution sits in a gated fallback's contribution
#: list. Legacy compatibility only: two mappings in one extractor disagree.
CONDITION_POSITIONS = ("first", "last")

#: What a group records for the member that holds the record. `selected` says
#: the file supplied the value; `condition` says the file's identity decided
#: which values were read but supplied none itself. Both occur.
MEMBERSHIP_STATUSES = ("selected", "condition")

#: What a candidate is called once an earlier one has won. `not_evaluated` is
#: the honest default — the chain short-circuited and never looked. `rejected`
#: exists because one legacy mapping records it that way.
AFTER_DECISION_STATUSES = ("not_evaluated", "rejected")

#: Whether the mapping's declared value state is written into the delivered
#: cell. `none` exists only to reproduce an existing provenance stream that
#: predates the typed state and carries the same meaning in its rule id.
VALUE_STATE_MODES = ("declared", "none")

#: Where a provenance row-number comes from. `source` is the row in the frame
#: as read; `none` records no row number (correct when the value did not come
#: from a source row at all).
ROW_NUMBER_MODES = ("source", "none")

#: Comparison predicates available to `conditional` and gated `fallback`
#: candidates. Deliberately small and total: each is decidable on one value
#: without evaluating code.
#: How a record maps to source rows.
ANCHOR_KINDS = ("external", "source", "grouping")

#: How a grouped anchor chooses the one source row it cites for a group.
#:
#: Closed, and deliberately tiny. `first_row` means the first row of the group
#: IN SOURCE READ ORDER — not sorted-first, and not "any of them". A grouped
#: record cites a row so a reader can go and look at it; which row that is has
#: to be reproducible, and read order is the only ordering the engine can claim
#: without the specification declaring one.
GROUPING_SELECTIONS = ("first_row",)

#: How a value's record key is built when it is not a literal. `participant` is
#: the default everywhere and needs no declaration; `group` is for a value whose
#: identity is the whole grouped key, not the participant alone.
RECORD_KEY_SOURCES = ("participant", "group")

#: Where a record came from. `observed` means the source data contained it;
#: `synthesized` means an approved rule declared that it should exist. A
#: specification names the difference so a mapping can describe each honestly —
#: a synthesized record has no source row to cite and must not pretend to one.
RECORD_ORIGINS = ("observed", "synthesized")

#: When an additional record is created. Closed, and currently one member: the
#: participant is known from some source but no record was constructed for them.
#: Anything richer would be a row-generation language, which this is not.
ADDITIONAL_RECORD_CONDITIONS = ("no_record_for_participant",)

#: What to do when several source rows share a record key.
DUPLICATE_POLICIES = ("first", "last", "error", "highest_confidence")

CARDINALITIES = ("one_to_one", "many_to_one", "one_to_many")

#: Value states a mapping may declare. Matches lib/value_states.py, but is
#: restated here so this module does not depend on a project's runtime library.
VALUE_STATES = ("delivered", "transformed", "provisional", "suppressed",
                "row_excluded", "source_missing", "not_applicable")

STATUSES = ("draft", "proposed", "approved", "superseded")

FILE_TYPES = ("csv", "tsv", "xlsx", "xls", "json", "parquet")

EXCLUSION_SCOPES = ("participant", "record", "sample")

_VERSION_RE = re.compile(r"^\d+\.\d+\.\d+$")
#: <kind>[.<qualifier>].v<n> — the convention the rule vocabulary already uses.
_RULE_ID_RE = re.compile(r"^[a-z0-9_]+(\.[a-z0-9_]+)*\.v\d+$")
_MAPPING_ID_RE = re.compile(r"^[a-z][a-z0-9_.-]*$")

# Accepted keys per level. Anything else is a load error.
_DOC_KEYS = {
    "spec_id", "spec_version", "project", "trial", "target_schema", "status",
    "description", "evidence", "sources", "source_groups", "record", "mappings",
    "exclusions", "allow_duplicate_targets", "post_rules", "reads", "conditions",
}
_READ_KEYS = {"id", "source", "column", "column_by_member", "coerce",
              "description"}
_CONDITION_KEYS = {"id", "read", "predicate", "value", "group", "member",
                   "record_origin", "description"}
_DERIVATION_KEYS = {"when", "record"}

#: The closed set of runtime facts a derivation record may reference. Each is
#: something the engine already knows while producing the value; none of them
#: is computed for the purpose. A reference outside this set is a load error.
#:
#: This is SUBSTITUTION, not programming: a specification says which facts to
#: record, and the engine fills in the ones it has. There is no arithmetic, no
#: expression, no conditional inside a value, and no way to reach into Python.
REF_MARKER = "$ref"
REF_VALUE = "value"                 # the delivered value
REF_RAW_VALUE = "raw_value"         # the value as read, before transformation
REF_SOURCE_COLUMN = "source_column"  # the column the value came from
REF_SOURCE_LABEL = "source_label"   # the logical label of the file it came from
#: Facts a GROUPED anchor knows about the record it just constructed. Named
#: `anchor_*` so they cannot be confused with the `group_*` references above,
#: which are about which FILE holds a record, not which rows compose one.
REF_ANCHOR_GROUP_SIZE = "anchor_group_size"     # how many source rows the group had
REF_ANCHOR_SELECTED_ROW = "anchor_selected_row"  # the row number of the cited row
REF_PREFIXES = ("read", "condition", "group_member", "group_value", "group_label")
REF_SIMPLE = (REF_VALUE, REF_RAW_VALUE, REF_SOURCE_COLUMN, REF_SOURCE_LABEL,
              REF_ANCHOR_GROUP_SIZE, REF_ANCHOR_SELECTED_ROW)
_POST_RULE_KEYS = {"id", "rule", "artifacts", "inputs", "params", "confidence",
                   "value_state", "provenance", "description", "canonical"}
_POST_PROV_KEYS = {"source_label", "stage", "supersedes_rule"}
_GROUP_KEYS = {"id", "select", "decided_by", "members", "description",
               "evidence_ref", "membership_status", "rule"}
_GROUP_MEMBER_KEYS = {"id", "value", "sources", "description"}
_SOURCE_KEYS = {"id", "path", "path_pattern", "file_type", "reader",
                "logical_label", "sheet", "record_keys", "participant_column",
                "role", "description", "on_missing_record"}
_MISSING_KEYS = {"mode", "emit_for", "confidence", "rule", "column"}
_RECORD_KEYS = {"anchor", "anchor_source", "participant_key",
                "record_key_column", "sample_key",
                "collection_event_key", "duplicates", "joins", "grouping",
                "additional_records", "description"}
_GROUPING_KEYS = {"key_columns", "select", "anchor_fields", "description"}
_ADDITIONAL_RECORD_KEYS = {"id", "participants_from", "when", "values", "rule",
                           "description"}
_JOIN_KEYS = {"id", "left", "right", "keys", "cardinality", "description"}
_DUP_KEYS = {"policy", "by"}
_MAPPING_KEYS = {
    "mapping_id", "target", "op", "source", "column", "value", "map", "on_unmapped",
    "candidates", "clauses", "otherwise", "factor", "members", "join",
    "lookup_column", "implementation", "inputs", "output_type", "value_state",
    "artifacts", "rule", "description", "confidence", "coerce", "round",
    "provenance", "params", "no_value_rule", "group", "column_by_member",
    "fallback_vocabulary", "unmapped_rule", "contributions", "when",
    "derivation",
}
_CONTRIB_KEYS = {"condition_position", "after_decision", "no_value_column",
                 "record", "conditions", "condition_rank"}

#: Whether a fallback records its candidate list. `none` exists because one
#: legacy mapping records the outcome only when the chain is exhausted.
CONTRIBUTION_RECORDING = ("all", "none")
_CONFIDENCE_KEYS = {"value", "no_value", "unmapped"}
#: Marker for a parameter whose value is a shared vocabulary, referenced rather
#: than copied. Resolved at load time; see lib/vocabularies.py.
VOCABULARY_MARKER = "$vocabulary"
_PROVENANCE_KEYS = {"source_ref", "source_label", "record_key",
                    "record_key_from", "row_number", "raw_value", "row_from",
                    "value_state", "contributions_from", "raw_from"}
_RULE_KEYS = {"id", "version", "rationale", "evidence", "approval_ref",
              "open_question_ref", "review_status"}
_CANDIDATE_KEYS = {"source", "column", "when", "label", "description"}
_CLAUSE_KEYS = {"when", "source", "column", "value", "read", "description"}
_MEMBER_KEYS = {"source", "value", "column"}
_WHEN_KEYS = {"source", "column", "predicate", "value", "read"}
_EXCLUSION_KEYS = {"id", "scope", "source", "match", "reason", "rule",
                   "applies_to", "description", "provenance", "canonical"}
_EXCL_PROV_KEYS = {"source_label", "evidence_path"}
_MATCH_KEYS = {"field", "predicate", "value"}
_TARGET_SCHEMA_KEYS = {"id", "version"}


class SpecError(ValueError):
    """The specification document is invalid. Raised only at load time."""


def _require(cond: bool, msg: str) -> None:
    if not cond:
        raise SpecError(msg)


def _check_keys(got: dict, allowed: set[str], where: str) -> None:
    unknown = sorted(set(got) - allowed)
    _require(not unknown,
             f"{where}: unknown propert{'y' if len(unknown) == 1 else 'ies'} "
             f"{unknown} (allowed: {sorted(allowed)})")


def _str(raw: dict, key: str, where: str, *, required: bool = True) -> str:
    v = raw.get(key)
    if v is None and not required:
        return ""
    _require(isinstance(v, str) and v.strip(), f"{where}: {key} is required")
    return v


# ── dataclasses ─────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class Source:
    """Where an input is expected — never where one was actually found."""

    id: str
    file_type: str
    path: str = ""
    path_pattern: str = ""
    #: How to open the file. `file_type` says what shape it is; `reader` says
    #: which vetted reader handles its quirks. Required to execute.
    reader: str = ""
    #: The stable name this source is known by in provenance and in any
    #: configuration keyed on it. Distinct from the resolved path and hash,
    #: which are runtime facts. Defaults to the basename actually opened.
    logical_label: str = ""
    sheet: str = ""
    #: Which columns identify one ROW of this file. This is source record
    #: identity and nothing else: it may be composite (a participant x timepoint
    #: file needs both), its order carries no meaning, and it is NOT the join
    #: key. A file whose participant column is spelled differently from the
    #: record's says so in `participant_column`, not here.
    record_keys: tuple[str, ...] = ()
    #: The column in THIS file that carries the logical participant identity
    #: named by `record.participant_key`. Native trial files spell it their own
    #: way — E4412's workbook says `CIMAC ID` where the canonical model says
    #: `cimac_part_id` — and the values are the same identifiers either way.
    #: Empty means "spelled the same as the record's participant key", which is
    #: the CIDC-prepared case and stays the default.
    participant_column: str = ""
    role: str = ""
    description: str = ""
    #: What this source does when it holds no row for the record being built.
    on_missing_record: "MissingRecord" = dc_field(default_factory=lambda: MissingRecord())


@dataclass(frozen=True)
class Read:
    """A named, per-record source lookup a specification can refer to.

    Exists so a predicate and a provenance record can talk about the same source
    value without repeating where it lives — and so a value read from one group's
    file can inform a mapping that reads another's.
    """

    id: str
    source: str = ""
    column: str = ""
    column_by_member: dict[str, str] = dc_field(default_factory=dict)
    coerce: str = "none"
    description: str = ""


@dataclass(frozen=True)
class Condition:
    """A named yes/no fact about one record.

    Either a predicate over a named read, or the question "did this source group
    resolve to that member". Both are things the engine already determines; a
    condition just gives the answer a name so provenance can record it.
    """

    id: str
    read: str = ""
    predicate: str = ""
    value: Any = None
    group: str = ""
    member: str = ""
    #: Tests how this record came to exist, rather than anything it contains.
    record_origin: str = ""
    description: str = ""


@dataclass(frozen=True)
class DerivationVariant:
    """One structured provenance record, optionally gated by a condition."""

    record: dict[str, Any]
    when: str = ""


@dataclass(frozen=True)
class MissingRecord:
    """What a source does when it holds no row for the record being built.

    Declared per source because that is where the fact lives: "this file has no
    row for this participant" is a property of the file, and what it means is a
    decision about that file, not about any one mapping.
    """

    mode: str = "error"
    #: Targets that receive an explicit lookup-miss cell. Not necessarily the
    #: targets this source otherwise supplies: an existing stream records a miss
    #: for a field this source never provides, and listing it here is what makes
    #: that visible instead of hidden.
    emit_for: tuple[str, ...] = ()
    confidence: float = 0.0
    rule: RuleRef | None = None
    #: The column name recorded on a miss. There is no real column — the record
    #: itself is absent — so the label says that rather than naming a field.
    column: str = ""


@dataclass(frozen=True)
class GroupMember:
    """One arm of a cohort split across parallel file sets."""

    id: str
    #: What membership of this arm MEANS — the value a `file_membership`
    #: mapping delivers when a record is found here. Clinical content, so it
    #: lives in the specification.
    value: Any = None
    #: role -> declared source id. Every member must offer the same roles, so a
    #: mapping can say "the demographics file of whichever arm this record is
    #: in" without knowing which arm that turns out to be.
    sources: dict[str, str] = dc_field(default_factory=dict)
    description: str = ""


@dataclass(frozen=True)
class SourceGroup:
    """Parallel sets of files, one set per arm of a split cohort.

    Some trials do not record the arm in a column: they put each arm in its own
    files, so the evidence for the arm is *which file holds the record*, and the
    files to read for everything else follow from that one decision.
    """

    id: str
    members: tuple[GroupMember, ...] = ()
    select: str = "first_containing_record"
    #: Which role's membership decides. The other roles follow the decision
    #: rather than being probed independently, because a record's arm is one
    #: fact, not one fact per file.
    decided_by: str = ""
    #: Where the member -> value table came from, recorded in the derivation of
    #: every value this group decides.
    evidence_ref: str = ""
    #: How the winning member is recorded on the membership contribution.
    membership_status: str = "selected"
    #: How the membership decision itself is identified in provenance. A group
    #: is a decision in its own right, so it names its own rule rather than
    #: borrowing one from whichever mapping happens to consume it.
    rule: "RuleRef | None" = None
    description: str = ""

    @property
    def roles(self) -> tuple[str, ...]:
        return tuple(sorted(self.members[0].sources)) if self.members else ()


@dataclass(frozen=True)
class Join:
    id: str
    left: str
    right: str
    on: tuple[str, ...]
    cardinality: str = "one_to_one"
    description: str = ""


@dataclass(frozen=True)
class Duplicates:
    policy: str = "error"
    by: tuple[str, ...] = ()


@dataclass(frozen=True)
class Grouping:
    """How one logical record is built from a GROUP of source rows.

    This is grouping, not aggregation. The engine collects the rows that share
    the declared key, keeps one of them as the row the record cites, and counts
    how many there were. It never combines their values — there is no minimum,
    no maximum, no sum and no concatenation, because a record whose values were
    computed across several rows could not honestly cite one of them.

    Every anchor field therefore comes from the group KEY, which by definition
    is the same in every row of the group. Anything else would be a choice the
    specification had not made.
    """

    #: The columns whose combination defines one record. Order matters: it is
    #: the order the grouped record key is written in.
    key_columns: tuple[str, ...] = ()
    select: str = "first_row"
    #: anchor field -> the key column that supplies it. A declared anchor field
    #: left out of this map has no source and is absent on every record.
    anchor_fields: dict[str, str] = dc_field(default_factory=dict)
    description: str = ""


@dataclass(frozen=True)
class AdditionalRecord:
    """A record an approved rule says must exist even though no source row is.

    Grouping constructs the records the data *contains*. Sometimes a trial's
    approved reading of its own data says a further record belongs there — a
    participant is known to exist, the event that would have produced a row was
    never recorded, and the harmonized table is expected to carry a row for them
    anyway.

    That is a decision, not a computation, so it is declared: which source
    enumerates the candidates, what makes one necessary, and what the resulting
    record says. There is no expression language here and no way to write one —
    the condition is a closed vocabulary of one, and the values are literals.
    """

    id: str
    #: The declared source whose participant key enumerates the candidates.
    participants_from: str
    when: str = "no_record_for_participant"
    #: record field -> the literal it takes. The participant key comes from the
    #: source and may not be restated here.
    values: dict[str, Any] = dc_field(default_factory=dict)
    rule: "RuleRef | None" = None
    description: str = ""


@dataclass(frozen=True)
class Record:
    """How one output row is constructed from source rows."""

    anchor: str = "source"
    anchor_source: str = ""
    participant_key: str = ""
    #: How this record's identity is SPELLED in provenance — the name in the
    #: `name=value` source record key. Distinct from `participant_key`, which
    #: names the logical identity the anchor and the canonical model use, and
    #: distinct from `Source.record_keys`, which is per-file row identity.
    #: Empty means "use `participant_key`", which is what ten of eleven trials
    #: do; E4412's legacy provenance writes `CIMAC ID=<pid>` on every cell,
    #: including derived ones, so the name is a property of the record rather
    #: than of whichever file a cell came from.
    record_key_column: str = ""
    sample_key: str = ""
    collection_event_key: str = ""
    duplicates: Duplicates = dc_field(default_factory=Duplicates)
    joins: tuple[Join, ...] = ()
    #: Present only when `anchor` is `grouping`.
    grouping: Grouping | None = None
    #: Records an approved rule adds after the constructed ones.
    additional_records: tuple[AdditionalRecord, ...] = ()
    description: str = ""


@dataclass(frozen=True)
class Predicate:
    source: str
    column: str
    predicate: str
    value: Any = None
    #: A named read, as an alternative to naming a source and column here. Lets
    #: a predicate and a provenance record talk about the same source value.
    read: str = ""


@dataclass(frozen=True)
class ContributionStyle:
    """Legacy-compatibility shape of a gated fallback's contribution list."""

    condition_position: str = "last"
    after_decision: str = "not_evaluated"
    #: The cell's source column when every candidate is exhausted. Defaults to
    #: every candidate column joined, which names what was tried.
    no_value_column: str = ""
    #: "" is `all`. `none` withholds the candidate list from the delivered cell.
    record: str = "all"
    #: Named reads recorded as `condition` contributions: values that informed
    #: which branch was taken but supplied no value of their own.
    conditions: tuple[str, ...] = ()
    #: The rank those condition contributions carry. -1 is "outside the ranked
    #: candidate list", which is what a condition normally is. A mapping whose
    #: condition IS its only input ranks it 0, because there was no list.
    condition_rank: int = -1


@dataclass(frozen=True)
class Candidate:
    """One step of a fallback chain, optionally gated by a predicate."""

    source: str
    column: str
    when: Predicate | None = None
    #: How this candidate names itself in the delivered cell's source column
    #: when it wins. Defaults to the column, which is what it is.
    label: str = ""
    description: str = ""


@dataclass(frozen=True)
class Clause:
    """One branch of a conditional: when this holds, take that value."""

    when: Predicate
    source: str = ""
    column: str = ""
    value: Any = None
    #: Yield a named read's value — the same value a predicate can test, so a
    #: clause can say "keep what we read" without repeating where it lives.
    read: str = ""
    description: str = ""


@dataclass(frozen=True)
class Member:
    """One arm of a file-membership rule: if the record is in this source ..."""

    source: str
    value: Any = None
    column: str = ""


@dataclass(frozen=True)
class RuleRef:
    """The provenance and approval identity of a mapping."""

    id: str
    version: str = "1"
    rationale: str = ""
    evidence: str = ""
    approval_ref: str = ""
    open_question_ref: str = ""
    review_status: str = ""


@dataclass(frozen=True)
class ProvenanceRef:
    """How a value's provenance identity is recorded.

    Every field is optional and every default is generic. This exists because a
    value that did not come from a source row still has to say where it came
    from, and because an executor reproducing an existing provenance stream may
    need to keep that stream's established labels. Those labels are data that
    belongs to the trial's specification, never to the reusable runtime.
    """

    #: Attribute the value to a declared file source, so the runtime records its
    #: resolved path and content hash even though the value was read elsewhere.
    source_ref: str = ""
    #: A literal non-file origin label, e.g. a configuration block. Mutually
    #: exclusive with `source_ref`.
    source_label: str = ""
    #: A literal record key, for values whose origin is not a source row.
    record_key: str = ""
    #: Which identity this value's record key states, when it is not a literal.
    #: "" is `participant`, which is what every value used before grouped
    #: records existed. `group` names the whole grouped key, for a value that
    #: belongs to the group rather than to the participant.
    record_key_from: str = ""
    #: "" means: `source` for a value read from a source row, `none` otherwise.
    row_number: str = ""
    #: "" is `value`. `none` omits the raw value, which only reproduces an
    #: existing stream that omitted it.
    raw_value: str = ""
    #: Take the source row number and record key from this declared source's
    #: matched row. A derived value has no source row of its own, but it is
    #: still *about* a record, and saying which one keeps it traceable.
    row_from: str = ""
    #: "" is `declared`. `none` withholds the declared value state from the
    #: delivered cell, for a stream that predates the typed vocabulary and
    #: carries the same meaning in its rule id instead.
    value_state: str = ""
    #: Record a named read as this value's raw value, for a mapping whose
    #: answer is a transformation of something read elsewhere.
    raw_from: str = ""
    #: Attach a source group's membership evidence to this value. Which file a
    #: record was read from is part of why the value is what it is, so a mapping
    #: that reads through a group can carry that evidence rather than leaving a
    #: reader to infer it.
    contributions_from: str = ""


@dataclass(frozen=True)
class Mapping:
    target: str
    op: str
    rule: RuleRef
    mapping_id: str = ""
    source: str = ""
    column: str = ""
    value: Any = None
    map: dict[str, Any] = dc_field(default_factory=dict)
    on_unmapped: str = "null"
    candidates: tuple[Candidate, ...] = ()
    clauses: tuple[Clause, ...] = ()
    otherwise: Any = None
    factor: float | None = None
    members: tuple[Member, ...] = ()
    join: str = ""
    lookup_column: str = ""
    implementation: str = ""
    inputs: tuple[str, ...] = ()
    output_type: str = ""
    value_state: str = ""
    artifacts: tuple[str, ...] = ()
    description: str = ""
    #: A number in [0, 1], or `from_rule` when a registered rule computes it.
    #: `None` means undeclared, which an executor must refuse to execute.
    confidence: float | str | None = None
    #: The confidence carried when the mapping produces no usable value. `None`
    #: means the mapping never declared one, which is correct for a mapping that
    #: always yields a value and is a refusal for one that does not.
    confidence_no_value: float | None = None
    #: How the no-value outcome is identified in provenance, when that differs
    #: from the ordinary rule. An exhausted fallback is not the same event as a
    #: successful one, and saying so is what makes the difference traceable.
    no_value_rule: RuleRef | None = None
    #: The confidence carried when a source value is PRESENT but no mapping
    #: entry recognises it. Distinct from `confidence_no_value`, which is about
    #: there being no value at all: "we do not know this code" and "there is
    #: nothing here" are different facts about the data.
    confidence_unmapped: float | None = None
    #: How the unrecognised-value outcome is identified in provenance.
    unmapped_rule: RuleRef | None = None
    #: Shape of the contribution list a gated fallback records.
    contributions: ContributionStyle = dc_field(default_factory=ContributionStyle)
    #: A named condition gating whether this mapping applies to a record at all.
    #: Two mappings for one target, gated on complementary conditions, is how a
    #: trial says "these records are decided by one rule and those by another"
    #: without either rule pretending to cover both.
    when: str = ""
    #: The structured provenance record this mapping emits, as declared
    #: variants. Values are literals or references to runtime facts.
    derivation: tuple[DerivationVariant, ...] = ()
    #: A shared vocabulary consulted when this mapping's own table has no entry
    #: for a source value. Referenced by `id@version`; resolved at load.
    fallback_vocabulary: str = ""
    #: The resolved ordered rules of that vocabulary. Populated at load so an
    #: executor never has to reach back out to the filesystem mid-run.
    fallback_rules: tuple = ()
    #: Runtime-only identity of the exact loaded vocabulary that supplied
    #: ``fallback_rules``.  It is additive and never serialized into the
    #: declarative specification.
    fallback_vocabulary_object: Any = None
    #: The source group a `file_membership` mapping decides over.
    group: str = ""
    #: member id -> column name, for a group whose members name the same
    #: harmonized fact in differently-titled columns. The harmonized meaning is
    #: one thing; only the label differs.
    column_by_member: dict[str, str] = dc_field(default_factory=dict)
    #: Declared parameters handed to a registered rule — a landmark threshold, a
    #: response vocabulary. These are clinical decisions and live here, never in
    #: the runtime registry.
    params: dict[str, Any] = dc_field(default_factory=dict)
    #: Runtime-only ``(parameter_name, loaded Vocabulary)`` pairs for exact
    #: top-level ``{$vocabulary: ref}`` parameters.  ``params`` keeps the
    #: resolved executable tables; this separately preserves replay evidence.
    parameter_vocabularies: tuple[tuple[str, Any], ...] = ()
    coerce: str = "none"
    round: bool = False
    provenance: ProvenanceRef = dc_field(default_factory=ProvenanceRef)


@dataclass(frozen=True)
class PostRule:
    """A decision that adjusts an artifact after its table has been assembled.

    Not a mapping: a mapping turns source data into a value, while this reads
    values the harmonization has already produced and changes some of them, for
    named artifacts only. It cannot run before the table exists, and it cannot
    be expressed as a field rule without pretending it can.
    """

    id: str
    rule: RuleRef
    #: True when this adjustment is part of getting the harmonization right —
    #: the default, because a decision recorded in an approved specification is
    #: normally one the canonical result depends on. A rule that exists only to
    #: reproduce a historical comparison sets it False.
    canonical: bool = True
    #: The artifacts this adjustment applies to, for a delivery layer that still
    #: distinguishes them. NOT how the canonical path decides anything: a new
    #: trial must be harmonizable without knowing any artifact name.
    artifacts: tuple[str, ...] = ()
    #: Target fields the rule reads from the assembled table.
    inputs: tuple[str, ...] = ()
    #: The clinical parameters of the decision — thresholds, value sets. These
    #: live here and never in the runtime registry.
    params: dict[str, Any] = dc_field(default_factory=dict)
    confidence: float = 1.0
    value_state: str = ""
    #: Compatibility labels for the provenance an existing stream records.
    source_label: str = ""
    stage: str = ""
    #: The rule id recorded as superseded. Left unset, the rule id of the cell
    #: this adjustment actually replaces is used, which is the accurate answer;
    #: an existing stream that records one id for several fields declares it.
    supersedes_rule: str = ""
    description: str = ""


@dataclass(frozen=True)
class Exclusion:
    id: str
    scope: str
    reason: str
    rule: RuleRef
    source: str = ""
    match: dict[str, Any] = dc_field(default_factory=dict)
    #: True when this exclusion is part of getting the harmonization right —
    #: the record does not belong in the harmonized result at all. False marks
    #: one that exists only for a historical comparison.
    canonical: bool = True
    applies_to: tuple[str, ...] = ()
    description: str = ""
    #: Compatibility labels for the provenance an existing stream records: where
    #: the decision is written down, and under what origin label.
    source_label: str = ""
    evidence_path: str = ""


@dataclass(frozen=True)
class HarmonizationSpec:
    spec_id: str
    spec_version: str
    trial: str
    target_schema_id: str
    target_schema_version: str
    project: str = ""
    status: str = "draft"
    description: str = ""
    evidence: str = ""
    sources: tuple[Source, ...] = ()
    record: Record = dc_field(default_factory=Record)
    source_groups: tuple[SourceGroup, ...] = ()
    reads: tuple[Read, ...] = ()
    conditions: tuple[Condition, ...] = ()
    mappings: tuple[Mapping, ...] = ()
    exclusions: tuple[Exclusion, ...] = ()
    post_rules: tuple[PostRule, ...] = ()
    allow_duplicate_targets: tuple[str, ...] = ()
    #: Shared vocabularies this specification references, as `id@version`.
    #: Referenced, not copied: cross-trial clinical knowledge lives once.
    vocabularies: tuple[str, ...] = ()
    sha256: str = ""
    source_path: str = ""

    @property
    def ref(self) -> str:
        return f"{self.spec_id}@{self.spec_version}"

    @property
    def target_schema_ref(self) -> str:
        return f"{self.target_schema_id}@{self.target_schema_version}"

    def source(self, sid: str) -> Source | None:
        for s in self.sources:
            if s.id == sid:
                return s
        return None

    def canonical_post_rules(self) -> tuple["PostRule", ...]:
        """Table-level rules the canonical harmonization depends on."""
        return tuple(r for r in self.post_rules if r.canonical)

    def canonical_exclusions(self) -> tuple["Exclusion", ...]:
        """Exclusions the canonical harmonization depends on."""
        return tuple(e for e in self.exclusions if e.canonical)

    def post_rules_for(self, artifact: str) -> tuple["PostRule", ...]:
        """Post-rules that apply to one artifact, in declared order."""
        return tuple(r for r in self.post_rules if artifact in r.artifacts)

    def exclusions_for(self, artifact: str) -> tuple["Exclusion", ...]:
        """Exclusions that drop records from one artifact, in declared order."""
        return tuple(e for e in self.exclusions if artifact in e.applies_to)

    def group(self, gid: str) -> SourceGroup | None:
        for g in self.source_groups:
            if g.id == gid:
                return g
        return None

    def mappings_for(self, target: str) -> tuple[Mapping, ...]:
        return tuple(m for m in self.mappings if m.target == target)

    def custom_rules(self) -> tuple[Mapping, ...]:
        return tuple(m for m in self.mappings if m.op == OP_CUSTOM_RULE)

    def coverage(self) -> dict:
        """How much of this spec is declarative, and how much is custom."""
        total = len(self.mappings)
        custom = len(self.custom_rules())
        return {"mappings": total, "generic": total - custom, "custom": custom,
                "generic_pct": (100.0 * (total - custom) / total) if total else 0.0}


# ── parsing ─────────────────────────────────────────────────────────────────

def _parse_rule(raw: Any, where: str) -> RuleRef:
    _require(isinstance(raw, dict), f"{where}: rule must be a mapping")
    _check_keys(raw, _RULE_KEYS, where)
    rid = _str(raw, "id", f"{where}.rule")
    _require(bool(_RULE_ID_RE.match(rid)),
             f"{where}: rule id {rid!r} does not match the <kind>[.<qualifier>].v<n> "
             f"convention")
    version = raw.get("version", "1")
    _require(isinstance(version, (str, int)), f"{where}: rule version must be a string or int")
    return RuleRef(
        id=rid, version=str(version),
        rationale=raw.get("rationale", "") or "",
        evidence=raw.get("evidence", "") or "",
        approval_ref=raw.get("approval_ref", "") or "",
        open_question_ref=raw.get("open_question_ref", "") or "",
        review_status=raw.get("review_status", "") or "",
    )


def _parse_predicate(raw: Any, where: str) -> Predicate:
    _require(isinstance(raw, dict), f"{where}: predicate must be a mapping")
    _check_keys(raw, _WHEN_KEYS, where)
    pred = _str(raw, "predicate", where)
    value = raw.get("value")
    try:
        validate_predicate_definition(pred, value)
    except PredicateDefinitionError as exc:
        raise SpecError(f"{where}: {exc}") from exc
    read = raw.get("read", "") or ""
    if read:
        _require(not raw.get("source") and not raw.get("column"),
                 f"{where}: a predicate names either a read or a source and "
                 f"column, not both")
        return Predicate(source="", column="", predicate=pred, value=value,
                         read=read)
    return Predicate(source=_str(raw, "source", where),
                     column=_str(raw, "column", where),
                     predicate=pred, value=value)


def _confidence_number(val: Any, where: str, label: str) -> float:
    _require(isinstance(val, (int, float)) and not isinstance(val, bool),
             f"{where}: confidence.{label} must be a number in [0, 1], got {val!r}")
    _require(0.0 <= float(val) <= 1.0,
             f"{where}: confidence.{label} must lie in [0, 1], got {val!r}")
    return float(val)


def _found_confidence(val: Any, where: str) -> float | str:
    """The confidence of a value the mapping actually found.

    One acceptance, used by both spellings of `confidence`. A number states the
    worth of the value outright; `from_rule` says a registered rule computes it
    and the specification will not assert a number the rule would override.
    Which spelling is used says nothing about the outcome being described, so
    the scalar form and the `value:` key of the mapping form accept the same
    thing.
    """
    if isinstance(val, str):
        _require(val == CONFIDENCE_FROM_RULE,
                 f"{where}: confidence must be a number in [0, 1] or "
                 f"{CONFIDENCE_FROM_RULE!r}, got {val!r}")
        return CONFIDENCE_FROM_RULE
    _require(isinstance(val, (int, float)) and not isinstance(val, bool),
             f"{where}: confidence must be a number in [0, 1] or "
             f"{CONFIDENCE_FROM_RULE!r}, got {val!r}")
    _require(0.0 <= float(val) <= 1.0,
             f"{where}: confidence must lie in [0, 1], got {val!r}")
    return float(val)


def _parse_confidence(raw: dict, where: str) -> float | str | None:
    """Parse a mapping's declared confidence.

    Absent is represented as None rather than defaulted. A confidence that a
    reviewer never wrote down is not a confidence of 0.7; it is a gap, and an
    executor is expected to refuse the mapping rather than invent a number.
    """
    if "confidence" not in raw:
        return None
    val = raw["confidence"]
    if isinstance(val, dict):
        # The two-outcome form: what this value is worth when the mapping finds
        # one, and what it is worth when it does not. Both are stated, because
        # the second is a policy decision and not the absence of one. A rule
        # that computes its own confidence still computes it for the value it
        # found; what an absence is worth is not the rule's to say, so
        # `from_rule` and `no_value` compose rather than conflict.
        _check_keys(val, _CONFIDENCE_KEYS, f"{where}.confidence")
        _require("value" in val,
                 f"{where}: confidence must declare `value` when written as a "
                 f"mapping")
        return _found_confidence(val["value"], where)
    return _found_confidence(val, where)


def _parse_provenance(raw: Any, where: str) -> ProvenanceRef:
    if raw is None:
        return ProvenanceRef()
    where = f"{where}.provenance"
    _require(isinstance(raw, dict), f"{where}: must be a mapping")
    _check_keys(raw, _PROVENANCE_KEYS, where)
    source_ref = raw.get("source_ref", "") or ""
    source_label = raw.get("source_label", "") or ""
    _require(not (source_ref and source_label),
             f"{where}: declare at most one of source_ref and source_label — a "
             f"value has one origin")
    row_number = raw.get("row_number", "") or ""
    if row_number:
        _require(row_number in ROW_NUMBER_MODES,
                 f"{where}: row_number must be one of {list(ROW_NUMBER_MODES)}, "
                 f"got {row_number!r}")
    raw_mode = raw.get("raw_value", "") or ""
    if raw_mode:
        _require(raw_mode in RAW_VALUE_MODES,
                 f"{where}: raw_value must be one of {list(RAW_VALUE_MODES)}, "
                 f"got {raw_mode!r}")
    state_mode = raw.get("value_state", "") or ""
    if state_mode:
        _require(state_mode in VALUE_STATE_MODES,
                 f"{where}: value_state must be one of {list(VALUE_STATE_MODES)}, "
                 f"got {state_mode!r}")
    key_from = raw.get("record_key_from", "") or ""
    if key_from:
        _require(key_from in RECORD_KEY_SOURCES,
                 f"{where}: record_key_from must be one of "
                 f"{list(RECORD_KEY_SOURCES)}, got {key_from!r}")
        _require(not raw.get("record_key"),
                 f"{where}: declare either a literal record_key or a "
                 f"record_key_from, not both — a value has one identity")
    return ProvenanceRef(source_ref=source_ref, source_label=source_label,
                         record_key=raw.get("record_key", "") or "",
                         record_key_from=key_from,
                         row_number=row_number, raw_value=raw_mode,
                         row_from=raw.get("row_from", "") or "",
                         value_state=state_mode,
                         contributions_from=raw.get("contributions_from", "") or "",
                         raw_from=raw.get("raw_from", "") or "")


def _parse_read(raw: Any, index: int) -> Read:
    where = f"reads[{index}]"
    _require(isinstance(raw, dict), f"{where}: must be a mapping")
    _check_keys(raw, _READ_KEYS, where)
    rid = _str(raw, "id", where)
    where = f"read {rid!r}"
    by_member = raw.get("column_by_member") or {}
    _require(isinstance(by_member, dict),
             f"{where}: column_by_member must map a member id to a column")
    _require(bool(raw.get("column")) or bool(by_member),
             f"{where}: a read names a column, or one column per group member")
    coerce = raw.get("coerce", "none") or "none"
    _require(coerce in COERCIONS,
             f"{where}: coerce must be one of {list(COERCIONS)}, got {coerce!r}")
    return Read(id=rid, source=_str(raw, "source", where),
                column=raw.get("column", "") or "",
                column_by_member={str(k): str(v) for k, v in by_member.items()},
                coerce=coerce, description=raw.get("description", "") or "")


def _parse_condition(raw: Any, index: int) -> Condition:
    where = f"conditions[{index}]"
    _require(isinstance(raw, dict), f"{where}: must be a mapping")
    _check_keys(raw, _CONDITION_KEYS, where)
    cid = _str(raw, "id", where)
    where = f"condition {cid!r}"
    read = raw.get("read", "") or ""
    group = raw.get("group", "") or ""
    origin = raw.get("record_origin", "") or ""
    declared = [bool(read), bool(group), bool(origin)]
    _require(sum(declared) == 1,
             f"{where}: a condition tests exactly one thing — a named read, a "
             f"group's membership, or how this record came to exist")
    if origin:
        _require(origin in RECORD_ORIGINS,
                 f"{where}: record_origin must be one of {list(RECORD_ORIGINS)}, "
                 f"got {origin!r}")
        return Condition(id=cid, record_origin=origin,
                         description=raw.get("description", "") or "")
    if group:
        return Condition(id=cid, group=group, member=_str(raw, "member", where),
                         description=raw.get("description", "") or "")
    pred = _str(raw, "predicate", where)
    value = raw.get("value")
    try:
        validate_predicate_definition(pred, value)
    except PredicateDefinitionError as exc:
        raise SpecError(f"{where}: {exc}") from exc
    return Condition(id=cid, read=read, predicate=pred, value=value,
                     description=raw.get("description", "") or "")


#: References that only a grouped anchor can answer.
_ANCHOR_GROUP_REFS = frozenset({REF_ANCHOR_GROUP_SIZE, REF_ANCHOR_SELECTED_ROW})


def _refs_used(value: Any) -> set[str]:
    """Every runtime fact a declared record names, at any depth."""
    if isinstance(value, dict):
        if set(value) == {REF_MARKER} and isinstance(value[REF_MARKER], str):
            return {value[REF_MARKER]}
        return set().union(*(_refs_used(v) for v in value.values())) \
            if value else set()
    if isinstance(value, list):
        return set().union(*(_refs_used(v) for v in value)) if value else set()
    return set()


def _check_derivation_refs(value: Any, where: str) -> None:
    """Every reference must name a fact from the closed set. Nothing else."""
    if isinstance(value, dict):
        if set(value) == {REF_MARKER}:
            ref = value[REF_MARKER]
            _require(isinstance(ref, str) and ref,
                     f"{where}: {REF_MARKER} must name a runtime fact")
            head, _, rest = ref.partition(".")
            if head in REF_SIMPLE:
                _require(not rest, f"{where}: {ref!r} takes no qualifier")
                return
            _require(head in REF_PREFIXES,
                     f"{where}: {ref!r} is not a known runtime fact; expected one "
                     f"of {list(REF_SIMPLE)} or a "
                     f"{list(REF_PREFIXES)} reference")
            _require(bool(rest), f"{where}: {ref!r} must name what it refers to")
            return
        _require(REF_MARKER not in value,
                 f"{where}: {REF_MARKER} cannot be mixed with other keys — a "
                 f"reference stands alone")
        for k, v in value.items():
            _check_derivation_refs(v, f"{where}.{k}")
        return
    if isinstance(value, list):
        for i, v in enumerate(value):
            _check_derivation_refs(v, f"{where}[{i}]")


def _parse_derivation(raw: Any, where: str) -> tuple[DerivationVariant, ...]:
    if raw is None:
        return ()
    where = f"{where}.derivation"
    if isinstance(raw, dict):
        # A bare mapping is the record itself — the one-variant shorthand. But a
        # mapping that already carries `record` or `when` is someone writing a
        # variant without the list, and wrapping it would silently produce a
        # record whose only key is "record". Refuse instead of mis-reading it.
        _require(not ({"record", "when"} & set(raw)),
                 f"{where}: a single derivation variant is written as a "
                 f"one-item list; a bare mapping here is the record itself and "
                 f"cannot also carry {sorted({'record', 'when'} & set(raw))}")
    entries = raw if isinstance(raw, list) else [{"record": raw}]
    out: list[DerivationVariant] = []
    for i, entry in enumerate(entries):
        ew = f"{where}[{i}]"
        _require(isinstance(entry, dict), f"{ew}: must be a mapping")
        _check_keys(entry, _DERIVATION_KEYS, ew)
        record = entry.get("record")
        _require(isinstance(record, dict) and record,
                 f"{ew}: record must be a non-empty mapping")
        _check_derivation_refs(record, ew)
        out.append(DerivationVariant(record=dict(record),
                                     when=entry.get("when", "") or ""))
    _require(bool(out), f"{where}: declares no record")
    for v in out[:-1]:
        _require(bool(v.when),
                 f"{where}: only the last variant may be unconditional; an "
                 f"earlier one without a condition would shadow the rest")
    return tuple(out)


def _parse_mapping(raw: Any, index: int) -> Mapping:
    where = f"mappings[{index}]"
    _require(isinstance(raw, dict), f"{where}: must be a mapping")
    _check_keys(raw, _MAPPING_KEYS, where)
    target = _str(raw, "target", where)
    mapping_id = raw.get("mapping_id", "") or ""
    _require(not mapping_id or bool(_MAPPING_ID_RE.fullmatch(mapping_id)),
             f"{where}: mapping_id must match {_MAPPING_ID_RE.pattern!r}")
    where = f"mapping for {target!r}"

    op = _str(raw, "op", where)
    _require(op in OPERATIONS,
             f"{where}: op must be one of {list(OPERATIONS)}, got {op!r}")
    _require("rule" in raw, f"{where}: a rule reference is required")
    rule = _parse_rule(raw["rule"], where)

    value_state = raw.get("value_state", "") or ""
    if value_state:
        _require(value_state in VALUE_STATES,
                 f"{where}: value_state must be one of {list(VALUE_STATES)}, "
                 f"got {value_state!r}")

    candidates: list[Candidate] = []
    clauses: list[Clause] = []
    members: list[Member] = []

    # Per-operation required parameters. Each branch states what the operation
    # cannot execute without, so an under-specified mapping fails at load rather
    # than at run time.
    if op in _COLUMN_OPS:
        _str(raw, "source", where)
        _str(raw, "column", where)
    if op == OP_CONSTANT:
        _require("value" in raw, f"{where}: constant requires a value (may be null)")
    if op == OP_FALLBACK and raw.get("map") is not None:
        _require(isinstance(raw["map"], dict) and raw["map"],
                 f"{where}: a fallback's map must be non-empty when declared")
    if op == OP_VALUE_MAP:
        vmap = raw.get("map")
        # A value map must have SOMEWHERE TO LOOK — its own table, a shared
        # vocabulary, or both. An empty table is legitimate: a trial that
        # contributes no coding of its own says so by declaring none and naming
        # the cross-trial vocabulary as the authority, which is more honest than
        # transcribing that vocabulary into the trial or inventing an entry that
        # can never match. An empty table with nothing behind it is still
        # refused: that mapping could never resolve anything.
        _require(isinstance(vmap, dict)
                 and (vmap or raw.get("fallback_vocabulary")),
                 f"{where}: value_map requires a non-empty map, or a "
                 f"fallback_vocabulary to consult when it has none")
        on_unmapped = raw.get("on_unmapped", "null")
        _require(on_unmapped in ("null", "passthrough", "error"),
                 f"{where}: on_unmapped must be null, passthrough or error")
    if op == OP_UNIT_CONVERSION:
        factor = raw.get("factor")
        _require(isinstance(factor, (int, float)) and not isinstance(factor, bool),
                 f"{where}: unit_conversion requires a numeric factor")
        _require(float(factor) != 0, f"{where}: unit_conversion factor must be non-zero")
    if op == OP_FALLBACK:
        raw_c = raw.get("candidates")
        _require(isinstance(raw_c, list) and len(raw_c) >= 2,
                 f"{where}: fallback requires at least two ordered candidates — "
                 f"a chain of one is a direct mapping")
        for i, c in enumerate(raw_c):
            cw = f"{where}.candidates[{i}]"
            _require(isinstance(c, dict), f"{cw}: must be a mapping")
            _check_keys(c, _CANDIDATE_KEYS, cw)
            candidates.append(Candidate(
                source=_str(c, "source", cw), column=_str(c, "column", cw),
                when=_parse_predicate(c["when"], f"{cw}.when") if c.get("when") else None,
                label=c.get("label", "") or "",
                description=c.get("description", "") or ""))
        seen = [(c.source, c.column) for c in candidates]
        dupes = sorted({s for s in seen if seen.count(s) > 1})
        _require(not dupes,
                 f"{where}: fallback repeats candidate(s) {dupes}; a repeated "
                 f"candidate can never be reached a second time")
    if op == OP_CONDITIONAL:
        raw_cl = raw.get("clauses")
        _require(isinstance(raw_cl, list) and raw_cl,
                 f"{where}: conditional requires at least one clause")
        for i, c in enumerate(raw_cl):
            cw = f"{where}.clauses[{i}]"
            _require(isinstance(c, dict), f"{cw}: must be a mapping")
            _check_keys(c, _CLAUSE_KEYS, cw)
            _require("when" in c, f"{cw}: a when predicate is required")
            has_value = "value" in c
            has_column = bool(c.get("column"))
            has_read = bool(c.get("read"))
            _require(sum([has_value, has_column, has_read]) == 1,
                     f"{cw}: a clause yields exactly one of a literal value, a "
                     f"column, or a named read")
            if has_column:
                _str(c, "source", cw)
            clauses.append(Clause(
                when=_parse_predicate(c["when"], f"{cw}.when"),
                source=c.get("source", "") or "", column=c.get("column", "") or "",
                value=c.get("value"), read=c.get("read", "") or "",
                description=c.get("description", "") or ""))
    if op == OP_FILE_MEMBERSHIP and raw.get("group"):
        _require(not raw.get("members"),
                 f"{where}: file_membership declares either a group or an inline "
                 f"member list, not both")
    elif op == OP_FILE_MEMBERSHIP:
        raw_m = raw.get("members")
        _require(isinstance(raw_m, list) and len(raw_m) >= 2,
                 f"{where}: file_membership requires at least two candidate sources")
        for i, m in enumerate(raw_m):
            mw = f"{where}.members[{i}]"
            _require(isinstance(m, dict), f"{mw}: must be a mapping")
            _check_keys(m, _MEMBER_KEYS, mw)
            has_value = "value" in m
            has_column = bool(m.get("column"))
            _require(has_value or has_column,
                     f"{mw}: a member must yield either a literal value or a column")
            members.append(Member(source=_str(m, "source", mw),
                                  value=m.get("value"),
                                  column=m.get("column", "") or ""))
        srcs = [m.source for m in members]
        dupes = sorted({s for s in srcs if srcs.count(s) > 1})
        _require(not dupes, f"{where}: file_membership lists source(s) {dupes} twice")
    if op == OP_JOIN_LOOKUP:
        _str(raw, "join", where)
        _str(raw, "lookup_column", where)
    if op == OP_NOT_APPLICABLE:
        state = raw.get("value_state") or ""
        _require(state in ("not_applicable", "source_missing"),
                 f"{where}: not_applicable requires value_state not_applicable or "
                 f"source_missing, so the reason for the absence is explicit")
    # `from_rule` says the confidence is computed by the rule named in `rule.id`.
    # It is a claim about that rule, not about the operation, so it is legal
    # anywhere — but an executor that cannot resolve the rule must defer the
    # mapping rather than guess a number.
    confidence = _parse_confidence(raw, where)
    raw_conf = raw.get("confidence")
    confidence_no_value = None
    confidence_unmapped = None
    if isinstance(raw_conf, dict) and "no_value" in raw_conf:
        confidence_no_value = _confidence_number(raw_conf["no_value"], where,
                                                 "no_value")
    if isinstance(raw_conf, dict) and "unmapped" in raw_conf:
        confidence_unmapped = _confidence_number(raw_conf["unmapped"], where,
                                                 "unmapped")

    unmapped_rule = (_parse_rule(raw["unmapped_rule"], f"{where}.unmapped_rule")
                     if raw.get("unmapped_rule") else None)
    _require(unmapped_rule is None or confidence_unmapped is not None,
             f"{where}: unmapped_rule describes an outcome the mapping never "
             f"declares a confidence for; add confidence.unmapped")

    raw_contrib = raw.get("contributions") or {}
    _require(isinstance(raw_contrib, dict),
             f"{where}: contributions must be a mapping")
    _check_keys(raw_contrib, _CONTRIB_KEYS, f"{where}.contributions")
    _require(not raw_contrib or op == OP_FALLBACK
             or set(raw_contrib) <= {"conditions", "condition_rank"},
             f"{where}: contributions describe a fallback's candidate list; "
             f"{op!r} may only add condition contributions")
    pos = raw_contrib.get("condition_position", "last")
    _require(pos in CONDITION_POSITIONS,
             f"{where}.contributions: condition_position must be one of "
             f"{list(CONDITION_POSITIONS)}, got {pos!r}")
    after = raw_contrib.get("after_decision", "not_evaluated")
    _require(after in AFTER_DECISION_STATUSES,
             f"{where}.contributions: after_decision must be one of "
             f"{list(AFTER_DECISION_STATUSES)}, got {after!r}")
    record = raw_contrib.get("record", "all")
    _require(record in CONTRIBUTION_RECORDING,
             f"{where}.contributions: record must be one of "
             f"{list(CONTRIBUTION_RECORDING)}, got {record!r}")
    contrib_conditions = tuple(str(c) for c
                               in (raw_contrib.get("conditions") or []))
    condition_rank = raw_contrib.get("condition_rank", -1)
    _require(isinstance(condition_rank, int)
             and not isinstance(condition_rank, bool),
             f"{where}.contributions: condition_rank must be an integer")
    _require(condition_rank == -1 or bool(contrib_conditions),
             f"{where}.contributions: condition_rank ranks the condition "
             f"contributions, and this mapping declares none")
    contribution_style = ContributionStyle(
        condition_position=pos, after_decision=after, record=record,
        conditions=contrib_conditions, condition_rank=condition_rank,
        no_value_column=raw_contrib.get("no_value_column", "") or "")

    fallback_vocabulary = raw.get("fallback_vocabulary", "") or ""
    _require(not fallback_vocabulary or op == OP_VALUE_MAP,
             f"{where}: fallback_vocabulary is consulted when a value map has no "
             f"entry; {op!r} has no map")
    _require(not confidence_unmapped or op == OP_VALUE_MAP,
             f"{where}: an unmapped outcome only arises from a value map; "
             f"{op!r} cannot produce one")

    no_value_rule = (_parse_rule(raw["no_value_rule"], f"{where}.no_value_rule")
                     if raw.get("no_value_rule") else None)
    _require(no_value_rule is None or confidence_no_value is not None,
             f"{where}: no_value_rule describes an outcome the mapping never "
             f"declares a confidence for; add confidence.no_value")

    coerce = raw.get("coerce", "none") or "none"
    _require(coerce in COERCIONS,
             f"{where}: coerce must be one of {list(COERCIONS)}, got {coerce!r}")
    # A fallback picks WHICH value; how that value is then read is still a
    # coercion, so the two compose. Every other operation states its own
    # transformation and takes none.
    _require(coerce == "none" or op in (OP_DIRECT, OP_FALLBACK),
             f"{where}: coerce applies to a direct or fallback mapping; {op!r} "
             f"states its own transformation")

    rounding = raw.get("round", False)
    _require(isinstance(rounding, bool),
             f"{where}: round must be true or false, got {rounding!r}")
    _require(not rounding or op == OP_UNIT_CONVERSION,
             f"{where}: round applies to unit_conversion only")

    provenance = _parse_provenance(raw.get("provenance"), where)

    by_member = raw.get("column_by_member") or {}
    _require(isinstance(by_member, dict),
             f"{where}: column_by_member must map a member id to a column name")
    _require(not by_member or op in _COLUMN_OPS + (OP_CONDITIONAL, OP_CONSTANT),
             f"{where}: column_by_member names the column this value is recorded "
             f"against; {op!r} records none")

    params = raw.get("params") or {}
    _require(isinstance(params, dict), f"{where}: params must be a mapping")
    _require(not params or op == OP_CUSTOM_RULE,
             f"{where}: params are handed to a registered rule; {op!r} takes none")

    if op == OP_CUSTOM_RULE:
        inputs = raw.get("inputs")
        _require(isinstance(inputs, list) and inputs,
                 f"{where}: custom_rule requires a non-empty inputs list — the "
                 f"escape hatch still declares its input contract")
        _str(raw, "output_type", where)

    return Mapping(
        target=target, op=op, rule=rule, mapping_id=mapping_id,
        source=raw.get("source", "") or "",
        column=raw.get("column", "") or "",
        value=raw.get("value"),
        map=dict(raw.get("map") or {}),
        on_unmapped=raw.get("on_unmapped", "null"),
        candidates=tuple(candidates), clauses=tuple(clauses),
        otherwise=raw.get("otherwise"),
        factor=(float(raw["factor"]) if raw.get("factor") is not None else None),
        members=tuple(members),
        join=raw.get("join", "") or "",
        group=raw.get("group", "") or "",
        column_by_member={str(k): str(v) for k, v in by_member.items()},
        lookup_column=raw.get("lookup_column", "") or "",
        implementation=raw.get("implementation", "") or "",
        inputs=tuple(str(i) for i in (raw.get("inputs") or [])),
        output_type=raw.get("output_type", "") or "",
        value_state=value_state,
        artifacts=tuple(str(a) for a in (raw.get("artifacts") or [])),
        description=raw.get("description", "") or "",
        confidence=confidence, confidence_no_value=confidence_no_value,
        no_value_rule=no_value_rule, confidence_unmapped=confidence_unmapped,
        unmapped_rule=unmapped_rule, fallback_vocabulary=fallback_vocabulary,
        contributions=contribution_style,
        when=raw.get("when", "") or "",
        derivation=_parse_derivation(raw.get("derivation"), where),
        coerce=coerce, round=rounding,
        provenance=provenance, params=dict(params),
    )


def _parse_source(raw: Any, index: int) -> Source:
    where = f"sources[{index}]"
    _require(isinstance(raw, dict), f"{where}: must be a mapping")
    # Checked before the generic unknown-key rule so the layer boundary gets its
    # own message: "unknown property" would be true but would not say why.
    for banned in ("sha256", "checksum", "n_rows", "resolved_path"):
        _require(banned not in raw,
                 f"{where}: {banned!r} is a runtime fact and must not appear in a "
                 f"static specification — it belongs to the SourceRef recorded at "
                 f"execution time")
    _check_keys(raw, _SOURCE_KEYS, where)
    sid = _str(raw, "id", where)
    where = f"source {sid!r}"
    ftype = _str(raw, "file_type", where)
    _require(ftype in FILE_TYPES,
             f"{where}: file_type must be one of {list(FILE_TYPES)}, got {ftype!r}")
    path, pattern = raw.get("path", "") or "", raw.get("path_pattern", "") or ""
    _require(bool(path) != bool(pattern),
             f"{where}: declare exactly one of path or path_pattern")
    keys = raw.get("record_keys")
    _require(isinstance(keys, list) and keys,
             f"{where}: record_keys is required and must be non-empty")
    # Optional, and deliberately NOT derived from record_keys: that list is row
    # identity, it is composite for grouped files, and its order means nothing.
    # A source that joins on a differently spelled column says so outright.
    part_col = raw.get("participant_column", "")
    _require(isinstance(part_col, str),
             f"{where}: participant_column must be a single column name")
    _require("participant_column" not in raw or part_col.strip(),
             f"{where}: participant_column is declared but empty — omit it to "
             f"use the record's participant_key")
    _require(sid != ANCHOR_SOURCE_ID,
             f"{where}: {ANCHOR_SOURCE_ID!r} is reserved for the external row anchor")
    reader = raw.get("reader", "") or ""
    if reader:
        _require(reader in READERS,
                 f"{where}: reader must be one of {list(READERS)}, got {reader!r} — "
                 f"a specification names a vetted reader, never Python")
    if ftype in ("xlsx", "xls"):
        _require(bool(raw.get("sheet")),
                 f"{where}: a spreadsheet source must name its sheet")
    miss_raw = raw.get("on_missing_record") or {}
    _require(isinstance(miss_raw, dict),
             f"{where}: on_missing_record must be a mapping")
    _check_keys(miss_raw, _MISSING_KEYS, f"{where}.on_missing_record")
    mode = miss_raw.get("mode", "error")
    _require(mode in MISSING_RECORD_MODES,
             f"{where}.on_missing_record: mode must be one of "
             f"{list(MISSING_RECORD_MODES)}, got {mode!r}")
    emit_for = tuple(str(t) for t in (miss_raw.get("emit_for") or []))
    if mode == "lookup_miss":
        _require(bool(emit_for),
                 f"{where}.on_missing_record: lookup_miss must name the targets "
                 f"it emits a miss for; an unlisted target stays an error")
        _require("rule" in miss_raw,
                 f"{where}.on_missing_record: a lookup miss is an event and needs "
                 f"a rule reference")
    else:
        _require(not emit_for,
                 f"{where}.on_missing_record: emit_for applies to lookup_miss")
    miss_conf = miss_raw.get("confidence", 0.0)
    _require(isinstance(miss_conf, (int, float)) and not isinstance(miss_conf, bool)
             and 0.0 <= float(miss_conf) <= 1.0,
             f"{where}.on_missing_record: confidence must be a number in [0, 1]")
    missing = MissingRecord(
        mode=mode, emit_for=emit_for, confidence=float(miss_conf),
        rule=(_parse_rule(miss_raw["rule"], f"{where}.on_missing_record")
              if miss_raw.get("rule") else None),
        column=miss_raw.get("column", "") or "")

    return Source(id=sid, file_type=ftype, path=path, path_pattern=pattern,
                  on_missing_record=missing,
                  reader=reader,
                  logical_label=raw.get("logical_label", "") or "",
                  sheet=raw.get("sheet", "") or "",
                  record_keys=tuple(str(k) for k in keys),
                  participant_column=part_col.strip(),
                  role=raw.get("role", "") or "",
                  description=raw.get("description", "") or "")


def _parse_source_group(raw: Any, index: int, source_ids: set[str]) -> SourceGroup:
    where = f"source_groups[{index}]"
    _require(isinstance(raw, dict), f"{where}: must be a mapping")
    _check_keys(raw, _GROUP_KEYS, where)
    gid = _str(raw, "id", where)
    where = f"source group {gid!r}"
    _require(GROUP_REF_SEP not in gid,
             f"{where}: a group id may not contain {GROUP_REF_SEP!r}, which "
             f"separates the group from the role in a source reference")
    select = raw.get("select", "first_containing_record")
    _require(select in GROUP_SELECTIONS,
             f"{where}: select must be one of {list(GROUP_SELECTIONS)}, "
             f"got {select!r}")

    raw_members = raw.get("members")
    _require(isinstance(raw_members, list) and len(raw_members) >= 2,
             f"{where}: a group needs at least two members — a split of one is "
             f"just a source")
    members: list[GroupMember] = []
    for i, m in enumerate(raw_members):
        mw = f"{where}.members[{i}]"
        _require(isinstance(m, dict), f"{mw}: must be a mapping")
        _check_keys(m, _GROUP_MEMBER_KEYS, mw)
        mid = _str(m, "id", mw)
        srcs = m.get("sources")
        _require(isinstance(srcs, dict) and srcs,
                 f"{mw}: sources must map a role to a declared source id")
        for role, sid in srcs.items():
            _require(sid in source_ids,
                     f"{mw}: role {role!r} names source {sid!r}, which is not "
                     f"declared")
        members.append(GroupMember(id=mid, value=m.get("value"),
                                   sources={str(k): str(v) for k, v in srcs.items()},
                                   description=m.get("description", "") or ""))

    ids = [m.id for m in members]
    dupes = sorted({i for i in ids if ids.count(i) > 1})
    _require(not dupes, f"{where}: duplicate member id(s) {dupes}")
    roles = [tuple(sorted(m.sources)) for m in members]
    _require(len(set(roles)) == 1,
             f"{where}: every member must offer the same roles, so a mapping can "
             f"name a role without knowing which member wins; got {roles}")

    decided_by = _str(raw, "decided_by", where)
    _require(decided_by in roles[0],
             f"{where}: decided_by {decided_by!r} is not one of the declared "
             f"roles {list(roles[0])}")
    status = raw.get("membership_status", "selected")
    _require(status in MEMBERSHIP_STATUSES,
             f"{where}: membership_status must be one of "
             f"{list(MEMBERSHIP_STATUSES)}, got {status!r}")

    group_rule = (_parse_rule(raw["rule"], f"{where}.rule")
                  if raw.get("rule") else None)

    return SourceGroup(id=gid, members=tuple(members), select=select,
                       decided_by=decided_by, membership_status=status,
                       rule=group_rule,
                       evidence_ref=raw.get("evidence_ref", "") or "",
                       description=raw.get("description", "") or "")


def _parse_record(raw: Any, source_ids: set[str]) -> Record:
    if raw is None:
        return Record()
    _require(isinstance(raw, dict), "record: must be a mapping")
    _check_keys(raw, _RECORD_KEYS, "record")
    anchor = raw.get("anchor", "source")
    _require(anchor in ANCHOR_KINDS,
             f"record: anchor must be one of {list(ANCHOR_KINDS)}, got {anchor!r}")
    anchor_source = raw.get("anchor_source", "") or ""
    if anchor in ("source", "grouping"):
        _require(bool(anchor_source),
                 f"record: anchor {anchor!r} requires anchor_source")
        _require(anchor_source in source_ids,
                 f"record: anchor_source {anchor_source!r} is not a declared source")
    else:
        _require(not anchor_source,
                 "record: anchor_source is meaningless for an external anchor")

    additional = _parse_additional_records(
        raw.get("additional_records"), anchor, source_ids,
        raw.get("participant_key", "") or "",
        {raw.get(k, "") or ""
         for k in ("participant_key", "sample_key",
                   "collection_event_key")} - {""},
        raw.get("anchor_source", "") or "")

    grouping = _parse_grouping(raw.get("grouping"), anchor,
                               raw.get("participant_key", "") or "",
                               {raw.get(k, "") or ""
                                for k in ("participant_key", "sample_key",
                                          "collection_event_key")} - {""})

    dup_raw = raw.get("duplicates") or {}
    _require(isinstance(dup_raw, dict), "record.duplicates: must be a mapping")
    _check_keys(dup_raw, _DUP_KEYS, "record.duplicates")
    policy = dup_raw.get("policy", "error")
    _require(policy in DUPLICATE_POLICIES,
             f"record.duplicates: policy must be one of {list(DUPLICATE_POLICIES)}")
    by = tuple(str(c) for c in (dup_raw.get("by") or []))
    _require(policy == "error" or by,
             "record.duplicates: a non-error policy must say which columns it "
             "deduplicates by, or the resolution is not deterministic")

    joins: list[Join] = []
    raw_joins = raw.get("joins") or []
    _require(isinstance(raw_joins, list), "record.joins: must be a list")
    for i, j in enumerate(raw_joins):
        jw = f"record.joins[{i}]"
        _require(isinstance(j, dict), f"{jw}: must be a mapping")
        _check_keys(j, _JOIN_KEYS, jw)
        jid = _str(j, "id", jw)
        left, right = _str(j, "left", jw), _str(j, "right", jw)
        for label, sid in (("left", left), ("right", right)):
            _require(sid in source_ids,
                     f"{jw}: {label} {sid!r} is not a declared source")
        _require(left != right, f"{jw}: a join must relate two different sources")
        # Named `keys`, not `on`: YAML 1.1 parses a bare `on` key as the
        # boolean true, which silently turns the join key into garbage.
        on = j.get("keys")
        _require(isinstance(on, list) and on,
                 f"{jw}: keys is required and must be a non-empty join-key list")
        card = j.get("cardinality", "one_to_one")
        _require(card in CARDINALITIES,
                 f"{jw}: cardinality must be one of {list(CARDINALITIES)}")
        joins.append(Join(id=jid, left=left, right=right,
                          on=tuple(str(k) for k in on), cardinality=card,
                          description=j.get("description", "") or ""))
    jids = [j.id for j in joins]
    dupes = sorted({j for j in jids if jids.count(j) > 1})
    _require(not dupes, f"record.joins: duplicate join ids {dupes}")

    key_col = raw.get("record_key_column", "")
    _require(isinstance(key_col, str),
             "record.record_key_column: must be a single column name")
    _require("record_key_column" not in raw or key_col.strip(),
             "record.record_key_column is declared but empty — omit it to name "
             "the record's identity by its participant_key")

    return Record(anchor=anchor, anchor_source=anchor_source,
                  participant_key=raw.get("participant_key", "") or "",
                  record_key_column=key_col.strip(),
                  sample_key=raw.get("sample_key", "") or "",
                  collection_event_key=raw.get("collection_event_key", "") or "",
                  duplicates=Duplicates(policy=policy, by=by),
                  joins=tuple(joins),
                  grouping=grouping,
                  additional_records=additional,
                  description=raw.get("description", "") or "")


def _parse_additional_records(raw: Any, anchor: str, source_ids: set[str],
                              participant_key: str, field_names: set[str],
                              anchor_source: str) -> tuple[AdditionalRecord, ...]:
    """Parse `record.additional_records`. Declarations, never computation."""
    if raw is None:
        return ()
    _require(anchor == "grouping",
             f"record.additional_records: records are added to the ones the "
             f"engine constructed, and record.anchor is {anchor!r}")
    _require(isinstance(raw, list) and raw,
             "record.additional_records: must be a non-empty list")
    out: list[AdditionalRecord] = []
    for i, entry in enumerate(raw):
        where = f"record.additional_records[{i}]"
        _require(isinstance(entry, dict), f"{where}: must be a mapping")
        _check_keys(entry, _ADDITIONAL_RECORD_KEYS, where)
        rid = _str(entry, "id", where)
        where = f"record.additional_records {rid!r}"
        src = _str(entry, "participants_from", where)
        _require(src in source_ids,
                 f"{where}: participants_from {src!r} is not a declared source")
        _require(src != anchor_source,
                 f"{where}: participants_from is the anchor source, so every "
                 f"participant in it already has a record and none could ever "
                 f"be added")
        when = entry.get("when", "no_record_for_participant")
        _require(when in ADDITIONAL_RECORD_CONDITIONS,
                 f"{where}: when must be one of "
                 f"{list(ADDITIONAL_RECORD_CONDITIONS)}, got {when!r}")
        values = entry.get("values") or {}
        _require(isinstance(values, dict) and values,
                 f"{where}: values is required — an added record has to say "
                 f"what it is, not only that it exists")
        for name in values:
            _require(name in field_names,
                     f"{where}.values: {name!r} is not a declared record field "
                     f"{sorted(field_names)}")
            _require(name != participant_key,
                     f"{where}.values: the participant key comes from "
                     f"{src!r}; restating it here could contradict the source")
        _require("rule" in entry,
                 f"{where}: a rule reference is required — adding a record is a "
                 f"decision and must name the one that approved it")
        out.append(AdditionalRecord(
            id=rid, participants_from=src, when=when,
            values={str(k): v for k, v in values.items()},
            rule=_parse_rule(entry["rule"], f"{where}.rule"),
            description=entry.get("description", "") or ""))
    ids = [a.id for a in out]
    dupes = sorted({i for i in ids if ids.count(i) > 1})
    _require(not dupes, f"record.additional_records: duplicate id(s) {dupes}")
    return tuple(out)


def _parse_grouping(raw: Any, anchor: str, participant_key: str,
                    anchor_field_names: set[str]) -> Grouping | None:
    """Parse `record.grouping`, which only a grouped anchor may declare."""
    if anchor != "grouping":
        _require(raw is None,
                 f"record.grouping: only a grouping anchor groups source rows, "
                 f"and record.anchor is {anchor!r}")
        return None
    _require(isinstance(raw, dict) and raw,
             "record.grouping: a grouping anchor must say how source rows are "
             "grouped into records")
    _check_keys(raw, _GROUPING_KEYS, "record.grouping")

    key_columns = raw.get("key_columns")
    _require(isinstance(key_columns, list) and key_columns,
             "record.grouping: key_columns is required and must be a non-empty "
             "list of source columns")
    key_columns = tuple(str(c) for c in key_columns)
    dupes = sorted({c for c in key_columns if key_columns.count(c) > 1})
    _require(not dupes,
             f"record.grouping: key_columns repeats {dupes}; a column can only "
             f"contribute to the key once")

    select = raw.get("select", "first_row")
    _require(select in GROUPING_SELECTIONS,
             f"record.grouping: select must be one of "
             f"{list(GROUPING_SELECTIONS)}, got {select!r}")

    fields = raw.get("anchor_fields") or {}
    _require(isinstance(fields, dict) and fields,
             "record.grouping: anchor_fields is required — it says which "
             "declared record field each key column supplies")
    fields = {str(k): str(v) for k, v in fields.items()}
    for field_name, column in fields.items():
        _require(field_name in anchor_field_names,
                 f"record.grouping.anchor_fields: {field_name!r} is not a "
                 f"declared record field {sorted(anchor_field_names)}")
        # The anchor is built from the key and nothing else. A non-key column
        # can differ between the rows of a group, so taking one would silently
        # choose a value the specification never decided on.
        _require(column in key_columns,
                 f"record.grouping.anchor_fields: {field_name!r} reads column "
                 f"{column!r}, which is not one of the grouping key columns "
                 f"{list(key_columns)}")
    _require(participant_key in fields,
             f"record.grouping.anchor_fields: the participant key "
             f"{participant_key!r} must be one of the fields the group key "
             f"supplies, or a constructed record has no participant")

    return Grouping(key_columns=key_columns, select=select,
                    anchor_fields=fields,
                    description=raw.get("description", "") or "")


def _parse_post_rule(raw: Any, index: int) -> PostRule:
    where = f"post_rules[{index}]"
    _require(isinstance(raw, dict), f"{where}: must be a mapping")
    _check_keys(raw, _POST_RULE_KEYS, where)
    rid = _str(raw, "id", where)
    where = f"post rule {rid!r}"
    _require("rule" in raw, f"{where}: a rule reference is required")
    rule = _parse_rule(raw["rule"], where)

    arts = tuple(str(a) for a in (raw.get("artifacts") or []))
    _require(bool(arts),
             f"{where}: must name the artifacts it applies to — an adjustment "
             f"that does not say where it lands is not reviewable")
    dupes = sorted({a for a in arts if arts.count(a) > 1})
    _require(not dupes, f"{where}: repeats artifact(s) {dupes}")

    inputs = tuple(str(i) for i in (raw.get("inputs") or []))
    _require(bool(inputs),
             f"{where}: must declare the fields it reads from the assembled "
             f"table")
    params = raw.get("params") or {}
    _require(isinstance(params, dict), f"{where}: params must be a mapping")

    conf = raw.get("confidence", 1.0)
    _require(isinstance(conf, (int, float)) and not isinstance(conf, bool)
             and 0.0 <= float(conf) <= 1.0,
             f"{where}: confidence must be a number in [0, 1]")
    state = raw.get("value_state", "") or ""
    if state:
        _require(state in VALUE_STATES,
                 f"{where}: value_state must be one of {list(VALUE_STATES)}")

    prov = raw.get("provenance") or {}
    _require(isinstance(prov, dict), f"{where}.provenance: must be a mapping")
    _check_keys(prov, _POST_PROV_KEYS, f"{where}.provenance")

    canonical = raw.get("canonical", True)
    _require(isinstance(canonical, bool),
             f"{where}: canonical must be true or false, got {canonical!r}")

    return PostRule(id=rid, rule=rule, artifacts=arts, inputs=inputs,
                    canonical=canonical,
                    params=dict(params), confidence=float(conf),
                    value_state=state,
                    source_label=prov.get("source_label", "") or "",
                    stage=prov.get("stage", "") or "",
                    supersedes_rule=prov.get("supersedes_rule", "") or "",
                    description=raw.get("description", "") or "")


def _parse_exclusion(raw: Any, index: int, source_ids: set[str]) -> Exclusion:
    where = f"exclusions[{index}]"
    _require(isinstance(raw, dict), f"{where}: must be a mapping")
    _check_keys(raw, _EXCLUSION_KEYS, where)
    eid = _str(raw, "id", where)
    where = f"exclusion {eid!r}"
    scope = _str(raw, "scope", where)
    _require(scope in EXCLUSION_SCOPES,
             f"{where}: scope must be one of {list(EXCLUSION_SCOPES)}, got {scope!r}")
    reason = _str(raw, "reason", where)
    _require("rule" in raw, f"{where}: a rule reference is required")
    rule = _parse_rule(raw["rule"], where)

    match = raw.get("match")
    _require(isinstance(match, dict) and match, f"{where}: match is required")
    _check_keys(match, _MATCH_KEYS, f"{where}.match")
    _str(match, "field", f"{where}.match")
    pred = match.get("predicate", "in")
    _require(pred in PREDICATES,
             f"{where}.match: predicate must be one of {list(PREDICATES)}")
    if pred in _SET_PREDICATES:
        _require(isinstance(match.get("value"), list) and match.get("value"),
                 f"{where}.match: predicate {pred!r} needs a non-empty list value")
    elif pred in _VALUE_PREDICATES:
        _require(match.get("value") is not None,
                 f"{where}.match: predicate {pred!r} needs a value")

    src = raw.get("source", "") or ""
    if src:
        _require(src in source_ids, f"{where}: source {src!r} is not declared")
    prov = raw.get("provenance") or {}
    _require(isinstance(prov, dict), f"{where}.provenance: must be a mapping")
    _check_keys(prov, _EXCL_PROV_KEYS, f"{where}.provenance")

    canonical = raw.get("canonical", True)
    _require(isinstance(canonical, bool),
             f"{where}: canonical must be true or false, got {canonical!r}")

    return Exclusion(id=eid, scope=scope, reason=reason, rule=rule, source=src,
                     match=dict(match), canonical=canonical,
                     applies_to=tuple(str(a) for a in (raw.get("applies_to") or [])),
                     description=raw.get("description", "") or "",
                     source_label=prov.get("source_label", "") or "",
                     evidence_path=prov.get("evidence_path", "") or "")


def _resolve_vocabularies(mappings: tuple[Mapping, ...], vocabulary_dir
                          ) -> tuple[tuple[Mapping, ...], tuple[str, ...]]:
    """Replace `{$vocabulary: id@version}` parameters with the resolved table.

    Resolution happens at load, so a specification that references a vocabulary
    that does not exist fails while it is being read rather than mid-run, and so
    the loaded object carries the terms an executor will actually use.
    """
    from . import vocabularies as vocab_mod

    used: set[str] = set()
    out: list[Mapping] = []
    for m in mappings:
        if m.fallback_vocabulary:
            try:
                kwargs = {"directory": vocabulary_dir} if vocabulary_dir else {}
                vocab = vocab_mod.resolve(m.fallback_vocabulary, **kwargs)
            except vocab_mod.VocabularyError as exc:
                raise SpecError(
                    f"mapping for {m.target!r}: fallback_vocabulary: {exc}") from exc
            _require(vocab.kind == vocab_mod.KIND_ORDERED_RULES,
                     f"mapping for {m.target!r}: fallback_vocabulary {vocab.ref} "
                     f"is a {vocab.kind!r} vocabulary; a fallback consulted in "
                     f"order needs ordered_rules")
            used.add(vocab.ref)
            m = dc_replace(
                m, fallback_rules=vocab.terms,
                fallback_vocabulary_object=vocab)
        if not m.params:
            out.append(m)
            continue
        resolved = {}
        parameter_vocabularies = []
        changed = False
        for key, val in m.params.items():
            if isinstance(val, dict) and set(val) == {VOCABULARY_MARKER}:
                ref = val[VOCABULARY_MARKER]
                try:
                    kwargs = ({"directory": vocabulary_dir} if vocabulary_dir
                              else {})
                    vocab = vocab_mod.resolve(ref, **kwargs)
                except vocab_mod.VocabularyError as exc:
                    raise SpecError(
                        f"mapping for {m.target!r}: parameter {key!r}: "
                        f"{exc}") from exc
                resolved[key] = vocab.terms
                parameter_vocabularies.append((key, vocab))
                used.add(vocab.ref)
                changed = True
            else:
                _require(VOCABULARY_MARKER not in str(val)
                         or not isinstance(val, dict),
                         f"mapping for {m.target!r}: parameter {key!r} mixes a "
                         f"{VOCABULARY_MARKER} reference with other keys")
                resolved[key] = val
        out.append(dc_replace(
            m, params=resolved,
            parameter_vocabularies=tuple(parameter_vocabularies)
        ) if changed else m)
    return tuple(out), tuple(sorted(used))


def parse(doc: Any, *, target_schema=None, sha256: str = "",
          source_path: str = "", require_executable: bool = False,
          vocabulary_dir=None,
          resolve_vocabularies: bool = True) -> HarmonizationSpec:
    """Build a `HarmonizationSpec`, validating it completely.

    When `target_schema` is supplied, every mapping target must be a field that
    schema declares. That cross-check is the link between the two layers and the
    reason a specification cannot quietly invent an output column.
    """
    _require(isinstance(doc, dict), "specification document must be a mapping")
    _check_keys(doc, _DOC_KEYS, "spec")

    spec_id = _str(doc, "spec_id", "spec")
    version = doc.get("spec_version")
    _require(isinstance(version, str) and _VERSION_RE.match(version or ""),
             f"spec: spec_version must be MAJOR.MINOR.PATCH, got {version!r}")
    trial = _str(doc, "trial", "spec")

    ts = doc.get("target_schema")
    _require(isinstance(ts, dict), "spec: target_schema is required")
    _check_keys(ts, _TARGET_SCHEMA_KEYS, "spec.target_schema")
    ts_id = _str(ts, "id", "spec.target_schema")
    ts_version = _str(ts, "version", "spec.target_schema")
    _require(bool(_VERSION_RE.match(ts_version)),
             f"spec.target_schema: version must be MAJOR.MINOR.PATCH, got {ts_version!r}")

    status = doc.get("status", "draft")
    _require(status in STATUSES,
             f"spec: status must be one of {list(STATUSES)}, got {status!r}")

    raw_sources = doc.get("sources")
    _require(isinstance(raw_sources, list) and raw_sources,
             "spec: sources must be a non-empty list")
    sources = tuple(_parse_source(s, i) for i, s in enumerate(raw_sources))
    sids = [s.id for s in sources]
    dupes = sorted({s for s in sids if sids.count(s) > 1})
    _require(not dupes, f"spec: duplicate source ids {dupes}")
    source_ids = set(sids)

    raw_groups = doc.get("source_groups") or []
    _require(isinstance(raw_groups, list), "spec: source_groups must be a list")
    source_groups = tuple(_parse_source_group(g, i, source_ids)
                          for i, g in enumerate(raw_groups))
    gids = [g.id for g in source_groups]
    dupes = sorted({g for g in gids if gids.count(g) > 1})
    _require(not dupes, f"spec: duplicate source group ids {dupes}")
    group_refs = {f"{g.id}{GROUP_REF_SEP}{r}" for g in source_groups
                  for r in g.roles}

    raw_reads = doc.get("reads") or []
    _require(isinstance(raw_reads, list), "spec: reads must be a list")
    reads = tuple(_parse_read(r, i) for i, r in enumerate(raw_reads))
    read_ids = [r.id for r in reads]
    dupes = sorted({r for r in read_ids if read_ids.count(r) > 1})
    _require(not dupes, f"spec: duplicate read ids {dupes}")

    raw_conditions = doc.get("conditions") or []
    _require(isinstance(raw_conditions, list), "spec: conditions must be a list")
    conditions = tuple(_parse_condition(c, i)
                       for i, c in enumerate(raw_conditions))
    cond_ids = [c.id for c in conditions]
    dupes = sorted({c for c in cond_ids if cond_ids.count(c) > 1})
    _require(not dupes, f"spec: duplicate condition ids {dupes}")

    record = _parse_record(doc.get("record"), source_ids)

    raw_mappings = doc.get("mappings")
    _require(isinstance(raw_mappings, list) and raw_mappings,
             "spec: mappings must be a non-empty list")
    mappings = tuple(_parse_mapping(m, i) for i, m in enumerate(raw_mappings))
    mapping_ids = [m.mapping_id for m in mappings if m.mapping_id]
    duplicate_mapping_ids = sorted({mid for mid in mapping_ids
                                    if mapping_ids.count(mid) > 1})
    _require(not duplicate_mapping_ids,
             f"spec: duplicate mapping_id values {duplicate_mapping_ids}")
    if resolve_vocabularies:
        mappings, vocabularies_used = _resolve_vocabularies(
            mappings, vocabulary_dir)
    else:
        # Replay must be able to refuse an external-vocabulary mapping from the
        # pinned specification itself without discovering or opening a
        # vocabulary directory.  Retain declared references, but deliberately
        # leave their rules/parameter tables unresolved.
        vocabulary_refs = {
            mapping.fallback_vocabulary for mapping in mappings
            if mapping.fallback_vocabulary
        }
        vocabulary_refs.update(
            value[VOCABULARY_MARKER]
            for mapping in mappings
            for value in mapping.params.values()
            if isinstance(value, dict)
            and set(value) == {VOCABULARY_MARKER}
            and isinstance(value[VOCABULARY_MARKER], str)
        )
        vocabularies_used = tuple(sorted(vocabulary_refs))

    for r in reads:
        where = f"read {r.id!r}"
        sid = r.source
        if GROUP_REF_SEP in sid:
            _require(sid in group_refs,
                     f"{where}: source {sid!r} is not a role of any declared "
                     f"source group")
        else:
            _require(sid in source_ids,
                     f"{where}: source {sid!r} is not a declared source")
    for c in conditions:
        where = f"condition {c.id!r}"
        if c.record_origin:
            _require(record.anchor == "grouping",
                     f"{where}: record_origin distinguishes constructed records "
                     f"from added ones, and record.anchor is {record.anchor!r}")
            _require(c.record_origin != "synthesized"
                     or record.additional_records,
                     f"{where}: tests for a synthesized record, but this "
                     f"specification declares no additional_records")
        elif c.group:
            group = next((g for g in source_groups if g.id == c.group), None)
            _require(group is not None,
                     f"{where}: group {c.group!r} is not declared")
            _require(c.member in {m.id for m in group.members},
                     f"{where}: {c.member!r} is not a member of group {c.group!r}")
        else:
            _require(c.read in set(read_ids),
                     f"{where}: read {c.read!r} is not declared")

    known_refs = ({f"read.{r}" for r in read_ids}
                  | {f"condition.{c}" for c in cond_ids}
                  | {f"{p}.{g.id}" for g in source_groups
                     for p in ("group_member", "group_value", "group_label")})

    def _check_known(value, where):
        if isinstance(value, dict):
            if set(value) == {REF_MARKER}:
                ref = value[REF_MARKER]
                _require(ref in REF_SIMPLE or ref in known_refs,
                         f"{where}: {ref!r} names nothing this specification "
                         f"declares")
                return
            for k, v in value.items():
                _check_known(v, f"{where}.{k}")
        elif isinstance(value, list):
            for i, v in enumerate(value):
                _check_known(v, f"{where}[{i}]")

    join_ids = {j.id for j in record.joins}
    for m in mappings:
        where = f"mapping for {m.target!r}"
        referenced = {m.source} if m.source else set()
        referenced |= {c.source for c in m.candidates}
        referenced |= {c.source for c in m.clauses if c.source}
        referenced |= {mem.source for mem in m.members}
        referenced |= {p.source for c in m.candidates if (p := c.when)}
        referenced |= {c.when.source for c in m.clauses}
        for sid in sorted(x for x in referenced if x):
            if sid == ANCHOR_SOURCE_ID:
                # A constructed anchor is still an anchor: its fields are the
                # group key, and a mapping may read them the same way.
                _require(record.anchor in ("external", "grouping"),
                         f"{where}: {ANCHOR_SOURCE_ID!r} reads the row anchor, "
                         f"but record.anchor is {record.anchor!r}")
                continue
            if GROUP_REF_SEP in sid:
                _require(sid in group_refs,
                         f"{where}: source {sid!r} is not a role of any declared "
                         f"source group (declared: {sorted(group_refs)})")
                continue
            _require(sid in source_ids,
                     f"{where}: source {sid!r} is not a declared source")
        if m.group:
            _require(m.group in gids,
                     f"{where}: group {m.group!r} is not a declared source group")
        _require(not m.when or m.when in set(cond_ids),
                 f"{where}: when {m.when!r} is not a declared condition")
        for vi, variant in enumerate(m.derivation):
            _require(not variant.when or variant.when in set(cond_ids),
                     f"{where}: derivation[{vi}] is gated on {variant.when!r}, "
                     f"which is not a declared condition")
            _check_known(variant.record, f"{where}.derivation[{vi}]")
            used = _refs_used(variant.record) & _ANCHOR_GROUP_REFS
            _require(not used or record.anchor == "grouping",
                     f"{where}.derivation[{vi}]: {sorted(used)} describe how a "
                     f"grouped record was built, and record.anchor is "
                     f"{record.anchor!r}")
        _require(m.provenance.record_key_from != "group"
                 or record.anchor == "grouping",
                 f"{where}: provenance.record_key_from is 'group', but "
                 f"record.anchor is {record.anchor!r} and there is no group key")
        _require(not m.provenance.raw_from
                 or m.provenance.raw_from in set(read_ids),
                 f"{where}: provenance.raw_from {m.provenance.raw_from!r} is not "
                 f"a declared read")
        for cname in m.contributions.conditions:
            _require(cname in set(read_ids),
                     f"{where}: contributions.conditions names read {cname!r}, "
                     f"which is not declared")
        for cl in m.clauses:
            if cl.read:
                _require(cl.read in set(read_ids),
                         f"{where}: clause yields read {cl.read!r}, which is not "
                         f"declared")
            if cl.when.read:
                _require(cl.when.read in set(read_ids),
                         f"{where}: clause reads {cl.when.read!r}, which is not "
                         f"declared")
        for cand in m.candidates:
            if cand.when and cand.when.read:
                _require(cand.when.read in set(read_ids),
                         f"{where}: candidate reads {cand.when.read!r}, which is "
                         f"not declared")

        cfrom = m.provenance.contributions_from
        _require(not cfrom or cfrom in gids,
                 f"{where}: provenance.contributions_from {cfrom!r} is not a "
                 f"declared source group")
        if m.column_by_member:
            gid = m.group or next(
                (ref.partition(GROUP_REF_SEP)[0] for ref in
                 (m.source, m.provenance.source_ref, m.provenance.row_from,
                  m.provenance.contributions_from)
                 if ref and GROUP_REF_SEP in ref),
                m.provenance.contributions_from)
            group = next((g for g in source_groups if g.id == gid), None)
            _require(group is not None,
                     f"{where}: column_by_member needs a source group; "
                     f"{m.source!r} names none")
            known = {mem.id for mem in group.members}
            unknown = sorted(set(m.column_by_member) - known)
            _require(not unknown,
                     f"{where}: column_by_member names unknown member(s) "
                     f"{unknown} of group {gid!r}")
            missing = sorted(known - set(m.column_by_member))
            _require(not missing,
                     f"{where}: column_by_member omits member(s) {missing}; a "
                     f"partial table would silently fall back to one column")
        for label, sid in (("source_ref", m.provenance.source_ref),
                           ("row_from", m.provenance.row_from)):
            if not sid:
                continue
            if GROUP_REF_SEP in sid:
                _require(sid in group_refs,
                         f"{where}: provenance.{label} {sid!r} is not a role of "
                         f"any declared source group")
                continue
            _require(sid in source_ids,
                     f"{where}: provenance.{label} {sid!r} is not a declared source")
        if m.op == OP_JOIN_LOOKUP:
            _require(m.join in join_ids,
                     f"{where}: join {m.join!r} is not declared in record.joins")

    allow_dupes = tuple(str(t) for t in (doc.get("allow_duplicate_targets") or []))
    targets = [m.target for m in mappings]
    conflicting = sorted({t for t in targets
                          if targets.count(t) > 1 and t not in allow_dupes
                          and not all(m.when for m in mappings if m.target == t)})
    _require(not conflicting,
             f"spec: multiple mappings for target(s) {conflicting} without listing "
             f"them in allow_duplicate_targets; which one wins would be ambiguous")

    if target_schema is not None:
        declared = {f.name for f in target_schema.fields}
        unknown = sorted({m.target for m in mappings} - declared)
        _require(not unknown,
                 f"spec: mapping target(s) {unknown} are not fields of target schema "
                 f"{target_schema.ref}")
        _require(target_schema.schema_id == ts_id,
                 f"spec: declares target schema {ts_id!r} but was validated against "
                 f"{target_schema.schema_id!r}")
        _require(target_schema.schema_version == ts_version,
                 f"spec: declares target schema version {ts_version!r} but was "
                 f"validated against {target_schema.schema_version!r}")

    if require_executable:
        # An "executable" specification is one an engine may be handed. Both
        # checks exist because the alternative is a silent Python default:
        # a reader guessed from the file extension, or a confidence keyed on a
        # rule ID. Either would move a decision out of the reviewed document.
        missing_reader = sorted(src.id for src in sources if not src.reader)
        _require(not missing_reader,
                 f"spec: source(s) {missing_reader} declare no reader; an "
                 f"executable specification must name one of {list(READERS)}")
        with_impl = sorted({m.target for m in mappings if m.implementation})
        _require(not with_impl,
                 f"spec: mapping(s) {with_impl} name a Python implementation "
                 f"path; an executable specification identifies a rule by id and "
                 f"version and lets the runtime registry resolve it")
        missing_conf = sorted({m.target for m in mappings if m.confidence is None})
        _require(not missing_conf,
                 f"spec: mapping(s) {missing_conf} declare no confidence; an "
                 f"executable specification states the confidence of every value "
                 f"it produces, or names {CONFIDENCE_FROM_RULE!r} where a "
                 f"registered rule computes it")

    raw_exclusions = doc.get("exclusions") or []
    _require(isinstance(raw_exclusions, list), "spec: exclusions must be a list")
    exclusions = tuple(_parse_exclusion(e, i, source_ids)
                       for i, e in enumerate(raw_exclusions))
    eids = [e.id for e in exclusions]
    dupes = sorted({e for e in eids if eids.count(e) > 1})
    _require(not dupes, f"spec: duplicate exclusion ids {dupes}")

    raw_post = doc.get("post_rules") or []
    _require(isinstance(raw_post, list), "spec: post_rules must be a list")
    post_rules = tuple(_parse_post_rule(r, i) for i, r in enumerate(raw_post))
    pids = [r.id for r in post_rules]
    dupes = sorted({r for r in pids if pids.count(r) > 1})
    _require(not dupes, f"spec: duplicate post rule ids {dupes}")
    if target_schema is not None:
        declared = {f.name for f in target_schema.fields}
        for r in post_rules:
            unknown = sorted(set(r.inputs) - declared)
            _require(not unknown,
                     f"post rule {r.id!r}: input(s) {unknown} are not fields of "
                     f"target schema {target_schema.ref}")

    return HarmonizationSpec(
        spec_id=spec_id, spec_version=version, trial=trial,
        target_schema_id=ts_id, target_schema_version=ts_version,
        project=doc.get("project", "") or "",
        status=status,
        description=doc.get("description", "") or "",
        evidence=doc.get("evidence", "") or "",
        sources=sources, source_groups=source_groups, record=record,
        reads=reads, conditions=conditions,
        mappings=mappings, exclusions=exclusions, post_rules=post_rules,
        allow_duplicate_targets=allow_dupes,
        vocabularies=vocabularies_used,
        sha256=sha256, source_path=source_path,
    )


def load(path: str | Path, *, target_schema=None,
         require_executable: bool = False,
         vocabulary_dir=None,
         resolve_vocabularies: bool = True) -> HarmonizationSpec:
    """Read and validate a specification file. Raises `SpecError` on any problem.

    `require_executable` additionally demands everything an engine needs but a
    reader of the document does not: a declared reader per source and a declared
    confidence per mapping. Executors load with it set.
    """
    from .source_ref import sha256_file       # local import: keeps this standalone

    p = Path(path)
    if not p.is_file():
        raise SpecError(f"specification file not found: {p}")
    try:
        doc = yaml.safe_load(p.read_text())
    except yaml.YAMLError as exc:
        raise SpecError(f"specification file is not valid YAML: {exc}") from exc
    return parse(doc, target_schema=target_schema, sha256=sha256_file(p),
                 source_path=str(p), require_executable=require_executable,
                 vocabulary_dir=vocabulary_dir,
                 resolve_vocabularies=resolve_vocabularies)

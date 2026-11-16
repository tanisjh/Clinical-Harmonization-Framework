"""spec_executor.py — execute a harmonization specification.

Phase 2c, Pilot 1. This is the first code that *runs* a specification rather
than merely validating one. It is deliberately small: it supports the four
generic operations plus external-anchor sourcing, which is exactly what the
S1400I pilot needs, and it refuses anything else instead of guessing.

The executor composes SourceSession for observed sources and CellFactory for
Cell-v1 provenance construction. BaseExtractor retains those same components
for compatibility users; modern execution does not inherit from it.

Source resolution belongs to this executor: the approved specification is the
sole authority on which exact files are opened, with containment and no basename
fallback. ``migrated_trials.py`` is never consulted.

A ``custom_rule`` mapping is executed by resolving its ``(rule_id, version)``
through :mod:`lib.rule_registry`. The specification never names Python. Rules of
this kind compute their own confidence, which the specification states by
declaring ``confidence: from_rule``, and they read target fields the pipeline
has already produced — so mappings are topologically ordered by their declared
inputs rather than run in document order.

Multiple sources are supported. The specification declares them, declares how
they relate in ``record.joins``, and the executor validates that relationship
before reading: declared join keys must exist, and the declared cardinality must
actually hold. A record missing from a joined source is an error, not a silent
null — this executor does not emit the diagnostic cells the legacy extractors
produce for a lookup miss, so it must not pretend one did not happen.

``fallback`` evaluates an ordered candidate list and records every candidate as
a structured contribution: which one supplied the value, which were examined and
found empty, and which were never consulted because the chain short-circuited.
When every candidate is empty the mapping may declare a distinct no-value rule,
because an exhausted chain is not the same event as a successful one.

``file_membership`` covers the cohort-split pattern: a trial that puts each arm
in its own files records the arm nowhere except in *which file holds the
record*. A ``source_group`` declares the parallel file sets, one decision picks a
member per record, and everything else reads through that member — so
``arm.demographics`` means "the demographics file of whichever arm this record
turned out to be in". The evidence is presence, not a column value, and the
provenance says so rather than naming a column that does not exist.

``not_applicable`` states that a target field has no meaning for a trial. That is
a different fact from a value the source failed to supply, and the specification
records which one it is.

Not implemented, and refused rather than skipped: ``conditional``,
``join_lookup``, and exclusions.

Determinism: output depends only on the specification, the schema and the
source bytes. No wall clock, no filesystem scan, no randomness.
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator

import pandas as pd

from lib import rule_registry, vocabularies
from lib.anchor_evidence import (
    ANCHOR_RECORD_EXTERNAL,
    ANCHOR_RECORD_GROUPED,
    ANCHOR_RECORD_SYNTHESIZED,
    EXTERNAL_ANCHOR_SOURCE_ID,
    RuntimeExternalAnchor,
    RuntimeExternalAnchorEvidence,
    RuntimeGroupingAnchorEvidence,
    RuntimeSourceRefs,
    is_external_anchor_evidence_mapping,
    is_grouping_anchor_direct_evidence_mapping,
)
from lib.conditions import (
    coerce_named_read,
    evaluate_predicate,
    is_missing_scalar,
    normalize_scalar,
)
from lib.confidence_policy import (
    ConfidencePolicyError,
    finalize_registered_rule_confidence,
)
from lib.fallback_policy import apply_inline_fallback_map
from lib.contributions import (
    STATUS_CONDITION,
    STATUS_INPUT,
    STATUS_NOT_EVALUATED,
    STATUS_REJECTED,
    STATUS_SELECTED,
    STATUS_UNAVAILABLE,
    Contribution,
    from_source_ref,
)
from lib.contributions import to_json as contributions_json
from lib.derivation import (
    derivation_references,
    is_stage16_fact,
    render_structured_derivation,
    select_derivation,
    select_derivation_record,
)
from lib.group_selection import (
    group_rule_id,
    is_closed_source_group_mapping,
    is_stage17_source_group_mapping,
    membership_role,
    select_first_containing_record,
    source_group_id_for_mapping,
)
from lib.cell_factory import (
    RAW_VALUE_ABSENT as _RAW_VALUE_ABSENT,
    CellFactory,
    anchor_str,
    compatibility_identity,
)
from lib.source_session import SourceSession
from lib.xlsx_reader import TrustedXlsxObservation
from lib.provenance import Cell, canonical_row_identity, provenance_event_ref
from lib.raw_policy import (
    COMPLETION_FALLBACK_EXHAUSTED,
    COMPLETION_VALUE,
    VerifiedOperationRouteResult,
    VerifiedRawCandidate,
    finalize_raw_value,
    raw_from_is_causal,
)
from lib.record_construction import (
    RecordConstructionError,
    construct_grouping_records,
    mapping_uses_record_construction,
    prepare_participant_frame,
)
from lib.replay_contract import encode_typed_raw
from lib.source_ref import SourceRef, assert_within_roots
from lib.value_states import (
    STATIC_PRESENT_DECLARATIONS,
    finalize_declared_value_state,
)
from lib.spec_model import (
    ANCHOR_SOURCE_ID,
    GROUP_REF_SEP,
    RECORD_ORIGINS,
    CONFIDENCE_FROM_RULE,
    OP_CONSTANT,
    OP_CUSTOM_RULE,
    OP_DIRECT,
    OP_FALLBACK,
    OP_CONDITIONAL,
    OP_FILE_MEMBERSHIP,
    OP_NOT_APPLICABLE,
    OP_NO_MAPPING_CONFIGURED,
    OP_UNIT_CONVERSION,
    OP_VALUE_MAP,
    REF_ANCHOR_GROUP_SIZE,
    REF_ANCHOR_SELECTED_ROW,
    HarmonizationSpec,
    Mapping,
    RuleRef,
)

LOG = logging.getLogger("spec_executor")

#: Operations this executor can run. Everything else is deferred with a reason.
SUPPORTED_OPS = (OP_DIRECT, OP_CONSTANT, OP_VALUE_MAP, OP_UNIT_CONVERSION,
                 OP_CUSTOM_RULE, OP_FALLBACK, OP_FILE_MEMBERSHIP,
                 OP_NOT_APPLICABLE, OP_NO_MAPPING_CONFIGURED, OP_CONDITIONAL)

#: Join cardinalities the executor can verify and honour.
SUPPORTED_CARDINALITIES = ("one_to_one", "many_to_one")

#: Readers this executor knows how to drive, mapped to the loader that opens
#: them. Declaring a reader the executor cannot drive is an error, not a
#: fallback to a guess.
_READER_LOADERS = {
    "cidc_csv@v1": "csv",
    "excel@v1": "xlsx",
    "xlsx_openpyxl@v1": "xlsx",
}


class ExecutionError(RuntimeError):
    """The specification cannot be executed as written."""


@dataclass(frozen=True)
class GroupedRecord:
    """What the engine knows about a record it constructed from a group.

    Kept beside the anchor row rather than inside it, so a mapping reading
    ``$anchor`` sees the record's fields and nothing else — these are facts
    ABOUT the construction, and they reach a specification only through the
    closed derivation-reference vocabulary.
    """

    #: key column -> its value for this record, in declared order.
    key: dict[str, Any]
    #: Row number, in the source as read, of the row this record cites.
    selected_row: int
    #: How many source rows composed the group.
    size: int
    #: The cited row itself, so an ordinary mapping can read a column from it.
    source_row: Any


@dataclass
class Diagnostics:
    """What the executor did, so a caller never has to infer it."""

    spec_ref: str = ""
    trial: str = ""
    anchor_rows: int = 0
    cells: int = 0
    executed: tuple[str, ...] = ()
    #: target -> why it was not executed.
    deferred: dict[str, str] = field(default_factory=dict)
    #: participant keys present in the anchor but absent from the source.
    lookup_misses: tuple[str, ...] = ()
    #: the order mappings ran in, after topological sorting.
    order: tuple[str, ...] = ()
    #: target -> the registered rule that produced it.
    rules_used: dict[str, str] = field(default_factory=dict)
    #: joins validated before any value was read.
    joins_validated: tuple[str, ...] = ()
    #: shared vocabularies the specification referenced.
    vocabularies: tuple[str, ...] = ()
    #: source group -> how many records each member claimed.
    group_membership: dict[str, dict[str, int]] = field(default_factory=dict)
    #: additional-record declaration -> how many records it added.
    additional_records: dict[str, int] = field(default_factory=dict)
    #: exclusions the specification declares and this stage did NOT apply.
    #: An exclusion drops a record from a named artifact, which is a decision
    #: about an assembled table, not about a value — so the field level records
    #: that it saw them and left them alone rather than silently ignoring them.
    exclusions_deferred: tuple[str, ...] = ()
    #: source id -> the SourceRef actually opened.
    sources: dict[str, SourceRef] = field(default_factory=dict)

    def summary(self) -> str:
        return (f"{self.spec_ref}: {self.cells} cells over {self.anchor_rows} anchor "
                f"rows; {len(self.executed)} mappings executed, "
                f"{len(self.deferred)} deferred")


def normalize_key(value: Any) -> str | None:
    """Compatibility name for the shared scalar normalization contract."""
    return normalize_scalar(value)


def _is_missing(value: Any) -> bool:
    return is_missing_scalar(value)


class SpecExecutor:
    """Execute a HarmonizationSpec against its declared sources."""

    def __init__(self, spec: HarmonizationSpec, project_root: Path,
                 anchor_df: pd.DataFrame | RuntimeExternalAnchor | None = None, *, contract=None,
                 targets: tuple[str, ...] | None = None, run_id: str = ""):
        """
        Args:
            spec:         a specification loaded with ``require_executable=True``.
            project_root: root that declared source paths resolve under.
            anchor_df:    the external row anchor for this trial. Omitted when
                          the specification declares a grouping anchor, because
                          then the records are a product of the specification
                          and there is nothing for a caller to supply.
            contract:     optional TargetContract; when given, every executed
                          target must be one of its columns.
            targets:      optional allow-list. Mappings outside it are deferred
                          with the reason ``out_of_pilot_scope``. This is how a
                          pilot runs part of a specification without pretending
                          the rest does not exist.
        """
        self.spec = spec
        self.contract = contract
        self._targets = tuple(targets) if targets is not None else None
        self._rules: dict[str, rule_registry.RegisteredRule] = {}
        # Legacy context attributes remain visible; modern resolution uses spec.
        # trial_dir_name also supplies source-session diagnostic context.
        self.trial_dir_name = spec.spec_id
        self.template_trial_name = spec.trial
        constructs_anchor = spec.record.anchor == "grouping"
        self._external_anchor: RuntimeExternalAnchor | None = None
        if isinstance(anchor_df, RuntimeExternalAnchor):
            if spec.record.anchor != "external":
                raise ExecutionError(
                    f"{spec.ref}: trusted external anchor supplied for "
                    f"record.anchor={spec.record.anchor!r}")
            try:
                runtime_anchor = anchor_df
                anchor_df = runtime_anchor.frame_copy(spec_ref=spec.ref)
            except (TypeError, ValueError) as exc:
                raise ExecutionError(
                    f"{spec.ref}: invalid external anchor runtime identity: "
                    f"{exc}") from exc
            self._external_anchor = runtime_anchor
        if anchor_df is None and not constructs_anchor:
            raise ExecutionError(
                f"{spec.ref}: record.anchor is {spec.record.anchor!r}, which "
                f"needs a row anchor; none was given")
        if anchor_df is not None and constructs_anchor:
            raise ExecutionError(
                f"{spec.ref}: record.anchor is 'grouping', so the records are "
                f"constructed from {spec.record.anchor_source!r}; supplying an "
                f"anchor would silently override them")
        # A grouping source is observed at extract(), after session setup.
        self._source_session = SourceSession(
            project_root, resolve_source=lambda key: self.src_path(key),
            context_name=lambda: self.trial_dir_name)
        self.trial_dir = self.project_root / self.trial_dir_name
        self.anchor = (pd.DataFrame() if anchor_df is None else anchor_df).reset_index(drop=True)
        # Retain these legacy compatibility attributes; modern execution uses spec.
        self.cfg = {}
        self.value_norms = {}
        self.derived_rules = {}
        self.run_id = run_id
        self.spec_ref = spec.ref
        self._cell_factory = CellFactory()
        #: The group behind each record, in record order, or None where the
        #: record was added by declaration rather than found in the data.
        self._grouped: list[GroupedRecord | None] = []
        #: "" where a record was observed, else the declaration that added it.
        self._origins: list[str] = []
        #: Exact runtime grouping construction retained for replay publication.
        self._anchor_evidence: (
            RuntimeGroupingAnchorEvidence | RuntimeExternalAnchorEvidence | None
        ) = (self._external_anchor.evidence
             if self._external_anchor is not None else None)
        if self._external_anchor is not None:
            self._source_session._register_observed_ref(
                EXTERNAL_ANCHOR_SOURCE_ID, self._external_anchor.source_ref)
        self.diagnostics = Diagnostics(spec_ref=spec.ref, trial=spec.trial,
                                       anchor_rows=len(self.anchor))

    # ── shared components and compatibility facades ───────────────────────
    def load_csv(self, key: str, logical_label: str | None = None) -> pd.DataFrame:
        """Read a configured CSV and record a SourceRef for it.

        Args:
            key:           logical input key in cfg['sources'].
            logical_label: stable label for config lookups and the existing
                           provenance `source_file` column. Defaults to the
                           basename actually opened. Repointed trials pass the
                           legacy basename here so `column_map` keys and
                           historical provenance values stay unchanged.
        """
        return self._source_session.load_csv(key, logical_label)

    def load_xlsx(self, key: str, sheet: str | int | None = None,
                  logical_label: str | None = None, *,
                  reader_id: str = "excel@v1") -> pd.DataFrame:
        return self._source_session.load_xlsx(
            key, sheet, logical_label, reader_id=reader_id)

    def ref(self, key: str) -> SourceRef:
        """SourceRef for a previously loaded logical input."""
        return self._source_session.ref(key)

    def source_refs(self) -> dict[str, SourceRef]:
        """All SourceRefs created during this extraction (for run manifests)."""
        return self._source_session.source_refs()

    # Live compatibility views: assignment replaces the session's state too.
    @property
    def project_root(self) -> Path:
        return self._source_session.project_root

    @project_root.setter
    def project_root(self, value: Path) -> None:
        self._source_session.project_root = value

    @property
    def _cache(self) -> dict[str, pd.DataFrame]:
        return self._source_session._cache

    @_cache.setter
    def _cache(self, value: dict[str, pd.DataFrame]) -> None:
        self._source_session._cache = value

    @property
    def _refs(self) -> dict[str, SourceRef]:
        return self._source_session._refs

    @_refs.setter
    def _refs(self, value: dict[str, SourceRef]) -> None:
        self._source_session._refs = value

    @property
    def _xlsx_observations(self) -> dict[str, TrustedXlsxObservation]:
        return self._source_session._xlsx_observations

    @_xlsx_observations.setter
    def _xlsx_observations(self, value: dict[str, TrustedXlsxObservation]) -> None:
        self._source_session._xlsx_observations = value

    # ── Cell construction ────────────────────────────────────────────────────
    @staticmethod
    def _anchor_str(value) -> str:
        """Compatibility facade for the shared Cell-v1 normalizer."""
        return anchor_str(value)

    @property
    def _legacy_label_uses(self) -> int:
        return self._cell_factory.legacy_label_uses

    @_legacy_label_uses.setter
    def _legacy_label_uses(self, value: int) -> None:
        self._cell_factory.legacy_label_uses = value

    def cell(self, anchor_row, harmonized_field, value, confidence, source_file,
             source_column, source_row_idx, extraction_method, notes="", *,
             raw_value=_RAW_VALUE_ABSENT, source_row_number=None, source_record_key=None,
             rule_id="", rule_version="", contributions="",
             authoritative_for="", derivation="", value_state="",
             mapping_id="", evidence_refs="", raw_evidence_recorded=True):
        """Build a Cell.

        `source_file` accepts either:
          * a `SourceRef` (migrated path) — the structured provenance fields
            (resolved path, sha256, sheet) are filled from the file that was
            actually opened, and the legacy `source_file` column keeps carrying
            `ref.logical_label` so existing consumers are unaffected; or
          * a plain string (not yet migrated, or a non-file source such as
            "CONFIG:trial_constants" / "DERIVED:BOR") — structured fields stay
            empty and the legacy-label counter is incremented.
        """
        source = self._cell_factory._prepare_source(source_file)
        identity = compatibility_identity(
            self.template_trial_name, anchor_row, get_anchor_str=lambda: self._anchor_str)
        return self._cell_factory._build_prepared(
            identity, harmonized_field, value, confidence, source, source_column,
            source_row_idx, extraction_method, notes,
            run_id=lambda: self.run_id, spec_ref=lambda: self.spec_ref,
            raw_value=raw_value, source_row_number=source_row_number,
            source_record_key=source_record_key, rule_id=rule_id,
            rule_version=rule_version, contributions=contributions,
            authoritative_for=authoritative_for, derivation=derivation,
            value_state=value_state, mapping_id=mapping_id,
            evidence_refs=evidence_refs, raw_evidence_recorded=raw_evidence_recorded)

    # Legacy scientific conveniences are unused by the modern runtime. Keep
    # lazy compatibility calls to the existing implementations until P1F review.
    def get_norm_rules(self, harmonized_field: str) -> list | dict | None:
        from lib.extractor_base import BaseExtractor
        return BaseExtractor.get_norm_rules(self, harmonized_field)

    def normalize_value(self, harmonized_field: str, raw_value):
        from lib.extractor_base import BaseExtractor
        return BaseExtractor.normalize_value(self, harmonized_field, raw_value)

    def derive_BOR_binary(self, bor_value):
        from lib.extractor_base import BaseExtractor
        return BaseExtractor.derive_BOR_binary(self, bor_value)

    def derive_bor_bin(self, bor_binary, pfs_time=None):
        from lib.extractor_base import BaseExtractor
        return BaseExtractor.derive_bor_bin(self, bor_binary, pfs_time)

    def derive_pfs_bin(self, pfs_time, pfs_stat=None):
        from lib.extractor_base import BaseExtractor
        return BaseExtractor.derive_pfs_bin(self, pfs_time, pfs_stat)

    # ── source authority: the specification, not the config ─────────────────
    def src_path(self, key: str) -> Path:
        """Resolve a declared source id to a path under the project root.

        The specification is the only source authority. Path containment is
        enforced and there is no basename fallback.
        """
        src = self.spec.source(key)
        if src is None:
            raise ExecutionError(f"{self.spec.ref}: no source {key!r} is declared")
        if not src.path:
            raise ExecutionError(
                f"{self.spec.ref}: source {key!r} declares path_pattern, which "
                f"Pilot 1 does not resolve; declare an exact path")
        return assert_within_roots(self.project_root / src.path, self.project_root)

    def _load_source(self, sid: str) -> pd.DataFrame:
        src = self.spec.source(sid)
        if src is None:
            raise ExecutionError(f"{self.spec.ref}: no source {sid!r} is declared")
        kind = _READER_LOADERS.get(src.reader)
        if kind is None:
            raise ExecutionError(
                f"{self.spec.ref}: source {sid!r} declares reader {src.reader!r}, "
                f"which this executor cannot drive "
                f"(knows {sorted(_READER_LOADERS)})")
        label = src.logical_label or None
        if kind == "csv":
            return self.load_csv(sid, logical_label=label)
        return self.load_xlsx(
            sid, sheet=src.sheet or None, logical_label=label,
            reader_id=src.reader)

    # ── record construction ─────────────────────────────────────────────────
    def _build_grouped_anchor(
            self) -> tuple[pd.DataFrame, list[GroupedRecord | None], list[str]]:
        """Construct one logical record per group of anchor-source rows.

        Some trials have no anchor to be given. Their records exist only as a
        pattern in a source file: the file is finer-grained than the record,
        several of its rows share the record's key, and the thing being
        harmonized is that group rather than any one row of it.

        The engine therefore groups the declared key columns, keeps ONE row per
        group as the row the record cites, and counts the group. It combines
        nothing: every anchor field comes from the key, which is identical in
        every row of the group by construction, so no value is invented and no
        row is silently preferred for anything but citation.

        `first_row` means first in SOURCE READ ORDER. Groups need not be
        contiguous in the file, so this is the first occurrence, not the first
        of a block, and it is stable because the read order is.
        """
        rec = self.spec.record
        sid = rec.anchor_source
        frame = self._load_source(sid)
        participant_frames = {
            declaration.participants_from:
                self._load_source(declaration.participants_from)
            for declaration in rec.additional_records
        }
        try:
            constructed = construct_grouping_records(
                self.spec, frame, participant_frames)
        except RecordConstructionError as exc:
            raise ExecutionError(f"{self.spec.ref}: {exc}") from exc
        rows = [item.anchor_row for item in constructed.records]
        grouped: list[GroupedRecord | None] = [
            (GroupedRecord(
                key=dict(item.group_key),
                selected_row=item.selected_source_row,
                size=item.group_size,
                source_row=item.source_row)
             if item.origin == "observed" else None)
            for item in constructed.records
        ]
        origins = [item.declaration_id for item in constructed.records]
        for declaration_id, count in constructed.additional_counts:
            self.diagnostics.additional_records[declaration_id] = count
            LOG.info("%s: declaration %r added %d record(s)",
                     self.spec.ref, declaration_id, count)
        LOG.info("%s: constructed %d grouped/declared records from %d rows of %r",
                 self.spec.ref, len(rows), len(frame), sid)
        try:
            self._anchor_evidence = RuntimeGroupingAnchorEvidence.capture(
                self.spec, self.source_refs(), record_count=len(rows))
        except ValueError as exc:
            raise ExecutionError(
                f"{self.spec.ref}: cannot capture grouping anchor evidence: {exc}") from exc
        return pd.DataFrame(rows), grouped, origins

    def _anchor_input_contribution(
            self, mapping: Mapping, raw: Any, grouped: GroupedRecord | None,
            declared_by: str, record_key: str, group_key: str,
            filtered_position: int | None = None) -> str:
        """Bind one consumed anchor scalar to its exact runtime route."""
        evidence = self._anchor_evidence
        if evidence is None:
            raise ExecutionError(
                f"{self.spec.ref}: anchor scalar used without runtime evidence")
        if self.spec.record.anchor == "external":
            runtime = self._external_anchor
            if (runtime is None or filtered_position is None
                    or isinstance(filtered_position, bool)
                    or not isinstance(filtered_position, int)
                    or filtered_position < 0
                    or filtered_position >= len(runtime.original_source_rows)):
                raise ExecutionError(
                    f"{self.spec.ref}: external anchor selection has no exact "
                    "runtime row identity")
            ref = runtime.source_ref
            contribution = from_source_ref(
                ref, role="anchor_input", status=STATUS_INPUT,
                source_column=mapping.column or mapping.target, raw_value=raw,
                source_record_key=record_key,
                source_row_number=runtime.original_source_rows[filtered_position],
                evidence_refs=evidence.evidence_ref_id,
                anchor_record_kind=ANCHOR_RECORD_EXTERNAL,
                anchor_filtered_position=filtered_position)
        elif grouped is not None:
            ref = self.ref(self.spec.record.anchor_source)
            contribution = from_source_ref(
                ref, role="anchor_input", status=STATUS_INPUT,
                source_column=mapping.column or mapping.target, raw_value=raw,
                source_record_key=group_key,
                source_row_number=grouped.selected_row,
                evidence_refs=evidence.evidence_ref_id,
                anchor_record_kind=ANCHOR_RECORD_GROUPED)
        else:
            declarations = [item for item in self.spec.record.additional_records
                            if item.id == declared_by]
            if len(declarations) != 1:
                raise ExecutionError(
                    f"{self.spec.ref}: synthesized anchor record has no exact "
                    f"declaration {declared_by!r}")
            ref = self.ref(declarations[0].participants_from)
            contribution = from_source_ref(
                ref, role="anchor_input", status=STATUS_INPUT,
                source_column=mapping.column or mapping.target, raw_value=raw,
                source_record_key=record_key, source_row_number="",
                evidence_refs=evidence.evidence_ref_id,
                anchor_record_kind=ANCHOR_RECORD_SYNTHESIZED,
                anchor_record_origin=declared_by)
        return contributions_json((contribution,))

    def _grouped_record_key(self, grouped: GroupedRecord) -> str:
        """The identity of a grouped record: the whole key, in declared order."""
        return ";".join(f"{c}={v}" for c, v in grouped.key.items())

    def _join_column(self, sid: str, participant_key: str) -> str:
        """The column in this source that carries the logical participant id.

        The identity is the record's; only its spelling is the file's. A
        CIDC-prepared file already writes `cimac_part_id` and declares nothing;
        a native trial file says which of its own columns means the same thing.
        Deliberately not read from `record_keys`: that list is row identity, it
        is composite for grouped files, and giving its first element a second
        job would make list order an API nobody agreed to.
        """
        src = self.spec.source(sid)
        return (src.participant_column if src and src.participant_column
                else participant_key)

    def _indexed_source(self, sid: str, participant_key: str) -> pd.DataFrame:
        """One source frame, deduplicated and indexed as the record declares."""
        try:
            return prepare_participant_frame(
                self.spec, sid, self._load_source(sid))
        except RecordConstructionError as exc:
            raise ExecutionError(f"{self.spec.ref}: {exc}") from exc

    # ── mapping triage ──────────────────────────────────────────────────────
    def _defer_reason(self, m: Mapping) -> str | None:
        if self._targets is not None and m.target not in self._targets:
            return "out_of_scope"
        if m.op not in SUPPORTED_OPS:
            return f"operation_not_implemented:{m.op}"
        if m.confidence == CONFIDENCE_FROM_RULE and m.op != OP_CUSTOM_RULE:
            # Only a registered rule can supply a confidence, and only a
            # custom_rule mapping invokes one.
            return f"confidence_from_rule_on_{m.op}"
        return None

    def _check_executable(self, m: Mapping) -> None:
        if m.confidence is None:
            raise ExecutionError(
                f"{self.spec.ref}: mapping for {m.target!r} declares no "
                f"confidence; refusing to invent one")
        if self.contract is not None and m.target not in self.contract.columns:
            raise ExecutionError(
                f"{self.spec.ref}: target {m.target!r} is not a column of "
                f"{self.contract.schema_ref}")
        if m.op != OP_CUSTOM_RULE:
            return
        if m.confidence != CONFIDENCE_FROM_RULE:
            raise ExecutionError(
                f"{self.spec.ref}: mapping for {m.target!r} invokes rule "
                f"{m.rule.id!r}, which computes its own confidence; the "
                f"specification must declare confidence {CONFIDENCE_FROM_RULE!r} "
                f"rather than assert a number the rule will override")
        rule = rule_registry.resolve(m.rule.id, m.rule.version,
                                     kind=rule_registry.KIND_FIELD)
        rule_registry.check_contract(rule, m.inputs, m.params)
        self._rules[m.target] = rule
        self.diagnostics.rules_used[m.target] = rule.ref

    def _vocabulary_evidence_refs(self, mapping: Mapping) -> tuple[str, ...]:
        """IDs for the exact runtime vocabularies used by one mapping."""
        loaded = []
        if mapping.fallback_vocabulary:
            vocab = mapping.fallback_vocabulary_object
            if vocab is None or vocab.ref != mapping.fallback_vocabulary:
                raise ExecutionError(
                    f"{self.spec.ref}: mapping for {mapping.target!r} has no "
                    "exact runtime fallback-vocabulary identity")
            if vocab.kind != vocabularies.KIND_ORDERED_RULES:
                raise ExecutionError(
                    f"{self.spec.ref}: mapping for {mapping.target!r} loaded "
                    f"fallback vocabulary kind {vocab.kind!r}, expected "
                    f"{vocabularies.KIND_ORDERED_RULES!r}")
            loaded.append(vocab)
        for parameter, vocab in mapping.parameter_vocabularies:
            if not parameter or vocab.kind != vocabularies.KIND_BUCKETS:
                raise ExecutionError(
                    f"{self.spec.ref}: mapping for {mapping.target!r} has an "
                    "invalid runtime bucket-vocabulary identity")
            loaded.append(vocab)
        try:
            return tuple(sorted({
                vocab.evidence_ref_id(self.project_root) for vocab in loaded
            }))
        except vocabularies.VocabularyError as exc:
            raise ExecutionError(
                f"{self.spec.ref}: mapping for {mapping.target!r} vocabulary "
                f"identity is not project-contained: {exc}") from exc

    # ── execution order ─────────────────────────────────────────────────────
    def _ordered(self, runnable: list[Mapping]) -> list[Mapping]:
        """Order mappings so a rule's inputs exist before it runs.

        Dependencies are inferred from the declared `inputs` of a rule, which
        name target fields. Nothing is hard-coded and no `depends_on` key is
        needed: a rule that has said what it reads has already said what it
        waits for.
        """
        by_target = {m.target: m for m in runnable}
        deps: dict[str, set[str]] = {}
        for m in runnable:
            needed = set()
            for name in m.inputs:
                if name == m.target:
                    raise ExecutionError(
                        f"{self.spec.ref}: mapping for {m.target!r} lists itself "
                        f"as an input")
                if name in by_target:
                    needed.add(name)
                    continue
                deferred = self.diagnostics.deferred.get(name)
                raise ExecutionError(
                    f"{self.spec.ref}: mapping for {m.target!r} depends on "
                    f"{name!r}, which "
                    + (f"is not executed ({deferred})" if deferred
                       else "no mapping in this specification produces"))
            deps[m.target] = needed

        # Kahn's algorithm, breaking ties by document order so the result is
        # deterministic rather than merely correct.
        order: list[Mapping] = []
        done: set[str] = set()
        remaining = list(runnable)
        while remaining:
            ready = [m for m in remaining if deps[m.target] <= done]
            if not ready:
                stuck = sorted(m.target for m in remaining)
                raise ExecutionError(
                    f"{self.spec.ref}: cyclic dependency among mapping(s) {stuck}; "
                    f"no execution order exists")
            for m in ready:
                order.append(m)
                done.add(m.target)
            remaining = [m for m in remaining if m.target not in done]
        return order

    # ── operations ──────────────────────────────────────────────────────────
    @staticmethod
    def _is_group_ref(sid: str) -> bool:
        return bool(sid) and GROUP_REF_SEP in sid

    def _source_value(self, m: Mapping, column: str, src_row) -> Any:
        """One value from a source row, refusing a column the file lacks.

        `Series.get` answers `None` for a key the row does not hold, which is
        indistinguishable from a column that exists and is empty. A misspelled
        column therefore used to produce a well-formed cell carrying no value,
        at whatever confidence the mapping declared for absence — a silent
        typo wearing the costume of missing data.

        The anchor branch above has always refused an unknown column by name.
        This is the same refusal for a file, and it says which mapping asked.
        """
        if column not in src_row.index:
            raise ExecutionError(
                f"{self.spec.ref}: mapping for {m.target!r} reads column "
                f"{column!r} from source {m.source!r}, which has no such "
                f"column")
        return src_row.get(column)

    def _read_column(self, m: Mapping, column: str, anchor_row, src_row) -> Any:
        """The raw value a mapping reads from `column`, before transformation."""
        if m.source == ANCHOR_SOURCE_ID:
            if column not in anchor_row.index:
                raise ExecutionError(
                    f"{self.spec.ref}: mapping for {m.target!r} reads anchor "
                    f"column {column!r}, which the anchor does not have")
            return str(anchor_row[column])
        if src_row is None:
            raise ExecutionError(
                f"{self.spec.ref}: mapping for {m.target!r} reads source "
                f"{m.source!r} but no source row is available")
        return self._source_value(m, column, src_row)

    def _read(self, m: Mapping, anchor_row, src_row) -> Any:
        """The raw value a mapping reads, before any transformation."""
        if m.source == ANCHOR_SOURCE_ID:
            if m.column not in anchor_row.index:
                raise ExecutionError(
                    f"{self.spec.ref}: mapping for {m.target!r} reads anchor "
                    f"column {m.column!r}, which the anchor does not have")
            return str(anchor_row[m.column])
        if src_row is None:
            raise ExecutionError(
                f"{self.spec.ref}: mapping for {m.target!r} reads source "
                f"{m.source!r} but no source row is available")
        return self._source_value(m, m.column, src_row)

    def _apply(self, m: Mapping, raw: Any) -> Any:
        if m.op == OP_CONSTANT:
            return m.value
        if m.op == OP_DIRECT:
            if m.coerce == "numeric":
                return self._coerce(raw)
            if m.coerce == "text":
                return normalize_key(raw)
            return raw
        if m.op == OP_VALUE_MAP:
            return self._evaluate_value_map(m, raw)[0]
        if m.op == OP_UNIT_CONVERSION:
            key = normalize_key(raw)
            if key is None:
                return None
            try:
                converted = float(key) * float(m.factor)
            except (TypeError, ValueError):
                return None
            return float(round(converted)) if m.round else converted
        raise ExecutionError(
            f"{self.spec.ref}: operation {m.op!r} is not implemented")

    # ── joins ───────────────────────────────────────────────────────────────
    def _validate_joins(self, frames: dict[str, pd.DataFrame],
                        participant_key: str) -> tuple[str, ...]:
        """Check the declared relationship actually holds before reading values.

        A join that is asserted but never checked is a comment. Each declared
        join must name loaded sources, join on the participant key this executor
        indexes by, and satisfy its declared cardinality — an unexpected
        duplicate fails here rather than being resolved by whichever row pandas
        happened to keep.
        """
        validated = []
        for join in self.spec.record.joins:
            if join.cardinality not in SUPPORTED_CARDINALITIES:
                raise ExecutionError(
                    f"{self.spec.ref}: join {join.id!r} declares cardinality "
                    f"{join.cardinality!r}, which is not implemented "
                    f"(supported: {list(SUPPORTED_CARDINALITIES)})")
            if tuple(join.on) != (participant_key,):
                raise ExecutionError(
                    f"{self.spec.ref}: join {join.id!r} joins on {list(join.on)}, "
                    f"but this executor relates sources by the record's "
                    f"participant key {participant_key!r} only")
            for side, sid in (("left", join.left), ("right", join.right)):
                if sid not in frames:
                    raise ExecutionError(
                        f"{self.spec.ref}: join {join.id!r} names {side} source "
                        f"{sid!r}, which no executed mapping reads")
            # The frames are indexed by the join key, so a duplicate key would
            # already have been resolved by the duplicate policy; verify that
            # the policy did not silently mask a cardinality the spec denies.
            sides = ([("left", join.left), ("right", join.right)]
                     if join.cardinality == "one_to_one"
                     else [("right", join.right)])
            for side, sid in sides:
                index = frames[sid].index
                if index.has_duplicates:
                    dupes = sorted(index[index.duplicated()].unique())[:5]
                    raise ExecutionError(
                        f"{self.spec.ref}: join {join.id!r} declares "
                        f"{join.cardinality}, but the {side} source {sid!r} has "
                        f"duplicate {participant_key} value(s) {dupes}")
            validated.append(join.id)
        return tuple(validated)

    # ── predicates ──────────────────────────────────────────────────────────
    # ── named reads, conditions and derivation records ──────────────────────
    def _read_value(self, read, rows: dict, members: dict):
        """The value a named read resolves to for this record."""
        sid = read.source
        member = None
        if self._is_group_ref(sid):
            gid = sid.partition(GROUP_REF_SEP)[0]
            member = members.get(gid)
            sid = self._resolve_group_ref(sid, member)
        row = rows.get(sid)
        if row is None:
            return None
        column = read.column
        if read.column_by_member and member is not None:
            column = read.column_by_member.get(member.id, column)
        if not column or column not in row.index:
            raise ExecutionError(
                f"{self.spec.ref}: named read {read.id!r} reads missing "
                f"column {column!r} from source {sid!r}")
        return coerce_named_read(row[column], read.coerce)

    def _resolve_reads(self, rows: dict, members: dict) -> dict:
        return {r.id: self._read_value(r, rows, members) for r in self.spec.reads}

    def _resolve_conditions(self, reads: dict, members: dict,
                            origin: str = "observed") -> dict:
        out = {}
        for c in self.spec.conditions:
            if c.record_origin:
                out[c.id] = c.record_origin == origin
            elif c.group:
                member = members.get(c.group)
                out[c.id] = member is not None and member.id == c.member
            else:
                out[c.id] = self._predicate_on(c.predicate, reads.get(c.read),
                                               c.value)
        return out

    @staticmethod
    def _predicate_on(predicate: str, raw, wanted) -> bool:
        """Evaluate one declared predicate against one already-read value."""
        return evaluate_predicate(predicate, raw, wanted)

    def _derivation_for(self, m: Mapping, facts: dict) -> str:
        """The structured provenance record this mapping declares, filled in.

        Substitution only: each value is a literal the specification wrote, or a
        reference to a fact the engine already had. Nothing is computed here.
        """
        selected = select_derivation_record(
            m.derivation, facts["conditions"])
        if selected is None:
            return ""
        resolved = {ref: self._fact(ref, facts)
                    for ref in derivation_references(selected)}
        return render_structured_derivation(
            m.derivation, resolved, facts["conditions"])

    def _fact(self, ref: str, facts: dict):
        head, _, rest = ref.partition(".")
        if head in ("value", "raw_value", "source_column", "source_label"):
            return facts[head]
        if head in (REF_ANCHOR_GROUP_SIZE, REF_ANCHOR_SELECTED_ROW):
            grouped = facts["grouped"]
            if grouped is None:
                raise ExecutionError(
                    f"{self.spec.ref}: {ref!r} describes a grouped record, and "
                    f"this record was not built from a group")
            return (grouped.size if head == REF_ANCHOR_GROUP_SIZE
                    else grouped.selected_row)
        if head == "read":
            return facts["reads"].get(rest)
        if head == "condition":
            return facts["conditions"].get(rest)
        member = facts["members"].get(rest)
        if member is None:
            return None
        if head == "group_member":
            return member.id
        if head == "group_value":
            return member.value
        group = self.spec.group(rest)
        return self.ref(member.sources[group.decided_by]).logical_label

    def _predicate_holds(self, pred, rows: dict, reads: dict | None = None) -> bool:
        """Evaluate a declared predicate against independently read operands."""
        if pred.read:
            return evaluate_predicate(
                pred.predicate, (reads or {}).get(pred.read), pred.value)
        row = rows.get(pred.source)
        if row is None:
            raise ExecutionError(
                f"{self.spec.ref}: predicate reads source {pred.source!r}, which "
                f"this record has no row for")
        if pred.column not in row.index:
            raise ExecutionError(
                f"{self.spec.ref}: predicate reads missing column "
                f"{pred.column!r} from source {pred.source!r}")
        return evaluate_predicate(
            pred.predicate, row[pred.column], pred.value)

    # ── source groups ───────────────────────────────────────────────────────
    def _group_sources(self) -> set[str]:
        """Every declared source any group member offers."""
        return {sid for g in self.spec.source_groups
                for m in g.members for sid in m.sources.values()}

    def _resolve_group_ref(self, ref: str, member) -> str:
        """`arm.demographics` -> the source id the matched member offers."""
        gid, _, role = ref.partition(GROUP_REF_SEP)
        group = self.spec.group(gid)
        if group is None:
            raise ExecutionError(
                f"{self.spec.ref}: source group {gid!r} is not declared")
        if member is None:
            raise ExecutionError(
                f"{self.spec.ref}: source {ref!r} was read for a record that "
                f"matched no member of group {gid!r}")
        sid = member.sources.get(role)
        if sid is None:
            raise ExecutionError(
                f"{self.spec.ref}: group {gid!r} member {member.id!r} offers no "
                f"role {role!r}")
        return sid

    def _membership_role(self, group) -> str:
        """The role name recorded on each membership contribution.

        Named after the target field the membership serves, because that is what
        a reader traces back from.
        """
        return membership_role(self.spec, group)

    def _decide_group(self, group, pid: str, frames: dict, record_key: str):
        """Which member holds this record, with every member's outcome recorded.

        Returns `(member, contributions_json)`. Candidates after the winner are
        `not_evaluated`: membership short-circuits, so the later files were never
        consulted, and saying "rejected" would claim a judgement never made.
        """
        presence = {
            member.id: pid in frames[member.sources[group.decided_by]].index
            for member in group.members
        }
        decision = select_first_containing_record(
            tuple(member.id for member in group.members), presence,
            membership_status=group.membership_status)
        contributions = []
        for rank, member in enumerate(group.members):
            sid = member.sources[group.decided_by]
            frame = frames[sid]
            present = presence[member.id]
            contributions.append(from_source_ref(
                self.ref(sid),
                role=self._membership_role(group),
                status=decision.statuses[rank],
                # Membership, not a column: naming one would be a fabricated
                # locator for evidence that is the presence of the record itself.
                source_column="",
                raw_value=(pid if present else ""),
                rank=rank,
                source_record_key=record_key,
                source_row_number=(int(frame.loc[pid, "_spec_src_row"])
                                   if present else ""),
                rule_id=self._group_rule_id(group)))
        matched = (
            next(member for member in group.members
                 if member.id == decision.selected_member_id)
            if decision.selected_member_id else None)
        return matched, contributions_json(contributions)

    def _skip_for_missing(self, m: Mapping, rows: dict) -> bool:
        """True when a mapping reads a source this record is absent from.

        Every source a mapping touches has to be present for it to run, and a
        source that declared what its absence means has already said what to do
        instead. A mapping that reads nothing is never skipped.
        """
        needed = set()
        if self._reads_source_row(m):
            needed.add(m.source)
        needed |= {c.source for c in m.candidates}
        needed |= {c.when.source for c in m.candidates if c.when}
        needed |= {c.when.source for c in m.clauses}
        needed |= {c.source for c in m.clauses if c.source}
        # `provenance.row_from` is deliberately NOT a dependency: it says where
        # to locate the record, not where to read a value, so its absence
        # removes the row number and nothing else.
        return any(sid and not self._is_group_ref(sid) and sid not in rows
                   for sid in needed)

    def _group_of(self, m: Mapping) -> str:
        """The source group this mapping's reads depend on, if any."""
        if m.group:
            return m.group
        if m.provenance.contributions_from:
            return m.provenance.contributions_from
        for ref in (m.source, m.provenance.row_from):
            if self._is_group_ref(ref):
                return ref.partition(GROUP_REF_SEP)[0]
        return ""

    def _membership_derivation(self, group, matched, value) -> str:
        """What the membership decision was, as a structured record.

        Distinct from `contributions`, which describes the candidates. This
        describes the rule: what was on offer, what matched, and what it means.
        """
        return json.dumps({
            "kind": "file_membership",
            "rule_id": self._group_rule_id(group),
            "candidates": [self.ref(mem.sources[group.decided_by]).logical_label
                           for mem in group.members],
            "matched": self.ref(matched.sources[group.decided_by]).logical_label,
            "matched_config_key": matched.sources[group.decided_by],
            "final_value": value,
            "map_ref": (group.evidence_ref
                        or f"spec:{self.spec.spec_id}.source_groups.{group.id}"),
        }, separators=(",", ":"), sort_keys=True)

    def _group_rule_id(self, group) -> str:
        """How the membership decision identifies itself in provenance."""
        return group_rule_id(self.spec, group)

    # ── fallback ────────────────────────────────────────────────────────────
    def _evaluate_fallback(self, m: Mapping, anchor_row, rows: dict,
                           row_numbers: dict, record_key: str,
                           reads: dict | None = None):
        """Take the first eligible candidate that has a value.

        A candidate may be *gated*: it is eligible only while its declared
        predicate holds. That gives four distinct outcomes per candidate, and
        the difference between them is the whole point of recording them —

          selected       eligible, had a value, and won;
          unavailable    eligible, but the source held nothing;
          rejected       had a value, but its gate said no;
          <after>        never consulted, because an earlier candidate won.

        The last one is declared rather than assumed: one legacy mapping calls
        it `not_evaluated` and another calls it `rejected`, and reproducing both
        exactly means the specification has to say which.

        Returns `(value, raw, column, contributions_json, exhausted)`.
        """
        style = m.contributions
        candidate_contribs = []
        chosen = None                       # (label, raw)
        for rank, cand in enumerate(m.candidates):
            if cand.source == ANCHOR_SOURCE_ID:
                raise ExecutionError(
                    f"{self.spec.ref}: fallback for {m.target!r} reads the row "
                    f"anchor, which has no source identity to record")
            row = rows[cand.source]
            if cand.column not in row.index:
                raise ExecutionError(
                    f"{self.spec.ref}: fallback candidate {rank} reads missing "
                    f"column {cand.column!r} from source {cand.source!r}")
            raw = row[cand.column]
            eligible = cand.when is None or self._predicate_holds(
                cand.when, rows, reads)
            if chosen is not None:
                status = (STATUS_REJECTED if style.after_decision == "rejected"
                          else STATUS_NOT_EVALUATED)
            elif _is_missing(raw):
                status = STATUS_UNAVAILABLE
            elif not eligible:
                status = STATUS_REJECTED
            else:
                status = STATUS_SELECTED
            candidate_contribs.append(from_source_ref(
                self.ref(cand.source), role=f"{m.target}_candidate",
                status=status, source_column=cand.column, raw_value=raw,
                rank=rank, source_record_key=record_key,
                source_row_number=row_numbers.get(cand.source),
                rule_id=m.rule.id))
            if status == STATUS_SELECTED:
                chosen = (cand.label or cand.column, raw)

        # Direct gate inputs are recorded separately from candidates. Named-read
        # gate inputs are appended by _with_conditions from their declarations.
        condition_contribs = []
        seen_gates = []
        for cand in m.candidates:
            gate = cand.when
            if gate is None or gate.read:
                continue
            if (gate.source, gate.column) in seen_gates:
                continue
            seen_gates.append((gate.source, gate.column))
            if gate.column not in rows[gate.source].index:
                raise ExecutionError(
                    f"{self.spec.ref}: fallback gate reads missing column "
                    f"{gate.column!r} from source {gate.source!r}")
            condition_contribs.append(from_source_ref(
                self.ref(gate.source), role=f"{m.target}_condition",
                status=STATUS_CONDITION, source_column=gate.column,
                raw_value=rows[gate.source][gate.column], rank=-1,
                source_record_key=record_key,
                source_row_number=row_numbers.get(gate.source),
                rule_id=m.rule.id))

        ordered = (condition_contribs + candidate_contribs
                   if style.condition_position == "first"
                   else candidate_contribs + condition_contribs)
        # A chain that records nothing still evaluated its candidates; it just
        # does not carry the list on a successful result.
        payload = "" if style.record == "none" else contributions_json(ordered)

        if chosen is None:
            column = (style.no_value_column
                      or "+".join(c.column for c in m.candidates))
            return None, None, column, payload, True
        label, raw = chosen
        if m.map:
            # The chain decides WHICH value; the map decides what it means. The
            # two compose, and keeping them separate is what lets a trial say
            # "first of these columns, then translate it" in one mapping.
            value = apply_inline_fallback_map(
                raw, mapping_table=m.map,
                on_unmapped=m.on_unmapped).value
        else:
            value = self._coerce(raw) if m.coerce == "numeric" else raw
        return value, raw, label, payload, False

    # ── conditional ─────────────────────────────────────────────────────────
    def _evaluate_conditional(self, m: Mapping, rows: dict, reads: dict):
        """First clause whose predicate holds supplies the value.

        Unlike a fallback, a clause may yield a literal rather than a column:
        the answer is not read from anywhere, it is what the condition being
        true *means*. Nothing falls through on a null — a clause that holds
        wins even if it yields nothing, because it was the declared answer.
        """
        for clause in m.clauses:
            if not self._predicate_holds(clause.when, rows, reads):
                continue
            if clause.read:
                return reads.get(clause.read)
            if clause.column:
                row = rows[clause.source]
                if clause.column not in row.index:
                    raise ExecutionError(
                        f"{self.spec.ref}: conditional clause reads missing "
                        f"column {clause.column!r} from source {clause.source!r}")
                return row[clause.column]
            return clause.value
        return m.otherwise

    def _evaluate_value_map(self, m: Mapping, raw: Any):
        """Resolve a value map to `(value, confidence, rule)`.

        The order is declared, not inferred: this mapping's own table first,
        then the shared vocabulary it names, then the declared unmapped
        behaviour. Each outcome carries the confidence and the rule identity
        that outcome states — nothing is derived from a confidence threshold.
        """
        if vocabularies.is_absent(raw):
            return None, m.confidence_no_value, m.no_value_rule or m.rule

        key = normalize_key(raw)
        if key in m.map:
            return m.map[key], m.confidence, m.rule

        if m.fallback_rules:
            hit = vocabularies.match(raw, m.fallback_rules)
            if hit is not None:
                value = hit.value if hit.value is not None else raw
                return value, hit.confidence, RuleRef(id=hit.rule_id,
                                                      version=hit.rule_version)

        if m.on_unmapped == "error":
            raise ExecutionError(
                f"{self.spec.ref}: mapping for {m.target!r} has no entry for "
                f"{key!r} and on_unmapped is 'error'")
        value = raw if m.on_unmapped == "passthrough" else None
        rule = m.unmapped_rule or m.rule
        return value, m.confidence_unmapped, rule

    def _confidence_for(self, m: Mapping, value: Any) -> float:
        """The declared confidence for the outcome that actually occurred."""
        if m.op in (OP_NOT_APPLICABLE, OP_NO_MAPPING_CONFIGURED):
            # A declared absence has one outcome, not two: the confidence is in
            # the assertion — that the field does not apply, or that no mapping
            # has been approved for it — and not in a value.
            return float(m.confidence)
        if not _is_missing(value):
            return float(m.confidence)
        if m.confidence_no_value is None:
            raise ExecutionError(
                f"{self.spec.ref}: mapping for {m.target!r} produced no value, "
                f"but declares no confidence.no_value; refusing to reuse the "
                f"confidence of a value that was found")
        return float(m.confidence_no_value)

    @staticmethod
    def _coerce(raw: Any):
        key = normalize_key(raw)
        if key is None or key in ("", "."):
            return None
        try:
            return float(key)
        except (TypeError, ValueError):
            return None

    # ── provenance identity ─────────────────────────────────────────────────
    def _origin(self, m: Mapping, member=None) -> Any:
        """What `cell()` records as the value's origin.

        A file-backed value is attributed to its ``SourceRef``, so the resolved
        path and content hash travel with it. A value that came from a constant,
        a declared map or the anchor has no file, so the specification may name
        its origin label; otherwise a generic one is derived from the operation.
        Trial-specific legacy labels live in the trial's specification, never
        here.
        """
        pv = m.provenance
        if pv.source_ref:
            sid = (self._resolve_group_ref(pv.source_ref, member)
                   if self._is_group_ref(pv.source_ref) else pv.source_ref)
            return self.ref(sid)
        if pv.source_label:
            return pv.source_label
        if m.op == OP_FILE_MEMBERSHIP and m.group and member is not None:
            group = self.spec.group(m.group)
            return self.ref(member.sources[group.decided_by])
        if self._is_group_ref(m.source):
            return self.ref(self._resolve_group_ref(m.source, member))
        if m.source and m.source != ANCHOR_SOURCE_ID:
            return self.ref(m.source)
        if m.source == ANCHOR_SOURCE_ID:
            return "ANCHOR"
        if m.candidates:
            return self.ref(m.candidates[0].source)
        return f"SPEC:{m.op}"

    def _with_conditions(self, m: Mapping, payload: str, reads: dict, rows: dict,
                         row_numbers: dict, record_key: str, members: dict,
                         read_ids=None) -> str:
        """Append the declared condition contributions to a contribution list.

        A condition informed which branch was taken but supplied no value of its
        own, so it is recorded as one — never as a candidate, which would claim
        a value it never gave.
        """
        existing = json.loads(payload) if payload else []
        by_id = {r.id: r for r in self.spec.reads}
        extra = []
        for name in (m.contributions.conditions if read_ids is None else read_ids):
            read = by_id[name]
            sid = read.source
            member = None
            if self._is_group_ref(sid):
                group_id, _, _role = sid.partition(GROUP_REF_SEP)
                member = members.get(group_id)
                sid = self._resolve_group_ref(sid, member)
            column = read.column
            if read.column_by_member and member is not None:
                column = read.column_by_member.get(member.id, column)
            extra.append(from_source_ref(
                self.ref(sid), role=name, status=STATUS_CONDITION,
                source_column=column, raw_value=reads.get(name),
                rank=m.contributions.condition_rank,
                source_record_key=record_key,
                source_row_number=row_numbers.get(sid),
                rule_id=m.rule.id).as_record())
        return json.dumps(existing + extra, separators=(",", ":"), sort_keys=True)

    def _supports_stage17_raw_derivation(self, m: Mapping) -> bool:
        """Whether execution is on the approved Stage 17 raw route boundary."""
        if m.op == OP_CUSTOM_RULE:
            return False
        if m.op in (OP_CONSTANT, OP_NOT_APPLICABLE,
                    OP_NO_MAPPING_CONFIGURED):
            return True
        if m.op in (OP_DIRECT, OP_VALUE_MAP, OP_UNIT_CONVERSION):
            return (
                is_stage17_source_group_mapping(self.spec, m)
                or bool(m.source and m.source != ANCHOR_SOURCE_ID
                        and not m.source.startswith(f"{ANCHOR_SOURCE_ID}."))
            )
        if m.op == OP_FILE_MEMBERSHIP:
            return is_stage17_source_group_mapping(self.spec, m)
        if m.op == OP_CONDITIONAL:
            return bool(m.clauses and not m.candidates and not m.map)
        if m.op == OP_FALLBACK:
            return bool(
                m.candidates and not m.map and not m.fallback_vocabulary
                and m.coerce in ("none", "numeric"))
        return False

    def _derivation_evidence_dependencies(
            self, m: Mapping, conditions: dict) -> tuple[tuple[str, ...], tuple[str, ...]]:
        """Return selected Stage 16 read/group dependencies in causal order."""
        if not m.derivation:
            return (), ()
        selection = select_derivation(m.derivation, conditions)
        if selection.record is None:
            refs = frozenset()
        else:
            refs = derivation_references(selection.record)
        has_raw_value = "raw_value" in refs
        if (not all(
                is_stage16_fact(ref) or ref == "raw_value"
                or (ref in {REF_ANCHOR_GROUP_SIZE, REF_ANCHOR_SELECTED_ROW}
                    and mapping_uses_record_construction(self.spec, m))
                for ref in refs)
                or (has_raw_value
                    and not self._supports_stage17_raw_derivation(m))):
            return (), ()

        by_condition = {condition.id: condition for condition in self.spec.conditions}
        read_ids: list[str] = []
        group_ids: list[str] = []

        def add_read(read_id: str) -> None:
            if read_id and read_id not in read_ids:
                read_ids.append(read_id)

        def add_group(group_id: str) -> None:
            if group_id and group_id not in group_ids:
                group_ids.append(group_id)

        def add_condition(condition_id: str) -> None:
            condition = by_condition.get(condition_id)
            if condition is None or condition.record_origin:
                return
            if condition.read:
                add_read(condition.read)
            elif condition.group:
                add_group(condition.group)

        for condition_id in selection.condition_ids:
            add_condition(condition_id)
        for ref in sorted(refs):
            head, _, name = ref.partition(".")
            if head == "read":
                add_read(name)
            elif head == "condition":
                add_condition(name)
            elif head in ("group_member", "group_value", "group_label"):
                add_group(name)

        by_read = {read.id: read for read in self.spec.reads}
        for read_id in read_ids:
            read = by_read.get(read_id)
            if read is not None and self._is_group_ref(read.source):
                group_id, _, _role = read.source.partition(GROUP_REF_SEP)
                add_group(group_id)
        return tuple(read_ids), tuple(group_ids)

    def _with_derivation_evidence(
            self, m: Mapping, payload: str, reads: dict, conditions: dict,
            rows: dict, row_numbers: dict, record_key: str, members: dict,
            memberships: dict, *, extra_read_ids=()) -> str:
        """Append selected derivation/raw-from facts using existing records."""
        read_ids, group_ids = self._derivation_evidence_dependencies(
            m, conditions)
        read_ids = list(read_ids)
        group_ids = list(group_ids)
        by_read = {read.id: read for read in self.spec.reads}
        for read_id in extra_read_ids:
            if read_id and read_id not in read_ids:
                read_ids.append(read_id)
            read = by_read.get(read_id)
            if read is not None and self._is_group_ref(read.source):
                group_id, _, _role = read.source.partition(GROUP_REF_SEP)
                if group_id not in group_ids:
                    group_ids.append(group_id)
        existing = json.loads(payload) if payload else []
        roles = {item.get("role") for item in existing}
        for group_id in group_ids:
            group = self.spec.group(group_id)
            role = membership_role(self.spec, group) if group is not None else ""
            if role and role not in roles:
                existing.extend(json.loads(memberships[group_id]))
                roles.add(role)
        payload = json.dumps(existing, separators=(",", ":"), sort_keys=True) \
            if existing else ""
        missing_reads = tuple(read_id for read_id in read_ids
                              if read_id not in roles)
        if missing_reads:
            payload = self._with_conditions(
                m, payload, reads, rows, row_numbers, record_key, members,
                read_ids=missing_reads)
        return payload

    def _origin_label(self, m: Mapping, member=None) -> str:
        """The logical label of whatever this value is attributed to."""
        origin = self._origin(m, member)
        return getattr(origin, "logical_label", origin)

    def _reads_source_row(self, m: Mapping) -> bool:
        return bool(m.source) and m.source != ANCHOR_SOURCE_ID

    def _row_source(self, m: Mapping, member=None) -> str:
        """Which source's matched row locates this value, if any."""
        if m.provenance.source_ref and self._is_group_ref(m.provenance.source_ref) \
                and not m.provenance.row_from and not self._reads_source_row(m):
            return self._resolve_group_ref(m.provenance.source_ref, member)
        if m.provenance.row_from:
            ref = m.provenance.row_from
            return (self._resolve_group_ref(ref, member)
                    if self._is_group_ref(ref) else ref)
        if self._reads_source_row(m):
            return (self._resolve_group_ref(m.source, member)
                    if self._is_group_ref(m.source) else m.source)
        return m.provenance.source_ref

    @staticmethod
    def _rule_raw_value(inputs: tuple[str, ...], values: list[Any]) -> Any:
        """What a derived cell records as its raw value.

        A rule with one input records that input's value. A rule with several
        records the inputs it was given, named, because no single one of them is
        "the" raw value and a reader has to be able to see all of them.
        """
        if len(inputs) == 1:
            return values[0]
        return ";".join(f"{n}={v!r}" for n, v in zip(inputs, values))

    def _event_row_identity(self, anchor_row) -> dict[str, str]:
        """Adapt the current record to the schema-declared replay row key."""
        keys = tuple(getattr(self.contract, "row_key", ()) or ())
        if not keys:
            # Backward-compatible non-canonical callers do not have a schema
            # contract.  Use the specification's declared record identity;
            # replay still requires the pinned schema adapter to agree.
            keys = tuple(dict.fromkeys(name for name in (
                self.spec.record.participant_key,
                self.spec.record.sample_key,
                self.spec.record.collection_event_key,
            ) if name))
        try:
            return canonical_row_identity(keys, anchor_row)
        except ValueError as exc:
            raise ExecutionError(
                f"{self.spec.ref}: cannot represent replay row identity: {exc}"
            ) from exc

    def _custom_input_contributions(
            self, mapping: Mapping, values: list[Any],
            produced_cells: dict[str, Cell], row_identity: dict[str, str],
            record_key: str) -> str:
        """Bind a custom-rule call to the exact upstream Cells it consumed."""
        items: list[Contribution] = []
        for rank, (name, value) in enumerate(zip(mapping.inputs, values)):
            upstream = produced_cells.get(name)
            if upstream is None:
                raise ExecutionError(
                    f"{self.spec.ref}: custom rule for {mapping.target!r} has "
                    f"no emitted upstream Cell for input {name!r}")
            reference = ""
            if self.run_id and upstream.mapping_id:
                reference = provenance_event_ref(
                    run_id=self.run_id,
                    spec_ref=self.spec.ref,
                    mapping_id=upstream.mapping_id,
                    row_identity=row_identity,
                    harmonized_field=upstream.harmonized_field,
                )
            try:
                typed = encode_typed_raw(value).to_json()
            except ValueError as exc:
                if self.run_id:
                    raise ExecutionError(
                        f"{self.spec.ref}: custom input {name!r} is not a "
                        f"replay scalar: {exc}") from exc
                typed = ""
            items.append(Contribution(
                role="registered_rule_input",
                status=STATUS_INPUT,
                source_column=name,
                raw_value=value,
                raw_value_json=typed,
                rank=rank,
                source_record_key=record_key,
                provenance_ref=reference,
            ))
        return contributions_json(items)

    # ── execution ───────────────────────────────────────────────────────────
    def extract(self) -> Iterator[Cell]:
        """Emit the Cell stream this specification describes."""
        rec = self.spec.record
        if rec.anchor not in ("external", "grouping"):
            raise ExecutionError(
                f"{self.spec.ref}: this executor sources records from an "
                f"external row anchor or by grouping a source, not "
                f"{rec.anchor!r}")
        participant_key = rec.participant_key
        if not participant_key:
            raise ExecutionError(
                f"{self.spec.ref}: record.participant_key is required")
        record_key_column = rec.record_key_column or participant_key

        # A grouped anchor is constructed before anything else reads a source:
        # every later step is about the records it produces.
        grouping_source = ""
        if rec.anchor == "grouping":
            grouping_source = rec.anchor_source
            anchor, self._grouped, self._origins = self._build_grouped_anchor()
            self.anchor = anchor
            self.diagnostics.anchor_rows = len(anchor)

        runnable: list[Mapping] = []
        for m in self.spec.mappings:
            reason = self._defer_reason(m)
            if reason:
                self.diagnostics.deferred[m.target] = reason
                continue
            self._check_executable(m)
            runnable.append(m)
        runnable = self._ordered(runnable)
        self.diagnostics.executed = tuple(m.target for m in runnable)
        self.diagnostics.order = self.diagnostics.executed

        # Load every source a runnable mapping touches, once. A group's members
        # are all loaded: which one holds a given record is decided per record.
        needed = {m.source for m in runnable if self._reads_source_row(m)}
        needed |= {m.provenance.row_from for m in runnable}
        needed |= {m.provenance.source_ref for m in runnable}
        needed |= {c.source for m in runnable for c in m.candidates}
        needed |= {c.when.source for m in runnable for c in m.candidates if c.when}
        needed |= {c.when.source for m in runnable for c in m.clauses}
        needed |= {c.source for m in runnable for c in m.clauses if c.source}
        needed = {sid for sid in needed if sid and not self._is_group_ref(sid)}
        if any(m.group or self._is_group_ref(m.source)
               or self._is_group_ref(m.provenance.row_from) for m in runnable):
            needed |= self._group_sources()
        # The grouping anchor source is NOT indexed one row per participant.
        # That is the whole point of it: it holds several rows per record, and
        # collapsing it to one would either discard the group or, under an
        # `error` duplicate policy, refuse a file that is doing nothing wrong.
        # Its per-record row is the row its own group selected. Every other
        # source keeps the duplicate protections unchanged.
        needed -= {grouping_source}
        frames = {sid: self._indexed_source(sid, participant_key)
                  for sid in sorted(needed)}
        self.diagnostics.joins_validated = self._validate_joins(
            frames, participant_key)
        self.diagnostics.vocabularies = self.spec.vocabularies
        self.diagnostics.exclusions_deferred = tuple(
            e.id for e in self.spec.exclusions)
        tallies = {g.id: {mem.id: 0 for mem in g.members}
                   for g in self.spec.source_groups}

        # A source is "required" only if some mapping reads it unconditionally.
        # A group member's files are read only for the records it claims, so a
        # participant absent from the other arm's files is not a lookup miss.
        required = {sid for sid in frames} - self._group_sources()

        misses: list[str] = []
        emitted = 0
        grouped_records = self._grouped or [None] * len(self.anchor)
        origins = self._origins or [""] * len(self.anchor)
        for (anchor_position, anchor_row), grouped, declared_by in zip(
                self.anchor.iterrows(), grouped_records, origins):
            pid = str(anchor_row[participant_key])
            # The identity is the anchor's; the NAME it is written under is the
            # record's own declaration, because a derived cell belongs to no
            # file and still has to say which record it is about.
            record_key = f"{record_key_column}={pid}"
            group_key = self._grouped_record_key(grouped) if grouped else ""

            rows: dict[str, Any] = {}
            row_numbers: dict[str, Any] = {}
            for sid, frame in frames.items():
                if pid in frame.index:
                    row = frame.loc[pid]
                    rows[sid] = row
                    row_numbers[sid] = int(row["_spec_src_row"])
            if grouped is not None:
                # The anchor source takes part like any other source; its row
                # for this record is simply the one the group selected.
                rows[grouping_source] = grouped.source_row
                row_numbers[grouping_source] = grouped.selected_row
            missing_from = sorted(required - set(rows))
            undeclared = [sid for sid in missing_from
                          if self.spec.source(sid).on_missing_record.mode == "error"]
            if undeclared:
                # Only a source that has SAID what a missing record means may
                # have one. Anywhere else it stays a finding, never a silent null.
                misses.append(pid)
                raise ExecutionError(
                    f"{self.spec.ref}: participant {pid!r} is in the row anchor "
                    f"but missing from source(s) {undeclared}, which declare no "
                    f"on_missing_record behaviour")
            if missing_from:
                misses.append(pid)

            # One membership decision per group, per record.
            members: dict[str, Any] = {}
            memberships: dict[str, str] = {}
            for group in self.spec.source_groups:
                member, payload = self._decide_group(group, pid, frames, record_key)
                if member is None:
                    raise ExecutionError(
                        f"{self.spec.ref}: participant {pid!r} belongs to no "
                        f"member of source group {group.id!r}; "
                        f"lookup-miss reporting is not implemented")
                members[group.id] = member
                memberships[group.id] = payload
                tallies[group.id][member.id] += 1

            # Named reads and conditions: facts about this record that a
            # predicate can test and a provenance record can quote, resolved
            # once so both see the same thing.
            reads = self._resolve_reads(rows, members)
            conditions = self._resolve_conditions(
                reads, members,
                "synthesized" if declared_by else "observed")

            # Values and exact Cells this record has already produced, so a
            # registered rule and its provenance bind to the same inputs.
            produced: dict[str, Any] = {}
            produced_cells: dict[str, Cell] = {}
            row_identity = self._event_row_identity(anchor_row)

            for m in runnable:
                if m.when and not conditions.get(m.when):
                    # This mapping does not apply to this record. Two mappings
                    # for one target, gated on complementary conditions, is how
                    # a trial says which rule decides which records.
                    continue
                if self._skip_for_missing(m, rows):
                    continue
                member = members.get(self._group_of(m))
                sid = (self._resolve_group_ref(m.source, member)
                       if self._is_group_ref(m.source)
                       else (m.source if self._reads_source_row(m) else ""))
                src_row = rows.get(sid) if sid else None
                contributions = ""
                derivation = ""
                exhausted = False
                confidence = None
                mapped_rule = m.rule
                column = m.column or m.target
                if m.column_by_member and member is not None:
                    column = m.column_by_member[member.id]

                if m.op == OP_CUSTOM_RULE:
                    rule_impl = self._rules[m.target]
                    values = [produced.get(name) for name in m.inputs]
                    contributions = self._custom_input_contributions(
                        m, values, produced_cells, row_identity, record_key)
                    try:
                        value, rule_confidence, _note = rule_registry.invoke(
                            rule_impl, values, dict(m.params))
                        confidence = finalize_registered_rule_confidence(
                            value, rule_confidence,
                            declared_value_policy=m.confidence,
                            declared_no_value_confidence=
                            m.confidence_no_value).value
                    except (rule_registry.RuleError,
                            ConfidencePolicyError) as exc:
                        raise ExecutionError(
                            f"{self.spec.ref}: mapping for {m.target!r} has "
                            f"invalid registered-rule confidence: {exc}") from exc
                    raw = self._rule_raw_value(m.inputs, values)
                elif m.op == OP_FILE_MEMBERSHIP:
                    group = self.spec.group(m.group)
                    matched = members[group.id]
                    value = matched.value
                    raw = pid
                    contributions = memberships[group.id]
                    derivation = self._membership_derivation(group, matched, value)
                elif m.op == OP_NOT_APPLICABLE:
                    # A declared absence. There is nothing to read and nothing to
                    # transform; the rule id and the value state carry the reason.
                    value = None
                    raw = None
                elif m.op == OP_NO_MAPPING_CONFIGURED:
                    # The source value IS read and kept as the raw value — the
                    # open question is what it should become, and a reviewer
                    # settling it needs to see what the file actually said.
                    raw = self._read_column(m, column, anchor_row, src_row)
                    value = None
                    if m.source == ANCHOR_SOURCE_ID and rec.anchor == "grouping":
                        contributions = self._anchor_input_contribution(
                            m, raw, grouped, declared_by, record_key, group_key)
                elif m.op == OP_CONDITIONAL:
                    raw = (self._read_column(m, column, anchor_row, src_row)
                           if m.source else None)
                    value = self._evaluate_conditional(m, rows, reads)
                elif m.op == OP_FALLBACK:
                    value, raw, chosen_label, contributions, exhausted = \
                        self._evaluate_fallback(m, anchor_row, rows,
                                                row_numbers, record_key, reads)
                    # A mapping that names its own column describes the chain as
                    # a whole and keeps that name whichever candidate wins; one
                    # that does not lets the winning candidate name itself.
                    if not m.column:
                        column = chosen_label
                elif m.op == OP_VALUE_MAP:
                    raw = self._read_column(m, column, anchor_row, src_row)
                    value, confidence, mapped_rule = \
                        self._evaluate_value_map(m, raw)
                    if (is_external_anchor_evidence_mapping(self.spec, m)
                            and self._anchor_evidence is not None):
                        contributions = self._anchor_input_contribution(
                            m, raw, grouped, declared_by, record_key, group_key,
                            filtered_position=int(anchor_position))
                    if confidence is None:
                        raise ExecutionError(
                            f"{self.spec.ref}: mapping for {m.target!r} reached an "
                            f"outcome it declares no confidence for; declare "
                            f"confidence.unmapped and/or confidence.no_value")
                elif m.op == OP_CONSTANT:
                    raw = m.value
                    value = self._apply(m, raw)
                else:
                    raw = self._read_column(m, column, anchor_row, src_row)
                    value = self._apply(m, raw)
                    if (is_grouping_anchor_direct_evidence_mapping(self.spec, m)
                            or (is_external_anchor_evidence_mapping(self.spec, m)
                                and self._anchor_evidence is not None)):
                        contributions = self._anchor_input_contribution(
                            m, raw, grouped, declared_by, record_key, group_key,
                            filtered_position=int(anchor_position))
                produced[m.target] = value

                if confidence is None:
                    confidence = self._confidence_for(m, value)
                if m.op == OP_VALUE_MAP:
                    # The value map already resolved which outcome occurred and
                    # therefore which rule describes it.
                    rule = mapped_rule
                else:
                    rule = m.rule
                    if _is_missing(value) and m.no_value_rule is not None:
                        rule = m.no_value_rule

                pv = m.provenance
                row_sid = self._row_source(m, member)
                if not row_sid and m.op == OP_FALLBACK and m.candidates:
                    row_sid = m.candidates[0].source
                if not row_sid and m.op == OP_FILE_MEMBERSHIP:
                    row_sid = members[m.group].sources[
                        self.spec.group(m.group).decided_by]
                if pv.row_number:
                    wants_row_number = pv.row_number == "source"
                else:
                    wants_row_number = bool(row_sid)
                row_no = row_numbers.get(row_sid) if wants_row_number else None

                if pv.record_key:
                    key = pv.record_key
                elif pv.record_key_from == "participant":
                    key = record_key
                elif pv.record_key_from == "group":
                    if not group_key:
                        raise ExecutionError(
                            f"{self.spec.ref}: mapping for {m.target!r} states "
                            f"a grouped record key, but this record was not "
                            f"built from a group")
                    key = group_key
                elif row_sid or m.op in (OP_NOT_APPLICABLE, OP_CUSTOM_RULE,
                                         OP_NO_MAPPING_CONFIGURED):
                    key = record_key
                else:
                    key = ""

                group_id = source_group_id_for_mapping(m)
                if (is_stage17_source_group_mapping(self.spec, m)
                        and group_id):
                    contributions = memberships[group_id]
                elif pv.contributions_from and not contributions:
                    # Which file a record was read from is part of why the value
                    # is what it is, so a mapping that reads through a group can
                    # carry that evidence instead of leaving it to be inferred.
                    contributions = memberships[pv.contributions_from]
                if m.contributions.conditions:
                    contributions = self._with_conditions(
                        m, contributions, reads, rows, row_numbers, record_key,
                        members)

                completion = (
                    COMPLETION_FALLBACK_EXHAUSTED
                    if exhausted else COMPLETION_VALUE)
                if m.op == OP_CUSTOM_RULE:
                    # Registered custom rules retain their existing synthesized
                    # raw contract; Stage 17 deliberately does not broaden it.
                    if pv.raw_value == "none":
                        raw_out = None
                        record_raw_snapshot = False
                    elif pv.raw_value == "coerced":
                        raw_out = value
                        record_raw_snapshot = True
                    else:
                        raw_out = reads.get(pv.raw_from) if pv.raw_from else raw
                        record_raw_snapshot = True
                else:
                    finalized_raw = finalize_raw_value(
                        VerifiedOperationRouteResult(
                            value=value,
                            raw_candidate=VerifiedRawCandidate(True, raw),
                            completion=completion),
                        reads, raw_from_id=pv.raw_from,
                        raw_policy=pv.raw_value)
                    raw_out = finalized_raw.value
                    record_raw_snapshot = finalized_raw.record_snapshot

                causal_raw_from = (
                    pv.raw_from
                    if m.op != OP_CUSTOM_RULE and raw_from_is_causal(
                        raw_from_id=pv.raw_from, raw_policy=pv.raw_value,
                        completion=completion)
                    else "")
                if m.derivation or causal_raw_from:
                    contributions = self._with_derivation_evidence(
                        m, contributions, reads, conditions, rows, row_numbers,
                        record_key, members, memberships,
                        extra_read_ids=((causal_raw_from,)
                                        if causal_raw_from else ()))
                state = (
                    finalize_declared_value_state(
                        m.value_state, pv.value_state).serialized_state
                    if m.value_state in STATIC_PRESENT_DECLARATIONS
                    else "" if pv.value_state == "none" else m.value_state)

                if m.derivation:
                    derivation = self._derivation_for(m, {
                        "value": value, "raw_value": raw_out,
                        "source_column": column,
                        "source_label": self._origin_label(m, member),
                        "reads": reads, "conditions": conditions,
                        "members": members, "grouped": grouped,
                    })

                cell = self.cell(
                    anchor_row, m.target, value, confidence,
                    self._origin(m, member),
                    column,
                    -1,                                   # deprecated locator
                    f"spec:{m.op}",
                    raw_value=raw_out,
                    raw_evidence_recorded=record_raw_snapshot,
                    source_row_number=row_no,
                    source_record_key=key,
                    rule_id=rule.id, rule_version=rule.version,
                    value_state=state,
                    authoritative_for=";".join(m.artifacts),
                    contributions=contributions,
                    derivation=derivation,
                    mapping_id=m.mapping_id,
                    evidence_refs=";".join(filter(None, (
                        *self._vocabulary_evidence_refs(m),
                        (self._anchor_evidence.evidence_ref_id
                         if ((m.op == OP_NO_MAPPING_CONFIGURED
                              and m.source == ANCHOR_SOURCE_ID)
                             or is_grouping_anchor_direct_evidence_mapping(
                                 self.spec, m)
                             or mapping_uses_record_construction(self.spec, m)
                             or is_external_anchor_evidence_mapping(
                                 self.spec, m))
                         and self._anchor_evidence is not None else ""),
                    ))),
                )
                produced_cells[m.target] = cell
                yield cell
                emitted += 1

            for sid in missing_from:
                miss = self.spec.source(sid).on_missing_record
                if miss.mode != "lookup_miss":
                    continue
                for target in miss.emit_for:
                    yield self.cell(
                        anchor_row, target, None, float(miss.confidence),
                        self.ref(sid), miss.column, -1, "spec:lookup_miss",
                        source_record_key=record_key,
                        rule_id=miss.rule.id, rule_version=miss.rule.version,
                    )
                    emitted += 1

        self.diagnostics.group_membership = tallies
        self.diagnostics.cells = emitted
        self.diagnostics.lookup_misses = tuple(misses)
        self.diagnostics.sources = RuntimeSourceRefs(
            self.source_refs(),
            anchor_evidence=((self._anchor_evidence,)
                             if self._anchor_evidence is not None else ()),
        )
        if misses:
            LOG.warning("%s: %d anchor participants absent from source",
                        self.spec.ref, len(misses))


def execute(spec: HarmonizationSpec, project_root: Path,
            anchor_df: pd.DataFrame | RuntimeExternalAnchor | None = None, *, contract=None,
            targets: tuple[str, ...] | None = None, run_id: str = "") -> tuple[list[Cell], Diagnostics]:
    """Run a specification and return its cells plus what happened."""
    ex = SpecExecutor(spec, project_root, anchor_df, contract=contract,
                      targets=targets, run_id=run_id)
    cells = list(ex.extract())
    return cells, ex.diagnostics

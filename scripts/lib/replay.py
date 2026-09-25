"""Strict replay for one supported harmonized cell.

The manifest supplied by the caller is the local trust root.  This module does
not discover files and does not invoke :class:`SpecExecutor` (or any other
harmonization, assembly, suppression, or post-rule pipeline).

The trust boundary is deliberate:

1. An operation-specific recomputation function receives only freshly verified
   source/dependency values and pinned mapping parameters.
2. The returned value is compared with provenance and delivered values only
   after recomputation has completed.
"""
from __future__ import annotations

import csv
import hashlib
import json
import math
import platform
import re
from dataclasses import dataclass, replace as dc_replace
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any, Mapping

import pandas as pd
import numpy as np
import openpyxl
import yaml

from . import rule_registry, vocabularies
from .anchor_evidence import (
    ANCHOR_RECORD_EXTERNAL, ANCHOR_RECORD_GROUPED, ANCHOR_RECORD_SYNTHESIZED,
    EXTERNAL_ANCHOR_CONSTRUCTION_VERSION,
    EXTERNAL_ANCHOR_FRAME_VERSION,
    EXTERNAL_ANCHOR_READER_FLAGS,
    EXTERNAL_ANCHOR_READER_ID,
    EXTERNAL_ANCHOR_SOURCE_ID,
    RuntimeExternalAnchorEvidence,
    RuntimeGroupingAnchorEvidence, grouping_anchor_identity,
    filter_external_anchor_frame,
    is_external_anchor_evidence_mapping,
    parse_external_anchor_bytes,
    typed_value,
)
from .cidc_io import read_cidc_csv
from .conditions import (
    coerce_named_read, evaluate_predicate, first_true_index, is_missing_scalar,
)
from .confidence_policy import (
    InvalidRuleConfidence,
    UnsupportedConfidencePolicy,
    finalize_registered_rule_confidence,
)
from .contributions import (
    STATUS_CONDITION,
    STATUS_INPUT,
    STATUS_NOT_EVALUATED,
    STATUS_REJECTED,
    STATUS_SELECTED,
    STATUS_UNAVAILABLE,
    from_json as contributions_from_json,
)
from .derivation import (
    DerivationError,
    MissingDerivationFact,
    STAGE14_FACTS,
    derivation_references,
    is_stage14_derivation,
    is_stage16_derivation,
    is_stage16_fact,
    is_stage17_derivation,
    is_stage19_fact,
    parse_canonical_derivation_json,
    render_structured_derivation,
    select_derivation,
)
from .group_selection import (
    group_rule_id,
    is_closed_source_group_mapping,
    is_stage17_source_group_mapping,
    membership_role,
    scalar_role_for_mapping,
    select_first_containing_record,
    source_group_for_mapping,
)
from .fallback_policy import (
    apply_inline_fallback_map, select_ordered_fallback,
)
from .provenance import (
    PROVENANCE_SCHEMA_VERSION,
    canonical_row_identity,
    is_provenance_event_ref,
    provenance_event_ref,
)
from .raw_policy import (
    COMPLETION_FALLBACK_EXHAUSTED,
    COMPLETION_VALUE,
    FinalizedRaw,
    RawAuthorityUnavailable,
    VerifiedOperationRouteResult,
    VerifiedRawCandidate,
    finalize_raw_value,
    raw_from_is_causal,
)
from .record_construction import (
    ORIGIN_OBSERVED,
    ORIGIN_SYNTHESIZED,
    RecordConstructionError,
    construct_grouping_records,
    mapping_uses_grouping_facts,
    mapping_uses_record_construction,
    mapping_uses_record_origin,
)
from .replay_contract import (
    REASON_CODES_VERSION,
    REPLAY_API_VERSION,
    REPLAY_STATUS_VERSION,
    ReplayStatus,
    TypedRawValue,
    encode_typed_raw,
)
from .replay_manifest import (
    ArtifactBinding,
    EvidenceBinding,
    FileBinding,
    ManifestError,
    ReplayManifest,
    SourceBinding,
    SpecificationBinding,
    UnsupportedManifestVersion,
    load as load_manifest,
    parse as parse_manifest,
    resolve_project_path,
)
from .schema_model import Field, SchemaError, TargetSchema, load as load_schema
from .source_ref import PathContainmentError, SourceRef, sha256_file
from .xlsx_reader import (
    XLSX_READER_DEPENDENCIES, XLSX_READER_FLAGS, XLSX_READER_ID,
    XLSX_SHEET_CANONICALIZATION_ID, XlsxReaderError, parse_xlsx_bytes,
    xlsx_sheet_sha256,
)
from .spec_model import (
    ANCHOR_SOURCE_ID,
    CONFIDENCE_FROM_RULE,
    GROUP_REF_SEP,
    OP_CONSTANT,
    OP_CUSTOM_RULE,
    OP_DIRECT,
    OP_FALLBACK,
    OP_CONDITIONAL,
    OP_NOT_APPLICABLE,
    OP_NO_MAPPING_CONFIGURED,
    OP_UNIT_CONVERSION,
    OP_VALUE_MAP,
    HarmonizationSpec,
    Mapping as SpecMapping,
    RuleRef,
    SpecError,
    OP_FILE_MEMBERSHIP,
    VOCABULARY_MARKER,
    load as load_spec,
)
from .value_states import finalize_declared_value_state


_CURRENT_PROVENANCE_KEY = (
    "trial", "cimac_part_id", "Cimac.id", "Collection_Event",
)
_PROVENANCE_COLUMNS = frozenset({
    *_CURRENT_PROVENANCE_KEY,
    "harmonized_field", "value", "source_file", "source_column",
    "confidence",
    "extraction_method", "source_logical_label", "source_resolved_path",
    "source_sha256", "source_sheet", "source_row_number",
    "source_record_key", "rule_id", "rule_version", "contributions",
    "authoritative_for", "derivation", "value_state", "run_id", "spec_ref",
    "mapping_id", "source_ref_id", "raw_value_json", "evidence_refs",
})
_CIDC_READER_FLAGS = frozenset({
    "encoding", "header", "low_memory", "skiprows",
})
_CIDC_ENCODINGS = frozenset({"utf-8", "latin-1", "cp1252"})
_SUPPORTED_OPERATIONS = frozenset({
    OP_DIRECT, OP_VALUE_MAP, OP_UNIT_CONVERSION, OP_CUSTOM_RULE, OP_FALLBACK,
})
_MAX_REPLAY_DEPTH = 32

_OUTCOME_VALUE = "value"
_OUTCOME_NO_VALUE = "no_value"
_OUTCOME_UNMAPPED = "unmapped"
_OUTCOME_VOCABULARY = "vocabulary"
_OUTCOME_CONSTANT_VALUE = "constant_value"
_OUTCOME_CONSTANT_NULL = "constant_null"
_OUTCOME_DECLARED_ABSENCE = "declared_absence"
_OUTCOME_NO_MAPPING_CONFIGURED = "no_mapping_configured"
_DECLARED_ABSENCE_STATES = frozenset({"not_applicable", "source_missing"})


@dataclass(frozen=True)
class ReplayRequest:
    """Explicit selector for one delivered cell."""

    artifact_id: str
    row_key: Mapping[str, Any]
    target_field: str
    api_version: int = REPLAY_API_VERSION


@dataclass(frozen=True)
class ReplayResult:
    """One version-1 replay outcome."""

    status: ReplayStatus
    reason_code: str
    message: str
    artifact_id: str = ""
    target_field: str = ""
    spec_ref: str = ""
    mapping_id: str = ""
    source_ref_id: str = ""
    delivered_value: Any = None
    recomputed_value: Any = None
    api_version: int = REPLAY_API_VERSION
    status_version: int = REPLAY_STATUS_VERSION
    reason_codes_version: int = REASON_CODES_VERSION

    def to_dict(self) -> dict[str, Any]:
        return {
            "api_version": self.api_version,
            "status_version": self.status_version,
            "reason_codes_version": self.reason_codes_version,
            "status": self.status.value,
            "reason_code": self.reason_code,
            "message": self.message,
            "artifact_id": self.artifact_id,
            "target_field": self.target_field,
            "spec_ref": self.spec_ref,
            "mapping_id": self.mapping_id,
            "source_ref_id": self.source_ref_id,
            "delivered_value": _json_scalar(self.delivered_value),
            "recomputed_value": _json_scalar(self.recomputed_value),
        }


class _Stop(Exception):
    def __init__(self, status: ReplayStatus, reason_code: str, message: str, *,
                 spec_ref: str = "", mapping_id: str = "",
                 source_ref_id: str = "", delivered_value: Any = None,
                 recomputed_value: Any = None):
        super().__init__(message)
        self.status = status
        self.reason_code = reason_code
        self.message = message
        self.spec_ref = spec_ref
        self.mapping_id = mapping_id
        self.source_ref_id = source_ref_id
        self.delivered_value = delivered_value
        self.recomputed_value = recomputed_value


class _CsvError(ValueError):
    pass


class _ValueMapUnmappedError(ValueError):
    pass


class _NonFiniteUnitConversionResult(ValueError):
    pass


class _NonFiniteFallbackResult(ValueError):
    pass


@dataclass(frozen=True)
class RecomputedOutcome:
    """A scalar result plus the mapping branch selected by source evidence."""

    value: Any
    branch: str
    rule: RuleRef | None = None
    confidence: float | None = None
    vocabulary_rule_index: int | None = None
    semantic_state: str = ""


@dataclass(frozen=True)
class RecomputedFallback:
    """A fallback result derived only from freshly verified reached scalars."""

    selected_rank: int | None
    raw_selected: Any
    value: Any
    branch: str


@dataclass(frozen=True)
class _VerifiedSource:
    binding: SourceBinding
    declaration: Any
    frame: Any


@dataclass(frozen=True)
class _VerifiedParticipantRow:
    binding: SourceBinding
    declaration: Any
    row: Any
    row_number: int


@dataclass(frozen=True)
class _VerifiedNamedRead:
    read_id: str
    source_id: str
    source_ref_id: str
    source_column: str
    row_number: int
    raw_source_value: Any
    value: Any
    selected_member_id: str = ""


@dataclass(frozen=True)
class _VerifiedGroupDecision:
    """One Stage 12 decision, retained for all Stage 16 fact consumers."""

    group: Any
    selected_member: Any
    selected_rank: int
    selected_deciding_binding: SourceBinding
    deciding_source_ref_ids: tuple[str, ...]


@dataclass(frozen=True)
class _VerifiedDerivationFacts:
    """Causal Stage 16 facts reconstructed before operation-value insertion."""

    facts: dict[str, Any]
    conditions: dict[str, bool]
    reads: dict[str, _VerifiedNamedRead]
    groups: dict[str, _VerifiedGroupDecision]
    selected_record: dict | None


@dataclass(frozen=True)
class _VerifiedAnchorScalar:
    """Causal anchor facts rebuilt without consulting produced values."""

    evidence_ref_id: str
    route: str
    record_key: str
    anchor_column: str
    raw_value: Any
    source_ref_id: str
    selected_source_row: int | None
    selected_filtered_position: int | None
    synthesized_declaration: str
    construction_source_ref_ids: tuple[str, ...]


@dataclass(frozen=True)
class _VerifiedConstructedRecord:
    """One independently reconstructed logical record and its projections."""

    row_identity: tuple[tuple[str, str], ...]
    origin: str
    anchor_items: tuple[tuple[str, Any], ...]
    group_key: tuple[tuple[str, str], ...] = ()
    selected_source_row: int | None = None
    group_size: int | None = None
    declaration_id: str = ""

    @property
    def anchor_row(self) -> dict[str, Any]:
        return dict(self.anchor_items)


@dataclass(frozen=True)
class _VerifiedConstructionUniverse:
    """Cached verified construction; event-local usage is checked separately."""

    evidence_ref_id: str
    records: tuple[_VerifiedConstructedRecord, ...]
    source_ref_ids: tuple[str, ...]
    source_bindings: tuple[tuple[str, SourceBinding], ...]


@dataclass(frozen=True)
class _VerifiedGroupScalar:
    """Source-group facts rebuilt without consulting recorded decisions."""

    group_id: str
    selected_member_id: str
    selected_rank: int
    source_ref_id: str
    source_row_number: int
    source_column: str
    raw_value: Any
    deciding_source_ref_ids: tuple[str, ...]
    selected_member_value: Any
    selected_deciding_source_ref_id: str
    selected_deciding_logical_label: str

@dataclass(frozen=True)
class _EventOutcome:
    value: Any
    spec_ref: str
    mapping_id: str
    source_ref_id: str = ""


def _route_result(value: Any, raw_value: Any, *, exhausted: bool = False
                  ) -> VerifiedOperationRouteResult:
    """Build the small finalized-raw input contract from verified route facts."""
    return VerifiedOperationRouteResult(
        value=value,
        raw_candidate=VerifiedRawCandidate(True, raw_value),
        completion=(COMPLETION_FALLBACK_EXHAUSTED
                    if exhausted else COMPLETION_VALUE),
    )


def _verify_finalized_raw_snapshot(
        event: Mapping[str, Any], finalized: FinalizedRaw, *,
        spec_ref: str, mapping_id: str, source_ref_id: str = "") -> None:
    """Compare independently finalized raw with the typed Cell snapshot."""
    context = {"spec_ref": spec_ref, "mapping_id": mapping_id,
               "source_ref_id": source_ref_id}
    payload = str(event.get("raw_value_json") or "")
    if not finalized.record_snapshot:
        if payload:
            _fail(
                "RAW_VALUE_MISMATCH",
                "typed raw snapshot exists although finalized raw is omitted",
                **context)
        return
    if not payload:
        _not_replayable(
            "RAW_VALUE_ENCODING_MISSING",
            "required finalized typed raw snapshot is absent", **context)
    try:
        recorded = TypedRawValue.from_json(payload)
        expected = encode_typed_raw(finalized.value)
    except ValueError as exc:
        _not_replayable("RAW_VALUE_KIND_UNSUPPORTED", str(exc), **context)
    if recorded != expected:
        _fail(
            "RAW_VALUE_MISMATCH",
            "typed raw snapshot contradicts independently finalized raw",
            **context)


def recompute_registered_rule(rule: rule_registry.RegisteredRule,
                              replayed_inputs: list[Any], *, params: dict[str, Any]
                              ) -> tuple[Any, float | None, str]:
    """Invoke one pinned registry implementation using replayed inputs only.

    Delivered and provenance output values are deliberately absent from this
    trust-boundary signature.
    """
    return rule_registry.invoke(rule, replayed_inputs, dict(params))


def recompute_constant(*, declared_value: Any) -> RecomputedOutcome:
    """Recompute one pinned scalar constant without consulting run outputs.

    Only built-in YAML scalar types cross this trust boundary.  In particular,
    ``bool`` is checked before ``int`` so its declared type is preserved rather
    than silently treated as Python's integer subclass.  Delivered values,
    provenance values, and recorded raw snapshots are deliberately absent from
    this signature.
    """
    if declared_value is None:
        return RecomputedOutcome(None, _OUTCOME_CONSTANT_NULL)
    if type(declared_value) in (str, bool, int):
        return RecomputedOutcome(declared_value, _OUTCOME_CONSTANT_VALUE)
    if type(declared_value) is float and math.isfinite(declared_value):
        return RecomputedOutcome(declared_value, _OUTCOME_CONSTANT_VALUE)
    raise ValueError(
        "constant literal must be null, string, boolean, integer, or finite float")


def recompute_no_mapping_configured() -> RecomputedOutcome:
    """Recompute pinned no-mapping semantics without any value input."""
    return RecomputedOutcome(None, _OUTCOME_NO_MAPPING_CONFIGURED)


def recompute_declared_absence(*, declared_state: str) -> RecomputedOutcome:
    """Recompute one source-independent absence from its pinned declaration.

    The output is always null, but the two approved semantic states remain
    distinct in the outcome.  Delivered/provenance values, serialized Cell
    state, raw evidence, and source evidence are deliberately absent from this
    trust-boundary signature.
    """
    if declared_state not in _DECLARED_ABSENCE_STATES:
        raise ValueError(
            "declared absence state must be 'not_applicable' or "
            "'source_missing'")
    return RecomputedOutcome(
        None, _OUTCOME_DECLARED_ABSENCE, semantic_state=declared_state)


def recompute_direct(raw_source_value: Any, *, coerce: str) -> Any:
    """Recompute an approved direct mapping from freshly read source evidence.

    This function intentionally has no delivered-value or provenance-output
    parameter.  Its closed coercion vocabulary is the one declared by the
    specification model.
    """
    if coerce == "none":
        # Pandas represents an empty CSV scalar as NaN.  At the mapping
        # boundary that is the schema's null value, not a numeric NaN result.
        return None if _is_missing(raw_source_value) else raw_source_value
    if coerce == "text":
        if _is_missing(raw_source_value):
            return None
        return re.sub(r"\s+", " ", str(raw_source_value)).strip()
    if coerce == "numeric":
        if _is_missing(raw_source_value):
            return None
        key = re.sub(r"\s+", " ", str(raw_source_value)).strip()
        if key in ("", "."):
            return None
        try:
            return float(key)
        except (TypeError, ValueError):
            return None
    raise ValueError(f"unsupported direct coercion {coerce!r}")

def recompute_file_membership(selected_member_id: str, *,
                              member_values: Mapping[str, Any]) -> Any:
    """Return the pinned literal for an independently selected group member.

    The selected member is the only runtime fact accepted here.  Recorded
    Contributions, provenance output, and delivery cannot cross this boundary.
    """
    if selected_member_id not in member_values:
        raise ValueError(
            f"selected member {selected_member_id!r} has no pinned value")
    return member_values[selected_member_id]



def recompute_value_map(raw_source_value: Any, *, mapping_table: Mapping,
                        on_unmapped: str) -> RecomputedOutcome:
    """Recompute one inline value map without seeing any output value."""
    if _value_map_is_absent(raw_source_value):
        return RecomputedOutcome(None, _OUTCOME_NO_VALUE)
    key = _normalize_key(raw_source_value)
    if key in mapping_table:
        return RecomputedOutcome(mapping_table[key], _OUTCOME_VALUE)
    if on_unmapped == "error":
        raise _ValueMapUnmappedError(
            f"inline value map has no entry for {key!r}")
    if on_unmapped == "passthrough":
        return RecomputedOutcome(raw_source_value, _OUTCOME_UNMAPPED)
    if on_unmapped == "null":
        return RecomputedOutcome(None, _OUTCOME_UNMAPPED)
    raise ValueError(f"unsupported on_unmapped behavior {on_unmapped!r}")


def recompute_vocabulary_value_map(
        raw_source_value: Any, *, mapping_table: Mapping,
        ordered_rules: tuple[vocabularies.MatchRule, ...], on_unmapped: str
        ) -> RecomputedOutcome:
    """Recompute inline-first, ordered-vocabulary value-map semantics.

    Only verified source input, the pinned inline declaration, and terms parsed
    from the exact verified vocabulary binding cross this trust boundary.
    Delivered and provenance outputs are deliberately absent.
    """
    if _value_map_is_absent(raw_source_value):
        return RecomputedOutcome(None, _OUTCOME_NO_VALUE)
    key = _normalize_key(raw_source_value)
    if key in mapping_table:
        return RecomputedOutcome(mapping_table[key], _OUTCOME_VALUE)
    hit = vocabularies.match(raw_source_value, ordered_rules)
    if hit is not None:
        index = ordered_rules.index(hit)
        value = hit.value if hit.value is not None else raw_source_value
        return RecomputedOutcome(
            value, _OUTCOME_VOCABULARY,
            rule=RuleRef(hit.rule_id, hit.rule_version),
            confidence=hit.confidence,
            vocabulary_rule_index=index,
        )
    if on_unmapped == "error":
        raise _ValueMapUnmappedError(
            f"inline map and ordered vocabulary have no entry for {key!r}")
    if on_unmapped == "passthrough":
        return RecomputedOutcome(raw_source_value, _OUTCOME_UNMAPPED)
    if on_unmapped == "null":
        return RecomputedOutcome(None, _OUTCOME_UNMAPPED)
    raise ValueError(f"unsupported on_unmapped behavior {on_unmapped!r}")


def recompute_unit_conversion(raw_source_value: Any, *, factor: float,
                              rounding: bool) -> RecomputedOutcome:
    """Recompute the executor's simple float multiplication operation."""
    if (isinstance(factor, bool) or not isinstance(factor, (int, float))
            or not math.isfinite(float(factor)) or float(factor) == 0.0):
        raise ValueError("unit-conversion factor must be finite and non-zero")
    key = _normalize_key(raw_source_value)
    if key is None:
        return RecomputedOutcome(None, _OUTCOME_NO_VALUE)
    try:
        converted = float(key) * float(factor)
    except (TypeError, ValueError, OverflowError):
        return RecomputedOutcome(None, _OUTCOME_NO_VALUE)
    if not math.isfinite(converted):
        raise _NonFiniteUnitConversionResult(
            "unit conversion produced a non-finite result")
    value = float(round(converted)) if rounding else converted
    if not math.isfinite(value):
        raise _NonFiniteUnitConversionResult(
            "unit conversion produced a non-finite rounded result")
    return RecomputedOutcome(value, _OUTCOME_VALUE)


def recompute_fallback(reached_candidate_scalars: list[Any], *, coerce: str
                       ) -> RecomputedFallback:
    """Recompute first-executor-present selection, then optional coercion.

    Delivered and provenance output values are deliberately absent.  The input
    list contains only candidate scalars whose ranks selection actually
    reached; later candidates used solely for source-presence eligibility are
    not accepted by this trust-boundary function.
    """
    selection = select_ordered_fallback(reached_candidate_scalars)
    if selection.exhausted:
        return RecomputedFallback(None, None, None, _OUTCOME_NO_VALUE)
    selected_rank = selection.selected_rank
    raw_selected = selection.raw_selected
    if coerce == "none":
        value = raw_selected
    elif coerce == "numeric":
        value = recompute_direct(raw_selected, coerce="numeric")
        if isinstance(value, float) and not math.isfinite(value):
            raise _NonFiniteFallbackResult(
                "fallback numeric coercion produced a non-finite result")
    else:
        raise ValueError(f"unsupported fallback coercion {coerce!r}")
    branch = (_OUTCOME_NO_VALUE if _fallback_raw_is_missing(value)
              else _OUTCOME_VALUE)
    return RecomputedFallback(selected_rank, raw_selected, value, branch)


def replay_cell(project_root: str | Path,
                manifest: ReplayManifest | str | Path,
                request: ReplayRequest) -> ReplayResult:
    """Strictly replay one supported cell.

    A manifest path, when used, must be explicit and project-contained.  No
    basename lookup, recursive search, or run discovery is performed.
    """
    artifact_id = request.artifact_id if isinstance(request, ReplayRequest) else ""
    target_field = request.target_field if isinstance(request, ReplayRequest) else ""
    try:
        return _replay_cell(Path(project_root), manifest, request)
    except _Stop as stop:
        return ReplayResult(
            status=stop.status,
            reason_code=stop.reason_code,
            message=stop.message,
            artifact_id=artifact_id,
            target_field=target_field,
            spec_ref=stop.spec_ref,
            mapping_id=stop.mapping_id,
            source_ref_id=stop.source_ref_id,
            delivered_value=stop.delivered_value,
            recomputed_value=stop.recomputed_value,
        )


def _replay_cell(root: Path, manifest_input, request: ReplayRequest) -> ReplayResult:
    if not isinstance(request, ReplayRequest):
        _request_error("REQUEST_MALFORMED", "request must be a ReplayRequest")
    if request.api_version != REPLAY_API_VERSION:
        _request_error("REQUEST_MALFORMED",
                       f"replay API version {request.api_version!r} is unsupported; "
                       f"expected {REPLAY_API_VERSION}")
    if not isinstance(request.artifact_id, str) or not request.artifact_id:
        _request_error("REQUEST_MALFORMED", "artifact_id must be a non-empty string")
    if not isinstance(request.target_field, str) or not request.target_field:
        _request_error("REQUEST_MALFORMED", "target_field must be a non-empty string")
    if not isinstance(request.row_key, Mapping):
        _request_error("REQUEST_MALFORMED", "row_key must be a mapping")

    session = _ReplaySession.open(Path(root), manifest_input, request)
    return session.replay_root(request)


class _ReplaySession:
    """One validated replay request with recursive event state and memoization."""

    def __init__(self, root: Path, manifest: ReplayManifest,
                 artifact: ArtifactBinding, schema: TargetSchema,
                 delivered_rows: list[dict[str, str]],
                 provenance_rows: list[dict[str, str]]):
        self.root = root
        self.manifest = manifest
        self.artifact = artifact
        self.schema = schema
        self.delivered_rows = delivered_rows
        self.provenance_rows = provenance_rows
        self.delivered_row: dict[str, str] = {}
        self.spec: HarmonizationSpec | None = None
        self.row_identity: dict[str, str] = {}
        self.event_index: dict[str, list[dict[str, str]]] = {}
        self.memo: dict[tuple[str, str], _EventOutcome | _Stop] = {}
        self.source_cache: dict[str, _VerifiedSource] = {}
        self.construction_cache: dict[str, _VerifiedConstructionUniverse] = {}

    @classmethod
    def open(cls, root: Path, manifest_input, request: ReplayRequest):
        root = root.resolve()
        if not root.is_dir():
            _request_error("REQUEST_MALFORMED", "project_root is not a directory")
        manifest = _validated_manifest(root, manifest_input)
        artifacts = [a for a in manifest.delivered_artifacts
                     if a.artifact_id == request.artifact_id]
        if not artifacts:
            _request_error("ARTIFACT_ID_UNKNOWN",
                           f"artifact_id {request.artifact_id!r} is not in the manifest")
        artifact = artifacts[0]
        if artifact.format != "csv":
            _not_replayable("OPERATION_UNSUPPORTED",
                            "strict scalar replay supports delivered CSV artifacts only")
        if manifest.provenance_artifact.format != "csv":
            _not_replayable("PROVENANCE_SCHEMA_UNSUPPORTED",
                            "strict replay supports provenance CSV artifacts only")
        if (manifest.provenance_artifact.provenance_schema_version
                != PROVENANCE_SCHEMA_VERSION):
            _not_replayable(
                "PROVENANCE_SCHEMA_UNSUPPORTED",
                f"provenance schema version "
                f"{manifest.provenance_artifact.provenance_schema_version!r} is "
                f"unsupported; expected {PROVENANCE_SCHEMA_VERSION}")
        _verify_hash(root, artifact, "DELIVERED_ARTIFACT_HASH_MISMATCH")
        _verify_hash(root, manifest.provenance_artifact,
                     "PROVENANCE_ARTIFACT_HASH_MISMATCH")
        _verify_hash(root, manifest.target_schema, "SCHEMA_HASH_MISMATCH")
        try:
            schema = load_schema(resolve_project_path(root, manifest.target_schema.path))
        except SchemaError as exc:
            _not_replayable("MANIFEST_REQUIRED_FIELD_MISSING",
                            f"pinned target schema cannot be loaded: {exc}")
        if (schema.ref != manifest.target_schema.ref
                or manifest.target_schema.identity != f"schema:{schema.ref}"):
            _fail("SCHEMA_HASH_MISMATCH",
                  "target-schema identity disagrees with the pinned schema file")
        if artifact.schema_ref != schema.ref:
            _fail("SCHEMA_HASH_MISMATCH",
                  "delivered artifact is not bound to the pinned target schema")
        artifact_name = Path(artifact.path).name
        if schema.artifacts and artifact_name not in schema.artifacts:
            _request_error("FIELD_NOT_APPLICABLE",
                           f"schema {schema.ref} does not apply to {artifact_name!r}")
        try:
            delivered_rows = _read_csv(resolve_project_path(root, artifact.path))
        except _CsvError as exc:
            _request_error("REQUEST_MALFORMED", str(exc))
        try:
            provenance_rows = _read_csv(
                resolve_project_path(root, manifest.provenance_artifact.path),
                required=_PROVENANCE_COLUMNS)
        except _CsvError as exc:
            _not_replayable("PROVENANCE_SCHEMA_UNSUPPORTED", str(exc))
        return cls(root, manifest, artifact, schema, delivered_rows, provenance_rows)

    def replay_root(self, request: ReplayRequest) -> ReplayResult:
        artifact_name = Path(self.artifact.path).name
        field = self.schema.field(request.target_field)
        if field is None:
            _request_error("FIELD_NOT_DECLARED",
                           f"field {request.target_field!r} is not declared")
        if not field.applies_to(artifact_name):
            _request_error("FIELD_NOT_APPLICABLE",
                           f"field {request.target_field!r} does not apply")
        selector = _validate_selector(request.row_key, self.schema)
        selected = _select_delivered(self.delivered_rows, selector, self.schema)
        if not selected:
            _request_error("TARGET_NOT_FOUND", "no delivered row matches the row key")
        if len(selected) != 1:
            _request_error("TARGET_AMBIGUOUS",
                           f"{len(selected)} delivered rows match the row key")
        self.delivered_row = selected[0]
        if request.target_field not in self.delivered_row:
            _request_error("FIELD_NOT_DECLARED",
                           f"delivered artifact has no {request.target_field!r} column")
        event, spec, _binding = _select_provenance(
            self.root, self.manifest, self.schema, self.provenance_rows,
            self.delivered_row, request.target_field, self.artifact)
        self.spec = spec
        if not event.get("run_id"):
            _not_replayable("RUN_ID_LINK_MISSING", "provenance run_id is absent",
                            spec_ref=spec.ref)
        if not event.get("mapping_id"):
            _not_replayable("MAPPING_NOT_IDENTIFIED",
                            "provenance mapping_id is absent", spec_ref=spec.ref)
        try:
            self.row_identity = canonical_row_identity(
                self.schema.row_key, self.delivered_row)
        except ValueError as exc:
            _not_replayable("ROW_KEY_NOT_REPRESENTABLE_IN_PROVENANCE", str(exc),
                            spec_ref=spec.ref)
        self._build_event_index()
        root_ref = self._reference_for(event, self.row_identity)
        outcome = self._replay_event(
            event, root_ref, request.target_field, self.row_identity,
            depth=0, active_stack=(root_ref,))
        delivered = _result_value(
            self.delivered_row[request.target_field], field, self.schema)
        operation = next(m.op for m in spec.mappings
                         if m.mapping_id == outcome.mapping_id)
        return ReplayResult(
            status=ReplayStatus.PASS,
            reason_code="REPLAY_MATCH",
            message=f"verified {operation} recomputation matches delivery",
            artifact_id=request.artifact_id,
            target_field=request.target_field,
            spec_ref=outcome.spec_ref,
            mapping_id=outcome.mapping_id,
            source_ref_id=outcome.source_ref_id,
            delivered_value=delivered,
            recomputed_value=outcome.value,
        )

    def _build_event_index(self) -> None:
        assert self.spec is not None
        for event in self.provenance_rows:
            if (event.get("spec_ref") != self.spec.ref
                    or not event.get("mapping_id")
                    or not event.get("harmonized_field")):
                continue
            try:
                identity = _row_identity_from_event(
                    self.spec, self.schema, event)
                reference = self._reference_for(event, identity)
            except ValueError:
                continue
            self.event_index.setdefault(reference, []).append(event)

    def _reference_for(self, event: dict[str, str], row_identity) -> str:
        return provenance_event_ref(
            run_id=event.get("run_id", ""),
            spec_ref=event.get("spec_ref", ""),
            mapping_id=event.get("mapping_id", ""),
            row_identity=row_identity,
            harmonized_field=event.get("harmonized_field", ""),
        )

    def replay_event(self, provenance_ref: str, expected_field: str,
                     expected_row_identity: dict[str, str], depth: int,
                     active_stack: tuple[str, ...]) -> _EventOutcome:
        if depth > _MAX_REPLAY_DEPTH:
            _not_replayable("REPLAY_DEPTH_EXCEEDED",
                            f"dependency replay exceeded depth {_MAX_REPLAY_DEPTH}")
        if not is_provenance_event_ref(provenance_ref):
            _not_replayable("DEPENDENCY_REFERENCE_MISSING",
                            "dependency provenance_ref is absent or malformed")
        if provenance_ref in active_stack:
            _not_replayable("REPLAY_CYCLE_DETECTED",
                            f"dependency cycle reaches {provenance_ref!r}")
        matches = self.event_index.get(provenance_ref, ())
        if not matches:
            _not_replayable("DEPENDENCY_REFERENCE_MISSING",
                            f"dependency reference {provenance_ref!r} is unresolved")
        if len(matches) != 1:
            _not_replayable("DEPENDENCY_REFERENCE_AMBIGUOUS",
                            f"dependency reference {provenance_ref!r} resolves to "
                            f"{len(matches)} events")
        event = matches[0]
        assert self.spec is not None
        try:
            identity = _row_identity_from_event(self.spec, self.schema, event)
            expected_anchors = _provenance_identity(
                self.spec, self.delivered_row)
        except ValueError as exc:
            _fail("DEPENDENCY_CONTEXT_MISMATCH", str(exc))
        actual_anchors = {name: event.get(name, "")
                          for name in _CURRENT_PROVENANCE_KEY}
        if (event.get("run_id") != self.manifest.run_id
                or event.get("spec_ref") != self.spec.ref
                or event.get("trial") != self.spec.trial
                or identity != expected_row_identity
                or actual_anchors != expected_anchors
                or event.get("harmonized_field") != expected_field
                or not _is_authoritative(event.get("authoritative_for", ""),
                                         self.artifact)):
            _fail("DEPENDENCY_CONTEXT_MISMATCH",
                  f"dependency {provenance_ref!r} contradicts its expected "
                  "run/spec/row/field/artifact context")
        key = (self.artifact.artifact_id, provenance_ref)
        cached = self.memo.get(key)
        if cached is not None:
            if isinstance(cached, _Stop):
                raise cached
            return cached
        try:
            outcome = self._replay_event(
                event, provenance_ref, expected_field, expected_row_identity,
                depth=depth, active_stack=active_stack + (provenance_ref,))
        except _Stop as stop:
            if stop.status is ReplayStatus.REQUEST_ERROR:
                stop = _Stop(
                    ReplayStatus.NOT_REPLAYABLE, "DEPENDENCY_NOT_REPLAYABLE",
                    f"internal dependency request failed: {stop.reason_code}: "
                    f"{stop.message}", spec_ref=stop.spec_ref,
                    mapping_id=stop.mapping_id,
                    source_ref_id=stop.source_ref_id)
            self.memo[key] = stop
            raise stop
        self.memo[key] = outcome
        return outcome

    def _verify_stage15_mapping_branch(
            self, mapping: SpecMapping, event, usage, *,
            constructed_record: _VerifiedConstructedRecord | None = None):
        """Prove same-target mapping eligibility without trusting mapping_id."""
        assert self.spec is not None
        spec = self.spec
        context = {"spec_ref": spec.ref, "mapping_id": mapping.mapping_id}
        if ((mapping.derivation or event.get("derivation"))
                and not (_is_stage16_derivation_mapping(mapping, spec)
                         or _is_stage17_derivation_mapping(mapping, spec)
                         or _is_stage18_derivation_mapping(mapping, spec)
                         or _is_stage19_derivation_mapping(mapping, spec))):
            _not_replayable(
                "DERIVATION_UNSUPPORTED_STAGE2",
                "conditional derivation is outside Stage 16 Option C", **context)
        contributions = self._stage15_contributions(
            event, mapping.mapping_id)
        conditions = {condition.id: condition for condition in spec.conditions}
        reads_cache = {}
        groups_cache = {}
        condition_results = {}

        siblings = [
            sibling for sibling in spec.mappings
            if sibling.target == mapping.target
            and (not sibling.artifacts or {
                self.artifact.artifact_id, Path(self.artifact.path).name
            }.intersection(sibling.artifacts))
        ]
        results = {}
        for sibling in siblings:
            if not sibling.when:
                results[sibling.mapping_id] = True
                continue
            condition = conditions.get(sibling.when)
            if condition is None:
                _not_replayable(
                    "NAMED_READ_EVIDENCE_MISSING",
                    f"sibling condition {sibling.when!r} is not pinned",
                    **context)
            if condition.read:
                if (condition.read not in mapping.contributions.conditions
                        and sum(
                            item.get("role") == condition.read
                            for item in contributions) != 1):
                    _not_replayable(
                        "NAMED_READ_EVIDENCE_MISSING",
                        f"sibling condition read {condition.read!r} lacks "
                        "usage-scoped Contribution evidence", **context)
                if condition.read not in reads_cache:
                    reads_cache[condition.read] = self._stage15_named_read(
                        condition.read, mapping, event, usage, contributions,
                        groups_cache=groups_cache)
                operand = reads_cache[condition.read].value
                condition_result = evaluate_predicate(
                    condition.predicate, operand, condition.value)
                condition_results[condition.id] = condition_result
                results[sibling.mapping_id] = condition_result
            elif condition.group:
                if condition.group not in groups_cache:
                    groups_cache[condition.group] = self._stage15_group_member(
                        condition.group, event, usage, mapping.mapping_id,
                        contributions)
                condition_result = (
                    groups_cache[condition.group].selected_member.id
                    == condition.member)
                condition_results[condition.id] = condition_result
                results[sibling.mapping_id] = condition_result
            elif condition.record_origin:
                if constructed_record is None:
                    _not_replayable(
                        "ANCHOR_EVIDENCE_MISSING",
                        "record-origin condition lacks verified construction",
                        **context)
                condition_result = (
                    condition.record_origin == constructed_record.origin)
                condition_results[condition.id] = condition_result
                results[sibling.mapping_id] = condition_result
            else:
                _not_replayable(
                    "MAPPING_CONDITION_UNSUPPORTED",
                    "record-origin branch authority is outside Stage 15",
                    **context)

        true_ids = [
            sibling.mapping_id for sibling in siblings
            if results.get(sibling.mapping_id, False)
        ]
        if len(true_ids) > 1:
            _not_replayable(
                "MULTI_EVENT_BRANCH_UNSUPPORTED",
                "multiple same-target mappings independently evaluate true; "
                "field replay cannot reconcile multiple valid events",
                **context)
        if results.get(mapping.mapping_id) is not True:
            _fail(
                "BRANCH_SELECTION_MISMATCH",
                "recorded mapping independently evaluates false", **context)
        if true_ids != [mapping.mapping_id]:
            _fail(
                "BRANCH_SELECTION_MISMATCH",
                "a different sole same-target mapping evaluates true", **context)

        condition_roles = set(mapping.contributions.conditions)
        operation_contributions = [
            item for item in contributions
            if item.get("role") not in condition_roles
        ]
        operation_event = dict(event)
        operation_event["contributions"] = (
            json.dumps(
                operation_contributions, separators=(",", ":"), sort_keys=True)
            if operation_contributions else "")
        operation_mapping = dc_replace(
            mapping, when="",
            contributions=dc_replace(
                mapping.contributions, conditions=()))
        operation_usage = usage
        if constructed_record is not None:
            operation_event["evidence_refs"] = ";".join(
                ref for ref in event.get("evidence_refs", "").split(";")
                if ref and not ref.startswith("anchor-"))
            operation_usage = dc_replace(
                operation_usage,
                evidence_ref_ids=tuple(
                    ref for ref in operation_usage.evidence_ref_ids
                    if not ref.startswith("anchor-")))
        if mapping.op not in (OP_FALLBACK, OP_CONDITIONAL):
            if event.get("source_ref_id"):
                operation_usage = dc_replace(
                    operation_usage,
                    source_ref_ids=(event["source_ref_id"],))
            elif not mapping.source:
                operation_usage = dc_replace(
                    operation_usage, source_ref_ids=())
        return (operation_mapping, operation_event, operation_usage,
                reads_cache, groups_cache, condition_results)

    def _replay_event(self, event: dict[str, str], event_ref: str,
                      target: str, row_identity: dict[str, str], *, depth: int,
                      active_stack: tuple[str, ...]) -> _EventOutcome:
        assert self.spec is not None
        spec = self.spec
        context = {"spec_ref": spec.ref}
        if not event.get("run_id"):
            _not_replayable("RUN_ID_LINK_MISSING", "provenance run_id is absent",
                            **context)
        if event["run_id"] != self.manifest.run_id:
            _not_replayable("RUN_ID_MISMATCH",
                            "provenance run_id does not match manifest", **context)
        if event.get("spec_ref") != spec.ref:
            _not_replayable("SPEC_LINK_MISSING",
                            "provenance spec_ref does not identify pinned spec",
                            **context)
        mapping_id = event.get("mapping_id", "")
        mappings = [m for m in spec.mappings if m.mapping_id == mapping_id]
        if len(mappings) != 1:
            _not_replayable("MAPPING_NOT_IDENTIFIED",
                            f"mapping_id {mapping_id!r} is not uniquely pinned",
                            spec_ref=spec.ref, mapping_id=mapping_id)
        mapping = mappings[0]
        declared_mapping = mapping
        if mapping.target != target:
            _fail("DEPENDENCY_CONTEXT_MISMATCH",
                  "mapping identity names a different target field",
                  spec_ref=spec.ref, mapping_id=mapping_id)
        usages = [u for u in self.manifest.mapping_usages
                  if u.spec_ref == spec.ref and u.mapping_id == mapping_id]
        if len(usages) != 1:
            _not_replayable("MAPPING_NOT_IDENTIFIED",
                            "manifest has no exact spec_ref + mapping_id usage",
                            spec_ref=spec.ref, mapping_id=mapping_id)
        usage = usages[0]
        _verify_usage_files(self.root, self.manifest, usage, self.artifact)
        branch_reads = {}
        branch_groups = {}
        branch_conditions = {}
        original_event = event
        original_usage = usage
        constructed_record = None
        construction_universe = None
        if mapping_uses_record_construction(spec, mapping):
            constructed_record, construction_universe = \
                self._verify_constructed_record(
                    mapping, event, usage, row_identity)
        if mapping.when:
            (mapping, event, usage, branch_reads, branch_groups,
             branch_conditions) = self._verify_stage15_mapping_branch(
                mapping, event, usage,
                constructed_record=constructed_record)
        stage16 = (
            _is_stage16_derivation_mapping(declared_mapping, spec)
            and not _is_stage14_derivation_mapping(declared_mapping))
        stage17 = _is_stage17_derivation_mapping(declared_mapping, spec)
        stage18 = _is_stage18_derivation_mapping(declared_mapping, spec)
        stage19 = _is_stage19_derivation_mapping(declared_mapping, spec)
        fact_stage = stage16 or stage17 or stage18 or stage19
        verified_derivation = None
        verified_group_scalar = None
        route_result = None
        finalized_raw = ...
        effective_source_column = declared_mapping.column or declared_mapping.target

        if (stage16 and (
                declared_mapping.op == OP_CONSTANT
                or source_group_for_mapping(spec, declared_mapping) is None)):
            verified_derivation = self._stage16_derivation_facts(
                declared_mapping, original_event, original_usage,
                reads_cache=branch_reads, groups_cache=branch_groups,
                conditions_cache=branch_conditions)

        if stage16 and declared_mapping.op == OP_CONSTANT:
            recomputed, source_ref_id, effective_source_column = \
                self._replay_stage16_constant(
                    declared_mapping, original_event, original_usage,
                    verified_derivation)
        elif mapping.op == OP_CUSTOM_RULE:
            recomputed = self._replay_custom(
                mapping, event, usage, row_identity, depth, active_stack)
            source_ref_id = ""
        elif mapping.op == OP_CONSTANT:
            route_result = self._replay_constant(mapping, event, usage)
            recomputed = route_result.value
            source_ref_id = ""
        elif mapping.op == OP_NOT_APPLICABLE:
            route_result, source_ref_id = self._replay_declared_absence(
                mapping, event, usage)
            recomputed = route_result.value
        elif mapping.op == OP_NO_MAPPING_CONFIGURED:
            recomputed, source_ref_id = self._replay_no_mapping_configured(
                mapping, event, usage)
        elif mapping.op == OP_CONDITIONAL:
            (route_result, source_ref_id, conditional_reads,
             conditional_groups) = self._replay_conditional(
                mapping, event, usage, reads_cache=branch_reads,
                groups_cache=branch_groups)
            recomputed = route_result.value
            branch_reads.update(conditional_reads)
            branch_groups.update(conditional_groups)
        elif ((fact_stage or declared_mapping.provenance.raw_from)
              and source_group_for_mapping(spec, declared_mapping)):
            group = source_group_for_mapping(spec, declared_mapping)
            role = membership_role(spec, group)
            contributions = self._stage15_contributions(
                original_event, declared_mapping.mapping_id)
            operation_event = dict(original_event)
            membership = [item for item in contributions
                          if item.get("role") == role]
            operation_event["contributions"] = json.dumps(
                membership, separators=(",", ":"), sort_keys=True)
            operation_event["derivation"] = ""
            operation_mapping = dc_replace(
                mapping, derivation=(),
                contributions=dc_replace(
                    mapping.contributions, conditions=()),
                provenance=dc_replace(
                    mapping.provenance, raw_from=""))
            route_result, source_ref_id, verified_group_scalar = \
                self._replay_source_group_operation(
                    operation_mapping, operation_event, original_usage,
                    exact_usage=False,
                    verify_file_derivation=not fact_stage)
            recomputed = route_result.value
            effective_source_column = verified_group_scalar.source_column
            selected_binding = next(
                item for item in self.manifest.sources
                if item.source_ref_id
                   == verified_group_scalar.selected_deciding_source_ref_id)
            selected_member = next(
                item for item in group.members
                if item.id == verified_group_scalar.selected_member_id)
            branch_groups.setdefault(
                group.id, _VerifiedGroupDecision(
                    group=group, selected_member=selected_member,
                    selected_rank=verified_group_scalar.selected_rank,
                    selected_deciding_binding=selected_binding,
                    deciding_source_ref_ids=(
                        verified_group_scalar.deciding_source_ref_ids)))
        elif is_closed_source_group_mapping(spec, mapping):
            route_result, source_ref_id, _group_scalar = \
                self._replay_source_group_operation(mapping, event, usage)
            recomputed = route_result.value
        elif (mapping.op in {OP_DIRECT, OP_VALUE_MAP}
              and mapping.source == ANCHOR_SOURCE_ID
              and spec.record.anchor == "external"):
            recomputed, source_ref_id = self._replay_external_anchor_operation(
                mapping, event, usage)
        elif mapping.op == OP_DIRECT and mapping.source == ANCHOR_SOURCE_ID:
            recomputed, source_ref_id = self._replay_grouping_anchor_direct(
                mapping, event, usage)
        elif mapping.op == OP_FALLBACK:
            route_result, source_ref_id = self._replay_fallback(
                mapping, event, usage)
            recomputed = route_result.value
        else:
            scalar_mapping = mapping
            scalar_event = event
            scalar_usage = usage
            if fact_stage or mapping.provenance.raw_from:
                scalar_mapping = dc_replace(
                    mapping, derivation=(),
                    contributions=dc_replace(
                        mapping.contributions, conditions=()),
                    provenance=dc_replace(
                        mapping.provenance, raw_from=""))
                scalar_event = dict(original_event)
                scalar_event["contributions"] = ""
                scalar_event["derivation"] = ""
                scalar_usage = dc_replace(
                    original_usage,
                    source_ref_ids=(original_event["source_ref_id"],))
            if (constructed_record is not None
                    and constructed_record.origin == ORIGIN_OBSERVED):
                route_result, source_ref_id = self._replay_scalar(
                    scalar_mapping, scalar_event, scalar_usage,
                    expected_source_row=constructed_record.selected_source_row,
                    record_key_verified=(
                        declared_mapping.provenance.record_key_from == "group"))
            else:
                route_result, source_ref_id = self._replay_scalar(
                    scalar_mapping, scalar_event, scalar_usage)
            recomputed = route_result.value

        if route_result is not None:
            raw_from = declared_mapping.provenance.raw_from
            if (raw_from_is_causal(
                    raw_from_id=raw_from,
                    raw_policy=declared_mapping.provenance.raw_value,
                    completion=route_result.completion)
                    and raw_from not in branch_reads):
                contributions = self._stage15_contributions(
                    original_event, declared_mapping.mapping_id)
                branch_reads[raw_from] = self._stage15_named_read(
                    raw_from, declared_mapping, original_event, original_usage,
                    contributions, groups_cache=branch_groups)
            try:
                finalized = finalize_raw_value(
                    route_result,
                    {name: item.value for name, item in branch_reads.items()},
                    raw_from_id=raw_from,
                    raw_policy=declared_mapping.provenance.raw_value)
            except RawAuthorityUnavailable as exc:
                _not_replayable(
                    "NAMED_READ_EVIDENCE_MISSING"
                    if raw_from else "RAW_VALUE_ENCODING_MISSING",
                    str(exc), spec_ref=spec.ref, mapping_id=mapping_id,
                    source_ref_id=source_ref_id)
            except ValueError as exc:
                _not_replayable(
                    "RAW_VALUE_KIND_UNSUPPORTED", str(exc),
                    spec_ref=spec.ref, mapping_id=mapping_id,
                    source_ref_id=source_ref_id)
            _verify_finalized_raw_snapshot(
                original_event, finalized, spec_ref=spec.ref,
                mapping_id=mapping_id, source_ref_id=source_ref_id)
            finalized_raw = finalized.value

        if declared_mapping.op not in {
                OP_NOT_APPLICABLE, OP_NO_MAPPING_CONFIGURED}:
            _verify_static_present_value_state(
                original_event, declared_mapping, source_ref_id=source_ref_id,
                spec_ref=spec.ref)

        if fact_stage and verified_derivation is None:
            verified_derivation = self._stage16_derivation_facts(
                declared_mapping, original_event, original_usage,
                reads_cache=branch_reads, groups_cache=branch_groups,
                conditions_cache=branch_conditions,
                constructed_record=constructed_record)
        if fact_stage:
            self._verify_stage16_usage_closure(
                declared_mapping, original_usage, verified_derivation,
                source_ref_id=source_ref_id,
                group_scalar=verified_group_scalar,
                construction_source_ref_ids=(
                    construction_universe.source_ref_ids
                    if construction_universe is not None else ()))
            self._verify_stage16_derivation(
                declared_mapping, original_event, recomputed,
                effective_source_column, verified_derivation,
                finalized_raw=finalized_raw)
        elif mapping.derivation:
            self._verify_stage14_derivation(mapping, event, recomputed)
        self._compare_event(
            original_event, declared_mapping, recomputed, source_ref_id)
        return _EventOutcome(recomputed, spec.ref, mapping_id, source_ref_id)

    def _replay_constant(
            self, mapping: SpecMapping, event, usage
            ) -> VerifiedOperationRouteResult:
        """Replay one source-independent scalar declared by the pinned spec."""
        assert self.spec is not None
        spec = self.spec
        context = {"spec_ref": spec.ref, "mapping_id": mapping.mapping_id}
        _check_constant_scope(mapping, spec, event, self.artifact)
        _check_yaml_declaration_environment(
            self.manifest, spec.ref, mapping.mapping_id)
        try:
            outcome = recompute_constant(declared_value=mapping.value)
        except ValueError as exc:
            _not_replayable("OPERATION_UNSUPPORTED", str(exc), **context)

        expected_method = "spec:constant"
        method = event.get("extraction_method", "")
        if method != expected_method:
            if method.startswith("post"):
                _not_replayable(
                    "POST_RULE_UNSUPPORTED_STAGE1",
                    "post-rule provenance is outside constant replay", **context)
            _fail("SOURCE_DECLARATION_MISMATCH",
                  f"provenance extraction_method is not {expected_method}",
                  **context)

        if usage.source_ref_ids or event.get("source_ref_id"):
            _fail("SOURCE_DECLARATION_MISMATCH",
                  "source-independent constant claims source evidence", **context)
        event_evidence = tuple(
            item for item in event.get("evidence_refs", "").split(";") if item)
        if usage.evidence_ref_ids or event_evidence:
            _fail("SOURCE_DECLARATION_MISMATCH",
                  "spec-local constant claims external evidence", **context)
        if event.get("contributions"):
            _fail("SOURCE_DECLARATION_MISMATCH",
                  "constant provenance contains undeclared derived context",
                  **context)
        if event.get("derivation") and not mapping.derivation:
            _fail("SOURCE_DECLARATION_MISMATCH",
                  "constant provenance contains undeclared derivation",
                  **context)

        expected_origin = mapping.provenance.source_label or "SPEC:constant"
        expected_column = mapping.column or mapping.target
        expected_key = _constant_record_key(
            mapping, spec, self.delivered_row)
        expected_authority = ";".join(mapping.artifacts)
        if (event.get("source_file") != expected_origin
                or event.get("source_logical_label") != expected_origin
                or event.get("source_column") != expected_column
                or event.get("source_record_key") != expected_key
                or event.get("source_resolved_path")
                or event.get("source_sha256")
                or event.get("source_sheet")
                or event.get("source_row_number")
                or event.get("authoritative_for") != expected_authority):
            _fail("SOURCE_DECLARATION_MISMATCH",
                  "constant provenance context contradicts its pinned policy",
                  **context)

        expected_rule, expected_confidence = _outcome_context(mapping, outcome)
        _verify_confidence(event, expected_confidence, mapping.op, spec.ref,
                           mapping.mapping_id, "")
        _verify_rule(event, expected_rule, usage, self.manifest, mapping.op,
                     spec.ref, mapping.mapping_id, "")
        expected_rule_ref = f"{expected_rule.id}@{expected_rule.version}"
        if set(usage.rule_refs) != {expected_rule_ref}:
            _fail("RULE_IDENTITY_MISMATCH",
                  "constant usage claims rules outside its deterministic outcome",
                  **context)
        return _route_result(outcome.value, mapping.value)

    def _replay_declared_absence(self, mapping: SpecMapping, event, usage
                                 ) -> tuple[VerifiedOperationRouteResult, str]:
        """Replay pinned absence semantics, with optional file attribution.

        The provenance source reference is verification context only.  The
        semantic recomputation remains the Stage 7 pure function and never
        receives a source, row, scalar, or recorded output.
        """
        assert self.spec is not None
        spec = self.spec
        context = {"spec_ref": spec.ref, "mapping_id": mapping.mapping_id}
        _check_declared_absence_scope(mapping, spec, event, self.artifact)
        _check_yaml_declaration_environment(
            self.manifest, spec.ref, mapping.mapping_id)
        try:
            outcome = recompute_declared_absence(
                declared_state=mapping.value_state)
        except ValueError as exc:
            _not_replayable("VALUE_STATE_UNSUPPORTED", str(exc), **context)

        expected_method = "spec:not_applicable"
        method = event.get("extraction_method", "")
        if method != expected_method:
            if method == "value_suppressed":
                _not_replayable(
                    "SUPPRESSION_UNSUPPORTED_STAGE1",
                    "suppression provenance is outside declared-absence "
                    "replay", **context)
            if method == "row_excluded":
                _not_replayable(
                    "EXCLUSION_REQUIRES_ROW_REPLAY",
                    "row-exclusion provenance is outside declared-absence "
                    "replay", **context)
            if method.startswith("post"):
                _not_replayable(
                    "POST_RULE_UNSUPPORTED_STAGE1",
                    "post-rule provenance is outside declared-absence replay",
                    **context)
            _fail("SOURCE_DECLARATION_MISMATCH",
                  f"provenance extraction_method is not {expected_method}",
                  **context)

        observed_state = event.get("value_state", "")
        expected_state = ("" if mapping.provenance.value_state == "none"
                          else outcome.semantic_state)
        if observed_state != expected_state:
            _fail(
                "VALUE_STATE_CONTEXT_MISMATCH",
                f"provenance value_state {observed_state!r} differs from "
                f"pinned effective state {expected_state!r}", **context)

        source_ref_id = ""
        verified_source = None
        if mapping.provenance.source_ref:
            verified_source, source_ref_id = \
                self._verify_attributed_absence_source(
                    mapping, event, usage)
        elif usage.source_ref_ids or event.get("source_ref_id"):
            _fail(
                "SOURCE_DECLARATION_MISMATCH",
                "source-independent declared absence claims source evidence",
                **context)
        event_evidence = tuple(
            item for item in event.get("evidence_refs", "").split(";") if item)
        if usage.evidence_ref_ids or event_evidence:
            _fail(
                "SOURCE_DECLARATION_MISMATCH",
                "spec-local declared absence claims external evidence",
                **context)
        if event.get("contributions") or event.get("derivation"):
            _fail(
                "SOURCE_DECLARATION_MISMATCH",
                "declared-absence provenance contains derived context",
                **context)

        expected_column = mapping.column or mapping.target
        expected_key = _declared_absence_record_key(
            mapping, spec, self.delivered_row)
        expected_authority = ";".join(mapping.artifacts)
        if verified_source is None:
            expected_origin = (
                mapping.provenance.source_label or "SPEC:not_applicable")
            locator_mismatch = (
                event.get("source_file") != expected_origin
                or event.get("source_logical_label") != expected_origin
                or event.get("source_resolved_path")
                or event.get("source_sha256")
                or event.get("source_sheet")
                or event.get("source_ref_id"))
        else:
            binding = verified_source.binding
            locator_mismatch = (
                event.get("source_file") != binding.logical_label
                or event.get("source_logical_label") != binding.logical_label
                or event.get("source_resolved_path") != binding.path
                or event.get("source_sha256") != binding.sha256
                or event.get("source_sheet") != binding.sheet
                or event.get("source_ref_id") != binding.source_ref_id)
        if (locator_mismatch
                or event.get("source_column") != expected_column
                or event.get("source_record_key") != expected_key
                or event.get("source_row_number")
                or event.get("authoritative_for") != expected_authority):
            _fail(
                "SOURCE_DECLARATION_MISMATCH",
                "declared-absence provenance contradicts its pinned policy",
                source_ref_id=source_ref_id, **context)

        expected_rule = mapping.no_value_rule or mapping.rule
        expected_confidence = mapping.confidence
        _verify_confidence(event, expected_confidence, mapping.op, spec.ref,
                           mapping.mapping_id, source_ref_id)
        _verify_rule(event, expected_rule, usage, self.manifest, mapping.op,
                     spec.ref, mapping.mapping_id, source_ref_id)
        expected_rule_ref = f"{expected_rule.id}@{expected_rule.version}"
        if set(usage.rule_refs) != {expected_rule_ref}:
            _fail(
                "RULE_IDENTITY_MISMATCH",
                "declared-absence usage claims rules outside its deterministic "
                "outcome", **context)
        return _route_result(outcome.value, None), source_ref_id

    def _constructed_row_identity(self, anchor_row: Mapping[str, Any],
                                  mapping_id: str
                                  ) -> dict[str, str]:
        """Canonicalize one pure record using only pinned identity literals."""
        assert self.spec is not None
        values = dict(anchor_row)
        for name in self.schema.row_key:
            if name in values:
                continue
            constants = [
                mapping for mapping in self.spec.mappings
                if mapping.target == name and mapping.op == OP_CONSTANT
                and not mapping.when
            ]
            if len(constants) != 1:
                _not_replayable(
                    "ANCHOR_EVIDENCE_MISSING",
                    f"constructed row identity field {name!r} has no unique "
                    "pinned literal authority",
                    spec_ref=self.spec.ref, mapping_id=mapping_id)
            values[name] = constants[0].value
        try:
            return canonical_row_identity(self.schema.row_key, values)
        except ValueError as exc:
            _not_replayable(
                "ANCHOR_EVIDENCE_MISSING",
                f"constructed row identity is incomplete: {exc}",
                spec_ref=self.spec.ref, mapping_id=mapping_id)

    def _verify_constructed_record(
            self, mapping: SpecMapping, event, usage,
            row_identity: Mapping[str, str], *,
            additional_source_ref_ids: frozenset[str] = frozenset(),
            scalar_anchor_consumer: bool = False,
            ) -> tuple[_VerifiedConstructedRecord,
                       _VerifiedConstructionUniverse]:
        """Verify event-scoped construction closure and select one record."""
        assert self.spec is not None
        spec = self.spec
        context = {"spec_ref": spec.ref, "mapping_id": mapping.mapping_id}
        if not scalar_anchor_consumer:
            _check_stage19_construction_scope(mapping, spec, event, self.artifact)

        event_refs = {
            item for item in event.get("evidence_refs", "").split(";") if item}
        usage_refs = set(usage.evidence_ref_ids)
        event_anchor_refs = {ref for ref in event_refs
                             if ref.startswith("anchor-")}
        usage_anchor_refs = {ref for ref in usage_refs
                             if ref.startswith("anchor-")}
        if not event_anchor_refs or not usage_anchor_refs:
            _not_replayable(
                "ANCHOR_EVIDENCE_MISSING",
                "record-construction event omits event-scoped anchor evidence",
                **context)
        if (len(event_anchor_refs) != 1
                or event_anchor_refs != usage_anchor_refs
                or event_refs != event_anchor_refs
                or usage_refs != usage_anchor_refs):
            _fail(
                "ANCHOR_IDENTITY_MISMATCH",
                "Cell and MappingUsage construction evidence links disagree",
                **context)
        evidence_ref_id = next(iter(event_anchor_refs))
        matches = [item for item in self.manifest.evidence
                   if item.evidence_ref_id == evidence_ref_id]
        if not matches:
            _not_replayable(
                "ANCHOR_EVIDENCE_MISSING",
                "manifest omits the event-scoped construction binding",
                **context)
        if len(matches) != 1:
            _fail(
                "ANCHOR_IDENTITY_MISMATCH",
                "manifest construction evidence identity is ambiguous",
                **context)
        binding = matches[0]
        if (binding.kind != "anchor"
                or binding.identity != f"anchor:{evidence_ref_id}"
                or binding.path or binding.sha256
                or binding.reference != f"{spec.ref}#record.anchor"):
            _fail(
                "ANCHOR_IDENTITY_MISMATCH",
                "construction binding contradicts the pinned record policy",
                **context)
        details = _thaw_manifest_value(dict(binding.details))
        source_ids = details.get("source_ref_ids")
        if not isinstance(source_ids, dict):
            _not_replayable(
                "ANCHOR_EVIDENCE_MISSING",
                "construction binding omits runtime source identities",
                **context)
        expected_source_ids = [spec.record.anchor_source]
        expected_source_ids.extend(
            declaration.participants_from
            for declaration in spec.record.additional_records)
        expected_source_ids = list(dict.fromkeys(expected_source_ids))
        if set(source_ids) != set(expected_source_ids):
            _fail(
                "ANCHOR_IDENTITY_MISMATCH",
                "construction binding source set contradicts the declaration",
                **context)
        construction_refs = set(source_ids.values())
        actual_usage_refs = set(usage.source_ref_ids)
        if not construction_refs.issubset(actual_usage_refs):
            _not_replayable(
                "ANCHOR_EVIDENCE_MISSING",
                "MappingUsage omits construction source evidence",
                **context)
        if actual_usage_refs != construction_refs | set(additional_source_ref_ids):
            _fail(
                "ANCHOR_IDENTITY_MISMATCH",
                "MappingUsage claims sources outside exact construction closure",
                **context)

        universe = self.construction_cache.get(evidence_ref_id)
        if universe is None:
            verified: dict[str, _VerifiedSource] = {}
            source_bindings: dict[str, SourceBinding] = {}
            for source_id in expected_source_ids:
                source_ref_id = str(source_ids.get(source_id) or "")
                candidates = [
                    item for item in self.manifest.sources
                    if item.source_ref_id == source_ref_id]
                if not source_ref_id or not candidates:
                    _not_replayable(
                        "ANCHOR_EVIDENCE_MISSING",
                        f"construction source {source_id!r} has no SourceBinding",
                        **context)
                if len(candidates) != 1:
                    _fail(
                        "ANCHOR_IDENTITY_MISMATCH",
                        f"construction source {source_id!r} is ambiguous",
                        **context)
                source_binding = candidates[0]
                declaration = spec.source(source_id)
                _check_anchor_source_declaration(
                    self.root, spec, source_id, declaration, source_binding,
                    mapping.mapping_id)
                source_bindings[source_id] = source_binding
                verified[source_id] = self._verify_source_identity(
                    source_binding, declaration, mapping.mapping_id)
            try:
                constructed = construct_grouping_records(
                    spec,
                    verified[spec.record.anchor_source].frame,
                    {
                        source_id: item.frame
                        for source_id, item in verified.items()
                        if source_id != spec.record.anchor_source
                    },
                )
            except RecordConstructionError as exc:
                code = (
                    "SOURCE_COLUMN_MISSING"
                    if "column" in str(exc) else "ANCHOR_SELECTION_MISMATCH")
                _fail(code, f"record construction failed: {exc}", **context)
            expected_runtime = RuntimeGroupingAnchorEvidence.capture(
                spec, source_bindings,
                record_count=len(constructed.records))
            expected_details = dict(expected_runtime.details)
            expected_details["environment"] = {
                "python": self.manifest.producer.python_version,
                "dependencies": dict(self.manifest.producer.dependencies),
            }
            if (evidence_ref_id != expected_runtime.evidence_ref_id
                    or binding.reference != expected_runtime.reference
                    or details != expected_details):
                _fail(
                    "ANCHOR_IDENTITY_MISMATCH",
                    "fresh construction contradicts the anchor binding",
                    **context)
            records = []
            for item in constructed.records:
                identity = self._constructed_row_identity(
                    item.anchor_row, mapping.mapping_id)
                records.append(_VerifiedConstructedRecord(
                    row_identity=tuple(identity.items()),
                    origin=item.origin,
                    anchor_items=tuple(item.anchor_items),
                    group_key=tuple(item.group_key),
                    selected_source_row=item.selected_source_row,
                    group_size=item.group_size,
                    declaration_id=item.declaration_id,
                ))
            universe = _VerifiedConstructionUniverse(
                evidence_ref_id=evidence_ref_id,
                records=tuple(records),
                source_ref_ids=tuple(sorted(construction_refs)),
                source_bindings=tuple(source_bindings.items()),
            )
            self.construction_cache[evidence_ref_id] = universe

        wanted_identity = tuple(row_identity.items())
        selected = [item for item in universe.records
                    if item.row_identity == wanted_identity]
        if len(selected) != 1:
            _fail(
                "ANCHOR_SELECTION_MISMATCH",
                "construction does not yield exactly one requested logical row",
                **context)
        record = selected[0]
        try:
            event_identity = _row_identity_from_event(spec, self.schema, event)
        except ValueError as exc:
            _fail(
                "ANCHOR_SELECTION_MISMATCH",
                f"recorded event identity is malformed: {exc}", **context)
        if event_identity != dict(record.row_identity):
            _fail(
                "ANCHOR_SELECTION_MISMATCH",
                "recorded event identity contradicts fresh construction",
                **context)

        provenance = mapping.provenance
        participant = str(
            record.anchor_row.get(spec.record.participant_key, ""))
        if provenance.record_key_from == "group":
            if record.origin != ORIGIN_OBSERVED or not record.group_key:
                _fail(
                    "ANCHOR_SELECTION_MISMATCH",
                    "group record-key policy selected a synthesized record",
                    **context)
            expected_key = ";".join(
                f"{name}={value}" for name, value in record.group_key)
            if event.get("source_record_key") != expected_key:
                _fail(
                    "ANCHOR_SELECTION_MISMATCH",
                    "recorded group key contradicts fresh construction",
                    **context)
        elif provenance.record_key_from == "participant":
            expected_key = _participant_record_key(spec.record, participant)
            if event.get("source_record_key") != expected_key:
                _fail(
                    "ANCHOR_SELECTION_MISMATCH",
                    "recorded participant key contradicts fresh construction",
                    **context)
        return record, universe

    def _verify_grouping_anchor_scalar(
            self, mapping: SpecMapping, event, usage, *,
            additional_source_ref_ids: frozenset[str] = frozenset()
            ) -> _VerifiedAnchorScalar:
        """Rebuild one selected grouping-anchor scalar from causal inputs."""
        assert self.spec is not None
        spec = self.spec
        context = {"spec_ref": spec.ref, "mapping_id": mapping.mapping_id}

        record, universe = self._verify_constructed_record(
            mapping, event, usage, self.row_identity,
            additional_source_ref_ids=additional_source_ref_ids,
            scalar_anchor_consumer=True)
        try:
            contributions = contributions_from_json(
                event.get("contributions", ""))
        except (TypeError, ValueError, json.JSONDecodeError) as exc:
            _not_replayable(
                "ANCHOR_EVIDENCE_MISSING",
                f"anchor contribution evidence is malformed: {exc}", **context)
        if len(contributions) != 1 or not isinstance(contributions[0], dict):
            _not_replayable(
                "ANCHOR_EVIDENCE_MISSING",
                "grouping-anchor replay requires one Contribution", **context)
        contribution = contributions[0]
        contribution_refs = {
            item for item in str(
                contribution.get("evidence_refs") or "").split(";") if item}
        if (contribution.get("role") != "anchor_input"
                or contribution.get("status") != STATUS_INPUT
                or contribution_refs != {universe.evidence_ref_id}
                or contribution.get("source_column") != mapping.column):
            _fail(
                "ANCHOR_SELECTION_MISMATCH",
                "anchor Contribution contradicts the consumed field",
                **context)

        bindings = dict(universe.source_bindings)
        if record.origin == ORIGIN_OBSERVED:
            route = ANCHOR_RECORD_GROUPED
            source_id = spec.record.anchor_source
            expected_key = ";".join(
                f"{name}={value}" for name, value in record.group_key)
            if (contribution.get("anchor_record_origin")
                    or contribution.get("source_row_number")
                       != record.selected_source_row):
                _fail(
                    "ANCHOR_SELECTION_MISMATCH",
                    "grouped Contribution contradicts selected first row",
                    **context)
            declaration_id = ""
        else:
            route = ANCHOR_RECORD_SYNTHESIZED
            declaration_id = record.declaration_id
            declarations = [
                item for item in spec.record.additional_records
                if item.id == declaration_id]
            if len(declarations) != 1:
                _fail(
                    "ANCHOR_SELECTION_MISMATCH",
                    "synthesized declaration identity is not pinned",
                    **context)
            source_id = declarations[0].participants_from
            participant = str(
                record.anchor_row.get(spec.record.participant_key, ""))
            expected_key = _participant_record_key(spec.record, participant)
            if (contribution.get("anchor_record_origin") != declaration_id
                    or contribution.get("source_row_number") not in ("", None)):
                _fail(
                    "ANCHOR_SELECTION_MISMATCH",
                    "synthesized Contribution invents grouping authority",
                    **context)
        if contribution.get("anchor_record_kind") != route:
            _fail(
                "ANCHOR_SELECTION_MISMATCH",
                "anchor Contribution claims the wrong construction route",
                **context)
        source_binding = bindings[source_id]
        source_ref_id = source_binding.source_ref_id
        _verify_anchor_contribution_source(
            contribution, source_binding, expected_key, source_ref_id,
            **context)
        anchor_row = record.anchor_row
        if mapping.column not in anchor_row:
            _fail(
                "ANCHOR_SELECTION_MISMATCH",
                f"constructed record omits anchor field {mapping.column!r}",
                **context)
        observed_raw = str(anchor_row[mapping.column])
        contribution_raw = str(contribution.get("raw_value_json") or "")
        event_raw = event.get("raw_value_json", "")
        if not contribution_raw or not event_raw:
            _not_replayable(
                "RAW_VALUE_ENCODING_MISSING",
                "anchor scalar lacks strict typed raw evidence", **context)
        try:
            observed_typed = encode_typed_raw(observed_raw)
            contribution_typed = TypedRawValue.from_json(contribution_raw)
            event_typed = TypedRawValue.from_json(event_raw)
        except ValueError as exc:
            _not_replayable("RAW_VALUE_KIND_UNSUPPORTED", str(exc), **context)
        if (contribution_typed != observed_typed
                or event_typed != observed_typed
                or contribution.get("raw_value") != observed_raw):
            _fail(
                "RAW_VALUE_MISMATCH",
                "reconstructed anchor scalar contradicts recorded evidence",
                source_ref_id=source_ref_id, **context)
        return _VerifiedAnchorScalar(
            evidence_ref_id=universe.evidence_ref_id,
            route=route,
            record_key=expected_key,
            anchor_column=mapping.column,
            raw_value=observed_raw,
            source_ref_id=source_ref_id,
            selected_source_row=record.selected_source_row,
            selected_filtered_position=None,
            synthesized_declaration=declaration_id,
            construction_source_ref_ids=universe.source_ref_ids,
        )


    def _replay_no_mapping_configured(
            self, mapping: SpecMapping, event, usage) -> tuple[Any, str]:
        """Replay the closed grouping-anchor no-mapping contract."""
        assert self.spec is not None
        spec = self.spec
        context = {"spec_ref": spec.ref, "mapping_id": mapping.mapping_id}
        _check_no_mapping_scope(mapping, spec, event, self.artifact)
        _check_stage2_environment(self.manifest, spec.ref, mapping.mapping_id)
        verified = self._verify_grouping_anchor_scalar(mapping, event, usage)

        outcome = recompute_no_mapping_configured()
        expected_state = ("" if mapping.provenance.value_state == "none"
                          else mapping.value_state)
        if event.get("value_state", "") != expected_state:
            _fail("VALUE_STATE_CONTEXT_MISMATCH",
                  "no-mapping Cell state contradicts pinned policy", **context)
        expected_rule = mapping.no_value_rule or mapping.rule
        _verify_confidence(event, mapping.confidence, mapping.op, spec.ref,
                           mapping.mapping_id, verified.source_ref_id)
        _verify_rule(event, expected_rule, usage, self.manifest, mapping.op,
                     spec.ref, mapping.mapping_id, verified.source_ref_id)
        if set(usage.rule_refs) != {
                f"{expected_rule.id}@{expected_rule.version}"}:
            _fail("RULE_IDENTITY_MISMATCH",
                  "no-mapping usage claims an unrelated rule", **context)

        record = spec.record
        participant = str(self.delivered_row.get(record.participant_key, ""))
        expected_origin = mapping.provenance.source_label or "ANCHOR"
        expected_event_key = _participant_record_key(record, participant)
        if (event.get("extraction_method") != "spec:no_mapping_configured"
                or event.get("source_file") != expected_origin
                or event.get("source_logical_label") != expected_origin
                or event.get("source_column") != mapping.column
                or event.get("source_record_key") != expected_event_key
                or event.get("source_resolved_path")
                or event.get("source_sha256")
                or event.get("source_sheet")
                or event.get("source_row_number")
                or event.get("source_ref_id")
                or event.get("derivation")
                or event.get("authoritative_for") != ";".join(mapping.artifacts)):
            _fail("SOURCE_DECLARATION_MISMATCH",
                  "no-mapping provenance contradicts pinned anchor policy",
                  source_ref_id=verified.source_ref_id, **context)
        return outcome.value, verified.source_ref_id

    def _resolve_no_row_attributed_source(
            self, mapping: SpecMapping, event, usage, *,
            conflicting_usage_is_failure: bool = False) -> SourceBinding:
        """Resolve mandatory attribution links without reading a scalar."""
        assert self.spec is not None
        spec = self.spec
        source_id = mapping.provenance.source_ref
        context = {"spec_ref": spec.ref, "mapping_id": mapping.mapping_id}
        bindings = [
            binding for binding in self.manifest.sources
            if binding.spec_ref == spec.ref and binding.source_id == source_id
        ]
        if not bindings:
            _not_replayable(
                "SOURCE_LINK_MISSING",
                "attributed mapping has no runtime SourceBinding", **context)
        if len(bindings) != 1:
            _fail(
                "SOURCE_DECLARATION_MISMATCH",
                "attributed mapping has ambiguous source bindings", **context)
        binding = bindings[0]
        context["source_ref_id"] = binding.source_ref_id
        if binding.source_ref_id not in usage.source_ref_ids:
            if conflicting_usage_is_failure and usage.source_ref_ids:
                _fail(
                    "SOURCE_DECLARATION_MISMATCH",
                    "MappingUsage contradicts the pinned attributed source",
                    **context)
            _not_replayable(
                "SOURCE_LINK_MISSING",
                "MappingUsage omits attributed runtime source evidence",
                **context)
        event_source_ref = event.get("source_ref_id", "")
        if not event_source_ref:
            _not_replayable(
                "SOURCE_LINK_MISSING",
                "provenance Cell omits attributed runtime source evidence",
                **context)
        if event_source_ref != binding.source_ref_id:
            _fail(
                "SOURCE_DECLARATION_MISMATCH",
                "provenance Cell contradicts the pinned attributed source",
                **context)
        return binding

    def _verify_no_row_attributed_source(
            self, mapping: SpecMapping, event, usage,
            binding: SourceBinding, *, expected_source_ref_ids: set[str]
            ) -> tuple[_VerifiedSource, str]:
        """Verify one exact attributed file without seeking a row or scalar."""
        assert self.spec is not None
        spec = self.spec
        context = {"spec_ref": spec.ref, "mapping_id": mapping.mapping_id,
                   "source_ref_id": binding.source_ref_id}
        if set(usage.source_ref_ids) != expected_source_ref_ids:
            _fail(
                "SOURCE_DECLARATION_MISMATCH",
                "MappingUsage contradicts the independent anchor and "
                "attribution source set", **context)
        if event.get("source_row_number"):
            _fail(
                "SOURCE_DECLARATION_MISMATCH",
                "no-row attribution unexpectedly claims a source row",
                **context)
        source_decl = spec.source(mapping.provenance.source_ref)
        _check_no_row_attributed_source_declaration(
            self.root, spec, mapping, source_decl, binding)
        verified = self._verify_source_identity(
            binding, source_decl, mapping.mapping_id)
        return verified, binding.source_ref_id

    def _verify_attributed_absence_source(
            self, mapping: SpecMapping, event, usage
            ) -> tuple[_VerifiedSource, str]:
        """Stage 8 wrapper around generic no-row source attribution."""
        binding = self._resolve_no_row_attributed_source(
            mapping, event, usage, conflicting_usage_is_failure=True)
        return self._verify_no_row_attributed_source(
            mapping, event, usage, binding,
            expected_source_ref_ids={binding.source_ref_id})

    def _replay_grouping_anchor_direct(
            self, mapping: SpecMapping, event, usage) -> tuple[Any, str]:
        """Replay direct from an independently rebuilt grouping anchor."""
        assert self.spec is not None
        spec = self.spec
        context = {"spec_ref": spec.ref, "mapping_id": mapping.mapping_id}
        _check_grouping_anchor_direct_scope(
            mapping, spec, event, self.artifact)
        _check_stage2_environment(self.manifest, spec.ref, mapping.mapping_id)

        attributed_binding = None
        additional_source_ref_ids: frozenset[str] = frozenset()
        if mapping.provenance.source_ref:
            attributed_binding = self._resolve_no_row_attributed_source(
                mapping, event, usage)
            additional_source_ref_ids = frozenset({
                attributed_binding.source_ref_id})
        verified = self._verify_grouping_anchor_scalar(
            mapping, event, usage,
            additional_source_ref_ids=additional_source_ref_ids)

        if attributed_binding is not None:
            expected_source_ref_ids = set(
                verified.construction_source_ref_ids)
            expected_source_ref_ids.add(attributed_binding.source_ref_id)
            self._verify_no_row_attributed_source(
                mapping, event, usage, attributed_binding,
                expected_source_ref_ids=expected_source_ref_ids)
            provenance_source_ref_id = attributed_binding.source_ref_id
        else:
            if event.get("source_ref_id"):
                _fail(
                    "SOURCE_DECLARATION_MISMATCH",
                    "anchor direct claims undeclared source attribution",
                    **context)
            provenance_source_ref_id = ""

        outcome = _recompute_mapping(mapping, verified.raw_value)
        expected_rule, expected_confidence = _outcome_context(mapping, outcome)
        result_source_ref_id = (
            provenance_source_ref_id or verified.source_ref_id)
        _verify_confidence(event, expected_confidence, mapping.op, spec.ref,
                           mapping.mapping_id, result_source_ref_id)
        _verify_rule(event, expected_rule, usage, self.manifest, mapping.op,
                     spec.ref, mapping.mapping_id, result_source_ref_id)

        record = spec.record
        participant = str(self.delivered_row.get(record.participant_key, ""))
        provenance = mapping.provenance
        if provenance.record_key:
            expected_event_key = provenance.record_key
        elif (provenance.record_key_from == "participant"
              or provenance.source_ref):
            expected_event_key = _participant_record_key(record, participant)
        else:
            expected_event_key = ""

        if attributed_binding is None:
            expected_origin = provenance.source_label or "ANCHOR"
            locator_mismatch = (
                event.get("source_file") != expected_origin
                or event.get("source_logical_label") != expected_origin
                or event.get("source_resolved_path")
                or event.get("source_sha256")
                or event.get("source_sheet")
                or event.get("source_ref_id"))
        else:
            locator_mismatch = (
                event.get("source_file") != attributed_binding.logical_label
                or event.get("source_logical_label")
                   != attributed_binding.logical_label
                or event.get("source_resolved_path") != attributed_binding.path
                or event.get("source_sha256") != attributed_binding.sha256
                or event.get("source_sheet") != attributed_binding.sheet
                or event.get("source_ref_id")
                   != attributed_binding.source_ref_id)
        if (locator_mismatch
                or event.get("source_column") != mapping.column
                or event.get("source_record_key") != expected_event_key
                or event.get("source_row_number")
                or event.get("derivation")
                or event.get("authoritative_for") != ";".join(mapping.artifacts)):
            _fail(
                "SOURCE_DECLARATION_MISMATCH",
                "anchor-direct provenance contradicts its pinned origin policy",
                source_ref_id=result_source_ref_id, **context)
        return outcome.value, result_source_ref_id

    def _verify_external_anchor_scalar(
            self, mapping: SpecMapping, event, usage, *,
            additional_source_ref_ids: frozenset[str] = frozenset()
            ) -> _VerifiedAnchorScalar:
        """Rebuild one external anchor scalar from its exact causal bytes."""
        assert self.spec is not None
        spec = self.spec
        context = {"spec_ref": spec.ref, "mapping_id": mapping.mapping_id}

        event_refs = {
            item for item in event.get("evidence_refs", "").split(";") if item}
        usage_refs = set(usage.evidence_ref_ids)
        anchor_refs = {ref for ref in usage_refs if ref.startswith("anchor-")}
        if not event_refs or not anchor_refs:
            _not_replayable(
                "ANCHOR_EVIDENCE_MISSING",
                "external-anchor event omits strict runtime evidence links",
                **context)
        if (len(anchor_refs) != 1 or usage_refs != anchor_refs
                or event_refs != anchor_refs):
            _fail("ANCHOR_IDENTITY_MISMATCH",
                  "Cell and MappingUsage external anchor links disagree",
                  **context)
        evidence_ref_id = next(iter(anchor_refs))
        evidence = [item for item in self.manifest.evidence
                    if item.evidence_ref_id == evidence_ref_id]
        if not evidence:
            _not_replayable(
                "ANCHOR_EVIDENCE_MISSING",
                "manifest omits the external anchor binding", **context)
        if len(evidence) != 1:
            _fail("ANCHOR_IDENTITY_MISMATCH",
                  "manifest external anchor identity is ambiguous", **context)
        anchor_binding = evidence[0]
        if (anchor_binding.kind != "anchor"
                or anchor_binding.identity != f"anchor:{evidence_ref_id}"
                or anchor_binding.path or anchor_binding.sha256
                or anchor_binding.reference != f"{spec.ref}#record.anchor"):
            _fail("ANCHOR_IDENTITY_MISMATCH",
                  "external anchor binding contradicts the pinned mapping",
                  **context)
        details = _thaw_manifest_value(dict(anchor_binding.details))
        source_ids = details.get("source_ref_ids")
        if not isinstance(source_ids, dict) or set(source_ids) != {
                EXTERNAL_ANCHOR_SOURCE_ID}:
            _not_replayable(
                "ANCHOR_EVIDENCE_MISSING",
                "external anchor binding omits its original input identity",
                **context)
        source_ref_id = str(source_ids.get(EXTERNAL_ANCHOR_SOURCE_ID) or "")
        sources = [item for item in self.manifest.sources
                   if item.source_ref_id == source_ref_id]
        if not source_ref_id or not sources:
            _not_replayable(
                "SOURCE_LINK_MISSING",
                "external anchor input has no SourceBinding", **context)
        if len(sources) != 1:
            _fail("ANCHOR_IDENTITY_MISMATCH",
                  "external anchor input binding is ambiguous", **context)
        source_binding = sources[0]
        context["source_ref_id"] = source_ref_id
        if (source_binding.identity != f"source:{source_ref_id}"
                or source_binding.spec_ref != spec.ref
                or source_binding.source_id != EXTERNAL_ANCHOR_SOURCE_ID
                or source_binding.config_key != EXTERNAL_ANCHOR_SOURCE_ID
                or source_binding.file_type != "csv" or source_binding.sheet):
            _fail("ANCHOR_IDENTITY_MISMATCH",
                  "external SourceBinding contradicts its runtime role",
                  **context)
        expected_sources = {source_ref_id} | set(additional_source_ref_ids)
        actual_sources = set(usage.source_ref_ids)
        if source_ref_id not in actual_sources:
            _not_replayable(
                "SOURCE_LINK_MISSING",
                "MappingUsage omits external anchor input evidence", **context)
        if actual_sources != expected_sources:
            _fail("ANCHOR_IDENTITY_MISMATCH",
                  "MappingUsage claims sources outside external anchor and "
                  "declared attribution", **context)

        _check_external_anchor_reader_contract(
            source_binding, self.manifest, mapping.mapping_id)
        source_path = resolve_project_path(self.root, source_binding.path)
        try:
            payload = source_path.read_bytes()
        except FileNotFoundError:
            _fail("SOURCE_FILE_MISSING",
                  f"source file is missing: {source_binding.path}", **context)
        except OSError as exc:
            _fail("SOURCE_FILE_MISSING",
                  f"external anchor source cannot be read: {exc}", **context)
        if hashlib.sha256(payload).hexdigest() != source_binding.sha256:
            _fail("SOURCE_HASH_MISMATCH",
                  "external anchor bytes differ from SourceBinding", **context)
        try:
            parsed = parse_external_anchor_bytes(payload)
        except Exception as exc:
            _fail("READER_OBSERVATION_MISMATCH",
                  f"external_anchor_csv@v1 cannot reproduce input: {exc}",
                  **context)
        if len(parsed) != source_binding.n_rows:
            _fail("SOURCE_ROW_COUNT_MISMATCH",
                  "external anchor pre-filter row count changed", **context)
        try:
            observed_ref = SourceRef.from_observed_bytes(
                config_key=EXTERNAL_ANCHOR_SOURCE_ID,
                path=source_path, project_root=self.root, payload=payload,
                file_type="csv", reader_id=EXTERNAL_ANCHOR_READER_ID,
                logical_label=source_binding.logical_label,
                reader_effective_flags=EXTERNAL_ANCHOR_READER_FLAGS,
                n_rows=len(parsed))
        except (PathContainmentError, FileNotFoundError) as exc:
            _not_replayable("MANIFEST_PATH_OUTSIDE_PROJECT", str(exc),
                            **context)
        if observed_ref.source_ref_id != source_ref_id:
            _fail("ANCHOR_IDENTITY_MISMATCH",
                  "external SourceRef cannot be rebuilt from parsed bytes",
                  **context)

        filter_doc = details.get("filter")
        required_filter = {
            "operation", "column", "selector", "equality", "null_semantics",
            "source_order", "reset_index",
        }
        if (not isinstance(filter_doc, dict)
                or set(filter_doc) != required_filter
                or filter_doc.get("operation") != "column_equals"
                or filter_doc.get("equality") != "pandas_scalar_eq@v1"
                or filter_doc.get("null_semantics")
                   != "parsed_null_never_matches"
                or filter_doc.get("source_order") != "preserved"
                or filter_doc.get("reset_index") != "drop_true"):
            _fail("ANCHOR_IDENTITY_MISMATCH",
                  "external anchor filter contract is contradictory", **context)
        try:
            selector = typed_value(filter_doc["selector"])
            filtered, original_rows = filter_external_anchor_frame(
                parsed, column=filter_doc["column"], selector=selector)
        except (KeyError, TypeError, ValueError) as exc:
            _fail("ANCHOR_IDENTITY_MISMATCH",
                  f"external anchor filter cannot be reconstructed: {exc}",
                  **context)
        expected_runtime = RuntimeExternalAnchorEvidence.capture(
            spec_ref=spec.ref, source_ref=observed_ref, frame=filtered,
            filter_column=filter_doc["column"], selector=selector,
            original_source_rows=original_rows)
        expected_details = dict(expected_runtime.details)
        expected_details["environment"] = {
            "python": self.manifest.producer.python_version,
            "dependencies": dict(self.manifest.producer.dependencies),
        }
        if (evidence_ref_id != expected_runtime.evidence_ref_id
                or details != expected_details
                or details.get("construction_contract_version")
                   != EXTERNAL_ANCHOR_CONSTRUCTION_VERSION
                or details.get("canonicalization_version")
                   != EXTERNAL_ANCHOR_FRAME_VERSION
                or details.get("canonicalization")
                   != "external_anchor_frame@v1"):
            _fail("ANCHOR_IDENTITY_MISMATCH",
                  "fresh external construction contradicts anchor binding",
                  **context)

        try:
            contributions = contributions_from_json(
                event.get("contributions", ""))
        except (TypeError, ValueError, json.JSONDecodeError) as exc:
            _not_replayable(
                "ANCHOR_EVIDENCE_MISSING",
                f"external anchor Contribution is malformed: {exc}", **context)
        if len(contributions) != 1 or not isinstance(contributions[0], dict):
            _not_replayable(
                "ANCHOR_EVIDENCE_MISSING",
                "external anchor replay requires one Contribution", **context)
        contribution = contributions[0]
        contribution_refs = {
            item for item in str(
                contribution.get("evidence_refs") or "").split(";") if item}
        if (contribution.get("role") != "anchor_input"
                or contribution.get("status") != STATUS_INPUT
                or contribution.get("anchor_record_kind")
                   != ANCHOR_RECORD_EXTERNAL
                or contribution.get("anchor_record_origin")
                or contribution_refs != {evidence_ref_id}
                or contribution.get("source_column") != mapping.column):
            _fail("ANCHOR_SELECTION_MISMATCH",
                  "external anchor Contribution contradicts its route/field",
                  **context)
        original_row = contribution.get("source_row_number")
        filtered_position = contribution.get("anchor_filtered_position")
        if (isinstance(original_row, bool) or not isinstance(original_row, int)
                or original_row < 0 or isinstance(filtered_position, bool)
                or not isinstance(filtered_position, int)
                or filtered_position < 0
                or filtered_position >= len(filtered)):
            _not_replayable(
                "ANCHOR_EVIDENCE_MISSING",
                "external anchor Contribution omits exact row locators",
                **context)
        if (original_rows[filtered_position] != original_row
                or int(details["original_source_rows"][filtered_position])
                   != original_row
                or int(details["filtered_positions"][filtered_position])
                   != filtered_position):
            _fail("ANCHOR_SELECTION_MISMATCH",
                  "external anchor row locators cannot be reproduced", **context)
        selected_row = filtered.iloc[filtered_position]
        record = spec.record
        for name in (record.participant_key, record.sample_key,
                     record.collection_event_key):
            if not name:
                continue
            if name not in selected_row.index:
                _fail("SOURCE_COLUMN_MISSING",
                      f"external anchor selected row omits {name!r}", **context)
            if _source_text(selected_row[name]) != self.delivered_row.get(name, ""):
                _fail("ANCHOR_SELECTION_MISMATCH",
                      "selected external row disagrees with delivered identity",
                      **context)
        participant = _source_text(selected_row[record.participant_key])
        expected_key = _participant_record_key(record, participant)
        _verify_anchor_contribution_source(
            contribution, source_binding, expected_key, source_ref_id,
            **context)
        if mapping.column not in selected_row.index:
            _fail("SOURCE_COLUMN_MISSING",
                  f"external anchor omits column {mapping.column!r}", **context)
        observed_raw = str(selected_row[mapping.column])
        contribution_raw = str(contribution.get("raw_value_json") or "")
        event_raw = event.get("raw_value_json", "")
        if not contribution_raw or not event_raw:
            _not_replayable(
                "RAW_VALUE_ENCODING_MISSING",
                "external anchor scalar lacks typed raw evidence", **context)
        try:
            observed_typed = encode_typed_raw(observed_raw)
            contribution_typed = TypedRawValue.from_json(contribution_raw)
            event_typed = TypedRawValue.from_json(event_raw)
        except ValueError as exc:
            _not_replayable("RAW_VALUE_KIND_UNSUPPORTED", str(exc), **context)
        if (contribution_typed != observed_typed
                or event_typed != observed_typed
                or contribution.get("raw_value") != observed_raw):
            _fail("RAW_VALUE_MISMATCH",
                  "reconstructed external scalar contradicts recorded evidence",
                  **context)
        return _VerifiedAnchorScalar(
            evidence_ref_id=evidence_ref_id,
            route=ANCHOR_RECORD_EXTERNAL,
            record_key=expected_key,
            anchor_column=mapping.column,
            raw_value=observed_raw,
            source_ref_id=source_ref_id,
            selected_source_row=original_row,
            selected_filtered_position=filtered_position,
            synthesized_declaration="",
            construction_source_ref_ids=(source_ref_id,),
        )

    def _replay_external_anchor_operation(
            self, mapping: SpecMapping, event, usage) -> tuple[Any, str]:
        """Replay direct or inline map from a verified external scalar."""
        assert self.spec is not None
        spec = self.spec
        context = {"spec_ref": spec.ref, "mapping_id": mapping.mapping_id}
        _check_external_anchor_scope(mapping, spec, event, self.artifact)
        _check_stage2_environment(self.manifest, spec.ref, mapping.mapping_id)

        attributed_binding = None
        additional_source_ref_ids: frozenset[str] = frozenset()
        if mapping.provenance.source_ref:
            attributed_binding = self._resolve_no_row_attributed_source(
                mapping, event, usage)
            additional_source_ref_ids = frozenset({
                attributed_binding.source_ref_id})
        verified = self._verify_external_anchor_scalar(
            mapping, event, usage,
            additional_source_ref_ids=additional_source_ref_ids)

        if attributed_binding is not None:
            self._verify_no_row_attributed_source(
                mapping, event, usage, attributed_binding,
                expected_source_ref_ids={
                    verified.source_ref_id, attributed_binding.source_ref_id})
            provenance_source_ref_id = attributed_binding.source_ref_id
        else:
            if event.get("source_ref_id"):
                _fail("SOURCE_DECLARATION_MISMATCH",
                      "external anchor mapping claims undeclared attribution",
                      **context)
            provenance_source_ref_id = ""

        outcome = _recompute_mapping(mapping, verified.raw_value)
        expected_rule, expected_confidence = _outcome_context(mapping, outcome)
        result_source_ref_id = provenance_source_ref_id or verified.source_ref_id
        _verify_confidence(event, expected_confidence, mapping.op, spec.ref,
                           mapping.mapping_id, result_source_ref_id)
        _verify_rule(event, expected_rule, usage, self.manifest, mapping.op,
                     spec.ref, mapping.mapping_id, result_source_ref_id)
        provenance = mapping.provenance
        participant = str(
            self.delivered_row.get(spec.record.participant_key, ""))
        if provenance.record_key:
            expected_event_key = provenance.record_key
        elif (provenance.record_key_from == "participant"
              or provenance.source_ref):
            expected_event_key = _participant_record_key(
                spec.record, participant)
        else:
            expected_event_key = ""
        if attributed_binding is None:
            expected_origin = provenance.source_label or "ANCHOR"
            locator_mismatch = (
                event.get("source_file") != expected_origin
                or event.get("source_logical_label") != expected_origin
                or event.get("source_resolved_path")
                or event.get("source_sha256")
                or event.get("source_sheet")
                or event.get("source_ref_id"))
        else:
            locator_mismatch = (
                event.get("source_file") != attributed_binding.logical_label
                or event.get("source_logical_label")
                   != attributed_binding.logical_label
                or event.get("source_resolved_path") != attributed_binding.path
                or event.get("source_sha256") != attributed_binding.sha256
                or event.get("source_sheet") != attributed_binding.sheet
                or event.get("source_ref_id")
                   != attributed_binding.source_ref_id)
        if (locator_mismatch
                or event.get("source_column") != mapping.column
                or event.get("source_record_key") != expected_event_key
                or event.get("source_row_number")
                or event.get("derivation")
                or event.get("authoritative_for") != ";".join(mapping.artifacts)):
            _fail("SOURCE_DECLARATION_MISMATCH",
                  "external anchor provenance contradicts pinned origin policy",
                  source_ref_id=result_source_ref_id, **context)
        return outcome.value, result_source_ref_id

    def _source_group_binding(self, source_id: str, usage,
                              mapping_id: str) -> SourceBinding:
        """Resolve one exact usage-scoped group SourceBinding."""
        assert self.spec is not None
        context = {"spec_ref": self.spec.ref, "mapping_id": mapping_id}
        matches = [
            binding for binding in self.manifest.sources
            if binding.spec_ref == self.spec.ref
            and binding.source_id == source_id
        ]
        if not matches:
            _not_replayable(
                "SOURCE_GROUP_EVIDENCE_MISSING",
                f"source-group member {source_id!r} has no SourceBinding",
                **context)
        if len(matches) != 1:
            _fail(
                "SOURCE_GROUP_SELECTION_MISMATCH",
                f"source-group member {source_id!r} has ambiguous bindings",
                **context)
        binding = matches[0]
        if binding.source_ref_id not in usage.source_ref_ids:
            _not_replayable(
                "SOURCE_GROUP_EVIDENCE_MISSING",
                f"MappingUsage omits source-group member {source_id!r}",
                source_ref_id=binding.source_ref_id, **context)
        return binding

    def _indexed_group_member(self, binding: SourceBinding, source_decl,
                              mapping_id: str):
        """Verify and index one group member using the pinned duplicate policy."""
        assert self.spec is not None
        spec = self.spec
        context = {"spec_ref": spec.ref, "mapping_id": mapping_id,
                   "source_ref_id": binding.source_ref_id}
        _check_source_group_source_declaration(
            self.root, spec, source_decl, binding, mapping_id)
        verified = self._verify_source_identity(
            binding, source_decl, mapping_id)
        frame = verified.frame.copy()
        frame["_spec_src_row"] = frame.index
        participant_key = spec.record.participant_key
        join_column = source_decl.participant_column or participant_key
        if join_column not in frame.columns:
            _fail(
                "SOURCE_COLUMN_MISSING",
                f"group member source omits participant column {join_column!r}",
                **context)
        frame[join_column] = frame[join_column].astype(str)
        duplicate = spec.record.duplicates
        by = [
            join_column if name == participant_key else name
            for name in duplicate.by
        ] or [join_column]
        missing = [name for name in by if name not in frame.columns]
        if missing:
            _fail(
                "SOURCE_COLUMN_MISSING",
                f"group duplicate policy names missing columns {missing}",
                **context)
        if duplicate.policy in ("first", "last"):
            frame = frame.drop_duplicates(subset=by, keep=duplicate.policy)
        elif duplicate.policy == "error":
            if frame.duplicated(subset=by).any():
                _fail(
                    "SOURCE_GROUP_SELECTION_MISMATCH",
                    "group member source violates the pinned duplicate policy",
                    **context)
        else:
            _not_replayable(
                "OPERATION_UNSUPPORTED",
                f"duplicate policy {duplicate.policy!r} is outside Stage 12",
                **context)
        indexed = frame.set_index(join_column, drop=False)
        if indexed.index.has_duplicates:
            _fail(
                "SOURCE_GROUP_SELECTION_MISMATCH",
                "group member remains ambiguous after the pinned duplicate "
                "policy", **context)
        return indexed

    def _verify_source_group_scalar(
            self, mapping: SpecMapping, event, usage, *, exact_usage=True
            ) -> _VerifiedGroupScalar:
        """Independently reconstruct an ordered group decision and scalar."""
        assert self.spec is not None
        spec = self.spec
        context = {"spec_ref": spec.ref, "mapping_id": mapping.mapping_id}
        group = source_group_for_mapping(spec, mapping)
        if group is None:
            _not_replayable(
                "SOURCE_GROUP_EVIDENCE_MISSING",
                "mapping has no exact source-group declaration", **context)
        payload = event.get("contributions", "")
        if not payload:
            _not_replayable(
                "SOURCE_GROUP_EVIDENCE_MISSING",
                "source-group Cell has no membership Contributions", **context)
        try:
            contributions = contributions_from_json(payload)
        except (TypeError, ValueError, json.JSONDecodeError) as exc:
            _not_replayable(
                "SOURCE_GROUP_EVIDENCE_MISSING",
                f"membership Contributions are malformed: {exc}", **context)
        if len(contributions) < len(group.members):
            _not_replayable(
                "SOURCE_GROUP_EVIDENCE_MISSING",
                "source-group Cell omits an ordered membership Contribution",
                **context)
        if len(contributions) != len(group.members):
            _fail(
                "SOURCE_GROUP_SELECTION_MISMATCH",
                "source-group Cell has extra membership Contributions",
                **context)

        participant = str(
            self.delivered_row.get(spec.record.participant_key, ""))
        record_key = _participant_record_key(spec.record, participant)
        indexed_by_source = {}
        bindings_by_source = {}
        rows_by_member = {}
        presence = {}
        deciding_ref_ids = []

        for member in group.members:
            source_id = member.sources[group.decided_by]
            binding = self._source_group_binding(
                source_id, usage, mapping.mapping_id)
            source_decl = spec.source(source_id)
            indexed = self._indexed_group_member(
                binding, source_decl, mapping.mapping_id)
            indexed_by_source[source_id] = indexed
            bindings_by_source[source_id] = binding
            deciding_ref_ids.append(binding.source_ref_id)
            present = participant in indexed.index
            presence[member.id] = present
            rows_by_member[member.id] = (
                indexed.loc[participant] if present else None)

        decision = select_first_containing_record(
            tuple(member.id for member in group.members), presence,
            membership_status=group.membership_status)
        if decision.selected_rank is None:
            _fail(
                "SOURCE_GROUP_SELECTION_MISMATCH",
                "no declared source-group member contains the delivered record",
                **context)
        selected = group.members[decision.selected_rank]
        expected_role = membership_role(spec, group)
        expected_group_rule = group_rule_id(spec, group)

        for rank, (member, contribution) in enumerate(
                zip(group.members, contributions)):
            if not isinstance(contribution, dict):
                _not_replayable(
                    "SOURCE_GROUP_EVIDENCE_MISSING",
                    f"membership Contribution {rank} is not an object",
                    **context)
            source_id = member.sources[group.decided_by]
            binding = bindings_by_source[source_id]
            row = rows_by_member[member.id]
            expected_row = (
                int(row["_spec_src_row"]) if row is not None else "")
            expected_raw = participant if row is not None else ""
            required = (
                "role", "status", "rank", "source_ref_id",
                "source_record_key", "rule_id", "raw_value_json",
            )
            if any(name not in contribution for name in required):
                _not_replayable(
                    "SOURCE_GROUP_EVIDENCE_MISSING",
                    f"membership Contribution {rank} is incomplete", **context)
            try:
                recorded_typed = TypedRawValue.from_json(
                    contribution["raw_value_json"])
                expected_typed = encode_typed_raw(expected_raw)
            except ValueError as exc:
                _not_replayable(
                    "SOURCE_GROUP_EVIDENCE_MISSING",
                    f"membership Contribution {rank} has invalid typed "
                    f"evidence: {exc}", **context)
            if (contribution.get("role") != expected_role
                    or contribution.get("status") != decision.statuses[rank]
                    or contribution.get("rank") != rank
                    or contribution.get("source_column") != ""
                    or contribution.get("source_ref_id")
                       != binding.source_ref_id
                    or contribution.get("source_logical_label")
                       != binding.logical_label
                    or contribution.get("source_resolved_path") != binding.path
                    or contribution.get("source_sha256") != binding.sha256
                    or contribution.get("source_sheet") != binding.sheet
                    or contribution.get("source_record_key") != record_key
                    or contribution.get("source_row_number") != expected_row
                    or contribution.get("rule_id") != expected_group_rule
                    or contribution.get("raw_value") != expected_raw
                    or recorded_typed != expected_typed):
                _fail(
                    "SOURCE_GROUP_SELECTION_MISMATCH",
                    f"membership Contribution {rank} contradicts the "
                    "independently reconstructed group decision",
                    source_ref_id=binding.source_ref_id, **context)

        scalar_role = scalar_role_for_mapping(mapping, group)
        scalar_source_id = selected.sources[scalar_role]
        scalar_binding = self._source_group_binding(
            scalar_source_id, usage, mapping.mapping_id)
        scalar_decl = spec.source(scalar_source_id)
        if scalar_source_id in indexed_by_source:
            scalar_indexed = indexed_by_source[scalar_source_id]
        else:
            scalar_indexed = self._indexed_group_member(
                scalar_binding, scalar_decl, mapping.mapping_id)
        if participant not in scalar_indexed.index:
            _fail(
                "SOURCE_GROUP_SELECTION_MISMATCH",
                "selected member's scalar-role source omits the record",
                source_ref_id=scalar_binding.source_ref_id, **context)
        scalar_row = scalar_indexed.loc[participant]
        scalar_row_number = int(scalar_row["_spec_src_row"])
        source_column = (
            mapping.column_by_member[selected.id]
            if mapping.column_by_member else (mapping.column or mapping.target))
        if (mapping.op != OP_FILE_MEMBERSHIP
                and source_column not in scalar_row.index):
            _fail(
                "SOURCE_COLUMN_MISSING",
                f"selected group scalar column {source_column!r} is absent",
                source_ref_id=scalar_binding.source_ref_id, **context)
        raw_value = (
            participant if mapping.op == OP_FILE_MEMBERSHIP
            else scalar_row[source_column])

        expected_sources = set(deciding_ref_ids)
        expected_sources.add(scalar_binding.source_ref_id)
        actual_sources = set(usage.source_ref_ids)
        missing_sources = expected_sources - actual_sources
        if missing_sources:
            _not_replayable(
                "SOURCE_GROUP_EVIDENCE_MISSING",
                f"MappingUsage omits group source links "
                f"{sorted(missing_sources)}", **context)
        if exact_usage and actual_sources != expected_sources:
            _fail(
                "SOURCE_GROUP_SELECTION_MISMATCH",
                "MappingUsage contains sources outside the exact group decision "
                "and selected scalar route", **context)

        event_source_ref = event.get("source_ref_id", "")
        if not event_source_ref:
            _not_replayable(
                "SOURCE_GROUP_EVIDENCE_MISSING",
                "Cell omits the selected scalar-role source link", **context)
        if event_source_ref != scalar_binding.source_ref_id:
            _fail(
                "SOURCE_GROUP_SELECTION_MISMATCH",
                "Cell source link contradicts the selected scalar role",
                source_ref_id=scalar_binding.source_ref_id, **context)
        locator_text = event.get("source_row_number", "")
        if locator_text == "":
            _not_replayable(
                "SOURCE_GROUP_EVIDENCE_MISSING",
                "Cell omits the selected scalar source-row locator",
                source_ref_id=scalar_binding.source_ref_id, **context)
        try:
            recorded_row_number = int(locator_text)
        except (TypeError, ValueError):
            _fail(
                "SOURCE_GROUP_SELECTION_MISMATCH",
                "Cell selected scalar row locator is not an integer",
                source_ref_id=scalar_binding.source_ref_id, **context)
        if (recorded_row_number != scalar_row_number
                or str(recorded_row_number) != str(locator_text)):
            _fail(
                "SOURCE_GROUP_SELECTION_MISMATCH",
                "Cell selected scalar row contradicts the reconstructed row",
                source_ref_id=scalar_binding.source_ref_id, **context)

        if not event.get("source_record_key"):
            _not_replayable(
                "SOURCE_GROUP_EVIDENCE_MISSING",
                "Cell omits the selected scalar record key",
                source_ref_id=scalar_binding.source_ref_id, **context)

        _verify_record_key(
            event.get("source_record_key", ""), spec, scalar_decl, scalar_row,
            self.delivered_row, spec.ref, mapping.mapping_id,
            scalar_binding.source_ref_id)
        _verify_declared_record_keys(
            scalar_decl, scalar_row, self.delivered_row, spec.ref,
            mapping.mapping_id, scalar_binding.source_ref_id)
        if (event.get("source_file") != scalar_binding.logical_label
                or event.get("source_logical_label")
                   != scalar_binding.logical_label
                or event.get("source_resolved_path") != scalar_binding.path
                or event.get("source_sha256") != scalar_binding.sha256
                or event.get("source_sheet") != scalar_binding.sheet
                or event.get("source_column") != source_column):
            _fail(
                "SOURCE_DECLARATION_MISMATCH",
                "Cell provenance contradicts the selected scalar source",
                source_ref_id=scalar_binding.source_ref_id, **context)

        return _VerifiedGroupScalar(
            group_id=group.id,
            selected_member_id=selected.id,
            selected_rank=decision.selected_rank,
            source_ref_id=scalar_binding.source_ref_id,
            source_row_number=scalar_row_number,
            source_column=source_column,
            raw_value=raw_value,
            deciding_source_ref_ids=tuple(deciding_ref_ids),
            selected_member_value=selected.value,
            selected_deciding_source_ref_id=(
                bindings_by_source[
                    selected.sources[group.decided_by]].source_ref_id),
            selected_deciding_logical_label=(
                bindings_by_source[
                    selected.sources[group.decided_by]].logical_label),
        )

    def _replay_source_group_operation(
            self, mapping: SpecMapping, event, usage, *, exact_usage=True,
            verify_file_derivation=True,
            ) -> tuple[VerifiedOperationRouteResult, str, _VerifiedGroupScalar]:
        """Replay the approved basic operations after independent group choice."""
        assert self.spec is not None
        spec = self.spec
        context = {"spec_ref": spec.ref, "mapping_id": mapping.mapping_id}
        _check_source_group_scope(mapping, spec, event, self.artifact)
        if mapping.op == OP_VALUE_MAP:
            _check_stage2_environment(
                self.manifest, spec.ref, mapping.mapping_id)
        verified = self._verify_source_group_scalar(
            mapping, event, usage, exact_usage=exact_usage)
        group = source_group_for_mapping(spec, mapping)
        selected = next(
            member for member in group.members
            if member.id == verified.selected_member_id)

        vocabulary = None
        if mapping.op == OP_VALUE_MAP and mapping.fallback_vocabulary:
            vocabulary = _verified_vocabulary(
                self.root, self.manifest, event, usage,
                expected_ref=mapping.fallback_vocabulary,
                expected_kind=vocabularies.KIND_ORDERED_RULES,
                spec_ref=spec.ref, mapping_id=mapping.mapping_id)
        try:
            if mapping.op == OP_FILE_MEMBERSHIP:
                value = recompute_file_membership(
                    verified.selected_member_id,
                    member_values={
                        member.id: member.value for member in group.members})
                outcome = RecomputedOutcome(
                    value, _OUTCOME_NO_VALUE if _is_missing(value)
                    else _OUTCOME_VALUE)
            elif vocabulary is not None:
                outcome = recompute_vocabulary_value_map(
                    verified.raw_value, mapping_table=mapping.map,
                    ordered_rules=vocabulary.terms,
                    on_unmapped=mapping.on_unmapped)
            else:
                outcome = _recompute_mapping(mapping, verified.raw_value)
        except _ValueMapUnmappedError as exc:
            _fail(
                "VALUE_MAP_UNMAPPED_ERROR",
                f"provenance claims value for error branch: {exc}", **context)
        except ValueError as exc:
            _not_replayable("OPERATION_UNSUPPORTED", str(exc), **context)

        expected_rule, expected_confidence = _outcome_context(mapping, outcome)
        _verify_confidence(
            event, expected_confidence, mapping.op, spec.ref,
            mapping.mapping_id, verified.source_ref_id)
        _verify_rule(
            event, expected_rule, usage, self.manifest, mapping.op,
            spec.ref, mapping.mapping_id, verified.source_ref_id)
        if mapping.op == OP_FILE_MEMBERSHIP and verify_file_derivation:
            expected_derivation = json.dumps({
                "kind": "file_membership",
                "rule_id": group_rule_id(spec, group),
                "candidates": [
                    self._source_group_binding(
                        member.sources[group.decided_by], usage,
                        mapping.mapping_id).logical_label
                    for member in group.members
                ],
                "matched": self._source_group_binding(
                    selected.sources[group.decided_by], usage,
                    mapping.mapping_id).logical_label,
                "matched_config_key": selected.sources[group.decided_by],
                "final_value": outcome.value,
                "map_ref": (
                    group.evidence_ref
                    or f"spec:{spec.spec_id}.source_groups.{group.id}"),
            }, separators=(",", ":"), sort_keys=True)
            if event.get("derivation", "") != expected_derivation:
                _fail(
                    "SOURCE_GROUP_SELECTION_MISMATCH",
                    "file-membership derivation contradicts the reconstructed "
                    "group decision", source_ref_id=verified.source_ref_id,
                    **context)
        elif event.get("derivation"):
            _not_replayable(
                "DERIVATION_UNSUPPORTED_STAGE2",
                "derived group scalar is outside Stage 12",
                source_ref_id=verified.source_ref_id, **context)

        return (_route_result(outcome.value, verified.raw_value),
                verified.source_ref_id, verified)

    def _verify_stage15_parent_scalar(
            self, mapping: SpecMapping, event, usage
            ) -> tuple[_VerifiedParticipantRow | None, Any, str]:
        """Verify a conditional mapping's own raw source before branch choice."""
        assert self.spec is not None
        context = {"spec_ref": self.spec.ref,
                   "mapping_id": mapping.mapping_id}
        if not mapping.source:
            if event.get("source_ref_id"):
                _fail(
                    "SOURCE_DECLARATION_MISMATCH",
                    "source-independent conditional claims a source link",
                    **context)
            return None, None, ""
        if GROUP_REF_SEP in mapping.source or mapping.source == ANCHOR_SOURCE_ID:
            _not_replayable(
                "OPERATION_UNSUPPORTED",
                "conditional parent source is outside Stage 15", **context)
        verified = self._stage15_participant_row(
            mapping.source, usage, mapping.mapping_id,
            missing_reason="NAMED_READ_EVIDENCE_MISSING")
        binding = verified.binding
        context["source_ref_id"] = binding.source_ref_id
        if event.get("source_ref_id") != binding.source_ref_id:
            _not_replayable(
                "SOURCE_LINK_MISSING",
                "conditional Cell source link disagrees with MappingUsage",
                **context)
        _check_source_declaration(
            self.root, self.spec, mapping, event,
            verified.declaration, binding)
        row_number = _source_row_number(
            event, self.spec.ref, mapping.mapping_id,
            binding.source_ref_id)
        if row_number != verified.row_number:
            _fail(
                "SOURCE_RECORD_KEY_MISMATCH",
                "conditional row locator contradicts participant lookup",
                **context)
        column = mapping.column or mapping.target
        if column not in verified.row.index:
            _fail(
                "SOURCE_COLUMN_MISSING",
                f"conditional source column {column!r} is absent", **context)
        _verify_record_key(
            event.get("source_record_key", ""), self.spec,
            verified.declaration, verified.row, self.delivered_row,
            self.spec.ref, mapping.mapping_id, binding.source_ref_id)
        _verify_declared_record_keys(
            verified.declaration, verified.row, self.delivered_row,
            self.spec.ref, mapping.mapping_id, binding.source_ref_id)
        raw = verified.row[column]
        return verified, raw, binding.source_ref_id

    def _verify_stage17_group_parent_scalar(
            self, mapping: SpecMapping, event, usage,
            contributions: list[dict], groups_cache: dict
            ) -> tuple[_VerifiedParticipantRow, Any, str]:
        """Verify a conditional parent scalar from one cached group decision."""
        assert self.spec is not None
        context = {"spec_ref": self.spec.ref,
                   "mapping_id": mapping.mapping_id}
        group_id, separator, role = mapping.source.partition(GROUP_REF_SEP)
        if not separator:
            _not_replayable(
                "OPERATION_UNSUPPORTED",
                "Stage 17 group-parent route lacks a group source", **context)
        if group_id not in groups_cache:
            groups_cache[group_id] = self._stage15_group_member(
                group_id, event, usage, mapping.mapping_id, contributions)
        decision = groups_cache[group_id]
        source_id = decision.selected_member.sources.get(role, "")
        if not source_id:
            _not_replayable(
                "SOURCE_GROUP_EVIDENCE_MISSING",
                f"selected member offers no parent role {role!r}", **context)
        verified = self._stage15_participant_row(
            source_id, usage, mapping.mapping_id,
            missing_reason="SOURCE_GROUP_EVIDENCE_MISSING")
        binding = verified.binding
        context["source_ref_id"] = binding.source_ref_id
        if event.get("source_ref_id") != binding.source_ref_id:
            _fail(
                "SOURCE_GROUP_SELECTION_MISMATCH",
                "conditional parent source link contradicts selected member",
                **context)
        column = (
            mapping.column_by_member.get(
                decision.selected_member.id, mapping.column)
            if mapping.column_by_member else (mapping.column or mapping.target))
        if not column or column not in verified.row.index:
            _fail(
                "SOURCE_COLUMN_MISSING",
                f"conditional group-parent column {column!r} is absent",
                **context)
        _verify_record_key(
            event.get("source_record_key", ""), self.spec,
            verified.declaration, verified.row, self.delivered_row,
            self.spec.ref, mapping.mapping_id, binding.source_ref_id)
        _verify_declared_record_keys(
            verified.declaration, verified.row, self.delivered_row,
            self.spec.ref, mapping.mapping_id, binding.source_ref_id)
        if (event.get("source_file") != binding.logical_label
                or event.get("source_logical_label") != binding.logical_label
                or event.get("source_resolved_path") != binding.path
                or event.get("source_sha256") != binding.sha256
                or event.get("source_sheet") != binding.sheet
                or event.get("source_column") != column
                or event.get("source_row_number") != str(verified.row_number)):
            _fail(
                "SOURCE_DECLARATION_MISMATCH",
                "conditional group-parent provenance contradicts selected "
                "source context", **context)
        return verified, verified.row[column], binding.source_ref_id

    def _replay_conditional(
            self, mapping: SpecMapping, event, usage, *, reads_cache=None,
            groups_cache=None
            ) -> tuple[VerifiedOperationRouteResult, str,
                       dict[str, _VerifiedNamedRead],
                       dict[str, _VerifiedGroupDecision]]:
        """Replay the closed first-match scalar conditional subset."""
        assert self.spec is not None
        spec = self.spec
        context = {"spec_ref": spec.ref, "mapping_id": mapping.mapping_id}
        _check_stage15_conditional_scope(
            mapping, spec, event, self.artifact)
        _check_stage2_environment(
            self.manifest, spec.ref, mapping.mapping_id)
        contributions = self._stage15_contributions(
            event, mapping.mapping_id)
        groups_cache = dict(groups_cache or {})
        reads_cache = dict(reads_cache or {})
        declared_condition_reads = list(mapping.contributions.conditions)
        if raw_from_is_causal(
                raw_from_id=mapping.provenance.raw_from,
                raw_policy=mapping.provenance.raw_value):
            if mapping.provenance.raw_from not in declared_condition_reads:
                declared_condition_reads.append(mapping.provenance.raw_from)
        declared_condition_reads = tuple(declared_condition_reads)
        if len(contributions) < len(declared_condition_reads):
            _not_replayable(
                "NAMED_READ_EVIDENCE_MISSING",
                "conditional omits declared named-read Contributions", **context)
        split = len(contributions) - len(declared_condition_reads)
        membership_contributions = contributions[:split]
        condition_contributions = contributions[split:]
        expected_order = [
            (read_id, STATUS_CONDITION,
             mapping.contributions.condition_rank)
            for read_id in declared_condition_reads
        ]
        actual_order = [
            (item.get("role"), item.get("status"), item.get("rank"))
            for item in condition_contributions
        ]
        if actual_order != expected_order:
            _fail(
                "NAMED_READ_MISMATCH",
                "conditional named-read Contributions contradict declared order",
                **context)
        if membership_contributions and not mapping.provenance.contributions_from:
            _fail(
                "NAMED_READ_MISMATCH",
                "conditional has undeclared non-condition Contributions",
                **context)
        if GROUP_REF_SEP in mapping.source:
            parent, route_raw, result_source = \
                self._verify_stage17_group_parent_scalar(
                    mapping, event, usage, contributions, groups_cache)
        else:
            parent, route_raw, result_source = \
                self._verify_stage15_parent_scalar(mapping, event, usage)

        required_reads = []
        for clause in mapping.clauses:
            if clause.when.read and clause.when.read not in required_reads:
                required_reads.append(clause.when.read)
            if clause.read and clause.read not in required_reads:
                required_reads.append(clause.read)
        for read_id in mapping.contributions.conditions:
            if read_id not in required_reads:
                required_reads.append(read_id)
        for read_id in declared_condition_reads:
            if read_id not in required_reads:
                required_reads.append(read_id)
        for read_id in required_reads:
            if read_id not in reads_cache:
                reads_cache[read_id] = self._stage15_named_read(
                    read_id, mapping, event, usage, contributions,
                    groups_cache=groups_cache)
        named = reads_cache

        results = []
        selected_rank = None
        for rank, clause in enumerate(mapping.clauses):
            predicate = clause.when
            if predicate.read:
                operand = named[predicate.read].value
            else:
                if predicate.source != mapping.source or parent is None:
                    _not_replayable(
                        "NAMED_READ_EVIDENCE_MISSING",
                        "direct conditional predicate is not bound to the "
                        "verified parent source", **context)
                if predicate.column not in parent.row.index:
                    _fail(
                        "SOURCE_COLUMN_MISSING",
                        f"conditional predicate column "
                        f"{predicate.column!r} is absent",
                        source_ref_id=parent.binding.source_ref_id, **context)
                operand = parent.row[predicate.column]
            result = evaluate_predicate(
                predicate.predicate, operand, predicate.value)
            results.append(result)
            if result:
                selected_rank = first_true_index(results)
                break

        if selected_rank is None:
            value = mapping.otherwise
        else:
            clause = mapping.clauses[selected_rank]
            if clause.read:
                value = named[clause.read].value
            elif clause.column:
                if clause.source != mapping.source or parent is None:
                    _not_replayable(
                        "NAMED_READ_EVIDENCE_MISSING",
                        "conditional yield column is not bound to the verified "
                        "parent source", **context)
                if clause.column not in parent.row.index:
                    _fail(
                        "SOURCE_COLUMN_MISSING",
                        f"conditional yield column {clause.column!r} is absent",
                        source_ref_id=parent.binding.source_ref_id, **context)
                value = parent.row[clause.column]
            else:
                value = clause.value

        expected_sources = set()
        if parent is not None:
            expected_sources.add(parent.binding.source_ref_id)
        expected_sources.update(item.source_ref_id for item in named.values())
        expected_sources.update(
            str(item.get("source_ref_id") or "") for item in contributions
            if item.get("source_ref_id"))
        actual_sources = set(usage.source_ref_ids)
        if not expected_sources.issubset(actual_sources):
            _not_replayable(
                "NAMED_READ_EVIDENCE_MISSING",
                "conditional MappingUsage omits causal sources", **context)
        if actual_sources != expected_sources:
            _fail(
                "NAMED_READ_MISMATCH",
                "conditional MappingUsage contains unrelated sources",
                **context)

        outcome = RecomputedOutcome(
            value, _OUTCOME_NO_VALUE if _is_missing(value)
            else _OUTCOME_VALUE)
        expected_rule, expected_confidence = _outcome_context(
            mapping, outcome)
        _verify_confidence(
            event, expected_confidence, mapping.op, spec.ref,
            mapping.mapping_id, result_source)
        _verify_rule(
            event, expected_rule, usage, self.manifest, mapping.op,
            spec.ref, mapping.mapping_id, result_source)
        return (_route_result(value, route_raw), result_source, named,
                groups_cache)

    def _replay_scalar(
            self, mapping: SpecMapping, event, usage, *,
            expected_source_row: int | None = None,
            record_key_verified: bool = False,
            ) -> tuple[VerifiedOperationRouteResult, str]:
        assert self.spec is not None
        spec = self.spec
        _check_operation_scope(mapping, spec, event, self.artifact)
        if mapping.op in (OP_VALUE_MAP, OP_UNIT_CONVERSION):
            _check_stage2_environment(self.manifest, spec.ref, mapping.mapping_id)
        vocabulary = None
        if mapping.op == OP_VALUE_MAP and mapping.fallback_vocabulary:
            vocabulary = _verified_vocabulary(
                self.root, self.manifest, event, usage,
                expected_ref=mapping.fallback_vocabulary,
                expected_kind=vocabularies.KIND_ORDERED_RULES,
                spec_ref=spec.ref, mapping_id=mapping.mapping_id)
        if len(usage.source_ref_ids) != 1:
            _not_replayable("SOURCE_NOT_SINGLE_FILE",
                            "strict scalar replay requires exactly one source",
                            spec_ref=spec.ref, mapping_id=mapping.mapping_id)
        source_ref_id = usage.source_ref_ids[0]
        sources = [s for s in self.manifest.sources
                   if s.source_ref_id == source_ref_id]
        if len(sources) != 1:
            _not_replayable("SOURCE_LINK_MISSING",
                            "manifest source reference cannot be resolved",
                            spec_ref=spec.ref, mapping_id=mapping.mapping_id)
        source_binding = sources[0]
        context = {"spec_ref": spec.ref, "mapping_id": mapping.mapping_id,
                   "source_ref_id": source_ref_id}
        if event.get("source_ref_id") != source_ref_id:
            _not_replayable("SOURCE_LINK_MISSING",
                            "provenance source_ref_id disagrees with usage",
                            **context)
        source_decl = spec.source(mapping.source)
        _check_source_declaration(
            self.root, spec, mapping, event, source_decl, source_binding)
        verified = self._verify_source_identity(
            source_binding, source_decl, mapping.mapping_id)
        row_number = _source_row_number(
            event, spec.ref, mapping.mapping_id, source_ref_id)
        if (expected_source_row is not None
                and row_number != expected_source_row):
            _fail(
                "ANCHOR_SELECTION_MISMATCH",
                "recorded source row contradicts independently selected anchor",
                **context)
        if expected_source_row is not None:
            row_number = expected_source_row
        if source_binding.reader.reader_id == XLSX_READER_ID:
            reconstructed_row = _xlsx_source_row_number(
                verified.frame, spec, source_decl, self.delivered_row,
                mapping.mapping_id, source_ref_id)
            if reconstructed_row != row_number:
                _fail("SOURCE_RECORD_KEY_MISMATCH",
                      "recorded XLSX row differs from independent participant lookup",
                      **context)
        if row_number >= len(verified.frame):
            _fail("SOURCE_ROW_OUT_OF_RANGE",
                  f"source row {row_number} is outside "
                  f"{len(verified.frame)} rows",
                  **context)
        if mapping.column not in verified.frame.columns:
            _fail("SOURCE_COLUMN_MISSING",
                  f"source column {mapping.column!r} is absent", **context)
        source_row = verified.frame.iloc[row_number]
        if not record_key_verified:
            _verify_record_key(
                event.get("source_record_key", ""), spec, source_decl,
                source_row, self.delivered_row, spec.ref,
                mapping.mapping_id, source_ref_id)
        _verify_declared_record_keys(source_decl, source_row, self.delivered_row,
                                     spec.ref, mapping.mapping_id, source_ref_id)
        raw_source_value = source_row[mapping.column]
        try:
            outcome = (
                recompute_vocabulary_value_map(
                    raw_source_value, mapping_table=mapping.map,
                    ordered_rules=vocabulary.terms,
                    on_unmapped=mapping.on_unmapped)
                if vocabulary is not None else
                _recompute_mapping(mapping, raw_source_value)
            )
        except _ValueMapUnmappedError as exc:
            _fail("VALUE_MAP_UNMAPPED_ERROR",
                  f"provenance claims value for error branch: {exc}", **context)
        except _NonFiniteUnitConversionResult as exc:
            _fail("UNIT_CONVERSION_NONFINITE_RESULT", str(exc), **context)
        expected_rule, expected_confidence = _outcome_context(mapping, outcome)
        _verify_confidence(event, expected_confidence, mapping.op, spec.ref,
                           mapping.mapping_id, source_ref_id)
        _verify_rule(event, expected_rule, usage, self.manifest, mapping.op,
                     spec.ref, mapping.mapping_id, source_ref_id)
        return _route_result(outcome.value, raw_source_value), source_ref_id

    def _stage16_condition(
            self, condition_id: str, mapping: SpecMapping, event, usage,
            contributions: list[dict], reads_cache: dict,
            groups_cache: dict, conditions_cache: dict,
            constructed_record: _VerifiedConstructedRecord | None = None
            ) -> bool:
        """Reconstruct one Stage 15 condition without recorded branch authority."""
        if condition_id in conditions_cache:
            return conditions_cache[condition_id]
        condition = next(
            (item for item in self.spec.conditions if item.id == condition_id),
            None)
        context = {"spec_ref": self.spec.ref,
                   "mapping_id": mapping.mapping_id}
        if condition is None:
            _not_replayable(
                "DERIVATION_EVIDENCE_MISSING",
                f"derivation condition {condition_id!r} is not pinned", **context)
        if condition.read:
            if condition.read not in reads_cache:
                reads_cache[condition.read] = self._stage15_named_read(
                    condition.read, mapping, event, usage, contributions,
                    groups_cache=groups_cache)
            result = evaluate_predicate(
                condition.predicate, reads_cache[condition.read].value,
                condition.value)
        elif condition.group:
            if condition.group not in groups_cache:
                groups_cache[condition.group] = self._stage15_group_member(
                    condition.group, event, usage, mapping.mapping_id,
                    contributions)
            result = (
                groups_cache[condition.group].selected_member.id
                == condition.member)
        elif condition.record_origin:
            if constructed_record is None:
                _not_replayable(
                    "ANCHOR_EVIDENCE_MISSING",
                    "record-origin derivation condition lacks verified "
                    "construction", **context)
            result = condition.record_origin == constructed_record.origin
        else:
            _not_replayable(
                "MAPPING_CONDITION_UNSUPPORTED",
                "record-origin derivation conditions are outside Stage 16",
                **context)
        conditions_cache[condition_id] = result
        return result

    def _stage16_derivation_facts(
            self, mapping: SpecMapping, event, usage, *,
            reads_cache=None, groups_cache=None, conditions_cache=None,
            constructed_record: _VerifiedConstructedRecord | None = None
            ) -> _VerifiedDerivationFacts:
        """Compose selected-template facts from existing strict replay results."""
        context = {"spec_ref": self.spec.ref,
                   "mapping_id": mapping.mapping_id}
        if not (_is_stage16_derivation_mapping(mapping, self.spec)
                or _is_stage17_derivation_mapping(mapping, self.spec)
                or _is_stage18_derivation_mapping(mapping, self.spec)
                or _is_stage19_derivation_mapping(mapping, self.spec)):
            _not_replayable(
                "DERIVATION_UNSUPPORTED_STAGE2",
                "structured derivation is outside Stage 16 Option C", **context)
        contributions = self._stage15_contributions(event, mapping.mapping_id)
        reads_cache = dict(reads_cache or {})
        groups_cache = dict(groups_cache or {})
        conditions_cache = dict(conditions_cache or {})

        selected_record = None
        for variant in mapping.derivation:
            if variant.when and not self._stage16_condition(
                    variant.when, mapping, event, usage, contributions,
                    reads_cache, groups_cache, conditions_cache,
                    constructed_record):
                continue
            selected_record = variant.record
            break
        refs = (derivation_references(selected_record)
                if selected_record is not None else frozenset())
        facts: dict[str, Any] = {}
        for ref in sorted(refs):
            head, _, name = ref.partition(".")
            if ref == "raw_value":
                continue
            if ref in STAGE14_FACTS:
                continue
            if head == "read":
                if name not in reads_cache:
                    reads_cache[name] = self._stage15_named_read(
                        name, mapping, event, usage, contributions,
                        groups_cache=groups_cache)
                facts[ref] = reads_cache[name].value
            elif head == "condition":
                facts[ref] = self._stage16_condition(
                    name, mapping, event, usage, contributions,
                    reads_cache, groups_cache, conditions_cache,
                    constructed_record)
            elif head in ("group_member", "group_value", "group_label"):
                if name not in groups_cache:
                    groups_cache[name] = self._stage15_group_member(
                        name, event, usage, mapping.mapping_id, contributions)
                verified = groups_cache[name]
                if head == "group_member":
                    facts[ref] = verified.selected_member.id
                elif head == "group_value":
                    facts[ref] = verified.selected_member.value
                else:
                    facts[ref] = \
                        verified.selected_deciding_binding.logical_label
            elif ref in {"anchor_group_size", "anchor_selected_row"}:
                if (constructed_record is None
                        or constructed_record.origin != ORIGIN_OBSERVED):
                    _not_replayable(
                        "DERIVATION_EVIDENCE_MISSING",
                        f"grouping fact {ref!r} lacks a verified grouped record",
                        **context)
                facts[ref] = (
                    constructed_record.group_size
                    if ref == "anchor_group_size"
                    else constructed_record.selected_source_row)
            else:
                _not_replayable(
                    "DERIVATION_UNSUPPORTED_STAGE2",
                    f"derivation fact {ref!r} is outside Stage 16", **context)
        return _VerifiedDerivationFacts(
            facts=facts, conditions=conditions_cache, reads=reads_cache,
            groups=groups_cache, selected_record=selected_record)

    def _verify_stage16_derivation(
            self, mapping: SpecMapping, event: dict[str, str], recomputed: Any,
            effective_source_column: str,
            verified: _VerifiedDerivationFacts, *, finalized_raw=...) -> None:
        """Render selected Stage 16 facts and compare recorded JSON last."""
        context = {
            "spec_ref": self.spec.ref, "mapping_id": mapping.mapping_id,
            "source_ref_id": event.get("source_ref_id", ""),
        }
        facts = dict(verified.facts)
        selected_refs = (
            derivation_references(verified.selected_record)
            if verified.selected_record is not None else frozenset())
        if "value" in selected_refs:
            facts["value"] = recomputed
        if "source_column" in selected_refs:
            if not effective_source_column:
                _not_replayable(
                    "DERIVATION_EVIDENCE_MISSING",
                    "effective source-column fact is unavailable", **context)
            facts["source_column"] = effective_source_column
        if "raw_value" in selected_refs:
            if finalized_raw is ...:
                _not_replayable(
                    "DERIVATION_EVIDENCE_MISSING",
                    "finalized raw fact is unavailable", **context)
            facts["raw_value"] = finalized_raw
        try:
            expected = render_structured_derivation(
                mapping.derivation, facts, verified.conditions)
        except MissingDerivationFact as exc:
            _not_replayable(
                "DERIVATION_EVIDENCE_MISSING", str(exc), **context)
        except DerivationError as exc:
            _not_replayable(
                "DERIVATION_EVIDENCE_MISSING",
                f"derivation cannot be rendered from verified facts: {exc}",
                **context)
        payload = event.get("derivation", "")
        if not expected:
            if payload:
                _fail(
                    "DERIVATION_MISMATCH",
                    "recorded derivation exists although no variant applies",
                    **context)
            return
        if not payload:
            _not_replayable(
                "DERIVATION_EVIDENCE_MISSING",
                "recorded structured derivation is absent", **context)
        try:
            recorded = parse_canonical_derivation_json(payload)
        except DerivationError as exc:
            _fail("DERIVATION_MISMATCH", str(exc), **context)
        if expected != recorded:
            _fail(
                "DERIVATION_MISMATCH",
                "recorded derivation contradicts independent fact projection",
                **context)

    def _replay_stage16_constant(
            self, mapping: SpecMapping, event, usage,
            verified: _VerifiedDerivationFacts) -> tuple[Any, str, str]:
        """Reuse constant recomputation with separately verified source context."""
        context = {"spec_ref": self.spec.ref,
                   "mapping_id": mapping.mapping_id}
        _check_yaml_declaration_environment(
            self.manifest, self.spec.ref, mapping.mapping_id)
        try:
            outcome = recompute_constant(declared_value=mapping.value)
        except ValueError as exc:
            _not_replayable("OPERATION_UNSUPPORTED", str(exc), **context)
        if event.get("extraction_method") != "spec:constant":
            _fail(
                "SOURCE_DECLARATION_MISMATCH",
                "provenance extraction_method is not spec:constant", **context)
        raw_from = mapping.provenance.raw_from
        if not raw_from or raw_from not in verified.reads:
            _not_replayable(
                "DERIVATION_EVIDENCE_MISSING",
                "sourced Stage 16 constant lacks its verified raw read", **context)
        named = verified.reads[raw_from]
        try:
            recorded_raw = TypedRawValue.from_json(event.get("raw_value_json", ""))
            expected_raw = encode_typed_raw(named.value)
        except ValueError as exc:
            _not_replayable("RAW_VALUE_KIND_UNSUPPORTED", str(exc), **context)
        if recorded_raw != expected_raw:
            _fail(
                "RAW_VALUE_MISMATCH",
                "constant raw snapshot differs from the verified named read",
                **context)

        bindings = [item for item in self.manifest.sources
                    if item.source_ref_id == named.source_ref_id]
        if len(bindings) != 1:
            _not_replayable(
                "SOURCE_LINK_MISSING",
                "sourced constant binding cannot be resolved", **context)
        binding = bindings[0]
        expected_column = mapping.column or mapping.target
        if named.selected_member_id and mapping.column_by_member:
            expected_column = mapping.column_by_member[named.selected_member_id]
        participant = str(
            self.delivered_row.get(self.spec.record.participant_key, ""))
        expected_key = _participant_record_key(self.spec.record, participant)
        if (event.get("source_ref_id") != binding.source_ref_id
                or event.get("source_file") != binding.logical_label
                or event.get("source_logical_label") != binding.logical_label
                or event.get("source_resolved_path") != binding.path
                or event.get("source_sha256") != binding.sha256
                or event.get("source_sheet") != binding.sheet
                or event.get("source_column") != expected_column
                or event.get("source_row_number") != str(named.row_number)
                or event.get("source_record_key") != expected_key):
            _fail(
                "SOURCE_DECLARATION_MISMATCH",
                "sourced constant provenance contradicts verified read context",
                source_ref_id=binding.source_ref_id, **context)
        expected_rule, expected_confidence = _outcome_context(mapping, outcome)
        _verify_confidence(
            event, expected_confidence, mapping.op, self.spec.ref,
            mapping.mapping_id, binding.source_ref_id)
        _verify_rule(
            event, expected_rule, usage, self.manifest, mapping.op,
            self.spec.ref, mapping.mapping_id, binding.source_ref_id)
        return outcome.value, binding.source_ref_id, expected_column

    def _verify_stage16_usage_closure(
            self, mapping: SpecMapping, usage,
            verified: _VerifiedDerivationFacts, *, source_ref_id: str = "",
            group_scalar: _VerifiedGroupScalar | None = None,
            construction_source_ref_ids: tuple[str, ...] = ()) -> None:
        """Require MappingUsage to equal the union of verified causal sources."""
        expected = {source_ref_id} if source_ref_id else set()
        expected.update(item.source_ref_id for item in verified.reads.values())
        for group in verified.groups.values():
            expected.update(group.deciding_source_ref_ids)
        if group_scalar is not None:
            expected.update(group_scalar.deciding_source_ref_ids)
            expected.add(group_scalar.source_ref_id)
        expected.update(construction_source_ref_ids)
        actual = set(usage.source_ref_ids)
        if not expected.issubset(actual):
            _not_replayable(
                "NAMED_READ_EVIDENCE_MISSING",
                "MappingUsage omits Stage 16 causal source links",
                spec_ref=self.spec.ref, mapping_id=mapping.mapping_id)
        if actual != expected:
            _fail(
                "NAMED_READ_MISMATCH",
                "MappingUsage contains sources outside the exact Stage 16 "
                "causal closure", spec_ref=self.spec.ref,
                mapping_id=mapping.mapping_id)

    def _verify_stage14_derivation(
            self, mapping: SpecMapping, event: dict[str, str],
            recomputed: Any) -> None:
        """Verify one closed fact projection after operation recomputation."""
        assert self.spec is not None
        context = {
            "spec_ref": self.spec.ref,
            "mapping_id": mapping.mapping_id,
            "source_ref_id": event.get("source_ref_id", ""),
        }
        if not _is_stage14_derivation_mapping(mapping):
            code = ("DERIVATION_UNSUPPORTED_STAGE1"
                    if mapping.op == OP_DIRECT
                    else "DERIVATION_UNSUPPORTED_STAGE2")
            _not_replayable(
                code, "structured derivation is outside Stage 14 Option A",
                **context)

        payload = event.get("derivation", "")
        if not payload:
            _not_replayable(
                "DERIVATION_EVIDENCE_MISSING",
                "recorded structured derivation is absent", **context)

        refs = frozenset().union(
            *(derivation_references(variant.record)
              for variant in mapping.derivation))
        facts: dict[str, Any] = {}
        if "value" in refs:
            facts["value"] = recomputed
        if "source_column" in refs:
            source_column = mapping.column or mapping.target
            if not source_column:
                _not_replayable(
                    "DERIVATION_EVIDENCE_MISSING",
                    "pinned source-column fact is unavailable", **context)
            facts["source_column"] = source_column
        if set(refs) - STAGE14_FACTS:
            _not_replayable(
                "DERIVATION_UNSUPPORTED_STAGE2",
                "derivation references facts outside Stage 14", **context)

        try:
            expected = render_structured_derivation(
                mapping.derivation, facts, {})
        except MissingDerivationFact as exc:
            _not_replayable(
                "DERIVATION_EVIDENCE_MISSING", str(exc), **context)
        except DerivationError as exc:
            _not_replayable(
                "DERIVATION_EVIDENCE_MISSING",
                f"derivation cannot be rendered from verified facts: {exc}",
                **context)
        try:
            recorded = parse_canonical_derivation_json(payload)
        except DerivationError as exc:
            _fail("DERIVATION_MISMATCH", str(exc), **context)
        if expected != recorded:
            _fail(
                "DERIVATION_MISMATCH",
                "recorded derivation contradicts independent fact projection",
                **context)

    def _stage15_contributions(self, event, mapping_id: str) -> list[dict]:
        """Parse recorded inputs only as verification targets."""
        payload = event.get("contributions", "")
        if not payload:
            return []
        try:
            contributions = contributions_from_json(payload)
        except (TypeError, ValueError, json.JSONDecodeError) as exc:
            _not_replayable(
                "NAMED_READ_EVIDENCE_MISSING",
                f"condition-input Contributions are malformed: {exc}",
                spec_ref=self.spec.ref, mapping_id=mapping_id)
        if any(not isinstance(item, dict) for item in contributions):
            _not_replayable(
                "NAMED_READ_EVIDENCE_MISSING",
                "condition-input Contributions must be objects",
                spec_ref=self.spec.ref, mapping_id=mapping_id)
        return contributions

    def _stage15_binding(self, source_id: str, usage, mapping_id: str,
                         *, missing_reason: str = "NAMED_READ_EVIDENCE_MISSING"
                         ) -> SourceBinding:
        """Resolve one exact usage-scoped source; run-wide bindings do not repair it."""
        assert self.spec is not None
        context = {"spec_ref": self.spec.ref, "mapping_id": mapping_id}
        matches = [
            binding for binding in self.manifest.sources
            if binding.spec_ref == self.spec.ref and binding.source_id == source_id
        ]
        if not matches:
            _not_replayable(
                missing_reason, f"source {source_id!r} has no SourceBinding",
                **context)
        if len(matches) != 1:
            _fail(
                "NAMED_READ_MISMATCH",
                f"source {source_id!r} has ambiguous SourceBindings", **context)
        binding = matches[0]
        if binding.source_ref_id not in usage.source_ref_ids:
            _not_replayable(
                missing_reason,
                f"MappingUsage omits causal source {source_id!r}",
                source_ref_id=binding.source_ref_id, **context)
        return binding

    def _stage15_indexed_source(
            self, source_id: str, usage, mapping_id: str, *,
            missing_reason: str = "NAMED_READ_EVIDENCE_MISSING"):
        """Verify one bound source and index it by the pinned participant policy."""
        assert self.spec is not None
        spec = self.spec
        binding = self._stage15_binding(
            source_id, usage, mapping_id, missing_reason=missing_reason)
        source_decl = spec.source(source_id)
        _check_stage15_source_declaration(
            self.root, spec, source_decl, binding, mapping_id,
            missing_reason=missing_reason)
        verified = self._verify_source_identity(
            binding, source_decl, mapping_id)
        frame = verified.frame.copy()
        frame["_spec_src_row"] = frame.index
        participant_key = spec.record.participant_key
        join_column = source_decl.participant_column or participant_key
        context = {"spec_ref": spec.ref, "mapping_id": mapping_id,
                   "source_ref_id": binding.source_ref_id}
        if join_column not in frame.columns:
            _fail(
                "SOURCE_COLUMN_MISSING",
                f"source {source_id!r} omits participant column {join_column!r}",
                **context)
        frame[join_column] = frame[join_column].astype(str)
        duplicate = spec.record.duplicates
        by = [
            join_column if name == participant_key else name
            for name in duplicate.by
        ] or [join_column]
        missing = [name for name in by if name not in frame.columns]
        if missing:
            _fail(
                "SOURCE_COLUMN_MISSING",
                f"duplicate policy names missing columns {missing}", **context)
        if duplicate.policy in ("first", "last"):
            frame = frame.drop_duplicates(subset=by, keep=duplicate.policy)
        elif duplicate.policy == "error":
            if frame.duplicated(subset=by).any():
                _fail(
                    "SOURCE_RECORD_KEY_MISMATCH",
                    "source violates the pinned duplicate policy", **context)
        else:
            _not_replayable(
                "OPERATION_UNSUPPORTED",
                f"duplicate policy {duplicate.policy!r} is unsupported",
                **context)
        indexed = frame.set_index(join_column, drop=False)
        if indexed.index.has_duplicates:
            _fail(
                "SOURCE_RECORD_KEY_MISMATCH",
                "source remains ambiguous after the pinned duplicate policy",
                **context)
        return binding, source_decl, indexed

    def _stage15_participant_row(
            self, source_id: str, usage, mapping_id: str, *,
            missing_reason: str = "NAMED_READ_EVIDENCE_MISSING"
            ) -> _VerifiedParticipantRow:
        assert self.spec is not None
        binding, source_decl, indexed = self._stage15_indexed_source(
            source_id, usage, mapping_id, missing_reason=missing_reason)
        participant = str(
            self.delivered_row.get(self.spec.record.participant_key, ""))
        if participant not in indexed.index:
            _not_replayable(
                missing_reason,
                f"source {source_id!r} has no independently reproducible row",
                source_ref_id=binding.source_ref_id,
                spec_ref=self.spec.ref, mapping_id=mapping_id)
        row = indexed.loc[participant]
        return _VerifiedParticipantRow(
            binding=binding, declaration=source_decl, row=row,
            row_number=int(row["_spec_src_row"]))

    def _stage15_group_member(
            self, group_id: str, event, usage, mapping_id: str,
            contributions: list[dict]):
        """Reuse the Stage 12 selector and verify its membership Contributions."""
        assert self.spec is not None
        spec = self.spec
        group = spec.group(group_id)
        context = {"spec_ref": spec.ref, "mapping_id": mapping_id}
        if group is None or group.select != "first_containing_record":
            _not_replayable(
                "NAMED_READ_EVIDENCE_MISSING",
                f"named read group {group_id!r} is not a closed source group",
                **context)
        expected_role = membership_role(spec, group)
        membership = [
            item for item in contributions if item.get("role") == expected_role
        ]
        if len(membership) < len(group.members):
            _not_replayable(
                "NAMED_READ_EVIDENCE_MISSING",
                "named-read group omits ordered membership evidence", **context)
        if len(membership) != len(group.members):
            _fail(
                "SOURCE_GROUP_SELECTION_MISMATCH",
                "named-read group has extra membership evidence", **context)

        participant = str(
            self.delivered_row.get(spec.record.participant_key, ""))
        record_key = _participant_record_key(spec.record, participant)
        presence = {}
        indexed_by_member = {}
        bindings = {}
        for member in group.members:
            source_id = member.sources[group.decided_by]
            binding, _decl, indexed = self._stage15_indexed_source(
                source_id, usage, mapping_id)
            bindings[member.id] = binding
            indexed_by_member[member.id] = indexed
            presence[member.id] = participant in indexed.index
        decision = select_first_containing_record(
            tuple(member.id for member in group.members), presence,
            membership_status=group.membership_status)
        if decision.selected_rank is None:
            _fail(
                "SOURCE_GROUP_SELECTION_MISMATCH",
                "no source-group member contains the record", **context)
        group_rule = group_rule_id(spec, group)
        for rank, (member, contribution) in enumerate(
                zip(group.members, membership)):
            binding = bindings[member.id]
            present = presence[member.id]
            row = (indexed_by_member[member.id].loc[participant]
                   if present else None)
            expected_row = int(row["_spec_src_row"]) if row is not None else ""
            expected_raw = participant if present else ""
            snapshot = str(contribution.get("raw_value_json") or "")
            if not snapshot:
                _not_replayable(
                    "NAMED_READ_EVIDENCE_MISSING",
                    f"membership Contribution {rank} has no typed snapshot",
                    **context)
            try:
                recorded = TypedRawValue.from_json(snapshot)
                observed = encode_typed_raw(expected_raw)
            except ValueError as exc:
                _not_replayable(
                    "NAMED_READ_EVIDENCE_MISSING", str(exc), **context)
            if (contribution.get("status") != decision.statuses[rank]
                    or contribution.get("rank") != rank
                    or contribution.get("source_ref_id") != binding.source_ref_id
                    or contribution.get("source_logical_label")
                       != binding.logical_label
                    or contribution.get("source_resolved_path") != binding.path
                    or contribution.get("source_sha256") != binding.sha256
                    or str(contribution.get("source_sheet") or "")
                       != binding.sheet
                    or contribution.get("source_column") != ""
                    or contribution.get("source_record_key") != record_key
                    or contribution.get("source_row_number") != expected_row
                    or contribution.get("rule_id") != group_rule
                    or recorded != observed):
                _fail(
                    "SOURCE_GROUP_SELECTION_MISMATCH",
                    f"membership Contribution {rank} contradicts selection",
                    source_ref_id=binding.source_ref_id, **context)
        selected = group.members[decision.selected_rank]
        return _VerifiedGroupDecision(
            group=group,
            selected_member=selected,
            selected_rank=decision.selected_rank,
            selected_deciding_binding=bindings[selected.id],
            deciding_source_ref_ids=tuple(
                bindings[member.id].source_ref_id for member in group.members),
        )

    def _verify_stage15_input_contribution(
            self, contribution: dict | None, verified: _VerifiedParticipantRow,
            *, mapping: SpecMapping, role: str, rank: int, column: str,
            value: Any, missing_reason: str = "NAMED_READ_EVIDENCE_MISSING",
            mismatch_reason: str = "NAMED_READ_MISMATCH") -> None:
        """Compare a rebuilt condition input with its recorded Contribution."""
        context = {
            "spec_ref": self.spec.ref, "mapping_id": mapping.mapping_id,
            "source_ref_id": verified.binding.source_ref_id,
        }
        if contribution is None:
            _not_replayable(
                missing_reason, f"condition input {role!r} is absent", **context)
        snapshot = str(contribution.get("raw_value_json") or "")
        if not snapshot:
            _not_replayable(
                missing_reason, f"condition input {role!r} has no typed snapshot",
                **context)
        try:
            recorded = TypedRawValue.from_json(snapshot)
            observed = encode_typed_raw(value)
        except ValueError as exc:
            _not_replayable(missing_reason, str(exc), **context)
        _verify_record_key(
            str(contribution.get("source_record_key") or ""),
            self.spec, verified.declaration, verified.row, self.delivered_row,
            self.spec.ref, mapping.mapping_id, verified.binding.source_ref_id)
        _verify_declared_record_keys(
            verified.declaration, verified.row, self.delivered_row,
            self.spec.ref, mapping.mapping_id, verified.binding.source_ref_id)
        if (contribution.get("role") != role
                or contribution.get("status") != STATUS_CONDITION
                or contribution.get("rank") != rank
                or contribution.get("source_ref_id")
                   != verified.binding.source_ref_id
                or contribution.get("source_column") != column
                or contribution.get("source_row_number") != verified.row_number
                or contribution.get("source_logical_label")
                   != verified.binding.logical_label
                or contribution.get("source_resolved_path")
                   != verified.binding.path
                or contribution.get("source_sha256") != verified.binding.sha256
                or str(contribution.get("source_sheet") or "")
                   != verified.binding.sheet
                or contribution.get("rule_id") != mapping.rule.id
                or contribution.get("provenance_ref")
                or recorded != observed):
            _fail(
                mismatch_reason,
                f"condition input {role!r} contradicts independent reconstruction",
                **context)

    def _stage15_named_read(
            self, read_id: str, mapping: SpecMapping, event, usage,
            contributions: list[dict], *, groups_cache=None
            ) -> _VerifiedNamedRead:
        """Reconstruct one declared read from source bytes, member and row."""
        assert self.spec is not None
        reads = [read for read in self.spec.reads if read.id == read_id]
        if len(reads) != 1:
            _not_replayable(
                "NAMED_READ_EVIDENCE_MISSING",
                f"named read {read_id!r} is not uniquely pinned",
                spec_ref=self.spec.ref, mapping_id=mapping.mapping_id)
        read = reads[0]
        source_id = read.source
        selected_member = None
        if GROUP_REF_SEP in source_id:
            group_id, _, role = source_id.partition(GROUP_REF_SEP)
            groups_cache = groups_cache if groups_cache is not None else {}
            if group_id not in groups_cache:
                groups_cache[group_id] = self._stage15_group_member(
                    group_id, event, usage, mapping.mapping_id, contributions)
            selected_member = groups_cache[group_id].selected_member
            source_id = selected_member.sources.get(role, "")
            if not source_id:
                _not_replayable(
                    "NAMED_READ_EVIDENCE_MISSING",
                    f"selected member offers no role {role!r}",
                    spec_ref=self.spec.ref, mapping_id=mapping.mapping_id)
        verified = self._stage15_participant_row(
            source_id, usage, mapping.mapping_id)
        column = read.column
        if read.column_by_member and selected_member is not None:
            column = read.column_by_member.get(selected_member.id, column)
        if not column or column not in verified.row.index:
            _fail(
                "SOURCE_COLUMN_MISSING",
                f"named read {read.id!r} column {column!r} is absent",
                source_ref_id=verified.binding.source_ref_id,
                spec_ref=self.spec.ref, mapping_id=mapping.mapping_id)
        raw_source = verified.row[column]
        try:
            value = coerce_named_read(raw_source, read.coerce)
        except ValueError as exc:
            _not_replayable(
                "NAMED_READ_EVIDENCE_MISSING", str(exc),
                spec_ref=self.spec.ref, mapping_id=mapping.mapping_id)
        matches = [item for item in contributions
                   if item.get("role") == read.id]
        if len(matches) < 1:
            _not_replayable(
                "NAMED_READ_EVIDENCE_MISSING",
                f"named read {read.id!r} has no condition Contribution",
                spec_ref=self.spec.ref, mapping_id=mapping.mapping_id)
        if len(matches) != 1:
            _fail(
                "NAMED_READ_MISMATCH",
                f"named read {read.id!r} has duplicate Contributions",
                spec_ref=self.spec.ref, mapping_id=mapping.mapping_id)
        self._verify_stage15_input_contribution(
            matches[0], verified, mapping=mapping, role=read.id,
            rank=mapping.contributions.condition_rank, column=column, value=value)
        return _VerifiedNamedRead(
            read_id=read.id, source_id=source_id,
            source_ref_id=verified.binding.source_ref_id,
            source_column=column, row_number=verified.row_number,
            raw_source_value=raw_source, value=value,
            selected_member_id=(selected_member.id if selected_member else ""))

    def _replay_stage15_fallback(
            self, mapping: SpecMapping, event, usage
            ) -> tuple[VerifiedOperationRouteResult, str]:
        """Replay a causally closed gated fallback through verified predicates."""
        assert self.spec is not None
        spec = self.spec
        context = {"spec_ref": spec.ref, "mapping_id": mapping.mapping_id}
        _check_stage15_fallback_scope(mapping, spec, event, self.artifact)
        _check_stage2_environment(self.manifest, spec.ref, mapping.mapping_id)
        contributions = self._stage15_contributions(
            event, mapping.mapping_id)

        direct_gates = []
        seen_direct = set()
        for candidate in mapping.candidates:
            gate = candidate.when
            if gate is None or gate.read:
                continue
            key = (gate.source, gate.column)
            if key not in seen_direct:
                seen_direct.add(key)
                direct_gates.append(gate)

        candidate_contributions = [
            item for item in contributions
            if item.get("role") == f"{mapping.target}_candidate"
        ]
        direct_condition_contributions = [
            item for item in contributions
            if item.get("role") == f"{mapping.target}_condition"
        ]
        named_roles = set(mapping.contributions.conditions)
        named_contributions = [
            item for item in contributions if item.get("role") in named_roles
        ]
        raw_from_contributions = [
            item for item in contributions
            if mapping.provenance.raw_from
            and mapping.provenance.raw_from not in named_roles
            and item.get("role") == mapping.provenance.raw_from
        ]
        expected_count = (
            len(mapping.candidates) + len(direct_gates)
            + len(mapping.contributions.conditions))
        if len(contributions) < expected_count:
            _not_replayable(
                "FALLBACK_EVIDENCE_INCOMPLETE",
                "gated fallback omits causal candidate/condition evidence",
                **context)
        if (len(contributions) != expected_count + len(raw_from_contributions)
                or len(candidate_contributions) != len(mapping.candidates)
                or len(direct_condition_contributions) != len(direct_gates)
                or len(named_contributions)
                   != len(mapping.contributions.conditions)
                or len(raw_from_contributions) > 1):
            _fail(
                "FALLBACK_CANDIDATE_CONTEXT_MISMATCH",
                "gated fallback has extra or misclassified Contributions",
                **context)

        expected_order = []
        direct_descriptors = [
            (f"{mapping.target}_condition", -1, gate.column)
            for gate in direct_gates
        ]
        candidate_descriptors = [
            (f"{mapping.target}_candidate", rank, candidate.column)
            for rank, candidate in enumerate(mapping.candidates)
        ]
        expected_order.extend(
            direct_descriptors + candidate_descriptors
            if mapping.contributions.condition_position == "first"
            else candidate_descriptors + direct_descriptors)
        expected_order.extend([
            (read_id, mapping.contributions.condition_rank, None)
            for read_id in mapping.contributions.conditions
        ])
        if raw_from_contributions:
            expected_order.append((
                mapping.provenance.raw_from,
                mapping.contributions.condition_rank, None))
        actual_order = [
            (item.get("role"), item.get("rank"), item.get("source_column"))
            for item in contributions
        ]
        for rank, (expected, actual) in enumerate(
                zip(expected_order, actual_order)):
            erole, erank, ecolumn = expected
            arole, arank, acolumn = actual
            if (arole != erole or arank != erank
                    or (ecolumn is not None and acolumn != ecolumn)):
                _fail(
                    "FALLBACK_CANDIDATE_CONTEXT_MISMATCH",
                    f"Contribution position {rank} contradicts declared order",
                    **context)

        candidate_contributions.sort(key=lambda item: item.get("rank"))
        bindings = []
        rows = []
        for rank, (candidate, contribution) in enumerate(
                zip(mapping.candidates, candidate_contributions)):
            binding = self._fallback_binding(candidate.source, rank)
            bindings.append(binding)
            _check_fallback_contribution_context(
                mapping, candidate, contribution, binding, rank, spec.ref)
            rows.append(self._verify_fallback_presence(
                mapping, candidate, contribution, binding, rank))

        direct_operands = {}
        for gate, contribution in zip(
                direct_gates, direct_condition_contributions):
            verified = self._stage15_participant_row(
                gate.source, usage, mapping.mapping_id,
                missing_reason="FALLBACK_EVIDENCE_INCOMPLETE")
            if gate.column not in verified.row.index:
                _fail(
                    "SOURCE_COLUMN_MISSING",
                    f"fallback gate column {gate.column!r} is absent",
                    source_ref_id=verified.binding.source_ref_id, **context)
            raw = verified.row[gate.column]
            self._verify_stage15_input_contribution(
                contribution, verified, mapping=mapping,
                role=f"{mapping.target}_condition", rank=-1,
                column=gate.column, value=raw,
                missing_reason="FALLBACK_EVIDENCE_INCOMPLETE",
                mismatch_reason="FALLBACK_CANDIDATE_CONTEXT_MISMATCH")
            direct_operands[(gate.source, gate.column)] = raw

        required_reads = []
        for candidate in mapping.candidates:
            if candidate.when and candidate.when.read:
                required_reads.append(candidate.when.read)
        for read_id in mapping.contributions.conditions:
            if read_id not in required_reads:
                required_reads.append(read_id)
        named = {
            read_id: self._stage15_named_read(
                read_id, mapping, event, usage, contributions)
            for read_id in required_reads
        }

        expected_sources = {binding.source_ref_id for binding in bindings}
        expected_sources.update(
            item.source_ref_id for item in named.values())
        expected_sources.update(
            str(item.get("source_ref_id") or "") for item in contributions
            if item.get("source_ref_id"))
        actual_sources = set(usage.source_ref_ids)
        if not expected_sources.issubset(actual_sources):
            _not_replayable(
                "SOURCE_LINK_MISSING",
                "MappingUsage omits gated-fallback causal sources", **context)
        if actual_sources != expected_sources:
            _fail(
                "FALLBACK_CANDIDATE_CONTEXT_MISMATCH",
                "MappingUsage contains unrelated gated-fallback sources",
                **context)

        selected_rank = None
        selected_raw = None
        for rank, (candidate, contribution, source_row, binding) in enumerate(
                zip(mapping.candidates, candidate_contributions, rows, bindings)):
            status = contribution.get("status")
            if selected_rank is not None:
                expected = (
                    STATUS_REJECTED
                    if mapping.contributions.after_decision == "rejected"
                    else STATUS_NOT_EVALUATED)
                if status != expected:
                    _fail(
                        "FALLBACK_SELECTION_MISMATCH",
                        f"candidate {rank} must be {expected!r} after winner",
                        source_ref_id=binding.source_ref_id, **context)
                continue
            if candidate.column not in source_row.index:
                _fail(
                    "SOURCE_COLUMN_MISSING",
                    f"fallback candidate column {candidate.column!r} is absent",
                    source_ref_id=binding.source_ref_id, **context)
            raw = source_row[candidate.column]
            snapshot = str(contribution.get("raw_value_json") or "")
            if not snapshot:
                _not_replayable(
                    "FALLBACK_EVIDENCE_INCOMPLETE",
                    f"reached candidate {rank} has no typed snapshot",
                    source_ref_id=binding.source_ref_id, **context)
            try:
                recorded = TypedRawValue.from_json(snapshot)
                observed = encode_typed_raw(raw)
            except ValueError as exc:
                _not_replayable(
                    "FALLBACK_CANDIDATE_UNSUPPORTED", str(exc),
                    source_ref_id=binding.source_ref_id, **context)
            if recorded != observed:
                _fail(
                    "RAW_VALUE_MISMATCH",
                    f"candidate {rank} scalar contradicts its snapshot",
                    source_ref_id=binding.source_ref_id, **context)

            if is_missing_scalar(raw):
                expected_status = STATUS_UNAVAILABLE
            else:
                gate = candidate.when
                eligible = True
                if gate is not None:
                    operand = (
                        named[gate.read].value if gate.read
                        else direct_operands[(gate.source, gate.column)])
                    eligible = evaluate_predicate(
                        gate.predicate, operand, gate.value)
                expected_status = (
                    STATUS_SELECTED if eligible else STATUS_REJECTED)
            if status != expected_status:
                _fail(
                    "FALLBACK_SELECTION_MISMATCH",
                    f"candidate {rank} is {status!r}, independently expected "
                    f"{expected_status!r}",
                    source_ref_id=binding.source_ref_id, **context)
            if expected_status == STATUS_SELECTED:
                selected_rank = rank
                selected_raw = raw

        try:
            recomputed = recompute_fallback(
                [] if selected_rank is None else [selected_raw],
                coerce=mapping.coerce)
        except _NonFiniteFallbackResult as exc:
            _fail("RECOMPUTED_VALUE_MISMATCH", str(exc), **context)
        value = recomputed.value
        branch = _OUTCOME_NO_VALUE if selected_rank is None else _OUTCOME_VALUE
        scalar_outcome = RecomputedOutcome(value, branch)
        result_source = (
            bindings[selected_rank].source_ref_id
            if selected_rank is not None else bindings[0].source_ref_id)
        expected_rule, expected_confidence = _outcome_context(
            mapping, scalar_outcome)
        _verify_confidence(
            event, expected_confidence, mapping.op, spec.ref,
            mapping.mapping_id, result_source)
        _verify_rule(
            event, expected_rule, usage, self.manifest, mapping.op,
            spec.ref, mapping.mapping_id, result_source)
        expected_column = (
            mapping.column
            or ((mapping.candidates[selected_rank].label
                 or mapping.candidates[selected_rank].column)
                if selected_rank is not None else
                (mapping.contributions.no_value_column or "+".join(
                    candidate.column for candidate in mapping.candidates))))
        if event.get("source_column") != expected_column:
            _fail(
                "FALLBACK_SELECTION_MISMATCH",
                "parent source column contradicts reconstructed winner",
                source_ref_id=result_source, **context)
        return _route_result(
            value, selected_raw, exhausted=selected_rank is None), result_source

    def _replay_fallback(self, mapping: SpecMapping, event, usage
                         ) -> tuple[VerifiedOperationRouteResult, str]:
        """Replay one narrow raw-source fallback with two evidence layers."""
        assert self.spec is not None
        if (any(candidate.when is not None for candidate in mapping.candidates)
                or mapping.contributions.conditions
                or mapping.contributions.after_decision != "not_evaluated"):
            return self._replay_stage15_fallback(mapping, event, usage)
        if _is_closed_same_row_inline_fallback(mapping):
            return self._replay_same_row_inline_fallback(mapping, event, usage)
        spec = self.spec
        context = {"spec_ref": spec.ref, "mapping_id": mapping.mapping_id}
        _check_fallback_scope(mapping, spec, event, self.artifact)
        _check_stage2_environment(self.manifest, spec.ref, mapping.mapping_id)

        payload = event.get("contributions", "")
        if not payload:
            _not_replayable(
                "FALLBACK_EVIDENCE_INCOMPLETE",
                "fallback provenance has no complete candidate evidence",
                **context)
        try:
            contributions = contributions_from_json(payload)
        except (TypeError, ValueError, json.JSONDecodeError) as exc:
            _not_replayable(
                "FALLBACK_EVIDENCE_INCOMPLETE",
                f"fallback contribution JSON is malformed: {exc}", **context)
        candidate_role = f"{mapping.target}_candidate"
        candidate_contributions = [
            item for item in contributions
            if isinstance(item, dict) and item.get("role") == candidate_role
        ]
        raw_from_contributions = [
            item for item in contributions
            if isinstance(item, dict)
            and item.get("role") == mapping.provenance.raw_from
        ] if mapping.provenance.raw_from else []
        allowed_ids = {id(item) for item in (
            candidate_contributions + raw_from_contributions)}
        if any(id(item) not in allowed_ids for item in contributions):
            _fail(
                "FALLBACK_CANDIDATE_CONTEXT_MISMATCH",
                "fallback provenance contains undeclared Contribution roles",
                **context)
        if len(raw_from_contributions) > 1:
            _fail(
                "NAMED_READ_MISMATCH",
                "fallback raw_from read has duplicate Contributions", **context)
        if len(candidate_contributions) < len(mapping.candidates):
            _not_replayable(
                "FALLBACK_EVIDENCE_INCOMPLETE",
                "fallback provenance omits one or more declared candidates",
                **context)
        if len(candidate_contributions) != len(mapping.candidates):
            _fail("FALLBACK_CANDIDATE_CONTEXT_MISMATCH",
                  "fallback provenance contains extra candidate evidence",
                  **context)

        bindings: list[SourceBinding] = []
        for rank, (candidate, contribution) in enumerate(
                zip(mapping.candidates, candidate_contributions)):
            if not isinstance(contribution, dict):
                _not_replayable(
                    "FALLBACK_EVIDENCE_INCOMPLETE",
                    f"fallback candidate {rank} is not an evidence object",
                    **context)
            binding = self._fallback_binding(candidate.source, rank)
            bindings.append(binding)
            _check_fallback_contribution_context(
                mapping, candidate, contribution, binding, rank, spec.ref)

        expected_source_ids = {binding.source_ref_id for binding in bindings}
        actual_source_ids = set(usage.source_ref_ids)
        if not expected_source_ids.issubset(actual_source_ids):
            missing = sorted(expected_source_ids - actual_source_ids)
            _not_replayable(
                "SOURCE_LINK_MISSING",
                f"fallback usage omits required eligibility source(s) {missing}",
                **context)
        optional_raw_sources = {
            str(item.get("source_ref_id") or "")
            for item in raw_from_contributions if item.get("source_ref_id")}
        if actual_source_ids != expected_source_ids | optional_raw_sources:
            _fail("FALLBACK_CANDIDATE_CONTEXT_MISMATCH",
                  "fallback usage includes sources outside its candidate/raw "
                  "evidence set",
                  **context)
        # Layer 1: every direct candidate source record is causal under the
        # executor's eager _skip_for_missing check, even after an early winner.
        rows = []
        for rank, (candidate, contribution, binding) in enumerate(
                zip(mapping.candidates, candidate_contributions, bindings)):
            rows.append(self._verify_fallback_presence(
                mapping, candidate, contribution, binding, rank))

        # Layer 2: read and verify candidate columns only through the first raw
        # winner.  Later contribution snapshots are audit-only.
        reached: list[Any] = []
        selected_rank = None
        for rank, (candidate, contribution, source_row, binding) in enumerate(
                zip(mapping.candidates, candidate_contributions, rows, bindings)):
            status = contribution.get("status")
            if selected_rank is not None:
                if status != STATUS_NOT_EVALUATED:
                    _fail("FALLBACK_SELECTION_MISMATCH",
                          f"candidate {rank} must be not_evaluated after rank "
                          f"{selected_rank} won", **context)
                continue
            if status == STATUS_NOT_EVALUATED:
                _fail("FALLBACK_SELECTION_MISMATCH",
                      f"candidate {rank} is not_evaluated before any winner",
                      **context)
            if candidate.column not in source_row.index:
                _fail("SOURCE_COLUMN_MISSING",
                      f"fallback candidate column {candidate.column!r} is absent",
                      source_ref_id=binding.source_ref_id, **context)
            raw = source_row[candidate.column]
            recorded_json = str(contribution.get("raw_value_json") or "")
            if not recorded_json:
                _not_replayable(
                    "FALLBACK_EVIDENCE_INCOMPLETE",
                    f"reached fallback candidate {rank} has no typed raw evidence",
                    source_ref_id=binding.source_ref_id, **context)
            try:
                recorded = TypedRawValue.from_json(recorded_json)
                observed = encode_typed_raw(raw)
            except ValueError as exc:
                _not_replayable(
                    "FALLBACK_CANDIDATE_UNSUPPORTED",
                    f"candidate {rank} scalar cannot be replayed: {exc}",
                    source_ref_id=binding.source_ref_id, **context)
            if recorded != observed:
                _fail("RAW_VALUE_MISMATCH",
                      f"candidate {rank} fresh scalar differs from typed evidence",
                      source_ref_id=binding.source_ref_id, **context)
            reached.append(raw)
            missing = _fallback_raw_is_missing(raw)
            expected_status = STATUS_UNAVAILABLE if missing else STATUS_SELECTED
            if status != expected_status:
                _fail("FALLBACK_SELECTION_MISMATCH",
                      f"candidate {rank} is {status!r}, independently expected "
                      f"{expected_status!r}", source_ref_id=binding.source_ref_id,
                      **context)
            if not missing:
                selected_rank = rank

        try:
            outcome = recompute_fallback(reached, coerce=mapping.coerce)
        except _NonFiniteFallbackResult as exc:
            _fail("RECOMPUTED_VALUE_MISMATCH", str(exc), **context)
        if outcome.selected_rank != selected_rank:
            _fail("FALLBACK_SELECTION_MISMATCH",
                  "pure fallback recomputation selected a different candidate",
                  **context)
        scalar_outcome = RecomputedOutcome(outcome.value, outcome.branch)
        expected_rule, expected_confidence = _outcome_context(
            mapping, scalar_outcome)
        result_source = (bindings[selected_rank].source_ref_id
                         if selected_rank is not None else bindings[0].source_ref_id)
        _verify_confidence(event, expected_confidence, mapping.op, spec.ref,
                           mapping.mapping_id, result_source)
        _verify_rule(event, expected_rule, usage, self.manifest, mapping.op,
                     spec.ref, mapping.mapping_id, result_source)

        expected_column = (mapping.column or
            ((mapping.candidates[selected_rank].label
              or mapping.candidates[selected_rank].column)
             if selected_rank is not None else
             (mapping.contributions.no_value_column or "+".join(
                 candidate.column for candidate in mapping.candidates))))
        if event.get("source_column") != expected_column:
            _fail("FALLBACK_SELECTION_MISMATCH",
                  "parent provenance source column contradicts selected fallback "
                  "outcome", source_ref_id=result_source, **context)
        return _route_result(
            outcome.value, outcome.raw_selected,
            exhausted=outcome.selected_rank is None), result_source

    def _replay_same_row_inline_fallback(
            self, mapping: SpecMapping, event, usage
            ) -> tuple[VerifiedOperationRouteResult, str]:
        """Replay a closed ungated chain from one exact bound source row."""
        assert self.spec is not None
        spec = self.spec
        context = {"spec_ref": spec.ref, "mapping_id": mapping.mapping_id}
        _check_same_row_inline_fallback_scope(
            mapping, spec, event, self.artifact)
        _check_stage2_environment(self.manifest, spec.ref, mapping.mapping_id)

        if event.get("contributions"):
            _fail(
                "FALLBACK_CANDIDATE_CONTEXT_MISMATCH",
                "record:none fallback unexpectedly contains serialized "
                "candidate evidence", **context)

        source_id = mapping.candidates[0].source
        binding = self._fallback_binding(source_id, 0)
        source_ref_id = binding.source_ref_id
        context["source_ref_id"] = source_ref_id
        actual_source_ids = set(usage.source_ref_ids)
        if source_ref_id not in actual_source_ids:
            _not_replayable(
                "SOURCE_LINK_MISSING",
                "MappingUsage omits the same-row fallback source", **context)
        if actual_source_ids != {source_ref_id}:
            _fail(
                "FALLBACK_CANDIDATE_CONTEXT_MISMATCH",
                "MappingUsage claims sources outside the closed same-row "
                "fallback", **context)
        event_source_ref_id = event.get("source_ref_id", "")
        if not event_source_ref_id:
            _not_replayable(
                "SOURCE_LINK_MISSING",
                "fallback Cell omits its event-scoped source link", **context)
        if event_source_ref_id != source_ref_id:
            _fail(
                "SOURCE_DECLARATION_MISMATCH",
                "fallback Cell source link contradicts MappingUsage and the "
                "pinned declaration", **context)

        source_decl = spec.source(source_id)
        _check_same_row_fallback_source_declaration(
            self.root, spec, mapping, event, source_decl, binding)
        verified = self._verify_source_identity(
            binding, source_decl, mapping.mapping_id)
        row_number = _source_row_number(
            event, spec.ref, mapping.mapping_id, source_ref_id)
        if row_number >= len(verified.frame):
            _fail(
                "SOURCE_ROW_OUT_OF_RANGE",
                f"source row {row_number} is outside {len(verified.frame)} rows",
                **context)
        source_row = verified.frame.iloc[row_number]
        _verify_record_key(
            event.get("source_record_key", ""), spec, source_decl, source_row,
            self.delivered_row, spec.ref, mapping.mapping_id, source_ref_id)
        _verify_declared_record_keys(
            source_decl, source_row, self.delivered_row, spec.ref,
            mapping.mapping_id, source_ref_id)
        _verify_unique_same_row_locator(
            event.get("source_record_key", ""), verified.frame, spec,
            source_decl, spec.ref, mapping.mapping_id, source_ref_id)

        raw_candidates = []
        for rank, candidate in enumerate(mapping.candidates):
            if candidate.column not in source_row.index:
                _fail(
                    "SOURCE_COLUMN_MISSING",
                    f"fallback candidate {rank} column "
                    f"{candidate.column!r} is absent", **context)
            raw_candidates.append(source_row[candidate.column])
        selection = select_ordered_fallback(raw_candidates)

        if selection.exhausted:
            value = None
        else:
            value = apply_inline_fallback_map(
                selection.raw_selected, mapping_table=mapping.map,
                on_unmapped=mapping.on_unmapped).value
        branch = _OUTCOME_NO_VALUE if _is_missing(value) else _OUTCOME_VALUE
        scalar_outcome = RecomputedOutcome(value, branch)
        expected_rule, expected_confidence = _outcome_context(
            mapping, scalar_outcome)
        _verify_confidence(
            event, expected_confidence, mapping.op, spec.ref,
            mapping.mapping_id, source_ref_id)
        _verify_rule(
            event, expected_rule, usage, self.manifest, mapping.op,
            spec.ref, mapping.mapping_id, source_ref_id)

        expected_column = (
            mapping.column
            or ((mapping.candidates[selection.selected_rank].label
                 or mapping.candidates[selection.selected_rank].column)
                if not selection.exhausted else
                (mapping.contributions.no_value_column or "+".join(
                    candidate.column for candidate in mapping.candidates))))
        if event.get("source_column") != expected_column:
            _fail(
                "FALLBACK_SELECTION_MISMATCH",
                "parent provenance source column contradicts the independently "
                "reconstructed same-row fallback outcome", **context)
        return _route_result(
            value, selection.raw_selected, exhausted=selection.exhausted), \
            source_ref_id

    def _fallback_binding(self, source_id: str, rank: int
                          ) -> SourceBinding:
        assert self.spec is not None
        matches = [binding for binding in self.manifest.sources
                   if binding.spec_ref == self.spec.ref
                   and binding.source_id == source_id]
        if not matches:
            _not_replayable(
                "SOURCE_LINK_MISSING",
                f"fallback candidate {rank} source {source_id!r} has no binding",
                spec_ref=self.spec.ref)
        if len(matches) != 1:
            _not_replayable(
                "SOURCE_LINK_MISSING",
                f"fallback candidate {rank} source {source_id!r} is ambiguous",
                spec_ref=self.spec.ref)
        return matches[0]

    def _verify_source_identity(
            self, binding: SourceBinding, source_decl, mapping_id: str
            ) -> _VerifiedSource:
        """Reopen and verify one bound source, without selecting a row.

        Row and scalar checks remain operation-specific.  This layer proves
        only the exact contained file, reader contract and observation, row
        count, and rebuilt runtime SourceRef identity.
        """
        assert self.spec is not None
        context = {"spec_ref": self.spec.ref, "mapping_id": mapping_id,
                   "source_ref_id": binding.source_ref_id}
        _check_reader_contract(
            binding, source_decl, self.manifest, mapping_id)
        verified = self.source_cache.get(binding.source_ref_id)
        if verified is not None:
            return verified
        if binding.reader.reader_id == XLSX_READER_ID:
            return self._verify_xlsx_source_identity(
                binding, source_decl, mapping_id)

        source_path = resolve_project_path(self.root, binding.path)
        read = read_cidc_csv(source_path)
        observed_flags = {
            "encoding": read.encoding, "header": 0, "low_memory": False,
            "skiprows": 1 if read.preamble_skipped else 0,
        }
        if read.note or observed_flags != binding.reader.flags:
            _fail("READER_OBSERVATION_MISMATCH",
                  "fresh reader observation differs from pinned metadata",
                  **context)
        if len(read.df) != binding.n_rows:
            _fail("SOURCE_ROW_COUNT_MISMATCH",
                  "fresh source row count differs from manifest", **context)
        try:
            observed_ref = SourceRef.build(
                config_key=binding.config_key, path=source_path,
                project_root=self.root, file_type="csv",
                reader_id="cidc_csv@v1",
                logical_label=binding.logical_label,
                reader_effective_flags=observed_flags, n_rows=len(read.df))
        except (PathContainmentError, FileNotFoundError) as exc:
            _not_replayable("MANIFEST_PATH_OUTSIDE_PROJECT", str(exc),
                            **context)
        if observed_ref.source_ref_id != binding.source_ref_id:
            _fail("SOURCE_DECLARATION_MISMATCH",
                  "source_ref_id is inconsistent with pinned source",
                  **context)
        verified = _VerifiedSource(binding, source_decl, read.df)
        self.source_cache[binding.source_ref_id] = verified
        return verified

    def _verify_xlsx_source_identity(
            self, binding: SourceBinding, source_decl, mapping_id: str
            ) -> _VerifiedSource:
        """Verify and parse one captured XLSX payload without reopening it."""
        assert self.spec is not None
        context = {"spec_ref": self.spec.ref, "mapping_id": mapping_id,
                   "source_ref_id": binding.source_ref_id}
        source_path = resolve_project_path(self.root, binding.path)
        if not source_path.is_file():
            _fail("SOURCE_FILE_MISSING",
                  f"source file is missing: {binding.path}", **context)
        try:
            payload = source_path.read_bytes()
        except OSError as exc:
            _fail("SOURCE_FILE_MISSING", str(exc), **context)
        if hashlib.sha256(payload).hexdigest() != binding.sha256:
            _fail("SOURCE_HASH_MISMATCH",
                  f"hash mismatch for {binding.identity!r}", **context)
        try:
            frame = parse_xlsx_bytes(payload, sheet_name=binding.sheet)
            sheet_digest = xlsx_sheet_sha256(
                frame, sheet_name=binding.sheet)
        except (TypeError, XlsxReaderError) as exc:
            _fail("READER_OBSERVATION_MISMATCH", str(exc), **context)
        if len(frame) != binding.n_rows:
            _fail("SOURCE_ROW_COUNT_MISMATCH",
                  "fresh XLSX row count differs from manifest", **context)
        if (tuple(frame.columns) != binding.parsed_columns
                or binding.parsed_canonicalization
                   != XLSX_SHEET_CANONICALIZATION_ID
                or sheet_digest != binding.parsed_sheet_sha256):
            _fail("READER_OBSERVATION_MISMATCH",
                  "fresh parsed-sheet identity differs from manifest", **context)
        try:
            observed_ref = SourceRef.from_observed_bytes(
                config_key=binding.config_key, path=source_path,
                project_root=self.root, payload=payload, file_type="xlsx",
                reader_id=XLSX_READER_ID,
                logical_label=binding.logical_label, sheet=binding.sheet,
                reader_effective_flags=XLSX_READER_FLAGS,
                n_rows=len(frame), parsed_columns=tuple(frame.columns),
                parsed_canonicalization=XLSX_SHEET_CANONICALIZATION_ID,
                parsed_sheet_sha256=sheet_digest)
        except (PathContainmentError, FileNotFoundError) as exc:
            _not_replayable("MANIFEST_PATH_OUTSIDE_PROJECT", str(exc),
                            **context)
        if observed_ref.source_ref_id != binding.source_ref_id:
            _fail("SOURCE_DECLARATION_MISMATCH",
                  "XLSX source_ref_id is inconsistent with pinned observation",
                  **context)
        verified = _VerifiedSource(binding, source_decl, frame)
        self.source_cache[binding.source_ref_id] = verified
        return verified

    def _verify_fallback_presence(self, mapping, candidate, contribution,
                                  binding, rank):
        """Verify a candidate source row without reading its value column."""
        assert self.spec is not None
        spec = self.spec
        source_decl = spec.source(candidate.source)
        context = {"spec_ref": spec.ref, "mapping_id": mapping.mapping_id,
                   "source_ref_id": binding.source_ref_id}
        _check_fallback_source_declaration(
            self.root, spec, source_decl, binding, contribution, rank,
            mapping.mapping_id)
        verified = self._verify_source_identity(
            binding, source_decl, mapping.mapping_id)
        row_number = _fallback_row_number(
            contribution, spec.ref, mapping.mapping_id, binding.source_ref_id,
            rank)
        if row_number >= len(verified.frame):
            _fail("SOURCE_ROW_OUT_OF_RANGE",
                  f"candidate {rank} row {row_number} is outside "
                  f"{len(verified.frame)} rows", **context)
        source_row = verified.frame.iloc[row_number]
        _verify_record_key(str(contribution.get("source_record_key") or ""),
                           spec, source_decl, source_row, self.delivered_row,
                           spec.ref, mapping.mapping_id, binding.source_ref_id)
        _verify_declared_record_keys(
            source_decl, source_row, self.delivered_row, spec.ref,
            mapping.mapping_id, binding.source_ref_id)
        return source_row

    def _replay_custom(self, mapping: SpecMapping, event, usage,
                       row_identity, depth, active_stack) -> Any:
        assert self.spec is not None
        spec = self.spec
        context = {"spec_ref": spec.ref, "mapping_id": mapping.mapping_id}
        _check_custom_scope(mapping, event, self.artifact)
        _check_stage2_environment(self.manifest, spec.ref, mapping.mapping_id)
        if usage.source_ref_ids or event.get("source_ref_id"):
            _fail("DEPENDENCY_CONTEXT_MISMATCH",
                  "custom parent incorrectly claims direct source evidence",
                  **context)
        vocabulary_parameters = _top_level_vocabulary_parameters(
            mapping.params, spec_ref=spec.ref,
            mapping_id=mapping.mapping_id)
        implementation_ref = f"{mapping.rule.id}@{mapping.rule.version}"
        if implementation_ref not in usage.rule_refs:
            _not_replayable("MANIFEST_REQUIRED_FIELD_MISSING",
                            "mapping usage omits registered field rule", **context)
        bindings = [r for r in self.manifest.rules
                    if r.ref == implementation_ref]
        if not bindings:
            _not_replayable("MANIFEST_REQUIRED_FIELD_MISSING",
                            "manifest omits registered field rule", **context)
        if bindings[0].kind != rule_registry.KIND_FIELD:
            _fail("RULE_IDENTITY_MISMATCH",
                  "registered custom rule is not bound as kind field", **context)
        try:
            registered = rule_registry.resolve(
                mapping.rule.id, mapping.rule.version,
                kind=rule_registry.KIND_FIELD)
        except rule_registry.RuleKindMismatch as exc:
            _fail("RULE_IDENTITY_MISMATCH", str(exc), **context)
        except (rule_registry.RuleNotRegistered,
                rule_registry.RuleVersionMismatch) as exc:
            _not_replayable("REGISTERED_RULE_UNAVAILABLE", str(exc), **context)
        payload = event.get("contributions", "")
        if not payload:
            _not_replayable("DEPENDENCY_REFERENCE_MISSING",
                            "custom-rule provenance has no dependency links",
                            **context)
        try:
            contributions = contributions_from_json(payload)
        except (TypeError, ValueError, json.JSONDecodeError) as exc:
            _not_replayable("DEPENDENCY_REFERENCE_MISSING",
                            f"dependency contribution JSON is malformed: {exc}",
                            **context)
        if len(contributions) < len(mapping.inputs):
            _not_replayable("DEPENDENCY_REFERENCE_MISSING",
                            "custom-rule provenance omits required inputs", **context)
        if len(contributions) != len(mapping.inputs):
            _fail("DEPENDENCY_CONTEXT_MISMATCH",
                  "custom-rule provenance has extra dependency inputs", **context)
        values: list[Any] = []
        for rank, (expected_field, contribution) in enumerate(
                zip(mapping.inputs, contributions)):
            if not isinstance(contribution, dict):
                _not_replayable("DEPENDENCY_REFERENCE_MISSING",
                                f"dependency {rank} is not an object", **context)
            if (contribution.get("status") != STATUS_INPUT
                    or contribution.get("rank") != rank
                    or contribution.get("source_column") != expected_field
                    or contribution.get("role") != "registered_rule_input"
                    or contribution.get("source_ref_id")):
                _fail("DEPENDENCY_CONTEXT_MISMATCH",
                      f"dependency {rank} contradicts declared input order",
                      **context)
            reference = str(contribution.get("provenance_ref") or "")
            try:
                dependency = self.replay_event(
                    reference, expected_field, row_identity, depth + 1,
                    active_stack)
            except _Stop as stop:
                if stop.reason_code in {
                    "DEPENDENCY_REFERENCE_MISSING",
                    "DEPENDENCY_REFERENCE_AMBIGUOUS",
                    "DEPENDENCY_CONTEXT_MISMATCH",
                    "REPLAY_CYCLE_DETECTED",
                    "REPLAY_DEPTH_EXCEEDED",
                }:
                    raise
                if stop.status is ReplayStatus.FAIL:
                    _fail("DEPENDENCY_REPLAY_FAILED",
                          f"dependency {reference!r} failed: "
                          f"{stop.reason_code}: {stop.message}", **context)
                _not_replayable(
                    "DEPENDENCY_NOT_REPLAYABLE",
                    f"dependency {reference!r} is not replayable: "
                    f"{stop.reason_code}: {stop.message}", **context)
            snapshot = str(contribution.get("raw_value_json") or "")
            if not snapshot:
                _not_replayable("DEPENDENCY_REFERENCE_MISSING",
                                f"dependency {rank} has no typed snapshot",
                                **context)
            try:
                recorded = TypedRawValue.from_json(snapshot)
                replayed = encode_typed_raw(dependency.value)
            except ValueError as exc:
                _not_replayable("DEPENDENCY_REFERENCE_MISSING", str(exc),
                                **context)
            if recorded != replayed:
                _fail("DEPENDENCY_CONTEXT_MISMATCH",
                      f"dependency {reference!r} typed snapshot contradicts "
                      "its independently replayed value", **context)
            values.append(dependency.value)
        params = dict(mapping.params)
        if vocabulary_parameters:
            parameter, expected_ref = next(iter(vocabulary_parameters.items()))
            vocabulary = _verified_vocabulary(
                self.root, self.manifest, event, usage,
                expected_ref=expected_ref,
                expected_kind=vocabularies.KIND_BUCKETS,
                spec_ref=spec.ref, mapping_id=mapping.mapping_id)
            params[parameter] = vocabulary.terms
        try:
            rule_registry.check_contract(registered, mapping.inputs, params)
        except rule_registry.RuleContractMismatch as exc:
            _not_replayable("REGISTERED_RULE_CONTRACT_MISMATCH", str(exc), **context)
        try:
            value, confidence, _note = recompute_registered_rule(
                registered, values, params=params)
            finalized_confidence = finalize_registered_rule_confidence(
                value, confidence,
                declared_value_policy=mapping.confidence,
                declared_no_value_confidence=mapping.confidence_no_value)
        except UnsupportedConfidencePolicy as exc:
            _not_replayable(
                "REGISTERED_RULE_CONTRACT_MISMATCH", str(exc), **context)
        except InvalidRuleConfidence as exc:
            _fail("REGISTERED_RULE_EXECUTION_FAILED", str(exc), **context)
        except Exception as exc:
            _fail("REGISTERED_RULE_EXECUTION_FAILED",
                  f"registered rule {registered.ref} failed: "
                  f"{type(exc).__name__}: {exc}", **context)
        expected_rule = mapping.rule
        if is_missing_scalar(value):
            if mapping.no_value_rule is not None:
                expected_rule = mapping.no_value_rule
        _verify_custom_rule(event, expected_rule, mapping.rule, usage,
                            self.manifest, spec.ref, mapping.mapping_id)
        _verify_confidence(
            event, finalized_confidence.value, mapping.op, spec.ref,
            mapping.mapping_id, "")
        return value

    def _compare_event(self, event, mapping, recomputed, source_ref_id) -> None:
        field = self.schema.field(mapping.target)
        context = {"spec_ref": self.spec.ref,
                   "mapping_id": mapping.mapping_id,
                   "source_ref_id": source_ref_id,
                   "delivered_value": self.delivered_row.get(mapping.target),
                   "recomputed_value": recomputed}
        if field is None or mapping.target not in self.delivered_row:
            _fail("DEPENDENCY_CONTEXT_MISMATCH",
                  "dependency target is absent from delivered schema", **context)
        try:
            delivered = _csv_semantic(
                self.delivered_row[mapping.target], field, self.schema)
            provenance = _csv_semantic(event.get("value", ""), field, self.schema)
            replayed = _python_semantic(recomputed, field)
        except ValueError as exc:
            _fail("RECOMPUTED_VALUE_MISMATCH",
                  f"value cannot be interpreted under schema: {exc}", **context)
        if provenance != delivered:
            _fail("PROVENANCE_VALUE_MISMATCH",
                  "authoritative provenance differs from delivered cell", **context)
        if replayed != delivered:
            _fail("RECOMPUTED_VALUE_MISMATCH",
                  "independently recomputed value differs from delivered cell",
                  **context)


def _validated_manifest(root: Path, manifest_input) -> ReplayManifest:
    try:
        if isinstance(manifest_input, ReplayManifest):
            return parse_manifest(manifest_input.to_dict(), project_root=root)
        if not isinstance(manifest_input, (str, Path)):
            _request_error("REQUEST_MALFORMED",
                           "manifest must be a ReplayManifest or explicit path")
        raw = Path(manifest_input)
        candidate = raw.resolve() if raw.is_absolute() else (root / raw).resolve()
        if not candidate.is_relative_to(root):
            _request_error("MANIFEST_PATH_INVALID",
                           "manifest path is outside project_root")
        return load_manifest(raw, project_root=root)
    except UnsupportedManifestVersion as exc:
        _not_replayable("MANIFEST_VERSION_UNSUPPORTED", str(exc))
    except ManifestError as exc:
        message = str(exc)
        code = (
            "SOURCE_LINK_MISSING"
            if "names unknown source" in message else
            "VOCABULARY_EVIDENCE_MISSING"
            if "names unknown evidence" in message and "vocab-" in message
            else "ANCHOR_EVIDENCE_MISSING"
            if "names unknown evidence" in message and "anchor-" in message
            else "MANIFEST_PATH_OUTSIDE_PROJECT"
            if "escape" in message or "project-root-relative" in message
            else "MANIFEST_REQUIRED_FIELD_MISSING"
        )
        _not_replayable(code, str(exc))


def _verify_hash(root: Path, binding: FileBinding, reason: str) -> None:
    path = resolve_project_path(root, binding.path)
    actual = sha256_file(path) if path.is_file() else ""
    if actual != binding.sha256:
        _fail(reason, f"hash mismatch for {binding.identity!r}")


def _verify_usage_files(root: Path, manifest: ReplayManifest, usage,
                        artifact: ArtifactBinding) -> None:
    try:
        files = manifest.files_for(
            mapping_id=usage.mapping_id, spec_ref=usage.spec_ref,
            artifact_id=artifact.artifact_id)
    except ManifestError as exc:
        _not_replayable("MANIFEST_REQUIRED_FIELD_MISSING", str(exc),
                        spec_ref=usage.spec_ref, mapping_id=usage.mapping_id)
    for binding in files:
        if binding.identity == manifest.target_schema.identity:
            reason = "SCHEMA_HASH_MISMATCH"
        elif binding.identity == manifest.provenance_artifact.identity:
            reason = "PROVENANCE_ARTIFACT_HASH_MISMATCH"
        elif binding.identity == artifact.identity:
            reason = "DELIVERED_ARTIFACT_HASH_MISMATCH"
        elif isinstance(binding, SpecificationBinding):
            reason = "SPEC_HASH_MISMATCH"
        elif isinstance(binding, SourceBinding):
            if binding.reader.reader_id == XLSX_READER_ID:
                # The XLSX verifier reads, hashes, and parses one cached payload.
                continue
            path = resolve_project_path(root, binding.path)
            if not path.is_file():
                _fail("SOURCE_FILE_MISSING",
                      f"source file is missing: {binding.path}",
                      spec_ref=usage.spec_ref, mapping_id=usage.mapping_id,
                      source_ref_id=binding.source_ref_id)
            reason = "SOURCE_HASH_MISMATCH"
        elif isinstance(binding, EvidenceBinding):
            reason = ("VOCABULARY_HASH_MISMATCH"
                      if binding.kind == "vocabulary"
                      else "CONFIG_HASH_MISMATCH")
        else:
            reason = "CONFIG_HASH_MISMATCH"
        _verify_hash(root, binding, reason)


def _read_csv(path: Path, required=frozenset()) -> list[dict[str, str]]:
    try:
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            reader = csv.reader(handle)
            try:
                header = next(reader)
            except StopIteration:
                raise _CsvError(f"{path}: CSV is empty")
            if not header or any(not name for name in header):
                raise _CsvError(f"{path}: CSV has an empty header")
            duplicates = sorted({name for name in header if header.count(name) > 1})
            if duplicates:
                raise _CsvError(f"{path}: duplicate CSV columns {duplicates}")
            missing = sorted(set(required) - set(header))
            if missing:
                raise _CsvError(f"{path}: missing required columns {missing}")
            rows = []
            for line_number, values in enumerate(reader, 2):
                if len(values) != len(header):
                    raise _CsvError(
                        f"{path}: row {line_number} has {len(values)} values; "
                        f"expected {len(header)}")
                rows.append(dict(zip(header, values)))
            return rows
    except (OSError, UnicodeError, csv.Error) as exc:
        raise _CsvError(f"{path}: CSV cannot be read: {exc}") from exc


def _validate_selector(selector: Mapping[str, Any],
                       schema: TargetSchema) -> dict[str, Any]:
    wanted, got = set(schema.row_key), set(selector)
    missing, extra = sorted(wanted - got), sorted(got - wanted)
    if missing:
        _request_error("SELECTOR_INCOMPLETE",
                       f"row_key is missing schema key(s) {missing}")
    if extra:
        _request_error("SELECTOR_HAS_EXTRA_KEYS",
                       f"row_key has non-schema key(s) {extra}")
    if not schema.row_key:
        _request_error("REQUEST_MALFORMED", "target schema has no row_key")
    out = {}
    for name in schema.row_key:
        field = schema.field(name)
        try:
            out[name] = _selector_semantic(selector[name], field)
        except ValueError as exc:
            _request_error("REQUEST_MALFORMED",
                           f"invalid selector value for {name!r}: {exc}")
    return out


def _select_delivered(rows: list[dict[str, str]], selector: dict[str, Any],
                      schema: TargetSchema) -> list[dict[str, str]]:
    out = []
    for row in rows:
        if any(name not in row for name in schema.row_key):
            _request_error("SELECTOR_INCOMPLETE",
                           "delivered artifact omits a schema row-key column")
        matched = True
        for name, wanted in selector.items():
            try:
                actual = _csv_semantic(row[name], schema.field(name), schema)
            except ValueError:
                matched = False
                break
            if actual != wanted:
                matched = False
                break
        if matched:
            out.append(row)
    return out


def _select_provenance(root: Path, manifest: ReplayManifest,
                       schema: TargetSchema, rows: list[dict[str, str]],
                       delivered_row: dict[str, str], target: str,
                       artifact: ArtifactBinding
                       ) -> tuple[dict[str, str], HarmonizationSpec,
                                  SpecificationBinding]:
    field_rows = [r for r in rows if r["harmonized_field"] == target]
    if not field_rows:
        _not_replayable("PROVENANCE_EVENT_NOT_FOUND",
                        "no provenance event exists for the target field")
    spec_bindings = {s.ref: s for s in manifest.specifications}

    # The current CIMAC schema already carries the four provenance anchors.
    # Use them directly so an unrelated trial/specification is never opened or
    # hash-checked merely because it maps the same target field.
    if all(name in delivered_row for name in _CURRENT_PROVENANCE_KEY):
        identity = {name: delivered_row[name]
                    for name in _CURRENT_PROVENANCE_KEY}
        candidates = [row for row in field_rows if all(
            row[column] == identity[column]
            for column in _CURRENT_PROVENANCE_KEY)]
        if not candidates:
            _not_replayable("PROVENANCE_EVENT_NOT_FOUND",
                            "no provenance event matches the selected row anchors")
        claiming = [row for row in candidates
                    if _is_authoritative(row["authoritative_for"], artifact)]
        if not claiming:
            _not_replayable("PROVENANCE_NOT_AUTHORITATIVE",
                            "no matching provenance event is authoritative")
        if len(claiming) != 1:
            spec_refs = {row.get("spec_ref", "") for row in claiming}
            if len(claiming) > 1 and len(spec_refs) == 1:
                candidate_ref = next(iter(spec_refs))
                candidate_binding = spec_bindings.get(candidate_ref)
                if candidate_binding is not None:
                    candidate_spec, candidate_binding = _load_pinned_spec(
                        root, schema, candidate_binding)
                    by_id = {
                        item.mapping_id: item for item in candidate_spec.mappings
                        if item.target == target
                    }
                    conditions = {
                        item.id: item for item in candidate_spec.conditions}
                    if (all(row.get("mapping_id") in by_id for row in claiming)
                            and all(
                                by_id[row["mapping_id"]].when
                                and conditions.get(
                                    by_id[row["mapping_id"]].when) is not None
                                and conditions[
                                    by_id[row["mapping_id"]].when].record_origin
                                for row in claiming)):
                        _not_replayable(
                            "PROVENANCE_AUTHORITY_AMBIGUOUS",
                            "multiple authoritative origin-sibling events "
                            "match the same logical row",
                            spec_ref=candidate_ref)
                    if (all(row.get("mapping_id") in by_id for row in claiming)
                            and any(by_id[row["mapping_id"]].when
                                    for row in claiming)):
                        event = claiming[0]
                        try:
                            adapted = _provenance_identity(
                                candidate_spec, delivered_row)
                        except ValueError as exc:
                            _not_replayable(
                                "ROW_KEY_NOT_REPRESENTABLE_IN_PROVENANCE",
                                str(exc), spec_ref=candidate_ref)
                        if adapted != identity:
                            _not_replayable(
                                "SPEC_LINK_MISSING",
                                "selected spec does not describe delivered anchors",
                                spec_ref=candidate_ref)
                        return event, candidate_spec, candidate_binding
            _not_replayable("PROVENANCE_AUTHORITY_AMBIGUOUS",
                            f"{len(claiming)} authoritative provenance events match")
        event = claiming[0]
        spec_ref = event["spec_ref"]
        if not spec_ref or spec_ref not in spec_bindings:
            _not_replayable("SPEC_LINK_MISSING",
                            "selected provenance has no pinned specification link")
        spec, binding = _load_pinned_spec(
            root, schema, spec_bindings[spec_ref])
        try:
            adapted = _provenance_identity(spec, delivered_row)
        except ValueError as exc:
            _not_replayable("ROW_KEY_NOT_REPRESENTABLE_IN_PROVENANCE", str(exc),
                            spec_ref=spec_ref)
        if adapted != identity:
            _not_replayable("SPEC_LINK_MISSING",
                            "selected spec does not describe the delivered anchors",
                            spec_ref=spec_ref)
        return event, spec, binding

    # A generic schema can use the specification's explicit Record fields to
    # map its delivered row into the current provenance shape.  Manifest v1
    # does not put the spec's trial/record adapter in the manifest itself, so a
    # multi-spec generic run cannot safely choose which unverified spec to open.
    if len(manifest.specifications) != 1:
        _not_replayable(
            "ROW_KEY_NOT_REPRESENTABLE_IN_PROVENANCE",
            "non-CIMAC row-key adaptation requires exactly one pinned spec")
    binding = manifest.specifications[0]
    spec, binding = _load_pinned_spec(root, schema, binding)
    try:
        identity = _provenance_identity(spec, delivered_row)
    except ValueError as exc:
        _not_replayable("ROW_KEY_NOT_REPRESENTABLE_IN_PROVENANCE", str(exc),
                        spec_ref=spec.ref)
    anchor_matches = [row for row in field_rows if all(
        row[column] == identity[column]
        for column in _CURRENT_PROVENANCE_KEY)]
    candidates = [row for row in anchor_matches if row["spec_ref"] == spec.ref]
    if not candidates:
        code = ("SPEC_LINK_MISSING" if anchor_matches
                else "PROVENANCE_EVENT_NOT_FOUND")
        _not_replayable(code,
                        "no provenance event matches the selected delivered row",
                        spec_ref=spec.ref)
    claiming = [row for row in candidates
                if _is_authoritative(row["authoritative_for"], artifact)]
    if not claiming:
        _not_replayable("PROVENANCE_NOT_AUTHORITATIVE",
                        "no matching provenance event is authoritative")
    if len(claiming) != 1:
        by_id = {
            item.mapping_id: item for item in spec.mappings
            if item.target == target
        }
        conditions = {item.id: item for item in spec.conditions}
        if (len(claiming) > 1
                and all(row.get("mapping_id") in by_id for row in claiming)
                and all(
                    by_id[row["mapping_id"]].when
                    and conditions.get(by_id[row["mapping_id"]].when) is not None
                    and conditions[by_id[row["mapping_id"]].when].record_origin
                    for row in claiming)):
            _not_replayable(
                "PROVENANCE_AUTHORITY_AMBIGUOUS",
                "multiple authoritative origin-sibling events match the same "
                "logical row", spec_ref=spec.ref)
        if (len(claiming) > 1
                and all(row.get("mapping_id") in by_id for row in claiming)
                and any(by_id[row["mapping_id"]].when for row in claiming)):
            # Choose no authority here. The first event is only an entry point;
            # Stage 15 independently evaluates every sibling and returns the
            # explicit multi-event NOT_REPLAYABLE result when several are true.
            return claiming[0], spec, binding
        _not_replayable("PROVENANCE_AUTHORITY_AMBIGUOUS",
                        f"{len(claiming)} authoritative provenance events match")
    return claiming[0], spec, binding


def _load_pinned_spec(root: Path, schema: TargetSchema,
                      binding: SpecificationBinding
                      ) -> tuple[HarmonizationSpec, SpecificationBinding]:
    _verify_hash(root, binding, "SPEC_HASH_MISMATCH")
    try:
        spec = load_spec(resolve_project_path(root, binding.path),
                         target_schema=schema, require_executable=True,
                         resolve_vocabularies=False)
    except SpecError as exc:
        _not_replayable("SPEC_LINK_MISSING",
                        f"pinned specification cannot be loaded: {exc}",
                        spec_ref=binding.ref)
    if (spec.ref != binding.ref
            or binding.identity != f"spec:{spec.ref}"
            or binding.target_schema_ref != schema.ref):
        _fail("SPEC_HASH_MISMATCH",
              "specification identity disagrees with its manifest binding",
              spec_ref=binding.ref)
    if binding.status != "approved" or spec.status != "approved":
        _not_replayable("SPEC_STATUS_NOT_APPROVED",
                        f"specification {spec.ref!r} is not approved",
                        spec_ref=spec.ref)
    return spec, binding


def _provenance_identity(spec: HarmonizationSpec,
                         row: dict[str, str]) -> dict[str, str]:
    record = spec.record
    if not record.participant_key or record.participant_key not in row:
        raise ValueError("participant identity is not representable")
    for name in (record.sample_key, record.collection_event_key):
        if name and name not in row:
            raise ValueError(f"record field {name!r} is absent")
    return {
        "trial": spec.trial,
        "cimac_part_id": row[record.participant_key],
        "Cimac.id": row[record.sample_key] if record.sample_key else "",
        "Collection_Event": (row[record.collection_event_key]
                             if record.collection_event_key else ""),
    }


def _row_identity_from_event(spec: HarmonizationSpec, schema: TargetSchema,
                             event: dict[str, str]) -> dict[str, str]:
    """Apply the current provenance-anchor adapter to a schema row identity.

    The fixed column names below belong only to the version-1 provenance
    adapter.  They are not part of the ``pev1`` payload, whose keys come from
    ``schema.row_key``.
    """
    record = spec.record
    values = {"trial": event.get("trial", "")}
    if record.participant_key:
        values[record.participant_key] = event.get("cimac_part_id", "")
    if record.sample_key:
        values[record.sample_key] = event.get("Cimac.id", "")
    if record.collection_event_key:
        values[record.collection_event_key] = event.get("Collection_Event", "")
    return canonical_row_identity(schema.row_key, values)


def _is_authoritative(value: str, artifact: ArtifactBinding) -> bool:
    if not value:
        return True
    if value == "(none)":
        return False
    names = {artifact.artifact_id, Path(artifact.path).name}
    return bool(names & {x for x in value.split(";") if x})


def _check_custom_scope(mapping: SpecMapping, event: dict[str, str],
                        artifact: ArtifactBinding) -> None:
    context = {"mapping_id": mapping.mapping_id}
    if mapping.op != OP_CUSTOM_RULE:
        _not_replayable("OPERATION_UNSUPPORTED",
                        f"operation {mapping.op!r} is outside custom replay",
                        **context)
    if mapping.when or mapping.contributions.conditions:
        _not_replayable("MAPPING_CONDITION_UNSUPPORTED",
                        "conditional custom rules are outside Stage 3", **context)
    if mapping.confidence != CONFIDENCE_FROM_RULE:
        _not_replayable("REGISTERED_RULE_CONTRACT_MISMATCH",
                        "custom rule must declare confidence: from_rule",
                        **context)
    if mapping.source or mapping.coerce != "none":
        _not_replayable("OPERATION_UNSUPPORTED",
                        "custom replay accepts declared harmonized inputs only",
                        **context)
    _require_static_present_state_policy(mapping, context)
    if event.get("extraction_method") != "spec:custom_rule":
        if event.get("extraction_method", "").startswith("post"):
            _not_replayable("POST_RULE_UNSUPPORTED_STAGE1",
                            "post rules are outside Stage 3", **context)
        _fail("DEPENDENCY_CONTEXT_MISMATCH",
              "custom provenance extraction_method is not spec:custom_rule",
              **context)
    artifact_names = {artifact.artifact_id, Path(artifact.path).name}
    if mapping.artifacts and not artifact_names.intersection(mapping.artifacts):
        _request_error("FIELD_NOT_APPLICABLE",
                       "custom mapping does not apply to selected artifact")


def _is_stage14_derivation_mapping(mapping: SpecMapping) -> bool:
    """Closed replay eligibility for unconditional existing-authority facts."""
    return (
        bool(mapping.derivation)
        and not mapping.when
        and not mapping.contributions.conditions
        and not mapping.candidates
        and not mapping.clauses
        and is_stage14_derivation(mapping.derivation)
    )


def _is_stage16_derivation_mapping(
        mapping: SpecMapping, spec: HarmonizationSpec) -> bool:
    """Closed fact-composition eligibility over existing replay authorities."""
    if (not mapping.derivation
            or mapping.candidates or mapping.clauses
            or not is_stage16_derivation(mapping.derivation)
            or mapping.op not in {
                OP_DIRECT, OP_VALUE_MAP, OP_UNIT_CONVERSION,
                OP_CONSTANT, OP_FILE_MEMBERSHIP,
            }):
        return False
    refs = frozenset().union(
        *(derivation_references(variant.record)
          for variant in mapping.derivation))
    reads = {read.id: read for read in spec.reads}
    groups = {group.id: group for group in spec.source_groups}
    conditions = {condition.id: condition for condition in spec.conditions}
    condition_ids = {
        variant.when for variant in mapping.derivation if variant.when}
    condition_ids.update(
        ref.partition(".")[2] for ref in refs
        if ref.startswith("condition."))
    if mapping.when:
        condition_ids.add(mapping.when)
    for condition_id in condition_ids:
        condition = conditions.get(condition_id)
        if condition is None or condition.record_origin:
            return False
        if condition.read:
            if condition.read not in reads:
                return False
        elif condition.group:
            if condition.group not in groups:
                return False
        else:
            return False
    for ref in refs:
        if not is_stage16_fact(ref):
            return False
        head, _, name = ref.partition(".")
        if head == "read" and name not in reads:
            return False
        if head in ("group_member", "group_value", "group_label") \
                and name not in groups:
            return False
    return True


def _is_stage17_raw_route(
        mapping: SpecMapping, spec: HarmonizationSpec) -> bool:
    """Closed operation-route eligibility for finalized raw authority."""
    if mapping.op == OP_CUSTOM_RULE:
        return False
    if mapping.op in {OP_CONSTANT, OP_NOT_APPLICABLE,
                      OP_NO_MAPPING_CONFIGURED}:
        return True
    if mapping.op in {OP_DIRECT, OP_VALUE_MAP, OP_UNIT_CONVERSION}:
        if is_stage17_source_group_mapping(spec, mapping):
            return True
        return bool(
            mapping.source and mapping.source != ANCHOR_SOURCE_ID
            and GROUP_REF_SEP not in mapping.source
            and not mapping.column_by_member)
    if mapping.op == OP_FILE_MEMBERSHIP:
        return is_stage17_source_group_mapping(spec, mapping)
    if mapping.op == OP_CONDITIONAL:
        if (not mapping.clauses or mapping.candidates or mapping.map
                or mapping.fallback_vocabulary):
            return False
        if not mapping.source:
            return not mapping.column_by_member
        if GROUP_REF_SEP in mapping.source:
            group = source_group_for_mapping(spec, mapping)
            return group is not None and group.select == "first_containing_record"
        source = spec.source(mapping.source)
        return bool(source is not None and source.path and not source.path_pattern)
    if mapping.op == OP_FALLBACK:
        if (not mapping.candidates or mapping.map or mapping.fallback_vocabulary
                or mapping.source or mapping.group or mapping.column_by_member
                or mapping.coerce not in ("none", "numeric")
                or mapping.contributions.record != "all"):
            return False
        return all(
            candidate.source and candidate.source != ANCHOR_SOURCE_ID
            and GROUP_REF_SEP not in candidate.source
            and spec.source(candidate.source) is not None
            and bool(spec.source(candidate.source).path)
            and not spec.source(candidate.source).path_pattern
            for candidate in mapping.candidates)
    return False


def _is_stage17_derivation_mapping(
        mapping: SpecMapping, spec: HarmonizationSpec) -> bool:
    """Stage 16 fact composition plus raw on an approved causal route."""
    if (not mapping.derivation or not is_stage17_derivation(mapping.derivation)
            or not _is_stage17_raw_route(mapping, spec)):
        return False
    refs = frozenset().union(
        *(derivation_references(variant.record)
          for variant in mapping.derivation))
    reads = {read.id: read for read in spec.reads}
    groups = {group.id: group for group in spec.source_groups}
    conditions = {condition.id: condition for condition in spec.conditions}
    condition_ids = {
        variant.when for variant in mapping.derivation if variant.when}
    condition_ids.update(
        ref.partition(".")[2] for ref in refs
        if ref.startswith("condition."))
    if mapping.when:
        condition_ids.add(mapping.when)
    for condition_id in condition_ids:
        condition = conditions.get(condition_id)
        if condition is None or condition.record_origin:
            return False
        if condition.read:
            if condition.read not in reads:
                return False
        elif condition.group:
            if condition.group not in groups:
                return False
        else:
            return False
    for ref in refs:
        if ref == "raw_value":
            continue
        if not is_stage16_fact(ref):
            return False
        head, _, name = ref.partition(".")
        if head == "read" and name not in reads:
            return False
        if head in ("group_member", "group_value", "group_label") \
                and name not in groups:
            return False
    return True


def _is_stage18_derivation_mapping(
        mapping: SpecMapping, spec: HarmonizationSpec) -> bool:
    """Narrow present-state composition over an already verified route.

    Existing Stage 14/16 routes need no new derivation semantics.  The only
    additional operation shape is a supported conditional whose derivation
    consumes the already verified value/source-column facts.
    """
    if (mapping.op != OP_CONDITIONAL or not mapping.derivation
            or not is_stage16_derivation(mapping.derivation)
            or not _is_stage17_raw_route(mapping, spec)):
        return False
    try:
        finalize_declared_value_state(
            mapping.value_state, mapping.provenance.value_state)
    except ValueError:
        return False
    refs = frozenset().union(
        *(derivation_references(variant.record)
          for variant in mapping.derivation))
    return refs <= STAGE14_FACTS


def _is_stage19_construction_route(
        mapping: SpecMapping, spec: HarmonizationSpec) -> bool:
    """Closed grouping-construction consumers approved for Stage 19."""
    record = spec.record
    grouping = record.grouping
    if (not mapping_uses_record_construction(spec, mapping)
            or record.anchor != "grouping" or grouping is None
            or grouping.select != "first_row"
            or record.duplicates.policy not in {"error", "first", "last"}
            or len(record.additional_records) > 1
            or mapping.contributions.conditions
            or mapping.group or mapping.column_by_member
            or mapping.candidates or mapping.clauses or mapping.inputs
            or mapping.params or mapping.fallback_vocabulary
            or mapping.fallback_rules or mapping.provenance.raw_from):
        return False
    if any(
            declaration.when != "no_record_for_participant"
            or not declaration.id or not declaration.participants_from
            or declaration.rule is None
            for declaration in record.additional_records):
        return False
    if mapping.op == OP_DIRECT:
        if (mapping.source != record.anchor_source or not mapping.column
                or mapping.coerce not in {"none", "text", "numeric"}
                or mapping.map):
            return False
    elif mapping.op == OP_CONSTANT:
        if mapping.source or mapping.map:
            return False
    else:
        return False
    conditions = {condition.id: condition for condition in spec.conditions}
    if mapping.when:
        condition = conditions.get(mapping.when)
        if condition is None or not condition.record_origin:
            return False
    if mapping.derivation:
        refs = frozenset().union(
            *(derivation_references(variant.record)
              for variant in mapping.derivation))
        if not all(is_stage19_fact(ref) for ref in refs):
            return False
        for variant in mapping.derivation:
            if variant.when:
                condition = conditions.get(variant.when)
                if condition is None or not condition.record_origin:
                    return False
    return True


def _is_stage19_derivation_mapping(
        mapping: SpecMapping, spec: HarmonizationSpec) -> bool:
    """Stage 19 construction-backed composition using the existing renderer."""
    return bool(
        mapping.derivation
        and _is_stage19_construction_route(mapping, spec)
        and (mapping_uses_grouping_facts(mapping)
             or mapping_uses_record_origin(spec, mapping))
    )


def _check_stage19_construction_scope(
        mapping: SpecMapping, spec: HarmonizationSpec,
        event: dict[str, str], artifact: ArtifactBinding) -> None:
    """Fail closed before record construction can broaden operation replay."""
    context = {"spec_ref": spec.ref, "mapping_id": mapping.mapping_id}
    if not _is_stage19_construction_route(mapping, spec):
        _not_replayable(
            "ANCHOR_SOURCE_UNSUPPORTED_STAGE1",
            "mapping is outside the closed Stage 19 construction boundary",
            **context)
    _require_static_present_state_policy(mapping, context)
    artifact_names = {artifact.artifact_id, Path(artifact.path).name}
    if mapping.artifacts and not artifact_names.intersection(mapping.artifacts):
        _request_error(
            "FIELD_NOT_APPLICABLE",
            "construction-backed mapping does not apply to selected artifact")


def _require_static_present_state_policy(
        mapping: SpecMapping, context: Mapping[str, Any]) -> None:
    """Require pinned static present-state authority, never recorded state."""
    try:
        finalize_declared_value_state(
            mapping.value_state, mapping.provenance.value_state)
    except ValueError as exc:
        code = (
            "SUPPRESSION_UNSUPPORTED_STAGE1"
            if mapping.value_state == "suppressed"
            else "EXCLUSION_REQUIRES_ROW_REPLAY"
            if mapping.value_state == "row_excluded"
            else "VALUE_STATE_UNSUPPORTED")
        _not_replayable(code, str(exc), **context)


def _verify_static_present_value_state(
        event: Mapping[str, str], mapping: SpecMapping, *,
        source_ref_id: str, spec_ref: str) -> None:
    """Compare a recorded Cell state after lower causal replay succeeds."""
    context = {
        "spec_ref": spec_ref, "mapping_id": mapping.mapping_id,
        "source_ref_id": source_ref_id,
    }
    try:
        finalized = finalize_declared_value_state(
            mapping.value_state, mapping.provenance.value_state)
    except ValueError as exc:
        _not_replayable("VALUE_STATE_UNSUPPORTED", str(exc), **context)
    observed = event.get("value_state", "")
    if observed != finalized.serialized_state:
        _fail(
            "VALUE_STATE_CONTEXT_MISMATCH",
            f"provenance value_state {observed!r} differs from pinned "
            f"effective state {finalized.serialized_state!r}", **context)


def _check_constant_scope(mapping: SpecMapping, spec: HarmonizationSpec,
                          event: dict[str, str],
                          artifact: ArtifactBinding) -> None:
    """Enforce the closed source-independent Stage 6 declaration subset."""
    context = {"spec_ref": spec.ref, "mapping_id": mapping.mapping_id}
    if mapping.op != OP_CONSTANT:
        _not_replayable("OPERATION_UNSUPPORTED",
                        "operation is outside Stage 6 constant replay", **context)
    if mapping.when:
        _not_replayable("MAPPING_CONDITION_UNSUPPORTED",
                        "conditional constants are outside Stage 6", **context)
    if mapping.source == ANCHOR_SOURCE_ID:
        _not_replayable("ANCHOR_SOURCE_UNSUPPORTED_STAGE1",
                        "anchor-context constants are outside Stage 6", **context)
    if (mapping.source or mapping.group or mapping.column_by_member
            or mapping.provenance.source_ref
            or mapping.provenance.row_from
            or mapping.provenance.contributions_from
            or mapping.provenance.row_number
            or mapping.provenance.record_key_from == "group"):
        _not_replayable(
            "SOURCE_NOT_SINGLE_FILE",
            "constant carries source, group, member, or row context", **context)
    if mapping.provenance.raw_from:
        _not_replayable("DERIVATION_UNSUPPORTED_STAGE2",
                        "constant raw evidence comes from a derived read",
                        **context)
    if mapping.derivation and not _is_stage14_derivation_mapping(mapping):
        _not_replayable("DERIVATION_UNSUPPORTED_STAGE2",
                        "structured constant derivation is outside Stage 6",
                        **context)
    if mapping.contributions.conditions:
        _not_replayable("CONTRIBUTIONS_UNSUPPORTED_STAGE2",
                        "condition contributions are outside Stage 6", **context)
    if mapping.provenance.raw_value not in ("", "none"):
        _not_replayable("RAW_VALUE_KIND_UNSUPPORTED",
                        "constant raw policy must be declared-value or none",
                        **context)
    _require_static_present_state_policy(mapping, context)
    artifact_names = {artifact.artifact_id, Path(artifact.path).name}
    if mapping.artifacts and not artifact_names.intersection(mapping.artifacts):
        _request_error("FIELD_NOT_APPLICABLE",
                       "constant mapping does not apply to selected artifact")


def _check_declared_absence_scope(
        mapping: SpecMapping, spec: HarmonizationSpec,
        event: dict[str, str], artifact: ArtifactBinding) -> None:
    """Enforce the closed Stage 7/8 declared-absence declaration subsets."""
    context = {"spec_ref": spec.ref, "mapping_id": mapping.mapping_id}
    if mapping.op != OP_NOT_APPLICABLE:
        _not_replayable(
            "OPERATION_UNSUPPORTED",
            "operation is outside Stage 7 declared-absence replay", **context)
    if mapping.when:
        _not_replayable(
            "MAPPING_CONDITION_UNSUPPORTED",
            "conditional declared absences are outside Stage 7", **context)
    if mapping.source == ANCHOR_SOURCE_ID:
        _not_replayable(
            "ANCHOR_SOURCE_UNSUPPORTED_STAGE1",
            "anchor-context declared absences are outside Stage 7", **context)
    provenance = mapping.provenance
    if (mapping.source or mapping.group or mapping.column_by_member
            or provenance.row_from or provenance.contributions_from):
        _not_replayable(
            "SOURCE_NOT_SINGLE_FILE",
            "declared absence carries source, group, member, or row context",
            **context)
    if provenance.source_ref:
        if (GROUP_REF_SEP in provenance.source_ref
                or provenance.source_label
                or provenance.row_number != "none"):
            _not_replayable(
                "SOURCE_NOT_SINGLE_FILE",
                "attributed declared absence requires one literal source "
                "reference and an explicitly absent row locator", **context)
        source = spec.source(provenance.source_ref)
        if source is None:
            _not_replayable(
                "SOURCE_LINK_MISSING",
                "attributed declared absence names no declared source",
                **context)
        if not source.path or source.path_pattern:
            _not_replayable(
                "SOURCE_NOT_SINGLE_FILE",
                "attributed declared absence requires one exact source path",
                **context)
        if (source.file_type != "csv" or source.reader != "cidc_csv@v1"
                or source.sheet):
            _not_replayable(
                "READER_UNSUPPORTED",
                "attributed declared absence requires cidc_csv@v1 without "
                "a sheet", **context)
        if (provenance.record_key_from == "group"
                and spec.record.grouping is None):
            _not_replayable(
                "SOURCE_NOT_SINGLE_FILE",
                "group record-key policy requires a grouping anchor", **context)
    elif (provenance.row_number not in ("", "none")
            or provenance.record_key_from == "group"):
        _not_replayable(
            "SOURCE_NOT_SINGLE_FILE",
            "source-independent declared absence carries row or group context",
            **context)
    if provenance.raw_from:
        _not_replayable(
            "DERIVATION_UNSUPPORTED_STAGE2",
            "declared-absence raw evidence comes from a derived read", **context)
    if mapping.derivation:
        _not_replayable(
            "DERIVATION_UNSUPPORTED_STAGE2",
            "structured declared-absence derivation is outside Stage 7",
            **context)
    if mapping.contributions.conditions:
        _not_replayable(
            "CONTRIBUTIONS_UNSUPPORTED_STAGE2",
            "condition contributions are outside Stage 7", **context)
    if provenance.raw_value not in ("", "value", "coerced", "none"):
        _not_replayable(
            "RAW_VALUE_KIND_UNSUPPORTED",
            "declared-absence raw policy must be value, coerced, or none",
            **context)
    if mapping.value_state not in _DECLARED_ABSENCE_STATES:
        _not_replayable(
            "VALUE_STATE_UNSUPPORTED",
            f"declared value_state {mapping.value_state!r} is outside Stage 7",
            **context)
    artifact_names = {artifact.artifact_id, Path(artifact.path).name}
    if mapping.artifacts and not artifact_names.intersection(mapping.artifacts):
        _request_error(
            "FIELD_NOT_APPLICABLE",
            "declared-absence mapping does not apply to selected artifact")


def _contains_vocabulary_reference(value: Any) -> bool:
    if isinstance(value, dict):
        if VOCABULARY_MARKER in value:
            return True
        return any(_contains_vocabulary_reference(item)
                   for item in value.values())
    if isinstance(value, (list, tuple)):
        return any(_contains_vocabulary_reference(item) for item in value)
    return False


def _top_level_vocabulary_parameters(
        params: Mapping[str, Any], *, spec_ref: str, mapping_id: str
        ) -> dict[str, str]:
    """Return the one approved top-level marker shape; refuse every other."""
    found: dict[str, str] = {}
    for parameter, value in params.items():
        if isinstance(value, dict) and set(value) == {VOCABULARY_MARKER}:
            ref = value.get(VOCABULARY_MARKER)
            if not isinstance(ref, str) or not ref:
                _not_replayable(
                    "EXTERNAL_VOCABULARY_UNSUPPORTED",
                    "vocabulary parameter reference is not a pinned id@version",
                    spec_ref=spec_ref, mapping_id=mapping_id)
            found[parameter] = ref
        elif _contains_vocabulary_reference(value):
            _not_replayable(
                "EXTERNAL_VOCABULARY_UNSUPPORTED",
                "nested or mixed vocabulary parameters are outside Stage 5",
                spec_ref=spec_ref, mapping_id=mapping_id)
    if len(found) > 1:
        _not_replayable(
            "EXTERNAL_VOCABULARY_UNSUPPORTED",
            "Stage 5 supports one top-level vocabulary parameter per mapping",
            spec_ref=spec_ref, mapping_id=mapping_id)
    return found


def _verified_vocabulary(root: Path, manifest: ReplayManifest,
                         event: dict[str, str], usage, *, expected_ref: str,
                         expected_kind: str, spec_ref: str,
                         mapping_id: str) -> vocabularies.Vocabulary:
    """Load only one exact usage-bound vocabulary and verify its identity."""
    context = {"spec_ref": spec_ref, "mapping_id": mapping_id}
    event_refs = {item for item in event.get("evidence_refs", "").split(";")
                  if item}
    usage_refs = set(usage.evidence_ref_ids)
    bindings_by_id = {binding.evidence_ref_id: binding
                      for binding in manifest.evidence}
    candidate_ids = {
        ref for ref in event_refs | usage_refs
        if ref.startswith("vocab-")
        or (ref in bindings_by_id
            and (bindings_by_id[ref].kind == "vocabulary"
                 or bindings_by_id[ref].reference == expected_ref))
    }
    if not candidate_ids:
        _not_replayable(
            "VOCABULARY_EVIDENCE_MISSING",
            f"mapping has no runtime evidence for vocabulary {expected_ref!r}",
            **context)
    if len(candidate_ids) != 1:
        _fail("VOCABULARY_IDENTITY_MISMATCH",
              "mapping claims more than one vocabulary binding", **context)
    evidence_ref_id = next(iter(candidate_ids))
    if evidence_ref_id not in event_refs:
        _not_replayable(
            "VOCABULARY_EVIDENCE_MISSING",
            "authoritative Cell omits its vocabulary evidence reference",
            **context)
    if evidence_ref_id not in usage_refs:
        _not_replayable(
            "VOCABULARY_EVIDENCE_MISSING",
            "MappingUsage omits its vocabulary evidence reference", **context)
    binding = bindings_by_id.get(evidence_ref_id)
    if binding is None:
        _not_replayable(
            "VOCABULARY_EVIDENCE_MISSING",
            "manifest omits the referenced vocabulary EvidenceBinding",
            **context)
    if (binding.kind != "vocabulary"
            or binding.identity != f"vocabulary:{evidence_ref_id}"
            or binding.reference != expected_ref
            or not binding.path or not binding.sha256):
        _fail("VOCABULARY_IDENTITY_MISMATCH",
              "vocabulary EvidenceBinding contradicts the pinned use",
              **context)
    try:
        path = resolve_project_path(root, binding.path)
        payload = path.read_bytes()
    except (ManifestError, OSError) as exc:
        _fail("VOCABULARY_IDENTITY_MISMATCH",
              f"bound vocabulary cannot be read as declared: {exc}", **context)
    if hashlib.sha256(payload).hexdigest() != binding.sha256:
        _fail("VOCABULARY_HASH_MISMATCH",
              "bound vocabulary bytes differ from the manifest hash", **context)
    try:
        loaded = vocabularies.load_bytes(payload, source_path=path)
    except vocabularies.VocabularyError as exc:
        _fail("VOCABULARY_IDENTITY_MISMATCH",
              f"bound vocabulary cannot be parsed as declared: {exc}", **context)
    try:
        loaded_path = loaded.contained_path(root)
        loaded_evidence_ref_id = loaded.evidence_ref_id(root)
    except vocabularies.VocabularyError as exc:
        _fail("VOCABULARY_IDENTITY_MISMATCH", str(exc), **context)
    details = dict(binding.details)
    required_details = set(loaded.evidence_details())
    if not required_details.issubset(details):
        _not_replayable(
            "VOCABULARY_EVIDENCE_MISSING",
            "vocabulary binding omits parser or declarative metadata",
            **context)
    if (loaded_path != binding.path
            or loaded_evidence_ref_id != evidence_ref_id
            or loaded.ref != binding.reference
            or details != loaded.evidence_details()):
        _fail("VOCABULARY_IDENTITY_MISMATCH",
              "bound path/ref/kind/metadata disagrees with parsed vocabulary",
              **context)
    if loaded.kind != expected_kind:
        _not_replayable(
            "EXTERNAL_VOCABULARY_UNSUPPORTED",
            f"vocabulary kind {loaded.kind!r} is outside this Stage 5 use",
            **context)
    return loaded


def _verify_custom_rule(event: dict[str, str], expected: RuleRef,
                        implementation: RuleRef, usage,
                        manifest: ReplayManifest, spec_ref: str,
                        mapping_id: str) -> None:
    context = {"spec_ref": spec_ref, "mapping_id": mapping_id}
    if not event.get("rule_id") or not event.get("rule_version"):
        _not_replayable("MANIFEST_REQUIRED_FIELD_MISSING",
                        "custom provenance rule identity is absent", **context)
    if (event["rule_id"], event["rule_version"]) != \
            (expected.id, expected.version):
        _fail("RULE_IDENTITY_MISMATCH",
              "custom provenance rule differs from recomputed outcome", **context)
    expected_ref = f"{expected.id}@{expected.version}"
    if expected_ref not in usage.rule_refs:
        _not_replayable("MANIFEST_REQUIRED_FIELD_MISSING",
                        f"mapping usage omits outcome rule {expected_ref!r}",
                        **context)
    bindings = [rule for rule in manifest.rules if rule.ref == expected_ref]
    if not bindings:
        _not_replayable("MANIFEST_REQUIRED_FIELD_MISSING",
                        f"manifest omits outcome rule {expected_ref!r}", **context)
    implementation_ref = f"{implementation.id}@{implementation.version}"
    required_kind = (rule_registry.KIND_FIELD
                     if expected_ref == implementation_ref else "mapping")
    if bindings[0].kind != required_kind:
        _fail("RULE_IDENTITY_MISMATCH",
              f"outcome rule {expected_ref!r} has wrong manifest kind",
              **context)
    recorded_evidence = tuple(
        item for item in event.get("evidence_refs", "").split(";") if item)
    if set(recorded_evidence) != set(usage.evidence_ref_ids):
        if not recorded_evidence and usage.evidence_ref_ids:
            _not_replayable("MANIFEST_REQUIRED_FIELD_MISSING",
                            "custom provenance omits scoped evidence links",
                            **context)
        _fail("DEPENDENCY_CONTEXT_MISMATCH",
              "custom provenance evidence links disagree with usage", **context)


def _thaw_manifest_value(value: Any) -> Any:
    """Convert frozen manifest detail tuples back to JSON-like values."""
    if isinstance(value, tuple):
        if all(isinstance(item, tuple) and len(item) == 2 for item in value):
            return {str(key): _thaw_manifest_value(item)
                    for key, item in value}
        return [_thaw_manifest_value(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _thaw_manifest_value(item)
                for key, item in value.items()}
    return value


def _participant_record_key(record, participant: str) -> str:
    return f"{record.record_key_column or record.participant_key}={participant}"


def _parse_grouping_record_key(payload: str, columns: tuple[str, ...], **context
                               ) -> dict[str, str]:
    parts = payload.split(";") if payload else []
    if len(parts) != len(columns):
        _fail("ANCHOR_SELECTION_MISMATCH",
              "grouping record key has the wrong arity", **context)
    out: dict[str, str] = {}
    for expected, part in zip(columns, parts):
        if "=" not in part:
            _fail("ANCHOR_SELECTION_MISMATCH",
                  "grouping record key is malformed", **context)
        name, value = part.split("=", 1)
        if name != expected or name in out:
            _fail("ANCHOR_SELECTION_MISMATCH",
                  "grouping record key contradicts declared key order",
                  **context)
        out[name] = value
    return out


def _verify_anchor_contribution_source(
        contribution: dict[str, Any], binding: SourceBinding,
        expected_record_key: str, expected_source_ref_id: str, **context) -> None:
    if (contribution.get("source_ref_id") != expected_source_ref_id
            or contribution.get("source_logical_label") != binding.logical_label
            or contribution.get("source_resolved_path") != binding.path
            or contribution.get("source_sha256") != binding.sha256
            or str(contribution.get("source_sheet") or "") != binding.sheet
            or contribution.get("source_record_key") != expected_record_key):
        _fail("ANCHOR_SELECTION_MISMATCH",
              "anchor Contribution source identity is contradictory",
              source_ref_id=expected_source_ref_id, **context)


def _check_anchor_source_declaration(
        root: Path, spec: HarmonizationSpec, source_id: str, source_decl,
        binding: SourceBinding, mapping_id: str) -> None:
    context = {"spec_ref": spec.ref, "mapping_id": mapping_id,
               "source_ref_id": binding.source_ref_id}
    if source_decl is None:
        _not_replayable("ANCHOR_EVIDENCE_MISSING",
                        f"anchor source {source_id!r} is undeclared", **context)
    if (binding.identity != f"source:{binding.source_ref_id}"
            or binding.spec_ref != spec.ref
            or binding.source_id != source_id
            or binding.config_key != source_id):
        _fail("ANCHOR_IDENTITY_MISMATCH",
              "anchor SourceBinding contradicts its declaration", **context)
    if (not source_decl.path or source_decl.path_pattern
            or source_decl.file_type != "csv"
            or source_decl.reader != "cidc_csv@v1"
            or source_decl.sheet or binding.sheet):
        _not_replayable(
            "ANCHOR_EVIDENCE_MISSING",
            "Stage 9 requires exact-path cidc_csv@v1 anchor inputs", **context)
    try:
        declared_path = resolve_project_path(root, source_decl.path)
        bound_path = resolve_project_path(root, binding.path)
    except ManifestError as exc:
        _not_replayable("MANIFEST_PATH_OUTSIDE_PROJECT", str(exc), **context)
    expected_label = source_decl.logical_label or declared_path.name
    if (declared_path != bound_path
            or binding.logical_label != expected_label
            or binding.file_type != "csv"
            or binding.reader.reader_id != "cidc_csv@v1"):
        _fail("ANCHOR_IDENTITY_MISMATCH",
              "anchor path, label, type, or reader is contradictory",
              **context)


def _check_no_mapping_scope(mapping: SpecMapping, spec: HarmonizationSpec,
                            event: dict[str, str],
                            artifact: ArtifactBinding) -> None:
    """Enforce the closed Stage 9 grouping-anchor declaration subset."""
    context = {"spec_ref": spec.ref, "mapping_id": mapping.mapping_id}
    record = spec.record
    grouping = record.grouping
    if mapping.op != OP_NO_MAPPING_CONFIGURED:
        _not_replayable("OPERATION_UNSUPPORTED",
                        "operation is outside Stage 9", **context)
    if (record.anchor != "grouping" or grouping is None
            or mapping.source != ANCHOR_SOURCE_ID):
        code = ("ANCHOR_SOURCE_UNSUPPORTED_STAGE1"
                if mapping.source == ANCHOR_SOURCE_ID
                else "OPERATION_UNSUPPORTED")
        _not_replayable(code,
                        "Stage 9 supports only grouping-anchor no-mapping",
                        **context)
    if mapping.when:
        _not_replayable("MAPPING_CONDITION_UNSUPPORTED",
                        "condition-bearing no-mapping is outside Stage 9",
                        **context)
    if (mapping.derivation or event.get("derivation")
            or mapping.contributions.conditions):
        _not_replayable("DERIVATION_UNSUPPORTED_STAGE1",
                        "derived or condition evidence is outside Stage 9",
                        **context)
    if (mapping.provenance.source_ref or mapping.provenance.row_from
            or mapping.provenance.raw_from
            or mapping.provenance.contributions_from
            or mapping.group or mapping.column_by_member):
        _not_replayable("ANCHOR_SOURCE_UNSUPPORTED_STAGE1",
                        "mapping carries source/group context outside its anchor",
                        **context)
    if (mapping.fallback_vocabulary or mapping.fallback_rules
            or mapping.candidates or mapping.clauses or mapping.inputs
            or mapping.params):
        _not_replayable("OPERATION_UNSUPPORTED",
                        "no-mapping declaration contains unsupported consumers",
                        **context)
    if (not mapping.column or mapping.column not in grouping.anchor_fields
            or grouping.select != "first_row"):
        _not_replayable("ANCHOR_SOURCE_UNSUPPORTED_STAGE1",
                        "anchor field projection/selection is outside Stage 9",
                        **context)
    if (mapping.provenance.raw_value not in ("", "value")
            or mapping.provenance.row_number not in ("", "none")
            or mapping.provenance.record_key
            or mapping.provenance.record_key_from not in ("", "participant")
            or mapping.value_state not in ("", "source_missing")):
        _not_replayable("VALUE_STATE_UNSUPPORTED",
                        "no-mapping provenance policy is outside Stage 9",
                        **context)
    if record.duplicates.policy not in {"error", "first", "last"}:
        _not_replayable("ANCHOR_SOURCE_UNSUPPORTED_STAGE1",
                        "anchor duplicate policy is unsupported", **context)
    if len(record.additional_records) > 1:
        _not_replayable(
            "ANCHOR_SOURCE_UNSUPPORTED_STAGE1",
            "Stage 9 supports at most one synthesized-record route", **context)
    for declaration in record.additional_records:
        if (declaration.when != "no_record_for_participant"
                or not declaration.id or not declaration.participants_from
                or declaration.rule is None
                or (mapping.column not in declaration.values
                    and mapping.column not in {
                        record.participant_key, record.sample_key,
                        record.collection_event_key})):
            _not_replayable(
                "ANCHOR_SOURCE_UNSUPPORTED_STAGE1",
                "additional-record route is outside Stage 9", **context)
    expected_method = "spec:no_mapping_configured"
    if event.get("extraction_method") != expected_method:
        method = event.get("extraction_method", "")
        if method.startswith("post"):
            _not_replayable("POST_RULE_UNSUPPORTED_STAGE1",
                            "post-rule event is outside Stage 9", **context)
        if method == "value_suppressed":
            _not_replayable("SUPPRESSION_UNSUPPORTED_STAGE1",
                            "suppression event is outside Stage 9", **context)
        if method == "row_excluded":
            _not_replayable("EXCLUSION_REQUIRES_ROW_REPLAY",
                            "exclusion event is outside Stage 9", **context)
        _fail("SOURCE_DECLARATION_MISMATCH",
              f"provenance extraction_method is not {expected_method}",
              **context)
    artifact_names = {artifact.artifact_id, Path(artifact.path).name}
    if mapping.artifacts and not artifact_names.intersection(mapping.artifacts):
        _request_error("FIELD_NOT_APPLICABLE",
                       "mapping does not apply to the selected artifact")



def _check_grouping_anchor_direct_scope(
        mapping: SpecMapping, spec: HarmonizationSpec,
        event: dict[str, str], artifact: ArtifactBinding) -> None:
    """Enforce the closed Stage 10 grouping-anchor direct subset."""
    context = {"spec_ref": spec.ref, "mapping_id": mapping.mapping_id}
    _require_static_present_state_policy(mapping, context)
    record = spec.record
    grouping = record.grouping
    if mapping.op != OP_DIRECT:
        _not_replayable(
            "OPERATION_UNSUPPORTED", "operation is outside Stage 10", **context)
    if (mapping.source != ANCHOR_SOURCE_ID
            or record.anchor != "grouping" or grouping is None):
        _not_replayable(
            "ANCHOR_SOURCE_UNSUPPORTED_STAGE1",
            "Stage 10 supports only grouping-anchor direct mappings", **context)
    if mapping.when:
        _not_replayable(
            "MAPPING_CONDITION_UNSUPPORTED",
            "condition-bearing anchor direct is outside Stage 10", **context)
    if (mapping.derivation or event.get("derivation")
            or mapping.contributions.conditions):
        _not_replayable(
            "DERIVATION_UNSUPPORTED_STAGE1",
            "derived or condition evidence is outside Stage 10", **context)
    if (mapping.group or mapping.column_by_member
            or mapping.provenance.row_from
            or mapping.provenance.raw_from):
        _not_replayable(
            "ANCHOR_SOURCE_UNSUPPORTED_STAGE1",
            "mapping carries source/group context outside its anchor", **context)
    if (mapping.map or mapping.fallback_vocabulary or mapping.fallback_rules
            or mapping.candidates or mapping.clauses or mapping.inputs
            or mapping.params):
        _not_replayable(
            "OPERATION_UNSUPPORTED",
            "anchor direct contains unsupported operation consumers", **context)
    if mapping.coerce not in ("none", "text", "numeric"):
        _not_replayable(
            "OPERATION_UNSUPPORTED",
            f"direct coercion {mapping.coerce!r} is unsupported", **context)
    if (not mapping.column or mapping.column not in grouping.anchor_fields
            or grouping.select != "first_row"):
        _not_replayable(
            "ANCHOR_SOURCE_UNSUPPORTED_STAGE1",
            "anchor field projection/selection is outside Stage 10", **context)
    provenance = mapping.provenance
    if (provenance.raw_value not in ("", "value")
            or provenance.record_key_from == "group"):
        _not_replayable(
            "RAW_VALUE_KIND_UNSUPPORTED",
            "anchor-direct raw/record-key policy is outside Stage 10", **context)
    if provenance.source_ref:
        if (GROUP_REF_SEP in provenance.source_ref
                or provenance.source_label
                or provenance.row_number != "none"):
            _not_replayable(
                "SOURCE_NOT_SINGLE_FILE",
                "anchor-direct attribution requires one literal source and "
                "an explicitly absent row locator", **context)
        source = spec.source(provenance.source_ref)
        if source is None:
            _not_replayable(
                "SOURCE_LINK_MISSING",
                "anchor-direct attribution names no declared source", **context)
        if not source.path or source.path_pattern:
            _not_replayable(
                "SOURCE_NOT_SINGLE_FILE",
                "anchor-direct attribution requires one exact source path",
                **context)
        if (source.file_type != "csv" or source.reader != "cidc_csv@v1"
                or source.sheet):
            _not_replayable(
                "READER_UNSUPPORTED",
                "anchor-direct attribution requires cidc_csv@v1 without a sheet",
                **context)
    elif provenance.row_number not in ("", "none"):
        _not_replayable(
            "SOURCE_NOT_SINGLE_FILE",
            "anchor-direct provenance claims an undeclared source row", **context)
    if record.duplicates.policy not in {"error", "first", "last"}:
        _not_replayable(
            "ANCHOR_SOURCE_UNSUPPORTED_STAGE1",
            "anchor duplicate policy is unsupported", **context)
    if len(record.additional_records) > 1:
        _not_replayable(
            "ANCHOR_SOURCE_UNSUPPORTED_STAGE1",
            "Stage 10 supports at most one synthesized-record route", **context)
    for declaration in record.additional_records:
        if (declaration.when != "no_record_for_participant"
                or not declaration.id or not declaration.participants_from
                or declaration.rule is None
                or (mapping.column not in declaration.values
                    and mapping.column not in {
                        record.participant_key, record.sample_key,
                        record.collection_event_key})):
            _not_replayable(
                "ANCHOR_SOURCE_UNSUPPORTED_STAGE1",
                "additional-record route is outside Stage 10", **context)
    expected_method = "spec:direct"
    if event.get("extraction_method") != expected_method:
        method = event.get("extraction_method", "")
        if method.startswith("post"):
            _not_replayable(
                "POST_RULE_UNSUPPORTED_STAGE1",
                "post-rule event is outside Stage 10", **context)
        if method == "value_suppressed":
            _not_replayable(
                "SUPPRESSION_UNSUPPORTED_STAGE1",
                "suppression event is outside Stage 10", **context)
        if method == "row_excluded":
            _not_replayable(
                "EXCLUSION_REQUIRES_ROW_REPLAY",
                "exclusion event is outside Stage 10", **context)
        _fail(
            "SOURCE_DECLARATION_MISMATCH",
            f"provenance extraction_method is not {expected_method}", **context)
    artifact_names = {artifact.artifact_id, Path(artifact.path).name}
    if mapping.artifacts and not artifact_names.intersection(mapping.artifacts):
        _request_error(
            "FIELD_NOT_APPLICABLE",
            "mapping does not apply to the selected artifact")


def _check_external_anchor_scope(
        mapping: SpecMapping, spec: HarmonizationSpec,
        event: dict[str, str], artifact: ArtifactBinding) -> None:
    """Enforce the closed Stage 11 external-anchor operation subset."""
    context = {"spec_ref": spec.ref, "mapping_id": mapping.mapping_id}
    _require_static_present_state_policy(mapping, context)
    if (mapping.source != ANCHOR_SOURCE_ID
            or spec.record.anchor != "external"):
        _not_replayable(
            "ANCHOR_SOURCE_UNSUPPORTED_STAGE1",
            "Stage 11 requires an external-anchor scalar source", **context)
    if mapping.op not in (OP_DIRECT, OP_VALUE_MAP):
        _not_replayable(
            "OPERATION_UNSUPPORTED",
            "operation is outside Stage 11 external-anchor replay", **context)
    if mapping.when:
        _not_replayable(
            "MAPPING_CONDITION_UNSUPPORTED",
            "condition-bearing external mappings are outside Stage 11",
            **context)
    if ((mapping.derivation or event.get("derivation"))
            and not _is_stage17_derivation_mapping(mapping, spec)):
        code = ("DERIVATION_UNSUPPORTED_STAGE1" if mapping.op == OP_DIRECT
                else "DERIVATION_UNSUPPORTED_STAGE2")
        _not_replayable(
            code, "external-anchor derivations are outside Stage 11", **context)
    if mapping.contributions.conditions:
        code = ("CONTRIBUTIONS_UNSUPPORTED_STAGE1"
                if mapping.op == OP_DIRECT
                else "CONTRIBUTIONS_UNSUPPORTED_STAGE2")
        _not_replayable(
            code, "candidate conditions are outside Stage 11", **context)
    if (mapping.group or mapping.column_by_member
            or mapping.provenance.row_from
            or mapping.provenance.raw_from):
        _not_replayable(
            "SOURCE_NOT_SINGLE_FILE",
            "mapping carries source/group context outside its external anchor",
            **context)
    if (mapping.candidates or mapping.clauses or mapping.inputs
            or mapping.params or mapping.fallback_vocabulary
            or mapping.fallback_rules):
        _not_replayable(
            "OPERATION_UNSUPPORTED",
            "external mapping contains unsupported operation consumers",
            **context)
    if mapping.op == OP_DIRECT:
        if mapping.map or mapping.coerce not in ("none", "text", "numeric"):
            _not_replayable(
                "OPERATION_UNSUPPORTED",
                "external direct coercion/declaration is outside Stage 11",
                **context)
    elif (not mapping.map or mapping.coerce != "none"):
        _not_replayable(
            "OPERATION_UNSUPPORTED",
            "Stage 11 value_map requires an inline map with coerce:none",
            **context)
    provenance = mapping.provenance
    if (provenance.raw_value not in ("", "value")
            or provenance.record_key_from == "group"):
        _not_replayable(
            "RAW_VALUE_KIND_UNSUPPORTED",
            "external-anchor raw/record-key policy is outside Stage 11",
            **context)
    if provenance.source_ref:
        if (GROUP_REF_SEP in provenance.source_ref
                or provenance.source_label
                or provenance.row_number != "none"):
            _not_replayable(
                "SOURCE_NOT_SINGLE_FILE",
                "external attribution requires one literal source and an "
                "explicitly absent row locator", **context)
        source = spec.source(provenance.source_ref)
        if source is None:
            _not_replayable(
                "SOURCE_LINK_MISSING",
                "external attribution names no declared source", **context)
        if not source.path or source.path_pattern:
            _not_replayable(
                "SOURCE_NOT_SINGLE_FILE",
                "external attribution requires one exact source path",
                **context)
        if (source.file_type != "csv" or source.reader != "cidc_csv@v1"
                or source.sheet):
            _not_replayable(
                "READER_UNSUPPORTED",
                "external attribution requires cidc_csv@v1 without a sheet",
                **context)
    elif provenance.row_number not in ("", "none"):
        _not_replayable(
            "SOURCE_NOT_SINGLE_FILE",
            "external provenance claims an undeclared source row", **context)
    if not is_external_anchor_evidence_mapping(spec, mapping):
        _not_replayable(
            "OPERATION_UNSUPPORTED",
            "mapping is outside the closed Stage 11 declaration shape",
            **context)
    expected_method = f"spec:{mapping.op}"
    if event.get("extraction_method") != expected_method:
        method = event.get("extraction_method", "")
        if method.startswith("post"):
            _not_replayable(
                "POST_RULE_UNSUPPORTED_STAGE1",
                "post-rule event is outside Stage 11", **context)
        if method == "value_suppressed":
            _not_replayable(
                "SUPPRESSION_UNSUPPORTED_STAGE1",
                "suppression event is outside Stage 11", **context)
        if method == "row_excluded":
            _not_replayable(
                "EXCLUSION_REQUIRES_ROW_REPLAY",
                "exclusion event is outside Stage 11", **context)
        _fail(
            "SOURCE_DECLARATION_MISMATCH",
            f"provenance extraction_method is not {expected_method}",
            **context)
    artifact_names = {artifact.artifact_id, Path(artifact.path).name}
    if mapping.artifacts and not artifact_names.intersection(mapping.artifacts):
        _request_error(
            "FIELD_NOT_APPLICABLE",
            "mapping does not apply to the selected artifact")


def _check_source_group_scope(
        mapping: SpecMapping, spec: HarmonizationSpec,
        event: dict[str, str], artifact: ArtifactBinding) -> None:
    """Enforce the exact closed Stage 12 group/basic-operation boundary."""
    context = {"spec_ref": spec.ref, "mapping_id": mapping.mapping_id}
    _require_static_present_state_policy(mapping, context)
    if not is_closed_source_group_mapping(spec, mapping):
        _not_replayable(
            "OPERATION_UNSUPPORTED",
            "mapping is outside the closed Stage 12 source-group boundary",
            **context)
    if not event.get("contributions"):
        _not_replayable(
            "SOURCE_GROUP_EVIDENCE_MISSING",
            "source-group Cell omits ordered membership evidence", **context)
    expected_method = f"spec:{mapping.op}"
    if event.get("extraction_method") != expected_method:
        if event.get("extraction_method", "").startswith("post"):
            _not_replayable(
                "POST_RULE_UNSUPPORTED_STAGE1",
                "post-rule provenance is outside Stage 12", **context)
        _fail(
            "SOURCE_DECLARATION_MISMATCH",
            f"provenance extraction_method is not {expected_method}",
            **context)
    if mapping.op != OP_FILE_MEMBERSHIP and event.get("derivation"):
        _not_replayable(
            "DERIVATION_UNSUPPORTED_STAGE2",
            "structured derivation is outside Stage 12", **context)
    artifact_names = {artifact.artifact_id, Path(artifact.path).name}
    if mapping.artifacts and not artifact_names.intersection(mapping.artifacts):
        _request_error(
            "FIELD_NOT_APPLICABLE",
            "mapping does not apply to the selected artifact")


def _check_stage15_conditional_scope(
        mapping: SpecMapping, spec: HarmonizationSpec,
        event: dict[str, str], artifact: ArtifactBinding) -> None:
    """Closed Stage 15 scalar-conditional declaration boundary."""
    context = {"spec_ref": spec.ref, "mapping_id": mapping.mapping_id}
    if mapping.op != OP_CONDITIONAL:
        _not_replayable(
            "OPERATION_UNSUPPORTED", "operation is not conditional", **context)
    if mapping.when:
        _not_replayable(
            "MAPPING_CONDITION_UNSUPPORTED",
            "mapping-level branch authority must be verified separately",
            **context)
    group_parent = GROUP_REF_SEP in mapping.source
    if (mapping.group or mapping.candidates or mapping.map
            or mapping.fallback_vocabulary
            or (mapping.column_by_member and not group_parent)):
        _not_replayable(
            "OPERATION_UNSUPPORTED",
            "conditional carries unsupported operation context", **context)
    if ((mapping.derivation or event.get("derivation"))
            and not (_is_stage17_derivation_mapping(mapping, spec)
                     or _is_stage18_derivation_mapping(mapping, spec))):
        _not_replayable(
            "DERIVATION_UNSUPPORTED_STAGE2",
            "Stage 15 does not broaden structured derivation", **context)
    _require_static_present_state_policy(mapping, context)
    if not mapping.clauses:
        _not_replayable(
            "OPERATION_UNSUPPORTED", "conditional has no clauses", **context)
    declared_reads = {read.id for read in spec.reads}
    for rank, clause in enumerate(mapping.clauses):
        predicate = clause.when
        if predicate.read:
            if (predicate.read not in declared_reads
                    or predicate.read not in mapping.contributions.conditions):
                _not_replayable(
                    "NAMED_READ_EVIDENCE_MISSING",
                    f"clause {rank} read predicate lacks Contribution evidence",
                    **context)
        elif (not mapping.source or predicate.source != mapping.source):
            _not_replayable(
                "NAMED_READ_EVIDENCE_MISSING",
                f"clause {rank} predicate is not bound to the parent source",
                **context)
        if clause.read and (
                clause.read not in declared_reads
                or clause.read not in mapping.contributions.conditions):
            _not_replayable(
                "NAMED_READ_EVIDENCE_MISSING",
                f"clause {rank} read yield lacks Contribution evidence",
                **context)
        if clause.column and (
                not mapping.source or clause.source != mapping.source):
            _not_replayable(
                "NAMED_READ_EVIDENCE_MISSING",
                f"clause {rank} column yield is not bound to parent source",
                **context)
    if group_parent:
        group = source_group_for_mapping(spec, mapping)
        if group is None or group.select != "first_containing_record":
            _not_replayable(
                "SOURCE_GROUP_EVIDENCE_MISSING",
                "conditional group parent is outside the Stage 12 selector",
                **context)
        _group_id, _, role = mapping.source.partition(GROUP_REF_SEP)
        for member in group.members:
            source = spec.source(member.sources.get(role, ""))
            if source is None or not source.path or source.path_pattern:
                _not_replayable(
                    "SOURCE_LINK_MISSING",
                    "conditional group-parent member lacks one exact source",
                    **context)
    elif mapping.source:
        source = spec.source(mapping.source)
        if source is None or not source.path or source.path_pattern:
            _not_replayable(
                "SOURCE_LINK_MISSING",
                "conditional parent source is not one exact file", **context)
    if event.get("extraction_method") != "spec:conditional":
        _fail(
            "SOURCE_DECLARATION_MISMATCH",
            "provenance extraction_method is not spec:conditional", **context)
    artifact_names = {artifact.artifact_id, Path(artifact.path).name}
    if mapping.artifacts and not artifact_names.intersection(mapping.artifacts):
        _request_error(
            "FIELD_NOT_APPLICABLE",
            "mapping does not apply to selected artifact")


def _check_operation_scope(mapping: SpecMapping, spec: HarmonizationSpec,
                           event: dict[str, str],
                           artifact: ArtifactBinding) -> None:
    context = {"spec_ref": spec.ref, "mapping_id": mapping.mapping_id}
    _require_static_present_state_policy(mapping, context)
    if mapping.op not in _SUPPORTED_OPERATIONS:
        _not_replayable("OPERATION_UNSUPPORTED",
                        f"operation {mapping.op!r} is outside Stage 2", **context)
    if mapping.when:
        _not_replayable("MAPPING_CONDITION_UNSUPPORTED",
                        "conditional scalar mappings are outside Stage 2", **context)
    if mapping.source == ANCHOR_SOURCE_ID:
        code = ("ANCHOR_SOURCE_UNSUPPORTED_STAGE1" if mapping.op == OP_DIRECT
                else "SOURCE_NOT_SINGLE_FILE")
        _not_replayable(
            code, "anchor-backed mappings are outside strict scalar replay",
            **context)
    if (not mapping.source or GROUP_REF_SEP in mapping.source
            or mapping.column_by_member
            or (mapping.provenance.source_ref
                and mapping.provenance.source_ref != mapping.source)
            or (mapping.provenance.row_from
                and mapping.provenance.row_from != mapping.source)):
        _not_replayable("SOURCE_NOT_SINGLE_FILE",
                        "mapping does not identify one scalar file source",
                        **context)
    if mapping.provenance.raw_from:
        code = ("DERIVATION_UNSUPPORTED_STAGE1" if mapping.op == OP_DIRECT
                else "DERIVATION_UNSUPPORTED_STAGE2")
        _not_replayable(
            code, "mapping raw evidence comes from a derived read", **context)
    if ((mapping.derivation or event["derivation"])
            and not _is_stage14_derivation_mapping(mapping)):
        code = ("DERIVATION_UNSUPPORTED_STAGE1" if mapping.op == OP_DIRECT
                else "DERIVATION_UNSUPPORTED_STAGE2")
        _not_replayable(code, "derivation evidence is outside scalar replay",
                        **context)
    if event["contributions"] or mapping.contributions.conditions:
        code = ("CONTRIBUTIONS_UNSUPPORTED_STAGE1"
                if mapping.op == OP_DIRECT
                else "CONTRIBUTIONS_UNSUPPORTED_STAGE2")
        _not_replayable(code, "contributions are outside scalar replay",
                        **context)
    if mapping.op in (OP_VALUE_MAP, OP_UNIT_CONVERSION) \
            and mapping.coerce != "none":
        _not_replayable("OPERATION_UNSUPPORTED",
                        "Stage 2 operations require coerce: none", **context)
    if mapping.op == OP_UNIT_CONVERSION:
        if (isinstance(mapping.factor, bool)
                or not isinstance(mapping.factor, (int, float))
                or not math.isfinite(float(mapping.factor))
                or float(mapping.factor) == 0.0
                or type(mapping.round) is not bool):
            _not_replayable(
                "UNIT_CONVERSION_PARAMETERS_INVALID",
                "unit conversion requires a finite non-zero factor and "
                "boolean round flag", **context)
    expected_method = f"spec:{mapping.op}"
    if event["extraction_method"] != expected_method:
        if event["extraction_method"].startswith("post"):
            _not_replayable("POST_RULE_UNSUPPORTED_STAGE1",
                            "post-rule provenance is outside scalar replay",
                            **context)
        _fail("SOURCE_DECLARATION_MISMATCH",
              f"provenance extraction_method is not {expected_method}",
              **context)
    artifact_names = {artifact.artifact_id, Path(artifact.path).name}
    if mapping.artifacts and not artifact_names.intersection(mapping.artifacts):
        _request_error("FIELD_NOT_APPLICABLE",
                       "mapping does not apply to the selected artifact")


def _check_stage15_source_declaration(
        root: Path, spec: HarmonizationSpec, source_decl,
        binding: SourceBinding, mapping_id: str, *,
        missing_reason: str = "NAMED_READ_EVIDENCE_MISSING") -> None:
    """Bind a Stage 15 read to one exact existing reader contract."""
    context = {"spec_ref": spec.ref, "mapping_id": mapping_id,
               "source_ref_id": binding.source_ref_id}
    if source_decl is None or not source_decl.path or source_decl.path_pattern:
        _not_replayable(
            missing_reason, "named-read source has no exact declaration",
            **context)
    if (binding.identity != f"source:{binding.source_ref_id}"
            or binding.spec_ref != spec.ref
            or binding.source_id != source_decl.id
            or binding.config_key != source_decl.id):
        _fail(
            "SOURCE_DECLARATION_MISMATCH",
            "named-read binding contradicts its declaration", **context)
    try:
        declared_path = resolve_project_path(root, source_decl.path)
        bound_path = resolve_project_path(root, binding.path)
    except ManifestError as exc:
        _not_replayable("MANIFEST_PATH_OUTSIDE_PROJECT", str(exc), **context)
    expected_label = source_decl.logical_label or declared_path.name
    if (declared_path != bound_path
            or binding.logical_label != expected_label
            or binding.file_type != source_decl.file_type
            or binding.sheet != (source_decl.sheet or "")):
        _fail(
            "SOURCE_DECLARATION_MISMATCH",
            "named-read path/type/label/sheet is contradictory", **context)
    supported = (
        (source_decl.file_type == "csv"
         and source_decl.reader == "cidc_csv@v1"
         and binding.reader.reader_id == "cidc_csv@v1")
        or
        (source_decl.file_type == "xlsx"
         and source_decl.reader == XLSX_READER_ID
         and binding.reader.reader_id == XLSX_READER_ID)
    )
    if not supported:
        _not_replayable(
            "READER_UNSUPPORTED",
            "named read is outside existing exact CSV/XLSX reader contracts",
            **context)


def _check_stage15_fallback_scope(
        mapping: SpecMapping, spec: HarmonizationSpec,
        event: dict[str, str], artifact: ArtifactBinding) -> None:
    """Closed Stage 15 gated-fallback eligibility."""
    context = {"spec_ref": spec.ref, "mapping_id": mapping.mapping_id}
    _require_static_present_state_policy(mapping, context)
    if mapping.op != OP_FALLBACK:
        _not_replayable(
            "OPERATION_UNSUPPORTED", "operation is not fallback", **context)
    if mapping.when:
        _not_replayable(
            "MAPPING_CONDITION_UNSUPPORTED",
            "mapping-level branch authority must be verified separately",
            **context)
    if (mapping.map or mapping.fallback_vocabulary
            or mapping.source or mapping.group or mapping.column_by_member
            or mapping.provenance.source_ref or mapping.provenance.row_from):
        _not_replayable(
            "FALLBACK_CANDIDATE_UNSUPPORTED",
            "gated fallback carries unsupported source/map context", **context)
    if ((mapping.derivation or event.get("derivation"))
            and not _is_stage17_derivation_mapping(mapping, spec)):
        _not_replayable(
            "DERIVATION_UNSUPPORTED_STAGE2",
            "Stage 15 does not broaden structured derivation", **context)
    if mapping.coerce not in ("none", "numeric"):
        _not_replayable(
            "FALLBACK_CANDIDATE_UNSUPPORTED",
            f"fallback coercion {mapping.coerce!r} is unsupported", **context)
    if mapping.contributions.record != "all":
        _not_replayable(
            "FALLBACK_EVIDENCE_INCOMPLETE",
            "strict gated fallback requires all Contributions", **context)
    declared_reads = {read.id for read in spec.reads}
    for rank, candidate in enumerate(mapping.candidates):
        if (not candidate.source or candidate.source == ANCHOR_SOURCE_ID
                or GROUP_REF_SEP in candidate.source):
            _not_replayable(
                "FALLBACK_CANDIDATE_UNSUPPORTED",
                f"candidate {rank} is not one exact file source", **context)
        source = spec.source(candidate.source)
        if source is None or not source.path or source.path_pattern:
            _not_replayable(
                "FALLBACK_CANDIDATE_UNSUPPORTED",
                f"candidate {rank} has no exact source", **context)
        gate = candidate.when
        if gate and gate.read and (
                gate.read not in declared_reads
                or gate.read not in mapping.contributions.conditions):
            _not_replayable(
                "NAMED_READ_EVIDENCE_MISSING",
                f"candidate {rank} read gate lacks declared Contribution evidence",
                **context)
    if event.get("extraction_method") != "spec:fallback":
        _fail(
            "SOURCE_DECLARATION_MISMATCH",
            "provenance extraction_method is not spec:fallback", **context)
    artifact_names = {artifact.artifact_id, Path(artifact.path).name}
    if mapping.artifacts and not artifact_names.intersection(mapping.artifacts):
        _request_error(
            "FIELD_NOT_APPLICABLE",
            "mapping does not apply to selected artifact")


def _check_fallback_scope(mapping: SpecMapping, spec: HarmonizationSpec,
                          event: dict[str, str],
                          artifact: ArtifactBinding) -> None:
    """Enforce the closed Stage 4 fallback subset without changing execution."""
    context = {"spec_ref": spec.ref, "mapping_id": mapping.mapping_id}
    _require_static_present_state_policy(mapping, context)
    if mapping.op != OP_FALLBACK:
        _not_replayable("OPERATION_UNSUPPORTED",
                        "operation is outside Stage 4 fallback replay", **context)
    if mapping.when:
        _not_replayable("MAPPING_CONDITION_UNSUPPORTED",
                        "conditional fallback mappings are outside Stage 4",
                        **context)
    if any(candidate.when is not None for candidate in mapping.candidates):
        _not_replayable("FALLBACK_CANDIDATE_UNSUPPORTED",
                        "gated fallback candidates are outside Stage 4",
                        **context)
    if mapping.map:
        _not_replayable("FALLBACK_CANDIDATE_UNSUPPORTED",
                        "inline fallback maps are outside Stage 4", **context)
    if mapping.fallback_vocabulary:
        _not_replayable("EXTERNAL_VOCABULARY_UNSUPPORTED",
                        "external vocabulary fallback is outside Stage 4",
                        **context)
    if mapping.coerce not in ("none", "numeric"):
        _not_replayable("FALLBACK_CANDIDATE_UNSUPPORTED",
                        f"fallback coercion {mapping.coerce!r} is unsupported",
                        **context)
    if (mapping.source or mapping.group or mapping.column_by_member
            or mapping.provenance.source_ref or mapping.provenance.row_from):
        _not_replayable("FALLBACK_CANDIDATE_UNSUPPORTED",
                        "fallback uses a non-candidate source or derived locator",
                        **context)
    if mapping.derivation or event.get("derivation"):
        _not_replayable("FALLBACK_CANDIDATE_UNSUPPORTED",
                        "derived fallback evidence is outside Stage 4", **context)
    if (mapping.contributions.record != "all"
            or mapping.contributions.after_decision != "not_evaluated"
            or mapping.contributions.conditions):
        _not_replayable(
            "FALLBACK_EVIDENCE_INCOMPLETE",
            "strict fallback replay requires contributions.record=all, no "
            "condition contributions, and after_decision=not_evaluated",
            **context)
    if len(mapping.candidates) < 2:
        _not_replayable("FALLBACK_CANDIDATE_UNSUPPORTED",
                        "fallback does not declare an ordered candidate chain",
                        **context)
    for rank, candidate in enumerate(mapping.candidates):
        if (not candidate.source or candidate.source == ANCHOR_SOURCE_ID
                or GROUP_REF_SEP in candidate.source):
            _not_replayable(
                "FALLBACK_CANDIDATE_UNSUPPORTED",
                f"candidate {rank} is not a raw file-backed source", **context)
        source = spec.source(candidate.source)
        if source is None or not source.path or source.path_pattern:
            _not_replayable(
                "FALLBACK_CANDIDATE_UNSUPPORTED",
                f"candidate {rank} does not bind one exact source file", **context)
        if source.file_type != "csv" or source.reader != "cidc_csv@v1" \
                or source.sheet:
            _not_replayable(
                "FALLBACK_CANDIDATE_UNSUPPORTED",
                f"candidate {rank} is outside the cidc_csv@v1 source contract",
                **context)
    if event.get("extraction_method") != "spec:fallback":
        if event.get("extraction_method", "").startswith("post"):
            _not_replayable("POST_RULE_UNSUPPORTED_STAGE1",
                            "post-rule provenance is outside Stage 4", **context)
        _fail("SOURCE_DECLARATION_MISMATCH",
              "provenance extraction_method is not spec:fallback", **context)
    artifact_names = {artifact.artifact_id, Path(artifact.path).name}
    if mapping.artifacts and not artifact_names.intersection(mapping.artifacts):
        _request_error("FIELD_NOT_APPLICABLE",
                       "mapping does not apply to the selected artifact")


def _is_closed_same_row_inline_fallback(mapping: SpecMapping) -> bool:
    """Recognize only the Stage 21 declaration shape before detailed checks."""
    source_ids = {candidate.source for candidate in mapping.candidates}
    return (
        mapping.op == OP_FALLBACK
        and mapping.contributions.record == "none"
        and bool(mapping.map)
        and len(mapping.candidates) >= 2
        and len(source_ids) == 1
        and all(candidate.when is None for candidate in mapping.candidates)
        and not mapping.when
        and not mapping.contributions.conditions
        and mapping.contributions.after_decision == "not_evaluated"
        and mapping.on_unmapped == "null"
        and mapping.coerce == "none"
        and not mapping.fallback_vocabulary
        and not mapping.source
        and not mapping.group
        and not mapping.column_by_member
        and not mapping.derivation
        and not mapping.provenance.source_ref
        and not mapping.provenance.row_from
        and not mapping.provenance.raw_from
        and not mapping.provenance.contributions_from
    )


def _check_same_row_inline_fallback_scope(
        mapping: SpecMapping, spec: HarmonizationSpec,
        event: dict[str, str], artifact: ArtifactBinding) -> None:
    """Enforce the closed Stage 21 same-source/same-row declaration subset."""
    context = {"spec_ref": spec.ref, "mapping_id": mapping.mapping_id}
    _require_static_present_state_policy(mapping, context)
    if not _is_closed_same_row_inline_fallback(mapping):
        _not_replayable(
            "FALLBACK_CANDIDATE_UNSUPPORTED",
            "fallback is outside the closed same-source/same-row inline-map "
            "route", **context)
    source_id = mapping.candidates[0].source
    if (not source_id or source_id == ANCHOR_SOURCE_ID
            or GROUP_REF_SEP in source_id):
        _not_replayable(
            "FALLBACK_CANDIDATE_UNSUPPORTED",
            "same-row fallback source is not a raw file-backed source",
            **context)
    source = spec.source(source_id)
    if source is None or not source.path or source.path_pattern:
        _not_replayable(
            "FALLBACK_CANDIDATE_UNSUPPORTED",
            "same-row fallback does not bind one exact source file", **context)
    if (source.file_type != "csv" or source.reader != "cidc_csv@v1"
            or source.sheet):
        _not_replayable(
            "FALLBACK_CANDIDATE_UNSUPPORTED",
            "same-row fallback is outside the cidc_csv@v1 source contract",
            **context)
    if event.get("derivation"):
        _not_replayable(
            "FALLBACK_CANDIDATE_UNSUPPORTED",
            "derived fallback evidence is outside the closed same-row route",
            **context)
    if event.get("extraction_method") != "spec:fallback":
        if event.get("extraction_method", "").startswith("post"):
            _not_replayable(
                "POST_RULE_UNSUPPORTED_STAGE1",
                "post-rule provenance is outside same-row fallback replay",
                **context)
        _fail(
            "SOURCE_DECLARATION_MISMATCH",
            "provenance extraction_method is not spec:fallback", **context)
    artifact_names = {artifact.artifact_id, Path(artifact.path).name}
    if mapping.artifacts and not artifact_names.intersection(mapping.artifacts):
        _request_error(
            "FIELD_NOT_APPLICABLE",
            "mapping does not apply to the selected artifact")


def _check_same_row_fallback_source_declaration(
        root: Path, spec: HarmonizationSpec, mapping: SpecMapping,
        event: dict[str, str], source_decl,
        binding: SourceBinding) -> None:
    """Bind a fallback parent Cell to one exact declared CSV artifact."""
    context = {"spec_ref": spec.ref, "mapping_id": mapping.mapping_id,
               "source_ref_id": binding.source_ref_id}
    if source_decl is None:
        _not_replayable(
            "SOURCE_LINK_MISSING",
            "same-row fallback source is not declared", **context)
    if (binding.identity != f"source:{binding.source_ref_id}"
            or binding.spec_ref != spec.ref
            or binding.source_id != source_decl.id
            or binding.config_key != source_decl.id):
        _fail(
            "SOURCE_DECLARATION_MISMATCH",
            "same-row fallback binding disagrees with its declaration",
            **context)
    try:
        declared_path = resolve_project_path(root, source_decl.path)
        bound_path = resolve_project_path(root, binding.path)
    except ManifestError as exc:
        _not_replayable("MANIFEST_PATH_OUTSIDE_PROJECT", str(exc), **context)
    expected_label = source_decl.logical_label or declared_path.name
    if (declared_path != bound_path
            or binding.logical_label != expected_label
            or binding.file_type != source_decl.file_type
            or binding.file_type != "csv"
            or binding.sheet or source_decl.sheet
            or binding.reader.reader_id != source_decl.reader
            or source_decl.reader != "cidc_csv@v1"):
        _fail(
            "SOURCE_DECLARATION_MISMATCH",
            "same-row fallback path/type/label/sheet/reader contradicts the "
            "pinned declaration", **context)
    if (event.get("source_resolved_path") != binding.path
            or event.get("source_sha256") != binding.sha256
            or event.get("source_sheet") != binding.sheet
            or event.get("source_logical_label") != binding.logical_label
            or event.get("source_file") != binding.logical_label):
        _fail(
            "SOURCE_DECLARATION_MISMATCH",
            "fallback Cell source locator disagrees with pinned source "
            "evidence", **context)


def _check_fallback_contribution_context(
        mapping: SpecMapping, candidate, contribution: dict[str, Any],
        binding: SourceBinding, rank: int, spec_ref: str) -> None:
    context = {"spec_ref": spec_ref, "mapping_id": mapping.mapping_id,
               "source_ref_id": binding.source_ref_id}
    if (contribution.get("rank") != rank
            or contribution.get("role") != f"{mapping.target}_candidate"
            or contribution.get("source_column") != candidate.column
            or contribution.get("source_ref_id") != binding.source_ref_id
            or contribution.get("rule_id") != mapping.rule.id
            or contribution.get("provenance_ref")):
        _fail("FALLBACK_CANDIDATE_CONTEXT_MISMATCH",
              f"fallback candidate {rank} identity/order disagrees with spec",
              **context)
    if contribution.get("status") not in {
            STATUS_SELECTED, STATUS_UNAVAILABLE, STATUS_REJECTED, STATUS_NOT_EVALUATED}:
        _fail("FALLBACK_SELECTION_MISMATCH",
              f"fallback candidate {rank} has unsupported status "
              f"{contribution.get('status')!r}", **context)


def _check_fallback_source_declaration(
        root: Path, spec: HarmonizationSpec, source_decl,
        binding: SourceBinding, contribution: dict[str, Any], rank: int,
        mapping_id: str) -> None:
    context = {"spec_ref": spec.ref, "mapping_id": mapping_id,
               "source_ref_id": binding.source_ref_id}
    if source_decl is None:
        _not_replayable("SOURCE_LINK_MISSING",
                        f"candidate {rank} source is not declared", **context)
    if (binding.identity != f"source:{binding.source_ref_id}"
            or binding.spec_ref != spec.ref
            or binding.source_id != source_decl.id
            or binding.config_key != source_decl.id):
        _fail("SOURCE_DECLARATION_MISMATCH",
              f"candidate {rank} binding disagrees with its source declaration",
              **context)
    try:
        declared_path = resolve_project_path(root, source_decl.path)
        bound_path = resolve_project_path(root, binding.path)
    except ManifestError as exc:
        _not_replayable("MANIFEST_PATH_OUTSIDE_PROJECT", str(exc), **context)
    expected_label = source_decl.logical_label or declared_path.name
    if (declared_path != bound_path
            or binding.logical_label != expected_label
            or binding.file_type != source_decl.file_type
            or binding.file_type != "csv"
            or binding.sheet or source_decl.sheet):
        _fail("SOURCE_DECLARATION_MISMATCH",
              f"candidate {rank} path/type/label/sheet is contradictory",
              **context)
    if (contribution.get("source_resolved_path") != binding.path
            or contribution.get("source_sha256") != binding.sha256
            or str(contribution.get("source_sheet") or "") != binding.sheet
            or contribution.get("source_logical_label") != binding.logical_label):
        _fail("FALLBACK_CANDIDATE_CONTEXT_MISMATCH",
              f"candidate {rank} contribution source identity is contradictory",
              **context)


def _fallback_row_number(contribution: dict[str, Any], spec_ref: str,
                         mapping_id: str, source_ref_id: str, rank: int) -> int:
    context = {"spec_ref": spec_ref, "mapping_id": mapping_id,
               "source_ref_id": source_ref_id}
    number = contribution.get("source_row_number")
    if isinstance(number, bool) or not isinstance(number, int) or number < 0:
        _not_replayable(
            "SOURCE_LOCATOR_INCOMPLETE",
            f"candidate {rank} source_row_number is not a non-negative integer",
            **context)
    return number


def _check_source_group_source_declaration(
        root: Path, spec: HarmonizationSpec, source_decl,
        binding: SourceBinding, mapping_id: str) -> None:
    """Bind one group role to its exact pinned CSV declaration."""
    context = {"spec_ref": spec.ref, "mapping_id": mapping_id,
               "source_ref_id": binding.source_ref_id}
    if source_decl is None:
        _not_replayable(
            "SOURCE_GROUP_EVIDENCE_MISSING",
            "source-group member has no pinned source declaration", **context)
    if (binding.identity != f"source:{binding.source_ref_id}"
            or binding.spec_ref != spec.ref
            or binding.source_id != source_decl.id
            or binding.config_key != source_decl.id):
        _fail(
            "SOURCE_DECLARATION_MISMATCH",
            "group SourceBinding disagrees with its pinned declaration",
            **context)
    if not source_decl.path or source_decl.path_pattern:
        _not_replayable(
            "SOURCE_GROUP_EVIDENCE_MISSING",
            "source-group member does not bind one exact path", **context)
    try:
        declared_path = resolve_project_path(root, source_decl.path)
        bound_path = resolve_project_path(root, binding.path)
    except ManifestError as exc:
        _not_replayable(
            "MANIFEST_PATH_OUTSIDE_PROJECT", str(exc), **context)
    expected_label = source_decl.logical_label or declared_path.name
    if (declared_path != bound_path
            or binding.logical_label != expected_label
            or binding.file_type != source_decl.file_type
            or binding.file_type != "csv"
            or binding.sheet or source_decl.sheet
            or binding.reader.reader_id != source_decl.reader
            or source_decl.reader != "cidc_csv@v1"):
        _fail(
            "SOURCE_DECLARATION_MISMATCH",
            "group source path/type/label/sheet/reader contradicts the pinned "
            "declaration", **context)


def _check_source_declaration(root: Path, spec: HarmonizationSpec,
                              mapping: SpecMapping, event: dict[str, str],
                              source_decl, binding: SourceBinding) -> None:
    context = {"spec_ref": spec.ref, "mapping_id": mapping.mapping_id,
               "source_ref_id": binding.source_ref_id}
    if source_decl is None:
        _not_replayable("SOURCE_LINK_MISSING",
                        "mapping source is not declared by the specification",
                        **context)
    if (binding.identity != f"source:{binding.source_ref_id}"
            or binding.spec_ref != spec.ref
            or binding.source_id != source_decl.id
            or binding.config_key != source_decl.id):
        _fail("SOURCE_DECLARATION_MISMATCH",
              "manifest source binding disagrees with the specification",
              **context)
    if not source_decl.path or source_decl.path_pattern:
        _not_replayable("SOURCE_NOT_SINGLE_FILE",
                        "strict scalar replay requires one exact source path",
                        **context)
    try:
        declared_path = resolve_project_path(root, source_decl.path)
        bound_path = resolve_project_path(root, binding.path)
    except ManifestError as exc:
        _not_replayable("MANIFEST_PATH_OUTSIDE_PROJECT", str(exc), **context)
    expected_label = source_decl.logical_label or declared_path.name
    csv_shape = (binding.file_type == "csv"
                 and not binding.sheet and not source_decl.sheet)
    xlsx_shape = (binding.file_type == "xlsx"
                  and bool(binding.sheet)
                  and binding.sheet == source_decl.sheet
                  and declared_path.suffix.lower() == ".xlsx")
    if (declared_path != bound_path
            or binding.logical_label != expected_label
            or binding.file_type != source_decl.file_type
            or binding.reader.reader_id != source_decl.reader
            or not (csv_shape or xlsx_shape)):
        _fail("SOURCE_DECLARATION_MISMATCH",
              "source path/type/label/sheet/reader disagrees across spec "
              "and manifest", **context)
    if (event["source_resolved_path"] != binding.path
            or event["source_sha256"] != binding.sha256
            or event["source_sheet"] != binding.sheet
            or event["source_logical_label"] != binding.logical_label
            or event["source_file"] != binding.logical_label
            or event["source_column"] != mapping.column):
        _fail("SOURCE_DECLARATION_MISMATCH",
              "provenance source locator disagrees with pinned source evidence",
              **context)


def _check_no_row_attributed_source_declaration(
        root: Path, spec: HarmonizationSpec, mapping: SpecMapping,
        source_decl, binding: SourceBinding) -> None:
    """Bind a no-row attribution to one exact pinned file declaration."""
    context = {"spec_ref": spec.ref, "mapping_id": mapping.mapping_id,
               "source_ref_id": binding.source_ref_id}
    source_id = mapping.provenance.source_ref
    if source_decl is None:
        _not_replayable(
            "SOURCE_LINK_MISSING",
            "attributed mapping names no declared source", **context)
    if (binding.identity != f"source:{binding.source_ref_id}"
            or binding.spec_ref != spec.ref
            or binding.source_id != source_id
            or binding.config_key != source_id):
        _fail(
            "SOURCE_DECLARATION_MISMATCH",
            "manifest source binding disagrees with attributed declaration",
            **context)
    if not source_decl.path or source_decl.path_pattern:
        _not_replayable(
            "SOURCE_NOT_SINGLE_FILE",
            "attributed declared absence requires one exact source path",
            **context)
    try:
        declared_path = resolve_project_path(root, source_decl.path)
        bound_path = resolve_project_path(root, binding.path)
    except ManifestError as exc:
        _not_replayable("MANIFEST_PATH_OUTSIDE_PROJECT", str(exc), **context)
    expected_label = source_decl.logical_label or declared_path.name
    if (declared_path != bound_path
            or binding.logical_label != expected_label
            or binding.file_type != source_decl.file_type
            or binding.file_type != "csv"
            or binding.sheet or source_decl.sheet
            or binding.reader.reader_id != source_decl.reader):
        _fail(
            "SOURCE_DECLARATION_MISMATCH",
            "source path/type/label/sheet/reader contradicts the pinned "
            "attribution", **context)


def _check_reader_contract(
        binding: SourceBinding, source_decl, manifest: ReplayManifest,
        mapping_id: str) -> None:
    context = {"spec_ref": binding.spec_ref, "mapping_id": mapping_id,
               "source_ref_id": binding.source_ref_id}
    reader = binding.reader
    if reader.reader_id == XLSX_READER_ID:
        _check_xlsx_reader_contract(binding, source_decl, manifest, mapping_id)
        return
    if not reader.reader_id or not reader.effective_flags or not reader.dependencies:
        _not_replayable("READER_METADATA_MISSING",
                        "reader ID, flags, and dependencies are required",
                        **context)
    if reader.reader_id != "cidc_csv@v1" or source_decl.reader != reader.reader_id:
        _not_replayable("READER_UNSUPPORTED",
                        "strict scalar replay supports only cidc_csv@v1",
                        **context)
    flags = reader.flags
    if set(flags) != _CIDC_READER_FLAGS:
        _not_replayable("READER_FLAGS_UNSUPPORTED",
                        "cidc_csv@v1 effective flags are incomplete or extended",
                        **context)
    valid = (
        flags["encoding"] in _CIDC_ENCODINGS
        and type(flags["header"]) is int and flags["header"] == 0
        and type(flags["low_memory"]) is bool and flags["low_memory"] is False
        and type(flags["skiprows"]) is int and flags["skiprows"] in (0, 1)
    )
    if not valid:
        _not_replayable("READER_FLAGS_UNSUPPORTED",
                        "cidc_csv@v1 effective flags are outside the v1 contract",
                        **context)
    dependencies = dict(reader.dependencies)
    if set(dependencies) != {"pandas"}:
        _not_replayable("READER_METADATA_MISSING",
                        "cidc_csv@v1 requires exactly the pandas dependency",
                        **context)
    if dependencies["pandas"] != pd.__version__:
        _not_replayable("READER_ENVIRONMENT_MISMATCH",
                        f"pandas {pd.__version__} does not match pinned "
                        f"{dependencies['pandas']}", **context)


def _check_xlsx_reader_contract(
        binding: SourceBinding, source_decl, manifest: ReplayManifest,
        mapping_id: str) -> None:
    """Verify the complete closed XLSX reader and environment identity."""
    context = {
        "spec_ref": binding.spec_ref,
        "mapping_id": mapping_id,
        "source_ref_id": binding.source_ref_id,
    }
    reader = binding.reader
    if (manifest.manifest_version < 2 or not reader.effective_flags
            or not reader.dependencies or not binding.parsed_columns
            or not binding.parsed_canonicalization
            or not binding.parsed_sheet_sha256):
        _not_replayable(
            "READER_METADATA_MISSING",
            "strict XLSX replay requires manifest-v2 reader and parsed-sheet metadata",
            **context)
    if source_decl.reader != XLSX_READER_ID:
        _fail("SOURCE_DECLARATION_MISMATCH",
              "XLSX declaration uses a different reader", **context)
    if reader.flags != XLSX_READER_FLAGS:
        _fail("READER_OBSERVATION_MISMATCH",
              f"XLSX effective flags contradict {XLSX_READER_ID}", **context)
    dependencies = dict(reader.dependencies)
    if set(dependencies) != set(XLSX_READER_DEPENDENCIES):
        _not_replayable(
            "READER_METADATA_MISSING",
            f"{XLSX_READER_ID} requires exactly pandas, openpyxl, and NumPy",
            **context)
    producer_dependencies = dict(manifest.producer.dependencies)
    required = set(XLSX_READER_DEPENDENCIES)
    if not required.issubset(producer_dependencies):
        _not_replayable(
            "READER_METADATA_MISSING",
            "producer environment omits XLSX reader dependencies", **context)
    if any(producer_dependencies[name] != dependencies[name]
           for name in required):
        _fail("READER_OBSERVATION_MISMATCH",
              "XLSX reader dependencies disagree with producer identity",
              **context)
    observed = {
        "python": platform.python_version(),
        "pandas": pd.__version__,
        "openpyxl": openpyxl.__version__,
        "numpy": np.__version__,
    }
    expected = {
        "python": manifest.producer.python_version,
        "pandas": dependencies["pandas"],
        "openpyxl": dependencies["openpyxl"],
        "numpy": dependencies["numpy"],
    }
    mismatches = [
        f"{name} observed {observed[name]!r}, pinned {expected[name]!r}"
        for name in ("python", "pandas", "openpyxl", "numpy")
        if observed[name] != expected[name]
    ]
    if mismatches:
        _fail("READER_ENVIRONMENT_MISMATCH", "; ".join(mismatches), **context)


def _check_external_anchor_reader_contract(
        binding: SourceBinding, manifest: ReplayManifest,
        mapping_id: str) -> None:
    """Verify the complete external_anchor_csv@v1 reader observation."""
    context = {
        "spec_ref": binding.spec_ref,
        "mapping_id": mapping_id,
        "source_ref_id": binding.source_ref_id,
    }
    reader = binding.reader
    if (not reader.reader_id or not reader.effective_flags
            or not reader.dependencies):
        _not_replayable(
            "READER_METADATA_MISSING",
            "external anchor reader ID, flags, and dependencies are required",
            **context)
    if reader.reader_id != EXTERNAL_ANCHOR_READER_ID:
        _fail(
            "READER_OBSERVATION_MISMATCH",
            "external anchor binding uses the wrong reader contract",
            **context)
    if reader.flags != EXTERNAL_ANCHOR_READER_FLAGS:
        _fail(
            "READER_OBSERVATION_MISMATCH",
            "external anchor effective flags contradict v1", **context)
    dependencies = dict(reader.dependencies)
    if set(dependencies) != {"numpy", "pandas"}:
        _not_replayable(
            "READER_METADATA_MISSING",
            "external_anchor_csv@v1 requires exactly pandas and NumPy",
            **context)
    producer_dependencies = dict(manifest.producer.dependencies)
    if not {"numpy", "pandas"}.issubset(producer_dependencies):
        _not_replayable(
            "MANIFEST_REQUIRED_FIELD_MISSING",
            "producer environment omits external reader dependencies",
            **context)
    expected = {
        "python": manifest.producer.python_version,
        "pandas": dependencies["pandas"],
        "numpy": dependencies["numpy"],
    }
    observed = {
        "python": platform.python_version(),
        "pandas": pd.__version__,
        "numpy": np.__version__,
    }
    if (producer_dependencies["pandas"] != expected["pandas"]
            or producer_dependencies["numpy"] != expected["numpy"]):
        _fail(
            "READER_OBSERVATION_MISMATCH",
            "external reader dependencies disagree with producer identity",
            **context)
    mismatches = [
        f"{name} observed {observed[name]!r}, pinned {expected[name]!r}"
        for name in ("python", "pandas", "numpy")
        if observed[name] != expected[name]
    ]
    if mismatches:
        _fail("READER_ENVIRONMENT_MISMATCH", "; ".join(mismatches), **context)


def _check_stage2_environment(manifest: ReplayManifest, spec_ref: str,
                              mapping_id: str) -> None:
    """Require the exact producer runtime used by Stage 2 scalar semantics."""
    context = {"spec_ref": spec_ref, "mapping_id": mapping_id}
    dependencies = dict(manifest.producer.dependencies)
    required = {"pandas", "yaml"}
    missing = sorted(required - set(dependencies))
    if missing:
        _not_replayable(
            "MANIFEST_REQUIRED_FIELD_MISSING",
            f"producer environment omits required dependency versions {missing}",
            **context)
    observed = {
        "python": platform.python_version(),
        "pandas": pd.__version__,
        "yaml": str(getattr(yaml, "__version__", "")),
    }
    expected = {
        "python": manifest.producer.python_version,
        "pandas": dependencies["pandas"],
        "yaml": dependencies["yaml"],
    }
    mismatches = [
        f"{name} observed {observed[name]!r}, pinned {expected[name]!r}"
        for name in ("python", "pandas", "yaml")
        if observed[name] != expected[name]
    ]
    if mismatches:
        _not_replayable(
            "PRODUCER_ENVIRONMENT_MISMATCH", "; ".join(mismatches),
            **context)


def _check_yaml_declaration_environment(
        manifest: ReplayManifest, spec_ref: str, mapping_id: str) -> None:
    """Pin only the runtime identities involved in YAML declaration parsing."""
    context = {"spec_ref": spec_ref, "mapping_id": mapping_id}
    dependencies = dict(manifest.producer.dependencies)
    if not dependencies.get("yaml"):
        _not_replayable(
            "MANIFEST_REQUIRED_FIELD_MISSING",
            "producer environment omits the PyYAML version", **context)
    observed = {
        "python": platform.python_version(),
        "yaml": str(getattr(yaml, "__version__", "")),
    }
    expected = {
        "python": manifest.producer.python_version,
        "yaml": dependencies["yaml"],
    }
    mismatches = [
        f"{name} observed {observed[name]!r}, pinned {expected[name]!r}"
        for name in ("python", "yaml") if observed[name] != expected[name]
    ]
    if mismatches:
        _not_replayable(
            "PRODUCER_ENVIRONMENT_MISMATCH", "; ".join(mismatches),
            **context)


def _constant_record_key(mapping: SpecMapping, spec: HarmonizationSpec,
                         delivered_row: Mapping[str, str]) -> str:
    """Reproduce the executor's source-free constant record-key policy."""
    provenance = mapping.provenance
    if provenance.record_key:
        return provenance.record_key
    if provenance.record_key_from == "participant":
        participant = spec.record.participant_key
        key_name = spec.record.record_key_column or participant
        if not participant or participant not in delivered_row or not key_name:
            _fail(
                "SOURCE_DECLARATION_MISMATCH",
                "constant participant record key is not representable",
                spec_ref=spec.ref, mapping_id=mapping.mapping_id)
        return f"{key_name}={delivered_row[participant]}"
    return ""


def _declared_absence_record_key(
        mapping: SpecMapping, spec: HarmonizationSpec,
        delivered_row: Mapping[str, str]) -> str:
    """Reproduce the executor's declared-absence record-key policy."""
    provenance = mapping.provenance
    if provenance.record_key:
        return provenance.record_key
    if provenance.record_key_from == "group":
        grouping = spec.record.grouping
        if grouping is None:
            _not_replayable(
                "RECORD_KEY_UNPARSABLE",
                "declared-absence group key has no grouping declaration",
                spec_ref=spec.ref, mapping_id=mapping.mapping_id)
        fields_by_column: dict[str, list[str]] = {}
        for field_name, source_column in grouping.anchor_fields.items():
            fields_by_column.setdefault(source_column, []).append(field_name)
        parts = []
        for source_column in grouping.key_columns:
            fields = fields_by_column.get(source_column, [])
            if (len(fields) != 1 or fields[0] not in delivered_row):
                _not_replayable(
                    "RECORD_KEY_UNPARSABLE",
                    "declared-absence grouped key is not representable from "
                    "the delivered record", spec_ref=spec.ref,
                    mapping_id=mapping.mapping_id)
            parts.append(f"{source_column}={delivered_row[fields[0]]}")
        return ";".join(parts)
    participant = spec.record.participant_key
    key_name = spec.record.record_key_column or participant
    if not participant or participant not in delivered_row or not key_name:
        _fail(
            "SOURCE_DECLARATION_MISMATCH",
            "declared-absence participant record key is not representable",
            spec_ref=spec.ref, mapping_id=mapping.mapping_id)
    return f"{key_name}={delivered_row[participant]}"


def _xlsx_source_row_number(
        frame: pd.DataFrame, spec: HarmonizationSpec, source_decl,
        delivered_row: Mapping[str, str], mapping_id: str,
        source_ref_id: str) -> int:
    """Rebuild executor participant lookup and retain the pre-dedup row."""
    context = {"spec_ref": spec.ref, "mapping_id": mapping_id,
               "source_ref_id": source_ref_id}
    participant_key = spec.record.participant_key
    if not participant_key or participant_key not in delivered_row:
        _fail("SOURCE_RECORD_KEY_MISMATCH",
              "delivered record lacks the declared participant identity",
              **context)
    join_column = source_decl.participant_column or participant_key
    if join_column not in frame.columns:
        _fail("SOURCE_COLUMN_MISSING",
              f"participant column {join_column!r} is absent", **context)
    indexed = frame.copy(deep=True)
    locator = "__stage13_source_row__"
    if locator in indexed.columns:
        _fail("READER_OBSERVATION_MISMATCH",
              f"XLSX source collides with internal column {locator!r}", **context)
    indexed[locator] = range(len(indexed))
    indexed[join_column] = indexed[join_column].astype(str)
    duplicate = spec.record.duplicates
    by = [join_column if name == participant_key else name
          for name in duplicate.by] or [join_column]
    missing = [name for name in by if name not in indexed.columns]
    if missing:
        _fail("SOURCE_COLUMN_MISSING",
              f"duplicate-policy columns are absent: {missing}", **context)
    if duplicate.policy in {"first", "last"}:
        indexed = indexed.drop_duplicates(subset=by, keep=duplicate.policy)
    elif duplicate.policy == "error":
        if indexed.duplicated(subset=by).any():
            _fail("READER_OBSERVATION_MISMATCH",
                  "XLSX now violates the declared duplicate-error policy",
                  **context)
    else:
        _not_replayable("OPERATION_UNSUPPORTED",
                        f"duplicate policy {duplicate.policy!r} is unsupported",
                        **context)
    indexed = indexed.set_index(join_column)
    participant = str(delivered_row[participant_key])
    if participant not in indexed.index:
        _fail("SOURCE_RECORD_KEY_MISMATCH",
              "participant is absent from the reconstructed XLSX index", **context)
    selected = indexed.loc[participant]
    if isinstance(selected, pd.DataFrame):
        _fail("READER_OBSERVATION_MISMATCH",
              "participant lookup is not unique after duplicate policy", **context)
    return int(selected[locator])


def _source_row_number(event: dict[str, str], spec_ref: str, mapping_id: str,
                       source_ref_id: str) -> int:
    text = event["source_row_number"]
    if not text:
        _not_replayable("SOURCE_LOCATOR_INCOMPLETE",
                        "source_row_number is absent", spec_ref=spec_ref,
                        mapping_id=mapping_id, source_ref_id=source_ref_id)
    try:
        number = int(text)
    except ValueError:
        _not_replayable("SOURCE_LOCATOR_INCOMPLETE",
                        "source_row_number is not an integer", spec_ref=spec_ref,
                        mapping_id=mapping_id, source_ref_id=source_ref_id)
    if number < 0 or str(number) != text:
        _not_replayable("SOURCE_LOCATOR_INCOMPLETE",
                        "source_row_number is not a canonical non-negative integer",
                        spec_ref=spec_ref, mapping_id=mapping_id,
                        source_ref_id=source_ref_id)
    return number


def _verify_record_key(payload: str, spec: HarmonizationSpec, source_decl,
                       source_row, delivered_row, spec_ref: str,
                       mapping_id: str, source_ref_id: str) -> None:
    context = {"spec_ref": spec_ref, "mapping_id": mapping_id,
               "source_ref_id": source_ref_id}
    if not payload:
        _not_replayable("SOURCE_LOCATOR_INCOMPLETE",
                        "source_record_key is absent", **context)
    if ";" in payload or "=" not in payload:
        _not_replayable("RECORD_KEY_UNPARSABLE",
                        "Stage 1 requires one name=value source record key",
                        **context)
    name, value = payload.split("=", 1)
    expected_name = spec.record.record_key_column or spec.record.participant_key
    source_column = source_decl.participant_column or spec.record.participant_key
    if not name or name != expected_name or not source_column:
        _fail("SOURCE_RECORD_KEY_MISMATCH",
              "source record-key name disagrees with the specification",
              **context)
    if source_column not in source_row.index:
        _fail("SOURCE_COLUMN_MISSING",
              f"record-key column {source_column!r} is absent", **context)
    observed = _source_text(source_row[source_column])
    delivered = delivered_row.get(spec.record.participant_key)
    if observed != value or delivered != value:
        _fail("SOURCE_RECORD_KEY_MISMATCH",
              "source record key disagrees with source or delivered identity",
              **context)


def _verify_declared_record_keys(source_decl, source_row, delivered_row,
                                 spec_ref: str, mapping_id: str,
                                 source_ref_id: str) -> None:
    context = {"spec_ref": spec_ref, "mapping_id": mapping_id,
               "source_ref_id": source_ref_id}
    for name in source_decl.record_keys:
        if name not in source_row.index:
            _fail("SOURCE_COLUMN_MISSING",
                  f"declared record-key column {name!r} is absent", **context)
        if (name in delivered_row
                and _source_text(source_row[name]) != delivered_row[name]):
            _fail("SOURCE_RECORD_KEY_MISMATCH",
                  f"declared record-key column {name!r} disagrees with delivery",
                  **context)


def _verify_unique_same_row_locator(
        payload: str, frame, spec: HarmonizationSpec, source_decl,
        spec_ref: str, mapping_id: str, source_ref_id: str) -> None:
    """Require the parent record identity to select one historical row."""
    context = {"spec_ref": spec_ref, "mapping_id": mapping_id,
               "source_ref_id": source_ref_id}
    if not payload or ";" in payload or "=" not in payload:
        return  # The preceding _verify_record_key reports the exact reason.
    _name, value = payload.split("=", 1)
    source_column = (
        source_decl.participant_column or spec.record.participant_key)
    if source_column not in frame.columns:
        _fail(
            "SOURCE_COLUMN_MISSING",
            f"record-key column {source_column!r} is absent", **context)
    matches = sum(
        _source_text(candidate) == value for candidate in frame[source_column])
    if matches != 1:
        _fail(
            "SOURCE_RECORD_KEY_MISMATCH",
            f"same-row fallback record identity resolves to {matches} rows; "
            "exactly one is required", **context)


def _verify_rule(event: dict[str, str], expected: RuleRef, usage,
                 manifest: ReplayManifest, operation: str, spec_ref: str,
                 mapping_id: str, source_ref_id: str) -> None:
    expected_ref = f"{expected.id}@{expected.version}"
    context = {"spec_ref": spec_ref, "mapping_id": mapping_id,
               "source_ref_id": source_ref_id}
    if not event["rule_id"] or not event["rule_version"]:
        _not_replayable("MANIFEST_REQUIRED_FIELD_MISSING",
                        "authoritative provenance rule identity is absent",
                        **context)
    # Compare the independently selected branch before consulting the
    # manifest's record of the emitted rule.  A producer that emitted a wrong
    # but otherwise declared branch rule is contradictory evidence, not merely
    # incomplete manifest context.
    if event["rule_id"] != expected.id or event["rule_version"] != expected.version:
        _fail("RULE_IDENTITY_MISMATCH",
              f"provenance rule identity differs from the selected {operation} "
              "outcome",
              **context)
    if expected_ref not in usage.rule_refs:
        _not_replayable("MANIFEST_REQUIRED_FIELD_MISSING",
                        f"mapping usage omits required rule {expected_ref!r}",
                        **context)
    pinned_rules = [rule for rule in manifest.rules if rule.ref == expected_ref]
    if not pinned_rules:
        _not_replayable("MANIFEST_REQUIRED_FIELD_MISSING",
                        f"manifest omits required rule {expected_ref!r}",
                        **context)
    if pinned_rules[0].kind != "mapping":
        _fail("RULE_IDENTITY_MISMATCH",
              "pinned scalar rule is not identified as a mapping rule",
              **context)
    recorded_evidence = tuple(
        item for item in event["evidence_refs"].split(";") if item)
    if set(recorded_evidence) != set(usage.evidence_ref_ids):
        if not recorded_evidence:
            _not_replayable("MANIFEST_REQUIRED_FIELD_MISSING",
                            "provenance omits mapping-scoped evidence links",
                            **context)
        _fail("SOURCE_DECLARATION_MISMATCH",
              "provenance evidence links disagree with mapping usage", **context)


def _verify_confidence(event: dict[str, str], expected_raw: Any,
                       operation: str, spec_ref: str, mapping_id: str,
                       source_ref_id: str) -> None:
    """Verify confidence selected independently by a recomputed outcome."""
    context = {"spec_ref": spec_ref, "mapping_id": mapping_id,
               "source_ref_id": source_ref_id}
    text = event.get("confidence", "")
    if text == "":
        _not_replayable(
            "MANIFEST_REQUIRED_FIELD_MISSING",
            "authoritative provenance confidence is absent", **context)
    try:
        observed = Decimal(text)
    except InvalidOperation:
        _fail("CONFIDENCE_CONTEXT_MISMATCH",
              "provenance confidence is not numeric", **context)
    if not observed.is_finite() or observed < 0 or observed > 1:
        _fail("CONFIDENCE_CONTEXT_MISMATCH",
              "provenance confidence is not finite in [0,1]", **context)

    if expected_raw is None or isinstance(expected_raw, str):
        _not_replayable(
            "MANIFEST_REQUIRED_FIELD_MISSING",
            f"pinned {operation} mapping does not declare numeric confidence "
            "for the recomputed outcome", **context)
    try:
        expected = Decimal(str(expected_raw))
    except InvalidOperation:
        _not_replayable(
            "MANIFEST_REQUIRED_FIELD_MISSING",
            f"pinned {operation} confidence is not numeric", **context)
    if not expected.is_finite() or expected < 0 or expected > 1:
        _not_replayable(
            "MANIFEST_REQUIRED_FIELD_MISSING",
            f"pinned {operation} confidence is not finite in [0,1]", **context)
    if observed != expected:
        _fail("CONFIDENCE_CONTEXT_MISMATCH",
              f"provenance confidence {text!r} differs from pinned outcome "
              f"confidence {str(expected)!r}", **context)


def _recompute_mapping(mapping: SpecMapping,
                       raw_source_value: Any) -> RecomputedOutcome:
    """Dispatch pure scalar recomputation using only source and pinned mapping."""
    if mapping.op == OP_DIRECT:
        value = recompute_direct(raw_source_value, coerce=mapping.coerce)
        branch = _OUTCOME_NO_VALUE if _is_missing(value) else _OUTCOME_VALUE
        return RecomputedOutcome(value, branch)
    if mapping.op == OP_VALUE_MAP:
        return recompute_value_map(
            raw_source_value, mapping_table=mapping.map,
            on_unmapped=mapping.on_unmapped)
    if mapping.op == OP_UNIT_CONVERSION:
        return recompute_unit_conversion(
            raw_source_value, factor=mapping.factor, rounding=mapping.round)
    raise ValueError(f"unsupported scalar operation {mapping.op!r}")


def _outcome_context(mapping: SpecMapping,
                     outcome: RecomputedOutcome) -> tuple[RuleRef, Any]:
    """Select rule/confidence solely from the independently computed branch."""
    if outcome.branch in (_OUTCOME_VALUE, _OUTCOME_CONSTANT_VALUE):
        return mapping.rule, mapping.confidence
    if outcome.branch in (_OUTCOME_NO_VALUE, _OUTCOME_CONSTANT_NULL):
        return (mapping.no_value_rule or mapping.rule,
                mapping.confidence_no_value)
    if outcome.branch == _OUTCOME_UNMAPPED:
        return (mapping.unmapped_rule or mapping.rule,
                mapping.confidence_unmapped)
    if outcome.branch == _OUTCOME_VOCABULARY:
        if outcome.rule is None or outcome.confidence is None:
            raise ValueError("vocabulary outcome omits rule/confidence context")
        return outcome.rule, outcome.confidence
    raise ValueError(f"unknown recomputation outcome {outcome.branch!r}")


def _selector_semantic(value: Any, field: Field) -> Any:
    if value is None:
        if field.nullable:
            return None
        raise ValueError("null is not allowed")
    if field.type == "string":
        if not isinstance(value, str):
            raise ValueError("string row keys require string selector values")
        return value
    if isinstance(value, bool):
        raise ValueError("boolean is not a numeric selector")
    try:
        number = Decimal(str(value))
    except (InvalidOperation, ValueError):
        raise ValueError("not a decimal number")
    if not number.is_finite():
        raise ValueError("number is not finite")
    if field.type == "integer" and number != number.to_integral_value():
        raise ValueError("integer row key requires an integral value")
    return number


def _csv_semantic(text: str, field: Field, schema: TargetSchema) -> Any:
    if text == schema.null_encoding:
        if not field.nullable:
            raise ValueError(f"non-nullable field {field.name!r} is null")
        return None
    if field.type == "string":
        return text
    try:
        number = Decimal(text)
    except InvalidOperation as exc:
        raise ValueError(f"{field.name!r} is not numeric") from exc
    if not number.is_finite():
        raise ValueError(f"{field.name!r} is not finite")
    if field.type == "integer" and number != number.to_integral_value():
        raise ValueError(f"{field.name!r} is not an integer")
    return number


def _python_semantic(value: Any, field: Field) -> Any:
    if _is_missing(value):
        if not field.nullable:
            raise ValueError(f"non-nullable field {field.name!r} recomputed null")
        return None
    if field.type == "string":
        return str(value)
    try:
        number = Decimal(str(value))
    except InvalidOperation as exc:
        raise ValueError(f"recomputed {field.name!r} is not numeric") from exc
    if not number.is_finite():
        raise ValueError(f"recomputed {field.name!r} is not finite")
    if field.type == "integer" and number != number.to_integral_value():
        raise ValueError(f"recomputed {field.name!r} is not an integer")
    return number


def _result_value(text: str, field: Field, schema: TargetSchema) -> Any:
    semantic = _csv_semantic(text, field, schema)
    if semantic is None or field.type == "string":
        return semantic
    return text


def _source_text(value: Any) -> str:
    return "" if _is_missing(value) else str(value)


def _normalize_key(value: Any) -> str | None:
    """Match the executor's stringification and whitespace-collapse behavior."""
    if value is None or (isinstance(value, float) and value != value):
        return None
    return re.sub(r"\s+", " ", str(value)).strip()


def _value_map_is_absent(value: Any) -> bool:
    """Match the executor's approved value-map absence vocabulary exactly."""
    if value is None or (isinstance(value, float) and value != value):
        return True
    text = str(value).strip()
    return text == "" or text.lower() in ("nan", "na", "n/a")


def _fallback_raw_is_missing(value: Any) -> bool:
    """Match SpecExecutor fallback selection, deliberately without pd.isna."""
    return value is None or (isinstance(value, float) and value != value)


def _is_missing(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, float):
        return math.isnan(value)
    try:
        missing = pd.isna(value)
    except (TypeError, ValueError):
        return False
    if isinstance(missing, bool):
        return missing
    if type(missing).__module__.split(".")[0] == "numpy":
        return bool(missing)
    return False


def _json_scalar(value: Any) -> Any:
    if isinstance(value, Decimal):
        return str(value)
    if _is_missing(value):
        return None
    if type(value).__module__.split(".")[0] == "numpy" and hasattr(value, "item"):
        return value.item()
    return value


def _request_error(code: str, message: str) -> None:
    raise _Stop(ReplayStatus.REQUEST_ERROR, code, message)


def _not_replayable(code: str, message: str, **context) -> None:
    raise _Stop(ReplayStatus.NOT_REPLAYABLE, code, message, **context)


def _fail(code: str, message: str, **context) -> None:
    raise _Stop(ReplayStatus.FAIL, code, message, **context)

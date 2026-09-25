"""Runtime identity for grouping anchors used by strict replay.

The evidence ID describes construction semantics and exact runtime source
identities. Per-record selection and scalar evidence stay on Contributions so
one Cell never depends on every record in an anchor.
"""
from __future__ import annotations

import hashlib
import io
import json
from dataclasses import dataclass, field
from decimal import Decimal
from pathlib import Path
from typing import Any, Mapping

import pandas as pd

from .replay_contract import TypedRawValue, encode_typed_raw
from .source_ref import SourceRef, assert_within_roots


ANCHOR_EVIDENCE_KIND = "anchor"
GROUPING_ANCHOR_CONTRACT_VERSION = 1
EXTERNAL_ANCHOR_CONSTRUCTION_VERSION = 1
EXTERNAL_ANCHOR_FRAME_VERSION = 1
EXTERNAL_ANCHOR_READER_ID = "external_anchor_csv@v1"
EXTERNAL_ANCHOR_SOURCE_ID = "$anchor"
ANCHOR_RECORD_GROUPED = "grouped"
ANCHOR_RECORD_SYNTHESIZED = "synthesized"
ANCHOR_RECORD_EXTERNAL = "external"
ANCHOR_RECORD_KINDS = frozenset({
    ANCHOR_RECORD_GROUPED, ANCHOR_RECORD_SYNTHESIZED,
    ANCHOR_RECORD_EXTERNAL,
})
_TRUSTED_EXTERNAL_ANCHOR_TOKEN = object()

# Every result-affecting pandas.read_csv option in the v1 contract is explicit.
# None means that the feature is intentionally disabled rather than inherited
# from a future pandas default. The parsed bytes are always uncompressed UTF-8.
EXTERNAL_ANCHOR_READER_FLAGS = {
    "comment": None,
    "compression": None,
    "converters": None,
    "decimal": ".",
    "doublequote": True,
    "dtype": None,
    "encoding": "utf-8",
    "encoding_errors": "strict",
    "engine": "c",
    "escapechar": None,
    "false_values": None,
    "float_precision": None,
    "header": 0,
    "keep_default_na": True,
    "lineterminator": None,
    "low_memory": True,
    "memory_map": False,
    "na_filter": True,
    "na_values": None,
    "nrows": None,
    "on_bad_lines": "error",
    "parse_dates": False,
    "quotechar": '"',
    "quoting": 0,
    "sep": ",",
    "skip_blank_lines": True,
    "skipfooter": 0,
    "skipinitialspace": False,
    "skiprows": None,
    "thousands": None,
    "true_values": None,
}


def is_grouping_anchor_direct_evidence_mapping(spec, mapping) -> bool:
    """Whether a direct mapping consumes the closed Stage 10 anchor shape."""
    record = spec.record
    grouping = record.grouping
    if (mapping.op != "direct" or mapping.source != "$anchor"
            or record.anchor != "grouping" or grouping is None):
        return False
    if (mapping.when or mapping.derivation or mapping.contributions.conditions
            or mapping.group or mapping.column_by_member
            or mapping.coerce not in {"none", "text", "numeric"}):
        return False
    if (mapping.map or mapping.fallback_vocabulary or mapping.fallback_rules
            or mapping.candidates or mapping.clauses or mapping.inputs
            or mapping.params):
        return False
    if (not mapping.column or mapping.column not in grouping.anchor_fields
            or grouping.select != "first_row"):
        return False
    provenance = mapping.provenance
    if (provenance.row_from or provenance.raw_from
            or provenance.contributions_from
            or provenance.raw_value not in {"", "value"}
            or provenance.record_key_from == "group"):
        return False
    if provenance.source_ref:
        source = spec.source(provenance.source_ref)
        if ("." in provenance.source_ref or provenance.source_label
                or provenance.row_number != "none" or source is None
                or not source.path or source.path_pattern
                or source.file_type != "csv" or source.reader != "cidc_csv@v1"
                or source.sheet):
            return False
    elif provenance.row_number not in {"", "none"}:
        return False
    if (record.duplicates.policy not in {"error", "first", "last"}
            or len(record.additional_records) > 1):
        return False
    anchor_keys = {
        record.participant_key, record.sample_key, record.collection_event_key,
    }
    return all(
        declaration.when == "no_record_for_participant"
        and bool(declaration.id and declaration.participants_from)
        and declaration.rule is not None
        and (mapping.column in declaration.values
             or mapping.column in anchor_keys)
        for declaration in record.additional_records
    )


def is_external_anchor_evidence_mapping(spec, mapping) -> bool:
    """Whether a mapping is in the closed Stage 11 external-anchor subset."""
    if (spec.record.anchor != "external"
            or mapping.source != EXTERNAL_ANCHOR_SOURCE_ID
            or mapping.op not in {"direct", "value_map"}
            or not mapping.column or mapping.when or mapping.derivation
            or mapping.contributions.conditions
            or mapping.group or mapping.column_by_member
            or mapping.candidates or mapping.clauses or mapping.inputs
            or mapping.params or mapping.fallback_vocabulary
            or mapping.fallback_rules or mapping.artifacts):
        return False
    if mapping.op == "direct":
        if mapping.map or mapping.coerce not in {"none", "text", "numeric"}:
            return False
    elif (not mapping.map or mapping.coerce != "none"):
        return False
    provenance = mapping.provenance
    if (provenance.row_from or provenance.raw_from
            or provenance.contributions_from
            or provenance.raw_value not in {"", "value"}
            or provenance.record_key_from == "group"):
        return False
    if provenance.source_ref:
        source = spec.source(provenance.source_ref)
        if ("." in provenance.source_ref or provenance.source_label
                or provenance.row_number != "none" or source is None
                or not source.path or source.path_pattern
                or source.file_type != "csv" or source.reader != "cidc_csv@v1"
                or source.sheet):
            return False
    elif provenance.row_number not in {"", "none"}:
        return False
    return True


def _canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"),
                      sort_keys=True)


def _rule_ref(rule) -> str:
    return f"{rule.id}@{rule.version}" if rule is not None else ""


def grouping_anchor_identity(spec, source_refs: Mapping[str, Any], *,
                             record_count: int) -> dict[str, Any]:
    """Return the stable causal identity of an actual grouping construction."""
    rec = spec.record
    grouping = rec.grouping
    if rec.anchor != "grouping" or grouping is None:
        raise ValueError("grouping anchor evidence requires a grouping record")
    source_ids = [rec.anchor_source]
    source_ids.extend(item.participants_from for item in rec.additional_records)
    source_ids = list(dict.fromkeys(source_ids))
    missing = [source_id for source_id in source_ids
               if source_id not in source_refs]
    if missing:
        raise ValueError(
            f"grouping anchor runtime sources are missing {sorted(missing)}")
    if isinstance(record_count, bool) or not isinstance(record_count, int) \
            or record_count < 0:
        raise ValueError("grouping anchor record count must be non-negative")
    return {
        "construction_contract_version": GROUPING_ANCHOR_CONTRACT_VERSION,
        "spec_ref": spec.ref,
        "anchor_kind": "grouping",
        "anchor_source": rec.anchor_source,
        "source_ref_ids": {
            source_id: source_refs[source_id].source_ref_id
            for source_id in source_ids
        },
        "grouping": {
            "key_columns": list(grouping.key_columns),
            "selection": grouping.select,
            "read_order": "source_read_order",
            "anchor_fields": dict(sorted(grouping.anchor_fields.items())),
        },
        "duplicates": {
            "policy": rec.duplicates.policy,
            "by": list(rec.duplicates.by),
        },
        "additional_records": [
            {
                "id": item.id,
                "participants_from": item.participants_from,
                "when": item.when,
                "values": dict(sorted(item.values.items())),
                "rule_ref": _rule_ref(item.rule),
            }
            for item in rec.additional_records
        ],
        "record_count": record_count,
    }


@dataclass(frozen=True)
class RuntimeGroupingAnchorEvidence:
    """Identity retained from the exact runtime grouping construction."""

    evidence_ref_id: str
    construction_digest: str
    reference: str
    details: dict[str, Any]

    @classmethod
    def capture(cls, spec, source_refs: Mapping[str, Any], *,
                record_count: int) -> "RuntimeGroupingAnchorEvidence":
        identity = grouping_anchor_identity(
            spec, source_refs, record_count=record_count)
        canonical = _canonical_json(identity).encode("utf-8")
        digest = hashlib.sha256(canonical).hexdigest()
        details = dict(identity)
        details["construction_digest"] = digest
        return cls(
            evidence_ref_id=f"anchor-{digest}",
            construction_digest=digest,
            reference=f"{spec.ref}#record.anchor",
            details=details,
        )


def parse_external_anchor_bytes(payload: bytes) -> pd.DataFrame:
    """Parse exactly one captured byte string under external_anchor_csv@v1."""
    if not isinstance(payload, bytes):
        raise TypeError("external anchor payload must be bytes")
    return pd.read_csv(io.BytesIO(payload), **EXTERNAL_ANCHOR_READER_FLAGS)


def _typed_json(value: Any) -> dict[str, Any]:
    if isinstance(value, Decimal) and not value.is_finite():
        raise ValueError("external anchor numeric cells must be finite")
    try:
        missing = pd.isna(value)
        if isinstance(missing, bool) and missing:
            value = None
    except (TypeError, ValueError):
        pass
    return json.loads(encode_typed_raw(value).to_json())


def typed_value(document: Mapping[str, Any]) -> Any:
    """Decode one canonical typed scalar used by construction evidence."""
    typed = TypedRawValue.from_json(_canonical_json(dict(document)))
    if typed.kind == "null":
        return None
    if typed.kind == "integer":
        return int(typed.value)
    if typed.kind == "number":
        return float(typed.value)
    return typed.value


def canonical_external_anchor_frame(frame: pd.DataFrame) -> bytes:
    """Return external_anchor_frame@v1 canonical UTF-8 JSON bytes.

    Column and row order participate; the pandas index does not. Numeric cells
    use TypedRawValue's canonical decimal contract: integers remain integers,
    finite floats use normalized base-10 decimal text, and negative zero is
    retained as ``-0.0``. NaN is the canonical null; infinities and unsupported
    objects are refused.
    """
    if not isinstance(frame, pd.DataFrame):
        raise TypeError("external anchor frame must be a DataFrame")
    columns = list(frame.columns)
    if any(not isinstance(name, str) or not name for name in columns):
        raise ValueError("external anchor columns must be non-empty strings")
    duplicates = sorted({name for name in columns if columns.count(name) > 1})
    if duplicates:
        raise ValueError(f"external anchor has duplicate columns {duplicates}")
    rows = [
        [_typed_json(value) for value in row]
        for row in frame.itertuples(index=False, name=None)
    ]
    document = {
        "canonicalization_version": EXTERNAL_ANCHOR_FRAME_VERSION,
        "columns": columns,
        "rows": rows,
    }
    return json.dumps(
        document, ensure_ascii=False, separators=(",", ":"), sort_keys=True,
        allow_nan=False,
    ).encode("utf-8")


def filter_external_anchor_frame(
        frame: pd.DataFrame, *, column: str, selector: Any
        ) -> tuple[pd.DataFrame, tuple[int, ...]]:
    """Apply the closed v1 typed equality filter in parsed source order."""
    if not isinstance(column, str) or not column:
        raise ValueError("external anchor filter column must be non-empty")
    if column not in frame.columns:
        raise ValueError(f"external anchor has no filter column {column!r}")
    if encode_typed_raw(selector).kind == "null":
        raise ValueError("external anchor v1 does not support a null selector")
    if not isinstance(frame.index, pd.RangeIndex) \
            or tuple(frame.index) != tuple(range(len(frame))):
        raise ValueError("external anchor parsed frame must have a zero-based RangeIndex")
    mask = frame[column].eq(selector)
    if hasattr(mask, "fillna"):
        mask = mask.fillna(False)
    positions = tuple(int(value) for value in frame.index[mask])
    filtered = frame.loc[mask].reset_index(drop=True).copy(deep=True)
    return filtered, positions


def external_anchor_identity(
        *, spec_ref: str, source_ref: SourceRef, frame: pd.DataFrame,
        filter_column: str, selector: Any,
        original_source_rows: tuple[int, ...]) -> dict[str, Any]:
    """Stable causal identity of one external-anchor construction."""
    frame_digest = hashlib.sha256(
        canonical_external_anchor_frame(frame)).hexdigest()
    return {
        "construction_contract_version": EXTERNAL_ANCHOR_CONSTRUCTION_VERSION,
        "spec_ref": spec_ref,
        "anchor_kind": "external",
        "source_ref_ids": {EXTERNAL_ANCHOR_SOURCE_ID: source_ref.source_ref_id},
        "filter": {
            "operation": "column_equals",
            "column": filter_column,
            "selector": _typed_json(selector),
            "equality": "pandas_scalar_eq@v1",
            "null_semantics": "parsed_null_never_matches",
            "source_order": "preserved",
            "reset_index": "drop_true",
        },
        "pre_filter_row_count": source_ref.n_rows,
        "post_filter_row_count": len(frame),
        "ordered_columns": list(frame.columns),
        "original_source_rows": list(original_source_rows),
        "filtered_positions": list(range(len(frame))),
        "canonicalization": "external_anchor_frame@v1",
        "canonicalization_version": EXTERNAL_ANCHOR_FRAME_VERSION,
        "filtered_frame_sha256": frame_digest,
    }


@dataclass(frozen=True)
class RuntimeExternalAnchorEvidence:
    """Immutable construction identity retained from a trusted external input."""

    evidence_ref_id: str
    construction_digest: str
    reference: str
    details: dict[str, Any]

    @classmethod
    def capture(
            cls, *, spec_ref: str, source_ref: SourceRef,
            frame: pd.DataFrame, filter_column: str, selector: Any,
            original_source_rows: tuple[int, ...]
            ) -> "RuntimeExternalAnchorEvidence":
        identity = external_anchor_identity(
            spec_ref=spec_ref, source_ref=source_ref, frame=frame,
            filter_column=filter_column, selector=selector,
            original_source_rows=original_source_rows)
        canonical = _canonical_json(identity).encode("utf-8")
        digest = hashlib.sha256(canonical).hexdigest()
        details = dict(identity)
        details["construction_digest"] = digest
        return cls(
            evidence_ref_id=f"anchor-{digest}",
            construction_digest=digest,
            reference=f"{spec_ref}#record.anchor",
            details=details,
        )


@dataclass(frozen=True)
class RuntimeExternalAnchor:
    """Trusted external anchor plus the evidence for the exact visible frame."""

    source_ref: SourceRef
    evidence: RuntimeExternalAnchorEvidence
    original_source_rows: tuple[int, ...]
    filtered_positions: tuple[int, ...]
    _frame: pd.DataFrame = field(repr=False, compare=False)
    _factory_token: object = field(repr=False, compare=False)

    def __post_init__(self) -> None:
        if self._factory_token is not _TRUSTED_EXTERNAL_ANCHOR_TOKEN:
            raise ValueError(
                "trusted external anchors must be created by "
                "load_external_anchor")

    def frame_copy(self, *, spec_ref: str) -> pd.DataFrame:
        """Validate retained identity and return a mutation-isolated frame."""
        details = self.evidence.details
        expected_positions = tuple(range(len(self._frame)))
        if (details.get("spec_ref") != spec_ref
                or details.get("anchor_kind") != "external"
                or self.filtered_positions != expected_positions
                or tuple(details.get("filtered_positions", ()))
                   != expected_positions
                or tuple(details.get("original_source_rows", ()))
                   != self.original_source_rows
                or len(self.original_source_rows) != len(self._frame)
                or details.get("post_filter_row_count") != len(self._frame)
                or tuple(details.get("ordered_columns", ()))
                   != tuple(self._frame.columns)
                or hashlib.sha256(canonical_external_anchor_frame(
                    self._frame)).hexdigest()
                   != details.get("filtered_frame_sha256")):
            raise ValueError("external anchor runtime identity no longer matches its frame")
        return self._frame.copy(deep=True)


def load_external_anchor(
        path: str | Path, *, project_root: str | Path, spec_ref: str,
        filter_column: str, selector: Any,
        config_key: str = EXTERNAL_ANCHOR_SOURCE_ID,
        logical_label: str | None = None) -> RuntimeExternalAnchor:
    """Capture, hash, parse, filter, and bind one trusted external anchor.

    The file is read exactly once. The SourceRef hash and parsed DataFrame are
    therefore derived from the same immutable byte string.
    """
    root = Path(project_root).resolve()
    resolved = assert_within_roots(path, root)
    if not resolved.is_file():
        raise FileNotFoundError(f"external anchor file does not exist: {resolved}")
    payload = resolved.read_bytes()
    parsed = parse_external_anchor_bytes(payload)
    source_ref = SourceRef.from_observed_bytes(
        config_key=config_key, path=resolved, project_root=root,
        payload=payload, file_type="csv", reader_id=EXTERNAL_ANCHOR_READER_ID,
        logical_label=logical_label,
        reader_flags=_canonical_json(EXTERNAL_ANCHOR_READER_FLAGS),
        reader_effective_flags=EXTERNAL_ANCHOR_READER_FLAGS,
        n_rows=len(parsed),
    )
    filtered, original_rows = filter_external_anchor_frame(
        parsed, column=filter_column, selector=selector)
    evidence = RuntimeExternalAnchorEvidence.capture(
        spec_ref=spec_ref, source_ref=source_ref, frame=filtered,
        filter_column=filter_column, selector=selector,
        original_source_rows=original_rows)
    return RuntimeExternalAnchor(
        source_ref=source_ref, evidence=evidence,
        original_source_rows=original_rows,
        filtered_positions=tuple(range(len(filtered))),
        _frame=filtered.copy(deep=True),
        _factory_token=_TRUSTED_EXTERNAL_ANCHOR_TOKEN,
    )


class RuntimeSourceRefs(dict):
    """SourceRef mapping carrying evidence captured during that execution."""

    def __init__(self, *args, anchor_evidence=(), **kwargs):
        super().__init__(*args, **kwargs)
        self.anchor_evidence = tuple(anchor_evidence)

    def copy(self):
        return type(self)(self, anchor_evidence=self.anchor_evidence)

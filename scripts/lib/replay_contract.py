"""Versioned evidence primitives shared by future strict replay tooling.

This module defines data contracts only.  It does not locate a cell, read a
source, recompute a value, or assign a replay outcome.
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from enum import Enum
from numbers import Integral, Real
from typing import Any


REPLAY_STATUS_VERSION = 1
REPLAY_API_VERSION = 1


class ReplayStatus(str, Enum):
    PASS = "PASS"
    FAIL = "FAIL"
    NOT_REPLAYABLE = "NOT_REPLAYABLE"
    REQUEST_ERROR = "REQUEST_ERROR"


REASON_CODES_VERSION = 10
REASON_CODES = {
    ReplayStatus.PASS: frozenset({"REPLAY_MATCH"}),
    ReplayStatus.FAIL: frozenset({
        "DELIVERED_ARTIFACT_HASH_MISMATCH", "PROVENANCE_ARTIFACT_HASH_MISMATCH",
        "SCHEMA_HASH_MISMATCH", "SPEC_HASH_MISMATCH",
        "VOCABULARY_HASH_MISMATCH", "CONFIG_HASH_MISMATCH",
        "SOURCE_FILE_MISSING", "SOURCE_HASH_MISMATCH",
        "SOURCE_DECLARATION_MISMATCH", "READER_OBSERVATION_MISMATCH",
        "SOURCE_ROW_COUNT_MISMATCH", "SOURCE_ROW_OUT_OF_RANGE",
        "SOURCE_COLUMN_MISSING", "SOURCE_RECORD_KEY_MISMATCH",
        "RAW_VALUE_MISMATCH", "RULE_IDENTITY_MISMATCH",
        "CONFIDENCE_CONTEXT_MISMATCH", "PROVENANCE_VALUE_MISMATCH",
        "RECOMPUTED_VALUE_MISMATCH", "VALUE_MAP_UNMAPPED_ERROR",
        "UNIT_CONVERSION_NONFINITE_RESULT",
        "DEPENDENCY_CONTEXT_MISMATCH", "DEPENDENCY_REPLAY_FAILED",
        "REGISTERED_RULE_EXECUTION_FAILED",
        "FALLBACK_SELECTION_MISMATCH",
        "FALLBACK_CANDIDATE_CONTEXT_MISMATCH",
        "VOCABULARY_IDENTITY_MISMATCH",
        "VALUE_STATE_CONTEXT_MISMATCH",
        "ANCHOR_IDENTITY_MISMATCH", "ANCHOR_SELECTION_MISMATCH",
        "SOURCE_GROUP_SELECTION_MISMATCH",
        "DERIVATION_MISMATCH", "NAMED_READ_MISMATCH",
        "BRANCH_SELECTION_MISMATCH",
    }),
    ReplayStatus.NOT_REPLAYABLE: frozenset({
        "MANIFEST_VERSION_UNSUPPORTED", "MANIFEST_REQUIRED_FIELD_MISSING",
        "MANIFEST_PATH_OUTSIDE_PROJECT", "PROVENANCE_SCHEMA_UNSUPPORTED",
        "RUN_ID_LINK_MISSING", "RUN_ID_MISMATCH", "SPEC_LINK_MISSING",
        "SOURCE_LINK_MISSING", "ROW_KEY_NOT_REPRESENTABLE_IN_PROVENANCE",
        "PROVENANCE_EVENT_NOT_FOUND", "PROVENANCE_AUTHORITY_AMBIGUOUS",
        "PROVENANCE_NOT_AUTHORITATIVE", "SPEC_STATUS_NOT_APPROVED",
        "MAPPING_NOT_IDENTIFIED", "MAPPING_IDENTITY_AMBIGUOUS",
        "MAPPING_CONDITION_UNSUPPORTED", "OPERATION_UNSUPPORTED",
        "SOURCE_NOT_SINGLE_FILE", "SOURCE_LOCATOR_INCOMPLETE",
        "READER_METADATA_MISSING", "READER_UNSUPPORTED",
        "READER_FLAGS_UNSUPPORTED", "READER_ENVIRONMENT_MISMATCH",
        "RAW_VALUE_ENCODING_MISSING", "RAW_VALUE_KIND_UNSUPPORTED",
        "RECORD_KEY_UNPARSABLE", "VALUE_STATE_UNSUPPORTED",
        "CONTRIBUTIONS_UNSUPPORTED_STAGE1", "DERIVATION_UNSUPPORTED_STAGE1",
        "ANCHOR_SOURCE_UNSUPPORTED_STAGE1", "SUPPRESSION_UNSUPPORTED_STAGE1",
        "POST_RULE_UNSUPPORTED_STAGE1", "EXCLUSION_REQUIRES_ROW_REPLAY",
        "EXTERNAL_VOCABULARY_UNSUPPORTED",
        "PRODUCER_ENVIRONMENT_MISMATCH",
        "UNIT_CONVERSION_PARAMETERS_INVALID",
        "CONTRIBUTIONS_UNSUPPORTED_STAGE2",
        "DERIVATION_UNSUPPORTED_STAGE2",
        "DEPENDENCY_REFERENCE_MISSING", "DEPENDENCY_REFERENCE_AMBIGUOUS",
        "DEPENDENCY_NOT_REPLAYABLE", "REPLAY_CYCLE_DETECTED",
        "REPLAY_DEPTH_EXCEEDED", "REGISTERED_RULE_UNAVAILABLE",
        "REGISTERED_RULE_CONTRACT_MISMATCH",
        "FALLBACK_EVIDENCE_INCOMPLETE",
        "FALLBACK_CANDIDATE_UNSUPPORTED",
        "VOCABULARY_EVIDENCE_MISSING",
        "ANCHOR_EVIDENCE_MISSING",
        "SOURCE_GROUP_EVIDENCE_MISSING",
        "DERIVATION_EVIDENCE_MISSING", "NAMED_READ_EVIDENCE_MISSING",
        "MULTI_EVENT_BRANCH_UNSUPPORTED",
    }),
    ReplayStatus.REQUEST_ERROR: frozenset({
        "REQUEST_MALFORMED", "MANIFEST_PATH_INVALID", "ARTIFACT_ID_UNKNOWN",
        "SELECTOR_INCOMPLETE", "SELECTOR_HAS_EXTRA_KEYS", "FIELD_NOT_DECLARED",
        "FIELD_NOT_APPLICABLE", "TARGET_NOT_FOUND", "TARGET_AMBIGUOUS",
    }),
}


def _canonical_decimal(value: Any) -> str:
    try:
        dec = value if isinstance(value, Decimal) else Decimal(str(value))
    except (InvalidOperation, ValueError) as exc:
        raise ValueError(f"not a finite decimal number: {value!r}") from exc
    if not dec.is_finite():
        raise ValueError(f"not a finite decimal number: {value!r}")
    text = format(dec.normalize(), "f")
    if "." not in text:
        text += ".0"
    return text


@dataclass(frozen=True)
class TypedRawValue:
    """Deterministic scalar evidence; absence is represented by no payload."""

    kind: str
    value: Any = None

    def __post_init__(self) -> None:
        if self.kind not in {"null", "string", "integer", "number", "boolean"}:
            raise ValueError(f"unsupported typed raw kind {self.kind!r}")
        if self.kind == "null":
            if self.value is not None:
                raise ValueError("typed null cannot carry a value")
        elif self.kind == "string":
            if not isinstance(self.value, str):
                raise ValueError("typed string value must be a string")
        elif self.kind == "integer":
            if (not isinstance(self.value, str) or
                    not self.value or str(int(self.value)) != self.value):
                raise ValueError("typed integer value must be a canonical integer string")
        elif self.kind == "number":
            if (not isinstance(self.value, str) or
                    _canonical_decimal(self.value) != self.value):
                raise ValueError("typed number value must be a canonical decimal string")
        elif not isinstance(self.value, bool):
            raise ValueError("typed boolean value must be boolean")

    def to_json(self) -> str:
        payload = {"kind": self.kind}
        if self.kind != "null":
            payload["value"] = self.value
        return json.dumps(payload, separators=(",", ":"), sort_keys=True)

    @classmethod
    def from_json(cls, payload: str) -> "TypedRawValue":
        if not payload:
            raise ValueError("typed raw evidence is absent")
        try:
            doc = json.loads(payload)
        except (TypeError, json.JSONDecodeError) as exc:
            raise ValueError("typed raw evidence is not valid JSON") from exc
        if not isinstance(doc, dict) or set(doc) - {"kind", "value"}:
            raise ValueError("typed raw evidence must contain only kind/value")
        kind = doc.get("kind")
        if kind == "null":
            if "value" in doc:
                raise ValueError("typed null must omit value")
            return cls("null")
        if "value" not in doc:
            raise ValueError(f"typed raw {kind!r} requires value")
        value = doc["value"]
        if kind == "string" and not isinstance(value, str):
            raise ValueError("typed string value must be a string")
        if kind == "integer" and (not isinstance(value, str) or not value.lstrip("-").isdigit()):
            raise ValueError("typed integer value must be a canonical integer string")
        if kind == "number":
            if not isinstance(value, str) or _canonical_decimal(value) != value:
                raise ValueError("typed number value must be a canonical decimal string")
        if kind == "boolean" and not isinstance(value, bool):
            raise ValueError("typed boolean value must be boolean")
        return cls(kind, value)


def encode_typed_raw(value: Any) -> TypedRawValue:
    """Encode a scalar exactly enough to distinguish text, number and null."""
    if type(value).__module__.split(".")[0] == "numpy" and hasattr(value, "item"):
        return encode_typed_raw(value.item())
    if value is None:
        return TypedRawValue("null")
    if isinstance(value, bool):
        return TypedRawValue("boolean", value)
    if isinstance(value, Integral):
        return TypedRawValue("integer", str(int(value)))
    if isinstance(value, Decimal):
        return TypedRawValue("number", _canonical_decimal(value))
    if isinstance(value, Real):
        if math.isnan(float(value)):
            return TypedRawValue("null")
        if not math.isfinite(float(value)):
            raise ValueError("infinite raw values are unsupported")
        return TypedRawValue("number", _canonical_decimal(value))
    if isinstance(value, str):
        return TypedRawValue("string", value)
    raise ValueError(f"raw value is not a supported scalar: {type(value).__name__}")

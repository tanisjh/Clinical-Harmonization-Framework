"""Pure confidence policy for registered field-rule results.

This module decides only which already-authoritative confidence applies.  It
does not acquire rule inputs, invoke rules, inspect provenance, or compare
delivered output.  Callers must establish those authorities before using the
policy here.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

from .conditions import is_missing_scalar


CONFIDENCE_FROM_RULE = "from_rule"
OUTCOME_VALUE = "value"
OUTCOME_NO_VALUE = "no_value"
AUTHORITY_RULE = "rule"
AUTHORITY_DECLARED_NO_VALUE = "declared_no_value"


class ConfidencePolicyError(ValueError):
    """A registered-rule confidence cannot be finalized safely."""


class UnsupportedConfidencePolicy(ConfidencePolicyError):
    """The declaration is outside the implemented registered-rule policy."""


class InvalidRuleConfidence(ConfidencePolicyError):
    """A registered rule returned a malformed confidence."""


@dataclass(frozen=True)
class FinalizedConfidence:
    """One independently finalized confidence and its policy branch."""

    value: float
    outcome: str
    authority: str


def validate_registered_rule_confidence(value: Any) -> float:
    """Validate and normalize the confidence part of a field-rule return.

    The accepted runtime language intentionally matches the repository's
    declarative numeric confidence language: Python integer/float values other
    than booleans, finite and within the closed interval [0, 1].
    """
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise InvalidRuleConfidence(
            "registered rule confidence must be numeric and not boolean")
    normalized = float(value)
    if not math.isfinite(normalized) or not 0.0 <= normalized <= 1.0:
        raise InvalidRuleConfidence(
            "registered rule confidence must be finite and within [0, 1]")
    return normalized


def _validate_declared_no_value_confidence(value: Any) -> float | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise UnsupportedConfidencePolicy(
            "declared confidence.no_value must be numeric and not boolean")
    normalized = float(value)
    if not math.isfinite(normalized) or not 0.0 <= normalized <= 1.0:
        raise UnsupportedConfidencePolicy(
            "declared confidence.no_value must be finite and within [0, 1]")
    return normalized


def finalize_registered_rule_confidence(
        semantic_value: Any, rule_confidence: Any, *,
        declared_value_policy: Any,
        declared_no_value_confidence: Any = None) -> FinalizedConfidence:
    """Finalize confidence from a valid rule return and pinned declaration.

    Rule confidence is validated before the optional no-value override.  Thus
    an override can choose the final confidence for a legitimate missing
    result, but it can never rescue a malformed registered-rule return.
    """
    if declared_value_policy != CONFIDENCE_FROM_RULE:
        raise UnsupportedConfidencePolicy(
            "registered field rules require confidence policy 'from_rule'")
    normalized_rule = validate_registered_rule_confidence(rule_confidence)
    normalized_no_value = _validate_declared_no_value_confidence(
        declared_no_value_confidence)
    missing = is_missing_scalar(semantic_value)
    if missing and normalized_no_value is not None:
        return FinalizedConfidence(
            normalized_no_value, OUTCOME_NO_VALUE,
            AUTHORITY_DECLARED_NO_VALUE)
    return FinalizedConfidence(
        normalized_rule,
        OUTCOME_NO_VALUE if missing else OUTCOME_VALUE,
        AUTHORITY_RULE)

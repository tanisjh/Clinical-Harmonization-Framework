"""Pure scalar condition semantics shared by execution and strict replay.

This module deliberately has no source, provenance, Cell, or replay imports.
Callers must establish the authority for an operand before asking whether a
predicate holds.  Keeping that boundary here prevents recorded branch evidence
from becoming an input to its own reconstruction.
"""
from __future__ import annotations

import re
from typing import Any


PREDICATES = (
    "is_null", "not_null", "equals", "not_equals", "in", "not_in",
    "gte", "gt", "lte", "lt",
)
NULLARY_PREDICATES = frozenset({"is_null", "not_null"})
SET_PREDICATES = frozenset({"in", "not_in"})
NUMERIC_PREDICATES = frozenset({"gte", "gt", "lte", "lt"})
VALUE_PREDICATES = frozenset({"equals", "not_equals"}) | NUMERIC_PREDICATES


class PredicateDefinitionError(ValueError):
    """A declared scalar predicate is outside the closed language."""


def normalize_scalar(value: Any) -> str | None:
    """Collapse whitespace and stringify exactly as executor map keys do."""
    if is_missing_scalar(value):
        return None
    return re.sub(r"\s+", " ", str(value)).strip()


def is_missing_scalar(value: Any) -> bool:
    """The executor's intentionally narrow missing-value test."""
    return value is None or (isinstance(value, float) and value != value)


def validate_predicate_definition(predicate: str, value: Any) -> None:
    """Validate one declaration without consulting runtime facts."""
    if predicate not in PREDICATES:
        raise PredicateDefinitionError(
            f"predicate must be one of {list(PREDICATES)}, got {predicate!r}")
    if predicate in NULLARY_PREDICATES:
        if value is not None:
            raise PredicateDefinitionError(
                f"predicate {predicate!r} takes no value")
        return
    if predicate in SET_PREDICATES:
        if not isinstance(value, list) or not value:
            raise PredicateDefinitionError(
                f"predicate {predicate!r} needs a non-empty list value")
        return
    if value is None:
        raise PredicateDefinitionError(
            f"predicate {predicate!r} needs a value")
    if predicate in NUMERIC_PREDICATES and (
            isinstance(value, bool) or not isinstance(value, (int, float))):
        raise PredicateDefinitionError(
            f"predicate {predicate!r} needs a numeric value, got {value!r}")


def evaluate_predicate(predicate: str, operand: Any, wanted: Any = None) -> bool:
    """Evaluate one validated scalar predicate against an authoritative value."""
    validate_predicate_definition(predicate, wanted)
    key = normalize_scalar(operand)
    present = not is_missing_scalar(operand) and key not in ("", None)
    if predicate == "is_null":
        return not present
    if predicate == "not_null":
        return present
    if not present:
        return False
    if predicate == "equals":
        return key == normalize_scalar(wanted)
    if predicate == "not_equals":
        return key != normalize_scalar(wanted)
    if predicate in SET_PREDICATES:
        inside = key in {normalize_scalar(item) for item in wanted}
        return inside if predicate == "in" else not inside
    try:
        left, right = float(key), float(wanted)
    except (TypeError, ValueError):
        return False
    return {
        "gte": left >= right,
        "gt": left > right,
        "lte": left <= right,
        "lt": left < right,
    }[predicate]


def coerce_named_read(value: Any, coercion: str) -> Any:
    """Apply the pinned named-read coercion to a verified source scalar."""
    if coercion == "none":
        return value
    key = normalize_scalar(value)
    if coercion == "text":
        return key
    if coercion == "numeric":
        if key is None or key in ("", "."):
            return None
        try:
            return float(key)
        except (TypeError, ValueError):
            return None
    raise ValueError(f"unsupported named-read coercion {coercion!r}")


def first_true_index(results: tuple[bool, ...] | list[bool]) -> int | None:
    """Return the first true ordered clause, or ``None`` for ``otherwise``."""
    return next((rank for rank, result in enumerate(results) if result), None)

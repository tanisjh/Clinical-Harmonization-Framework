"""Pure structured-derivation rendering and canonical JSON comparison."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from collections.abc import Mapping, Sequence
from typing import Any


REF_MARKER = "$ref"
STAGE14_FACTS = frozenset({"value", "source_column"})
STAGE16_FACT_PREFIXES = frozenset({
    "read", "condition", "group_member", "group_value", "group_label",
})
STAGE19_FACTS = frozenset({"anchor_group_size", "anchor_selected_row"})


class DerivationError(ValueError):
    """A derivation template or JSON value is not deterministic and closed."""


class MissingDerivationFact(DerivationError):
    """A selected template names a fact the caller did not reconstruct."""


@dataclass(frozen=True)
class DerivationSelection:
    """A declared-order selection and only the conditions causal to it."""

    record: dict | None
    condition_ids: tuple[str, ...]


def derivation_references(value: Any) -> frozenset[str]:
    """Return every exact fact reference in a nested derivation value."""
    refs: set[str] = set()

    def visit(item: Any) -> None:
        if isinstance(item, dict):
            if REF_MARKER in item:
                if set(item) != {REF_MARKER}:
                    raise DerivationError(
                        f"{REF_MARKER} reference must be a one-key object")
                ref = item[REF_MARKER]
                if not isinstance(ref, str) or not ref:
                    raise DerivationError(
                        f"{REF_MARKER} must name a non-empty fact")
                refs.add(ref)
                return
            for child in item.values():
                visit(child)
        elif isinstance(item, list):
            for child in item:
                visit(child)

    visit(value)
    return frozenset(refs)


def is_stage14_derivation(variants: Sequence[Any]) -> bool:
    """Whether variants are the closed unconditional Stage 14 fact shape."""
    if len(variants) != 1 or any(variant.when for variant in variants):
        return False
    try:
        refs = frozenset().union(
            *(derivation_references(variant.record) for variant in variants))
    except DerivationError:
        return False
    return refs <= STAGE14_FACTS


def is_stage16_fact(ref: str) -> bool:
    """Whether one fact name is inside the closed Stage 16 composition set."""
    if ref in STAGE14_FACTS:
        return True
    head, separator, rest = ref.partition(".")
    return bool(separator and rest and head in STAGE16_FACT_PREFIXES)


def is_stage16_derivation(variants: Sequence[Any]) -> bool:
    """Whether every template reference is in the closed Stage 16 namespace."""
    if not variants:
        return False
    try:
        refs = frozenset().union(
            *(derivation_references(variant.record) for variant in variants))
    except DerivationError:
        return False
    return all(is_stage16_fact(ref) for ref in refs)


def is_stage17_fact(ref: str) -> bool:
    """Whether a fact is Stage 16 authority or finalized route raw."""
    return ref == "raw_value" or is_stage16_fact(ref)


def is_stage17_derivation(variants: Sequence[Any]) -> bool:
    """Recognize templates whose only new fact is finalized ``raw_value``.

    Mapping/route eligibility is intentionally enforced by replay separately;
    this syntax predicate alone never makes a mapping replayable.
    """
    if not variants:
        return False
    try:
        refs = frozenset().union(
            *(derivation_references(variant.record) for variant in variants))
    except DerivationError:
        return False
    return "raw_value" in refs and all(is_stage17_fact(ref) for ref in refs)


def is_stage19_fact(ref: str) -> bool:
    """Whether a fact has existing authority or verified record construction."""
    return ref in STAGE19_FACTS or is_stage17_fact(ref)


def is_stage19_derivation(variants: Sequence[Any]) -> bool:
    """Recognize templates composed with Stage 19 grouping-anchor facts."""
    if not variants:
        return False
    try:
        refs = frozenset().union(
            *(derivation_references(variant.record) for variant in variants))
    except DerivationError:
        return False
    return bool(refs & STAGE19_FACTS) and all(
        is_stage19_fact(ref) for ref in refs)


def select_derivation(
        variants: Sequence[Any], conditions: Mapping[str, Any]
        ) -> DerivationSelection:
    """Select in declaration order and retain only conditions actually tested."""
    deciding: list[str] = []
    for variant in variants:
        if variant.when:
            deciding.append(variant.when)
            if not conditions.get(variant.when):
                continue
        return DerivationSelection(variant.record, tuple(deciding))
    return DerivationSelection(None, tuple(deciding))


def select_derivation_record(
        variants: Sequence[Any], conditions: Mapping[str, Any]) -> dict | None:
    """Select a variant with the executor's declared-order semantics."""
    return select_derivation(variants, conditions).record


def render_derivation_record(value: Any, facts: Mapping[str, Any]) -> Any:
    """Recursively substitute exact reference objects from a closed fact map."""
    if isinstance(value, dict):
        if REF_MARKER in value:
            if set(value) != {REF_MARKER}:
                raise DerivationError(
                    f"{REF_MARKER} reference must be a one-key object")
            ref = value[REF_MARKER]
            if not isinstance(ref, str) or not ref:
                raise DerivationError(
                    f"{REF_MARKER} must name a non-empty fact")
            if ref not in facts:
                raise MissingDerivationFact(
                    f"required derivation fact {ref!r} is missing")
            return facts[ref]
        return {key: render_derivation_record(child, facts)
                for key, child in value.items()}
    if isinstance(value, list):
        return [render_derivation_record(child, facts) for child in value]
    return value


def matches_derivation_template(template: Any, rendered: Any) -> bool:
    """Match one rendered record shape while treating exact refs as wildcards."""
    if isinstance(template, dict):
        if REF_MARKER in template:
            return (set(template) == {REF_MARKER}
                    and isinstance(template[REF_MARKER], str)
                    and bool(template[REF_MARKER]))
        return (isinstance(rendered, dict)
                and set(rendered) == set(template)
                and all(matches_derivation_template(template[key], rendered[key])
                        for key in template))
    if isinstance(template, list):
        return (isinstance(rendered, list)
                and len(rendered) == len(template)
                and all(matches_derivation_template(left, right)
                        for left, right in zip(template, rendered)))
    return type(template) is type(rendered) and template == rendered


def _validate_json_value(value: Any) -> None:
    if value is None or isinstance(value, (str, bool)):
        return
    if isinstance(value, int):
        return
    if isinstance(value, float):
        if not math.isfinite(value):
            raise DerivationError("derivation JSON contains a non-finite number")
        return
    if isinstance(value, list):
        for child in value:
            _validate_json_value(child)
        return
    if isinstance(value, dict):
        for key, child in value.items():
            if not isinstance(key, str):
                raise DerivationError("derivation JSON object keys must be strings")
            _validate_json_value(child)
        return
    raise DerivationError(
        f"derivation JSON contains unsupported {type(value).__name__}")


def canonical_derivation_json(value: Any) -> str:
    """Encode one derivation object as deterministic strict JSON."""
    _validate_json_value(value)
    return json.dumps(
        value, separators=(",", ":"), sort_keys=True, allow_nan=False)


def render_structured_derivation(
        variants: Sequence[Any], facts: Mapping[str, Any],
        conditions: Mapping[str, Any]) -> str:
    """Select, substitute, and canonically encode one derivation record."""
    record = select_derivation_record(variants, conditions)
    if record is None:
        return ""
    return canonical_derivation_json(render_derivation_record(record, facts))


def parse_canonical_derivation_json(payload: str) -> str:
    """Strictly parse recorded JSON and return its canonical encoding."""
    def object_pairs(pairs):
        out = {}
        for key, value in pairs:
            if key in out:
                raise DerivationError(
                    f"derivation JSON contains duplicate key {key!r}")
            out[key] = value
        return out

    def reject_constant(token: str):
        raise DerivationError(
            f"derivation JSON contains non-standard constant {token!r}")

    try:
        value = json.loads(
            payload, object_pairs_hook=object_pairs,
            parse_constant=reject_constant)
    except (json.JSONDecodeError, TypeError) as exc:
        raise DerivationError(f"derivation JSON is malformed: {exc}") from exc
    if not isinstance(value, dict):
        raise DerivationError("recorded derivation must be a JSON object")
    return canonical_derivation_json(value)

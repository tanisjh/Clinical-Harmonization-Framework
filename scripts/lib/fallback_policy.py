"""Pure ordered-fallback selection and inline-map composition.

This module deliberately knows nothing about source bindings, provenance,
recorded outputs, or delivery. Callers must first establish authority for the
candidate scalars. Keeping selection here makes execution and strict replay use
the same narrow missing predicate and selection-before-map ordering.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from .conditions import is_missing_scalar, normalize_scalar


MAPPING_MATCHED = "matched"
MAPPING_UNMAPPED_NULL = "unmapped_null"
MAPPING_UNMAPPED_PASSTHROUGH = "unmapped_passthrough"


@dataclass(frozen=True)
class OrderedFallbackSelection:
    """The first nonmissing raw candidate, or an exhausted chain."""

    selected_rank: int | None
    raw_selected: Any

    @property
    def exhausted(self) -> bool:
        return self.selected_rank is None


@dataclass(frozen=True)
class InlineFallbackValue:
    """A selected raw candidate after deterministic inline mapping."""

    value: Any
    outcome: str


def select_ordered_fallback(
        raw_candidates: Sequence[Any]) -> OrderedFallbackSelection:
    """Select the first candidate present under the executor predicate."""
    for rank, raw in enumerate(raw_candidates):
        if not is_missing_scalar(raw):
            return OrderedFallbackSelection(rank, raw)
    return OrderedFallbackSelection(None, None)


def apply_inline_fallback_map(
        raw_selected: Any, *, mapping_table: Mapping[str, Any],
        on_unmapped: str) -> InlineFallbackValue:
    """Map an already-selected raw scalar without reconsidering fallback.

    The executor historically treats every fallback ``on_unmapped`` mode other
    than ``passthrough`` as a null result. Declaration-level replay scope is
    responsible for admitting only a closed supported mode; this function
    preserves execution behavior for existing declarations.
    """
    key = normalize_scalar(raw_selected)
    if key in mapping_table:
        return InlineFallbackValue(mapping_table[key], MAPPING_MATCHED)
    if on_unmapped == "passthrough":
        return InlineFallbackValue(
            raw_selected, MAPPING_UNMAPPED_PASSTHROUGH)
    return InlineFallbackValue(None, MAPPING_UNMAPPED_NULL)

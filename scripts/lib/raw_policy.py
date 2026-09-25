"""Pure finalized-raw policy shared by execution and strict replay.

This module performs no evidence acquisition.  Callers must supply an
independently established operation value, route raw candidate, completion
state, and (when causally relevant) named-read results.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping


COMPLETION_VALUE = "value"
COMPLETION_FALLBACK_EXHAUSTED = "fallback_exhausted"
_RAW_POLICIES = frozenset({"", "value", "coerced", "none"})


class RawAuthorityUnavailable(ValueError):
    """A causally required route or named-read raw value was not supplied."""


@dataclass(frozen=True)
class VerifiedRawCandidate:
    """A typed causal scalar, distinct from unavailable authority."""

    available: bool
    value: Any = None


@dataclass(frozen=True)
class VerifiedOperationRouteResult:
    """The minimum operation result needed by finalized-raw policy."""

    value: Any
    raw_candidate: VerifiedRawCandidate
    completion: str = COMPLETION_VALUE


@dataclass(frozen=True)
class FinalizedRaw:
    """The executor-equivalent final raw value and snapshot policy."""

    value: Any
    record_snapshot: bool


def raw_from_is_causal(*, raw_from_id: str, raw_policy: str,
                       completion: str = COMPLETION_VALUE) -> bool:
    """Whether ``raw_from`` controls finalized raw under the pinned policy."""
    _validate_policy(raw_policy, completion)
    return bool(
        raw_from_id
        and completion != COMPLETION_FALLBACK_EXHAUSTED
        and (raw_policy or "value") == "value"
    )


def finalize_raw_value(
        route_result: VerifiedOperationRouteResult,
        named_reads: Mapping[str, Any], *, raw_from_id: str = "",
        raw_policy: str = "") -> FinalizedRaw:
    """Apply executor final-raw precedence to already verified causal inputs.

    Fallback exhaustion and explicit suppression mask every lower-precedence
    input.  ``coerced`` uses the independently recomputed operation value.
    Only default/``value`` needs either the declared named-read result or the
    operation route's pre-transformation raw candidate.
    """
    _validate_policy(raw_policy, route_result.completion)
    policy = raw_policy or "value"
    if route_result.completion == COMPLETION_FALLBACK_EXHAUSTED:
        return FinalizedRaw(None, False)
    if policy == "none":
        return FinalizedRaw(None, False)
    if policy == "coerced":
        return FinalizedRaw(route_result.value, True)
    if raw_from_id:
        if raw_from_id not in named_reads:
            raise RawAuthorityUnavailable(
                f"causal raw_from read {raw_from_id!r} is unavailable")
        return FinalizedRaw(named_reads[raw_from_id], True)
    if not route_result.raw_candidate.available:
        raise RawAuthorityUnavailable("causal route raw is unavailable")
    return FinalizedRaw(route_result.raw_candidate.value, True)


def _validate_policy(raw_policy: str, completion: str) -> None:
    if raw_policy not in _RAW_POLICIES:
        raise ValueError(f"unsupported raw-value policy {raw_policy!r}")
    if completion not in {COMPLETION_VALUE, COMPLETION_FALLBACK_EXHAUSTED}:
        raise ValueError(f"unsupported route completion {completion!r}")

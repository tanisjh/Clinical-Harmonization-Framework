"""value_states.py — What happened to a value, as a typed field.

A trace has to distinguish outcomes that all look like "empty" or all look like
"a value" in the output table, and that were previously separable only by reading
free-text notes:

    delivered       the value reached the artifact as extracted
    transformed     a rule computed a value the inputs never held
                    (ABTC1603 clinical benefit from a PFS landmark)
    provisional     delivered, but interim — better data is expected
                    (14C0059G OS/PFS pending trial-team delivery, RAE C11)
    suppressed      a value existed but was withheld by a threshold or rule
    row_excluded    the whole record was dropped from an artifact
    source_missing  the source has the field but no value for this record
    not_applicable  the source has no such field, or it has no meaning here

The distinction that matters most clinically is `source_missing` versus
`not_applicable`. A blank age for 14C0059G is not a patient whose age is
unknown — the demographics file has no age column at all. Presenting both as
"NA" invites a reader to assume the former.

`provisional` is a delivery state, not a failure: the value IS in the artifact.
It marks values a trial team has been asked to replace.

An empty string means `delivered`, so every row written before this field
existed keeps its meaning.
"""
from __future__ import annotations

from dataclasses import dataclass

DELIVERED = "delivered"
TRANSFORMED = "transformed"
PROVISIONAL = "provisional"
SUPPRESSED = "suppressed"
ROW_EXCLUDED = "row_excluded"
SOURCE_MISSING = "source_missing"
NOT_APPLICABLE = "not_applicable"

VALID = frozenset({
    DELIVERED, TRANSFORMED, PROVISIONAL, SUPPRESSED,
    ROW_EXCLUDED, SOURCE_MISSING, NOT_APPLICABLE,
})

#: States in which a value is actually present in an artifact.
PRESENT = frozenset({DELIVERED, TRANSFORMED, PROVISIONAL})

#: States that explain an ABSENCE.
ABSENT = frozenset({SUPPRESSED, ROW_EXCLUDED, SOURCE_MISSING, NOT_APPLICABLE})

# Mapping-level states whose authority is entirely the pinned declaration.
# Absence and artifact-stage states deliberately retain their specialized
# runtime authorities.
STATIC_PRESENT_DECLARATIONS = frozenset({"", *PRESENT})
STATIC_SERIALIZATION_MODES = frozenset({"", "declared", "none"})


@dataclass(frozen=True)
class FinalizedValueState:
    """One static semantic state and its exact Cell serialization."""

    semantic_state: str
    serialized_state: str


def finalize_declared_value_state(
        declared_state: str, provenance_state_mode: str
        ) -> FinalizedValueState:
    """Finalize pure mapping-level present-state policy.

    The result depends only on the pinned mapping declaration.  In particular,
    it accepts no recorded Cell, operation result, derivation, or delivery.
    """
    if declared_state not in STATIC_PRESENT_DECLARATIONS:
        raise ValueError(
            f"declared value_state {declared_state!r} is not a static "
            "present state")
    if provenance_state_mode not in STATIC_SERIALIZATION_MODES:
        raise ValueError(
            f"provenance value_state mode {provenance_state_mode!r} is "
            "unsupported")
    return FinalizedValueState(
        semantic_state=normalize(declared_state),
        serialized_state=(
            "" if provenance_state_mode == "none" else declared_state),
    )


def normalize(value: str) -> str:
    """Empty means delivered — the default for every pre-existing row."""
    return value or DELIVERED


def is_present(value: str) -> bool:
    return normalize(value) in PRESENT

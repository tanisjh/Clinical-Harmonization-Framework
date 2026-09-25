"""rule_ids.py — Stable transformation-rule identifiers.

Phase 1a introduced these as module constants inside the S1400I extractor. Phase
1b lifts the shared ones here so migrated trials agree on a vocabulary instead of
each inventing its own.

This is deliberately NOT the rule registry described in the Phase 1 plan §6.2 —
there is no lookup table, no config back-pointer and no approval metadata. It is
a shared list of names, so that `value_map.trial.v1` means the same thing in
every migrated extractor.

Convention:  <kind>[.<qualifier>].v<n>
  * lowercase, dot-separated, [a-z0-9_] segments
  * the trailing .v<n> is the rule version, also emitted separately in the
    provenance `rule_version` column so it can be filtered numerically
"""
from __future__ import annotations

# ── simple single-source rules ──────────────────────────────────────────────
DIRECT_COPY      = "direct_copy.v1"          # verbatim copy of a source column
DIRECT_NUMERIC   = "direct_numeric.v1"       # copy + numeric coercion
CONSTANT         = "constant.v1"             # value from config trial_constants
CONSTANT_NA      = "constant_na.v1"          # deliberate NA asserted by config
TEMPLATE_ANCHOR  = "template_anchor.v1"      # value taken from the row anchor

# ── vocabulary mapping ──────────────────────────────────────────────────────
VMAP_TRIAL       = "value_map.trial.v1"      # per-trial value_maps
VMAP_GLOBAL      = "value_map.global.v1"     # global value_normalizations
VMAP_CE_ALT      = "value_map.collection_event_alt.v1"

# ── unit conversion ─────────────────────────────────────────────────────────
TIME_UNIT_CONV   = "time_unit_conversion.v1"

# ── multi-candidate selection ───────────────────────────────────────────────
FIRST_NON_NULL   = "first_non_null.v1"       # first non-null of an ordered list
FIRST_NON_NULL_EXHAUSTED = "first_non_null_exhausted.v1"  # every candidate null

# Conditional composites: an ordered candidate list where a step is gated on
# another field, so a candidate can be REJECTED (present but declined) rather
# than merely unavailable.
OS_TIME_COMPOSITE  = "derived.os_time_composite.v1"   # dead ? days_to_death : days_to_last_contact
PFS_TIME_COMPOSITE = "derived.pfs_time_composite.v1"  # progression > death(if dead) > last contact
STATUS_FROM_NONNULL = "derived.status_from_nonnull.v1"  # 1 iff a date column is populated

# Evidence is which FILE a record appears in, not a column value. Used where a
# trial splits its cohort into per-arm files (9204).
VALUE_FROM_FILE_MEMBERSHIP = "derived.value_from_file_membership.v1"

# ── derivations ─────────────────────────────────────────────────────────────
DER_BOR_BINARY   = "derived.bor_binary.v1"
DER_BOR_BIN_120  = "derived.bor_bin_120d.v1"
DER_PFS_BIN_120  = "derived.pfs_bin_120d.v1"

# ── trial-specific ──────────────────────────────────────────────────────────
AGE_ENROLL_SUBST = "age_enrollment_substitute.v1"   # S1400I reviewer decision
ARM_FROM_SEX     = "template_bug.arm_from_sex.v1"   # deliberate template repro
PCR_CLINICAL_BENEFIT = "derived.pcr_clinical_benefit.v1"  # 10013 per-patient pCR (RAE C6)
TREATMENT_PER_ARM = "derived.treatment_per_arm.v1"        # arm code -> regimen label
ANCHOR_FROM_GROUPING = "derived.anchor_from_grouping.v1"  # anchor row = a group of source rows
ANCHOR_SYNTHESIZED = "derived.anchor_synthesized.v1"      # anchor exists in no source file

# 10104: randomized arm, arm-origin file precedence, crossover course, and
# survival read from a per-arm file whose column name states the anchor.
RANDOMIZED_ARM        = "derived.randomized_arm.v1"
ARM_ORIGIN_PRECEDENCE = "derived.arm_origin_precedence.v1"
CROSSOVER_TREATMENT   = "derived.crossover_sequential_course.v1"
SURVIVAL_FROM_SOURCE  = "derived.survival_time_from_source.v1"

# ── post-pivot overrides ────────────────────────────────────────────────────
# Applied by the orchestrator AFTER the long->wide pivot, and authoritative only
# for the artifacts named on the event. See lib/artifacts.py.
CLIN_BENEFIT_FROM_PFS_120D = "derived.clinical_benefit_from_pfs_120d.v1"

# ── non-values ──────────────────────────────────────────────────────────────
LOOKUP_MISS      = "lookup_miss.v1"          # participant absent from the source
NOT_IN_SOURCE    = "not_in_source.v1"        # the source has no such column at all
FIELD_NOT_APPLICABLE = "field_not_applicable.v1"   # target field has no meaning for this trial
PROVISIONAL      = "provisional_pending_delivery.v1"  # interim value; better data expected

# ── row- and cell-level suppression (Phase 1d) ──────────────────────────────
ROW_EXCLUDED     = "row_excluded.v1"         # participant dropped from an artifact
VALUE_SUPPRESSED = "value_suppressed_below_threshold.v1"

ALL = {
    name: value for name, value in list(globals().items())
    if name.isupper() and name != "ALL" and isinstance(value, str)
}

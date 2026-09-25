"""artifacts.py — Output artifacts and provenance authority.

The pipeline writes three harmonized tables from one cell stream, and they do not
always agree. ABTC1603 `clinical_benefit` is the clear case: the extractor finds
no BOR column in the source, so the template-reproduction artifact delivers NA,
while the gold deliverables carry a value derived post-pivot from a 120-day PFS
landmark (Dr. Ye, 2026-06-17).

Before Phase 1c, provenance described only the extractor's view. That made it
wrong for 290 delivered cells: a reader tracing `clinical_benefit='Clin-Ben'` in
harmonized_11trials.csv was told the source contained no value.

The fix is to let a provenance row say WHICH ARTIFACTS it speaks for.

Authority rules
  * `authoritative_for == ""`  — the row is authoritative for every artifact in
    which its anchor is delivered. This is the default and covers every row the
    pipeline produced before Phase 1c, so nothing else had to change.
  * `authoritative_for == "a.csv;b.csv"` — authoritative only for those.

Exactly one row may be authoritative for a given (anchor, field, artifact). Two
rows claiming the same cell in the same artifact is a defect, not a preference,
and the audit fails on it.
"""
from __future__ import annotations

#: The template-reproduction QC artifact. Retains screen-failure rows and is NOT
#: subject to the gold-only clinical-benefit override, so it stays comparable to
#: the reference 9-trial table row by row.
REPRODUCED = "harmonized_9trials_reproduced.csv"

#: The corrected 9-trial deliverable.
GOLD9 = "harmonized_9trials_gold.csv"

#: The corrected 11-trial deliverable.
ELEVEN = "harmonized_11trials.csv"

ALL_ARTIFACTS = (REPRODUCED, GOLD9, ELEVEN)

#: Artifacts the gold-only rules apply to.
GOLD_ARTIFACTS = (GOLD9, ELEVEN)

SEP = ";"

#: Explicit "authoritative for no artifact". Needed because an empty string
#: already means "all artifacts", and Phase 1d introduces rows whose value is
#: delivered NOWHERE — a participant excluded from every artifact, or a value
#: suppressed below its confidence threshold. Recording those as "all" would be
#: a false claim; recording them as "" is indistinguishable from the default.
NONE = "(none)"


def join(artifacts) -> str:
    """Serialize an artifact list for the `authoritative_for` column.

    An empty selection serializes to NONE, not to "" — see the note above.
    """
    seen = [a for a in ALL_ARTIFACTS if a in set(artifacts)]   # stable order
    return SEP.join(seen) if seen else NONE


def parse(value: str) -> tuple[str, ...]:
    """Parse `authoritative_for`. Empty means 'all artifacts'; NONE means none."""
    if not value:
        return ALL_ARTIFACTS
    if value == NONE:
        return ()
    return tuple(a for a in value.split(SEP) if a)


def is_authoritative_for(value: str, artifact: str) -> bool:
    """Does a row with this `authoritative_for` speak for `artifact`?"""
    return artifact in parse(value)

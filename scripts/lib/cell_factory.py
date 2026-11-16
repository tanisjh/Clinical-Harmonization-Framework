"""Explicit Cell-v1 identity adaptation and construction.

The compatibility adapter alone reads legacy anchor keys. The factory consumes
explicit identities and source observations; it does not select scientific or
mapping policy, resolve sources, or infer schema row identities.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

from .provenance import Cell
from .replay_contract import encode_typed_raw
from .source_ref import SourceRef

RAW_VALUE_ABSENT = object()


@dataclass(frozen=True)
class CellIdentity:
    """Values for the four existing Cell-v1 compatibility identity slots."""

    trial: str
    participant: str
    sample: str
    collection_event: str


def anchor_str(value) -> str:
    """Preserve the legacy None/float-NaN-only anchor normalization."""
    if value is None:
        return ""
    if isinstance(value, float) and value != value:      # NaN
        return ""
    return str(value)


def compatibility_identity(
        trial, anchor_row, *,
        get_anchor_str: Callable[[], Callable[[Any], str]] = lambda: anchor_str
        ) -> CellIdentity:
    """Read only the existing aliases, with no inferred identity or fallbacks.

    The optional accessor preserves a legacy consumer's normalizer overrides
    and their lookup timing for each slot. Trial is passed through verbatim.
    """
    return CellIdentity(
        trial=trial,
        participant=get_anchor_str()(anchor_row["cimac_part_id"]),
        sample=get_anchor_str()(anchor_row["Cimac.id"]),
        collection_event=get_anchor_str()(anchor_row["Collection_Event"]))


@dataclass(frozen=True)
class _SourceMetadata:
    legacy_label: Any
    resolved_path: str
    source_sha256: str
    source_sheet: str
    source_ref_id: str


class CellFactory:
    """One consumer's Cell construction and legacy-label accounting."""

    def __init__(self):
        self.legacy_label_uses: int = 0

    def build(self, identity: CellIdentity, harmonized_field, value, confidence,
              source_file, source_column, source_row_idx, extraction_method,
              notes="", *, run_id="", spec_ref="",
              raw_value=RAW_VALUE_ABSENT, source_row_number=None, source_record_key=None,
              rule_id="", rule_version="", contributions="",
              authoritative_for="", derivation="", value_state="",
              mapping_id="", evidence_refs="", raw_evidence_recorded=True) -> Cell:
        """Build a Cell from explicit identity and current provenance values."""
        source = self._prepare_source(source_file)
        return self._build_prepared(
            identity, harmonized_field, value, confidence, source, source_column,
            source_row_idx, extraction_method, notes,
            run_id=lambda: run_id, spec_ref=lambda: spec_ref,
            raw_value=raw_value, source_row_number=source_row_number,
            source_record_key=source_record_key, rule_id=rule_id,
            rule_version=rule_version, contributions=contributions,
            authoritative_for=authoritative_for, derivation=derivation,
            value_state=value_state, mapping_id=mapping_id,
            evidence_refs=evidence_refs, raw_evidence_recorded=raw_evidence_recorded)

    def _prepare_source(self, source_file) -> _SourceMetadata:
        """Capture source metadata and account before any identity lookup."""
        if isinstance(source_file, SourceRef):
            ref = source_file
            legacy_label = ref.logical_label
            resolved_path = ref.resolved_path
            source_sha256 = ref.sha256
            source_sheet = ref.sheet or ""
            source_ref_id = ref.source_ref_id
        else:
            ref = None
            legacy_label = source_file
            resolved_path = ""
            source_sha256 = ""
            source_sheet = ""
            source_ref_id = ""
            self.legacy_label_uses += 1

        return _SourceMetadata(
            legacy_label, resolved_path, source_sha256, source_sheet, source_ref_id)

    def _build_prepared(
            self, identity: CellIdentity, harmonized_field, value, confidence,
            source: _SourceMetadata, source_column, source_row_idx,
            extraction_method, notes="", *,
            run_id: Callable[[], str], spec_ref: Callable[[], str],
            raw_value=RAW_VALUE_ABSENT, source_row_number=None, source_record_key=None,
            rule_id="", rule_version="", contributions="",
            authoritative_for="", derivation="", value_state="",
            mapping_id="", evidence_refs="", raw_evidence_recorded=True) -> Cell:
        """Finish after source accounting and compatibility identity adaptation.

        Context accessors keep the Base facade's run/spec reads at their
        historical position, after confidence/locator handling and before raw
        encoding. Public build wraps explicit context values in these accessors.
        """
        return Cell(
            trial=identity.trial,
            cimac_part_id=identity.participant,
            Cimac_id=identity.sample,
            Collection_Event=identity.collection_event,
            harmonized_field=harmonized_field,
            value=value,
            confidence=float(confidence),
            source_file=source.legacy_label,
            source_column=source_column,
            source_row_idx=source_row_idx,
            extraction_method=extraction_method,
            notes=notes,
            source_logical_label=source.legacy_label,
            source_resolved_path=source.resolved_path,
            source_sha256=source.source_sha256,
            source_sheet=source.source_sheet,
            source_row_number=("" if source_row_number is None else source_row_number),
            source_record_key=(source_record_key or ""),
            raw_value=("" if raw_value is RAW_VALUE_ABSENT or raw_value is None
                       else raw_value),
            run_id=run_id(),
            spec_ref=spec_ref(),
            mapping_id=mapping_id,
            source_ref_id=source.source_ref_id,
            raw_value_json=("" if raw_value is RAW_VALUE_ABSENT
                            or not raw_evidence_recorded else
                            encode_typed_raw(raw_value).to_json()),
            evidence_refs=evidence_refs,
            rule_id=rule_id,
            rule_version=rule_version,
            contributions=contributions,
            authoritative_for=authoritative_for,
            derivation=derivation,
            value_state=value_state,
        )

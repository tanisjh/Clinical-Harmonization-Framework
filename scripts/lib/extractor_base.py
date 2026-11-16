"""
extractor_base.py — Base class for per-trial extractors.

Each trial extractor subclasses BaseExtractor and implements `extract()`,
which yields Cell objects (one per (cimac_part_id, Cimac.id, Collection_Event, field) tuple).

The base class provides:
  - Path-resolving file accessors (with snapshot selection)
  - CIDC-aware CSV reading
  - Value normalization helpers
  - Standard yield helpers
"""
from __future__ import annotations

from pathlib import Path
from typing import Iterator

import pandas as pd

from .normalize import (
    derive_bor_binary,
    derive_bor_bin,
    derive_pfs_bin,
    normalize,
)
from .provenance import Cell
from .replay_contract import encode_typed_raw
from .source_ref import SourceRef, assert_within_roots
from .source_session import SourceSession
from .xlsx_reader import TrustedXlsxObservation

_RAW_VALUE_ABSENT = object()


class BaseExtractor:
    """Subclass per trial; override `extract()` to emit Cell objects."""

    trial_dir_name: str            # e.g. "EAY131-Z1D-clinical"
    template_trial_name: str       # e.g. "EAY131_Z1D"

    def __init__(self, project_root: Path, anchor_df: pd.DataFrame, trial_config: dict,
                 value_norms: dict, derived_rules: dict | None = None, *,
                 run_id: str = "", spec_ref: str = ""):
        """
        Args:
            project_root: filesystem root that contains *-clinical/ dirs
            anchor_df:    rows from the template CSV for this trial (the row anchors)
            trial_config: section of harmonization_config.yaml for this trial
            value_norms:  the global `value_normalizations` block
            derived_rules: the global `derived_rules` block (landmark parameters)
        """
        self._source_session = SourceSession(
            project_root, resolve_source=lambda key: self.src_path(key),
            context_name=lambda: self.trial_dir_name)
        self.trial_dir = self.project_root / self.trial_dir_name
        self.anchor = anchor_df.reset_index(drop=True)
        self.cfg = trial_config or {}
        self.value_norms = value_norms or {}
        self.derived_rules = derived_rules or {}
        self.run_id = run_id
        self.spec_ref = spec_ref
        # Counts cell() calls that still pass a bare string instead of a
        # SourceRef. Must reach 0 as extractors are migrated (plan §7 A6).
        self._legacy_label_uses: int = 0

    # ── source-file access ───────────────────────────────────────────────────
    def src_path(self, key: str) -> Path:
        """Resolve a key in cfg['sources'] to a Path.

        By default files resolve under the trial dir (``<root>/<trial_dir_name>``).
        A trial may set ``source_dir`` in its config (a path relative to
        ``project_root``) to read its source files from an alternate location —
        e.g. the CIDC cross-trial bucket ``cimac-cidc-x001/<trial>``. Trials
        without ``source_dir`` are unaffected.
        """
        rel = (self.cfg.get("sources") or {}).get(key)
        if rel is None:
            raise KeyError(f"No source '{key}' configured for trial {self.trial_dir_name}")
        source_dir = self.cfg.get("source_dir")
        base = (self.project_root / source_dir) if source_dir else self.trial_dir
        # Phase 1a (R3): reject any path that escapes an approved input root
        # before it is ever opened or recorded. There is deliberately NO
        # basename fallback — if the configured path is wrong we fail loudly
        # rather than opening a same-named file somewhere else (defect D2).
        return assert_within_roots(base / rel, self.project_root)

    def load_csv(self, key: str, logical_label: str | None = None) -> pd.DataFrame:
        """Read a configured CSV and record a SourceRef for it.

        Args:
            key:           logical input key in cfg['sources'].
            logical_label: stable label for config lookups and the existing
                           provenance `source_file` column. Defaults to the
                           basename actually opened. Repointed trials pass the
                           legacy basename here so `column_map` keys and
                           historical provenance values stay unchanged.
        """
        return self._source_session.load_csv(key, logical_label)

    def load_xlsx(self, key: str, sheet: str | int | None = None,
                  logical_label: str | None = None, *,
                  reader_id: str = "excel@v1") -> pd.DataFrame:
        return self._source_session.load_xlsx(
            key, sheet, logical_label, reader_id=reader_id)

    def ref(self, key: str) -> SourceRef:
        """SourceRef for a previously loaded logical input."""
        return self._source_session.ref(key)

    def source_refs(self) -> dict[str, SourceRef]:
        """All SourceRefs created during this extraction (for run manifests)."""
        return self._source_session.source_refs()

    # Live compatibility views: assignment replaces the session's state too.
    @property
    def project_root(self) -> Path:
        return self._source_session.project_root

    @project_root.setter
    def project_root(self, value: Path) -> None:
        self._source_session.project_root = value

    @property
    def _cache(self) -> dict[str, pd.DataFrame]:
        return self._source_session._cache

    @_cache.setter
    def _cache(self, value: dict[str, pd.DataFrame]) -> None:
        self._source_session._cache = value

    @property
    def _refs(self) -> dict[str, SourceRef]:
        return self._source_session._refs

    @_refs.setter
    def _refs(self, value: dict[str, SourceRef]) -> None:
        self._source_session._refs = value

    @property
    def _xlsx_observations(self) -> dict[str, TrustedXlsxObservation]:
        return self._source_session._xlsx_observations

    @_xlsx_observations.setter
    def _xlsx_observations(self, value: dict[str, TrustedXlsxObservation]) -> None:
        self._source_session._xlsx_observations = value

    # ── value-extraction helpers ────────────────────────────────────────────
    def get_norm_rules(self, harmonized_field: str) -> list | dict | None:
        return self.value_norms.get(harmonized_field)

    def normalize_value(self, harmonized_field: str, raw_value):
        rules = self.get_norm_rules(harmonized_field)
        rules_list = rules if isinstance(rules, list) else None
        return normalize(harmonized_field, raw_value, rules_list)

    # ── derived-field helpers ───────────────────────────────────────────────
    def derive_BOR_binary(self, bor_value):
        # Method name kept for internal callers; the YAML key + output column
        # were renamed BOR.binary → clinical_benefit.binary on 2026-05-20.
        rules = self.value_norms.get("clinical_benefit.binary")
        return derive_bor_binary(bor_value, rules if isinstance(rules, dict) else None)

    def derive_bor_bin(self, bor_binary, pfs_time=None):
        """120-day SD-landmark rule. Landmark pulled from derived_rules in YAML
        config (default 120). See scripts/lib/normalize.py for the rule."""
        landmark = float(self.derived_rules.get("bor_bin_landmark_days", 120))
        return derive_bor_bin(bor_binary, pfs_time, landmark_days=landmark)

    def derive_pfs_bin(self, pfs_time, pfs_stat=None):
        """120-day landmark rule. Landmark pulled from derived_rules in YAML
        config (default 120). See scripts/lib/normalize.py for the rule."""
        landmark = float(self.derived_rules.get("pfs_bin_landmark_days", 120))
        return derive_pfs_bin(pfs_time, pfs_stat, landmark_days=landmark)

    # ── Cell construction ────────────────────────────────────────────────────
    @staticmethod
    def _anchor_str(value) -> str:
        """Render an anchor component, mapping a null to the empty string.

        A plain `str(None)` yields the four-character string `"None"`, which then
        travels into the anchor key, the provenance `Cimac.id` column and the
        delivered CSV — where it reads as a value rather than an absence. Two
        trials (10013, 14C0059G) have no sample-level `Cimac.id`, so this was
        writing 218 literal `"None"` cells into `harmonized_11trials.csv` while
        every other null in the same file was an empty field.

        Empty is also the more detectable failure: a genuinely missing
        `cimac_part_id` becomes `""`, which the target schema rejects outright,
        whereas `"None"` would pass as a plausible seven-character-ish token.
        """
        if value is None:
            return ""
        if isinstance(value, float) and value != value:      # NaN
            return ""
        return str(value)

    def cell(self, anchor_row, harmonized_field, value, confidence, source_file,
             source_column, source_row_idx, extraction_method, notes="", *,
             raw_value=_RAW_VALUE_ABSENT, source_row_number=None, source_record_key=None,
             rule_id="", rule_version="", contributions="",
             authoritative_for="", derivation="", value_state="",
             mapping_id="", evidence_refs="", raw_evidence_recorded=True):
        """Build a Cell.

        `source_file` accepts either:
          * a `SourceRef` (migrated path) — the structured provenance fields
            (resolved path, sha256, sheet) are filled from the file that was
            actually opened, and the legacy `source_file` column keeps carrying
            `ref.logical_label` so existing consumers are unaffected; or
          * a plain string (not yet migrated, or a non-file source such as
            "CONFIG:trial_constants" / "DERIVED:BOR") — structured fields stay
            empty and the legacy-label counter is incremented.
        """
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
            self._legacy_label_uses += 1

        return Cell(
            trial=self.template_trial_name,
            cimac_part_id=self._anchor_str(anchor_row["cimac_part_id"]),
            Cimac_id=self._anchor_str(anchor_row["Cimac.id"]),
            Collection_Event=self._anchor_str(anchor_row["Collection_Event"]),
            harmonized_field=harmonized_field,
            value=value,
            confidence=float(confidence),
            source_file=legacy_label,
            source_column=source_column,
            source_row_idx=source_row_idx,
            extraction_method=extraction_method,
            notes=notes,
            source_logical_label=legacy_label,
            source_resolved_path=resolved_path,
            source_sha256=source_sha256,
            source_sheet=source_sheet,
            source_row_number=("" if source_row_number is None else source_row_number),
            source_record_key=(source_record_key or ""),
            raw_value=("" if raw_value is _RAW_VALUE_ABSENT or raw_value is None
                       else raw_value),
            run_id=self.run_id,
            spec_ref=self.spec_ref,
            mapping_id=mapping_id,
            source_ref_id=source_ref_id,
            raw_value_json=("" if raw_value is _RAW_VALUE_ABSENT
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

    # ── must override ────────────────────────────────────────────────────────
    def extract(self) -> Iterator[Cell]:
        raise NotImplementedError(f"{self.__class__.__name__}.extract() not implemented")

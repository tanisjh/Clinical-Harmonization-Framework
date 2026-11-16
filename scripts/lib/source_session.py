"""Per-extraction source observations, reader caches, and SourceRefs.

Resolution policy and diagnostic context belong to the caller. This component
retains the existing reader-specific cache and observation behavior; creating a
session does not resolve or open a source.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Callable

import pandas as pd

from .cidc_io import read_cidc_csv
from .source_ref import SourceRef
from .xlsx_reader import XLSX_READER_ID, TrustedXlsxObservation, load_xlsx_source

LOG = logging.getLogger("extractor")


class SourceSession:
    """One consumer's observed sources, with caller-supplied resolution."""

    def __init__(self, project_root: Path, *,
                 resolve_source: Callable[[str], Path],
                 context_name: Callable[[], str]):
        self.project_root = Path(project_root)
        self._resolve_source = resolve_source
        self._context_name = context_name
        # CSV and legacy Excel return the same mutable cached frame.
        self._cache: dict[str, pd.DataFrame] = {}
        # Trusted XLSX retains one checked observation per logical source key.
        self._xlsx_observations: dict[str, TrustedXlsxObservation] = {}
        self._refs: dict[str, SourceRef] = {}

    def resolve(self, key: str) -> Path:
        """Invoke the caller's current source-resolution policy."""
        return self._resolve_source(key)

    def load_csv(self, key: str, logical_label: str | None = None) -> pd.DataFrame:
        """Read a configured CSV and record a SourceRef for it.

        Args:
            key:           logical input key accepted by the resolver.
            logical_label: stable label for config lookups and the existing
                           provenance `source_file` column. Defaults to the
                           basename actually opened. Repointed trials pass the
                           legacy basename here so `column_map` keys and
                           historical provenance values stay unchanged.
        """
        path = self.resolve(key)
        ck = str(path)
        if ck not in self._cache:
            r = read_cidc_csv(path)
            if r.note:
                LOG.warning("CSV read note for %s: %s", path.name, r.note)
            self._cache[ck] = r.df
            self._refs[key] = SourceRef.build(
                config_key=key,
                path=path,
                project_root=self.project_root,
                file_type="csv",
                reader_id="cidc_csv@v1",
                logical_label=logical_label,
                reader_flags=f"preamble_skipped={r.preamble_skipped};encoding={r.encoding}",
                reader_effective_flags={
                    "encoding": r.encoding,
                    "header": 0,
                    "low_memory": False,
                    "skiprows": 1 if r.preamble_skipped else 0,
                },
                n_rows=len(r.df),
            )
        elif key not in self._refs and logical_label:
            # Same physical file reached under a second logical key. Reuse the
            # effective parser contract from the read that populated the cache.
            relative = str(path.relative_to(self.project_root.resolve()))
            prior = next(ref for ref in self._refs.values()
                         if ref.resolved_path == relative)
            self._refs[key] = SourceRef.build(
                config_key=key, path=path, project_root=self.project_root,
                file_type="csv", reader_id=prior.reader_id,
                logical_label=logical_label, reader_flags=prior.reader_flags,
                reader_effective_flags=dict(prior.reader_effective_flags),
                n_rows=len(self._cache[ck]),
            )
        return self._cache[ck]

    def load_xlsx(self, key: str, sheet: str | int | None = None,
                  logical_label: str | None = None, *,
                  reader_id: str = "excel@v1") -> pd.DataFrame:
        path = self.resolve(key)
        if reader_id == XLSX_READER_ID:
            if not isinstance(sheet, str) or not sheet:
                raise ValueError(
                    "xlsx_openpyxl@v1 requires a non-empty exact sheet name")
            observation = self._xlsx_observations.get(key)
            if observation is None:
                observation = load_xlsx_source(
                    path, project_root=self.project_root, config_key=key,
                    sheet_name=sheet, logical_label=logical_label)
                self._xlsx_observations[key] = observation
                self._refs[key] = observation.source_ref
            elif observation.source_ref.sheet != sheet:
                raise ValueError(
                    f"source {key!r} was already observed at sheet "
                    f"{observation.source_ref.sheet!r}, not {sheet!r}")
            return observation.frame_copy()
        ck = f"{path}::{sheet}"
        if ck not in self._cache:
            self._cache[ck] = pd.read_excel(path, sheet_name=sheet) if sheet is not None else pd.read_excel(path)
            self._refs[key] = SourceRef.build(
                config_key=key,
                path=path,
                project_root=self.project_root,
                file_type="xlsx",
                reader_id="excel@v1",
                logical_label=logical_label,
                sheet=str(sheet) if sheet is not None else None,
                n_rows=len(self._cache[ck]),
            )
        return self._cache[ck]

    def ref(self, key: str) -> SourceRef:
        """SourceRef for a previously loaded logical input."""
        if key not in self._refs:
            raise KeyError(
                f"No SourceRef for '{key}' in {self._context_name()}; "
                "call load_csv()/load_xlsx() first"
            )
        return self._refs[key]

    def source_refs(self) -> dict[str, SourceRef]:
        """All SourceRefs created during this extraction (for run manifests)."""
        return dict(self._refs)

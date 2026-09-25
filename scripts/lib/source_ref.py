"""source_ref.py — Reader-emitted identity of a source dataset.

Phase 1a. Motivation (see docs/design/phase1_provenance_traceability_plan_2026-08-03.md,
defects D1/D2): provenance used to record a filename that each extractor supplied
by hand, independently of the file the pipeline actually opened. After the CIDC
repoint those two drifted apart for nine trials, and because the legacy files are
still on disk with different content, a naive trace could open the WRONG file and
return a confident wrong answer.

The fix is structural: a `SourceRef` is created ONLY by the loader that actually
opened the file, and it carries the resolved path and content hash of that file.
Extractors never construct one by hand.

Two distinct name fields are deliberately kept:

  * ``logical_label`` — the stable key extractors and configuration use (the
    legacy basename, e.g. "Clinical Dataset 2023_03_14.csv"). ``column_map`` in
    harmonization_config.yaml is keyed by this, so it MUST NOT change. It is also
    what continues to populate the existing provenance ``source_file`` column, so
    downstream consumers keep working.
  * ``resolved_path`` / ``sha256`` — the truth about what was opened.

Keeping both is what lets Phase 1a fix provenance without touching
harmonization_config.yaml or any existing provenance column.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping


class PathContainmentError(ValueError):
    """A source path resolved outside every approved input root."""


def sha256_file(path: str | Path, chunk: int = 1 << 16) -> str:
    """SHA-256 of a file's bytes."""
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def approved_input_roots(project_root: str | Path) -> list[Path]:
    """Roots a source file is permitted to resolve under.

    Currently the repository root only. ``harmonization_outputs/`` is explicitly
    excluded by :func:`assert_within_roots` — the pipeline must never treat its
    own generated artifacts as source inputs.
    """
    return [Path(project_root).resolve()]


def assert_within_roots(path: str | Path, project_root: str | Path) -> Path:
    """Return the fully resolved path, or raise if it escapes an approved root.

    Resolves symlinks first, so a symlink pointing outside the repository is
    rejected rather than silently followed.
    """
    root = Path(project_root).resolve()
    resolved = Path(path).resolve()

    if not any(resolved.is_relative_to(r) for r in approved_input_roots(root)):
        raise PathContainmentError(
            f"source path resolves outside approved input roots:\n"
            f"  path : {resolved}\n"
            f"  roots: {[str(r) for r in approved_input_roots(root)]}"
        )

    outputs = (root / "harmonization_outputs").resolve()
    if resolved.is_relative_to(outputs):
        raise PathContainmentError(
            f"source path resolves inside harmonization_outputs/, which holds "
            f"generated artifacts and is never a source input:\n  path: {resolved}"
        )
    return resolved


@dataclass(frozen=True)
class SourceRef:
    """Identity of one source dataset, emitted by the reader that opened it.

    Constructed only by trusted repository readers or runtime input factories.
    """

    config_key: str            # logical input key in cfg['sources'], e.g. "clinical"
    logical_label: str         # stable label used by config column_map + provenance
    resolved_path: str         # repo-relative path ACTUALLY opened
    basename: str              # basename of the file actually opened
    sha256: str                # content hash at read time
    file_type: str             # "csv" | "xlsx"
    input_root: str            # approved root it resolved under (repo-relative or ".")
    reader_id: str             # e.g. "cidc_csv@v1", "excel@v1"
    sheet: str | None = None   # Excel sheet name; None for CSV
    reader_flags: str = ""     # e.g. "preamble_skipped=True;encoding=utf-8"
    n_rows: int = -1           # rows in the frame handed to the extractor
    source_ref_id: str = ""    # compact stable binding used by replay manifests
    reader_effective_flags: tuple[tuple[str, Any], ...] = ()
    parsed_columns: tuple[str, ...] = ()
    parsed_canonicalization: str = ""
    parsed_sheet_sha256: str = ""

    @classmethod
    def build(
        cls,
        *,
        config_key: str,
        path: str | Path,
        project_root: str | Path,
        file_type: str,
        reader_id: str,
        logical_label: str | None = None,
        sheet: str | None = None,
        reader_flags: str = "",
        reader_effective_flags: Mapping[str, Any] | None = None,
        n_rows: int = -1,
        parsed_columns: tuple[str, ...] = (),
        parsed_canonicalization: str = "",
        parsed_sheet_sha256: str = "",
    ) -> "SourceRef":
        """Build a ref for a file that has just been opened.

        Validates containment (R3) before hashing, so an out-of-root path fails
        loudly rather than being recorded.
        """
        root = Path(project_root).resolve()
        resolved = assert_within_roots(path, root)
        if not resolved.is_file():
            raise FileNotFoundError(f"source file does not exist: {resolved}")

        digest = sha256_file(resolved)
        return cls._from_observation(
            config_key=config_key,
            resolved=resolved,
            root=root,
            digest=digest,
            file_type=file_type,
            reader_id=reader_id,
            logical_label=logical_label,
            sheet=sheet,
            reader_flags=reader_flags,
            reader_effective_flags=reader_effective_flags,
            n_rows=n_rows,
            parsed_columns=parsed_columns,
            parsed_canonicalization=parsed_canonicalization,
            parsed_sheet_sha256=parsed_sheet_sha256,
        )

    @classmethod
    def from_observed_bytes(
        cls,
        *,
        config_key: str,
        path: str | Path,
        project_root: str | Path,
        payload: bytes,
        file_type: str,
        reader_id: str,
        logical_label: str | None = None,
        sheet: str | None = None,
        reader_flags: str = "",
        reader_effective_flags: Mapping[str, Any] | None = None,
        n_rows: int = -1,
        parsed_columns: tuple[str, ...] = (),
        parsed_canonicalization: str = "",
        parsed_sheet_sha256: str = "",
    ) -> "SourceRef":
        """Bind the exact captured bytes that a reader parsed.

        Unlike :meth:`build`, this constructor does not reopen the file to
        obtain its digest. It is for loaders that capture one immutable byte
        string and then both hash and parse that same object, eliminating a
        hash/read time-of-check/time-of-use gap.
        """
        if not isinstance(payload, bytes):
            raise TypeError("observed source payload must be bytes")
        root = Path(project_root).resolve()
        resolved = assert_within_roots(path, root)
        if not resolved.is_file():
            raise FileNotFoundError(f"source file does not exist: {resolved}")
        return cls._from_observation(
            config_key=config_key,
            resolved=resolved,
            root=root,
            digest=hashlib.sha256(payload).hexdigest(),
            file_type=file_type,
            reader_id=reader_id,
            logical_label=logical_label,
            sheet=sheet,
            reader_flags=reader_flags,
            reader_effective_flags=reader_effective_flags,
            n_rows=n_rows,
            parsed_columns=parsed_columns,
            parsed_canonicalization=parsed_canonicalization,
            parsed_sheet_sha256=parsed_sheet_sha256,
        )

    @classmethod
    def _from_observation(
        cls,
        *,
        config_key: str,
        resolved: Path,
        root: Path,
        digest: str,
        file_type: str,
        reader_id: str,
        logical_label: str | None,
        sheet: str | None,
        reader_flags: str,
        reader_effective_flags: Mapping[str, Any] | None,
        n_rows: int,
        parsed_columns: tuple[str, ...],
        parsed_canonicalization: str,
        parsed_sheet_sha256: str,
    ) -> "SourceRef":
        """Construct one identity from an already verified observation."""
        effective_flags = tuple(sorted(
            (str(key), value)
            for key, value in (reader_effective_flags or {}).items()
        ))
        identity = {
            "config_key": config_key,
            "resolved_path": str(resolved.relative_to(root)),
            "sha256": digest,
            "sheet": sheet,
            "reader_id": reader_id,
            "effective_flags": effective_flags,
        }
        if parsed_canonicalization or parsed_sheet_sha256 or parsed_columns:
            identity.update({
                "n_rows": int(n_rows),
                "parsed_columns": list(parsed_columns),
                "parsed_canonicalization": parsed_canonicalization,
                "parsed_sheet_sha256": parsed_sheet_sha256,
            })
        source_ref_id = "src-" + hashlib.sha256(
            json.dumps(identity, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()[:24]

        return cls(
            config_key=config_key,
            logical_label=logical_label or resolved.name,
            resolved_path=str(resolved.relative_to(root)),
            basename=resolved.name,
            sha256=digest,
            file_type=file_type,
            input_root=str(root.name),
            reader_id=reader_id,
            sheet=sheet,
            reader_flags=reader_flags,
            n_rows=int(n_rows),
            source_ref_id=source_ref_id,
            reader_effective_flags=effective_flags,
            parsed_columns=tuple(parsed_columns),
            parsed_canonicalization=parsed_canonicalization,
            parsed_sheet_sha256=parsed_sheet_sha256,
        )

    def verify(self, project_root: str | Path) -> tuple[bool, str]:
        """Re-read the file and confirm it still hashes to the recorded value.

        Returns ``(ok, detail)``. Never raises for a mismatch — callers decide
        whether a drift is fatal.
        """
        root = Path(project_root).resolve()
        p = root / self.resolved_path
        if not p.is_file():
            return False, f"missing: {self.resolved_path}"
        actual = sha256_file(p)
        if actual != self.sha256:
            return False, f"hash drift: recorded {self.sha256[:12]}… actual {actual[:12]}…"
        return True, "ok"

    def __str__(self) -> str:
        sheet = f"::{self.sheet}" if self.sheet else ""
        return f"{self.resolved_path}{sheet}@{self.sha256[:12]}"

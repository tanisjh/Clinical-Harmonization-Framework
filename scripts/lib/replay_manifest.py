"""Versioned run evidence manifest for future strict recomputation.

The manifest is a passive, generic contract.  It captures and validates
identities, links and hashes; it never reads provenance to recompute a value.
"""
from __future__ import annotations

import importlib
import json
import os
import platform
import re
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping

from .source_ref import SourceRef, sha256_file


MANIFEST_SCHEMA_ID = "harmonization/replay-manifest"
MANIFEST_VERSION = 2
SUPPORTED_MANIFEST_VERSIONS = frozenset({1, 2})
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_KINDS = frozenset({"configuration", "thresholds", "vocabulary", "anchor"})


class ManifestError(ValueError):
    """The replay manifest is malformed, inconsistent, or unsafe."""


class UnsupportedManifestVersion(ManifestError):
    pass


@dataclass(frozen=True)
class HashCheck:
    identity: str
    path: str
    expected: str
    actual: str

    @property
    def ok(self) -> bool:
        return bool(self.expected) and self.expected == self.actual


@dataclass(frozen=True)
class FileBinding:
    identity: str
    path: str
    sha256: str


@dataclass(frozen=True)
class TargetSchemaBinding(FileBinding):
    schema_id: str = ""
    schema_version: str = ""

    @property
    def ref(self) -> str:
        return f"{self.schema_id}@{self.schema_version}"


@dataclass(frozen=True)
class SpecificationBinding(FileBinding):
    spec_id: str = ""
    spec_version: str = ""
    status: str = "approved"
    target_schema_ref: str = ""
    vocabulary_refs: tuple[str, ...] = ()

    @property
    def ref(self) -> str:
        return f"{self.spec_id}@{self.spec_version}"


@dataclass(frozen=True)
class ProducerBinding:
    application: str
    application_version: str
    python_version: str
    dependencies: tuple[tuple[str, str], ...] = ()


@dataclass(frozen=True)
class ReaderBinding:
    reader_id: str
    effective_flags: tuple[tuple[str, Any], ...] = ()
    dependencies: tuple[tuple[str, str], ...] = ()

    @property
    def flags(self) -> dict[str, Any]:
        return dict(self.effective_flags)


@dataclass(frozen=True)
class SourceBinding(FileBinding):
    source_ref_id: str = ""
    spec_ref: str = ""
    source_id: str = ""
    config_key: str = ""
    logical_label: str = ""
    file_type: str = "csv"
    sheet: str = ""
    reader: ReaderBinding = field(default_factory=lambda: ReaderBinding(""))
    n_rows: int = -1
    parsed_columns: tuple[str, ...] = ()
    parsed_canonicalization: str = ""
    parsed_sheet_sha256: str = ""


@dataclass(frozen=True)
class ArtifactBinding(FileBinding):
    artifact_id: str = ""
    format: str = "csv"
    n_rows: int = -1
    schema_ref: str = ""
    provenance_schema_version: int = 0


@dataclass(frozen=True)
class EvidenceBinding(FileBinding):
    evidence_ref_id: str = ""
    kind: str = "configuration"
    reference: str = ""
    details: tuple[tuple[str, Any], ...] = ()


@dataclass(frozen=True)
class RuleBinding:
    rule_id: str
    rule_version: str
    kind: str
    parameters: tuple[tuple[str, Any], ...] = ()

    @property
    def ref(self) -> str:
        return f"{self.rule_id}@{self.rule_version}"


@dataclass(frozen=True)
class MappingUsage:
    mapping_id: str
    spec_ref: str
    source_ref_ids: tuple[str, ...] = ()
    rule_refs: tuple[str, ...] = ()
    evidence_ref_ids: tuple[str, ...] = ()


@dataclass(frozen=True)
class ReplayManifest:
    run_id: str
    created_at_utc: str
    producer: ProducerBinding
    target_schema: TargetSchemaBinding
    specifications: tuple[SpecificationBinding, ...]
    sources: tuple[SourceBinding, ...]
    delivered_artifacts: tuple[ArtifactBinding, ...]
    provenance_artifact: ArtifactBinding
    rules: tuple[RuleBinding, ...] = ()
    evidence: tuple[EvidenceBinding, ...] = ()
    mapping_usages: tuple[MappingUsage, ...] = ()
    manifest_schema: str = MANIFEST_SCHEMA_ID
    manifest_version: int = MANIFEST_VERSION

    def __post_init__(self) -> None:
        if self.manifest_schema != MANIFEST_SCHEMA_ID:
            raise ManifestError(f"unsupported manifest schema {self.manifest_schema!r}")
        if self.manifest_version not in SUPPORTED_MANIFEST_VERSIONS:
            raise UnsupportedManifestVersion(
                f"manifest version {self.manifest_version!r} is unsupported; "
                f"supported versions are {sorted(SUPPORTED_MANIFEST_VERSIONS)}")
        if not self.run_id or not self.created_at_utc:
            raise ManifestError("run_id and created_at_utc are required")
        _unique("specification ref", [s.ref for s in self.specifications])
        _unique("source_ref_id", [s.source_ref_id for s in self.sources])
        _unique("artifact_id", [a.artifact_id for a in self.delivered_artifacts])
        _unique("evidence_ref_id", [e.evidence_ref_id for e in self.evidence])
        _unique("mapping usage key",
                [f"{m.spec_ref}::{m.mapping_id}" for m in self.mapping_usages])
        _unique("rule ref", [r.ref for r in self.rules])
        if self.manifest_version == 1 and any(
                source.parsed_columns or source.parsed_canonicalization
                or source.parsed_sheet_sha256 for source in self.sources):
            raise ManifestError(
                "manifest v1 cannot carry parsed-sheet source identity")
        self._validate_links()

    def _validate_links(self) -> None:
        specs = {s.ref for s in self.specifications}
        sources = {s.source_ref_id for s in self.sources}
        rules = {r.ref for r in self.rules}
        evidence = {e.evidence_ref_id for e in self.evidence}
        if not self.provenance_artifact.provenance_schema_version:
            raise ManifestError("provenance artifact needs provenance_schema_version")
        for spec in self.specifications:
            if spec.status != "approved":
                raise ManifestError(f"specification {spec.ref!r} is not approved")
            if spec.target_schema_ref != self.target_schema.ref:
                raise ManifestError(
                    f"specification {spec.ref!r} targets {spec.target_schema_ref!r}, "
                    f"not {self.target_schema.ref!r}")
        for artifact in self.delivered_artifacts:
            if artifact.schema_ref != self.target_schema.ref:
                raise ManifestError(
                    f"artifact {artifact.artifact_id!r} has schema_ref "
                    f"{artifact.schema_ref!r}, not {self.target_schema.ref!r}")
        for source in self.sources:
            if source.spec_ref not in specs:
                raise ManifestError(
                    f"source {source.source_ref_id!r} names unknown spec {source.spec_ref!r}")
            if not source.reader.reader_id:
                raise ManifestError(f"source {source.source_ref_id!r} has no reader_id")
            if not source.reader.dependencies or any(
                    not version or version in {"unknown", "absent"}
                    for _, version in source.reader.dependencies):
                raise ManifestError(
                    f"source {source.source_ref_id!r} needs exact reader dependency versions")
            if source.n_rows < 0:
                raise ManifestError(f"source {source.source_ref_id!r} has invalid n_rows")
            if source.reader.reader_id == "xlsx_openpyxl@v1":
                if self.manifest_version < 2:
                    raise ManifestError(
                        "xlsx_openpyxl@v1 requires manifest v2 parsed-sheet identity")
                if (not source.sheet or not source.parsed_columns
                        or source.parsed_canonicalization != "xlsx_sheet_frame@v1"
                        or not _SHA256.fullmatch(source.parsed_sheet_sha256)):
                    raise ManifestError(
                        f"source {source.source_ref_id!r} has incomplete XLSX "
                        "parsed-sheet identity")
        for usage in self.mapping_usages:
            if usage.spec_ref not in specs:
                raise ManifestError(
                    f"mapping {usage.mapping_id!r} names unknown spec {usage.spec_ref!r}")
            _require_subset("source", usage.mapping_id, usage.source_ref_ids, sources)
            _require_subset("rule", usage.mapping_id, usage.rule_refs, rules)
            _require_subset("evidence", usage.mapping_id,
                            usage.evidence_ref_ids, evidence)

    def to_dict(self) -> dict[str, Any]:
        return {
            "manifest_schema": self.manifest_schema,
            "manifest_version": self.manifest_version,
            "run_id": self.run_id,
            "created_at_utc": self.created_at_utc,
            "producer": _producer_dict(self.producer),
            "target_schema": _schema_dict(self.target_schema),
            "specifications": [_spec_dict(x) for x in self.specifications],
            "sources": [_source_dict(x, self.manifest_version)
                        for x in self.sources],
            "delivered_artifacts": [_artifact_dict(x)
                                    for x in self.delivered_artifacts],
            "provenance_artifact": _artifact_dict(self.provenance_artifact),
            "rules": [_rule_dict(x) for x in self.rules],
            "evidence": [_evidence_dict(x) for x in self.evidence],
            "mapping_usages": [_usage_dict(x) for x in self.mapping_usages],
        }

    def write(self, path: str | Path) -> Path:
        p = Path(path)
        with p.open("w") as handle:
            json.dump(self.to_dict(), handle, indent=2, sort_keys=True)
            handle.write("\n")
        return p

    def write_atomic(self, path: str | Path) -> Path:
        """Publish this manifest atomically without replacing an existing one.

        The temporary file is created beside the destination, so no shared
        temporary directory or cross-filesystem rename is involved.  A hard
        link is the commit operation: it is atomic and fails if the completed
        manifest name already exists.
        """
        p = Path(path)
        if not p.parent.is_dir():
            raise ManifestError(f"manifest directory does not exist: {p.parent}")
        if p.exists():
            raise ManifestError(f"completed replay manifest already exists: {p}")
        fd, temporary = tempfile.mkstemp(
            prefix=f".{p.name}.", suffix=".staging", dir=str(p.parent))
        staged = Path(temporary)
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as handle:
                json.dump(self.to_dict(), handle, indent=2, sort_keys=True)
                handle.write("\n")
                handle.flush()
                os.fsync(handle.fileno())
            try:
                os.link(staged, p)
            except FileExistsError as exc:
                raise ManifestError(
                    f"completed replay manifest already exists: {p}") from exc
            _fsync_directory(p.parent)
            return p
        finally:
            try:
                staged.unlink()
            except FileNotFoundError:
                pass

    def files_for(self, *, mapping_id: str = "", spec_ref: str = "",
                  artifact_id: str = ""
                  ) -> tuple[FileBinding, ...]:
        """Files relevant to a run or one mapping usage.

        With a mapping ID, unrelated sources/evidence/specifications are
        intentionally excluded. Schema, provenance and the selected delivered
        artifact remain run/cell essentials.
        """
        files: list[FileBinding] = [self.target_schema, self.provenance_artifact]
        delivered = self.delivered_artifacts
        if artifact_id:
            delivered = tuple(a for a in delivered if a.artifact_id == artifact_id)
            if not delivered:
                raise ManifestError(f"unknown artifact_id {artifact_id!r}")
        files.extend(delivered)
        if not mapping_id:
            files.extend(self.specifications)
            files.extend(self.sources)
            files.extend(e for e in self.evidence if e.path)
            return _dedupe_files(files)
        matches = [u for u in self.mapping_usages
                   if u.mapping_id == mapping_id
                   and (not spec_ref or u.spec_ref == spec_ref)]
        if not matches:
            raise ManifestError(f"unknown mapping usage {spec_ref!r}:{mapping_id!r}")
        if len(matches) > 1:
            raise ManifestError(
                f"mapping_id {mapping_id!r} is ambiguous without spec_ref")
        usage = matches[0]
        files.extend(s for s in self.specifications if s.ref == usage.spec_ref)
        files.extend(s for s in self.sources if s.source_ref_id in usage.source_ref_ids)
        files.extend(e for e in self.evidence
                     if e.evidence_ref_id in usage.evidence_ref_ids and e.path)
        return _dedupe_files(files)

    def verify_hashes(self, project_root: str | Path, *, mapping_id: str = "",
                      spec_ref: str = "", artifact_id: str = "") -> tuple[HashCheck, ...]:
        root = Path(project_root).resolve()
        out = []
        for binding in self.files_for(mapping_id=mapping_id, spec_ref=spec_ref,
                                      artifact_id=artifact_id):
            path = resolve_project_path(root, binding.path)
            actual = sha256_file(path) if path.is_file() else ""
            out.append(HashCheck(binding.identity, binding.path,
                                 binding.sha256, actual))
        return tuple(out)


def producer_binding(application: str, application_version: str, names: Iterable[str] = (
        "pandas", "numpy", "yaml")) -> ProducerBinding:
    return ProducerBinding(
        application=application, application_version=application_version,
        python_version=platform.python_version(),
        dependencies=tuple(sorted(runtime_dependency_versions(names).items())),
    )


def runtime_dependency_versions(names: Iterable[str] = (
        "pandas", "numpy", "yaml", "openpyxl")) -> dict[str, str]:
    out: dict[str, str] = {}
    for name in names:
        try:
            module = importlib.import_module(name)
            out[name] = str(getattr(module, "__version__", "unknown"))
        except ImportError:
            out[name] = "absent"
    return out


def source_binding_from_ref(ref: SourceRef, *, spec_ref: str, source_id: str,
                            dependencies: Mapping[str, str]) -> SourceBinding:
    return SourceBinding(
        identity=f"source:{ref.source_ref_id}", path=ref.resolved_path,
        sha256=ref.sha256, source_ref_id=ref.source_ref_id,
        spec_ref=spec_ref, source_id=source_id, config_key=ref.config_key,
        logical_label=ref.logical_label, file_type=ref.file_type,
        sheet=ref.sheet or "", n_rows=ref.n_rows,
        parsed_columns=tuple(ref.parsed_columns),
        parsed_canonicalization=ref.parsed_canonicalization,
        parsed_sheet_sha256=ref.parsed_sheet_sha256,
        reader=ReaderBinding(
            ref.reader_id, tuple(ref.reader_effective_flags),
            tuple(sorted((str(k), str(v)) for k, v in dependencies.items()))),
    )


def capture_file(project_root: str | Path, path: str | Path, *,
                 identity: str) -> FileBinding:
    root = Path(project_root).resolve()
    resolved = resolve_project_path(root, path)
    if not resolved.is_file():
        raise ManifestError(f"manifest file does not exist: {resolved}")
    return FileBinding(identity, str(resolved.relative_to(root)),
                       sha256_file(resolved))


def resolve_project_path(project_root: str | Path, path: str | Path) -> Path:
    root = Path(project_root).resolve()
    raw = Path(path)
    if raw.is_absolute():
        raise ManifestError(f"manifest paths must be project-root-relative: {path}")
    resolved = (root / raw).resolve()
    if not resolved.is_relative_to(root):
        raise ManifestError(f"manifest path escapes project root: {path}")
    return resolved


def load(path: str | Path, *, project_root: str | Path) -> ReplayManifest:
    manifest_path = Path(path)
    if not manifest_path.is_absolute():
        manifest_path = resolve_project_path(project_root, manifest_path)
    else:
        root = Path(project_root).resolve()
        manifest_path = manifest_path.resolve()
        if not manifest_path.is_relative_to(root):
            raise ManifestError("manifest path escapes project root")
    try:
        doc = json.loads(manifest_path.read_text())
    except FileNotFoundError as exc:
        raise ManifestError(f"manifest not found: {manifest_path}") from exc
    except json.JSONDecodeError as exc:
        raise ManifestError(f"manifest is not valid JSON: {exc}") from exc
    return parse(doc, project_root=project_root)


def parse(doc: Any, *, project_root: str | Path) -> ReplayManifest:
    if not isinstance(doc, dict):
        raise ManifestError("manifest must be a JSON object")
    version = doc.get("manifest_version")
    if version not in SUPPORTED_MANIFEST_VERSIONS:
        raise UnsupportedManifestVersion(
            f"manifest version {version!r} is unsupported; supported versions "
            f"are {sorted(SUPPORTED_MANIFEST_VERSIONS)}")
    required = {"manifest_schema", "manifest_version", "run_id",
                "created_at_utc", "producer", "target_schema",
                "specifications", "sources", "delivered_artifacts",
                "provenance_artifact", "rules", "evidence", "mapping_usages"}
    missing = sorted(required - set(doc))
    unknown = sorted(set(doc) - required)
    if missing or unknown:
        raise ManifestError(f"manifest keys invalid; missing={missing}, unknown={unknown}")
    root = Path(project_root).resolve()
    manifest = ReplayManifest(
        manifest_schema=_text(doc, "manifest_schema"),
        manifest_version=version,
        run_id=_text(doc, "run_id"),
        created_at_utc=_text(doc, "created_at_utc"),
        producer=_parse_producer(doc.get("producer")),
        target_schema=_parse_schema(doc["target_schema"]),
        specifications=tuple(_parse_spec(x) for x in _list(doc, "specifications")),
        sources=tuple(_parse_source(x, version) for x in _list(doc, "sources")),
        delivered_artifacts=tuple(_parse_artifact(x) for x in
                                  _list(doc, "delivered_artifacts")),
        provenance_artifact=_parse_artifact(doc["provenance_artifact"]),
        rules=tuple(_parse_rule(x) for x in _list(doc, "rules")),
        evidence=tuple(_parse_evidence(x) for x in _list(doc, "evidence")),
        mapping_usages=tuple(_parse_usage(x) for x in _list(doc, "mapping_usages")),
    )
    for binding in manifest.files_for():
        resolve_project_path(root, binding.path)
        _valid_hash(binding.sha256, binding.identity)
    return manifest


def _parse_producer(d: dict) -> ProducerBinding:
    _keys(d, {"application", "application_version", "python_version",
              "dependencies"}, "producer")
    dependencies = _freeze_mapping(d.get("dependencies"), "producer.dependencies")
    if not dependencies or any(not version or version in {"unknown", "absent"}
                               for _, version in dependencies):
        raise ManifestError("producer needs exact runtime dependency versions")
    return ProducerBinding(_text(d, "application"),
                           _text(d, "application_version"),
                           _text(d, "python_version"), dependencies)


def _parse_schema(d: dict) -> TargetSchemaBinding:
    _keys(d, {"identity", "path", "sha256", "schema_id", "schema_version"}, "target_schema")
    return TargetSchemaBinding(_text(d, "identity"), _text(d, "path"),
                               _hash(d), _text(d, "schema_id"),
                               _text(d, "schema_version"))


def _parse_spec(d: dict) -> SpecificationBinding:
    _keys(d, {"identity", "path", "sha256", "spec_id", "spec_version",
              "status", "target_schema_ref", "vocabulary_refs"}, "specification")
    return SpecificationBinding(
        _text(d, "identity"), _text(d, "path"), _hash(d),
        _text(d, "spec_id"), _text(d, "spec_version"), _text(d, "status"),
        _text(d, "target_schema_ref"), tuple(str(x) for x in d["vocabulary_refs"]))


def _parse_source(d: dict, manifest_version: int) -> SourceBinding:
    keys = {"identity", "path", "sha256", "source_ref_id", "spec_ref",
            "source_id", "config_key", "logical_label", "file_type", "sheet",
            "reader", "n_rows"}
    if manifest_version >= 2:
        keys.update({"parsed_columns", "parsed_canonicalization",
                     "parsed_sheet_sha256"})
    _keys(d, keys, "source")
    reader = d.get("reader")
    _keys(reader, {"reader_id", "effective_flags", "dependencies"}, "reader")
    return SourceBinding(
        _text(d, "identity"), _text(d, "path"), _hash(d),
        _text(d, "source_ref_id"), _text(d, "spec_ref"), _text(d, "source_id"),
        _text(d, "config_key"), _text(d, "logical_label"),
        _text(d, "file_type"), str(d.get("sheet") or ""),
        ReaderBinding(_text(reader, "reader_id"),
                      _freeze_mapping(reader.get("effective_flags"), "effective_flags"),
                      _freeze_mapping(reader.get("dependencies"), "dependencies")),
        _integer(d, "n_rows"),
        tuple(str(value) for value in d.get("parsed_columns", ())),
        str(d.get("parsed_canonicalization") or ""),
        str(d.get("parsed_sheet_sha256") or ""))


def _parse_artifact(d: dict) -> ArtifactBinding:
    _keys(d, {"identity", "path", "sha256", "artifact_id", "format", "n_rows",
              "schema_ref", "provenance_schema_version"}, "artifact")
    return ArtifactBinding(
        _text(d, "identity"), _text(d, "path"), _hash(d),
        _text(d, "artifact_id"), _text(d, "format"), _integer(d, "n_rows"),
        str(d.get("schema_ref") or ""), int(d.get("provenance_schema_version") or 0))


def _parse_evidence(d: dict) -> EvidenceBinding:
    _keys(d, {"identity", "path", "sha256", "evidence_ref_id", "kind",
              "reference", "details"}, "evidence")
    kind = _text(d, "kind")
    if kind not in _KINDS:
        raise ManifestError(f"unknown evidence kind {kind!r}")
    path = str(d.get("path") or "")
    digest = str(d.get("sha256") or "")
    if bool(path) != bool(digest):
        raise ManifestError("evidence path and sha256 must be supplied together")
    return EvidenceBinding(_text(d, "identity"), path, digest,
                           _text(d, "evidence_ref_id"), kind,
                           str(d.get("reference") or ""),
                           _freeze_mapping(d.get("details"), "details"))


def _parse_rule(d: dict) -> RuleBinding:
    _keys(d, {"rule_id", "rule_version", "kind", "parameters"}, "rule")
    return RuleBinding(_text(d, "rule_id"), _text(d, "rule_version"),
                       _text(d, "kind"),
                       _freeze_mapping(d.get("parameters"), "parameters"))


def _parse_usage(d: dict) -> MappingUsage:
    _keys(d, {"mapping_id", "spec_ref", "source_ref_ids", "rule_refs",
              "evidence_ref_ids"}, "mapping_usage")
    return MappingUsage(_text(d, "mapping_id"), _text(d, "spec_ref"),
                        tuple(str(x) for x in d["source_ref_ids"]),
                        tuple(str(x) for x in d["rule_refs"]),
                        tuple(str(x) for x in d["evidence_ref_ids"]))


def _producer_dict(x):
    return {"application": x.application,
            "application_version": x.application_version,
            "python_version": x.python_version,
            "dependencies": dict(x.dependencies)}


def _schema_dict(x):
    return {"identity": x.identity, "path": x.path, "sha256": x.sha256,
            "schema_id": x.schema_id, "schema_version": x.schema_version}


def _spec_dict(x):
    return {"identity": x.identity, "path": x.path, "sha256": x.sha256,
            "spec_id": x.spec_id, "spec_version": x.spec_version,
            "status": x.status, "target_schema_ref": x.target_schema_ref,
            "vocabulary_refs": list(x.vocabulary_refs)}


def _source_dict(x, manifest_version):
    document = {"identity": x.identity, "path": x.path, "sha256": x.sha256,
            "source_ref_id": x.source_ref_id, "spec_ref": x.spec_ref,
            "source_id": x.source_id, "config_key": x.config_key,
            "logical_label": x.logical_label, "file_type": x.file_type,
            "sheet": x.sheet, "n_rows": x.n_rows,
            "reader": {"reader_id": x.reader.reader_id,
                       "effective_flags": _thaw(dict(x.reader.effective_flags)),
                       "dependencies": dict(x.reader.dependencies)}}
    if manifest_version >= 2:
        document.update({
            "parsed_columns": list(x.parsed_columns),
            "parsed_canonicalization": x.parsed_canonicalization,
            "parsed_sheet_sha256": x.parsed_sheet_sha256,
        })
    return document


def _artifact_dict(x):
    return {"identity": x.identity, "path": x.path, "sha256": x.sha256,
            "artifact_id": x.artifact_id, "format": x.format,
            "n_rows": x.n_rows, "schema_ref": x.schema_ref,
            "provenance_schema_version": x.provenance_schema_version}


def _evidence_dict(x):
    return {"identity": x.identity, "path": x.path, "sha256": x.sha256,
            "evidence_ref_id": x.evidence_ref_id, "kind": x.kind,
            "reference": x.reference, "details": _thaw(dict(x.details))}


def _rule_dict(x):
    return {"rule_id": x.rule_id, "rule_version": x.rule_version,
            "kind": x.kind, "parameters": _thaw(dict(x.parameters))}


def _usage_dict(x):
    return {"mapping_id": x.mapping_id, "spec_ref": x.spec_ref,
            "source_ref_ids": list(x.source_ref_ids),
            "rule_refs": list(x.rule_refs),
            "evidence_ref_ids": list(x.evidence_ref_ids)}


def _dedupe_files(items):
    out, seen = [], set()
    for x in items:
        key = (x.identity, x.path)
        if key not in seen:
            out.append(x)
            seen.add(key)
    return tuple(out)


def _unique(label, values):
    if any(not v for v in values):
        raise ManifestError(f"{label} cannot be empty")
    dupes = sorted({v for v in values if values.count(v) > 1})
    if dupes:
        raise ManifestError(f"duplicate {label}: {dupes}")


def _require_subset(kind, mapping_id, values, known):
    missing = sorted(set(values) - set(known))
    if missing:
        raise ManifestError(f"mapping {mapping_id!r} names unknown {kind}(s) {missing}")


def _freeze_mapping(value, where):
    if value is None:
        return ()
    if not isinstance(value, dict):
        raise ManifestError(f"{where} must be an object")
    return tuple(sorted((str(k), _freeze(v)) for k, v in value.items()))


def _freeze(value):
    if isinstance(value, dict):
        return tuple(sorted((str(k), _freeze(v)) for k, v in value.items()))
    if isinstance(value, list):
        return tuple(_freeze(v) for v in value)
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    raise ManifestError(f"manifest value is not JSON-compatible: {value!r}")


def _thaw(value):
    if isinstance(value, tuple):
        if all(isinstance(x, tuple) and len(x) == 2 for x in value):
            return {str(k): _thaw(v) for k, v in value}
        return [_thaw(v) for v in value]
    if isinstance(value, dict):
        return {str(k): _thaw(v) for k, v in value.items()}
    return value


def _keys(d, required, where):
    if not isinstance(d, dict):
        raise ManifestError(f"{where} must be an object")
    missing = sorted(required - set(d))
    unknown = sorted(set(d) - required)
    if missing or unknown:
        raise ManifestError(f"{where} keys invalid; missing={missing}, unknown={unknown}")


def _text(d, key):
    value = d.get(key) if isinstance(d, dict) else None
    if not isinstance(value, str) or not value:
        raise ManifestError(f"{key} must be a non-empty string")
    return value


def _integer(d, key):
    value = d.get(key)
    if not isinstance(value, int) or isinstance(value, bool):
        raise ManifestError(f"{key} must be an integer")
    return value


def _hash(d):
    value = _text(d, "sha256")
    _valid_hash(value, d.get("identity", "file"))
    return value


def _valid_hash(value, identity):
    if not _SHA256.match(value or ""):
        raise ManifestError(f"{identity}: sha256 must be 64 lowercase hex characters")


def _list(d, key):
    value = d.get(key)
    if not isinstance(value, list):
        raise ManifestError(f"{key} must be a list")
    return value


def _fsync_directory(path: Path) -> None:
    """Make a just-published directory entry durable where supported."""
    flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
    fd = os.open(str(path), flags)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)

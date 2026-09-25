"""Production orchestration for one strict replay evidence bundle.

This module owns run lifecycle and publication.  The replay manifest remains a
passive trust contract, and the replay engine remains a reader of that contract.
No source discovery occurs here: source bindings come only from ``SourceRef``
objects returned by the readers during the harmonization execution.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping

import yaml

from . import rule_registry, vocabularies
from .anchor_evidence import (
    RuntimeExternalAnchorEvidence,
    RuntimeGroupingAnchorEvidence,
    is_external_anchor_evidence_mapping,
    is_grouping_anchor_direct_evidence_mapping,
)
from .contributions import (
    STATUS_CONDITION,
    STATUS_REJECTED,
    STATUS_INPUT,
    STATUS_NOT_EVALUATED,
    STATUS_SELECTED,
    STATUS_UNAVAILABLE,
    from_json as contributions_from_json,
)
from .confidence_policy import (
    ConfidencePolicyError,
    OUTCOME_NO_VALUE,
    finalize_registered_rule_confidence,
    validate_registered_rule_confidence,
)
from .derivation import (
    derivation_references,
    is_stage16_derivation,
    is_stage17_derivation,
    is_stage19_derivation,
    matches_derivation_template,
)
from .group_selection import (
    group_rule_id,
    is_closed_source_group_mapping,
    is_stage17_source_group_mapping,
    membership_role,
    scalar_role_for_mapping,
    select_first_containing_record,
    source_group_for_mapping,
)
from .provenance import (
    Cell,
    PROVENANCE_SCHEMA_VERSION,
    canonical_row_identity,
    is_provenance_event_ref,
    provenance_event_ref,
)
from .raw_policy import (
    COMPLETION_FALLBACK_EXHAUSTED,
    COMPLETION_VALUE,
    raw_from_is_causal,
)
from .record_construction import mapping_uses_record_construction
from .replay_contract import TypedRawValue, encode_typed_raw
from .replay_manifest import (
    ArtifactBinding,
    EvidenceBinding,
    ManifestError,
    MappingUsage,
    ProducerBinding,
    ReplayManifest,
    RuleBinding,
    SourceBinding,
    SpecificationBinding,
    TargetSchemaBinding,
    parse as parse_manifest,
    producer_binding,
    runtime_dependency_versions,
    source_binding_from_ref,
)
from .source_ref import SourceRef, sha256_file
from .spec_model import (
    ANCHOR_SOURCE_ID,
    GROUP_REF_SEP,
    OPERATIONS,
    OP_CUSTOM_RULE,
    OP_CONDITIONAL,
    OP_CONSTANT,
    OP_DIRECT,
    OP_FALLBACK,
    OP_FILE_MEMBERSHIP,
    OP_NOT_APPLICABLE,
    OP_NO_MAPPING_CONFIGURED,
    OP_UNIT_CONVERSION,
    OP_VALUE_MAP,
)
from .value_states import finalize_declared_value_state


DEFAULT_MANIFEST_NAME = "replay_manifest.json"
_RUN_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:-]{0,127}$")
_GENERATED_RUN_ID_RE = re.compile(
    r"^run-\d{8}T\d{6}Z-[0-9a-f]{32}$")
_READER_DEPENDENCIES = {
    "cidc_csv@v1": ("pandas",),
    "external_anchor_csv@v1": ("pandas", "numpy"),
    "excel@v1": ("pandas", "openpyxl"),
    "xlsx_openpyxl@v1": ("pandas", "openpyxl", "numpy"),
}
_SPEC_MAPPING_METHODS = frozenset(f"spec:{op}" for op in OPERATIONS)


class ReplayOrchestrationError(RuntimeError):
    """A run cannot be represented or safely published for strict replay."""


@dataclass(frozen=True)
class LoadedYamlEvidence:
    """One YAML input and the identity of the exact bytes that were parsed."""

    path: str
    sha256: str
    document: Any

    def binding(self, *, evidence_ref_id: str, kind: str,
                reference: str = "",
                details: Mapping[str, Any] | None = None) -> EvidenceBinding:
        return EvidenceBinding(
            identity=f"{kind}:{evidence_ref_id}",
            path=self.path,
            sha256=self.sha256,
            evidence_ref_id=evidence_ref_id,
            kind=kind,
            reference=reference,
            details=tuple(sorted((details or {}).items())),
        )


def load_yaml_evidence(project_root: str | Path,
                       path: str | Path) -> LoadedYamlEvidence:
    """Load a YAML input once and retain the identity of the parsed bytes."""
    root = Path(project_root).resolve()
    resolved = _contained_path(root, path, "YAML evidence")
    if not resolved.is_file():
        raise ReplayOrchestrationError(
            f"YAML evidence file does not exist: {resolved}")
    payload = resolved.read_bytes()
    try:
        document = yaml.safe_load(payload)
    except yaml.YAMLError as exc:
        raise ReplayOrchestrationError(
            f"YAML evidence is invalid: {resolved}: {exc}") from exc
    return LoadedYamlEvidence(
        path=str(resolved.relative_to(root)),
        sha256=hashlib.sha256(payload).hexdigest(),
        document=document,
    )


def vocabulary_evidence_binding(project_root: str | Path,
                                vocabulary) -> EvidenceBinding:
    """Bind one actual runtime-loaded Vocabulary without reopening its file."""
    root = Path(project_root).resolve()
    if not isinstance(vocabulary, vocabularies.Vocabulary):
        raise ReplayOrchestrationError(
            "runtime vocabulary evidence must be a loaded Vocabulary")
    try:
        path = vocabulary.contained_path(root)
        evidence_ref_id = vocabulary.evidence_ref_id(root)
    except vocabularies.VocabularyError as exc:
        raise ReplayOrchestrationError(str(exc)) from exc
    digest = _loaded_hash(
        vocabulary.source_sha256, f"vocabulary {vocabulary.ref}")
    if (vocabulary.kind not in vocabularies.KINDS
            or not vocabulary.parser_id or not vocabulary.parser_version):
        raise ReplayOrchestrationError(
            f"loaded vocabulary {vocabulary.ref!r} has incomplete runtime "
            "identity")
    return EvidenceBinding(
        identity=f"vocabulary:{evidence_ref_id}",
        path=path,
        sha256=digest,
        evidence_ref_id=evidence_ref_id,
        kind="vocabulary",
        reference=vocabulary.ref,
        details=tuple(sorted(vocabulary.evidence_details().items())),
    )


def anchor_evidence_binding(
        anchor: RuntimeGroupingAnchorEvidence | RuntimeExternalAnchorEvidence,
        producer: ProducerBinding) -> EvidenceBinding:
    """Bind actual anchor construction evidence without rediscovery."""
    if not isinstance(anchor, (
            RuntimeGroupingAnchorEvidence, RuntimeExternalAnchorEvidence)):
        raise ReplayOrchestrationError(
            "runtime anchor evidence must be a captured trusted anchor")
    details = dict(anchor.details)
    details["environment"] = {
        "python": producer.python_version,
        "dependencies": dict(producer.dependencies),
    }
    return EvidenceBinding(
        identity=f"anchor:{anchor.evidence_ref_id}", path="", sha256="",
        evidence_ref_id=anchor.evidence_ref_id, kind="anchor",
        reference=anchor.reference, details=tuple(sorted(details.items())),
    )


def generate_run_id() -> str:
    """Return a unique, sortable run ID that is not a scientific input."""
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return f"run-{stamp}-{uuid.uuid4().hex}"


def validate_run_id(run_id: str) -> str:
    if not isinstance(run_id, str) or not _RUN_ID_RE.fullmatch(run_id):
        raise ReplayOrchestrationError(
            "run_id must be 1-128 characters using letters, digits, '.', '_', "
            "':' or '-', and must start with a letter or digit")
    return run_id


@dataclass(frozen=True)
class ReplayRunContext:
    """One identity owner for a complete harmonization execution."""

    project_root: Path
    run_id: str
    schema: object
    specs: tuple[object, ...]
    target_schema_binding: TargetSchemaBinding
    specification_bindings: tuple[SpecificationBinding, ...]
    producer: ProducerBinding

    @classmethod
    def create(cls, project_root: str | Path, schema, specs: Iterable,
               *, run_id: str = "", application: str = "harmonize_spec",
               application_version: str = "1") -> "ReplayRunContext":
        root = Path(project_root).resolve()
        if not root.is_dir():
            raise ReplayOrchestrationError(
                f"project_root is not a directory: {root}")
        rid = validate_run_id(run_id or generate_run_id())
        loaded_specs = tuple(specs)
        if not loaded_specs:
            raise ReplayOrchestrationError("at least one loaded specification is required")

        schema_path = _loaded_path(root, schema.source_path, "target schema")
        schema_binding = TargetSchemaBinding(
            identity=f"schema:{schema.ref}",
            path=str(schema_path.relative_to(root)),
            sha256=_loaded_hash(schema.sha256, "target schema"),
            schema_id=schema.schema_id,
            schema_version=schema.schema_version,
        )

        by_ref: dict[str, SpecificationBinding] = {}
        for spec in loaded_specs:
            if spec.status != "approved":
                raise ReplayOrchestrationError(
                    f"specification {spec.ref!r} is not approved")
            missing_mapping_ids = [
                f"{mapping.target} ({mapping.op})"
                for mapping in spec.mappings if not mapping.mapping_id
            ]
            if missing_mapping_ids:
                raise ReplayOrchestrationError(
                    f"specification {spec.ref!r} has mapping(s) without "
                    f"mapping_id: {missing_mapping_ids}")
            if spec.target_schema_ref != schema.ref:
                raise ReplayOrchestrationError(
                    f"specification {spec.ref!r} targets "
                    f"{spec.target_schema_ref!r}, not {schema.ref!r}")
            path = _loaded_path(root, spec.source_path,
                                f"specification {spec.ref}")
            binding = SpecificationBinding(
                identity=f"spec:{spec.ref}",
                path=str(path.relative_to(root)),
                sha256=_loaded_hash(spec.sha256, f"specification {spec.ref}"),
                spec_id=spec.spec_id,
                spec_version=spec.spec_version,
                status=spec.status,
                target_schema_ref=spec.target_schema_ref,
                vocabulary_refs=tuple(sorted(spec.vocabularies)),
            )
            previous = by_ref.get(spec.ref)
            if previous is not None and previous != binding:
                raise ReplayOrchestrationError(
                    f"conflicting specification identity {spec.ref!r}")
            by_ref[spec.ref] = binding

        producer = producer_binding(
            application, application_version, ("numpy", "openpyxl", "pandas", "yaml"))
        return cls(
            project_root=root,
            run_id=rid,
            schema=schema,
            specs=loaded_specs,
            target_schema_binding=schema_binding,
            specification_bindings=tuple(by_ref[key] for key in sorted(by_ref)),
            producer=producer,
        )


class ReplayEvidenceCollector:
    """Collect deterministic replay bindings from actual execution facts."""

    def __init__(self, context: ReplayRunContext):
        self.context = context
        self._sources: dict[str, SourceBinding] = {}
        self._rules: dict[str, RuleBinding] = {}
        self._evidence: dict[str, EvidenceBinding] = {}
        self._usages: dict[tuple[str, str], dict[str, set[str]]] = {}
        self._specs = {spec.ref: spec for spec in context.specs}

    @property
    def sources(self) -> tuple[SourceBinding, ...]:
        return tuple(self._sources[key] for key in sorted(self._sources))

    @property
    def rules(self) -> tuple[RuleBinding, ...]:
        return tuple(self._rules[key] for key in sorted(self._rules))

    @property
    def evidence(self) -> tuple[EvidenceBinding, ...]:
        return tuple(self._evidence[key] for key in sorted(self._evidence))

    @property
    def mapping_usages(self) -> tuple[MappingUsage, ...]:
        out = []
        for spec_ref, mapping_id in sorted(self._usages):
            parts = self._usages[(spec_ref, mapping_id)]
            out.append(MappingUsage(
                mapping_id=mapping_id,
                spec_ref=spec_ref,
                source_ref_ids=tuple(sorted(parts["sources"])),
                rule_refs=tuple(sorted(parts["rules"])),
                evidence_ref_ids=tuple(sorted(parts["evidence"])),
            ))
        return tuple(out)

    def add_source(self, binding: SourceBinding) -> None:
        self._add_unique(self._sources, binding.source_ref_id, binding, "source")

    def add_rule(self, binding: RuleBinding) -> None:
        self._add_unique(self._rules, binding.ref, binding, "rule")

    def add_evidence(self, binding: EvidenceBinding) -> None:
        self._add_unique(
            self._evidence, binding.evidence_ref_id, binding, "evidence")

    def collect(self, spec, source_refs: Mapping[str, SourceRef],
                cells: Iterable[Cell]) -> None:
        """Observe one execution without rereading or rediscovering inputs."""
        loaded = self._specs.get(spec.ref)
        if loaded is None or loaded.sha256 != spec.sha256:
            raise ReplayOrchestrationError(
                f"specification {spec.ref!r} is not the context's loaded identity")

        for source_id in sorted(source_refs):
            ref = source_refs[source_id]
            if not isinstance(ref, SourceRef):
                raise ReplayOrchestrationError(
                    f"source {source_id!r} was not supplied as a SourceRef")
            if ref.config_key != source_id:
                raise ReplayOrchestrationError(
                    f"SourceRef config_key {ref.config_key!r} disagrees with "
                    f"runtime source id {source_id!r}")
            dependency_names = _READER_DEPENDENCIES.get(ref.reader_id)
            if dependency_names is None:
                raise ReplayOrchestrationError(
                    f"reader {ref.reader_id!r} has no orchestration dependency "
                    "contract")
            self.add_source(source_binding_from_ref(
                ref,
                spec_ref=spec.ref,
                source_id=source_id,
                dependencies=runtime_dependency_versions(dependency_names),
            ))

        runtime_anchors = tuple(getattr(source_refs, "anchor_evidence", ()))
        anchor_bindings = {}
        for anchor in runtime_anchors:
            binding = anchor_evidence_binding(anchor, self.context.producer)
            previous = anchor_bindings.get(binding.evidence_ref_id)
            if previous is not None and previous != binding:
                raise ReplayOrchestrationError(
                    f"conflicting anchor identity {binding.evidence_ref_id!r}")
            anchor_bindings[binding.evidence_ref_id] = binding
            self.add_evidence(binding)

        mappings = {m.mapping_id: m for m in spec.mappings if m.mapping_id}
        cells = list(cells)
        event_index: dict[str, list[tuple[Cell, dict[str, str]]]] = {}
        if any(m.op == OP_CUSTOM_RULE for m in mappings.values()):
            for candidate in cells:
                if not candidate.mapping_id:
                    continue
                identity = _cell_row_identity(
                    candidate, spec, self.context.schema.row_key)
                reference = provenance_event_ref(
                    run_id=candidate.run_id,
                    spec_ref=candidate.spec_ref,
                    mapping_id=candidate.mapping_id,
                    row_identity=identity,
                    harmonized_field=candidate.harmonized_field,
                )
                event_index.setdefault(reference, []).append((candidate, identity))

        for cell in cells:
            if not cell.mapping_id:
                if cell.extraction_method in _SPEC_MAPPING_METHODS:
                    raise ReplayOrchestrationError(
                        "normal specification mapping Cell has no mapping_id: "
                        f"method={cell.extraction_method!r}, "
                        f"field={cell.harmonized_field!r}")
                continue
            if cell.run_id != self.context.run_id:
                raise ReplayOrchestrationError(
                    f"mapping cell {cell.mapping_id!r} has run_id "
                    f"{cell.run_id!r}, expected {self.context.run_id!r}")
            if cell.spec_ref != spec.ref:
                raise ReplayOrchestrationError(
                    f"mapping cell {cell.mapping_id!r} has spec_ref "
                    f"{cell.spec_ref!r}, expected {spec.ref!r}")
            mapping = mappings.get(cell.mapping_id)
            if mapping is None:
                raise ReplayOrchestrationError(
                    f"cell names unknown mapping_id {cell.mapping_id!r}")
            if mapping.target != cell.harmonized_field:
                raise ReplayOrchestrationError(
                    f"mapping identity {spec.ref}:{cell.mapping_id} names target "
                    f"{mapping.target!r}, not cell field {cell.harmonized_field!r}")
            try:
                expected_state = finalize_declared_value_state(
                    mapping.value_state,
                    mapping.provenance.value_state).serialized_state
            except ValueError:
                # Absence, suppression/exclusion, and post-rule states retain
                # their existing specialized publication authorities.
                expected_state = None
            if (expected_state is not None
                    and cell.value_state != expected_state):
                raise ReplayOrchestrationError(
                    f"mapping cell {cell.mapping_id!r} value_state "
                    "contradicts its pinned declaration")
            vocabulary_bindings = tuple(
                vocabulary_evidence_binding(self.context.project_root, vocab)
                for vocab in _mapping_vocabularies(mapping, spec))
            for binding in vocabulary_bindings:
                self.add_evidence(binding)
            expected_vocabulary_refs = {
                binding.evidence_ref_id for binding in vocabulary_bindings}
            cell_evidence_refs = set(_split_refs(cell.evidence_refs))
            missing_vocabulary_refs = sorted(
                expected_vocabulary_refs - cell_evidence_refs)
            claimed_vocabulary_refs = {
                ref for ref in cell_evidence_refs if ref.startswith("vocab-")}
            if missing_vocabulary_refs:
                raise ReplayOrchestrationError(
                    f"mapping cell {cell.mapping_id!r} omits runtime "
                    f"vocabulary evidence {missing_vocabulary_refs}")
            if claimed_vocabulary_refs != expected_vocabulary_refs:
                raise ReplayOrchestrationError(
                    f"mapping cell {cell.mapping_id!r} claims conflicting "
                    "runtime vocabulary identity")
            grouping_anchor_scalar_usage = (
                (mapping.op == OP_NO_MAPPING_CONFIGURED
                 and mapping.source == ANCHOR_SOURCE_ID
                 and spec.record.anchor == "grouping")
                or is_grouping_anchor_direct_evidence_mapping(spec, mapping)
            )
            external_anchor_usage = is_external_anchor_evidence_mapping(
                spec, mapping)
            construction_usage = mapping_uses_record_construction(spec, mapping)
            scalar_anchor_usage = (
                grouping_anchor_scalar_usage or external_anchor_usage)
            anchor_usage = scalar_anchor_usage or construction_usage
            expected_anchor_refs = (
                set(anchor_bindings) if anchor_usage else set())
            claimed_anchor_refs = {
                ref for ref in cell_evidence_refs if ref.startswith("anchor-")}
            if anchor_usage and not expected_anchor_refs:
                raise ReplayOrchestrationError(
                    f"mapping cell {cell.mapping_id!r} has no runtime anchor binding")
            if claimed_anchor_refs != expected_anchor_refs:
                raise ReplayOrchestrationError(
                    f"mapping cell {cell.mapping_id!r} claims incomplete or "
                    "conflicting runtime anchor identity")
            if not cell.rule_id or not cell.rule_version:
                raise ReplayOrchestrationError(
                    f"mapping cell {cell.mapping_id!r} has no rule identity")
            rule_ref = f"{cell.rule_id}@{cell.rule_version}"
            if rule_ref not in _mapping_rule_refs(mapping):
                raise ReplayOrchestrationError(
                    f"mapping cell {cell.mapping_id!r} names undeclared rule "
                    f"{rule_ref!r}")
            implementation_ref = f"{mapping.rule.id}@{mapping.rule.version}"
            if mapping.op == OP_CUSTOM_RULE:
                self.add_rule(RuleBinding(
                    rule_id=mapping.rule.id,
                    rule_version=mapping.rule.version,
                    kind="field",
                ))
            if rule_ref != implementation_ref or mapping.op != OP_CUSTOM_RULE:
                self.add_rule(RuleBinding(
                    rule_id=cell.rule_id,
                    rule_version=cell.rule_version,
                    kind="mapping",
                ))

            key = (spec.ref, cell.mapping_id)
            usage = self._usages.setdefault(
                key, {"sources": set(), "rules": set(), "evidence": set()})
            usage["rules"].add(rule_ref)
            if mapping.op == OP_CUSTOM_RULE:
                usage["rules"].add(implementation_ref)
            if cell.source_ref_id:
                if cell.source_ref_id not in self._sources:
                    raise ReplayOrchestrationError(
                        f"mapping cell {cell.mapping_id!r} names unknown "
                        f"source_ref_id {cell.source_ref_id!r}")
                usage["sources"].add(cell.source_ref_id)
            for evidence_ref in _split_refs(cell.evidence_refs):
                usage["evidence"].add(evidence_ref)
            if construction_usage:
                if len(expected_anchor_refs) != 1:
                    raise ReplayOrchestrationError(
                        f"mapping cell {cell.mapping_id!r} has ambiguous "
                        "grouping construction evidence")
                anchor_binding = anchor_bindings[
                    next(iter(expected_anchor_refs))]
                details = dict(anchor_binding.details)
                source_ids = dict(details.get("source_ref_ids") or ())
                expected_source_ids = [spec.record.anchor_source]
                expected_source_ids.extend(
                    declaration.participants_from
                    for declaration in spec.record.additional_records)
                expected_source_ids = list(dict.fromkeys(expected_source_ids))
                if (details.get("anchor_kind") != "grouping"
                        or details.get("spec_ref") != spec.ref
                        or set(source_ids) != set(expected_source_ids)):
                    raise ReplayOrchestrationError(
                        f"mapping cell {cell.mapping_id!r} contradicts its "
                        "pinned grouping construction identity")
                for source_id, source_ref_id in source_ids.items():
                    runtime_ref = source_refs.get(source_id)
                    if (runtime_ref is None
                            or runtime_ref.source_ref_id != source_ref_id):
                        raise ReplayOrchestrationError(
                            f"mapping cell {cell.mapping_id!r} has incomplete "
                            f"construction source identity for {source_id!r}")
                    usage["sources"].add(source_ref_id)
            try:
                contributions = contributions_from_json(cell.contributions)
            except (TypeError, ValueError) as exc:
                raise ReplayOrchestrationError(
                    f"mapping cell {cell.mapping_id!r} has malformed "
                    "contributions JSON") from exc
            stage16_sources = _validate_stage16_runtime_evidence(
                cell, mapping, spec, contributions)
            usage["sources"].update(stage16_sources)
            if mapping.op == OP_CUSTOM_RULE:
                custom_values = _validate_custom_dependencies(
                    cell, mapping, contributions, event_index,
                    _cell_row_identity(cell, spec, self.context.schema.row_key),
                )
                _validate_custom_confidence(cell, mapping, custom_values)
            elif is_stage17_source_group_mapping(spec, mapping):
                group = source_group_for_mapping(spec, mapping)
                group_role = membership_role(spec, group)
                group_contributions = [
                    item for item in contributions
                    if isinstance(item, dict)
                    and item.get("role") == group_role
                ]
                group_sources = _validate_source_group_runtime_evidence(
                    cell, mapping, spec, group_contributions, source_refs)
                usage["sources"].update(group_sources)
            elif _is_stage4_fallback(mapping, spec):
                fallback_contributions = [
                    item for item in contributions
                    if isinstance(item, dict)
                    and item.get("role") == f"{mapping.target}_candidate"
                ]
                fallback_sources = _validate_fallback_runtime_evidence(
                    cell, mapping, fallback_contributions, source_refs)
                usage["sources"].update(fallback_sources)
            elif _is_stage15_fallback(mapping, spec):
                stage15_sources = _validate_stage15_fallback_runtime_evidence(
                    cell, mapping, spec, contributions, source_refs)
                usage["sources"].update(stage15_sources)
            elif mapping.op == OP_CONDITIONAL and mapping.contributions.conditions:
                conditional_sources = \
                    _validate_stage15_conditional_runtime_evidence(
                        cell, mapping, spec, contributions, source_refs)
                usage["sources"].update(conditional_sources)
            elif scalar_anchor_usage:
                if len(contributions) != 1:
                    raise ReplayOrchestrationError(
                        f"mapping cell {cell.mapping_id!r} needs one anchor contribution")
                contribution = contributions[0]
                if (not isinstance(contribution, dict)
                        or contribution.get("role") != "anchor_input"
                        or contribution.get("status") != STATUS_INPUT
                        or set(_split_refs(contribution.get("evidence_refs", "")))
                           != expected_anchor_refs
                        or contribution.get("anchor_record_kind")
                           not in ({"grouped", "synthesized"}
                                   if grouping_anchor_scalar_usage
                                   else {"external"})
                        or (contribution.get("anchor_record_kind") == "grouped"
                            and contribution.get("anchor_record_origin"))
                        or (contribution.get("anchor_record_kind") == "synthesized"
                            and not contribution.get("anchor_record_origin"))
                        or (contribution.get("anchor_record_kind") == "external"
                            and (contribution.get("anchor_record_origin")
                                 or isinstance(
                                     contribution.get("source_row_number"), bool)
                                 or not isinstance(
                                     contribution.get("source_row_number"), int)
                                 or contribution.get("source_row_number") < 0
                                 or isinstance(
                                     contribution.get("anchor_filtered_position"),
                                     bool)
                                 or not isinstance(
                                     contribution.get("anchor_filtered_position"),
                                     int)
                                 or contribution.get(
                                     "anchor_filtered_position") < 0))):
                    raise ReplayOrchestrationError(
                        f"mapping cell {cell.mapping_id!r} has incomplete anchor "
                        "selection evidence")
                if (contribution.get("source_column") != mapping.column
                        or not contribution.get("source_ref_id")
                        or not contribution.get("raw_value_json")):
                    raise ReplayOrchestrationError(
                        f"mapping cell {cell.mapping_id!r} has incomplete anchor "
                        "scalar evidence")
                try:
                    TypedRawValue.from_json(contribution["raw_value_json"])
                except ValueError as exc:
                    raise ReplayOrchestrationError(
                        f"mapping cell {cell.mapping_id!r} has invalid anchor "
                        "scalar evidence") from exc
                if external_anchor_usage:
                    if len(expected_anchor_refs) != 1:
                        raise ReplayOrchestrationError(
                            f"mapping cell {cell.mapping_id!r} has ambiguous "
                            "external anchor evidence")
                    anchor_binding = anchor_bindings[
                        next(iter(expected_anchor_refs))]
                    details = dict(anchor_binding.details)
                    source_ids = dict(details.get("source_ref_ids") or ())
                    anchor_ref = source_refs.get(ANCHOR_SOURCE_ID)
                    filtered_position = contribution.get(
                        "anchor_filtered_position")
                    original_rows = tuple(
                        details.get("original_source_rows") or ())
                    if (details.get("anchor_kind") != "external"
                            or set(source_ids) != {ANCHOR_SOURCE_ID}
                            or anchor_ref is None
                            or source_ids[ANCHOR_SOURCE_ID]
                               != anchor_ref.source_ref_id
                            or contribution.get("source_ref_id")
                               != anchor_ref.source_ref_id
                            or contribution.get("source_resolved_path")
                               != anchor_ref.resolved_path
                            or contribution.get("source_sha256")
                               != anchor_ref.sha256
                            or filtered_position >= len(original_rows)
                            or contribution.get("source_row_number")
                               != original_rows[filtered_position]):
                        raise ReplayOrchestrationError(
                            f"mapping cell {cell.mapping_id!r} contradicts "
                            "external anchor construction/selection evidence")
                if (is_grouping_anchor_direct_evidence_mapping(spec, mapping)
                        or external_anchor_usage):
                    attributed_source = mapping.provenance.source_ref
                    if attributed_source:
                        attributed_ref = source_refs.get(attributed_source)
                        if attributed_ref is None:
                            raise ReplayOrchestrationError(
                                f"mapping cell {cell.mapping_id!r} has no "
                                "runtime attributed source binding")
                        if cell.source_ref_id != attributed_ref.source_ref_id:
                            raise ReplayOrchestrationError(
                                f"mapping cell {cell.mapping_id!r} omits or "
                                "contradicts its attributed source link")
                        if cell.source_row_number not in ("", None):
                            raise ReplayOrchestrationError(
                                f"mapping cell {cell.mapping_id!r} claims an "
                                "attributed source row")
                    elif cell.source_ref_id:
                        raise ReplayOrchestrationError(
                            f"mapping cell {cell.mapping_id!r} claims an "
                            "undeclared attributed source link")
                for binding in anchor_bindings.values():
                    details = dict(binding.details)
                    source_ids = dict(details.get("source_ref_ids") or ())
                    usage["sources"].update(source_ids.values())
            for contribution in contributions:
                if not isinstance(contribution, dict):
                    raise ReplayOrchestrationError(
                        f"mapping cell {cell.mapping_id!r} has a non-object "
                        "contribution")
                contribution_source = str(
                    contribution.get("source_ref_id") or "")
                if contribution_source:
                    if contribution_source not in self._sources:
                        raise ReplayOrchestrationError(
                            f"mapping cell {cell.mapping_id!r} contribution "
                            f"names unknown source_ref_id "
                            f"{contribution_source!r}")
                    usage["sources"].add(contribution_source)
                for evidence_ref in _split_refs(
                        contribution.get("evidence_refs", "")):
                    usage["evidence"].add(evidence_ref)

    def build_manifest(self, delivered_artifact: ArtifactBinding,
                       provenance_artifact: ArtifactBinding) -> ReplayManifest:
        unknown_evidence = sorted(
            {ref for usage in self.mapping_usages
             for ref in usage.evidence_ref_ids} - set(self._evidence))
        if unknown_evidence:
            raise ReplayOrchestrationError(
                f"mapping usages name unregistered evidence {unknown_evidence}")
        manifest = ReplayManifest(
            run_id=self.context.run_id,
            created_at_utc=datetime.now(timezone.utc).strftime(
                "%Y-%m-%dT%H:%M:%SZ"),
            producer=self.context.producer,
            target_schema=self.context.target_schema_binding,
            specifications=self.context.specification_bindings,
            sources=self.sources,
            delivered_artifacts=(delivered_artifact,),
            provenance_artifact=provenance_artifact,
            rules=self.rules,
            evidence=self.evidence,
            mapping_usages=self.mapping_usages,
        )
        try:
            return parse_manifest(
                manifest.to_dict(), project_root=self.context.project_root)
        except ManifestError as exc:
            raise ReplayOrchestrationError(
                f"constructed replay manifest is invalid: {exc}") from exc

    @staticmethod
    def _add_unique(store: dict, key: str, value, label: str) -> None:
        if not key:
            raise ReplayOrchestrationError(f"{label} identity cannot be empty")
        previous = store.get(key)
        if previous is not None and previous != value:
            raise ReplayOrchestrationError(
                f"conflicting {label} identity {key!r}")
        store[key] = value


@dataclass(frozen=True)
class PublishedReplayRun:
    run_dir: Path
    delivered_path: Path
    provenance_path: Path
    manifest_path: Path
    manifest: ReplayManifest


class ReplayPublication:
    """Stage and atomically publish one single-spec CSV replay bundle."""

    def __init__(self, context: ReplayRunContext,
                 collector: ReplayEvidenceCollector,
                 run_dir: str | Path):
        self.context = context
        self.collector = collector
        self.run_dir = _contained_path(context.project_root, run_dir, "run directory")
        if self.run_dir == context.project_root:
            raise ReplayOrchestrationError(
                "replay run directory cannot be the project root")
        if self.run_dir.exists():
            raise ReplayOrchestrationError(
                f"replay run destination already exists: {self.run_dir}")

    def publish(self, delivered, provenance, *, artifact_id: str,
                delivered_name: str, provenance_name: str,
                manifest_name: str = DEFAULT_MANIFEST_NAME
                ) -> PublishedReplayRun:
        if provenance is None:
            raise ReplayOrchestrationError(
                "replay-enabled publication requires provenance output")
        if not artifact_id:
            raise ReplayOrchestrationError("artifact_id is required")
        names = tuple(_file_name(name) for name in (
            delivered_name, provenance_name, manifest_name))
        delivered_name, provenance_name, manifest_name = names
        if len(set(names)) != len(names):
            raise ReplayOrchestrationError(
                "delivered, provenance, and manifest names must be distinct")
        if self.run_dir.exists():
            raise ReplayOrchestrationError(
                f"replay run destination already exists: {self.run_dir}")
        self._verify_provenance_run(provenance)

        self.run_dir.parent.mkdir(parents=True, exist_ok=True)
        stage = self.run_dir.parent / (
            f".{self.run_dir.name}.staging-{uuid.uuid4().hex}")
        stage.mkdir()
        stage_delivered = stage / delivered_name
        stage_provenance = stage / provenance_name
        final_delivered = self.run_dir / delivered_name
        final_provenance = self.run_dir / provenance_name
        final_manifest = self.run_dir / manifest_name
        published = False
        try:
            self._write_csv(delivered, stage_delivered)
            self._write_csv(provenance, stage_provenance)
            _fsync_file(stage_delivered)
            _fsync_file(stage_provenance)

            delivered_binding = ArtifactBinding(
                identity=f"artifact:{artifact_id}",
                path=str(final_delivered.relative_to(self.context.project_root)),
                sha256=sha256_file(stage_delivered),
                artifact_id=artifact_id,
                format="csv",
                n_rows=len(delivered),
                schema_ref=self.context.target_schema_binding.ref,
            )
            provenance_binding = ArtifactBinding(
                identity=f"artifact:{self.context.run_id}:provenance",
                path=str(final_provenance.relative_to(self.context.project_root)),
                sha256=sha256_file(stage_provenance),
                artifact_id=f"{self.context.run_id}:provenance",
                format="csv",
                n_rows=len(provenance),
                provenance_schema_version=PROVENANCE_SCHEMA_VERSION,
            )
            manifest = self.collector.build_manifest(
                delivered_binding, provenance_binding)

            _fsync_directory(stage)
            if self.run_dir.exists():
                raise ReplayOrchestrationError(
                    f"replay run destination already exists: {self.run_dir}")
            os.rename(stage, self.run_dir)
            published = True
            _fsync_directory(self.run_dir.parent)
            self._verify_final_hashes(manifest)
            manifest.write_atomic(final_manifest)
            return PublishedReplayRun(
                run_dir=self.run_dir,
                delivered_path=final_delivered,
                provenance_path=final_provenance,
                manifest_path=final_manifest,
                manifest=manifest,
            )
        except ReplayOrchestrationError:
            raise
        except Exception as exc:
            raise ReplayOrchestrationError(
                f"replay publication failed: {exc}") from exc
        finally:
            if not published and stage.exists():
                shutil.rmtree(stage)

    @staticmethod
    def _write_csv(frame, path: Path) -> None:
        frame.to_csv(path, index=False)

    def _verify_final_hashes(self, manifest: ReplayManifest) -> None:
        failures = [check for check in manifest.verify_hashes(
            self.context.project_root) if not check.ok]
        if failures:
            details = ", ".join(
                f"{item.identity}:{item.expected}->{item.actual}"
                for item in failures)
            raise ReplayOrchestrationError(
                f"final replay evidence hash verification failed: {details}")

    def _verify_provenance_run(self, provenance) -> None:
        columns = set(getattr(provenance, "columns", ()))
        if "run_id" not in columns:
            raise ReplayOrchestrationError(
                "provenance output has no run_id column")
        run_ids = {str(value) for value in provenance["run_id"]}
        if run_ids != {self.context.run_id}:
            raise ReplayOrchestrationError(
                f"provenance run IDs {sorted(run_ids)!r} do not equal "
                f"context run ID {self.context.run_id!r}")


def _loaded_path(root: Path, value: str, label: str) -> Path:
    if not value:
        raise ReplayOrchestrationError(f"loaded {label} has no source_path")
    try:
        return _contained_path(root, value, label)
    except ReplayOrchestrationError:
        raise


def _loaded_hash(value: str, label: str) -> str:
    if not isinstance(value, str) or not re.fullmatch(r"[0-9a-f]{64}", value):
        raise ReplayOrchestrationError(
            f"loaded {label} has no valid sha256 identity")
    return value


def _contained_path(root: Path, path: str | Path, label: str) -> Path:
    raw = Path(path)
    resolved = raw.resolve() if raw.is_absolute() else (root / raw).resolve()
    if not resolved.is_relative_to(root):
        raise ReplayOrchestrationError(
            f"{label} escapes project root: {path}")
    return resolved


def _file_name(value: str) -> str:
    if not isinstance(value, str) or not value or Path(value).name != value:
        raise ReplayOrchestrationError(
            f"replay artifact name must be one filename, got {value!r}")
    if value in {".", ".."}:
        raise ReplayOrchestrationError(
            f"invalid replay artifact name {value!r}")
    return value


def _split_refs(value: str) -> tuple[str, ...]:
    return tuple(item for item in str(value or "").split(";") if item)


def _mapping_rule_refs(mapping) -> set[str]:
    refs = {f"{mapping.rule.id}@{mapping.rule.version}"}
    for rule in (mapping.no_value_rule, mapping.unmapped_rule):
        if rule is not None:
            refs.add(f"{rule.id}@{rule.version}")
    for rule in mapping.fallback_rules:
        refs.add(f"{rule.rule_id}@{rule.rule_version}")
    return refs


def _mapping_vocabularies(mapping, spec) -> tuple[vocabularies.Vocabulary, ...]:
    """Exact loaded objects used at this mapping's declared use sites."""
    loaded = []
    if mapping.fallback_vocabulary:
        vocab = mapping.fallback_vocabulary_object
        if (not isinstance(vocab, vocabularies.Vocabulary)
                or vocab.ref != mapping.fallback_vocabulary
                or vocab.kind != vocabularies.KIND_ORDERED_RULES):
            raise ReplayOrchestrationError(
                f"mapping {mapping.mapping_id!r} has incomplete or conflicting "
                "runtime fallback-vocabulary identity")
        loaded.append(vocab)
    elif mapping.fallback_vocabulary_object is not None:
        raise ReplayOrchestrationError(
            f"mapping {mapping.mapping_id!r} retains an undeclared fallback "
            "vocabulary")
    parameter_names = set()
    for parameter, vocab in mapping.parameter_vocabularies:
        if (not parameter or parameter in parameter_names
                or not isinstance(vocab, vocabularies.Vocabulary)
                or vocab.kind != vocabularies.KIND_BUCKETS):
            raise ReplayOrchestrationError(
                f"mapping {mapping.mapping_id!r} has incomplete or conflicting "
                "runtime parameter-vocabulary identity")
        parameter_names.add(parameter)
        loaded.append(vocab)
    undeclared = sorted({vocab.ref for vocab in loaded} - set(spec.vocabularies))
    if undeclared:
        raise ReplayOrchestrationError(
            f"mapping {mapping.mapping_id!r} loaded undeclared vocabularies "
            f"{undeclared}")
    return tuple(loaded)


def _cell_row_identity(cell: Cell, spec, row_key) -> dict[str, str]:
    """Adapt fixed compatibility anchors to schema-defined identity names."""
    values = {"trial": cell.trial}
    record = spec.record
    if record.participant_key:
        values[record.participant_key] = cell.cimac_part_id
    if record.sample_key:
        values[record.sample_key] = cell.Cimac_id
    if record.collection_event_key:
        values[record.collection_event_key] = cell.Collection_Event
    try:
        return canonical_row_identity(row_key, values)
    except ValueError as exc:
        raise ReplayOrchestrationError(
            f"cell {cell.mapping_id!r} row identity is not representable: {exc}"
        ) from exc


def _validate_custom_dependencies(cell: Cell, mapping, contributions,
                                  event_index, row_identity) -> tuple[Any, ...]:
    if len(contributions) != len(mapping.inputs):
        raise ReplayOrchestrationError(
            f"custom mapping cell {cell.mapping_id!r} has "
            f"{len(contributions)} input contribution(s), expected "
            f"{len(mapping.inputs)}")
    values = []
    for rank, (field, contribution) in enumerate(
            zip(mapping.inputs, contributions)):
        if not isinstance(contribution, dict):
            raise ReplayOrchestrationError(
                f"custom mapping cell {cell.mapping_id!r} has a non-object input")
        if (contribution.get("status") != STATUS_INPUT
                or contribution.get("rank") != rank
                or contribution.get("source_column") != field
                or contribution.get("role") != "registered_rule_input"):
            raise ReplayOrchestrationError(
                f"custom mapping cell {cell.mapping_id!r} input {rank} does not "
                "match its declared position and field")
        reference = str(contribution.get("provenance_ref") or "")
        if not is_provenance_event_ref(reference):
            raise ReplayOrchestrationError(
                f"custom mapping cell {cell.mapping_id!r} input {rank} has no "
                "valid pev1 provenance_ref")
        matches = event_index.get(reference, ())
        if len(matches) != 1:
            raise ReplayOrchestrationError(
                f"custom mapping cell {cell.mapping_id!r} input {rank} reference "
                f"resolves to {len(matches)} runtime Cells")
        upstream, upstream_identity = matches[0]
        if (upstream.run_id != cell.run_id
                or upstream.spec_ref != cell.spec_ref
                or upstream_identity != row_identity
                or upstream.harmonized_field != field):
            raise ReplayOrchestrationError(
                f"custom mapping cell {cell.mapping_id!r} input {rank} reference "
                "does not identify the declared same-row upstream Cell")
        if not contribution.get("raw_value_json"):
            raise ReplayOrchestrationError(
                f"custom mapping cell {cell.mapping_id!r} input {rank} has no "
                "typed snapshot")
        if contribution.get("source_ref_id"):
            raise ReplayOrchestrationError(
                f"custom mapping cell {cell.mapping_id!r} input {rank} "
                "incorrectly copies a transitive source_ref_id")
        try:
            recorded = TypedRawValue.from_json(contribution["raw_value_json"])
            observed = encode_typed_raw(upstream.value)
        except ValueError as exc:
            raise ReplayOrchestrationError(
                f"custom mapping cell {cell.mapping_id!r} input {rank} has an "
                f"invalid typed snapshot: {exc}") from exc
        if recorded != observed:
            raise ReplayOrchestrationError(
                f"custom mapping cell {cell.mapping_id!r} input {rank} typed "
                "snapshot disagrees with its runtime upstream Cell")
        values.append(upstream.value)
    return tuple(values)


def _validate_custom_confidence(cell: Cell, mapping,
                                values: tuple[Any, ...]) -> None:
    """Recompute a runtime custom Cell's confidence without trusting the Cell."""
    try:
        registered = rule_registry.resolve(
            mapping.rule.id, mapping.rule.version,
            kind=rule_registry.KIND_FIELD)
        params = dict(mapping.params)
        rule_registry.check_contract(registered, mapping.inputs, params)
        value, rule_confidence, _note = rule_registry.invoke(
            registered, list(values), params)
        finalized = finalize_registered_rule_confidence(
            value, rule_confidence,
            declared_value_policy=mapping.confidence,
            declared_no_value_confidence=mapping.confidence_no_value)
        recorded = validate_registered_rule_confidence(cell.confidence)
    except (rule_registry.RuleError, ConfidencePolicyError,
            TypeError, ValueError) as exc:
        raise ReplayOrchestrationError(
            f"custom mapping cell {cell.mapping_id!r} has invalid runtime "
            f"registered-rule confidence: {exc}") from exc
    expected_rule = mapping.rule
    if (finalized.outcome == OUTCOME_NO_VALUE
            and mapping.no_value_rule is not None):
        expected_rule = mapping.no_value_rule
    if (cell.rule_id, cell.rule_version) != (
            expected_rule.id, expected_rule.version):
        raise ReplayOrchestrationError(
            f"custom mapping cell {cell.mapping_id!r} rule identity "
            "contradicts its independently recomputed outcome")
    if recorded != finalized.value:
        raise ReplayOrchestrationError(
            f"custom mapping cell {cell.mapping_id!r} confidence contradicts "
            "its independently recomputed registered-rule result")


def _is_stage17_publication_route(mapping, spec) -> bool:
    """Mirror the closed raw-route boundary for producer validation only."""
    if mapping.op == OP_CUSTOM_RULE:
        return False
    if mapping.op in {OP_CONSTANT, OP_NOT_APPLICABLE,
                      OP_NO_MAPPING_CONFIGURED}:
        return True
    if mapping.op in {OP_DIRECT, OP_VALUE_MAP, OP_UNIT_CONVERSION}:
        return (
            is_stage17_source_group_mapping(spec, mapping)
            or bool(mapping.source and mapping.source != ANCHOR_SOURCE_ID
                    and GROUP_REF_SEP not in mapping.source
                    and not mapping.column_by_member))
    if mapping.op == OP_FILE_MEMBERSHIP:
        return is_stage17_source_group_mapping(spec, mapping)
    if mapping.op == OP_CONDITIONAL:
        return bool(mapping.clauses and not mapping.candidates and not mapping.map)
    if mapping.op == OP_FALLBACK:
        return bool(
            mapping.candidates and not mapping.map
            and not mapping.fallback_vocabulary
            and mapping.coerce in ("none", "numeric")
            and mapping.contributions.record == "all")
    return False


def _validate_stage16_runtime_evidence(
        cell: Cell, mapping, spec, contributions) -> set[str]:
    """Require derivation-only causal links before manifest publication.

    This is deliberately a publication check, not replay authority.  It uses
    the derivation just produced by execution only to identify the selected
    pinned variant, then requires the existing usage-scoped Contributions for
    every read/group fact and every condition that could decide that variant.
    Replay independently reconstructs all of those facts later.
    """
    variants = tuple(mapping.derivation)
    stage16 = bool(variants and is_stage16_derivation(variants))
    stage17 = bool(
        variants and is_stage17_derivation(variants)
        and _is_stage17_publication_route(mapping, spec))
    stage19 = bool(
        variants and is_stage19_derivation(variants)
        and mapping_uses_record_construction(spec, mapping))
    fallback_exhausted = bool(
        mapping.op == OP_FALLBACK
        and not any(
            isinstance(item, dict)
            and item.get("role") == f"{mapping.target}_candidate"
            and item.get("status") == STATUS_SELECTED
            for item in contributions))
    causal_raw_from = raw_from_is_causal(
        raw_from_id=mapping.provenance.raw_from,
        raw_policy=mapping.provenance.raw_value,
        completion=(COMPLETION_FALLBACK_EXHAUSTED
                    if fallback_exhausted else COMPLETION_VALUE))
    if not stage16 and not stage17 and not stage19 and not causal_raw_from:
        return set()
    new_refs = {
        ref for variant in variants
        for ref in derivation_references(variant.record)
        if ref.startswith((
            "read.", "condition.", "group_member.", "group_value.",
            "group_label."))
    }
    if (not new_refs and not mapping.when and not causal_raw_from
            and not any(variant.when for variant in variants)):
        return set()
    selected_index = None
    if stage16 or stage17 or stage19:
        try:
            rendered = json.loads(cell.derivation)
        except (TypeError, ValueError) as exc:
            raise ReplayOrchestrationError(
                f"mapping cell {cell.mapping_id!r} has malformed derivation JSON"
            ) from exc

        matches = [
            index for index, variant in enumerate(variants)
            if matches_derivation_template(variant.record, rendered)
        ]
        if not matches and rendered == {}:
            selected_index = None
        elif len(matches) == 1:
            selected_index = matches[0]
        else:
            raise ReplayOrchestrationError(
                f"mapping cell {cell.mapping_id!r} derivation does not identify "
                "one pinned variant for causal publication")

    conditions = {condition.id: condition for condition in spec.conditions}
    reads = {read.id: read for read in spec.reads}
    read_ids: set[str] = set()
    group_ids: set[str] = set()
    selected_refs: set[str] = set()
    deciding_condition_ids: list[str] = []
    if mapping.when:
        deciding_condition_ids.append(mapping.when)
    limit = len(variants) if selected_index is None else selected_index + 1
    for variant in variants[:limit]:
        if variant.when:
            deciding_condition_ids.append(variant.when)
    if selected_index is not None:
        selected_refs = derivation_references(
            variants[selected_index].record)
    if causal_raw_from:
        read_ids.add(mapping.provenance.raw_from)

    for ref in selected_refs:
        namespace, _, identity = ref.partition(".")
        if namespace == "read":
            read_ids.add(identity)
        elif namespace == "condition":
            deciding_condition_ids.append(identity)
        elif namespace in {"group_member", "group_value", "group_label"}:
            group_ids.add(identity)
    for condition_id in deciding_condition_ids:
        condition = conditions.get(condition_id)
        if condition is None:
            raise ReplayOrchestrationError(
                f"mapping cell {cell.mapping_id!r} names unknown condition "
                f"{condition_id!r}")
        if condition.read:
            read_ids.add(condition.read)
        elif condition.group and condition.member:
            group_ids.add(condition.group)
        elif (condition.record_origin
              and mapping_uses_record_construction(spec, mapping)):
            # Construction linkage was validated separately above.  It is
            # record-level authority, not a scalar Contribution.
            continue
        else:
            raise ReplayOrchestrationError(
                f"mapping cell {cell.mapping_id!r} condition "
                f"{condition_id!r} is outside Stage 16 authority")
    for read_id in tuple(read_ids):
        read = reads.get(read_id)
        if read is None:
            raise ReplayOrchestrationError(
                f"mapping cell {cell.mapping_id!r} names unknown read "
                f"{read_id!r}")
        if GROUP_REF_SEP in read.source:
            group_ids.add(read.source.partition(GROUP_REF_SEP)[0])

    by_role: dict[str, list[dict[str, Any]]] = {}
    for contribution in contributions:
        if isinstance(contribution, dict):
            by_role.setdefault(str(contribution.get("role") or ""), []).append(
                contribution)
    required_sources: set[str] = set()
    for read_id in sorted(read_ids):
        matches = by_role.get(read_id, ())
        if len(matches) != 1 or not matches[0].get("source_ref_id"):
            raise ReplayOrchestrationError(
                f"mapping cell {cell.mapping_id!r} omits usage-scoped "
                f"named-read evidence for {read_id!r}")
        required_sources.add(str(matches[0]["source_ref_id"]))
    for group_id in sorted(group_ids):
        group = spec.group(group_id)
        if group is None:
            raise ReplayOrchestrationError(
                f"mapping cell {cell.mapping_id!r} names unknown source group "
                f"{group_id!r}")
        role = membership_role(spec, group)
        matches = by_role.get(role, ())
        if (len(matches) != len(group.members)
                or any(not item.get("source_ref_id") for item in matches)):
            raise ReplayOrchestrationError(
                f"mapping cell {cell.mapping_id!r} omits usage-scoped "
                f"membership evidence for {group_id!r}")
        required_sources.update(str(item["source_ref_id"]) for item in matches)
    return required_sources


def _validate_source_group_runtime_evidence(
        cell: Cell, mapping, spec, contributions,
        source_refs: Mapping[str, SourceRef]) -> set[str]:
    """Validate complete runtime group evidence before manifest publication."""
    group = source_group_for_mapping(spec, mapping)
    if group is None:
        raise ReplayOrchestrationError(
            f"mapping cell {cell.mapping_id!r} has no exact source group")
    expected_count = len(group.members)
    if len(contributions) != expected_count:
        raise ReplayOrchestrationError(
            f"mapping cell {cell.mapping_id!r} needs {expected_count} ordered "
            "membership contributions")

    winner_status = (
        STATUS_CONDITION if group.membership_status == "condition"
        else STATUS_SELECTED)
    winners = [
        rank for rank, contribution in enumerate(contributions)
        if isinstance(contribution, dict)
        and contribution.get("status") == winner_status
    ]
    if len(winners) != 1:
        raise ReplayOrchestrationError(
            f"mapping cell {cell.mapping_id!r} has no unique group winner")
    selected_rank = winners[0]
    presence = {
        member.id: rank == selected_rank
        for rank, member in enumerate(group.members)
    }
    decision = select_first_containing_record(
        tuple(member.id for member in group.members), presence,
        membership_status=group.membership_status)
    expected_role = membership_role(spec, group)
    expected_rule = group_rule_id(spec, group)
    required_refs: set[str] = set()

    for rank, (member, contribution) in enumerate(
            zip(group.members, contributions)):
        if not isinstance(contribution, dict):
            raise ReplayOrchestrationError(
                f"mapping cell {cell.mapping_id!r} membership {rank} is not "
                "an object")
        source_id = member.sources[group.decided_by]
        ref = source_refs.get(source_id)
        if ref is None:
            raise ReplayOrchestrationError(
                f"mapping cell {cell.mapping_id!r} has no deciding source "
                f"{source_id!r}")
        required_refs.add(ref.source_ref_id)
        locator = contribution.get("source_row_number")
        if rank == selected_rank:
            valid_locator = (
                not isinstance(locator, bool)
                and isinstance(locator, int) and locator >= 0)
        elif rank < selected_rank:
            valid_locator = locator in ("", None)
        else:
            valid_locator = (
                locator in ("", None)
                or (not isinstance(locator, bool)
                    and isinstance(locator, int) and locator >= 0))
        if not valid_locator:
            raise ReplayOrchestrationError(
                f"mapping cell {cell.mapping_id!r} membership {rank} has an "
                "invalid source-row locator")
        if not contribution.get("raw_value_json"):
            raise ReplayOrchestrationError(
                f"mapping cell {cell.mapping_id!r} membership {rank} omits "
                "typed presence evidence")
        try:
            TypedRawValue.from_json(contribution["raw_value_json"])
        except ValueError as exc:
            raise ReplayOrchestrationError(
                f"mapping cell {cell.mapping_id!r} membership {rank} has an "
                f"invalid typed snapshot: {exc}") from exc
        if (contribution.get("rank") != rank
                or contribution.get("role") != expected_role
                or contribution.get("status") != decision.statuses[rank]
                or contribution.get("source_column") != ""
                or contribution.get("source_ref_id") != ref.source_ref_id
                or contribution.get("source_logical_label") != ref.logical_label
                or contribution.get("source_resolved_path") != ref.resolved_path
                or contribution.get("source_sha256") != ref.sha256
                or contribution.get("source_sheet") != (ref.sheet or "")
                or contribution.get("source_record_key")
                   != cell.source_record_key
                or contribution.get("rule_id") != expected_rule):
            raise ReplayOrchestrationError(
                f"mapping cell {cell.mapping_id!r} membership {rank} "
                "contradicts the runtime group decision")

    selected = group.members[selected_rank]
    scalar_source_id = selected.sources[
        scalar_role_for_mapping(mapping, group)]
    scalar_ref = source_refs.get(scalar_source_id)
    if scalar_ref is None:
        raise ReplayOrchestrationError(
            f"mapping cell {cell.mapping_id!r} has no selected scalar-role "
            f"source {scalar_source_id!r}")
    required_refs.add(scalar_ref.source_ref_id)
    expected_column = (
        mapping.column_by_member[selected.id]
        if mapping.column_by_member else (mapping.column or mapping.target))
    if (cell.source_ref_id != scalar_ref.source_ref_id
            or cell.source_resolved_path != scalar_ref.resolved_path
            or cell.source_sha256 != scalar_ref.sha256
            or cell.source_logical_label != scalar_ref.logical_label
            or cell.source_sheet != (scalar_ref.sheet or "")
            or cell.source_column != expected_column
            or isinstance(cell.source_row_number, bool)
            or not isinstance(cell.source_row_number, int)
            or cell.source_row_number < 0
            or not cell.source_record_key):
        raise ReplayOrchestrationError(
            f"mapping cell {cell.mapping_id!r} has incomplete selected "
            "scalar-role evidence")
    return required_refs


def _validate_stage15_group_membership_prefix(
        cell: Cell, spec, group_id: str, contributions,
        source_refs: Mapping[str, SourceRef]):
    """Validate ordered group membership evidence used by a named read."""
    group = spec.group(group_id)
    if group is None or group.select != "first_containing_record":
        raise ReplayOrchestrationError(
            f"conditional mapping cell {cell.mapping_id!r} names unsupported "
            f"source group {group_id!r}")
    if len(contributions) != len(group.members):
        raise ReplayOrchestrationError(
            f"conditional mapping cell {cell.mapping_id!r} needs complete "
            f"membership evidence for source group {group_id!r}")
    winner_status = (
        STATUS_CONDITION if group.membership_status == "condition"
        else STATUS_SELECTED)
    winners = [
        rank for rank, contribution in enumerate(contributions)
        if isinstance(contribution, dict)
        and contribution.get("status") == winner_status
    ]
    if len(winners) != 1:
        raise ReplayOrchestrationError(
            f"conditional mapping cell {cell.mapping_id!r} has no unique "
            f"source-group winner for {group_id!r}")
    selected_rank = winners[0]
    presence = {
        member.id: rank == selected_rank
        for rank, member in enumerate(group.members)
    }
    decision = select_first_containing_record(
        tuple(member.id for member in group.members), presence,
        membership_status=group.membership_status)
    expected_role = membership_role(spec, group)
    expected_rule = group_rule_id(spec, group)
    required_refs = set()
    for rank, (member, contribution) in enumerate(
            zip(group.members, contributions)):
        source_id = member.sources[group.decided_by]
        ref = source_refs.get(source_id)
        if ref is None:
            raise ReplayOrchestrationError(
                f"conditional mapping cell {cell.mapping_id!r} has no runtime "
                f"deciding source {source_id!r}")
        required_refs.add(ref.source_ref_id)
        locator = contribution.get("source_row_number")
        valid_locator = (
            not isinstance(locator, bool) and isinstance(locator, int)
            and locator >= 0) if rank == selected_rank else locator in ("", None)
        if rank > selected_rank:
            valid_locator = valid_locator or (
                not isinstance(locator, bool) and isinstance(locator, int)
                and locator >= 0)
        snapshot = str(contribution.get("raw_value_json") or "")
        if not snapshot:
            raise ReplayOrchestrationError(
                f"conditional mapping cell {cell.mapping_id!r} membership "
                f"{rank} has no typed presence evidence")
        try:
            TypedRawValue.from_json(snapshot)
        except ValueError as exc:
            raise ReplayOrchestrationError(
                f"conditional mapping cell {cell.mapping_id!r} membership "
                f"{rank} has invalid typed evidence") from exc
        if (not valid_locator
                or contribution.get("rank") != rank
                or contribution.get("role") != expected_role
                or contribution.get("status") != decision.statuses[rank]
                or contribution.get("source_column") != ""
                or contribution.get("source_ref_id") != ref.source_ref_id
                or contribution.get("source_logical_label") != ref.logical_label
                or contribution.get("source_resolved_path") != ref.resolved_path
                or contribution.get("source_sha256") != ref.sha256
                or str(contribution.get("source_sheet") or "")
                   != (ref.sheet or "")
                or contribution.get("source_record_key")
                   != cell.source_record_key
                or contribution.get("rule_id") != expected_rule):
            raise ReplayOrchestrationError(
                f"conditional mapping cell {cell.mapping_id!r} membership "
                f"{rank} contradicts its runtime group decision")
    return group.members[selected_rank], required_refs


def _validate_stage15_conditional_runtime_evidence(
        cell: Cell, mapping, spec, contributions,
        source_refs: Mapping[str, SourceRef]) -> set[str]:
    """Fail closed on named-read evidence before conditional publication."""
    read_ids = tuple(mapping.contributions.conditions)
    if len(contributions) < len(read_ids):
        raise ReplayOrchestrationError(
            f"conditional mapping cell {cell.mapping_id!r} omits named-read "
            "Contributions")
    split = len(contributions) - len(read_ids)
    prefix = contributions[:split]
    condition_inputs = contributions[split:]
    reads = {read.id: read for read in spec.reads}
    group_id = mapping.provenance.contributions_from
    selected_member = None
    required_refs = set()
    if group_id:
        selected_member, group_refs = _validate_stage15_group_membership_prefix(
            cell, spec, group_id, prefix, source_refs)
        required_refs.update(group_refs)
    elif prefix:
        raise ReplayOrchestrationError(
            f"conditional mapping cell {cell.mapping_id!r} has undeclared "
            "pre-condition Contributions")

    for position, (read_id, contribution) in enumerate(
            zip(read_ids, condition_inputs)):
        read = reads.get(read_id)
        if read is None:
            raise ReplayOrchestrationError(
                f"conditional mapping cell {cell.mapping_id!r} names unknown "
                f"read {read_id!r}")
        source_id = read.source
        member = None
        if GROUP_REF_SEP in source_id:
            read_group, _, role = source_id.partition(GROUP_REF_SEP)
            if read_group != group_id or selected_member is None:
                raise ReplayOrchestrationError(
                    f"conditional mapping cell {cell.mapping_id!r} lacks "
                    f"usage-scoped membership evidence for read {read_id!r}")
            member = selected_member
            source_id = member.sources.get(role, "")
        ref = source_refs.get(source_id)
        column = read.column
        if read.column_by_member and member is not None:
            column = read.column_by_member.get(member.id, column)
        snapshot = str(contribution.get("raw_value_json") or "")
        if ref is None or not source_id or not column or not snapshot:
            raise ReplayOrchestrationError(
                f"conditional mapping cell {cell.mapping_id!r} read "
                f"{read_id!r} has incomplete runtime evidence")
        try:
            TypedRawValue.from_json(snapshot)
        except ValueError as exc:
            raise ReplayOrchestrationError(
                f"conditional mapping cell {cell.mapping_id!r} read "
                f"{read_id!r} has invalid typed evidence") from exc
        if (contribution.get("role") != read_id
                or contribution.get("status") != STATUS_CONDITION
                or contribution.get("rank")
                   != mapping.contributions.condition_rank
                or contribution.get("source_column") != column
                or contribution.get("source_ref_id") != ref.source_ref_id
                or contribution.get("source_logical_label") != ref.logical_label
                or contribution.get("source_resolved_path") != ref.resolved_path
                or contribution.get("source_sha256") != ref.sha256
                or str(contribution.get("source_sheet") or "")
                   != (ref.sheet or "")
                or contribution.get("source_record_key")
                   != cell.source_record_key
                or isinstance(contribution.get("source_row_number"), bool)
                or not isinstance(contribution.get("source_row_number"), int)
                or contribution.get("source_row_number") < 0
                or contribution.get("rule_id") != mapping.rule.id
                or contribution.get("provenance_ref")):
            raise ReplayOrchestrationError(
                f"conditional mapping cell {cell.mapping_id!r} read "
                f"{read_id!r} is incomplete, contradictory, or out of order")
        required_refs.add(ref.source_ref_id)
    return required_refs


def _is_stage4_fallback(mapping, spec) -> bool:
    """Whether a mapping is in the narrow Stage 4 production-evidence subset."""
    if (mapping.op != OP_FALLBACK or mapping.when or mapping.map
            or mapping.fallback_vocabulary
            or mapping.coerce not in ("none", "numeric")
            or mapping.source or mapping.group or mapping.column_by_member
            or mapping.provenance.source_ref or mapping.provenance.row_from
            or (mapping.derivation and not (
                is_stage17_derivation(mapping.derivation)
                and _is_stage17_publication_route(mapping, spec)))
            or mapping.contributions.record != "all"
            or mapping.contributions.after_decision != "not_evaluated"
            or mapping.contributions.conditions
            or any(candidate.when is not None
                   for candidate in mapping.candidates)):
        return False
    for candidate in mapping.candidates:
        if (not candidate.source or candidate.source == ANCHOR_SOURCE_ID
                or GROUP_REF_SEP in candidate.source):
            return False
        source = spec.source(candidate.source)
        if (source is None or not source.path or source.path_pattern
                or source.file_type != "csv"
                or source.reader != "cidc_csv@v1" or source.sheet):
            return False
    return True


def _is_stage15_fallback(mapping, spec) -> bool:
    """Whether publication must validate the closed gated-fallback evidence."""
    if (mapping.op != OP_FALLBACK or mapping.when or mapping.map
            or mapping.fallback_vocabulary
            or mapping.coerce not in ("none", "numeric")
            or mapping.source or mapping.group or mapping.column_by_member
            or mapping.provenance.source_ref or mapping.provenance.row_from
            or (mapping.derivation and not (
                is_stage17_derivation(mapping.derivation)
                and _is_stage17_publication_route(mapping, spec)))
            or mapping.contributions.record != "all"):
        return False
    if not (any(candidate.when is not None
                for candidate in mapping.candidates)
            or mapping.contributions.conditions
            or mapping.contributions.after_decision != "not_evaluated"):
        return False
    declared_reads = {read.id for read in spec.reads}
    for candidate in mapping.candidates:
        if (not candidate.source or candidate.source == ANCHOR_SOURCE_ID
                or GROUP_REF_SEP in candidate.source):
            return False
        source = spec.source(candidate.source)
        if (source is None or not source.path or source.path_pattern
                or source.file_type != "csv"
                or source.reader != "cidc_csv@v1" or source.sheet):
            return False
        if (candidate.when and candidate.when.read
                and (candidate.when.read not in declared_reads
                     or candidate.when.read
                        not in mapping.contributions.conditions)):
            return False
    return True


def _validate_stage15_fallback_runtime_evidence(
        cell: Cell, mapping, spec, contributions,
        source_refs: Mapping[str, SourceRef]) -> set[str]:
    """Fail closed on gated-fallback Contribution identity before publication."""
    direct_gates = []
    seen = set()
    for candidate in mapping.candidates:
        gate = candidate.when
        if gate is None or gate.read:
            continue
        key = (gate.source, gate.column)
        if key not in seen:
            seen.add(key)
            direct_gates.append(gate)
    expected = []
    direct = [
        ("gate", gate, f"{mapping.target}_condition", -1, gate.column)
        for gate in direct_gates
    ]
    candidates = [
        ("candidate", candidate, f"{mapping.target}_candidate", rank,
         candidate.column)
        for rank, candidate in enumerate(mapping.candidates)
    ]
    expected.extend(
        direct + candidates
        if mapping.contributions.condition_position == "first"
        else candidates + direct)
    reads = {read.id: read for read in spec.reads}
    for read_id in mapping.contributions.conditions:
        read = reads.get(read_id)
        if read is None:
            raise ReplayOrchestrationError(
                f"fallback mapping cell {cell.mapping_id!r} names unknown "
                f"condition read {read_id!r}")
        expected.append(
            ("read", read, read_id, mapping.contributions.condition_rank, None))
    if (mapping.provenance.raw_from
            and mapping.provenance.raw_from
               not in mapping.contributions.conditions
            and any(
                isinstance(item, dict)
                and item.get("role") == mapping.provenance.raw_from
                for item in contributions)):
        read = reads.get(mapping.provenance.raw_from)
        if read is None:
            raise ReplayOrchestrationError(
                f"fallback mapping cell {cell.mapping_id!r} names unknown "
                f"raw_from read {mapping.provenance.raw_from!r}")
        expected.append((
            "read", read, mapping.provenance.raw_from,
            mapping.contributions.condition_rank, None))
    if len(contributions) != len(expected):
        raise ReplayOrchestrationError(
            f"fallback mapping cell {cell.mapping_id!r} has incomplete or "
            "extra Stage 15 Contributions")

    source_ids = set()
    selected = False
    for position, (contribution, descriptor) in enumerate(
            zip(contributions, expected)):
        if not isinstance(contribution, dict):
            raise ReplayOrchestrationError(
                f"fallback mapping cell {cell.mapping_id!r} Contribution "
                f"{position} is not an object")
        kind, declaration, role, rank, column = descriptor
        if (contribution.get("role") != role
                or contribution.get("rank") != rank
                or (column is not None
                    and contribution.get("source_column") != column)
                or contribution.get("status")
                   not in ({STATUS_CONDITION} if kind in ("gate", "read")
                           else {STATUS_SELECTED, STATUS_UNAVAILABLE,
                                 STATUS_REJECTED, STATUS_NOT_EVALUATED})
                or not contribution.get("source_ref_id")
                or not contribution.get("source_record_key")
                or isinstance(contribution.get("source_row_number"), bool)
                or not isinstance(contribution.get("source_row_number"), int)
                or contribution.get("source_row_number") < 0
                or not contribution.get("raw_value_json")):
            raise ReplayOrchestrationError(
                f"fallback mapping cell {cell.mapping_id!r} Contribution "
                f"{position} is incomplete or out of order")
        try:
            TypedRawValue.from_json(contribution["raw_value_json"])
        except ValueError as exc:
            raise ReplayOrchestrationError(
                f"fallback mapping cell {cell.mapping_id!r} Contribution "
                f"{position} has invalid typed evidence") from exc
        source_ref_id = contribution["source_ref_id"]
        source_ids.add(source_ref_id)
        if kind in ("candidate", "gate"):
            source_id = declaration.source
            ref = source_refs.get(source_id)
            if (ref is None or ref.source_ref_id != source_ref_id
                    or contribution.get("source_resolved_path")
                       != ref.resolved_path
                    or contribution.get("source_sha256") != ref.sha256
                    or contribution.get("source_logical_label")
                       != ref.logical_label
                    or str(contribution.get("source_sheet") or "")
                       != (ref.sheet or "")):
                raise ReplayOrchestrationError(
                    f"fallback mapping cell {cell.mapping_id!r} Contribution "
                    f"{position} contradicts runtime source identity")
        if kind == "candidate":
            status = contribution["status"]
            if selected and status != mapping.contributions.after_decision:
                raise ReplayOrchestrationError(
                    f"fallback mapping cell {cell.mapping_id!r} candidate "
                    f"{rank} contradicts post-winner status")
            selected = selected or status == STATUS_SELECTED
    return source_ids


def _validate_fallback_runtime_evidence(cell: Cell, mapping, contributions,
                                        source_refs: Mapping[str, SourceRef]
                                        ) -> set[str]:
    """Validate runtime fallback facts and return the usage-scoped source union."""
    if len(contributions) != len(mapping.candidates):
        raise ReplayOrchestrationError(
            f"fallback mapping cell {cell.mapping_id!r} has "
            f"{len(contributions)} candidate contribution(s), expected "
            f"{len(mapping.candidates)}")
    selected = False
    eligibility_sources: set[str] = set()
    scalar_sources: set[str] = set()
    for rank, (candidate, contribution) in enumerate(
            zip(mapping.candidates, contributions)):
        if not isinstance(contribution, dict):
            raise ReplayOrchestrationError(
                f"fallback mapping cell {cell.mapping_id!r} candidate {rank} "
                "is not an object")
        ref = source_refs.get(candidate.source)
        if ref is None:
            raise ReplayOrchestrationError(
                f"fallback mapping cell {cell.mapping_id!r} candidate {rank} "
                f"has no runtime SourceRef for {candidate.source!r}")
        eligibility_sources.add(ref.source_ref_id)
        if (contribution.get("rank") != rank
                or contribution.get("role") != f"{mapping.target}_candidate"
                or contribution.get("source_column") != candidate.column
                or contribution.get("source_ref_id") != ref.source_ref_id
                or contribution.get("source_resolved_path") != ref.resolved_path
                or contribution.get("source_sha256") != ref.sha256
                or contribution.get("source_logical_label") != ref.logical_label
                or str(contribution.get("source_sheet") or "") != (ref.sheet or "")
                or contribution.get("rule_id") != mapping.rule.id
                or contribution.get("provenance_ref")):
            raise ReplayOrchestrationError(
                f"fallback mapping cell {cell.mapping_id!r} candidate {rank} "
                "does not match its declared runtime source context")
        row_number = contribution.get("source_row_number")
        if (isinstance(row_number, bool) or not isinstance(row_number, int)
                or row_number < 0 or not contribution.get("source_record_key")):
            raise ReplayOrchestrationError(
                f"fallback mapping cell {cell.mapping_id!r} candidate {rank} "
                "has incomplete source-presence evidence")
        status = contribution.get("status")
        if selected:
            if status != STATUS_NOT_EVALUATED:
                raise ReplayOrchestrationError(
                    f"fallback mapping cell {cell.mapping_id!r} candidate "
                    f"{rank} is not not_evaluated after the winner")
            continue
        scalar_sources.add(ref.source_ref_id)
        snapshot = str(contribution.get("raw_value_json") or "")
        if not snapshot:
            raise ReplayOrchestrationError(
                f"fallback mapping cell {cell.mapping_id!r} candidate {rank} "
                "has no reached typed scalar evidence")
        try:
            typed = TypedRawValue.from_json(snapshot)
        except ValueError as exc:
            raise ReplayOrchestrationError(
                f"fallback mapping cell {cell.mapping_id!r} candidate {rank} "
                f"has invalid typed scalar evidence: {exc}") from exc
        expected = STATUS_UNAVAILABLE if typed.kind == "null" else STATUS_SELECTED
        if status != expected:
            raise ReplayOrchestrationError(
                f"fallback mapping cell {cell.mapping_id!r} candidate {rank} "
                f"status {status!r} contradicts runtime typed evidence")
        selected = status == STATUS_SELECTED
    return eligibility_sources | scalar_sources


def _fsync_file(path: Path) -> None:
    with path.open("rb") as handle:
        os.fsync(handle.fileno())


def _fsync_directory(path: Path) -> None:
    flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
    fd = os.open(str(path), flags)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def is_generated_run_id(value: str) -> bool:
    """Public predicate used by focused tests and callers displaying IDs."""
    return bool(_GENERATED_RUN_ID_RE.fullmatch(value or ""))

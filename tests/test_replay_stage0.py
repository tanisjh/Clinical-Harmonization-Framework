"""Focused tests for the Stage 0 strict-replay evidence contract.

These tests deliberately do not recompute or compare harmonized values.
"""
from __future__ import annotations

import json
from dataclasses import replace
from decimal import Decimal
from pathlib import Path

import pandas as pd
import pytest
import yaml

from lib.contributions import Contribution, from_source_ref
from lib.extractor_base import BaseExtractor
from lib.provenance import Cell, PROVENANCE_SCHEMA_VERSION
from lib.replay_contract import (
    REASON_CODES,
    REASON_CODES_VERSION,
    REPLAY_API_VERSION,
    REPLAY_STATUS_VERSION,
    ReplayStatus,
    TypedRawValue,
    encode_typed_raw,
)
from lib.replay_manifest import (
    ArtifactBinding,
    EvidenceBinding,
    ManifestError,
    MappingUsage,
    ReplayManifest,
    RuleBinding,
    SpecificationBinding,
    TargetSchemaBinding,
    UnsupportedManifestVersion,
    load as load_manifest,
    parse as parse_manifest,
    producer_binding,
    runtime_dependency_versions,
    source_binding_from_ref,
)
from lib.schema_model import load as load_schema
from lib.source_ref import SourceRef, sha256_file
from lib.spec_model import SpecError, load as load_spec, parse as parse_spec

ROOT = Path(__file__).resolve().parents[1]
FIXTURE = ROOT / "tests/fixtures/synthetic_replay"
SPEC_REF = "synthetic/device_readings_direct@1.0.0"
RUN_ID = "synthetic-run-0001"


def _relative(path: Path) -> str:
    return str(path.resolve().relative_to(ROOT.resolve()))


def _manifest(tmp_path: Path) -> ReplayManifest:
    delivered = tmp_path / "delivered.csv"
    provenance = tmp_path / "provenance.csv"
    delivered.write_text("device_id,reading_id,reading_text\nDEVICE-A,READ-001,001\n")
    provenance.write_text("run_id,mapping_id\nsynthetic-run-0001,direct.reading_text\n")

    source_path = FIXTURE / "sources/device_readings.csv"
    ref = SourceRef.build(
        config_key="readings",
        path=source_path,
        project_root=ROOT,
        file_type="csv",
        reader_id="cidc_csv@v1",
        logical_label="device_readings.csv",
        reader_effective_flags={
            "encoding": "utf-8",
            "header": 0,
            "low_memory": False,
            "skiprows": 0,
        },
        n_rows=2,
    )
    source = source_binding_from_ref(
        ref,
        spec_ref=SPEC_REF,
        source_id="readings",
        dependencies=runtime_dependency_versions(("pandas",)),
    )
    return ReplayManifest(
        run_id=RUN_ID,
        created_at_utc="2026-09-16T12:00:00Z",
        producer=producer_binding("synthetic-test", "1", ("pandas",)),
        target_schema=TargetSchemaBinding(
            identity="schema:synthetic/device_readings@1.0.0",
            path=_relative(FIXTURE / "schema.yaml"),
            sha256=sha256_file(FIXTURE / "schema.yaml"),
            schema_id="synthetic/device_readings",
            schema_version="1.0.0",
        ),
        specifications=(SpecificationBinding(
            identity=f"spec:{SPEC_REF}",
            path=_relative(FIXTURE / "spec.yaml"),
            sha256=sha256_file(FIXTURE / "spec.yaml"),
            spec_id="synthetic/device_readings_direct",
            spec_version="1.0.0",
            status="approved",
            target_schema_ref="synthetic/device_readings@1.0.0",
        ),),
        sources=(source,),
        delivered_artifacts=(ArtifactBinding(
            identity="artifact:synthetic-delivered",
            path=_relative(delivered),
            sha256=sha256_file(delivered),
            artifact_id="synthetic-delivered",
            format="csv",
            n_rows=1,
            schema_ref="synthetic/device_readings@1.0.0",
        ),),
        provenance_artifact=ArtifactBinding(
            identity="artifact:synthetic-provenance",
            path=_relative(provenance),
            sha256=sha256_file(provenance),
            artifact_id="synthetic-provenance",
            format="csv",
            n_rows=1,
            provenance_schema_version=PROVENANCE_SCHEMA_VERSION,
        ),
        rules=(RuleBinding("direct.copy.v1", "1", "mapping"),),
        evidence=(EvidenceBinding(
            identity="configuration:synthetic/device_calibration@1.0.0",
            path=_relative(FIXTURE / "config.yaml"),
            sha256=sha256_file(FIXTURE / "config.yaml"),
            evidence_ref_id="config:calibration",
            kind="configuration",
            reference="synthetic/device_calibration@1.0.0",
        ),),
        mapping_usages=(MappingUsage(
            mapping_id="direct.reading_text",
            spec_ref=SPEC_REF,
            source_ref_ids=(ref.source_ref_id,),
            rule_refs=("direct.copy.v1@1",),
            evidence_ref_ids=(),
        ),),
    )


def test_manifest_round_trip_and_links(tmp_path):
    manifest = _manifest(tmp_path)
    path = tmp_path / "manifest.json"
    manifest.write(path)
    loaded = load_manifest(_relative(path), project_root=ROOT)
    assert loaded == manifest
    assert loaded.run_id == RUN_ID
    assert loaded.mapping_usages[0].source_ref_ids == (
        loaded.sources[0].source_ref_id,
    )
    assert all(check.ok for check in loaded.verify_hashes(ROOT))


def test_manifest_rejects_unsupported_version_and_path_escape(tmp_path):
    doc = _manifest(tmp_path).to_dict()
    doc["manifest_version"] = 99
    with pytest.raises(UnsupportedManifestVersion):
        parse_manifest(doc, project_root=ROOT)

    doc = _manifest(tmp_path).to_dict()
    doc["target_schema"]["path"] = "../outside.yaml"
    with pytest.raises(ManifestError, match="escapes project root"):
        parse_manifest(doc, project_root=ROOT)


def test_hash_capture_and_mismatch_detection(tmp_path):
    manifest = _manifest(tmp_path)
    checks = manifest.verify_hashes(ROOT, mapping_id="direct.reading_text")
    assert checks and all(c.ok for c in checks)

    bad_schema = replace(manifest.target_schema, sha256="0" * 64)
    drifted = replace(manifest, target_schema=bad_schema)
    by_identity = {c.identity: c for c in drifted.verify_hashes(
        ROOT, mapping_id="direct.reading_text")}
    assert not by_identity[bad_schema.identity].ok


def test_csv_reader_metadata_and_exact_dependency_version(tmp_path):
    manifest = _manifest(tmp_path)
    source = manifest.sources[0]
    assert source.reader.reader_id == "cidc_csv@v1"
    assert source.reader.flags == {
        "encoding": "utf-8", "header": 0,
        "low_memory": False, "skiprows": 0,
    }
    assert dict(source.reader.dependencies)["pandas"] == pd.__version__
    assert source.sheet == ""
    assert source.n_rows == 2


def test_mapping_id_parsing_back_compatibility_and_uniqueness():
    schema = load_schema(FIXTURE / "schema.yaml")
    spec = load_spec(FIXTURE / "spec.yaml", target_schema=schema,
                     require_executable=True)
    assert [m.mapping_id for m in spec.mappings] == [
        "direct.reading_text", "direct.sequence_number", "direct.enabled_text",
        "direct.optional_note"
    ]

    doc = {
        "spec_id": "synthetic/legacy", "spec_version": "1.0.0",
        "trial": "synthetic", "target_schema": {
            "id": "synthetic/device_readings", "version": "1.0.0"},
        "sources": [{"id": "s", "path": "x.csv", "file_type": "csv",
                         "record_keys": ["device_id"]}],
        "record": {"anchor": "external", "participant_key": "device_id"},
        "mappings": [{"target": "reading_text", "op": "direct",
                      "source": "s", "column": "x",
                      "rule": {"id": "direct.copy.v1"}}],
    }
    assert parse_spec(doc).mappings[0].mapping_id == ""
    doc["mappings"].append({**doc["mappings"][0],
                            "target": "optional_note", "mapping_id": "same"})
    doc["mappings"][0]["mapping_id"] = "same"
    with pytest.raises(SpecError, match="duplicate mapping_id"):
        parse_spec(doc)


def test_invalid_mapping_id_rejected_and_explicit_ids_ignore_order():
    doc = yaml.safe_load((FIXTURE / "spec.yaml").read_text())
    original = {mapping["target"]: mapping["mapping_id"]
                for mapping in doc["mappings"]}
    reordered = dict(doc)
    reordered["mappings"] = list(reversed(doc["mappings"]))
    parsed = parse_spec(reordered)
    assert {mapping.target: mapping.mapping_id for mapping in parsed.mappings} \
        == original

    invalid = yaml.safe_load((FIXTURE / "spec.yaml").read_text())
    invalid["mappings"][0]["mapping_id"] = "Invalid Mapping ID"
    with pytest.raises(SpecError, match="mapping_id"):
        parse_spec(invalid)


@pytest.mark.parametrize("value,kind,encoded", [
    (None, "null", '{"kind":"null"}'),
    ("001", "string", '{"kind":"string","value":"001"}'),
    (1, "integer", '{"kind":"integer","value":"1"}'),
    (Decimal("1.0"), "number", '{"kind":"number","value":"1.0"}'),
    (True, "boolean", '{"kind":"boolean","value":true}'),
])
def test_typed_raw_evidence(value, kind, encoded):
    evidence = encode_typed_raw(value)
    assert evidence.kind == kind
    assert evidence.to_json() == encoded
    assert TypedRawValue.from_json(encoded) == evidence


def test_typed_null_is_distinct_from_absent_raw_evidence():
    old = Cell("t", "p", "s", "e", "f", None, 1.0,
               "source.csv", "x", 0, "direct")
    assert old.raw_value_json == ""

    class SyntheticExtractor(BaseExtractor):
        trial_dir_name = "tests/fixtures/synthetic_replay"
        template_trial_name = "synthetic_device_lab"

    base = SyntheticExtractor(ROOT, pd.DataFrame(), {}, {}, run_id=RUN_ID,
                              spec_ref=SPEC_REF)
    anchor = {"cimac_part_id": "DEVICE-A", "Cimac.id": "READ-001",
              "Collection_Event": "bench-west"}
    absent = base.cell(anchor, "optional_note", None, 1.0, "source.csv",
                       "optional_note", 0, "direct")
    explicit_null = base.cell(anchor, "optional_note", None, 1.0, "source.csv",
                              "optional_note", 0, "direct", raw_value=None,
                              mapping_id="direct.optional_note")
    assert absent.raw_value_json == ""
    assert explicit_null.raw_value_json == '{"kind":"null"}'
    assert explicit_null.run_id == RUN_ID
    assert explicit_null.spec_ref == SPEC_REF
    assert explicit_null.mapping_id == "direct.optional_note"
    deliberately_omitted = base.cell(
        anchor, "optional_note", None, 1.0, "source.csv", "optional_note",
        0, "direct", raw_value=None, raw_evidence_recorded=False)
    assert deliberately_omitted.raw_value_json == ""


def test_source_ref_and_contribution_backward_compatibility(tmp_path):
    old = Contribution("candidate", "selected", "reading")
    assert old.source_ref_id == old.raw_value_json == old.evidence_refs == ""

    ref = SourceRef.build(
        config_key="readings", path=FIXTURE / "sources/device_readings.csv",
        project_root=ROOT, file_type="csv", reader_id="cidc_csv@v1",
        reader_effective_flags={"encoding": "utf-8", "skiprows": 0},
        n_rows=2,
    )
    contribution = from_source_ref(
        ref, role="candidate", status="selected", source_column="reading_text")
    assert ref.source_ref_id.startswith("src-")
    assert contribution.source_ref_id == ref.source_ref_id


def test_reason_code_contract_is_versioned_and_stable():
    assert REPLAY_API_VERSION == 1
    assert REPLAY_STATUS_VERSION == 1
    assert REASON_CODES_VERSION == 10
    assert {s.value for s in ReplayStatus} == {
        "PASS", "FAIL", "NOT_REPLAYABLE", "REQUEST_ERROR"}
    assert REASON_CODES[ReplayStatus.PASS] == frozenset({"REPLAY_MATCH"})
    assert "SOURCE_HASH_MISMATCH" in REASON_CODES[ReplayStatus.FAIL]
    assert "VALUE_MAP_UNMAPPED_ERROR" in REASON_CODES[ReplayStatus.FAIL]
    assert "UNIT_CONVERSION_NONFINITE_RESULT" in REASON_CODES[ReplayStatus.FAIL]
    assert "DEPENDENCY_CONTEXT_MISMATCH" in REASON_CODES[ReplayStatus.FAIL]
    assert "DEPENDENCY_REPLAY_FAILED" in REASON_CODES[ReplayStatus.FAIL]
    assert "REGISTERED_RULE_EXECUTION_FAILED" in REASON_CODES[ReplayStatus.FAIL]
    assert "FALLBACK_SELECTION_MISMATCH" in REASON_CODES[ReplayStatus.FAIL]
    assert "FALLBACK_CANDIDATE_CONTEXT_MISMATCH" in \
        REASON_CODES[ReplayStatus.FAIL]
    assert "VOCABULARY_IDENTITY_MISMATCH" in \
        REASON_CODES[ReplayStatus.FAIL]
    assert "VALUE_STATE_CONTEXT_MISMATCH" in \
        REASON_CODES[ReplayStatus.FAIL]
    assert "ANCHOR_IDENTITY_MISMATCH" in REASON_CODES[ReplayStatus.FAIL]
    assert "ANCHOR_SELECTION_MISMATCH" in REASON_CODES[ReplayStatus.FAIL]
    assert "DERIVATION_MISMATCH" in REASON_CODES[ReplayStatus.FAIL]
    assert "MAPPING_NOT_IDENTIFIED" in REASON_CODES[ReplayStatus.NOT_REPLAYABLE]
    assert "EXTERNAL_VOCABULARY_UNSUPPORTED" in \
        REASON_CODES[ReplayStatus.NOT_REPLAYABLE]
    assert "PRODUCER_ENVIRONMENT_MISMATCH" in \
        REASON_CODES[ReplayStatus.NOT_REPLAYABLE]
    for code in (
        "DEPENDENCY_REFERENCE_MISSING", "DEPENDENCY_REFERENCE_AMBIGUOUS",
        "DEPENDENCY_NOT_REPLAYABLE", "REPLAY_CYCLE_DETECTED",
        "REPLAY_DEPTH_EXCEEDED", "REGISTERED_RULE_UNAVAILABLE",
        "REGISTERED_RULE_CONTRACT_MISMATCH",
    ):
        assert code in REASON_CODES[ReplayStatus.NOT_REPLAYABLE]
    assert "FALLBACK_EVIDENCE_INCOMPLETE" in \
        REASON_CODES[ReplayStatus.NOT_REPLAYABLE]
    assert "FALLBACK_CANDIDATE_UNSUPPORTED" in \
        REASON_CODES[ReplayStatus.NOT_REPLAYABLE]
    assert "VOCABULARY_EVIDENCE_MISSING" in \
        REASON_CODES[ReplayStatus.NOT_REPLAYABLE]
    assert "ANCHOR_EVIDENCE_MISSING" in \
        REASON_CODES[ReplayStatus.NOT_REPLAYABLE]
    assert "TARGET_AMBIGUOUS" in REASON_CODES[ReplayStatus.REQUEST_ERROR]
    assert "DERIVATION_EVIDENCE_MISSING" in \
        REASON_CODES[ReplayStatus.NOT_REPLAYABLE]


def test_usage_scoped_files_exclude_unreferenced_configuration(tmp_path):
    manifest = _manifest(tmp_path)
    all_paths = {f.path for f in manifest.files_for()}
    used_paths = {f.path for f in manifest.files_for(
        mapping_id="direct.reading_text", artifact_id="synthetic-delivered")}
    config_path = _relative(FIXTURE / "config.yaml")
    assert config_path in all_paths
    assert config_path not in used_paths
    assert _relative(FIXTURE / "spec.yaml") in used_paths
    assert _relative(FIXTURE / "sources/device_readings.csv") in used_paths


def test_csv_loader_emits_structured_effective_flags():
    class SyntheticCsvExtractor(BaseExtractor):
        trial_dir_name = "tests/fixtures/synthetic_replay/sources"
        template_trial_name = "synthetic_device_lab"

    extractor = SyntheticCsvExtractor(
        ROOT, pd.DataFrame(), {"sources": {"readings": "device_readings.csv"}}, {})
    frame = extractor.load_csv("readings")
    ref = extractor.ref("readings")
    assert len(frame) == ref.n_rows == 2
    assert ref.reader_id == "cidc_csv@v1"
    assert dict(ref.reader_effective_flags) == {
        "encoding": "utf-8", "header": 0,
        "low_memory": False, "skiprows": 0,
    }


def test_manifest_requires_exact_reader_dependency_versions(tmp_path):
    manifest = _manifest(tmp_path)
    source = manifest.sources[0]
    reader_without_versions = replace(source.reader, dependencies=())
    with pytest.raises(ManifestError, match="exact reader dependency versions"):
        replace(manifest, sources=(replace(source, reader=reader_without_versions),))

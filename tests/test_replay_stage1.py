"""Synthetic-only tests for strict Stage 1 direct replay."""
from __future__ import annotations

import csv
import inspect
from dataclasses import dataclass, replace
from pathlib import Path

import pandas as pd
import pytest
import yaml

import lib.replay as replay_module
from lib.cidc_io import read_cidc_csv
from lib.provenance import Cell, PROVENANCE_SCHEMA_VERSION
from lib.replay import ReplayRequest, recompute_direct, replay_cell
from lib.replay_contract import (
    REASON_CODES,
    REASON_CODES_VERSION,
    REPLAY_API_VERSION,
    REPLAY_STATUS_VERSION,
    ReplayStatus,
    encode_typed_raw,
)
from lib.replay_manifest import (
    ArtifactBinding,
    MappingUsage,
    ReplayManifest,
    RuleBinding,
    SpecificationBinding,
    TargetSchemaBinding,
    producer_binding,
    runtime_dependency_versions,
    source_binding_from_ref,
)
from lib.source_ref import SourceRef, sha256_file
from lib.spec_model import load as load_spec


ROOT = Path(__file__).resolve().parents[1]
FIXTURE = ROOT / "tests/fixtures/synthetic_replay"
SCHEMA = FIXTURE / "schema.yaml"
SOURCE_FIXTURE = FIXTURE / "sources/device_readings.csv"
SPEC_FIXTURE = FIXTURE / "spec.yaml"
SPEC_REF = "synthetic/device_readings_direct@1.0.0"
SCHEMA_REF = "synthetic/device_readings@1.0.0"
RUN_ID = "synthetic-stage1-run-0001"


@dataclass
class Case:
    manifest: ReplayManifest
    request: ReplayRequest
    source_path: Path
    delivered_path: Path
    provenance_path: Path
    expected_recomputed: object


def _relative(path: Path) -> str:
    return str(path.resolve().relative_to(ROOT.resolve()))


def _csv_scalar(value):
    if value is None:
        return ""
    try:
        if bool(pd.isna(value)):
            return ""
    except (TypeError, ValueError):
        pass
    return value


def _write_rows(path: Path, fieldnames, rows) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _case(tmp_path: Path, mapping_id: str, *,
          delivered_value=...,
          provenance_value=...,
          provenance_updates=None,
          spec_mutator=None,
          duplicate_delivered=False,
          duplicate_provenance=False,
          raw_override=...) -> Case:
    case_dir = tmp_path / mapping_id.replace(".", "_")
    case_dir.mkdir(parents=True)
    source_path = case_dir / "device_readings.csv"
    source_path.write_bytes(SOURCE_FIXTURE.read_bytes())

    spec_doc = yaml.safe_load(SPEC_FIXTURE.read_text())
    spec_doc["sources"][0]["path"] = _relative(source_path)
    if spec_mutator:
        spec_mutator(spec_doc)
    spec_path = case_dir / "spec.yaml"
    spec_path.write_text(yaml.safe_dump(spec_doc, sort_keys=False))
    spec = load_spec(spec_path, require_executable=True)
    mapping = next(m for m in spec.mappings if m.mapping_id == mapping_id)

    read = read_cidc_csv(source_path)
    row = read.df.iloc[0]
    observed_flags = {
        "encoding": read.encoding,
        "header": 0,
        "low_memory": False,
        "skiprows": 1 if read.preamble_skipped else 0,
    }
    source_ref = SourceRef.build(
        config_key="readings",
        path=source_path,
        project_root=ROOT,
        file_type="csv",
        reader_id="cidc_csv@v1",
        logical_label="device_readings.csv",
        reader_effective_flags=observed_flags,
        n_rows=len(read.df),
    )

    if raw_override is ...:
        raw = row[mapping.column] if mapping.column in row.index else None
    else:
        raw = raw_override
    recomputed = recompute_direct(raw, coerce=mapping.coerce)
    if delivered_value is ...:
        delivered_value = recomputed
    if provenance_value is ...:
        provenance_value = delivered_value

    delivered_row = {
        "device_id": "DEVICE-A",
        "reading_id": "READ-001",
        "location": "bench-west",
        "reading_text": "",
        "sequence_number": "1",
        "enabled": "",
        "optional_note": "",
    }
    delivered_row[mapping.target] = _csv_scalar(delivered_value)
    delivered_path = case_dir / "delivered.csv"
    delivered_rows = [delivered_row]
    if duplicate_delivered:
        delivered_rows.append(dict(delivered_row))
    _write_rows(delivered_path, list(delivered_row), delivered_rows)

    record = Cell(
        trial=spec.trial,
        cimac_part_id="DEVICE-A",
        Cimac_id="READ-001",
        Collection_Event="bench-west",
        harmonized_field=mapping.target,
        value=_csv_scalar(provenance_value),
        confidence=1.0,
        source_file=source_ref.logical_label,
        source_column=mapping.column,
        source_row_idx=-1,
        extraction_method="spec:direct",
        source_logical_label=source_ref.logical_label,
        source_resolved_path=source_ref.resolved_path,
        source_sha256=source_ref.sha256,
        source_sheet="",
        source_row_number=0,
        source_record_key="device_id=DEVICE-A",
        rule_id=mapping.rule.id,
        rule_version=mapping.rule.version,
        run_id=RUN_ID,
        spec_ref=spec.ref,
        mapping_id=mapping.mapping_id,
        source_ref_id=source_ref.source_ref_id,
        raw_value_json=encode_typed_raw(raw).to_json(),
    ).as_record()
    record.update(provenance_updates or {})
    provenance_path = case_dir / "provenance.csv"
    provenance_rows = [record]
    if duplicate_provenance:
        provenance_rows.append(dict(record))
    _write_rows(provenance_path, list(record), provenance_rows)

    source_binding = source_binding_from_ref(
        source_ref,
        spec_ref=SPEC_REF,
        source_id="readings",
        dependencies=runtime_dependency_versions(("pandas",)),
    )
    manifest = ReplayManifest(
        run_id=RUN_ID,
        created_at_utc="2026-09-17T12:00:00Z",
        producer=producer_binding("synthetic-stage1-test", "1", ("pandas",)),
        target_schema=TargetSchemaBinding(
            identity=f"schema:{SCHEMA_REF}",
            path=_relative(SCHEMA),
            sha256=sha256_file(SCHEMA),
            schema_id="synthetic/device_readings",
            schema_version="1.0.0",
        ),
        specifications=(SpecificationBinding(
            identity=f"spec:{SPEC_REF}",
            path=_relative(spec_path),
            sha256=sha256_file(spec_path),
            spec_id="synthetic/device_readings_direct",
            spec_version="1.0.0",
            status="approved",
            target_schema_ref=SCHEMA_REF,
        ),),
        sources=(source_binding,),
        delivered_artifacts=(ArtifactBinding(
            identity="artifact:synthetic-delivered",
            path=_relative(delivered_path),
            sha256=sha256_file(delivered_path),
            artifact_id="synthetic-delivered",
            format="csv",
            n_rows=len(delivered_rows),
            schema_ref=SCHEMA_REF,
        ),),
        provenance_artifact=ArtifactBinding(
            identity="artifact:synthetic-provenance",
            path=_relative(provenance_path),
            sha256=sha256_file(provenance_path),
            artifact_id="synthetic-provenance",
            format="csv",
            n_rows=len(provenance_rows),
            provenance_schema_version=PROVENANCE_SCHEMA_VERSION,
        ),
        rules=(RuleBinding(
            mapping.rule.id, mapping.rule.version, "mapping"),),
        mapping_usages=(MappingUsage(
            mapping_id=mapping.mapping_id,
            spec_ref=SPEC_REF,
            source_ref_ids=(source_ref.source_ref_id,),
            rule_refs=(f"{mapping.rule.id}@{mapping.rule.version}",),
        ),),
    )
    request = ReplayRequest(
        artifact_id="synthetic-delivered",
        row_key={"device_id": "DEVICE-A", "reading_id": "READ-001"},
        target_field=mapping.target,
    )
    return Case(manifest, request, source_path, delivered_path,
                provenance_path, recomputed)


@pytest.mark.parametrize("mapping_id,expected", [
    ("direct.reading_text", "001"),
    ("direct.enabled_text", "active state"),
    ("direct.sequence_number", 1.0),
    ("direct.optional_note", None),
])
def test_direct_passes(mapping_id, expected, tmp_path):
    case = _case(tmp_path, mapping_id)
    result = replay_cell(ROOT, case.manifest, case.request)
    assert result.status is ReplayStatus.PASS
    assert result.reason_code == "REPLAY_MATCH"
    if isinstance(expected, float):
        assert result.recomputed_value == expected
    else:
        assert result.recomputed_value is expected or result.recomputed_value == expected


def test_direct_confidence_is_required(tmp_path):
    case = _case(
        tmp_path, "direct.reading_text",
        provenance_updates={"confidence": ""})
    result = replay_cell(ROOT, case.manifest, case.request)
    assert (result.status, result.reason_code) == (
        ReplayStatus.NOT_REPLAYABLE, "MANIFEST_REQUIRED_FIELD_MISSING")


def test_direct_confidence_must_match_pinned_outcome(tmp_path):
    case = _case(
        tmp_path, "direct.reading_text",
        provenance_updates={"confidence": "0.5"})
    result = replay_cell(ROOT, case.manifest, case.request)
    assert (result.status, result.reason_code) == (
        ReplayStatus.FAIL, "CONFIDENCE_CONTEXT_MISMATCH")


@pytest.mark.parametrize("confidence", ["nan", "Infinity", "-0.1", "1.1"])
def test_direct_confidence_must_be_finite_and_in_range(confidence, tmp_path):
    case = _case(
        tmp_path, "direct.reading_text",
        provenance_updates={"confidence": confidence})
    result = replay_cell(ROOT, case.manifest, case.request)
    assert (result.status, result.reason_code) == (
        ReplayStatus.FAIL, "CONFIDENCE_CONTEXT_MISMATCH")


def test_delivered_value_mismatch(tmp_path):
    case = _case(tmp_path, "direct.reading_text",
                 delivered_value="WRONG", provenance_value="WRONG")
    result = replay_cell(ROOT, case.manifest, case.request)
    assert (result.status, result.reason_code) == (
        ReplayStatus.FAIL, "RECOMPUTED_VALUE_MISMATCH")
    assert result.recomputed_value == "001"


def test_provenance_value_mismatch(tmp_path):
    case = _case(tmp_path, "direct.reading_text",
                 provenance_value="WRONG")
    result = replay_cell(ROOT, case.manifest, case.request)
    assert (result.status, result.reason_code) == (
        ReplayStatus.FAIL, "PROVENANCE_VALUE_MISMATCH")
    assert result.recomputed_value == "001"


def test_source_hash_drift(tmp_path):
    case = _case(tmp_path, "direct.reading_text")
    case.source_path.write_text(case.source_path.read_text() + "\n")
    result = replay_cell(ROOT, case.manifest, case.request)
    assert (result.status, result.reason_code) == (
        ReplayStatus.FAIL, "SOURCE_HASH_MISMATCH")


def test_raw_value_drift(tmp_path):
    case = _case(
        tmp_path, "direct.reading_text",
        provenance_updates={
            "raw_value_json": '{"kind":"string","value":"different"}'})
    result = replay_cell(ROOT, case.manifest, case.request)
    assert (result.status, result.reason_code) == (
        ReplayStatus.FAIL, "RAW_VALUE_MISMATCH")


def test_record_key_drift(tmp_path):
    case = _case(
        tmp_path, "direct.reading_text",
        provenance_updates={"source_record_key": "device_id=DEVICE-Z"})
    result = replay_cell(ROOT, case.manifest, case.request)
    assert (result.status, result.reason_code) == (
        ReplayStatus.FAIL, "SOURCE_RECORD_KEY_MISMATCH")


@pytest.mark.parametrize("updates,reason", [
    ({"run_id": ""}, "RUN_ID_LINK_MISSING"),
    ({"run_id": "wrong-run"}, "RUN_ID_MISMATCH"),
    ({"spec_ref": ""}, "SPEC_LINK_MISSING"),
    ({"spec_ref": "synthetic/wrong@1.0.0"}, "SPEC_LINK_MISSING"),
    ({"mapping_id": ""}, "MAPPING_NOT_IDENTIFIED"),
    ({"mapping_id": "direct.wrong"}, "MAPPING_NOT_IDENTIFIED"),
    ({"source_ref_id": ""}, "SOURCE_LINK_MISSING"),
    ({"source_ref_id": "src-wrong"}, "SOURCE_LINK_MISSING"),
])
def test_wrong_or_missing_links(tmp_path, updates, reason):
    case = _case(tmp_path, "direct.reading_text",
                 provenance_updates=updates)
    result = replay_cell(ROOT, case.manifest, case.request)
    assert (result.status, result.reason_code) == (
        ReplayStatus.NOT_REPLAYABLE, reason)


def test_missing_reader_metadata(tmp_path):
    case = _case(tmp_path, "direct.reading_text")
    source = case.manifest.sources[0]
    manifest = replace(
        case.manifest,
        sources=(replace(
            source,
            reader=replace(source.reader, effective_flags=())),),
    )
    result = replay_cell(ROOT, manifest, case.request)
    assert (result.status, result.reason_code) == (
        ReplayStatus.NOT_REPLAYABLE, "READER_METADATA_MISSING")


def test_path_escape_rejected(tmp_path):
    case = _case(tmp_path, "direct.reading_text")
    source = case.manifest.sources[0]
    manifest = replace(
        case.manifest,
        sources=(replace(source, path="../escape.csv"),),
    )
    result = replay_cell(ROOT, manifest, case.request)
    assert (result.status, result.reason_code) == (
        ReplayStatus.NOT_REPLAYABLE, "MANIFEST_PATH_OUTSIDE_PROJECT")


def test_spec_declared_source_path_escape_rejected(tmp_path):
    def escaping_source(doc):
        doc["sources"][0]["path"] = "../escape.csv"

    case = _case(
        tmp_path, "direct.reading_text", spec_mutator=escaping_source)
    result = replay_cell(ROOT, case.manifest, case.request)
    assert (result.status, result.reason_code) == (
        ReplayStatus.NOT_REPLAYABLE, "MANIFEST_PATH_OUTSIDE_PROJECT")


def test_source_row_out_of_range(tmp_path):
    case = _case(
        tmp_path, "direct.reading_text",
        provenance_updates={"source_row_number": "99"})
    result = replay_cell(ROOT, case.manifest, case.request)
    assert (result.status, result.reason_code) == (
        ReplayStatus.FAIL, "SOURCE_ROW_OUT_OF_RANGE")


def test_source_column_missing(tmp_path):
    def missing_column(doc):
        mapping = next(
            m for m in doc["mappings"]
            if m["mapping_id"] == "direct.reading_text")
        mapping["column"] = "missing_column"

    case = _case(
        tmp_path, "direct.reading_text",
        spec_mutator=missing_column, raw_override=None)
    result = replay_cell(ROOT, case.manifest, case.request)
    assert (result.status, result.reason_code) == (
        ReplayStatus.FAIL, "SOURCE_COLUMN_MISSING")


def test_ambiguous_delivered_target_is_request_error(tmp_path):
    case = _case(tmp_path, "direct.reading_text", duplicate_delivered=True)
    result = replay_cell(ROOT, case.manifest, case.request)
    assert (result.status, result.reason_code) == (
        ReplayStatus.REQUEST_ERROR, "TARGET_AMBIGUOUS")


def test_ambiguous_provenance_is_not_replayable(tmp_path):
    case = _case(tmp_path, "direct.reading_text", duplicate_provenance=True)
    result = replay_cell(ROOT, case.manifest, case.request)
    assert (result.status, result.reason_code) == (
        ReplayStatus.NOT_REPLAYABLE, "PROVENANCE_AUTHORITY_AMBIGUOUS")


def test_draft_spec_with_complete_ids_is_not_approved(tmp_path):
    def draft(doc):
        doc["status"] = "draft"

    case = _case(
        tmp_path, "direct.reading_text", spec_mutator=draft)
    result = replay_cell(ROOT, case.manifest, case.request)
    assert (result.status, result.reason_code) == (
        ReplayStatus.NOT_REPLAYABLE, "SPEC_STATUS_NOT_APPROVED")


def test_unsupported_operation_is_not_replayable(tmp_path):
    def no_mapping_configured(doc):
        mapping = next(
            m for m in doc["mappings"]
            if m["mapping_id"] == "direct.reading_text")
        mapping["op"] = "no_mapping_configured"

    case = _case(
        tmp_path, "direct.reading_text", spec_mutator=no_mapping_configured)
    result = replay_cell(ROOT, case.manifest, case.request)
    assert (result.status, result.reason_code) == (
        ReplayStatus.NOT_REPLAYABLE, "OPERATION_UNSUPPORTED")


def test_recomputation_receives_no_delivered_value(monkeypatch, tmp_path):
    case = _case(tmp_path, "direct.reading_text")
    original = replay_module.recompute_direct
    calls = []

    def guarded(*args, **kwargs):
        calls.append((args, kwargs))
        assert len(args) == 1
        assert set(kwargs) == {"coerce"}
        return original(*args, **kwargs)

    monkeypatch.setattr(replay_module, "recompute_direct", guarded)
    result = replay_cell(ROOT, case.manifest, case.request)
    assert result.status is ReplayStatus.PASS
    assert calls and calls[0][0][0] == "001"
    assert tuple(inspect.signature(original).parameters) == (
        "raw_source_value", "coerce")


def test_changing_delivery_does_not_change_recomputed_value(tmp_path):
    good = _case(tmp_path / "good", "direct.reading_text")
    wrong = _case(tmp_path / "wrong", "direct.reading_text",
                  delivered_value="changed", provenance_value="changed")
    good_result = replay_cell(ROOT, good.manifest, good.request)
    wrong_result = replay_cell(ROOT, wrong.manifest, wrong.request)
    assert good_result.recomputed_value == wrong_result.recomputed_value == "001"
    assert good_result.status is ReplayStatus.PASS
    assert wrong_result.reason_code == "RECOMPUTED_VALUE_MISMATCH"


def test_numeric_comparison_has_no_epsilon(tmp_path):
    case = _case(tmp_path, "direct.sequence_number",
                 delivered_value="1.0000000000000001",
                 provenance_value="1.0000000000000001")
    result = replay_cell(ROOT, case.manifest, case.request)
    assert (result.status, result.reason_code) == (
        ReplayStatus.FAIL, "RECOMPUTED_VALUE_MISMATCH")


def test_manifest_can_be_supplied_by_explicit_path(tmp_path):
    case = _case(tmp_path, "direct.reading_text")
    path = tmp_path / "manifest.json"
    case.manifest.write(path)
    result = replay_cell(ROOT, _relative(path), case.request)
    assert result.status is ReplayStatus.PASS


@pytest.mark.parametrize("which,reason", [
    ("delivered", "DELIVERED_ARTIFACT_HASH_MISMATCH"),
    ("provenance", "PROVENANCE_ARTIFACT_HASH_MISMATCH"),
])
def test_artifact_hash_drift(which, reason, tmp_path):
    case = _case(tmp_path, "direct.reading_text")
    path = (case.delivered_path if which == "delivered"
            else case.provenance_path)
    path.write_text(path.read_text() + "\n")
    result = replay_cell(ROOT, case.manifest, case.request)
    assert (result.status, result.reason_code) == (ReplayStatus.FAIL, reason)


def test_schema_and_spec_hash_drift(tmp_path):
    case = _case(tmp_path, "direct.reading_text")
    bad_schema = replace(
        case.manifest,
        target_schema=replace(case.manifest.target_schema, sha256="0" * 64))
    schema_result = replay_cell(ROOT, bad_schema, case.request)
    assert schema_result.reason_code == "SCHEMA_HASH_MISMATCH"

    bad_spec = replace(
        case.manifest,
        specifications=(replace(
            case.manifest.specifications[0], sha256="0" * 64),))
    spec_result = replay_cell(ROOT, bad_spec, case.request)
    assert spec_result.reason_code == "SPEC_HASH_MISMATCH"


def test_reader_environment_and_observation_are_exact(tmp_path):
    case = _case(tmp_path, "direct.reading_text")
    source = case.manifest.sources[0]
    wrong_environment = replace(
        case.manifest,
        sources=(replace(
            source,
            reader=replace(
                source.reader, dependencies=(("pandas", "0.0.0"),))),),
    )
    env_result = replay_cell(ROOT, wrong_environment, case.request)
    assert env_result.reason_code == "READER_ENVIRONMENT_MISMATCH"

    wrong_flags = dict(source.reader.flags)
    wrong_flags["skiprows"] = 1
    wrong_observation = replace(
        case.manifest,
        sources=(replace(
            source,
            reader=replace(
                source.reader,
                effective_flags=tuple(sorted(wrong_flags.items())))),),
    )
    observation_result = replay_cell(ROOT, wrong_observation, case.request)
    assert observation_result.reason_code == "READER_OBSERVATION_MISMATCH"


def test_source_row_count_is_exact(tmp_path):
    case = _case(tmp_path, "direct.reading_text")
    source = case.manifest.sources[0]
    manifest = replace(
        case.manifest, sources=(replace(source, n_rows=source.n_rows + 1),))
    result = replay_cell(ROOT, manifest, case.request)
    assert (result.status, result.reason_code) == (
        ReplayStatus.FAIL, "SOURCE_ROW_COUNT_MISMATCH")


def test_selector_shape_and_api_version_are_request_errors(tmp_path):
    case = _case(tmp_path, "direct.reading_text")
    incomplete = replace(
        case.request, row_key={"device_id": "DEVICE-A"})
    extra = replace(
        case.request,
        row_key={**case.request.row_key, "location": "bench-west"})
    wrong_version = replace(case.request, api_version=99)
    assert (replay_cell(ROOT, case.manifest, incomplete).reason_code
            == "SELECTOR_INCOMPLETE")
    assert (replay_cell(ROOT, case.manifest, extra).reason_code
            == "SELECTOR_HAS_EXTRA_KEYS")
    assert (replay_cell(ROOT, case.manifest, wrong_version).reason_code
            == "REQUEST_MALFORMED")


def test_result_exposes_versioned_contract(tmp_path):
    case = _case(tmp_path, "direct.reading_text")
    result = replay_cell(ROOT, case.manifest, case.request)
    payload = result.to_dict()
    assert payload["api_version"] == REPLAY_API_VERSION
    assert payload["status_version"] == REPLAY_STATUS_VERSION
    assert payload["reason_codes_version"] == REASON_CODES_VERSION
    assert result.reason_code in REASON_CODES[result.status]

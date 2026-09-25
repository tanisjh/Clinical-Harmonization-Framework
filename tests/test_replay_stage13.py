"""Synthetic-only tests for closed exact-byte XLSX strict replay (Stage 13)."""
from __future__ import annotations

import csv
import datetime as dt
import hashlib
import io
import json
import zipfile
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np
import openpyxl
import pandas as pd
import pytest
import yaml

import harmonize_spec
from lib import rule_registry
import lib.replay as replay_module
from lib.replay import ReplayRequest, replay_cell
from lib.replay_contract import ReplayStatus, encode_typed_raw
from lib.replay_manifest import (
    MANIFEST_VERSION, ManifestError, ReaderBinding, parse as parse_manifest,
    runtime_dependency_versions, source_binding_from_ref,
)
from lib.replay_orchestration import (
    ReplayEvidenceCollector, ReplayPublication, ReplayRunContext,
)
from lib.rule_registry import RegisteredRule
from lib.schema_model import contract_for, load as load_schema
from lib.source_ref import SourceRef, sha256_file
from lib.spec_executor import execute
from lib.spec_model import load as load_spec
from lib.spec_preflight import source_columns
from lib.xlsx_reader import (
    XLSX_READER_DEPENDENCIES, XLSX_READER_FLAGS, XLSX_READER_ID,
    XLSX_SHEET_CANONICALIZATION_ID, XlsxReaderError,
    canonical_xlsx_sheet, load_xlsx_source, parse_xlsx_bytes,
    xlsx_sheet_sha256,
)


ROOT = Path(__file__).resolve().parents[1]
FIXTURE = ROOT / "tests/fixtures/synthetic_replay"
SCHEMA = FIXTURE / "schema_stage13.yaml"
SPEC = FIXTURE / "spec_stage13.yaml"
VOCABULARY = FIXTURE / "vocabularies/stage13_ordered.yaml"
RUN_ID = "synthetic-stage13-run-0001"
ARTIFACT_ID = "synthetic-stage13-delivered"


def _one(value):
    if value is None:
        return None, 0.2, "missing"
    return float(value) + 1.0, 0.88, "synthetic one-input"


def _multi(left, right):
    if left is None or right is None:
        return None, 0.2, "missing"
    return float(left) + float(right), 0.89, "synthetic multi-input"


@pytest.fixture(autouse=True)
def _rules(monkeypatch):
    monkeypatch.setitem(
        rule_registry.REGISTRY,
        ("synthetic.stage13.one.v1", "1"),
        RegisteredRule(
            "synthetic.stage13.one.v1", "1", rule_registry.KIND_FIELD,
            _one, arity=1),
    )
    monkeypatch.setitem(
        rule_registry.REGISTRY,
        ("synthetic.stage13.multi.v1", "1"),
        RegisteredRule(
            "synthetic.stage13.multi.v1", "1", rule_registry.KIND_FIELD,
            _multi, arity=2),
    )


@dataclass
class Case:
    manifest: object
    request: ReplayRequest
    spec: object
    cells: list
    source_path: Path
    spec_path: Path
    delivered_path: Path
    provenance_path: Path


def _relative(path: Path) -> str:
    return str(path.resolve().relative_to(ROOT.resolve()))


def _anchor(participant="DEVICE-A"):
    return pd.DataFrame([{
        "device_id": participant,
        "reading_id": f"READ-{participant}",
        "location": "bench-west",
        "cimac_part_id": participant,
        "Cimac.id": f"READ-{participant}",
        "Collection_Event": "bench-west",
    }])


def _workbook(path: Path, *, formula=False, merged=False,
              duplicate_header=False, empty_header=False,
              second_sheet=True) -> None:
    workbook = openpyxl.Workbook()
    sheet = workbook.active
    sheet.title = "Readings"
    header = [
        "device_id", "raw_none", "raw_text", "raw_numeric",
        "inline_code", "raw_units", "vocabulary_code", "blank_value",
        "bool_value",
    ]
    if duplicate_header:
        header[-1] = "raw_none"
    if empty_header:
        header[-1] = None
    sheet.append(header)
    sheet.append([
        "DEVICE-A", "plain", "  spaced   text  ", 12.5,
        "A", 2, "SPECIAL", None, True,
    ])
    sheet.append([
        "DEVICE-A", "later", "later", 99, "A", 9, "SPECIAL", None, False,
    ])
    sheet.append([
        "DEVICE-B", "other", "other", 7, "A", 3, "SPECIAL", None, True,
    ])
    if formula:
        sheet["D2"] = "=6+6.5"
    if merged:
        sheet.merge_cells("H2:I2")
    if second_sheet:
        other = workbook.create_sheet("Other")
        other.append(header)
        other.append([
            "DEVICE-A", "other-sheet", "other-sheet", 1,
            "A", 1, "SPECIAL", None, False,
        ])
    workbook.save(path)


def _write_rows(path: Path, fieldnames, rows) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _read_rows(path: Path):
    with path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        return reader.fieldnames, list(reader)


def _bundle(tmp_path: Path, *, target="direct_numeric",
            duplicate_policy="first") -> Case:
    case_dir = tmp_path / target.replace(".", "_")
    case_dir.mkdir(parents=True)
    source_path = case_dir / "readings.xlsx"
    _workbook(source_path)
    vocabulary_dir = case_dir / "vocabularies"
    vocabulary_dir.mkdir()
    (vocabulary_dir / VOCABULARY.name).write_bytes(VOCABULARY.read_bytes())

    document = yaml.safe_load(SPEC.read_text())
    document["sources"][0]["path"] = _relative(source_path)
    document["record"]["duplicates"]["policy"] = duplicate_policy
    spec_path = case_dir / "spec.yaml"
    spec_path.write_text(yaml.safe_dump(document, sort_keys=False))
    schema = load_schema(SCHEMA)
    spec = load_spec(
        spec_path, target_schema=schema, require_executable=True,
        vocabulary_dir=vocabulary_dir)
    context = ReplayRunContext.create(
        ROOT, schema, (spec,), run_id=RUN_ID,
        application="synthetic-stage13-test")
    cells, diagnostics = execute(
        spec, ROOT, _anchor(), contract=contract_for(schema), run_id=RUN_ID)
    collector = ReplayEvidenceCollector(context)
    collector.collect(spec, diagnostics.sources, cells)

    row = {name: None for name in schema.column_order}
    row.update({"device_id": "DEVICE-A", "reading_id": "READ-DEVICE-A",
                "location": "bench-west"})
    for cell in cells:
        row[cell.harmonized_field] = cell.value
    delivered = pd.DataFrame([row], columns=schema.column_order)
    provenance = pd.DataFrame([
        {name: getattr(cell, name, "")
         for name in harmonize_spec.PROVENANCE_COLUMNS}
        for cell in cells
    ]).rename(columns={"Cimac_id": "Cimac.id"})
    published = ReplayPublication(
        context, collector, case_dir / "published").publish(
            delivered, provenance, artifact_id=ARTIFACT_ID,
            delivered_name="delivered.csv", provenance_name="provenance.csv")
    request = ReplayRequest(
        artifact_id=ARTIFACT_ID,
        row_key={"device_id": "DEVICE-A", "reading_id": "READ-DEVICE-A"},
        target_field=target)
    return Case(
        published.manifest, request, spec, cells, source_path, spec_path,
        published.delivered_path, published.provenance_path)


def _mapping(case: Case):
    return next(mapping for mapping in case.spec.mappings
                if mapping.target == case.request.target_field)


def _usage(case: Case):
    mapping = _mapping(case)
    return next(usage for usage in case.manifest.mapping_usages
                if usage.mapping_id == mapping.mapping_id)


def _source(case: Case):
    return next(source for source in case.manifest.sources
                if source.source_id == "readings")


def _replace_source(case: Case, source) -> None:
    current = _source(case)
    case.manifest = replace(
        case.manifest,
        sources=tuple(source if item == current else item
                      for item in case.manifest.sources))


def _mutate_provenance(case: Case, **updates) -> None:
    fields, rows = _read_rows(case.provenance_path)
    mapping = _mapping(case)
    row = next(item for item in rows if item["mapping_id"] == mapping.mapping_id)
    row.update(updates)
    _write_rows(case.provenance_path, fields, rows)
    case.manifest = replace(
        case.manifest,
        provenance_artifact=replace(
            case.manifest.provenance_artifact,
            sha256=sha256_file(case.provenance_path)))


def _mutate_delivery(case: Case, value) -> None:
    fields, rows = _read_rows(case.delivered_path)
    rows[0][case.request.target_field] = value
    _write_rows(case.delivered_path, fields, rows)
    artifact = case.manifest.delivered_artifacts[0]
    case.manifest = replace(
        case.manifest,
        delivered_artifacts=(replace(
            artifact, sha256=sha256_file(case.delivered_path)),))


@pytest.mark.parametrize("target,expected", [
    ("direct_none", "plain"),
    ("direct_text", "spaced text"),
    ("direct_numeric", 12.5),
    ("inline_label", "mapped-a"),
    ("converted_value", 60.0),
    ("vocabulary_label", "vocabulary-hit"),
    ("recursive_one", 13.5),
    ("recursive_multi", 73.5),
])
def test_stage13_operations_and_recursion_pass(tmp_path, target, expected):
    case = _bundle(tmp_path, target=target)
    result = replay_cell(ROOT, case.manifest, case.request)
    assert (result.status, result.reason_code, result.recomputed_value) == (
        ReplayStatus.PASS, "REPLAY_MATCH", expected), result.message


def test_existing_operation_helpers_are_reused(tmp_path, monkeypatch):
    case = _bundle(tmp_path, target="recursive_multi")
    calls = {"direct": 0, "unit": 0, "rule": 0}
    direct = replay_module.recompute_direct
    unit = replay_module.recompute_unit_conversion
    rule = replay_module.recompute_registered_rule

    def direct_spy(*args, **kwargs):
        calls["direct"] += 1
        return direct(*args, **kwargs)

    def unit_spy(*args, **kwargs):
        calls["unit"] += 1
        return unit(*args, **kwargs)

    def rule_spy(*args, **kwargs):
        calls["rule"] += 1
        return rule(*args, **kwargs)

    monkeypatch.setattr(replay_module, "recompute_direct", direct_spy)
    monkeypatch.setattr(replay_module, "recompute_unit_conversion", unit_spy)
    monkeypatch.setattr(replay_module, "recompute_registered_rule", rule_spy)
    assert replay_cell(ROOT, case.manifest, case.request).status is ReplayStatus.PASS
    assert calls == {"direct": 1, "unit": 1, "rule": 2}

    inline = _bundle(tmp_path / "inline", target="inline_label")
    vocabulary = _bundle(tmp_path / "vocab", target="vocabulary_label")
    map_calls = []
    vocabulary_calls = []
    map_impl = replay_module.recompute_value_map
    vocab_impl = replay_module.recompute_vocabulary_value_map
    monkeypatch.setattr(
        replay_module, "recompute_value_map",
        lambda *args, **kwargs: (map_calls.append(1),
                                 map_impl(*args, **kwargs))[1])
    monkeypatch.setattr(
        replay_module, "recompute_vocabulary_value_map",
        lambda *args, **kwargs: (vocabulary_calls.append(1),
                                 vocab_impl(*args, **kwargs))[1])
    assert replay_cell(ROOT, inline.manifest, inline.request).status is ReplayStatus.PASS
    assert replay_cell(ROOT, vocabulary.manifest, vocabulary.request).status is ReplayStatus.PASS
    assert map_calls == [1]
    assert vocabulary_calls == [1]


def test_manifest_v2_binds_complete_xlsx_observation(tmp_path):
    case = _bundle(tmp_path)
    source = _source(case)
    assert MANIFEST_VERSION == case.manifest.manifest_version == 2
    assert source.reader.reader_id == XLSX_READER_ID
    assert source.reader.flags == XLSX_READER_FLAGS
    assert set(dict(source.reader.dependencies)) == set(XLSX_READER_DEPENDENCIES)
    assert source.sheet == "Readings"
    assert source.n_rows == 3
    assert source.parsed_columns[0] == "device_id"
    assert source.parsed_canonicalization == XLSX_SHEET_CANONICALIZATION_ID
    assert len(source.parsed_sheet_sha256) == 64


def test_preflight_uses_the_closed_xlsx_reader(tmp_path):
    case = _bundle(tmp_path)
    assert source_columns(case.spec, "readings", ROOT) == _source(case).parsed_columns


def test_manifest_publication_contract_fails_closed_without_sheet_identity(tmp_path):
    case = _bundle(tmp_path)
    source = _source(case)
    incomplete = replace(
        source, parsed_columns=(), parsed_canonicalization="",
        parsed_sheet_sha256="")
    with pytest.raises(ManifestError, match="incomplete XLSX"):
        replace(case.manifest, sources=(incomplete,))


def test_producer_observation_reads_path_once_and_is_toctou_safe(
        tmp_path, monkeypatch):
    path = tmp_path / "once.xlsx"
    _workbook(path)
    original_payload = path.read_bytes()
    original = Path.read_bytes
    reads = []

    def changing_read(candidate):
        payload = original(candidate)
        if candidate == path:
            reads.append(candidate)
            candidate.write_bytes(b"changed-after-capture")
        return payload

    monkeypatch.setattr(Path, "read_bytes", changing_read)
    observation = load_xlsx_source(
        path, project_root=tmp_path, config_key="readings",
        sheet_name="Readings")
    assert reads == [path]
    assert observation.source_ref.sha256 == hashlib.sha256(original_payload).hexdigest()
    assert observation.frame_copy().iloc[0]["raw_none"] == "plain"


def test_replay_reads_workbook_once_and_caches_same_payload(tmp_path, monkeypatch):
    case = _bundle(tmp_path, target="recursive_multi")
    original = Path.read_bytes
    reads = []

    def counted(candidate):
        if candidate == case.source_path:
            reads.append(candidate)
        return original(candidate)

    monkeypatch.setattr(Path, "read_bytes", counted)
    assert replay_cell(ROOT, case.manifest, case.request).status is ReplayStatus.PASS
    assert reads == [case.source_path]


@pytest.mark.parametrize("missing", [False, True])
def test_changed_or_missing_workbook_fails(tmp_path, missing):
    case = _bundle(tmp_path)
    if missing:
        case.source_path.unlink()
        expected = "SOURCE_FILE_MISSING"
    else:
        _workbook(case.source_path, second_sheet=False)
        expected = "SOURCE_HASH_MISMATCH"
    result = replay_cell(ROOT, case.manifest, case.request)
    assert (result.status, result.reason_code) == (ReplayStatus.FAIL, expected)


@pytest.mark.parametrize("flag,value", [
    ("engine", "calamine"),
    ("engine_read_only", False),
    ("engine_data_only", False),
    ("engine_keep_links", True),
    ("header", 1),
])
def test_reader_flag_contradictions_fail(tmp_path, flag, value):
    case = _bundle(tmp_path)
    source = _source(case)
    flags = dict(source.reader.effective_flags)
    flags[flag] = value
    _replace_source(case, replace(
        source, reader=replace(
            source.reader, effective_flags=tuple(sorted(flags.items())))))
    result = replay_cell(ROOT, case.manifest, case.request)
    assert (result.status, result.reason_code) == (
        ReplayStatus.FAIL, "READER_OBSERVATION_MISMATCH")


@pytest.mark.parametrize("dependency", ["pandas", "openpyxl", "numpy"])
def test_reader_dependency_contradictions_fail(tmp_path, dependency):
    case = _bundle(tmp_path)
    source = _source(case)
    dependencies = dict(source.reader.dependencies)
    dependencies[dependency] = "0.invalid"
    _replace_source(case, replace(
        source, reader=replace(
            source.reader, dependencies=tuple(sorted(dependencies.items())))))
    result = replay_cell(ROOT, case.manifest, case.request)
    assert (result.status, result.reason_code) == (
        ReplayStatus.FAIL, "READER_OBSERVATION_MISMATCH")


def test_python_and_consistent_dependency_environment_drift_fail(tmp_path):
    case = _bundle(tmp_path / "python")
    case.manifest = replace(
        case.manifest,
        producer=replace(case.manifest.producer, python_version="0.invalid"))
    result = replay_cell(ROOT, case.manifest, case.request)
    assert (result.status, result.reason_code) == (
        ReplayStatus.FAIL, "READER_ENVIRONMENT_MISMATCH")

    case = _bundle(tmp_path / "dependency")
    source = _source(case)
    reader_dependencies = dict(source.reader.dependencies)
    reader_dependencies["openpyxl"] = "0.invalid"
    producer_dependencies = dict(case.manifest.producer.dependencies)
    producer_dependencies["openpyxl"] = "0.invalid"
    _replace_source(case, replace(
        source, reader=replace(source.reader, dependencies=tuple(sorted(
            reader_dependencies.items())))))
    case.manifest = replace(
        case.manifest,
        producer=replace(case.manifest.producer, dependencies=tuple(sorted(
            producer_dependencies.items()))))
    result = replay_cell(ROOT, case.manifest, case.request)
    assert (result.status, result.reason_code) == (
        ReplayStatus.FAIL, "READER_ENVIRONMENT_MISMATCH")


def test_parsed_sheet_identity_contradictions_fail(tmp_path):
    case = _bundle(tmp_path / "digest")
    source = _source(case)
    _replace_source(case, replace(source, parsed_sheet_sha256="0" * 64))
    result = replay_cell(ROOT, case.manifest, case.request)
    assert (result.status, result.reason_code) == (
        ReplayStatus.FAIL, "READER_OBSERVATION_MISMATCH")

    case = _bundle(tmp_path / "columns")
    source = _source(case)
    _replace_source(case, replace(
        source, parsed_columns=tuple(reversed(source.parsed_columns))))
    result = replay_cell(ROOT, case.manifest, case.request)
    assert (result.status, result.reason_code) == (
        ReplayStatus.FAIL, "READER_OBSERVATION_MISMATCH")


def test_independent_row_key_column_and_raw_checks(tmp_path):
    case = _bundle(tmp_path / "row")
    _mutate_provenance(case, source_row_number="1")
    result = replay_cell(ROOT, case.manifest, case.request)
    assert (result.status, result.reason_code) == (
        ReplayStatus.FAIL, "SOURCE_RECORD_KEY_MISMATCH")

    case = _bundle(tmp_path / "key")
    _mutate_provenance(case, source_record_key="device_id=DEVICE-B")
    result = replay_cell(ROOT, case.manifest, case.request)
    assert (result.status, result.reason_code) == (
        ReplayStatus.FAIL, "SOURCE_RECORD_KEY_MISMATCH")

    case = _bundle(tmp_path / "column")
    _mutate_provenance(case, source_column="raw_text")
    result = replay_cell(ROOT, case.manifest, case.request)
    assert (result.status, result.reason_code) == (
        ReplayStatus.FAIL, "SOURCE_DECLARATION_MISMATCH")

    case = _bundle(tmp_path / "raw")
    _mutate_provenance(
        case, raw_value_json=encode_typed_raw(999).to_json())
    result = replay_cell(ROOT, case.manifest, case.request)
    assert (result.status, result.reason_code) == (
        ReplayStatus.FAIL, "RAW_VALUE_MISMATCH")


def test_fresh_sheet_missing_declared_source_column_fails(tmp_path):
    case = _bundle(tmp_path)
    workbook = openpyxl.load_workbook(case.source_path)
    sheet = workbook["Readings"]
    numeric_position = next(
        cell.column for cell in sheet[1] if cell.value == "raw_numeric")
    sheet.delete_cols(numeric_position)
    workbook.save(case.source_path)
    observation = load_xlsx_source(
        case.source_path, project_root=ROOT, config_key="readings",
        sheet_name="Readings", logical_label="stage13_readings.xlsx")
    binding = source_binding_from_ref(
        observation.source_ref, spec_ref=case.spec.ref, source_id="readings",
        dependencies=runtime_dependency_versions(XLSX_READER_DEPENDENCIES))
    old_id = _source(case).source_ref_id
    fields, rows = _read_rows(case.provenance_path)
    for row in rows:
        if row["source_ref_id"] == old_id:
            row["source_ref_id"] = binding.source_ref_id
            row["source_sha256"] = binding.sha256
    _write_rows(case.provenance_path, fields, rows)
    case.manifest = replace(
        case.manifest,
        sources=(binding,),
        provenance_artifact=replace(
            case.manifest.provenance_artifact,
            sha256=sha256_file(case.provenance_path)),
        mapping_usages=tuple(replace(
            usage,
            source_ref_ids=tuple(
                binding.source_ref_id if value == old_id else value
                for value in usage.source_ref_ids))
            for usage in case.manifest.mapping_usages))
    result = replay_cell(ROOT, case.manifest, case.request)
    assert (result.status, result.reason_code) == (
        ReplayStatus.FAIL, "SOURCE_COLUMN_MISSING")


def test_duplicate_first_and_error_semantics(tmp_path):
    first = _bundle(tmp_path / "first", target="direct_none")
    assert replay_cell(ROOT, first.manifest, first.request).recomputed_value == "plain"
    with pytest.raises(Exception, match="duplicate policy is 'error'"):
        _bundle(tmp_path / "error", duplicate_policy="error")


def test_outputs_and_recorded_raw_cannot_drive_reconstruction(tmp_path):
    case = _bundle(tmp_path / "delivery")
    _mutate_delivery(case, "999")
    result = replay_cell(ROOT, case.manifest, case.request)
    assert (result.status, result.reason_code, result.recomputed_value) == (
        ReplayStatus.FAIL, "PROVENANCE_VALUE_MISMATCH", 12.5)

    case = _bundle(tmp_path / "provenance")
    _mutate_provenance(case, value="999")
    result = replay_cell(ROOT, case.manifest, case.request)
    assert (result.status, result.reason_code, result.recomputed_value) == (
        ReplayStatus.FAIL, "PROVENANCE_VALUE_MISMATCH", 12.5)

    case = _bundle(tmp_path / "raw")
    _mutate_provenance(case, raw_value_json=encode_typed_raw(999).to_json())
    result = replay_cell(ROOT, case.manifest, case.request)
    assert result.reason_code == "RAW_VALUE_MISMATCH"
    assert result.recomputed_value is None


def test_usage_scope_cannot_be_repaired_by_run_wide_other_sheet(tmp_path):
    case = _bundle(tmp_path)
    other = load_xlsx_source(
        case.source_path, project_root=ROOT, config_key="readings",
        sheet_name="Other", logical_label="stage13_readings.xlsx")
    other_binding = source_binding_from_ref(
        other.source_ref, spec_ref=case.spec.ref, source_id="readings",
        dependencies=runtime_dependency_versions(XLSX_READER_DEPENDENCIES))
    usage = _usage(case)
    case.manifest = replace(
        case.manifest,
        sources=case.manifest.sources + (other_binding,),
        mapping_usages=tuple(
            replace(item, source_ref_ids=()) if item == usage else item
            for item in case.manifest.mapping_usages))
    result = replay_cell(ROOT, case.manifest, case.request)
    assert (result.status, result.reason_code) == (
        ReplayStatus.NOT_REPLAYABLE, "SOURCE_NOT_SINGLE_FILE")


def test_old_excel_history_remains_not_replayable(tmp_path):
    case = _bundle(tmp_path)
    document = yaml.safe_load(case.spec_path.read_text())
    document["sources"][0]["reader"] = "excel@v1"
    case.spec_path.write_text(yaml.safe_dump(document, sort_keys=False))
    old_ref = SourceRef.build(
        config_key="readings", path=case.source_path, project_root=ROOT,
        file_type="xlsx", reader_id="excel@v1",
        logical_label="stage13_readings.xlsx", sheet="Readings", n_rows=3)
    old_binding = source_binding_from_ref(
        old_ref, spec_ref=case.spec.ref, source_id="readings",
        dependencies=runtime_dependency_versions(("pandas", "openpyxl")))

    fields, rows = _read_rows(case.provenance_path)
    for row in rows:
        if row["source_ref_id"]:
            row["source_ref_id"] = old_ref.source_ref_id
    _write_rows(case.provenance_path, fields, rows)
    old_source_id = _source(case).source_ref_id
    specification = case.manifest.specifications[0]
    case.manifest = replace(
        case.manifest,
        manifest_version=1,
        specifications=(replace(
            specification, sha256=sha256_file(case.spec_path)),),
        sources=(old_binding,),
        provenance_artifact=replace(
            case.manifest.provenance_artifact,
            sha256=sha256_file(case.provenance_path)),
        mapping_usages=tuple(replace(
            usage,
            source_ref_ids=tuple(
                old_ref.source_ref_id if source == old_source_id else source
                for source in usage.source_ref_ids))
            for usage in case.manifest.mapping_usages))
    result = replay_cell(ROOT, case.manifest, case.request)
    assert (result.status, result.reason_code) == (
        ReplayStatus.NOT_REPLAYABLE, "READER_METADATA_MISSING")


def test_manifest_v1_non_xlsx_shape_remains_readable(tmp_path):
    case = _bundle(tmp_path)
    source = _source(case)
    historical = replace(
        source, file_type="csv", sheet="",
        reader=ReaderBinding(
            "cidc_csv@v1",
            (("encoding", "utf-8"), ("header", 0),
             ("low_memory", False), ("skiprows", 0)),
            (("pandas", pd.__version__),)),
        parsed_columns=(), parsed_canonicalization="", parsed_sheet_sha256="")
    manifest = replace(
        case.manifest, manifest_version=1, sources=(historical,),
        mapping_usages=tuple(replace(
            usage, source_ref_ids=(historical.source_ref_id,)
            if usage.source_ref_ids else ())
            for usage in case.manifest.mapping_usages))
    parsed = parse_manifest(manifest.to_dict(), project_root=ROOT)
    assert parsed.manifest_version == 1
    assert parsed.sources[0].parsed_columns == ()


def test_wrong_reader_and_sheet_identity_are_not_substituted(tmp_path):
    case = _bundle(tmp_path / "reader")
    source = _source(case)
    _replace_source(case, replace(
        source, reader=replace(source.reader, reader_id="excel@v1")))
    result = replay_cell(ROOT, case.manifest, case.request)
    assert result.status is ReplayStatus.FAIL

    path = tmp_path / "sheets.xlsx"
    _workbook(path)
    payload = path.read_bytes()
    readings = parse_xlsx_bytes(payload, sheet_name="Readings")
    other = parse_xlsx_bytes(payload, sheet_name="Other")
    assert xlsx_sheet_sha256(
        readings, sheet_name="Readings") != xlsx_sheet_sha256(
            other, sheet_name="Other")
    with pytest.raises(XlsxReaderError, match="no exact sheet"):
        parse_xlsx_bytes(payload, sheet_name="Missing")
    with pytest.raises(XlsxReaderError, match="exact sheet name"):
        parse_xlsx_bytes(payload, sheet_name=0)
    with pytest.raises(XlsxReaderError, match="exact sheet name"):
        parse_xlsx_bytes(payload, sheet_name=None)


@pytest.mark.parametrize("shape", ["formula", "cached_formula", "merged",
                                    "duplicate", "empty"])
def test_closed_workbook_shapes_are_refused(tmp_path, shape):
    path = tmp_path / f"{shape}.xlsx"
    _workbook(
        path, formula=shape in {"formula", "cached_formula"},
        merged=shape == "merged", duplicate_header=shape == "duplicate",
        empty_header=shape == "empty")
    if shape == "cached_formula":
        source = path.read_bytes()
        output = io.BytesIO()
        with zipfile.ZipFile(io.BytesIO(source)) as incoming, \
                zipfile.ZipFile(output, "w") as outgoing:
            for item in incoming.infolist():
                data = incoming.read(item.filename)
                if item.filename == "xl/worksheets/sheet1.xml":
                    data = data.replace(b"<v></v>", b"<v>12.5</v>", 1)
                outgoing.writestr(item, data)
        path.write_bytes(output.getvalue())
    message = {
        "formula": "formula", "cached_formula": "formula",
        "merged": "merged", "duplicate": "duplicate", "empty": "heading",
    }[shape]
    with pytest.raises(XlsxReaderError, match=message):
        load_xlsx_source(
            path, project_root=tmp_path, config_key="readings",
            sheet_name="Readings")


@pytest.mark.parametrize("suffix", [".xls", ".xlsb", ".xlsm", ".ods"])
def test_unsupported_workbook_types_are_refused(tmp_path, suffix):
    real = tmp_path / "real.xlsx"
    _workbook(real)
    path = tmp_path / f"wrong{suffix}"
    path.write_bytes(real.read_bytes())
    with pytest.raises(XlsxReaderError, match="only .xlsx"):
        load_xlsx_source(
            path, project_root=tmp_path, config_key="readings",
            sheet_name="Readings")


def test_canonical_sheet_is_stable_ordered_typed_and_index_free():
    frame = pd.DataFrame({
        "text": [" café  ", None],
        "integer": [1, 2],
        "number": [1.5, np.nan],
        "boolean": [True, False],
        "date": [dt.date(2026, 1, 2), dt.date(2026, 1, 3)],
        "time": [dt.time(4, 5, 6, 7000), dt.time(4, 5)],
        "datetime": [dt.datetime(2026, 1, 2, 3, 4, 5, 6000),
                     pd.Timestamp("2026-01-03 04:05:06.007")],
    })
    first = canonical_xlsx_sheet(frame, sheet_name="Readings")
    second = canonical_xlsx_sheet(frame.copy(), sheet_name="Readings")
    changed_index = frame.copy()
    changed_index.index = [10, 20]
    assert first == second == canonical_xlsx_sheet(
        changed_index, sheet_name="Readings")
    document = json.loads(first)
    assert document["rows"][0][0] == {"kind": "string", "value": " café  "}
    assert document["rows"][0][1] == {"kind": "integer", "value": "1"}
    assert document["rows"][0][3] == {"kind": "boolean", "value": True}
    assert [cell["kind"] for cell in document["rows"][0][4:]] == [
        "date", "time", "datetime"]
    assert canonical_xlsx_sheet(
        frame.iloc[::-1].reset_index(drop=True), sheet_name="Readings") != first
    assert canonical_xlsx_sheet(
        frame[list(reversed(frame.columns))], sheet_name="Readings") != first
    assert canonical_xlsx_sheet(frame, sheet_name="Other") != first


@pytest.mark.parametrize("value", [
    float("inf"), complex(1, 2), object(), pd.Timedelta(days=1),
    pd.Period("2026-01", freq="M"),
    pd.Timestamp("2026-01-01", tz="UTC"),
    dt.time(1, 2, tzinfo=dt.timezone.utc),
])
def test_canonical_sheet_refuses_unsupported_values(value):
    with pytest.raises(XlsxReaderError):
        canonical_xlsx_sheet(pd.DataFrame({"value": [value]}),
                             sheet_name="Readings")


def test_mutated_protected_frame_is_detected(tmp_path):
    path = tmp_path / "mutation.xlsx"
    _workbook(path)
    observation = load_xlsx_source(
        path, project_root=tmp_path, config_key="readings",
        sheet_name="Readings")
    execution_copy = observation.frame_copy()
    execution_copy.loc[0, "raw_none"] = "caller mutation"
    assert observation.frame_copy().loc[0, "raw_none"] == "plain"
    observation._frame.loc[0, "raw_none"] = "internal mutation"
    with pytest.raises(XlsxReaderError, match="no longer matches"):
        observation.frame_copy()


def test_consumed_temporal_remains_outside_typed_raw_contract():
    with pytest.raises(ValueError, match="supported scalar"):
        encode_typed_raw(pd.Timestamp("2026-01-01"))

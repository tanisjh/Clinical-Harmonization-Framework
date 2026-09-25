"""Synthetic-only tests for strict Stage 2 scalar replay."""
from __future__ import annotations

import csv
import inspect
from dataclasses import dataclass, replace
from pathlib import Path

import pandas as pd
import pytest
import yaml

import lib.vocabularies as vocabularies_module
from lib.cidc_io import read_cidc_csv
from lib.provenance import Cell, PROVENANCE_SCHEMA_VERSION
from lib.replay import (
    ReplayRequest,
    recompute_unit_conversion,
    recompute_value_map,
    replay_cell,
)
from lib.replay_contract import ReplayStatus, encode_typed_raw
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
from lib.schema_model import load as load_schema
from lib.source_ref import SourceRef, sha256_file
from lib.spec_model import OP_UNIT_CONVERSION, OP_VALUE_MAP, load as load_spec


ROOT = Path(__file__).resolve().parents[1]
FIXTURE = ROOT / "tests/fixtures/synthetic_replay"
SCHEMA = FIXTURE / "schema_stage2.yaml"
SOURCE_FIXTURE = FIXTURE / "sources/stage2_readings.csv"
SPEC_FIXTURE = FIXTURE / "spec_stage2.yaml"
SPEC_REF = "synthetic/device_stage2_scalar@1.0.0"
SCHEMA_REF = "synthetic/device_stage2@1.0.0"
RUN_ID = "synthetic-stage2-run-0001"


@dataclass
class Case:
    manifest: ReplayManifest
    request: ReplayRequest
    source_path: Path
    spec_path: Path
    delivered_path: Path
    provenance_path: Path
    recomputed: object


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


def _replace_source_values(path: Path, updates: dict[str, object]) -> None:
    with path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)
        fieldnames = reader.fieldnames
    rows[0].update(updates)
    _write_rows(path, fieldnames, rows)


def _mapping_rules(mapping):
    rules = [mapping.rule]
    for candidate in (mapping.no_value_rule, mapping.unmapped_rule):
        if candidate is not None:
            rules.append(candidate)
    by_ref = {f"{rule.id}@{rule.version}": rule for rule in rules}
    return tuple(by_ref[key] for key in sorted(by_ref))


def _outcome_context(mapping, outcome):
    if outcome.branch == "value":
        return mapping.rule, mapping.confidence
    if outcome.branch == "no_value":
        return mapping.no_value_rule or mapping.rule, mapping.confidence_no_value
    return mapping.unmapped_rule or mapping.rule, mapping.confidence_unmapped


def _case(tmp_path: Path, mapping_id: str, *, source_updates=None,
          spec_mutator=None, delivered_value=..., provenance_value=...,
          provenance_updates=None, permit_recompute_error=False) -> Case:
    case_dir = tmp_path / mapping_id.replace(".", "_")
    case_dir.mkdir(parents=True)
    source_path = case_dir / "stage2_readings.csv"
    source_path.write_bytes(SOURCE_FIXTURE.read_bytes())
    if source_updates:
        _replace_source_values(source_path, source_updates)

    spec_doc = yaml.safe_load(SPEC_FIXTURE.read_text())
    spec_doc["sources"][0]["path"] = _relative(source_path)
    if spec_mutator:
        spec_mutator(spec_doc)
    spec_path = case_dir / "spec.yaml"
    spec_path.write_text(yaml.safe_dump(spec_doc, sort_keys=False))
    schema = load_schema(SCHEMA)
    spec = load_spec(spec_path, target_schema=schema, require_executable=True,
                     resolve_vocabularies=False)
    mapping = next(item for item in spec.mappings
                   if item.mapping_id == mapping_id)

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
        logical_label="stage2_readings.csv",
        reader_effective_flags=observed_flags,
        n_rows=len(read.df),
    )
    raw = row[mapping.column]

    outcome = None
    try:
        if mapping.op == OP_VALUE_MAP:
            outcome = recompute_value_map(
                raw, mapping_table=mapping.map,
                on_unmapped=mapping.on_unmapped)
        elif mapping.op == OP_UNIT_CONVERSION:
            outcome = recompute_unit_conversion(
                raw, factor=mapping.factor, rounding=mapping.round)
        else:  # pragma: no cover - fixture invariant
            raise AssertionError(mapping.op)
    except ValueError:
        if not permit_recompute_error:
            raise

    recomputed = None if outcome is None else outcome.value
    if delivered_value is ...:
        delivered_value = recomputed
    if provenance_value is ...:
        provenance_value = delivered_value

    delivered_row = {field.name: "" for field in schema.fields}
    delivered_row.update({
        "device_id": "DEVICE-A",
        "reading_id": "READ-001",
        "location": "bench-west",
        mapping.target: _csv_scalar(delivered_value),
    })
    delivered_path = case_dir / "delivered.csv"
    _write_rows(delivered_path, schema.column_order, [delivered_row])

    selected_rule = mapping.rule
    selected_confidence = mapping.confidence
    if outcome is not None:
        selected_rule, selected_confidence = _outcome_context(mapping, outcome)
    # A deliberately incomplete branch needs a syntactically valid event so
    # replay, rather than fixture construction, can issue the refusal.
    event_confidence = (0.5 if selected_confidence is None
                        else selected_confidence)
    record = Cell(
        trial=spec.trial,
        cimac_part_id="DEVICE-A",
        Cimac_id="READ-001",
        Collection_Event="bench-west",
        harmonized_field=mapping.target,
        value=_csv_scalar(provenance_value),
        confidence=event_confidence,
        source_file=source_ref.logical_label,
        source_column=mapping.column,
        source_row_idx=-1,
        extraction_method=f"spec:{mapping.op}",
        source_logical_label=source_ref.logical_label,
        source_resolved_path=source_ref.resolved_path,
        source_sha256=source_ref.sha256,
        source_sheet="",
        source_row_number=0,
        source_record_key="device_id=DEVICE-A",
        rule_id=selected_rule.id,
        rule_version=selected_rule.version,
        run_id=RUN_ID,
        spec_ref=spec.ref,
        mapping_id=mapping.mapping_id,
        source_ref_id=source_ref.source_ref_id,
        raw_value_json=encode_typed_raw(raw).to_json(),
    ).as_record()
    record.update(provenance_updates or {})
    provenance_path = case_dir / "provenance.csv"
    _write_rows(provenance_path, list(record), [record])

    rules = _mapping_rules(mapping)
    source_binding = source_binding_from_ref(
        source_ref,
        spec_ref=SPEC_REF,
        source_id="readings",
        dependencies=runtime_dependency_versions(("pandas",)),
    )
    manifest = ReplayManifest(
        run_id=RUN_ID,
        created_at_utc="2026-09-17T12:00:00Z",
        producer=producer_binding(
            "synthetic-stage2-test", "1", ("pandas", "yaml")),
        target_schema=TargetSchemaBinding(
            identity=f"schema:{SCHEMA_REF}",
            path=_relative(SCHEMA),
            sha256=sha256_file(SCHEMA),
            schema_id="synthetic/device_stage2",
            schema_version="1.0.0",
        ),
        specifications=(SpecificationBinding(
            identity=f"spec:{SPEC_REF}",
            path=_relative(spec_path),
            sha256=sha256_file(spec_path),
            spec_id="synthetic/device_stage2_scalar",
            spec_version="1.0.0",
            status="approved",
            target_schema_ref=SCHEMA_REF,
            vocabulary_refs=tuple(spec.vocabularies),
        ),),
        sources=(source_binding,),
        delivered_artifacts=(ArtifactBinding(
            identity="artifact:synthetic-stage2-delivered",
            path=_relative(delivered_path),
            sha256=sha256_file(delivered_path),
            artifact_id="synthetic-stage2-delivered",
            format="csv",
            n_rows=1,
            schema_ref=SCHEMA_REF,
        ),),
        provenance_artifact=ArtifactBinding(
            identity="artifact:synthetic-stage2-provenance",
            path=_relative(provenance_path),
            sha256=sha256_file(provenance_path),
            artifact_id="synthetic-stage2-provenance",
            format="csv",
            n_rows=1,
            provenance_schema_version=PROVENANCE_SCHEMA_VERSION,
        ),
        rules=tuple(RuleBinding(rule.id, rule.version, "mapping")
                    for rule in rules),
        mapping_usages=(MappingUsage(
            mapping_id=mapping.mapping_id,
            spec_ref=SPEC_REF,
            source_ref_ids=(source_ref.source_ref_id,),
            rule_refs=tuple(f"{rule.id}@{rule.version}" for rule in rules),
        ),),
    )
    request = ReplayRequest(
        artifact_id="synthetic-stage2-delivered",
        row_key={"device_id": "DEVICE-A", "reading_id": "READ-001"},
        target_field=mapping.target,
    )
    return Case(manifest, request, source_path, spec_path, delivered_path,
                provenance_path, recomputed)


@pytest.mark.parametrize("raw", ["ACTIVE", "  ACTIVE\n "])
def test_inline_value_map_exact_and_whitespace_normalized_hits_pass(tmp_path, raw):
    case = _case(tmp_path, "map.status", source_updates={"status_code": raw})
    result = replay_cell(ROOT, case.manifest, case.request)
    assert (result.status, result.reason_code, result.recomputed_value) == (
        ReplayStatus.PASS, "REPLAY_MATCH", "ready")


def test_inline_value_map_absent_input_passes_typed_null(tmp_path):
    case = _case(tmp_path, "map.status", source_updates={"status_code": ""})
    result = replay_cell(ROOT, case.manifest, case.request)
    assert result.status is ReplayStatus.PASS
    assert result.recomputed_value is None


def test_inline_value_map_passthrough_preserves_original_scalar(tmp_path):
    case = _case(tmp_path, "map.passthrough")
    result = replay_cell(ROOT, case.manifest, case.request)
    assert result.status is ReplayStatus.PASS
    assert result.recomputed_value == "  unknown  "


def test_inline_value_map_unmapped_null_passes(tmp_path):
    case = _case(tmp_path, "map.null")
    result = replay_cell(ROOT, case.manifest, case.request)
    assert result.status is ReplayStatus.PASS
    assert result.recomputed_value is None


def test_inline_value_map_unmapped_error_is_contradictory_failure(tmp_path):
    case = _case(tmp_path, "map.error", source_updates={"error_code": "UNKNOWN"},
                 delivered_value="claimed",
                 provenance_value="claimed", permit_recompute_error=True)
    result = replay_cell(ROOT, case.manifest, case.request)
    assert (result.status, result.reason_code) == (
        ReplayStatus.FAIL, "VALUE_MAP_UNMAPPED_ERROR")


def test_value_map_wrong_outcome_rule_fails(tmp_path):
    case = _case(tmp_path, "map.null", provenance_updates={
        "rule_id": "map.inline.v1", "rule_version": "1"})
    result = replay_cell(ROOT, case.manifest, case.request)
    assert (result.status, result.reason_code) == (
        ReplayStatus.FAIL, "RULE_IDENTITY_MISMATCH")


def test_missing_outcome_rule_context_is_not_replayable(tmp_path):
    case = _case(tmp_path, "map.null", provenance_updates={
        "rule_id": "", "rule_version": ""})
    result = replay_cell(ROOT, case.manifest, case.request)
    assert (result.status, result.reason_code) == (
        ReplayStatus.NOT_REPLAYABLE, "MANIFEST_REQUIRED_FIELD_MISSING")


def test_value_map_wrong_outcome_confidence_fails(tmp_path):
    case = _case(tmp_path, "map.null", provenance_updates={"confidence": "0.9"})
    result = replay_cell(ROOT, case.manifest, case.request)
    assert (result.status, result.reason_code) == (
        ReplayStatus.FAIL, "CONFIDENCE_CONTEXT_MISMATCH")


def test_changed_inline_map_bytes_fail_spec_hash(tmp_path):
    case = _case(tmp_path, "map.status")
    doc = yaml.safe_load(case.spec_path.read_text())
    mapping = next(item for item in doc["mappings"]
                   if item["mapping_id"] == "map.status")
    mapping["map"]["ACTIVE"] = "changed"
    case.spec_path.write_text(yaml.safe_dump(doc, sort_keys=False))
    result = replay_cell(ROOT, case.manifest, case.request)
    assert (result.status, result.reason_code) == (
        ReplayStatus.FAIL, "SPEC_HASH_MISMATCH")


def test_historical_external_vocabulary_without_evidence_is_not_replayable(
        tmp_path, monkeypatch):
    def fallback(doc):
        mapping = next(item for item in doc["mappings"]
                       if item["mapping_id"] == "map.status")
        mapping["fallback_vocabulary"] = "synthetic/missing@1"

    case = _case(tmp_path, "map.status", spec_mutator=fallback)

    def forbidden(*args, **kwargs):
        raise AssertionError("replay attempted external vocabulary discovery")

    monkeypatch.setattr(vocabularies_module, "resolve", forbidden)
    result = replay_cell(ROOT, case.manifest, case.request)
    assert (result.status, result.reason_code) == (
        ReplayStatus.NOT_REPLAYABLE, "VOCABULARY_EVIDENCE_MISSING")


def test_missing_value_map_outcome_confidence_is_not_replayable(tmp_path):
    def remove_unmapped_confidence(doc):
        mapping = next(item for item in doc["mappings"]
                       if item["mapping_id"] == "map.null")
        mapping.pop("unmapped_rule")
        mapping["confidence"].pop("unmapped")

    case = _case(tmp_path, "map.null", spec_mutator=remove_unmapped_confidence)
    result = replay_cell(ROOT, case.manifest, case.request)
    assert (result.status, result.reason_code) == (
        ReplayStatus.NOT_REPLAYABLE, "MANIFEST_REQUIRED_FIELD_MISSING")


def test_basic_round_false_unit_conversion_passes(tmp_path):
    case = _case(tmp_path, "unit.basic")
    result = replay_cell(ROOT, case.manifest, case.request)
    assert (result.status, result.reason_code, result.recomputed_value) == (
        ReplayStatus.PASS, "REPLAY_MATCH", 10.0)


def test_round_true_unit_conversion_passes(tmp_path):
    case = _case(tmp_path, "unit.rounded",
                 source_updates={"rounded_input": "7"})
    result = replay_cell(ROOT, case.manifest, case.request)
    assert (result.status, result.recomputed_value) == (ReplayStatus.PASS, 4.0)


def test_unit_conversion_rounding_uses_python_ties_to_even(tmp_path):
    case = _case(tmp_path, "unit.rounded")
    result = replay_cell(ROOT, case.manifest, case.request)
    assert (result.status, result.recomputed_value) == (ReplayStatus.PASS, 2.0)


@pytest.mark.parametrize("mapping_id", ["unit.missing", "unit.bad"])
def test_unit_conversion_missing_or_unparsable_passes_typed_null(
        tmp_path, mapping_id):
    case = _case(tmp_path, mapping_id)
    result = replay_cell(ROOT, case.manifest, case.request)
    assert result.status is ReplayStatus.PASS
    assert result.recomputed_value is None


@pytest.mark.parametrize("change", ["factor", "round"])
def test_changed_unit_parameters_fail_spec_hash(tmp_path, change):
    case = _case(tmp_path, "unit.basic")
    doc = yaml.safe_load(case.spec_path.read_text())
    mapping = next(item for item in doc["mappings"]
                   if item["mapping_id"] == "unit.basic")
    mapping[change] = 3.0 if change == "factor" else True
    case.spec_path.write_text(yaml.safe_dump(doc, sort_keys=False))
    result = replay_cell(ROOT, case.manifest, case.request)
    assert (result.status, result.reason_code) == (
        ReplayStatus.FAIL, "SPEC_HASH_MISMATCH")


def test_nonfinite_unit_factor_is_not_replayable(tmp_path):
    def nonfinite(doc):
        mapping = next(item for item in doc["mappings"]
                       if item["mapping_id"] == "unit.basic")
        mapping["factor"] = float("inf")

    case = _case(tmp_path, "unit.basic", spec_mutator=nonfinite,
                 delivered_value="10", provenance_value="10",
                 permit_recompute_error=True)
    result = replay_cell(ROOT, case.manifest, case.request)
    assert (result.status, result.reason_code) == (
        ReplayStatus.NOT_REPLAYABLE, "UNIT_CONVERSION_PARAMETERS_INVALID")


def test_nonfinite_unit_result_cannot_pass(tmp_path):
    case = _case(tmp_path, "unit.basic",
                 source_updates={"measurement": "1e308"},
                 delivered_value="10", provenance_value="10",
                 permit_recompute_error=True)
    result = replay_cell(ROOT, case.manifest, case.request)
    assert (result.status, result.reason_code) == (
        ReplayStatus.FAIL, "UNIT_CONVERSION_NONFINITE_RESULT")


@pytest.mark.parametrize("updates,reason", [
    ({"rule_id": "unit.convert.v1", "rule_version": "1"},
     "RULE_IDENTITY_MISMATCH"),
    ({"confidence": "0.95"}, "CONFIDENCE_CONTEXT_MISMATCH"),
])
def test_unit_no_value_wrong_rule_or_confidence_fails(tmp_path, updates, reason):
    case = _case(tmp_path, "unit.missing", provenance_updates=updates)
    result = replay_cell(ROOT, case.manifest, case.request)
    assert (result.status, result.reason_code) == (ReplayStatus.FAIL, reason)


@pytest.mark.parametrize("component", ["python", "yaml"])
def test_stage2_environment_mismatch_is_not_replayable(tmp_path, component):
    case = _case(tmp_path, "unit.basic")
    producer = case.manifest.producer
    if component == "python":
        producer = replace(producer, python_version="0.0.0")
    else:
        dependencies = dict(producer.dependencies)
        dependencies["yaml"] = "0.0.0"
        producer = replace(producer,
                           dependencies=tuple(sorted(dependencies.items())))
    manifest = replace(case.manifest, producer=producer)
    result = replay_cell(ROOT, manifest, case.request)
    assert (result.status, result.reason_code) == (
        ReplayStatus.NOT_REPLAYABLE, "PRODUCER_ENVIRONMENT_MISMATCH")


def test_unit_numeric_comparison_has_no_epsilon(tmp_path):
    case = _case(tmp_path, "unit.basic", delivered_value="10.000000000000001",
                 provenance_value="10.000000000000001")
    result = replay_cell(ROOT, case.manifest, case.request)
    assert (result.status, result.reason_code) == (
        ReplayStatus.FAIL, "RECOMPUTED_VALUE_MISMATCH")


@pytest.mark.parametrize("function,keywords", [
    (recompute_value_map, {"mapping_table", "on_unmapped"}),
    (recompute_unit_conversion, {"factor", "rounding"}),
])
def test_stage2_recomputation_signatures_exclude_output_values(function, keywords):
    signature = inspect.signature(function)
    assert tuple(signature.parameters)[0] == "raw_source_value"
    assert set(signature.parameters) == {"raw_source_value", *keywords}
    assert "delivered" not in str(signature)
    assert "provenance" not in str(signature)


@pytest.mark.parametrize("mapping_id", ["map.status", "unit.basic"])
def test_changing_delivery_does_not_change_stage2_recomputation(
        tmp_path, mapping_id):
    expected = "ready" if mapping_id == "map.status" else 10.0
    changed = "different" if mapping_id == "map.status" else "11"
    case = _case(tmp_path, mapping_id, delivered_value=changed,
                 provenance_value=changed)
    result = replay_cell(ROOT, case.manifest, case.request)
    assert (result.status, result.reason_code) == (
        ReplayStatus.FAIL, "RECOMPUTED_VALUE_MISMATCH")
    assert result.recomputed_value == expected


@pytest.mark.parametrize("mapping_id", ["map.status", "unit.basic"])
@pytest.mark.parametrize("drift,reason", [
    ("source", "SOURCE_HASH_MISMATCH"),
    ("raw", "RAW_VALUE_MISMATCH"),
    ("key", "SOURCE_RECORD_KEY_MISMATCH"),
])
def test_stage2_preserves_source_raw_and_key_drift_failures(
        tmp_path, mapping_id, drift, reason):
    updates = None
    if drift == "raw":
        updates = {"raw_value_json": encode_typed_raw("wrong").to_json()}
    elif drift == "key":
        updates = {"source_record_key": "device_id=OTHER"}
    case = _case(tmp_path, mapping_id, provenance_updates=updates)
    if drift == "source":
        case.source_path.write_bytes(case.source_path.read_bytes() + b"\n")
    result = replay_cell(ROOT, case.manifest, case.request)
    assert (result.status, result.reason_code) == (ReplayStatus.FAIL, reason)


def test_stage2_result_keeps_api_versions_stable(tmp_path):
    case = _case(tmp_path, "map.status")
    result = replay_cell(ROOT, case.manifest, case.request)
    assert (result.api_version, result.status_version,
           result.reason_codes_version) == (1, 1, 10)

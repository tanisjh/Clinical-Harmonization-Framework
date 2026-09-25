"""Synthetic-only tests for source-independent scalar constant replay."""
from __future__ import annotations

import csv
import inspect
from dataclasses import dataclass, replace
from pathlib import Path

import pandas as pd
import pytest

import harmonize_spec
import lib.replay as replay_module
from lib import rule_registry
from lib.replay import ReplayRequest, recompute_constant, replay_cell
from lib.replay_contract import ReplayStatus, encode_typed_raw
from lib.replay_orchestration import (
    ReplayEvidenceCollector,
    ReplayPublication,
    ReplayRunContext,
)
from lib.rule_registry import RegisteredRule
from lib.schema_model import contract_for, load as load_schema
from lib.source_ref import sha256_file
from lib.spec_executor import execute
from lib.spec_model import load as load_spec


ROOT = Path(__file__).resolve().parents[1]
FIXTURE = ROOT / "tests/fixtures/synthetic_replay"
SCHEMA = FIXTURE / "schema_stage6.yaml"
SPEC = FIXTURE / "spec_stage6.yaml"
RUN_ID = "synthetic-stage6-run-0001"
ARTIFACT_ID = "synthetic-stage6-delivered"


def _pair(left, right):
    if left is None or right is None:
        return None, 0.2, "missing"
    return f"{left}:{right}", 0.88, "synthetic pair"


@pytest.fixture(autouse=True)
def _rules(monkeypatch):
    monkeypatch.setitem(rule_registry.REGISTRY,
        ("synthetic.constant_pair.v1", "1"), RegisteredRule(
            "synthetic.constant_pair.v1", "1", rule_registry.KIND_FIELD,
            _pair, arity=2))


@dataclass
class Case:
    manifest: object
    request: ReplayRequest
    spec: object
    cells: list
    delivered_path: Path
    provenance_path: Path
    spec_path: Path


def _anchor():
    return pd.DataFrame([{
        "device_id": "DEVICE-A", "reading_id": "READ-001",
        "location": "bench-west", "cimac_part_id": "DEVICE-A",
        "Cimac.id": "READ-001", "Collection_Event": "bench-west",
    }])


def _write_rows(path: Path, fieldnames, rows) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _read_rows(path: Path):
    with path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        return reader.fieldnames, list(reader)


def _bundle(tmp_path: Path, *, target="constant_string") -> Case:
    case_dir = tmp_path / target
    case_dir.mkdir(parents=True)
    spec_path = case_dir / "spec.yaml"
    spec_path.write_bytes(SPEC.read_bytes())
    schema = load_schema(SCHEMA)
    spec = load_spec(spec_path, target_schema=schema, require_executable=True)
    context = ReplayRunContext.create(
        ROOT, schema, (spec,), run_id=RUN_ID,
        application="synthetic-stage6-test")
    cells, diagnostics = execute(
        spec, ROOT, _anchor(), contract=contract_for(schema), run_id=RUN_ID)
    collector = ReplayEvidenceCollector(context)
    collector.collect(spec, diagnostics.sources, cells)

    row = {name: None for name in schema.column_order}
    row.update({"device_id": "DEVICE-A", "reading_id": "READ-001",
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
        row_key={"device_id": "DEVICE-A", "reading_id": "READ-001"},
        target_field=target)
    return Case(published.manifest, request, spec, cells,
                published.delivered_path, published.provenance_path, spec_path)


def _event(rows, mapping_id):
    return next(row for row in rows if row["mapping_id"] == mapping_id)


def _mapping_id(case: Case, target=None):
    wanted = target or case.request.target_field
    return next(mapping.mapping_id for mapping in case.spec.mappings
                if mapping.target == wanted)


def _mutate_provenance(case: Case, mapping_id=None, **updates) -> None:
    fields, rows = _read_rows(case.provenance_path)
    _event(rows, mapping_id or _mapping_id(case)).update(updates)
    _write_rows(case.provenance_path, fields, rows)
    case.manifest = replace(
        case.manifest,
        provenance_artifact=replace(
            case.manifest.provenance_artifact,
            sha256=sha256_file(case.provenance_path)))


def _mutate_delivery(case: Case, field: str, value: str) -> None:
    fields, rows = _read_rows(case.delivered_path)
    rows[0][field] = value
    _write_rows(case.delivered_path, fields, rows)
    binding = case.manifest.delivered_artifacts[0]
    case.manifest = replace(
        case.manifest,
        delivered_artifacts=(replace(
            binding, sha256=sha256_file(case.delivered_path)),))


@pytest.mark.parametrize("target,expected,expected_type", [
    ("constant_string", "alpha", str),
    ("constant_integer", 7, int),
    ("constant_float", 2.5, float),
    ("constant_boolean", True, bool),
    ("constant_null", None, type(None)),
    ("constant_null_default_rule", None, type(None)),
    ("constant_transformed", "transformed-value", str),
    ("constant_delivered", "delivered-value", str),
    ("constant_state_none", "state-omitted", str),
    ("constant_raw_none", "no-snapshot", str),
])
def test_scalar_constant_passes_with_exact_declared_type(
        tmp_path, target, expected, expected_type):
    case = _bundle(tmp_path, target=target)
    result = replay_cell(ROOT, case.manifest, case.request)
    assert (result.status, result.reason_code, result.recomputed_value) == (
        ReplayStatus.PASS, "REPLAY_MATCH", expected)
    assert type(result.recomputed_value) is expected_type


def test_pure_constant_helper_has_closed_output_only_signature():
    signature = inspect.signature(recompute_constant)
    assert tuple(signature.parameters) == ("declared_value",)
    assert signature.parameters["declared_value"].kind is \
        inspect.Parameter.KEYWORD_ONLY
    values = [None, "alpha", True, 7, 2.5]
    outcomes = [recompute_constant(declared_value=value) for value in values]
    assert [outcome.branch for outcome in outcomes] == [
        "constant_null", "constant_value", "constant_value",
        "constant_value", "constant_value"]
    assert [type(outcome.value) for outcome in outcomes] == [
        type(None), str, bool, int, float]


@pytest.mark.parametrize("value", [float("nan"), float("inf"),
                                   float("-inf"), ["x"], {"x": 1}, ("x",)])
def test_pure_constant_helper_refuses_non_scalar_or_nonfinite(value):
    with pytest.raises(ValueError, match="constant literal"):
        recompute_constant(declared_value=value)


def test_numeric_comparison_has_no_epsilon(tmp_path):
    case = _bundle(tmp_path, target="constant_float")
    _mutate_delivery(case, "constant_float", "2.5000000001")
    _mutate_provenance(case, value="2.5000000001")
    result = replay_cell(ROOT, case.manifest, case.request)
    assert (result.status, result.reason_code, result.recomputed_value) == (
        ReplayStatus.FAIL, "RECOMPUTED_VALUE_MISMATCH", 2.5)


def test_boolean_is_not_recomputed_as_integer(tmp_path):
    case = _bundle(tmp_path, target="constant_boolean")
    _mutate_delivery(case, "constant_boolean", "1")
    _mutate_provenance(case, value="1")
    result = replay_cell(ROOT, case.manifest, case.request)
    assert (result.status, result.reason_code, result.recomputed_value) == (
        ReplayStatus.FAIL, "RECOMPUTED_VALUE_MISMATCH", True)
    assert type(result.recomputed_value) is bool


@pytest.mark.parametrize("target,rule,confidence", [
    ("constant_null", "constant.null.v1", 0.21),
    ("constant_null_default_rule", "constant.scalar.v1", 0.22),
])
def test_null_uses_exact_no_value_rule_and_confidence(
        tmp_path, target, rule, confidence):
    case = _bundle(tmp_path, target=target)
    cell = next(item for item in case.cells
                if item.mapping_id == _mapping_id(case))
    assert (cell.value, cell.rule_id, cell.confidence, cell.value_state) == (
        None, rule, confidence, "")
    assert cell.raw_value_json == encode_typed_raw(None).to_json()
    result = replay_cell(ROOT, case.manifest, case.request)
    assert result.status is ReplayStatus.PASS


@pytest.mark.parametrize("target,expected_state", [
    ("constant_string", ""),
    ("constant_delivered", "delivered"),
    ("constant_transformed", "transformed"),
    ("constant_state_none", ""),
])
def test_effective_value_state_matches_executor_directive(
        tmp_path, target, expected_state):
    case = _bundle(tmp_path, target=target)
    cell = next(item for item in case.cells
                if item.mapping_id == _mapping_id(case))
    assert cell.value_state == expected_state
    assert replay_cell(ROOT, case.manifest, case.request).status is \
        ReplayStatus.PASS


@pytest.mark.parametrize("field,value,reason", [
    ("rule_id", "wrong.rule", "RULE_IDENTITY_MISMATCH"),
    ("confidence", "0.01", "CONFIDENCE_CONTEXT_MISMATCH"),
    ("value_state", "transformed", "VALUE_STATE_CONTEXT_MISMATCH"),
    ("source_file", "WRONG:origin", "SOURCE_DECLARATION_MISMATCH"),
    ("source_logical_label", "WRONG:label",
     "SOURCE_DECLARATION_MISMATCH"),
    ("source_column", "wrong_column", "SOURCE_DECLARATION_MISMATCH"),
    ("source_record_key", "wrong=key", "SOURCE_DECLARATION_MISMATCH"),
    ("source_resolved_path", "some/file.csv",
     "SOURCE_DECLARATION_MISMATCH"),
    ("source_sha256", "0" * 64, "SOURCE_DECLARATION_MISMATCH"),
    ("source_sheet", "Sheet1", "SOURCE_DECLARATION_MISMATCH"),
    ("source_row_number", "0", "SOURCE_DECLARATION_MISMATCH"),
    ("extraction_method", "spec:direct", "SOURCE_DECLARATION_MISMATCH"),
    ("source_ref_id", "src-claimed", "SOURCE_DECLARATION_MISMATCH"),
    ("evidence_refs", "config-claimed", "SOURCE_DECLARATION_MISMATCH"),
    ("contributions", '[{"status":"input"}]',
     "SOURCE_DECLARATION_MISMATCH"),
    ("derivation", '{"kind":"claimed"}', "SOURCE_DECLARATION_MISMATCH"),
])
def test_constant_context_contradictions_fail(
        tmp_path, field, value, reason):
    case = _bundle(tmp_path)
    _mutate_provenance(case, **{field: value})
    result = replay_cell(ROOT, case.manifest, case.request)
    assert (result.status, result.reason_code) == (ReplayStatus.FAIL, reason)


def test_wrong_constant_artifact_authority_is_not_replayable(tmp_path):
    case = _bundle(tmp_path)
    _mutate_provenance(case, authoritative_for="other.csv")
    result = replay_cell(ROOT, case.manifest, case.request)
    assert (result.status, result.reason_code) == (
        ReplayStatus.NOT_REPLAYABLE, "PROVENANCE_NOT_AUTHORITATIVE")


@pytest.mark.parametrize("state", [
    "source_missing", "not_applicable", "suppressed", "row_excluded",
])
def test_recorded_state_cannot_change_pinned_constant_authority(
        tmp_path, state):
    case = _bundle(tmp_path)
    _mutate_provenance(case, value_state=state)
    result = replay_cell(ROOT, case.manifest, case.request)
    assert (result.status, result.reason_code) == (
        ReplayStatus.FAIL, "VALUE_STATE_CONTEXT_MISMATCH")


def test_post_rule_event_remains_not_replayable(tmp_path):
    case = _bundle(tmp_path, target="constant_parent")
    _mutate_provenance(case, extraction_method="post:synthetic_override")
    result = replay_cell(ROOT, case.manifest, case.request)
    assert (result.status, result.reason_code) == (
        ReplayStatus.NOT_REPLAYABLE, "POST_RULE_UNSUPPORTED_STAGE1")


def test_typed_raw_snapshot_is_verified_but_not_an_input(tmp_path):
    case = _bundle(tmp_path)
    _mutate_provenance(
        case, raw_value_json=encode_typed_raw("tampered").to_json())
    result = replay_cell(ROOT, case.manifest, case.request)
    assert (result.status, result.reason_code) == (
        ReplayStatus.FAIL, "RAW_VALUE_MISMATCH")
    assert tuple(inspect.signature(recompute_constant).parameters) == (
        "declared_value",)


def test_declared_raw_omission_requires_exact_absence(tmp_path):
    case = _bundle(tmp_path, target="constant_raw_none")
    result = replay_cell(ROOT, case.manifest, case.request)
    assert result.status is ReplayStatus.PASS
    _mutate_provenance(
        case, raw_value_json=encode_typed_raw("no-snapshot").to_json())
    result = replay_cell(ROOT, case.manifest, case.request)
    assert (result.status, result.reason_code) == (
        ReplayStatus.FAIL, "RAW_VALUE_MISMATCH")


def test_missing_required_raw_snapshot_is_historical_incompleteness(tmp_path):
    case = _bundle(tmp_path)
    _mutate_provenance(case, raw_value_json="")
    result = replay_cell(ROOT, case.manifest, case.request)
    assert (result.status, result.reason_code) == (
        ReplayStatus.NOT_REPLAYABLE, "RAW_VALUE_ENCODING_MISSING")


@pytest.mark.parametrize("identity", ["python", "yaml"])
def test_constant_environment_pins_python_and_yaml(tmp_path, identity):
    case = _bundle(tmp_path)
    producer = case.manifest.producer
    if identity == "python":
        producer = replace(producer, python_version="0.0.0")
    else:
        dependencies = dict(producer.dependencies)
        dependencies["yaml"] = "0.0.0"
        producer = replace(producer,
                           dependencies=tuple(sorted(dependencies.items())))
    case.manifest = replace(case.manifest, producer=producer)
    result = replay_cell(ROOT, case.manifest, case.request)
    assert (result.status, result.reason_code) == (
        ReplayStatus.NOT_REPLAYABLE, "PRODUCER_ENVIRONMENT_MISMATCH")


def test_pandas_mismatch_does_not_block_constant_replay(tmp_path):
    case = _bundle(tmp_path)
    dependencies = dict(case.manifest.producer.dependencies)
    dependencies["pandas"] = "0.0.0"
    producer = replace(
        case.manifest.producer,
        dependencies=tuple(sorted(dependencies.items())))
    case.manifest = replace(case.manifest, producer=producer)
    result = replay_cell(ROOT, case.manifest, case.request)
    assert result.status is ReplayStatus.PASS


@pytest.mark.parametrize("target,reason", [
    ("constant_conditional", "NAMED_READ_EVIDENCE_MISSING"),
    ("constant_anchor", "ANCHOR_SOURCE_UNSUPPORTED_STAGE1"),
    ("absent_declared", "ANCHOR_SOURCE_UNSUPPORTED_STAGE1"),
    ("no_mapping_value", "ANCHOR_SOURCE_UNSUPPORTED_STAGE1"),
    ("constant_collection", "OPERATION_UNSUPPORTED"),
    ("constant_nonfinite", "OPERATION_UNSUPPORTED"),
])
def test_unsupported_shapes_remain_not_replayable(tmp_path, target, reason):
    case = _bundle(tmp_path, target=target)
    result = replay_cell(ROOT, case.manifest, case.request)
    assert (result.status, result.reason_code) == (
        ReplayStatus.NOT_REPLAYABLE, reason)


@pytest.mark.parametrize("changes", [
    {"column_by_member": {"member": "column"}},
    {"source": "unused_readings"},
    {"group": "synthetic_group"},
])
def test_source_context_constant_scope_is_closed(tmp_path, changes):
    case = _bundle(tmp_path)
    mapping = next(item for item in case.spec.mappings
                   if item.mapping_id == "constant.string")
    mapping = replace(mapping, **changes)
    fields, rows = _read_rows(case.provenance_path)
    event = _event(rows, "constant.string")
    with pytest.raises(replay_module._Stop) as caught:
        replay_module._check_constant_scope(
            mapping, case.spec, event, case.manifest.delivered_artifacts[0])
    assert (caught.value.status, caught.value.reason_code) == (
        ReplayStatus.NOT_REPLAYABLE, "SOURCE_NOT_SINGLE_FILE")


def test_delivery_and_provenance_cannot_change_recomputation(tmp_path):
    case = _bundle(tmp_path)
    _mutate_delivery(case, "constant_string", "tampered")
    _mutate_provenance(case, value="tampered")
    result = replay_cell(ROOT, case.manifest, case.request)
    assert (result.status, result.reason_code, result.recomputed_value) == (
        ReplayStatus.FAIL, "RECOMPUTED_VALUE_MISMATCH", "alpha")

    other = _bundle(tmp_path / "provenance-only")
    _mutate_provenance(other, value="tampered")
    result = replay_cell(ROOT, other.manifest, other.request)
    assert (result.status, result.reason_code, result.recomputed_value) == (
        ReplayStatus.FAIL, "PROVENANCE_VALUE_MISMATCH", "alpha")


def test_complete_v2_event_passes_but_missing_usage_does_not(tmp_path):
    case = _bundle(tmp_path)
    assert case.manifest.manifest_version == 2
    assert replay_cell(ROOT, case.manifest, case.request).status is ReplayStatus.PASS
    case.manifest = replace(
        case.manifest,
        mapping_usages=tuple(
            usage for usage in case.manifest.mapping_usages
            if usage.mapping_id != "constant.string"))
    result = replay_cell(ROOT, case.manifest, case.request)
    assert (result.status, result.reason_code) == (
        ReplayStatus.NOT_REPLAYABLE, "MAPPING_NOT_IDENTIFIED")


def test_exact_spec_hash_remains_required(tmp_path):
    case = _bundle(tmp_path)
    case.spec_path.write_bytes(case.spec_path.read_bytes() + b"\n")
    result = replay_cell(ROOT, case.manifest, case.request)
    assert (result.status, result.reason_code) == (
        ReplayStatus.FAIL, "SPEC_HASH_MISMATCH")


def test_constant_parent_and_stage14_sibling_unlock_downstream(tmp_path):
    parent = _bundle(tmp_path / "parent", target="constant_parent")
    assert replay_cell(ROOT, parent.manifest, parent.request).status is \
        ReplayStatus.PASS

    downstream = _bundle(tmp_path / "downstream", target="downstream_blocked")
    result = replay_cell(ROOT, downstream.manifest, downstream.request)
    assert (result.status, result.reason_code, result.recomputed_value) == (
        ReplayStatus.PASS, "REPLAY_MATCH", "other:derived-context")


def test_stage6_versions_change_only_reason_contract(tmp_path):
    case = _bundle(tmp_path)
    result = replay_cell(ROOT, case.manifest, case.request)
    assert (result.api_version, result.status_version,
            result.reason_codes_version,
            case.manifest.manifest_version) == (1, 1, 10, 2)

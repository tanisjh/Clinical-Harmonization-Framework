"""Synthetic-only tests for source-independent declared-absence replay."""
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
from lib.replay import (
    ReplayRequest,
    recompute_declared_absence,
    replay_cell,
)
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
SCHEMA = FIXTURE / "schema_stage7.yaml"
SPEC = FIXTURE / "spec_stage7.yaml"
RUN_ID = "synthetic-stage7-run-0001"
ARTIFACT_ID = "synthetic-stage7-delivered"


def _pfs_bin(time_value, status_value, *, landmark_days):
    if time_value is None or status_value is None:
        return None, 0.2, "missing declared inputs"
    value = int(float(time_value) >= float(landmark_days) or status_value == 0)
    return value, 0.9, "synthetic landmark"


@pytest.fixture(autouse=True)
def _rules(monkeypatch):
    monkeypatch.setitem(
        rule_registry.REGISTRY,
        ("synthetic.pfs_bin_120d.v1", "1"),
        RegisteredRule(
            "synthetic.pfs_bin_120d.v1", "1", rule_registry.KIND_FIELD,
            _pfs_bin, arity=2, params=("landmark_days",)))


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


def _bundle(tmp_path: Path, *, target="absence_not_applicable") -> Case:
    case_dir = tmp_path / target
    case_dir.mkdir(parents=True)
    spec_path = case_dir / "spec.yaml"
    spec_path.write_bytes(SPEC.read_bytes())
    schema = load_schema(SCHEMA)
    spec = load_spec(spec_path, target_schema=schema, require_executable=True)
    context = ReplayRunContext.create(
        ROOT, schema, (spec,), run_id=RUN_ID,
        application="synthetic-stage7-test")
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


def _mapping(case: Case, target=None):
    wanted = target or case.request.target_field
    return next(mapping for mapping in case.spec.mappings
                if mapping.target == wanted)


def _cell(case: Case, target=None):
    mapping = _mapping(case, target)
    return next(cell for cell in case.cells
                if cell.mapping_id == mapping.mapping_id)


def _mutate_provenance(
        case: Case, selected_mapping_id=None, **updates) -> None:
    fields, rows = _read_rows(case.provenance_path)
    _event(
        rows, selected_mapping_id or _mapping(case).mapping_id).update(updates)
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


def _scope_stop(case: Case, mapping, reason: str) -> None:
    with pytest.raises(replay_module._Stop) as caught:
        replay_module._check_declared_absence_scope(
            mapping, case.spec, {}, case.manifest.delivered_artifacts[0])
    assert (caught.value.status, caught.value.reason_code) == (
        ReplayStatus.NOT_REPLAYABLE, reason)


@pytest.mark.parametrize("target,declared_state,serialized_state", [
    ("absence_not_applicable", "not_applicable", "not_applicable"),
    ("absence_source_missing", "source_missing", ""),
    ("absence_state_none", "not_applicable", ""),
])
def test_both_exact_semantic_states_and_serialization_policies_pass(
        tmp_path, target, declared_state, serialized_state):
    case = _bundle(tmp_path, target=target)
    cell = _cell(case)
    assert (_mapping(case).value_state, cell.value, cell.value_state) == (
        declared_state, None, serialized_state)
    result = replay_cell(ROOT, case.manifest, case.request)
    assert (result.status, result.reason_code, result.recomputed_value) == (
        ReplayStatus.PASS, "REPLAY_MATCH", None)


def test_pure_helper_preserves_state_and_has_closed_signature():
    signature = inspect.signature(recompute_declared_absence)
    assert tuple(signature.parameters) == ("declared_state",)
    assert signature.parameters["declared_state"].kind is \
        inspect.Parameter.KEYWORD_ONLY
    outcomes = [
        recompute_declared_absence(declared_state=state)
        for state in ("not_applicable", "source_missing")
    ]
    assert [(item.value, item.branch, item.semantic_state)
            for item in outcomes] == [
        (None, "declared_absence", "not_applicable"),
        (None, "declared_absence", "source_missing"),
    ]
    assert not set(signature.parameters) & {
        "delivered_value", "provenance_value", "value_state", "raw_value",
        "source", "source_ref",
    }


@pytest.mark.parametrize("state", ["", "delivered", "transformed",
                                   "no_mapping_configured", None])
def test_pure_helper_rejects_unsupported_semantic_state(state):
    with pytest.raises(ValueError, match="declared absence state"):
        recompute_declared_absence(declared_state=state)


@pytest.mark.parametrize("target,serialized_state", [
    ("absence_not_applicable", ""),
    ("absence_not_applicable", "source_missing"),
    ("absence_not_applicable", "suppressed"),
    ("absence_not_applicable", "row_excluded"),
    ("absence_source_missing", "source_missing"),
    ("absence_state_none", "not_applicable"),
])
def test_cell_state_contradictions_fail(
        tmp_path, target, serialized_state):
    case = _bundle(tmp_path, target=target)
    _mutate_provenance(case, value_state=serialized_state)
    result = replay_cell(ROOT, case.manifest, case.request)
    assert (result.status, result.reason_code) == (
        ReplayStatus.FAIL, "VALUE_STATE_CONTEXT_MISMATCH")


def test_scope_refuses_unsupported_declared_semantic_state(tmp_path):
    case = _bundle(tmp_path)
    _scope_stop(
        case, replace(_mapping(case), value_state="future_absence"),
        "VALUE_STATE_UNSUPPORTED")


def test_ordinary_confidence_and_optional_no_value_rule_match_executor(tmp_path):
    case = _bundle(tmp_path, target="absence_no_value_rule")
    cell = _cell(case)
    assert (cell.value, cell.rule_id, cell.confidence) == (
        None, "absence.asserted_null.v1", 0.73)
    assert cell.confidence != 0.11
    assert replay_cell(ROOT, case.manifest, case.request).status is \
        ReplayStatus.PASS


@pytest.mark.parametrize("field,value,reason", [
    ("rule_id", "wrong.rule", "RULE_IDENTITY_MISMATCH"),
    ("confidence", "0.11", "CONFIDENCE_CONTEXT_MISMATCH"),
])
def test_rule_and_confidence_contradictions_fail(
        tmp_path, field, value, reason):
    case = _bundle(tmp_path, target="absence_no_value_rule")
    _mutate_provenance(case, **{field: value})
    result = replay_cell(ROOT, case.manifest, case.request)
    assert (result.status, result.reason_code) == (ReplayStatus.FAIL, reason)


@pytest.mark.parametrize("target,expects_snapshot", [
    ("absence_not_applicable", True),
    ("absence_raw_value", True),
    ("absence_raw_coerced", True),
    ("absence_raw_none", False),
])
def test_exact_raw_policy_passes(tmp_path, target, expects_snapshot):
    case = _bundle(tmp_path, target=target)
    assert bool(_cell(case).raw_value_json) is expects_snapshot
    if expects_snapshot:
        assert _cell(case).raw_value_json == encode_typed_raw(None).to_json()
    assert replay_cell(ROOT, case.manifest, case.request).status is \
        ReplayStatus.PASS


def test_wrong_typed_raw_snapshot_fails_without_becoming_input(tmp_path):
    case = _bundle(tmp_path)
    _mutate_provenance(
        case, raw_value_json=encode_typed_raw("tampered").to_json())
    result = replay_cell(ROOT, case.manifest, case.request)
    assert (result.status, result.reason_code, result.recomputed_value) == (
        ReplayStatus.FAIL, "RAW_VALUE_MISMATCH", None)


def test_missing_required_raw_snapshot_is_historical_incompleteness(tmp_path):
    case = _bundle(tmp_path)
    _mutate_provenance(case, raw_value_json="")
    result = replay_cell(ROOT, case.manifest, case.request)
    assert (result.status, result.reason_code) == (
        ReplayStatus.NOT_REPLAYABLE, "RAW_VALUE_ENCODING_MISSING")


def test_raw_omission_policy_rejects_a_snapshot(tmp_path):
    case = _bundle(tmp_path, target="absence_raw_none")
    _mutate_provenance(
        case, raw_value_json=encode_typed_raw(None).to_json())
    result = replay_cell(ROOT, case.manifest, case.request)
    assert (result.status, result.reason_code) == (
        ReplayStatus.FAIL, "RAW_VALUE_MISMATCH")


def test_source_free_origin_column_key_and_locator_policies(tmp_path):
    explicit = _bundle(tmp_path / "explicit")
    cell = _cell(explicit)
    assert (cell.source_file, cell.source_logical_label, cell.source_column,
            cell.source_record_key) == (
        "CONFIG:declared_absence", "CONFIG:declared_absence",
        "declared_absence", "device_id=DEVICE-A")

    default = _bundle(tmp_path / "default", target="absence_default_origin")
    cell = _cell(default)
    assert (cell.source_file, cell.source_logical_label, cell.source_column,
            cell.source_record_key) == (
        "SPEC:not_applicable", "SPEC:not_applicable",
        "absence_default_origin", "device_id=DEVICE-A")

    literal = _bundle(tmp_path / "literal", target="absence_literal_key")
    assert _cell(literal).source_record_key == \
        "config:synthetic.absence_literal_key"

    for case in (explicit, default, literal):
        cell = _cell(case)
        assert not any((cell.source_resolved_path, cell.source_sha256,
                        cell.source_sheet, cell.source_row_number,
                        cell.source_ref_id, cell.evidence_refs,
                        cell.contributions, cell.derivation))
        usage = next(item for item in case.manifest.mapping_usages
                     if item.mapping_id == _mapping(case).mapping_id)
        assert usage.source_ref_ids == ()
        assert usage.evidence_ref_ids == ()
        assert replay_cell(ROOT, case.manifest, case.request).status is \
            ReplayStatus.PASS


@pytest.mark.parametrize("field,value", [
    ("source_file", "WRONG:origin"),
    ("source_logical_label", "WRONG:label"),
    ("source_column", "wrong_column"),
    ("source_record_key", "wrong=key"),
    ("source_resolved_path", "some/file.csv"),
    ("source_sha256", "0" * 64),
    ("source_sheet", "Sheet1"),
    ("source_row_number", "0"),
    ("source_ref_id", "src-claimed"),
    ("evidence_refs", "config-claimed"),
    ("contributions", '[{"status":"input"}]'),
    ("derivation", '{"kind":"claimed"}'),
])
def test_provenance_context_contradictions_fail(tmp_path, field, value):
    case = _bundle(tmp_path)
    _mutate_provenance(case, **{field: value})
    result = replay_cell(ROOT, case.manifest, case.request)
    assert (result.status, result.reason_code) == (
        ReplayStatus.FAIL, "SOURCE_DECLARATION_MISMATCH")


def test_wrong_artifact_authority_is_not_replayable(tmp_path):
    case = _bundle(tmp_path)
    _mutate_provenance(case, authoritative_for="other-artifact")
    result = replay_cell(ROOT, case.manifest, case.request)
    assert (result.status, result.reason_code) == (
        ReplayStatus.NOT_REPLAYABLE, "PROVENANCE_NOT_AUTHORITATIVE")


@pytest.mark.parametrize("identity", ["python", "yaml"])
def test_environment_pins_only_python_and_yaml(tmp_path, identity):
    case = _bundle(tmp_path)
    producer = case.manifest.producer
    if identity == "python":
        producer = replace(producer, python_version="0.0.0")
    else:
        dependencies = dict(producer.dependencies)
        dependencies["yaml"] = "0.0.0"
        producer = replace(
            producer, dependencies=tuple(sorted(dependencies.items())))
    case.manifest = replace(case.manifest, producer=producer)
    result = replay_cell(ROOT, case.manifest, case.request)
    assert (result.status, result.reason_code) == (
        ReplayStatus.NOT_REPLAYABLE, "PRODUCER_ENVIRONMENT_MISMATCH")


def test_pandas_mismatch_does_not_block_declared_absence(tmp_path):
    case = _bundle(tmp_path)
    dependencies = dict(case.manifest.producer.dependencies)
    dependencies["pandas"] = "0.0.0"
    case.manifest = replace(
        case.manifest,
        producer=replace(
            case.manifest.producer,
            dependencies=tuple(sorted(dependencies.items()))))
    assert replay_cell(ROOT, case.manifest, case.request).status is \
        ReplayStatus.PASS


def test_delivery_and_provenance_cannot_change_recomputation(
        tmp_path, monkeypatch):
    calls = []
    original = replay_module.recompute_declared_absence

    def guarded(**kwargs):
        calls.append(kwargs)
        return original(**kwargs)

    monkeypatch.setattr(replay_module, "recompute_declared_absence", guarded)
    case = _bundle(tmp_path / "both")
    _mutate_delivery(case, "absence_not_applicable", "tampered")
    _mutate_provenance(case, value="tampered")
    result = replay_cell(ROOT, case.manifest, case.request)
    assert (result.status, result.reason_code, result.recomputed_value) == (
        ReplayStatus.FAIL, "RECOMPUTED_VALUE_MISMATCH", None)
    assert calls == [{"declared_state": "not_applicable"}]

    calls.clear()
    other = _bundle(tmp_path / "provenance")
    _mutate_provenance(other, value="tampered")
    result = replay_cell(ROOT, other.manifest, other.request)
    assert (result.status, result.reason_code, result.recomputed_value) == (
        ReplayStatus.FAIL, "PROVENANCE_VALUE_MISMATCH", None)
    assert calls == [{"declared_state": "not_applicable"}]


def test_complete_v2_event_and_intentional_state_omission_can_pass(tmp_path):
    case = _bundle(tmp_path, target="absence_source_missing")
    assert case.manifest.manifest_version == 2
    assert _cell(case).value_state == ""
    assert replay_cell(ROOT, case.manifest, case.request).status is \
        ReplayStatus.PASS


@pytest.mark.parametrize("missing", ["usage", "mapping_id", "spec_ref"])
def test_incomplete_historical_identity_is_not_replayable(tmp_path, missing):
    case = _bundle(tmp_path)
    if missing == "usage":
        case.manifest = replace(
            case.manifest,
            mapping_usages=tuple(
                usage for usage in case.manifest.mapping_usages
                if usage.mapping_id != "absence.not_applicable"))
    else:
        _mutate_provenance(case, **{missing: ""})
    result = replay_cell(ROOT, case.manifest, case.request)
    assert result.status is ReplayStatus.NOT_REPLAYABLE
    assert result.reason_code in {
        "MAPPING_NOT_IDENTIFIED", "SPEC_LINK_MISSING"}


def test_exact_spec_hash_remains_required(tmp_path):
    case = _bundle(tmp_path)
    case.spec_path.write_bytes(case.spec_path.read_bytes() + b"\n")
    result = replay_cell(ROOT, case.manifest, case.request)
    assert (result.status, result.reason_code) == (
        ReplayStatus.FAIL, "SPEC_HASH_MISMATCH")


@pytest.mark.parametrize("target,reason", [
    ("anchor_absence", "ANCHOR_SOURCE_UNSUPPORTED_STAGE1"),
    ("derived_absence", "DERIVATION_UNSUPPORTED_STAGE2"),
    ("unsupported_no_mapping", "OPERATION_UNSUPPORTED"),
])
def test_explicitly_excluded_operations_and_contexts_remain_unsupported(
        tmp_path, target, reason):
    case = _bundle(tmp_path, target=target)
    result = replay_cell(ROOT, case.manifest, case.request)
    assert (result.status, result.reason_code) == (
        ReplayStatus.NOT_REPLAYABLE, reason)


def test_group_key_attribution_requires_a_grouping_anchor(tmp_path):
    case = _bundle(tmp_path, target="source_attributed_absence")
    mapping = _mapping(case)
    provenance = replace(mapping.provenance, record_key_from="group")
    _scope_stop(
        case, replace(mapping, provenance=provenance),
        "SOURCE_NOT_SINGLE_FILE")


@pytest.mark.parametrize("change,reason", [
    ({"when": "synthetic_condition"}, "MAPPING_CONDITION_UNSUPPORTED"),
    ({"group": "synthetic_group"}, "SOURCE_NOT_SINGLE_FILE"),
    ({"column_by_member": {"member": "column"}}, "SOURCE_NOT_SINGLE_FILE"),
])
def test_declaration_scope_is_closed(tmp_path, change, reason):
    case = _bundle(tmp_path)
    _scope_stop(case, replace(_mapping(case), **change), reason)


@pytest.mark.parametrize("provenance_change,reason", [
    ({"row_from": "readings"}, "SOURCE_NOT_SINGLE_FILE"),
    ({"raw_from": "synthetic_read"}, "DERIVATION_UNSUPPORTED_STAGE2"),
    ({"contributions_from": "synthetic_group"}, "SOURCE_NOT_SINGLE_FILE"),
    ({"row_number": "source"}, "SOURCE_NOT_SINGLE_FILE"),
    ({"record_key_from": "group"}, "SOURCE_NOT_SINGLE_FILE"),
])
def test_source_row_raw_and_group_indirection_remain_unsupported(
        tmp_path, provenance_change, reason):
    case = _bundle(tmp_path)
    mapping = _mapping(case)
    provenance = replace(mapping.provenance, **provenance_change)
    _scope_stop(case, replace(mapping, provenance=provenance), reason)


def test_condition_contributions_remain_unsupported(tmp_path):
    case = _bundle(tmp_path)
    mapping = _mapping(case)
    contributions = replace(
        mapping.contributions, conditions=("synthetic_condition",))
    _scope_stop(
        case, replace(mapping, contributions=contributions),
        "CONTRIBUTIONS_UNSUPPORTED_STAGE2")


@pytest.mark.parametrize("method,state,reason", [
    ("value_suppressed", "suppressed", "SUPPRESSION_UNSUPPORTED_STAGE1"),
    ("row_excluded", "row_excluded", "EXCLUSION_REQUIRES_ROW_REPLAY"),
    ("post:synthetic_override", "not_applicable",
     "POST_RULE_UNSUPPORTED_STAGE1"),
])
def test_artifact_stage_authority_remains_unsupported(
        tmp_path, method, state, reason):
    case = _bundle(tmp_path)
    _mutate_provenance(
        case, extraction_method=method, value_state=state)
    result = replay_cell(ROOT, case.manifest, case.request)
    assert (result.status, result.reason_code) == (
        ReplayStatus.NOT_REPLAYABLE, reason)


@pytest.mark.parametrize("target", [
    "pfs_source_missing", "pfs_not_applicable",
])
def test_declared_absence_parents_replay_before_registered_pfs_rule(
        tmp_path, monkeypatch, target):
    calls = []

    def spy(time_value, status_value, *, landmark_days):
        calls.append((time_value, status_value, landmark_days))
        return _pfs_bin(time_value, status_value, landmark_days=landmark_days)

    monkeypatch.setitem(
        rule_registry.REGISTRY,
        ("synthetic.pfs_bin_120d.v1", "1"),
        RegisteredRule(
            "synthetic.pfs_bin_120d.v1", "1", rule_registry.KIND_FIELD,
            spy, arity=2, params=("landmark_days",)))
    case = _bundle(tmp_path, target=target)
    calls.clear()
    result = replay_cell(ROOT, case.manifest, case.request)
    assert (result.status, result.reason_code, result.recomputed_value) == (
        ReplayStatus.PASS, "REPLAY_MATCH", None)
    assert calls == [(None, None, 120)]


def test_parent_state_contradiction_blocks_recursive_rule(tmp_path):
    case = _bundle(tmp_path, target="pfs_source_missing")
    _mutate_provenance(
        case, selected_mapping_id="absence.source_missing_time",
        value_state="source_missing")
    result = replay_cell(ROOT, case.manifest, case.request)
    assert (result.status, result.reason_code) == (
        ReplayStatus.FAIL, "DEPENDENCY_REPLAY_FAILED")
    assert "VALUE_STATE_CONTEXT_MISMATCH" in result.message


def test_unsupported_sibling_dependency_still_blocks_downstream_rule(tmp_path):
    case = _bundle(tmp_path, target="downstream_blocked")
    result = replay_cell(ROOT, case.manifest, case.request)
    assert (result.status, result.reason_code) == (
        ReplayStatus.NOT_REPLAYABLE, "DEPENDENCY_NOT_REPLAYABLE")
    assert "OPERATION_UNSUPPORTED" in result.message


def test_stage7_changes_no_contract_version(tmp_path):
    case = _bundle(tmp_path)
    result = replay_cell(ROOT, case.manifest, case.request)
    assert (result.api_version, result.status_version,
            result.reason_codes_version,
            case.manifest.manifest_version) == (1, 1, 10, 2)

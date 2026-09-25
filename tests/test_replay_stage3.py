"""Synthetic-only tests for recursive registered field-rule replay."""
from __future__ import annotations

import csv
import inspect
import json
from dataclasses import dataclass, replace
from pathlib import Path

import pandas as pd
import pytest
import yaml

import harmonize_spec
import lib.replay as replay_module
from lib import rule_registry
from lib.contributions import STATUS_INPUT, from_json as contributions_from_json
from lib.provenance import (
    canonical_row_identity,
    is_provenance_event_ref,
    provenance_event_ref,
)
from lib.replay import ReplayRequest, recompute_registered_rule, replay_cell
from lib.replay_contract import ReplayStatus, encode_typed_raw
from lib.replay_orchestration import (
    ReplayEvidenceCollector,
    ReplayOrchestrationError,
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
SCHEMA_FIXTURE = FIXTURE / "schema_stage3.yaml"
SPEC_FIXTURE = FIXTURE / "spec_stage3.yaml"
SOURCE_FIXTURE = FIXTURE / "sources/stage3_readings.csv"
RUN_ID = "synthetic-stage3-run-0001"
ARTIFACT_ID = "synthetic-stage3-delivered"


def _adjusted(left, right, *, scale):
    if left is None or right is None:
        return None, 0.2, "missing input"
    return float(left) * float(scale) + float(right), 0.9, "synthetic rule"


def _identity(value):
    return value, 0.8, "synthetic identity"


@pytest.fixture(autouse=True)
def _test_rules(monkeypatch):
    monkeypatch.setitem(rule_registry.REGISTRY,
        ("synthetic.adjusted.v1", "1"), RegisteredRule(
            "synthetic.adjusted.v1", "1", rule_registry.KIND_FIELD,
            _adjusted, arity=2, params=("scale",)))
    monkeypatch.setitem(rule_registry.REGISTRY,
        ("synthetic.identity.v1", "1"), RegisteredRule(
            "synthetic.identity.v1", "1", rule_registry.KIND_FIELD,
            _identity, arity=1))


@dataclass
class Case:
    manifest: object
    request: ReplayRequest
    source_path: Path
    spec_path: Path
    delivered_path: Path
    provenance_path: Path
    cells: list


def _relative(path: Path) -> str:
    return str(path.resolve().relative_to(ROOT.resolve()))


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


def _bundle(tmp_path: Path, *, target="adjusted_measurement",
            source_updates=None, spec_mutator=None) -> Case:
    case_dir = tmp_path / target
    case_dir.mkdir(parents=True)
    source_path = case_dir / "stage3_readings.csv"
    fields, rows = _read_rows(SOURCE_FIXTURE)
    rows[0].update(source_updates or {})
    _write_rows(source_path, fields, rows)

    spec_doc = yaml.safe_load(SPEC_FIXTURE.read_text())
    spec_doc["sources"][0]["path"] = _relative(source_path)
    if spec_mutator:
        spec_mutator(spec_doc)
    spec_path = case_dir / "spec.yaml"
    spec_path.write_text(yaml.safe_dump(spec_doc, sort_keys=False))
    schema = load_schema(SCHEMA_FIXTURE)
    spec = load_spec(spec_path, target_schema=schema, require_executable=True)
    context = ReplayRunContext.create(
        ROOT, schema, (spec,), run_id=RUN_ID,
        application="synthetic-stage3-test")
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
    return Case(published.manifest, request, source_path, spec_path,
                published.delivered_path, published.provenance_path, cells)


def _mutate_provenance(case: Case, mutate) -> None:
    fields, rows = _read_rows(case.provenance_path)
    mutate(rows)
    _write_rows(case.provenance_path, fields, rows)
    binding = replace(
        case.manifest.provenance_artifact,
        sha256=sha256_file(case.provenance_path), n_rows=len(rows))
    case.manifest = replace(case.manifest, provenance_artifact=binding)


def _mutate_delivered(case: Case, mutate) -> None:
    fields, rows = _read_rows(case.delivered_path)
    mutate(rows)
    _write_rows(case.delivered_path, fields, rows)
    old = case.manifest.delivered_artifacts[0]
    case.manifest = replace(case.manifest, delivered_artifacts=(replace(
        old, sha256=sha256_file(case.delivered_path), n_rows=len(rows)),))


def _event(rows, mapping_id):
    return next(row for row in rows if row["mapping_id"] == mapping_id)


def _set_contribution(event, position, **updates):
    items = json.loads(event["contributions"])
    items[position].update(updates)
    event["contributions"] = json.dumps(
        items, separators=(",", ":"), sort_keys=True)


def test_two_input_recursive_pass_through_production_orchestration(tmp_path):
    case = _bundle(tmp_path)
    result = replay_cell(ROOT, case.manifest, case.request)
    assert (result.status, result.reason_code, result.recomputed_value) == (
        ReplayStatus.PASS, "REPLAY_MATCH", 23.0)
    parent = next(c for c in case.cells if c.mapping_id == "custom.adjusted")
    contributions = contributions_from_json(parent.contributions)
    assert [item["status"] for item in contributions] == [STATUS_INPUT] * 2
    assert [item["rank"] for item in contributions] == [0, 1]
    assert [item["source_column"] for item in contributions] == [
        "base_measurement", "calibration_offset"]
    assert all(is_provenance_event_ref(item["provenance_ref"])
               for item in contributions)
    assert all(not item.get("source_ref_id") for item in contributions)
    rules = {rule.ref: rule.kind for rule in case.manifest.rules}
    assert rules["synthetic.adjusted.v1@1"] == "field"
    usages = {usage.mapping_id: usage
              for usage in case.manifest.mapping_usages}
    assert usages["custom.adjusted"].source_ref_ids == ()
    assert usages["custom.adjusted"].rule_refs == (
        "synthetic.adjusted.v1@1",)
    assert usages["direct.base"].source_ref_ids


def test_collector_rejects_dependency_snapshot_not_from_runtime_cell(tmp_path):
    case = _bundle(tmp_path)
    schema = load_schema(SCHEMA_FIXTURE)
    spec = load_spec(case.spec_path, target_schema=schema,
                     require_executable=True)
    context = ReplayRunContext.create(ROOT, schema, (spec,), run_id=RUN_ID)
    cells, diagnostics = execute(
        spec, ROOT, _anchor(), contract=contract_for(schema), run_id=RUN_ID)
    parent = next(cell for cell in cells
                  if cell.mapping_id == "custom.adjusted")
    payload = json.loads(parent.contributions)
    payload[0]["raw_value_json"] = encode_typed_raw(999).to_json()
    parent.contributions = json.dumps(
        payload, separators=(",", ":"), sort_keys=True)
    with pytest.raises(ReplayOrchestrationError, match="typed snapshot"):
        ReplayEvidenceCollector(context).collect(
            spec, diagnostics.sources, cells)


def test_pev1_is_generic_deterministic_and_value_independent():
    identity = canonical_row_identity(
        ("device_id", "reading_id"),
        {"device_id": "DEVICE-A", "reading_id": "READ-001"})
    kwargs = dict(run_id=RUN_ID, spec_ref="synthetic/spec@1",
                  mapping_id="direct.base", row_identity=identity,
                  harmonized_field="base_measurement")
    assert provenance_event_ref(**kwargs) == provenance_event_ref(**kwargs)
    assert set(identity) == {"device_id", "reading_id"}
    assert "cimac" not in provenance_event_ref(**kwargs).lower()


def test_exact_input_order_and_pinned_parameter_reach_callable(tmp_path, monkeypatch):
    seen = []
    def spy(left, right, *, scale):
        seen.append((left, right, scale))
        return _adjusted(left, right, scale=scale)
    monkeypatch.setitem(rule_registry.REGISTRY,
        ("synthetic.adjusted.v1", "1"), RegisteredRule(
            "synthetic.adjusted.v1", "1", rule_registry.KIND_FIELD,
            spy, arity=2, params=("scale",)))
    case = _bundle(tmp_path)
    seen.clear()  # exclude normal execution; inspect replay invocation only
    assert replay_cell(ROOT, case.manifest, case.request).status is ReplayStatus.PASS
    assert seen == [(10.0, 3.0, 2.0)]


@pytest.mark.parametrize("updates", [
    {"provenance_ref": ""},
    {"provenance_ref": "not-a-pev1"},
    {"provenance_ref": "pev1-" + "0" * 64},
])
def test_missing_malformed_and_unresolvable_dependency_refs(tmp_path, updates):
    case = _bundle(tmp_path)
    _mutate_provenance(case, lambda rows: _set_contribution(
        _event(rows, "custom.adjusted"), 0, **updates))
    result = replay_cell(ROOT, case.manifest, case.request)
    assert (result.status, result.reason_code) == (
        ReplayStatus.NOT_REPLAYABLE, "DEPENDENCY_REFERENCE_MISSING")


def test_duplicate_dependency_reference_is_ambiguous(tmp_path):
    case = _bundle(tmp_path)
    def duplicate(rows):
        rows.append(dict(_event(rows, "direct.base")))
    _mutate_provenance(case, duplicate)
    result = replay_cell(ROOT, case.manifest, case.request)
    assert (result.status, result.reason_code) == (
        ReplayStatus.NOT_REPLAYABLE, "DEPENDENCY_REFERENCE_AMBIGUOUS")


def test_wrong_existing_event_reference_is_context_failure(tmp_path):
    case = _bundle(tmp_path)
    def wrong(rows):
        parent = _event(rows, "custom.adjusted")
        offset = next(c for c in case.cells if c.mapping_id == "direct.offset")
        row_identity = {"device_id": "DEVICE-A", "reading_id": "READ-001"}
        ref = provenance_event_ref(
            run_id=RUN_ID, spec_ref=offset.spec_ref,
            mapping_id=offset.mapping_id, row_identity=row_identity,
            harmonized_field=offset.harmonized_field)
        _set_contribution(parent, 0, provenance_ref=ref)
    _mutate_provenance(case, wrong)
    result = replay_cell(ROOT, case.manifest, case.request)
    assert (result.status, result.reason_code) == (
        ReplayStatus.FAIL, "DEPENDENCY_CONTEXT_MISMATCH")


def test_dependency_non_key_anchor_drift_is_context_failure(tmp_path):
    case = _bundle(tmp_path)
    _mutate_provenance(case, lambda rows: _event(
        rows, "direct.base").update(Collection_Event="wrong-location"))
    result = replay_cell(ROOT, case.manifest, case.request)
    assert (result.status, result.reason_code) == (
        ReplayStatus.FAIL, "DEPENDENCY_CONTEXT_MISMATCH")


def test_dependency_source_hash_drift_fails_parent(tmp_path):
    case = _bundle(tmp_path)
    case.source_path.write_text(case.source_path.read_text().replace(",10,3", ",11,3"))
    result = replay_cell(ROOT, case.manifest, case.request)
    assert (result.status, result.reason_code) == (
        ReplayStatus.FAIL, "DEPENDENCY_REPLAY_FAILED")
    assert "SOURCE_HASH_MISMATCH" in result.message


def test_dependency_typed_raw_drift_fails_parent(tmp_path):
    case = _bundle(tmp_path)
    def drift(rows):
        _event(rows, "direct.base")["raw_value_json"] = \
            encode_typed_raw(99).to_json()
    _mutate_provenance(case, drift)
    result = replay_cell(ROOT, case.manifest, case.request)
    assert (result.status, result.reason_code) == (
        ReplayStatus.FAIL, "DEPENDENCY_REPLAY_FAILED")
    assert "RAW_VALUE_MISMATCH" in result.message


def test_dependency_record_key_drift_fails_parent(tmp_path):
    case = _bundle(tmp_path)
    _mutate_provenance(case, lambda rows: _event(
        rows, "direct.base").update(source_record_key="device_id=OTHER"))
    result = replay_cell(ROOT, case.manifest, case.request)
    assert (result.status, result.reason_code) == (
        ReplayStatus.FAIL, "DEPENDENCY_REPLAY_FAILED")
    assert "SOURCE_RECORD_KEY_MISMATCH" in result.message


def test_upstream_delivered_mismatch_fails_parent(tmp_path):
    case = _bundle(tmp_path)
    _mutate_delivered(case, lambda rows: rows[0].update(base_measurement="99"))
    result = replay_cell(ROOT, case.manifest, case.request)
    assert (result.status, result.reason_code) == (
        ReplayStatus.FAIL, "DEPENDENCY_REPLAY_FAILED")
    assert "PROVENANCE_VALUE_MISMATCH" in result.message


def test_unsupported_dependency_makes_parent_not_replayable(tmp_path):
    case = _bundle(tmp_path, target="unsupported_result")
    result = replay_cell(ROOT, case.manifest, case.request)
    assert (result.status, result.reason_code) == (
        ReplayStatus.NOT_REPLAYABLE, "DEPENDENCY_NOT_REPLAYABLE")
    assert "OPERATION_UNSUPPORTED" in result.message


def test_direct_self_cycle_is_not_replayable(tmp_path):
    case = _bundle(tmp_path)
    def cycle(rows):
        parent = _event(rows, "custom.adjusted")
        ref = provenance_event_ref(
            run_id=RUN_ID, spec_ref=parent["spec_ref"],
            mapping_id=parent["mapping_id"],
            row_identity={"device_id": "DEVICE-A", "reading_id": "READ-001"},
            harmonized_field=parent["harmonized_field"])
        _set_contribution(parent, 0, provenance_ref=ref,
                          source_column="base_measurement")
    _mutate_provenance(case, cycle)
    result = replay_cell(ROOT, case.manifest, case.request)
    assert (result.status, result.reason_code) == (
        ReplayStatus.NOT_REPLAYABLE, "REPLAY_CYCLE_DETECTED")


def test_multi_node_cycle_is_not_replayable(tmp_path):
    case = _bundle(tmp_path, target="final_measurement")
    def cycle(rows):
        adjusted = _event(rows, "custom.adjusted")
        final = _event(rows, "custom.final")
        ref = provenance_event_ref(
            run_id=RUN_ID, spec_ref=final["spec_ref"],
            mapping_id=final["mapping_id"],
            row_identity={"device_id": "DEVICE-A", "reading_id": "READ-001"},
            harmonized_field=final["harmonized_field"])
        _set_contribution(adjusted, 0, provenance_ref=ref,
                          source_column="base_measurement")
    _mutate_provenance(case, cycle)
    result = replay_cell(ROOT, case.manifest, case.request)
    assert (result.status, result.reason_code) == (
        ReplayStatus.NOT_REPLAYABLE, "REPLAY_CYCLE_DETECTED")


def test_wrong_rule_confidence_and_contract_failures(tmp_path, monkeypatch):
    case = _bundle(tmp_path)
    _mutate_provenance(case, lambda rows: _event(
        rows, "custom.adjusted").update(confidence="0.1"))
    result = replay_cell(ROOT, case.manifest, case.request)
    assert (result.status, result.reason_code) == (
        ReplayStatus.FAIL, "CONFIDENCE_CONTEXT_MISMATCH")

    case = _bundle(tmp_path / "rule", target="adjusted_measurement")
    _mutate_provenance(case, lambda rows: _event(
        rows, "custom.adjusted").update(rule_version="999"))
    result = replay_cell(ROOT, case.manifest, case.request)
    assert (result.status, result.reason_code) == (
        ReplayStatus.FAIL, "RULE_IDENTITY_MISMATCH")

    case = _bundle(tmp_path / "arity", target="adjusted_measurement")
    monkeypatch.setitem(rule_registry.REGISTRY,
        ("synthetic.adjusted.v1", "1"), RegisteredRule(
            "synthetic.adjusted.v1", "1", rule_registry.KIND_FIELD,
            _adjusted, arity=1, params=("scale",)))
    result = replay_cell(ROOT, case.manifest, case.request)
    assert (result.status, result.reason_code) == (
        ReplayStatus.NOT_REPLAYABLE, "REGISTERED_RULE_CONTRACT_MISMATCH")


def test_wrong_registry_parameter_contract_is_not_replayable(tmp_path, monkeypatch):
    case = _bundle(tmp_path)
    monkeypatch.setitem(rule_registry.REGISTRY,
        ("synthetic.adjusted.v1", "1"), RegisteredRule(
            "synthetic.adjusted.v1", "1", rule_registry.KIND_FIELD,
            _adjusted, arity=2, params=("different_parameter",)))
    result = replay_cell(ROOT, case.manifest, case.request)
    assert (result.status, result.reason_code) == (
        ReplayStatus.NOT_REPLAYABLE, "REGISTERED_RULE_CONTRACT_MISMATCH")


def test_unavailable_rule_and_wrong_registry_kind(tmp_path, monkeypatch):
    case = _bundle(tmp_path)
    monkeypatch.delitem(rule_registry.REGISTRY,
                        ("synthetic.adjusted.v1", "1"))
    result = replay_cell(ROOT, case.manifest, case.request)
    assert (result.status, result.reason_code) == (
        ReplayStatus.NOT_REPLAYABLE, "REGISTERED_RULE_UNAVAILABLE")

    monkeypatch.setitem(rule_registry.REGISTRY,
        ("synthetic.adjusted.v1", "1"), RegisteredRule(
            "synthetic.adjusted.v1", "1", rule_registry.KIND_POST,
            _adjusted, arity=2, params=("scale",)))
    result = replay_cell(ROOT, case.manifest, case.request)
    assert (result.status, result.reason_code) == (
        ReplayStatus.FAIL, "RULE_IDENTITY_MISMATCH")


def test_registered_rule_execution_failure_is_explicit(tmp_path, monkeypatch):
    case = _bundle(tmp_path)
    def broken(_left, _right, *, scale):
        raise RuntimeError(f"synthetic failure at scale {scale}")
    monkeypatch.setitem(rule_registry.REGISTRY,
        ("synthetic.adjusted.v1", "1"), RegisteredRule(
            "synthetic.adjusted.v1", "1", rule_registry.KIND_FIELD,
            broken, arity=2, params=("scale",)))
    result = replay_cell(ROOT, case.manifest, case.request)
    assert (result.status, result.reason_code) == (
        ReplayStatus.FAIL, "REGISTERED_RULE_EXECUTION_FAILED")


def test_wrong_manifest_rule_kind_fails(tmp_path):
    case = _bundle(tmp_path)
    rules = tuple(replace(rule, kind="mapping")
                  if rule.ref == "synthetic.adjusted.v1@1" else rule
                  for rule in case.manifest.rules)
    case.manifest = replace(case.manifest, rules=rules)
    result = replay_cell(ROOT, case.manifest, case.request)
    assert (result.status, result.reason_code) == (
        ReplayStatus.FAIL, "RULE_IDENTITY_MISMATCH")


@pytest.mark.parametrize("updates", [
    {"rank": 1}, {"status": "selected"},
    {"source_column": "calibration_offset"},
])
def test_wrong_contribution_rank_status_or_order_fails(tmp_path, updates):
    case = _bundle(tmp_path)
    _mutate_provenance(case, lambda rows: _set_contribution(
        _event(rows, "custom.adjusted"), 0, **updates))
    result = replay_cell(ROOT, case.manifest, case.request)
    assert (result.status, result.reason_code) == (
        ReplayStatus.FAIL, "DEPENDENCY_CONTEXT_MISMATCH")


def test_wrong_contribution_arity_and_snapshot(tmp_path):
    case = _bundle(tmp_path)
    def missing(rows):
        parent = _event(rows, "custom.adjusted")
        parent["contributions"] = json.dumps(
            json.loads(parent["contributions"])[:1],
            separators=(",", ":"), sort_keys=True)
    _mutate_provenance(case, missing)
    result = replay_cell(ROOT, case.manifest, case.request)
    assert (result.status, result.reason_code) == (
        ReplayStatus.NOT_REPLAYABLE, "DEPENDENCY_REFERENCE_MISSING")

    case = _bundle(tmp_path / "snapshot")
    _mutate_provenance(case, lambda rows: _set_contribution(
        _event(rows, "custom.adjusted"), 0,
        raw_value_json=encode_typed_raw(999).to_json()))
    result = replay_cell(ROOT, case.manifest, case.request)
    assert (result.status, result.reason_code) == (
        ReplayStatus.FAIL, "DEPENDENCY_CONTEXT_MISMATCH")


def test_supported_derived_null_passes(tmp_path):
    case = _bundle(tmp_path, source_updates={"base_measurement": ""})
    result = replay_cell(ROOT, case.manifest, case.request)
    assert (result.status, result.recomputed_value) == (ReplayStatus.PASS, None)


def test_historical_vocabulary_parameter_without_evidence_is_not_replayable(
        tmp_path):
    case = _bundle(tmp_path)
    doc = yaml.safe_load(case.spec_path.read_text())
    mapping = next(m for m in doc["mappings"]
                   if m["mapping_id"] == "custom.adjusted")
    mapping["params"]["scale"] = {"$vocabulary": "synthetic/scale@1"}
    case.spec_path.write_text(yaml.safe_dump(doc, sort_keys=False))
    spec_binding = replace(case.manifest.specifications[0],
                           sha256=sha256_file(case.spec_path),
                           vocabulary_refs=("synthetic/scale@1",))
    case.manifest = replace(case.manifest, specifications=(spec_binding,))
    result = replay_cell(ROOT, case.manifest, case.request)
    assert (result.status, result.reason_code) == (
        ReplayStatus.NOT_REPLAYABLE, "VOCABULARY_EVIDENCE_MISSING")


def test_changed_pinned_parameter_recomputes_independently_and_fails(tmp_path):
    case = _bundle(tmp_path)
    doc = yaml.safe_load(case.spec_path.read_text())
    mapping = next(m for m in doc["mappings"]
                   if m["mapping_id"] == "custom.adjusted")
    mapping["params"]["scale"] = 3.0
    case.spec_path.write_text(yaml.safe_dump(doc, sort_keys=False))
    spec_binding = replace(case.manifest.specifications[0],
                           sha256=sha256_file(case.spec_path))
    case.manifest = replace(case.manifest, specifications=(spec_binding,))
    result = replay_cell(ROOT, case.manifest, case.request)
    assert (result.status, result.reason_code, result.recomputed_value) == (
        ReplayStatus.FAIL, "RECOMPUTED_VALUE_MISMATCH", 33.0)


def test_stage3_requires_exact_producer_environment(tmp_path):
    case = _bundle(tmp_path)
    case.manifest = replace(
        case.manifest,
        producer=replace(case.manifest.producer, python_version="0.0.0"))
    result = replay_cell(ROOT, case.manifest, case.request)
    assert (result.status, result.reason_code) == (
        ReplayStatus.NOT_REPLAYABLE, "PRODUCER_ENVIRONMENT_MISMATCH")


def test_changing_derived_delivery_does_not_change_recomputation(tmp_path):
    case = _bundle(tmp_path)
    _mutate_delivered(case, lambda rows: rows[0].update(
        adjusted_measurement="999"))
    result = replay_cell(ROOT, case.manifest, case.request)
    assert result.status is ReplayStatus.FAIL
    assert result.reason_code == "PROVENANCE_VALUE_MISMATCH"
    assert result.recomputed_value == 23.0


def test_recomputation_signature_has_no_output_values():
    names = set(inspect.signature(recompute_registered_rule).parameters)
    assert names == {"rule", "replayed_inputs", "params"}
    assert not names & {"delivered_value", "provenance_value", "output_value"}


def test_shared_dependency_is_memoized(tmp_path, monkeypatch):
    case = _bundle(tmp_path, target="final_measurement")
    calls = []
    original = replay_module._ReplaySession._replay_scalar
    def spy(self, mapping, event, usage):
        calls.append(mapping.mapping_id)
        return original(self, mapping, event, usage)
    monkeypatch.setattr(replay_module._ReplaySession, "_replay_scalar", spy)
    assert replay_cell(ROOT, case.manifest, case.request).status is ReplayStatus.PASS
    assert calls.count("direct.base") == 1


def test_historical_custom_row_without_links_is_not_replayable(tmp_path):
    case = _bundle(tmp_path)
    _mutate_provenance(case, lambda rows: _event(
        rows, "custom.adjusted").update(contributions=""))
    result = replay_cell(ROOT, case.manifest, case.request)
    assert (result.status, result.reason_code) == (
        ReplayStatus.NOT_REPLAYABLE, "DEPENDENCY_REFERENCE_MISSING")


def test_depth_guard_over_32(tmp_path):
    fields = ["base"] + [f"node_{i}" for i in range(33)]
    schema_doc = {
        "schema_id": "synthetic/deep", "schema_version": "1.0.0",
        "row_key": ["device_id"],
        "column_order": ["device_id", *fields], "null_encoding": "",
        "fields": [{"name": "device_id", "type": "string", "nullable": False}]
                  + [{"name": name, "type": "number", "nullable": True}
                     for name in fields],
    }
    source = tmp_path / "deep.csv"
    _write_rows(source, ["device_id", "base"],
                [{"device_id": "DEVICE-A", "base": "1"}])
    mappings = [{
        "mapping_id": "direct.base", "target": "base", "op": "direct",
        "source": "readings", "column": "base", "coerce": "numeric",
        "rule": {"id": "direct.copy.v1", "version": 1},
        "confidence": 1.0,
        "provenance": {"record_key_from": "participant", "row_number": "source"},
    }]
    previous = "base"
    for i in range(33):
        target = f"node_{i}"
        mappings.append({
            "mapping_id": f"custom.node_{i}", "target": target,
            "op": "custom_rule", "inputs": [previous],
            "output_type": "number", "params": {},
            "rule": {"id": "synthetic.identity.v1", "version": 1},
            "confidence": "from_rule",
            "provenance": {"record_key_from": "participant"},
        })
        previous = target
    spec_doc = {
        "spec_id": "synthetic/deep", "spec_version": "1.0.0",
        "trial": "synthetic_deep", "target_schema": {
            "id": "synthetic/deep", "version": "1.0.0"},
        "status": "approved", "sources": [{
            "id": "readings", "path": _relative(source), "file_type": "csv",
            "reader": "cidc_csv@v1", "logical_label": "deep.csv",
            "record_keys": ["device_id"], "participant_column": "device_id"}],
        "record": {"anchor": "external", "participant_key": "device_id",
                   "duplicates": {"policy": "error", "by": ["device_id"]}},
        "mappings": mappings,
    }
    schema_path, spec_path = tmp_path / "schema.yaml", tmp_path / "spec.yaml"
    schema_path.write_text(yaml.safe_dump(schema_doc, sort_keys=False))
    spec_path.write_text(yaml.safe_dump(spec_doc, sort_keys=False))
    schema = load_schema(schema_path)
    spec = load_spec(spec_path, target_schema=schema, require_executable=True)
    context = ReplayRunContext.create(ROOT, schema, (spec,), run_id=RUN_ID)
    anchor = pd.DataFrame([{"device_id": "DEVICE-A", "cimac_part_id": "DEVICE-A",
                            "Cimac.id": "", "Collection_Event": ""}])
    cells, diagnostics = execute(
        spec, ROOT, anchor, contract=contract_for(schema), run_id=RUN_ID)
    collector = ReplayEvidenceCollector(context)
    collector.collect(spec, diagnostics.sources, cells)
    delivered_row = {"device_id": "DEVICE-A"}
    delivered_row.update({cell.harmonized_field: cell.value for cell in cells})
    delivered = pd.DataFrame([delivered_row], columns=schema.column_order)
    provenance = pd.DataFrame([
        {name: getattr(cell, name, "")
         for name in harmonize_spec.PROVENANCE_COLUMNS} for cell in cells
    ]).rename(columns={"Cimac_id": "Cimac.id"})
    published = ReplayPublication(
        context, collector, tmp_path / "published").publish(
            delivered, provenance, artifact_id=ARTIFACT_ID,
            delivered_name="delivered.csv", provenance_name="provenance.csv")
    result = replay_cell(ROOT, published.manifest, ReplayRequest(
        ARTIFACT_ID, {"device_id": "DEVICE-A"}, "node_32"))
    assert (result.status, result.reason_code) == (
        ReplayStatus.NOT_REPLAYABLE, "REPLAY_DEPTH_EXCEEDED")

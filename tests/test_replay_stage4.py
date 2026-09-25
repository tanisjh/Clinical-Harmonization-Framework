"""Synthetic-only tests for strict ordered raw-source fallback replay."""
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
from lib.contributions import (
    STATUS_NOT_EVALUATED,
    STATUS_SELECTED,
    STATUS_UNAVAILABLE,
    from_json as contributions_from_json,
)
from lib.replay import ReplayRequest, recompute_fallback, replay_cell
from lib.replay_contract import ReplayStatus, encode_typed_raw
from lib.replay_orchestration import (
    ReplayEvidenceCollector,
    ReplayPublication,
    ReplayRunContext,
)
from lib.schema_model import contract_for, load as load_schema
from lib.source_ref import sha256_file
from lib.spec_executor import ExecutionError, execute
from lib.spec_model import load as load_spec


ROOT = Path(__file__).resolve().parents[1]
FIXTURE = ROOT / "tests/fixtures/synthetic_replay"
SCHEMA_FIXTURE = FIXTURE / "schema_stage4.yaml"
SPEC_FIXTURE = FIXTURE / "spec_stage4.yaml"
SOURCE_FIXTURES = {
    name: FIXTURE / f"sources/stage4_{name}.csv"
    for name in ("primary", "secondary", "tertiary")
}
RUN_ID = "synthetic-stage4-run-0001"
ARTIFACT_ID = "synthetic-stage4-delivered"


@dataclass
class Case:
    manifest: object
    request: ReplayRequest
    source_paths: dict[str, Path]
    spec_path: Path
    delivered_path: Path
    provenance_path: Path
    cells: list


def _relative(path: Path) -> str:
    return str(path.resolve().relative_to(ROOT.resolve()))


def _read_rows(path: Path):
    with path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        return reader.fieldnames, list(reader)


def _write_rows(path: Path, fields, rows) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _anchor(device: str) -> pd.DataFrame:
    suffix = device.removeprefix("DEVICE-")
    return pd.DataFrame([{
        "device_id": device, "reading_id": f"READ-{suffix}",
        "location": "bench-west", "cimac_part_id": device,
        "Cimac.id": f"READ-{suffix}", "Collection_Event": "bench-west",
    }])


def _bundle(tmp_path: Path, *, device="DEVICE-A",
            target="preferred_measurement", spec_mutate=None,
            source_mutate=None) -> Case:
    case_dir = tmp_path / f"{device}-{target}"
    case_dir.mkdir(parents=True)
    source_paths = {}
    for name, fixture in SOURCE_FIXTURES.items():
        path = case_dir / fixture.name
        path.write_bytes(fixture.read_bytes())
        source_paths[name] = path
    if source_mutate:
        source_mutate(source_paths)

    spec_doc = yaml.safe_load(SPEC_FIXTURE.read_text())
    for source in spec_doc["sources"]:
        source["path"] = _relative(source_paths[source["id"]])
    if spec_mutate:
        spec_mutate(spec_doc)
    spec_path = case_dir / "spec.yaml"
    spec_path.write_text(yaml.safe_dump(spec_doc, sort_keys=False))

    schema = load_schema(SCHEMA_FIXTURE)
    spec = load_spec(spec_path, target_schema=schema, require_executable=True)
    context = ReplayRunContext.create(
        ROOT, schema, (spec,), run_id=RUN_ID,
        application="synthetic-stage4-test")
    cells, diagnostics = execute(
        spec, ROOT, _anchor(device), contract=contract_for(schema), run_id=RUN_ID)
    collector = ReplayEvidenceCollector(context)
    collector.collect(spec, diagnostics.sources, cells)

    row = {name: None for name in schema.column_order}
    row.update({"device_id": device,
                "reading_id": f"READ-{device.removeprefix('DEVICE-')}",
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
        row_key={"device_id": device,
                 "reading_id": f"READ-{device.removeprefix('DEVICE-')}"},
        target_field=target)
    return Case(published.manifest, request, source_paths, spec_path,
                published.delivered_path, published.provenance_path, cells)


def _mutate_provenance(case: Case, mutate) -> None:
    fields, rows = _read_rows(case.provenance_path)
    mutate(rows)
    _write_rows(case.provenance_path, fields, rows)
    case.manifest = replace(case.manifest, provenance_artifact=replace(
        case.manifest.provenance_artifact,
        sha256=sha256_file(case.provenance_path), n_rows=len(rows)))


def _mutate_delivered(case: Case, mutate) -> None:
    fields, rows = _read_rows(case.delivered_path)
    mutate(rows)
    _write_rows(case.delivered_path, fields, rows)
    old = case.manifest.delivered_artifacts[0]
    case.manifest = replace(case.manifest, delivered_artifacts=(replace(
        old, sha256=sha256_file(case.delivered_path), n_rows=len(rows)),))


def _mutate_spec(case: Case, mutate) -> None:
    document = yaml.safe_load(case.spec_path.read_text())
    mutate(document)
    case.spec_path.write_text(yaml.safe_dump(document, sort_keys=False))
    old = case.manifest.specifications[0]
    case.manifest = replace(case.manifest, specifications=(replace(
        old, sha256=sha256_file(case.spec_path)),))


def _event(rows, mapping_id="fallback.numeric"):
    return next(row for row in rows if row["mapping_id"] == mapping_id)


def _set_contribution(event, position, **updates):
    items = json.loads(event["contributions"])
    items[position].update(updates)
    event["contributions"] = json.dumps(
        items, separators=(",", ":"), sort_keys=True)


@pytest.mark.parametrize("device,expected,statuses", [
    ("DEVICE-A", 5.0,
     [STATUS_SELECTED, STATUS_NOT_EVALUATED, STATUS_NOT_EVALUATED]),
    ("DEVICE-B", 20.0,
     [STATUS_UNAVAILABLE, STATUS_SELECTED, STATUS_NOT_EVALUATED]),
    ("DEVICE-C", 30.0,
     [STATUS_UNAVAILABLE, STATUS_UNAVAILABLE, STATUS_SELECTED]),
    ("DEVICE-D", None,
     [STATUS_UNAVAILABLE, STATUS_UNAVAILABLE, STATUS_UNAVAILABLE]),
])
def test_ordered_fallback_selection_passes_through_orchestration(
        tmp_path, device, expected, statuses):
    case = _bundle(tmp_path, device=device)
    result = replay_cell(ROOT, case.manifest, case.request)
    assert (result.status, result.reason_code, result.recomputed_value) == (
        ReplayStatus.PASS, "REPLAY_MATCH", expected)
    parent = next(cell for cell in case.cells
                  if cell.mapping_id == "fallback.numeric")
    contributions = contributions_from_json(parent.contributions)
    assert [item["rank"] for item in contributions] == [0, 1, 2]
    assert [item["status"] for item in contributions] == statuses
    usage = next(item for item in case.manifest.mapping_usages
                 if item.mapping_id == "fallback.numeric")
    assert len(usage.source_ref_ids) == 3


@pytest.mark.parametrize("device,expected", [
    ("DEVICE-G", 0.0),
    ("DEVICE-E", None),
    ("DEVICE-F", None),
    ("DEVICE-H", None),
])
def test_numeric_selection_precedes_coercion_and_never_resumes(
        tmp_path, device, expected):
    case = _bundle(tmp_path, device=device)
    result = replay_cell(ROOT, case.manifest, case.request)
    assert result.status is ReplayStatus.PASS
    assert result.recomputed_value == expected
    event = next(cell for cell in case.cells
                 if cell.mapping_id == "fallback.numeric")
    statuses = [item["status"]
                for item in contributions_from_json(event.contributions)]
    assert statuses == [STATUS_SELECTED, STATUS_NOT_EVALUATED,
                        STATUS_NOT_EVALUATED]


def test_raw_false_is_a_legitimate_winner(tmp_path):
    case = _bundle(tmp_path, device="DEVICE-H", target="preferred_raw")
    result = replay_cell(ROOT, case.manifest, case.request)
    assert (result.status, result.recomputed_value) == (ReplayStatus.PASS, "False")


def test_later_snapshot_is_audit_only_but_later_presence_is_required(tmp_path):
    case = _bundle(tmp_path)
    _mutate_provenance(case, lambda rows: _set_contribution(
        _event(rows), 2, raw_value_json="not-json", raw_value="changed"))
    assert replay_cell(ROOT, case.manifest, case.request).status is ReplayStatus.PASS

    usage = next(item for item in case.manifest.mapping_usages
                 if item.mapping_id == "fallback.numeric")
    reduced = replace(usage, source_ref_ids=usage.source_ref_ids[:-1])
    case.manifest = replace(case.manifest, mapping_usages=tuple(
        reduced if item == usage else item
        for item in case.manifest.mapping_usages))
    result = replay_cell(ROOT, case.manifest, case.request)
    assert (result.status, result.reason_code) == (
        ReplayStatus.NOT_REPLAYABLE, "SOURCE_LINK_MISSING")


def test_later_source_hash_is_still_eligibility_evidence(tmp_path):
    case = _bundle(tmp_path)
    case.source_paths["tertiary"].write_bytes(
        case.source_paths["tertiary"].read_bytes() + b"\n")
    result = replay_cell(ROOT, case.manifest, case.request)
    assert (result.status, result.reason_code) == (
        ReplayStatus.FAIL, "SOURCE_HASH_MISMATCH")


def test_later_locator_is_required_even_after_first_winner(tmp_path):
    case = _bundle(tmp_path)
    _mutate_provenance(case, lambda rows: _set_contribution(
        _event(rows), 2, source_row_number=""))
    result = replay_cell(ROOT, case.manifest, case.request)
    assert (result.status, result.reason_code) == (
        ReplayStatus.NOT_REPLAYABLE, "SOURCE_LOCATOR_INCOMPLETE")


def test_later_unsupported_reader_blocks_presence_verification(tmp_path):
    case = _bundle(tmp_path)
    sources = []
    for source in case.manifest.sources:
        if source.source_id == "tertiary":
            source = replace(
                source, reader=replace(source.reader, reader_id="other_csv@v1"))
        sources.append(source)
    case.manifest = replace(case.manifest, sources=tuple(sources))
    result = replay_cell(ROOT, case.manifest, case.request)
    assert (result.status, result.reason_code) == (
        ReplayStatus.NOT_REPLAYABLE, "READER_UNSUPPORTED")


@pytest.mark.parametrize("position,updates,reason", [
    (0, {"raw_value_json": encode_typed_raw("wrong").to_json()},
     "RAW_VALUE_MISMATCH"),
    (0, {"status": STATUS_UNAVAILABLE}, "FALLBACK_SELECTION_MISMATCH"),
    (1, {"status": STATUS_SELECTED}, "FALLBACK_SELECTION_MISMATCH"),
    (0, {"status": STATUS_NOT_EVALUATED}, "FALLBACK_SELECTION_MISMATCH"),
    (1, {"rank": 7}, "FALLBACK_CANDIDATE_CONTEXT_MISMATCH"),
    (1, {"source_column": "other"},
     "FALLBACK_CANDIDATE_CONTEXT_MISMATCH"),
])
def test_scalar_and_selection_contradictions_fail(
        tmp_path, position, updates, reason):
    case = _bundle(tmp_path)
    _mutate_provenance(case, lambda rows: _set_contribution(
        _event(rows), position, **updates))
    result = replay_cell(ROOT, case.manifest, case.request)
    assert (result.status, result.reason_code) == (ReplayStatus.FAIL, reason)


def test_incomplete_candidate_list_is_not_replayable(tmp_path):
    case = _bundle(tmp_path)
    def mutate(rows):
        event = _event(rows)
        event["contributions"] = json.dumps(
            json.loads(event["contributions"])[:-1],
            separators=(",", ":"), sort_keys=True)
    _mutate_provenance(case, mutate)
    result = replay_cell(ROOT, case.manifest, case.request)
    assert (result.status, result.reason_code) == (
        ReplayStatus.NOT_REPLAYABLE, "FALLBACK_EVIDENCE_INCOMPLETE")


@pytest.mark.parametrize("device,position", [
    ("DEVICE-B", 0),
    ("DEVICE-B", 1),
])
def test_reached_unavailable_and_selected_raw_drift_fail(
        tmp_path, device, position):
    case = _bundle(tmp_path, device=device)
    _mutate_provenance(case, lambda rows: _set_contribution(
        _event(rows), position, raw_value_json=encode_typed_raw("wrong").to_json()))
    result = replay_cell(ROOT, case.manifest, case.request)
    assert (result.status, result.reason_code) == (
        ReplayStatus.FAIL, "RAW_VALUE_MISMATCH")


def test_selected_record_key_drift_fails(tmp_path):
    case = _bundle(tmp_path, device="DEVICE-B")
    _mutate_provenance(case, lambda rows: _set_contribution(
        _event(rows), 1, source_record_key="device_id=OTHER"))
    result = replay_cell(ROOT, case.manifest, case.request)
    assert (result.status, result.reason_code) == (
        ReplayStatus.FAIL, "SOURCE_RECORD_KEY_MISMATCH")


@pytest.mark.parametrize("policy", ["error", "lookup_miss", "omit"])
def test_source_policy_does_not_waive_presence_for_an_emitted_cell(
        tmp_path, policy):
    def mutate(doc):
        for source in doc["sources"]:
            if policy == "lookup_miss":
                source["on_missing_record"] = {
                    "mode": policy, "emit_for": ["preferred_measurement"],
                    "confidence": 0.1,
                    "rule": {"id": "lookup.miss.v1", "version": 1,
                             "review_status": "approved"},
                }
            else:
                source["on_missing_record"] = {"mode": policy}
    case = _bundle(tmp_path, spec_mutate=mutate)
    assert replay_cell(ROOT, case.manifest, case.request).status is ReplayStatus.PASS


@pytest.mark.parametrize("policy", ["error", "lookup_miss", "omit"])
def test_missing_later_record_preserves_current_executor_policy(
        tmp_path, policy):
    case_dir = tmp_path / policy
    case_dir.mkdir(parents=True)
    paths = {}
    for name, fixture in SOURCE_FIXTURES.items():
        path = case_dir / fixture.name
        path.write_bytes(fixture.read_bytes())
        paths[name] = path
    fields, rows = _read_rows(paths["tertiary"])
    _write_rows(paths["tertiary"], fields,
                [row for row in rows if row["device_id"] != "DEVICE-A"])
    document = yaml.safe_load(SPEC_FIXTURE.read_text())
    for source in document["sources"]:
        source["path"] = _relative(paths[source["id"]])
        if source["id"] == "tertiary":
            if policy == "lookup_miss":
                source["on_missing_record"] = {
                    "mode": policy, "emit_for": ["preferred_measurement"],
                    "confidence": 0.1,
                    "rule": {"id": "lookup.miss.v1", "version": 1,
                             "review_status": "approved"},
                }
            else:
                source["on_missing_record"] = {"mode": policy}
    spec_path = case_dir / "spec.yaml"
    spec_path.write_text(yaml.safe_dump(document, sort_keys=False))
    schema = load_schema(SCHEMA_FIXTURE)
    spec = load_spec(spec_path, target_schema=schema, require_executable=True)
    if policy == "error":
        with pytest.raises(ExecutionError, match="missing from source"):
            execute(spec, ROOT, _anchor("DEVICE-A"),
                    contract=contract_for(schema), run_id=RUN_ID)
        return
    cells, _diagnostics = execute(
        spec, ROOT, _anchor("DEVICE-A"),
        contract=contract_for(schema), run_id=RUN_ID)
    assert not any(cell.extraction_method == "spec:fallback" for cell in cells)
    assert not any(cell.mapping_id.startswith("fallback.") for cell in cells)


@pytest.mark.parametrize("mutation", [
    lambda mapping: mapping.update({"map": {"5": "5"}}),
    lambda mapping: mapping.update({"coerce": "text"}),
    lambda mapping: mapping["contributions"].update({"record": "none"}),
])
def test_deliberately_unsupported_or_historical_fallback_is_not_replayable(
        tmp_path, mutation):
    def mutate(doc):
        mapping = next(item for item in doc["mappings"]
                       if item["mapping_id"] == "fallback.numeric")
        mutation(mapping)
    case = _bundle(tmp_path, spec_mutate=mutate)
    result = replay_cell(ROOT, case.manifest, case.request)
    assert result.status is ReplayStatus.NOT_REPLAYABLE
    assert result.reason_code in {
        "FALLBACK_CANDIDATE_UNSUPPORTED", "FALLBACK_EVIDENCE_INCOMPLETE"}


def test_mapping_condition_needs_usage_scoped_read_evidence_and_gate_passes(
        tmp_path):
    def conditional(doc):
        doc["reads"] = [{"id": "primary_value", "source": "primary",
                         "column": "candidate_value"}]
        doc["conditions"] = [{"id": "use_fallback", "read": "primary_value",
                              "predicate": "not_null"}]
        mapping = next(item for item in doc["mappings"]
                       if item["mapping_id"] == "fallback.numeric")
        mapping["when"] = "use_fallback"
    case = _bundle(tmp_path, spec_mutate=conditional)
    result = replay_cell(ROOT, case.manifest, case.request)
    assert (result.status, result.reason_code) == (
        ReplayStatus.NOT_REPLAYABLE, "NAMED_READ_EVIDENCE_MISSING")

    def gated(doc):
        mapping = next(item for item in doc["mappings"]
                       if item["mapping_id"] == "fallback.numeric")
        mapping["candidates"][0]["when"] = {
            "source": "primary", "column": "candidate_value",
            "predicate": "not_null"}
    case = _bundle(tmp_path / "gate", spec_mutate=gated)
    result = replay_cell(ROOT, case.manifest, case.request)
    assert (result.status, result.reason_code) == (
        ReplayStatus.PASS, "REPLAY_MATCH")

@pytest.mark.parametrize("kind", ["anchor", "group", "excel"])
def test_anchor_group_and_excel_candidates_remain_out_of_scope(tmp_path, kind):
    case = _bundle(tmp_path)
    def mutate(doc):
        mapping = next(item for item in doc["mappings"]
                       if item["mapping_id"] == "fallback.numeric")
        if kind == "anchor":
            mapping["candidates"][2]["source"] = "$anchor"
        elif kind == "group":
            doc["source_groups"] = [{
                "id": "split", "decided_by": "measurement",
                "members": [
                    {"id": "left", "sources": {"measurement": "primary"}},
                    {"id": "right", "sources": {"measurement": "secondary"}},
                ],
            }]
            mapping["candidates"][2]["source"] = "split.measurement"
        else:
            source = next(item for item in doc["sources"]
                          if item["id"] == "tertiary")
            source.update({"file_type": "xlsx", "reader": "excel@v1",
                           "sheet": "Sheet1"})
    _mutate_spec(case, mutate)
    result = replay_cell(ROOT, case.manifest, case.request)
    assert (result.status, result.reason_code) == (
        ReplayStatus.NOT_REPLAYABLE, "FALLBACK_CANDIDATE_UNSUPPORTED")


def test_wrong_rule_and_confidence_are_verified_from_recomputed_branch(tmp_path):
    case = _bundle(tmp_path)
    _mutate_provenance(case, lambda rows: _event(rows).update(
        {"rule_id": "fallback.no_value.v1"}))
    result = replay_cell(ROOT, case.manifest, case.request)
    assert (result.status, result.reason_code) == (
        ReplayStatus.FAIL, "RULE_IDENTITY_MISMATCH")

    case = _bundle(tmp_path / "confidence")
    _mutate_provenance(case, lambda rows: _event(rows).update(
        {"confidence": "0.2"}))
    result = replay_cell(ROOT, case.manifest, case.request)
    assert (result.status, result.reason_code) == (
        ReplayStatus.FAIL, "CONFIDENCE_CONTEXT_MISMATCH")


def test_missing_confidence_is_not_replayable(tmp_path):
    case = _bundle(tmp_path)
    _mutate_provenance(case, lambda rows: _event(rows).update({"confidence": ""}))
    result = replay_cell(ROOT, case.manifest, case.request)
    assert (result.status, result.reason_code) == (
        ReplayStatus.NOT_REPLAYABLE, "MANIFEST_REQUIRED_FIELD_MISSING")


def test_changed_delivery_does_not_change_fallback_recomputation(tmp_path):
    case = _bundle(tmp_path)
    _mutate_delivered(case, lambda rows: rows[0].update(
        {"preferred_measurement": "999"}))
    _mutate_provenance(case, lambda rows: _event(rows).update({"value": "999"}))
    result = replay_cell(ROOT, case.manifest, case.request)
    assert (result.status, result.reason_code, result.recomputed_value) == (
        ReplayStatus.FAIL, "RECOMPUTED_VALUE_MISMATCH", 5.0)


def test_parent_raw_snapshot_is_comparison_only(tmp_path):
    case = _bundle(tmp_path)
    _mutate_provenance(case, lambda rows: _event(rows).update(
        {"raw_value": "999", "raw_value_json": encode_typed_raw(999).to_json()}))
    result = replay_cell(ROOT, case.manifest, case.request)
    assert (result.status, result.reason_code) == (
        ReplayStatus.FAIL, "RAW_VALUE_MISMATCH")


def test_numeric_comparison_has_no_epsilon(tmp_path):
    case = _bundle(tmp_path)
    changed = "5.000000000000001"
    _mutate_delivered(case, lambda rows: rows[0].update(
        {"preferred_measurement": changed}))
    _mutate_provenance(case, lambda rows: _event(rows).update({"value": changed}))
    result = replay_cell(ROOT, case.manifest, case.request)
    assert (result.status, result.reason_code) == (
        ReplayStatus.FAIL, "RECOMPUTED_VALUE_MISMATCH")


def test_recompute_boundary_excludes_delivered_and_provenance_outputs():
    signature = inspect.signature(recompute_fallback)
    assert set(signature.parameters) == {"reached_candidate_scalars", "coerce"}
    assert "delivered" not in str(signature)
    assert "provenance" not in str(signature)
    assert recompute_fallback([float("nan"), ".", "10"],
                              coerce="numeric").value is None
    false_winner = recompute_fallback([False, "10"], coerce="none")
    assert false_winner.selected_rank == 0
    assert false_winner.value is False


def test_nonfinite_numeric_result_cannot_pass(tmp_path):
    def source_mutate(paths):
        fields, rows = _read_rows(paths["primary"])
        rows[0]["candidate_value"] = "1e309"
        _write_rows(paths["primary"], fields, rows)
    case = _bundle(tmp_path, source_mutate=source_mutate)
    result = replay_cell(ROOT, case.manifest, case.request)
    assert result.status is ReplayStatus.FAIL
    assert result.reason_code == "RECOMPUTED_VALUE_MISMATCH"

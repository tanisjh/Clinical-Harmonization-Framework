"""Synthetic-only tests for Stage 19 record-construction authority."""
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
from lib.record_construction import (
    ORIGIN_OBSERVED,
    ORIGIN_SYNTHESIZED,
    RecordConstructionError,
    construct_grouping_records,
    prepare_participant_frame,
)
from lib.replay import ReplayRequest, replay_cell
from lib.replay_contract import REASON_CODES_VERSION, ReplayStatus
from lib.replay_orchestration import (
    ReplayEvidenceCollector,
    ReplayOrchestrationError,
    ReplayPublication,
    ReplayRunContext,
)
from lib.schema_model import load as load_schema
from lib.source_ref import sha256_file
from lib.spec_executor import execute
from lib.spec_model import Duplicates, load as load_spec


ROOT = Path(__file__).resolve().parents[1]
FIXTURE = ROOT / "tests/fixtures/synthetic_replay"
SCHEMA = FIXTURE / "schema_stage19.yaml"
SPEC = FIXTURE / "spec_stage19.yaml"
GROUP_SOURCE = FIXTURE / "sources/stage10_grouping.csv"
PARTICIPANT_SOURCE = FIXTURE / "sources/stage10_participants.csv"
RUN_ID = "synthetic-stage19-run-0001"
ARTIFACT_ID = "synthetic-stage19-delivered"


@dataclass
class Case:
    manifest: object
    request: ReplayRequest
    spec: object
    cells: list
    delivered_path: Path
    provenance_path: Path
    group_path: Path
    participant_path: Path


def _relative(path: Path) -> str:
    return str(path.resolve().relative_to(ROOT.resolve()))


def _write_rows(path: Path, fieldnames, rows) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _read_rows(path: Path):
    with path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        return reader.fieldnames, list(reader)


def _bundle(tmp_path: Path, *, participant="DEVICE-A",
            target="grouped_value", spec_mutator=None) -> Case:
    case_dir = tmp_path / f"{participant}-{target}"
    case_dir.mkdir(parents=True)
    group_path = case_dir / "grouping.csv"
    participant_path = case_dir / "participants.csv"
    group_path.write_bytes(GROUP_SOURCE.read_bytes())
    participant_path.write_bytes(PARTICIPANT_SOURCE.read_bytes())
    document = yaml.safe_load(SPEC.read_text())
    document["sources"][0]["path"] = _relative(group_path)
    document["sources"][1]["path"] = _relative(participant_path)
    if spec_mutator is not None:
        spec_mutator(document)
    spec_path = case_dir / "spec.yaml"
    spec_path.write_text(yaml.safe_dump(document, sort_keys=False))

    schema = load_schema(SCHEMA)
    spec = load_spec(spec_path, target_schema=schema, require_executable=True)
    context = ReplayRunContext.create(
        ROOT, schema, (spec,), run_id=RUN_ID,
        application="synthetic-stage19-test")
    cells, diagnostics = execute(
        spec, ROOT, contract=None, run_id=RUN_ID)
    collector = ReplayEvidenceCollector(context)
    collector.collect(spec, diagnostics.sources, cells)

    by_key = {}
    for cell in cells:
        key = (cell.cimac_part_id, cell.Cimac_id, cell.Collection_Event)
        row = by_key.setdefault(
            key, {name: None for name in schema.column_order})
        row.update({
            "cimac_part_id": key[0], "Cimac.id": key[1],
            "Collection_Event": key[2], cell.harmonized_field: cell.value,
        })
    delivered = pd.DataFrame(by_key.values(), columns=schema.column_order)
    provenance = pd.DataFrame([
        {name: getattr(cell, name, "")
         for name in harmonize_spec.PROVENANCE_COLUMNS}
        for cell in cells
    ]).rename(columns={"Cimac_id": "Cimac.id"})
    published = ReplayPublication(
        context, collector, case_dir / "published").publish(
            delivered, provenance, artifact_id=ARTIFACT_ID,
            delivered_name="delivered.csv", provenance_name="provenance.csv")
    selected = delivered.loc[
        delivered["cimac_part_id"] == participant].iloc[0]
    request = ReplayRequest(
        artifact_id=ARTIFACT_ID,
        row_key={name: selected[name] for name in schema.row_key},
        target_field=target)
    return Case(
        published.manifest, request, spec, cells,
        published.delivered_path, published.provenance_path,
        group_path, participant_path)


def _mapping(case: Case, mapping_id=None):
    mapping_id = mapping_id or next(
        cell.mapping_id for cell in case.cells
        if cell.harmonized_field == case.request.target_field
        and cell.cimac_part_id == case.request.row_key["cimac_part_id"])
    return next(item for item in case.spec.mappings
                if item.mapping_id == mapping_id)


def _usage(case: Case, mapping_id=None):
    mapping = _mapping(case, mapping_id)
    return next(item for item in case.manifest.mapping_usages
                if item.spec_ref == case.spec.ref
                and item.mapping_id == mapping.mapping_id)


def _replace_usage(case: Case, usage) -> None:
    old = _usage(case, usage.mapping_id)
    case.manifest = replace(
        case.manifest,
        mapping_usages=tuple(
            usage if item == old else item
            for item in case.manifest.mapping_usages))


def _mutate_provenance(case: Case, mapping_id=None, **updates) -> None:
    fields, rows = _read_rows(case.provenance_path)
    mapping = _mapping(case, mapping_id)
    row = next(item for item in rows
               if item["mapping_id"] == mapping.mapping_id
               and item["cimac_part_id"]
                  == case.request.row_key["cimac_part_id"]
               and item["Cimac.id"] == case.request.row_key["Cimac.id"])
    row.update(updates)
    _write_rows(case.provenance_path, fields, rows)
    case.manifest = replace(
        case.manifest,
        provenance_artifact=replace(
            case.manifest.provenance_artifact,
            sha256=sha256_file(case.provenance_path)))


def _pure_spec(tmp_path: Path, mutator=None):
    document = yaml.safe_load(SPEC.read_text())
    if mutator:
        mutator(document)
    path = tmp_path / "pure-spec.yaml"
    path.write_text(yaml.safe_dump(document, sort_keys=False))
    return load_spec(
        path, target_schema=load_schema(SCHEMA), require_executable=True)


def test_pure_construction_preserves_group_size_first_row_and_origin(tmp_path):
    spec = _pure_spec(tmp_path)
    result = construct_grouping_records(
        spec, pd.read_csv(GROUP_SOURCE),
        {"participants": pd.read_csv(PARTICIPANT_SOURCE)})
    observed = next(item for item in result.records
                    if item.anchor_row["cimac_part_id"] == "DEVICE-A")
    synthesized = next(item for item in result.records
                       if item.anchor_row["cimac_part_id"] == "DEVICE-C")
    assert (observed.origin, observed.group_size,
            observed.selected_source_row) == (ORIGIN_OBSERVED, 2, 0)
    assert observed.group_key == (
        ("cimac_part_id", "DEVICE-A"),
        ("Cimac.id", "READ-001"),
        ("Collection_Event", "  bench   west  "),
    )
    assert (synthesized.origin, synthesized.declaration_id,
            synthesized.group_key, synthesized.selected_source_row,
            synthesized.group_size) == (
                ORIGIN_SYNTHESIZED, "missing_reading", (), None, None)
    assert result.additional_counts == (("missing_reading", 1),)


def test_pure_construction_counts_noncontiguous_duplicate_rows(tmp_path):
    spec = _pure_spec(tmp_path)
    frame = pd.DataFrame([
        {"cimac_part_id": "A", "Cimac.id": "1", "Collection_Event": "x"},
        {"cimac_part_id": "B", "Cimac.id": "2", "Collection_Event": "y"},
        {"cimac_part_id": "A", "Cimac.id": "1", "Collection_Event": "x"},
    ])
    participants = pd.DataFrame({"cimac_part_id": ["A", "B"]})
    first = construct_grouping_records(
        spec, frame, {"participants": participants})
    second = construct_grouping_records(
        spec, frame, {"participants": participants})
    assert first == second
    record = first.records[0]
    assert (record.group_size, record.selected_source_row) == (2, 0)
    assert [item.anchor_row["cimac_part_id"] for item in first.records] == [
        "A", "B"]


def test_pure_construction_stringifies_keys_and_projects_only_declared_fields(
        tmp_path):
    spec = _pure_spec(tmp_path)
    frame = pd.DataFrame([{
        "cimac_part_id": 7, "Cimac.id": 3.0, "Collection_Event": None,
        "not_projected": "forbidden",
    }])
    result = construct_grouping_records(
        spec, frame,
        {"participants": pd.DataFrame({"cimac_part_id": [7]})})
    record = result.records[0]
    assert record.group_key == (
        ("cimac_part_id", "7"), ("Cimac.id", "3.0"),
        ("Collection_Event", "None"))
    assert "not_projected" not in record.anchor_row


@pytest.mark.parametrize("policy,expected", [("first", 0), ("last", 1)])
def test_participant_duplicate_policy_preserves_source_order(
        tmp_path, policy, expected):
    spec = _pure_spec(tmp_path)
    spec = replace(
        spec, record=replace(
            spec.record, duplicates=Duplicates(
                policy=policy, by=("cimac_part_id",))))
    frame = pd.DataFrame({"cimac_part_id": ["A", "A"], "marker": [1, 2]})
    prepared = prepare_participant_frame(spec, "participants", frame)
    assert int(prepared.loc["A", "_spec_src_row"]) == expected


def test_participant_duplicate_error_is_fail_closed(tmp_path):
    spec = _pure_spec(tmp_path)
    spec = replace(
        spec, record=replace(
            spec.record, duplicates=Duplicates(
                policy="error", by=("cimac_part_id",))))
    frame = pd.DataFrame({"cimac_part_id": ["A", "A"]})
    with pytest.raises(RecordConstructionError, match="duplicate records"):
        prepare_participant_frame(spec, "participants", frame)


def test_pure_helper_has_no_io_or_recorded_output_inputs():
    signature = inspect.signature(construct_grouping_records)
    assert tuple(signature.parameters) == (
        "spec", "anchor_frame", "participant_frames")
    source = inspect.getsource(construct_grouping_records)
    for forbidden in ("Cell", "delivery", "provenance", "read_csv", "open("):
        assert forbidden not in source


@pytest.mark.parametrize("participant,target,expected", [
    ("DEVICE-A", "grouped_value", "  bench   west  "),
    ("DEVICE-B", "origin_value", "bench-east"),
    ("DEVICE-C", "origin_value", "declared"),
])
def test_grouping_anchor_and_both_origin_routes_pass(
        tmp_path, participant, target, expected):
    case = _bundle(tmp_path, participant=participant, target=target)
    result = replay_cell(ROOT, case.manifest, case.request)
    assert (result.status, result.reason_code, result.recomputed_value) == (
        ReplayStatus.PASS, "REPLAY_MATCH", expected), result.message
    usage = _usage(case)
    assert len(usage.evidence_ref_ids) == 1
    assert usage.evidence_ref_ids[0].startswith("anchor-")
    assert len(usage.source_ref_ids) == 2


def test_anchor_facts_are_independently_rendered(tmp_path):
    case = _bundle(tmp_path)
    cell = next(item for item in case.cells
                if item.mapping_id == "construction.grouped_value"
                and item.cimac_part_id == "DEVICE-A")
    assert json.loads(cell.derivation) == {
        "final_value": "  bench   west  ",
        "kind": "anchor_from_grouping",
        "n_source_rows": 2,
        "selected_row": 0,
    }
    forged = json.loads(cell.derivation)
    forged["n_source_rows"] = 99
    _mutate_provenance(
        case, mapping_id="construction.grouped_value",
        derivation=json.dumps(forged, separators=(",", ":"), sort_keys=True))
    result = replay_cell(ROOT, case.manifest, case.request)
    assert (result.status, result.reason_code) == (
        ReplayStatus.FAIL, "DERIVATION_MISMATCH")


def test_missing_anchor_derivation_remains_not_replayable(tmp_path):
    case = _bundle(tmp_path)
    _mutate_provenance(
        case, mapping_id="construction.grouped_value", derivation="")
    result = replay_cell(ROOT, case.manifest, case.request)
    assert (result.status, result.reason_code) == (
        ReplayStatus.NOT_REPLAYABLE, "DERIVATION_EVIDENCE_MISSING")


def test_raw_mismatch_precedes_derivation(tmp_path):
    case = _bundle(tmp_path)
    _mutate_provenance(
        case, mapping_id="construction.grouped_value",
        raw_value_json='{"kind":"string","value":"forged"}',
        derivation='{"forged":"derivation"}')
    result = replay_cell(ROOT, case.manifest, case.request)
    assert (result.status, result.reason_code) == (
        ReplayStatus.FAIL, "RAW_VALUE_MISMATCH")


def test_value_state_mismatch_precedes_derivation(tmp_path):
    case = _bundle(tmp_path)
    _mutate_provenance(
        case, mapping_id="construction.grouped_value",
        value_state="provisional", derivation='{"forged":"derivation"}')
    result = replay_cell(ROOT, case.manifest, case.request)
    assert (result.status, result.reason_code) == (
        ReplayStatus.FAIL, "VALUE_STATE_CONTEXT_MISMATCH")


def test_selected_source_row_is_construction_authority(tmp_path):
    case = _bundle(tmp_path, participant="DEVICE-A", target="origin_value")
    _mutate_provenance(
        case, mapping_id="origin.observed", source_row_number="1")
    result = replay_cell(ROOT, case.manifest, case.request)
    assert (result.status, result.reason_code) == (
        ReplayStatus.FAIL, "ANCHOR_SELECTION_MISMATCH")


def test_group_key_is_comparison_only(tmp_path):
    case = _bundle(tmp_path, participant="DEVICE-A", target="origin_value")
    _mutate_provenance(
        case, mapping_id="origin.observed",
        source_record_key="cimac_part_id=FORGED")
    result = replay_cell(ROOT, case.manifest, case.request)
    assert (result.status, result.reason_code) == (
        ReplayStatus.FAIL, "ANCHOR_SELECTION_MISMATCH")


@pytest.mark.parametrize("link", ["cell", "usage"])
def test_missing_event_scoped_anchor_link_is_not_replayable(tmp_path, link):
    case = _bundle(tmp_path, participant="DEVICE-B", target="origin_value")
    if link == "cell":
        _mutate_provenance(case, mapping_id="origin.observed", evidence_refs="")
    else:
        usage = _usage(case, "origin.observed")
        _replace_usage(case, replace(usage, evidence_ref_ids=()))
    result = replay_cell(ROOT, case.manifest, case.request)
    assert (result.status, result.reason_code) == (
        ReplayStatus.NOT_REPLAYABLE, "ANCHOR_EVIDENCE_MISSING")


def test_missing_participant_universe_link_cannot_prove_synthesis(tmp_path):
    case = _bundle(tmp_path, participant="DEVICE-C", target="origin_value")
    usage = _usage(case, "origin.synthesized")
    participant_binding = next(
        item for item in case.manifest.sources
        if item.source_id == "participants")
    _replace_usage(case, replace(
        usage, source_ref_ids=tuple(
            ref for ref in usage.source_ref_ids
            if ref != participant_binding.source_ref_id)))
    result = replay_cell(ROOT, case.manifest, case.request)
    assert (result.status, result.reason_code) == (
        ReplayStatus.NOT_REPLAYABLE, "ANCHOR_EVIDENCE_MISSING")


def test_bound_source_hash_failure_precedes_origin_branch(tmp_path):
    case = _bundle(tmp_path, participant="DEVICE-A", target="origin_value")
    case.group_path.write_text(case.group_path.read_text() + "\n")
    result = replay_cell(ROOT, case.manifest, case.request)
    assert (result.status, result.reason_code) == (
        ReplayStatus.FAIL, "SOURCE_HASH_MISMATCH")


def test_wrong_origin_sibling_is_rejected_after_construction(tmp_path):
    case = _bundle(tmp_path, participant="DEVICE-A", target="origin_value")
    fields, rows = _read_rows(case.provenance_path)
    correct = next(row for row in rows
                   if row["mapping_id"] == "origin.observed"
                   and row["cimac_part_id"] == "DEVICE-A")
    wrong = next(row for row in rows
                 if row["mapping_id"] == "origin.synthesized"
                 and row["cimac_part_id"] == "DEVICE-C")
    wrong = dict(wrong)
    for name in ("trial", "cimac_part_id", "Cimac.id", "Collection_Event"):
        wrong[name] = correct[name]
    wrong["source_record_key"] = "cimac_part_id=DEVICE-A"
    rows.remove(correct)
    rows.append(wrong)
    _write_rows(case.provenance_path, fields, rows)
    case.manifest = replace(
        case.manifest,
        provenance_artifact=replace(
            case.manifest.provenance_artifact,
            sha256=sha256_file(case.provenance_path)))
    result = replay_cell(ROOT, case.manifest, case.request)
    assert (result.status, result.reason_code) == (
        ReplayStatus.FAIL, "BRANCH_SELECTION_MISMATCH")


@pytest.mark.parametrize("extra_first", [False, True])
def test_extra_authoritative_origin_sibling_is_rejected_in_both_orders(
        tmp_path, extra_first):
    case = _bundle(tmp_path, participant="DEVICE-A", target="origin_value")
    fields, rows = _read_rows(case.provenance_path)
    correct = next(row for row in rows
                   if row["mapping_id"] == "origin.observed"
                   and row["cimac_part_id"] == "DEVICE-A")
    false_sibling = dict(next(
        row for row in rows
        if row["mapping_id"] == "origin.synthesized"
        and row["cimac_part_id"] == "DEVICE-C"))
    for name in ("trial", "cimac_part_id", "Cimac.id", "Collection_Event"):
        false_sibling[name] = correct[name]
    rows.remove(correct)
    pair = [false_sibling, correct] if extra_first else [correct, false_sibling]
    rows.extend(pair)
    _write_rows(case.provenance_path, fields, rows)
    case.manifest = replace(
        case.manifest,
        provenance_artifact=replace(
            case.manifest.provenance_artifact,
            sha256=sha256_file(case.provenance_path)))
    result = replay_cell(ROOT, case.manifest, case.request)
    assert (result.status, result.reason_code) == (
        ReplayStatus.NOT_REPLAYABLE, "PROVENANCE_AUTHORITY_AMBIGUOUS")


def test_publication_rejects_missing_construction_evidence(tmp_path):
    case_dir = tmp_path / "publication"
    case_dir.mkdir()
    document = yaml.safe_load(SPEC.read_text())
    group_path = case_dir / "grouping.csv"
    participant_path = case_dir / "participants.csv"
    group_path.write_bytes(GROUP_SOURCE.read_bytes())
    participant_path.write_bytes(PARTICIPANT_SOURCE.read_bytes())
    document["sources"][0]["path"] = _relative(group_path)
    document["sources"][1]["path"] = _relative(participant_path)
    spec_path = case_dir / "spec.yaml"
    spec_path.write_text(yaml.safe_dump(document, sort_keys=False))
    schema = load_schema(SCHEMA)
    spec = load_spec(spec_path, target_schema=schema, require_executable=True)
    context = ReplayRunContext.create(
        ROOT, schema, (spec,), run_id=RUN_ID,
        application="synthetic-stage19-publication")
    cells, diagnostics = execute(spec, ROOT, contract=None, run_id=RUN_ID)
    cells = [
        replace(cell, evidence_refs="")
        if cell.mapping_id == "origin.observed" else cell
        for cell in cells
    ]
    collector = ReplayEvidenceCollector(context)
    with pytest.raises(
            ReplayOrchestrationError,
            match="incomplete or conflicting runtime anchor identity"):
        collector.collect(spec, diagnostics.sources, cells)


def test_contract_versions_remain_unchanged():
    assert REASON_CODES_VERSION == 10


def test_stage19_scope_does_not_broaden_unrelated_routes(tmp_path):
    spec = _pure_spec(tmp_path)
    mapping = next(item for item in spec.mappings
                   if item.mapping_id == "construction.grouped_value")
    assert replay_module._is_stage19_construction_route(mapping, spec)
    assert not replay_module._is_stage19_construction_route(
        replace(mapping, op="custom_rule", implementation="forbidden"), spec)
    highest = replace(
        spec, record=replace(
            spec.record, duplicates=Duplicates(
                policy="highest_confidence", by=("cimac_part_id",))))
    assert not replay_module._is_stage19_construction_route(mapping, highest)
    external = replace(spec, record=replace(
        spec.record, anchor="external", anchor_source="", grouping=None,
        additional_records=()))
    assert not replay_module._is_stage19_construction_route(mapping, external)

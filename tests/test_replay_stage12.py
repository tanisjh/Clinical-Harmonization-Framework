"""Synthetic-only tests for ordered source-group strict replay (Stage 12)."""
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
from lib import rule_registry
import lib.replay as replay_module
from lib.contributions import STATUS_CONDITION, STATUS_NOT_EVALUATED
from lib.group_selection import (
    is_closed_source_group_mapping, select_first_containing_record,
)
from lib.replay import ReplayRequest, recompute_file_membership, replay_cell
from lib.replay_contract import (
    REASON_CODES, REASON_CODES_VERSION, ReplayStatus,
)
from lib.replay_orchestration import (
    ReplayEvidenceCollector, ReplayOrchestrationError, ReplayPublication,
    ReplayRunContext,
)
from lib.rule_registry import RegisteredRule
from lib.schema_model import contract_for, load as load_schema
from lib.source_ref import sha256_file
from lib.spec_executor import execute
from lib.spec_model import load as load_spec

ROOT = Path(__file__).resolve().parents[1]
FIXTURE = ROOT / "tests/fixtures/synthetic_replay"
SCHEMA = FIXTURE / "schema_stage12.yaml"
SPEC = FIXTURE / "spec_stage12.yaml"
SOURCES = tuple(sorted((FIXTURE / "sources").glob("stage12_*.csv")))
VOCABULARY = FIXTURE / "vocabularies/stage12_ordered.yaml"
RUN_ID = "synthetic-stage12-run-0001"
ARTIFACT_ID = "synthetic-stage12-delivered"


def _identity(value):
    if value is None:
        return None, 0.2, "missing"
    return value, 0.88, "synthetic identity"


@pytest.fixture(autouse=True)
def _rules(monkeypatch):
    monkeypatch.setitem(
        rule_registry.REGISTRY,
        ("synthetic.stage12.identity.v1", "1"),
        RegisteredRule(
            "synthetic.stage12.identity.v1", "1", rule_registry.KIND_FIELD,
            _identity, arity=1),
    )


@dataclass
class Case:
    manifest: object
    request: ReplayRequest
    spec: object
    cells: list
    source_paths: dict[str, Path]
    delivered_path: Path
    provenance_path: Path


def _relative(path: Path) -> str:
    return str(path.resolve().relative_to(ROOT.resolve()))


def _anchor(participant: str) -> pd.DataFrame:
    return pd.DataFrame([{
        "device_id": participant,
        "reading_id": f"READ-{participant}",
        "location": "bench-west",
        "cimac_part_id": participant,
        "Cimac.id": f"READ-{participant}",
        "Collection_Event": "bench-west",
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


def _bundle(tmp_path: Path, *, participant="DEVICE-A", target="direct_none",
            membership_status="selected", cell_mutator=None,
            spec_mutate=None) -> Case:
    case_dir = tmp_path / f"{participant}-{target}"
    case_dir.mkdir(parents=True)
    source_paths = {}
    for source in SOURCES:
        path = case_dir / source.name
        path.write_bytes(source.read_bytes())
        source_paths[source.stem.removeprefix("stage12_")] = path
    vocabulary_dir = case_dir / "vocabularies"
    vocabulary_dir.mkdir()
    (vocabulary_dir / VOCABULARY.name).write_bytes(VOCABULARY.read_bytes())

    document = yaml.safe_load(SPEC.read_text())
    for source in document["sources"]:
        source["path"] = _relative(source_paths[source["id"]])
    if membership_status == "condition":
        document["source_groups"][0]["membership_status"] = "condition"
    if spec_mutate is not None:
        spec_mutate(document)
    spec_path = case_dir / "spec.yaml"
    spec_path.write_text(yaml.safe_dump(document, sort_keys=False))

    schema = load_schema(SCHEMA)
    spec = load_spec(
        spec_path, target_schema=schema, require_executable=True,
        vocabulary_dir=vocabulary_dir)
    context = ReplayRunContext.create(
        ROOT, schema, (spec,), run_id=RUN_ID,
        application="synthetic-stage12-test")
    cells, diagnostics = execute(
        spec, ROOT, _anchor(participant), contract=contract_for(schema),
        run_id=RUN_ID)
    if cell_mutator is not None:
        cells = [cell_mutator(cell) for cell in cells]
    collector = ReplayEvidenceCollector(context)
    collector.collect(spec, diagnostics.sources, cells)

    row = {name: None for name in schema.column_order}
    row.update({"device_id": participant,
                "reading_id": f"READ-{participant}",
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
        row_key={"device_id": participant,
                 "reading_id": f"READ-{participant}"},
        target_field=target)
    return Case(
        published.manifest, request, spec, cells, source_paths,
        published.delivered_path, published.provenance_path)


def _mapping(case: Case):
    return next(mapping for mapping in case.spec.mappings
                if mapping.target == case.request.target_field)


def _usage(case: Case):
    mapping = _mapping(case)
    return next(usage for usage in case.manifest.mapping_usages
                if usage.spec_ref == case.spec.ref
                and usage.mapping_id == mapping.mapping_id)


def _cell(case: Case):
    mapping = _mapping(case)
    return next(cell for cell in case.cells if cell.mapping_id == mapping.mapping_id)


def _source(case: Case, source_id: str):
    return next(source for source in case.manifest.sources
                if source.spec_ref == case.spec.ref and source.source_id == source_id)


def _replace_usage(case: Case, usage) -> None:
    old = _usage(case)
    case.manifest = replace(
        case.manifest,
        mapping_usages=tuple(usage if item == old else item
                             for item in case.manifest.mapping_usages))


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


def _mutate_contributions(case: Case, mutate) -> None:
    fields, rows = _read_rows(case.provenance_path)
    mapping = _mapping(case)
    row = next(item for item in rows if item["mapping_id"] == mapping.mapping_id)
    contributions = json.loads(row["contributions"])
    mutate(contributions)
    row["contributions"] = json.dumps(
        contributions, separators=(",", ":"), sort_keys=True)
    _write_rows(case.provenance_path, fields, rows)
    case.manifest = replace(
        case.manifest,
        provenance_artifact=replace(
            case.manifest.provenance_artifact,
            sha256=sha256_file(case.provenance_path)))


def _mutate_delivered(case: Case, value) -> None:
    fields, rows = _read_rows(case.delivered_path)
    rows[0][case.request.target_field] = value
    _write_rows(case.delivered_path, fields, rows)
    artifact = case.manifest.delivered_artifacts[0]
    case.manifest = replace(
        case.manifest,
        delivered_artifacts=(replace(
            artifact, sha256=sha256_file(case.delivered_path)),))


@pytest.mark.parametrize("participant,target,expected", [
    ("DEVICE-A", "membership_label", "alpha-value"),
    ("DEVICE-B", "membership_label", "beta-value"),
    ("DEVICE-A", "direct_none", "raw-alpha"),
    ("DEVICE-B", "direct_none", "raw-beta"),
    ("DEVICE-A", "direct_text", "spaced alpha"),
    ("DEVICE-B", "direct_numeric", 7.0),
    ("DEVICE-A", "status_label", "ready"),
    ("DEVICE-B", "status_label", None),
    ("DEVICE-A", "vocabulary_label", "vocabulary-hit"),
    ("DEVICE-A", "recursive_label", "spaced alpha"),
])
def test_group_operations_and_recursive_dependency_pass(
        tmp_path, participant, target, expected):
    case = _bundle(tmp_path, participant=participant, target=target)
    result = replay_cell(ROOT, case.manifest, case.request)
    assert (result.status, result.reason_code, result.recomputed_value) == (
        ReplayStatus.PASS, "REPLAY_MATCH", expected), result.message


@pytest.mark.parametrize("participant,expected", [
    ("DEVICE-BOTH", "first-wins"),
    ("DEVICE-DUP", "duplicate-first"),
])
def test_order_and_duplicate_policy_are_reconstructed(tmp_path, participant, expected):
    case = _bundle(tmp_path, participant=participant)
    contributions = json.loads(_cell(case).contributions)
    result = replay_cell(ROOT, case.manifest, case.request)
    assert result.status == ReplayStatus.PASS, result.message
    assert result.recomputed_value == expected
    assert [item["status"] for item in contributions] == [
        "selected", "not_evaluated"]


def test_condition_membership_status_is_preserved(tmp_path):
    case = _bundle(tmp_path, participant="DEVICE-B", membership_status="condition")
    contributions = json.loads(_cell(case).contributions)
    assert [item["status"] for item in contributions] == [
        "unavailable", STATUS_CONDITION]
    assert replay_cell(ROOT, case.manifest, case.request).status == ReplayStatus.PASS


def test_deciding_role_and_scalar_role_are_independent(tmp_path):
    case = _bundle(tmp_path, participant="DEVICE-B")
    contributions = json.loads(_cell(case).contributions)
    deciding = {_source(case, "deciding_alpha").source_ref_id,
                _source(case, "deciding_beta").source_ref_id}
    assert {item["source_ref_id"] for item in contributions} == deciding
    assert _cell(case).source_ref_id == _source(case, "scalar_beta").source_ref_id
    assert replay_cell(ROOT, case.manifest, case.request).status == ReplayStatus.PASS


def test_pure_selection_helper_has_no_recorded_evidence_inputs():
    signature = inspect.signature(select_first_containing_record)
    assert set(signature.parameters) == {
        "ordered_member_ids", "presence_by_member", "membership_status"}
    decision = select_first_containing_record(
        ("alpha", "beta"), {"alpha": False, "beta": True})
    assert (decision.selected_member_id, decision.statuses) == (
        "beta", ("unavailable", "selected"))


def test_existing_operation_helpers_are_reused(tmp_path, monkeypatch):
    direct = _bundle(tmp_path, target="direct_none")
    calls = []
    original = replay_module.recompute_direct
    monkeypatch.setattr(
        replay_module, "recompute_direct",
        lambda raw, *, coerce: (
            calls.append((raw, coerce)), original(raw, coerce=coerce))[1])
    assert replay_cell(ROOT, direct.manifest, direct.request).status == ReplayStatus.PASS
    assert calls == [("raw-alpha", "none")]

    mapped = _bundle(tmp_path, target="status_label")
    calls = []
    original_map = replay_module.recompute_value_map
    monkeypatch.setattr(
        replay_module, "recompute_value_map",
        lambda *args, **kwargs: (
            calls.append((args, kwargs)), original_map(*args, **kwargs))[1])
    assert replay_cell(ROOT, mapped.manifest, mapped.request).status == ReplayStatus.PASS
    assert len(calls) == 1

    vocabulary = _bundle(tmp_path, target="vocabulary_label")
    calls = []
    original_vocab = replay_module.recompute_vocabulary_value_map
    monkeypatch.setattr(
        replay_module, "recompute_vocabulary_value_map",
        lambda *args, **kwargs: (
            calls.append((args, kwargs)), original_vocab(*args, **kwargs))[1])
    assert replay_cell(
        ROOT, vocabulary.manifest, vocabulary.request).status == ReplayStatus.PASS
    assert len(calls) == 1


@pytest.mark.parametrize("mutation", [
    "empty_contributions", "missing_contribution", "missing_deciding_link",
    "missing_scalar_link", "missing_cell_source", "missing_row", "missing_record_key",
])
def test_missing_group_evidence_is_not_replayable(tmp_path, mutation):
    case = _bundle(tmp_path, participant="DEVICE-B")
    if mutation == "empty_contributions":
        _mutate_provenance(case, contributions="")
    elif mutation == "missing_contribution":
        _mutate_contributions(case, lambda items: items.pop(0))
    elif mutation in ("missing_deciding_link", "missing_scalar_link"):
        source_id = ("deciding_alpha" if mutation == "missing_deciding_link"
                     else "scalar_beta")
        usage = _usage(case)
        ref = _source(case, source_id).source_ref_id
        _replace_usage(case, replace(
            usage, source_ref_ids=tuple(
                item for item in usage.source_ref_ids if item != ref)))
    elif mutation == "missing_cell_source":
        _mutate_provenance(case, source_ref_id="")
    elif mutation == "missing_row":
        _mutate_provenance(case, source_row_number="")
    else:
        _mutate_provenance(case, source_record_key="")
    result = replay_cell(ROOT, case.manifest, case.request)
    assert (result.status, result.reason_code) == (
        ReplayStatus.NOT_REPLAYABLE, "SOURCE_GROUP_EVIDENCE_MISSING"), result.message


@pytest.mark.parametrize("field,value", [
    ("rank", 9), ("status", STATUS_NOT_EVALUATED),
    ("role", "forged"), ("source_ref_id", "source:forged"),
])
def test_contribution_contradictions_fail(tmp_path, field, value):
    case = _bundle(tmp_path)
    _mutate_contributions(case, lambda items: items[0].__setitem__(field, value))
    result = replay_cell(ROOT, case.manifest, case.request)
    assert (result.status, result.reason_code) == (
        ReplayStatus.FAIL, "SOURCE_GROUP_SELECTION_MISMATCH"), result.message


def test_claimed_later_winner_cannot_choose_member(tmp_path):
    case = _bundle(tmp_path, participant="DEVICE-BOTH")
    def forge(items):
        items[0]["status"] = "unavailable"
        items[1]["status"] = "selected"
    _mutate_contributions(case, forge)
    result = replay_cell(ROOT, case.manifest, case.request)
    assert (result.status, result.reason_code) == (
        ReplayStatus.FAIL, "SOURCE_GROUP_SELECTION_MISMATCH")


def test_wrong_scalar_row_and_raw_fail_precisely(tmp_path):
    row_case = _bundle(tmp_path / "row")
    _mutate_provenance(row_case, source_row_number="1")
    row_result = replay_cell(ROOT, row_case.manifest, row_case.request)
    assert (row_result.status, row_result.reason_code) == (
        ReplayStatus.FAIL, "SOURCE_GROUP_SELECTION_MISMATCH")

    raw_case = _bundle(tmp_path / "raw")
    _mutate_provenance(
        raw_case, raw_value_json='{"kind":"string","value":"forged"}')
    raw_result = replay_cell(ROOT, raw_case.manifest, raw_case.request)
    assert (raw_result.status, raw_result.reason_code) == (
        ReplayStatus.FAIL, "RAW_VALUE_MISMATCH")


def test_changed_source_bytes_fail_hash_verification(tmp_path):
    case = _bundle(tmp_path, participant="DEVICE-B")
    path = case.source_paths["deciding_alpha"]
    path.write_text(path.read_text() + "DEVICE-X,x\n")
    result = replay_cell(ROOT, case.manifest, case.request)
    assert (result.status, result.reason_code) == (
        ReplayStatus.FAIL, "SOURCE_HASH_MISMATCH")


def test_unrelated_runwide_binding_cannot_repair_usage_scope(tmp_path):
    case = _bundle(tmp_path, participant="DEVICE-B")
    usage = _usage(case)
    omitted = _source(case, "deciding_alpha").source_ref_id
    assert any(source.source_ref_id == omitted for source in case.manifest.sources)
    _replace_usage(case, replace(
        usage, source_ref_ids=tuple(
            ref for ref in usage.source_ref_ids if ref != omitted)))
    result = replay_cell(ROOT, case.manifest, case.request)
    assert (result.status, result.reason_code) == (
        ReplayStatus.NOT_REPLAYABLE, "SOURCE_GROUP_EVIDENCE_MISSING")


def test_delivery_and_provenance_never_drive_selection(tmp_path):
    delivered = _bundle(tmp_path / "delivery", participant="DEVICE-BOTH")
    _mutate_delivered(delivered, "forged")
    result = replay_cell(ROOT, delivered.manifest, delivered.request)
    assert (result.status, result.reason_code, result.recomputed_value) == (
        ReplayStatus.FAIL, "PROVENANCE_VALUE_MISMATCH", "first-wins")

    provenance = _bundle(tmp_path / "provenance", participant="DEVICE-BOTH")
    _mutate_provenance(provenance, value="forged")
    result = replay_cell(ROOT, provenance.manifest, provenance.request)
    assert (result.status, result.reason_code, result.recomputed_value) == (
        ReplayStatus.FAIL, "PROVENANCE_VALUE_MISMATCH", "first-wins")


def test_contract_version_and_reason_codes():
    assert REASON_CODES_VERSION == 10
    assert "SOURCE_GROUP_EVIDENCE_MISSING" in REASON_CODES[ReplayStatus.NOT_REPLAYABLE]
    assert "SOURCE_GROUP_SELECTION_MISMATCH" in REASON_CODES[ReplayStatus.FAIL]


def test_missing_deciding_source_binding_is_group_evidence_missing(tmp_path):
    case = _bundle(tmp_path, participant="DEVICE-B")
    omitted = _source(case, "deciding_alpha").source_ref_id
    case.manifest = replace(
        case.manifest,
        sources=tuple(source for source in case.manifest.sources
                      if source.source_ref_id != omitted),
        mapping_usages=tuple(
            replace(item, source_ref_ids=tuple(
                ref for ref in item.source_ref_ids if ref != omitted))
            for item in case.manifest.mapping_usages))
    result = replay_cell(ROOT, case.manifest, case.request)
    assert (result.status, result.reason_code) == (
        ReplayStatus.NOT_REPLAYABLE, "SOURCE_GROUP_EVIDENCE_MISSING"), result.message


def test_publication_refuses_incomplete_group_evidence(tmp_path):
    def strip(cell):
        if cell.mapping_id == "group.direct.none":
            return replace(cell, contributions="")
        return cell
    with pytest.raises(ReplayOrchestrationError):
        _bundle(tmp_path, cell_mutator=strip)


def test_closed_boundary_rejects_neighboring_mechanisms(tmp_path):
    case = _bundle(tmp_path)
    mapping = _mapping(case)
    assert is_closed_source_group_mapping(case.spec, mapping)
    assert not is_closed_source_group_mapping(
        case.spec, replace(mapping, when="unsupported-condition"))
    assert not is_closed_source_group_mapping(
        case.spec, replace(mapping, derivation=(object(),)))
    assert is_closed_source_group_mapping(
        case.spec, replace(mapping, value_state="provisional"))
    assert not is_closed_source_group_mapping(
        case.spec, replace(mapping, value_state="suppressed"))


def test_file_membership_recomputation_is_pure():
    signature = inspect.signature(recompute_file_membership)
    assert set(signature.parameters) == {"selected_member_id", "member_values"}
    assert recompute_file_membership(
        "beta", member_values={"alpha": "A", "beta": "B"}) == "B"
    with pytest.raises(ValueError):
        recompute_file_membership(
            "missing", member_values={"alpha": "A", "beta": "B"})


def test_group_vocabulary_requires_usage_scoped_evidence(tmp_path):
    case = _bundle(tmp_path, target="vocabulary_label")
    usage = _usage(case)
    assert usage.evidence_ref_ids
    _replace_usage(case, replace(usage, evidence_ref_ids=()))
    result = replay_cell(ROOT, case.manifest, case.request)
    assert (result.status, result.reason_code) == (
        ReplayStatus.NOT_REPLAYABLE, "VOCABULARY_EVIDENCE_MISSING"), result.message

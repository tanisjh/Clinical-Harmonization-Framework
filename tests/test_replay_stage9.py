"""Synthetic-only tests for grouping-anchor no-mapping replay."""
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
from lib.replay import (
    ReplayRequest,
    recompute_no_mapping_configured,
    replay_cell,
)
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
from lib.spec_model import load as load_spec


ROOT = Path(__file__).resolve().parents[1]
FIXTURE = ROOT / "tests/fixtures/synthetic_replay"
SCHEMA = FIXTURE / "schema_stage9.yaml"
SPEC = FIXTURE / "spec_stage9.yaml"
GROUP_SOURCE = FIXTURE / "sources/stage9_grouping.csv"
PARTICIPANT_SOURCE = FIXTURE / "sources/stage9_participants.csv"
RUN_ID = "synthetic-stage9-run-0001"
ARTIFACT_ID = "synthetic-stage9-delivered"


@dataclass
class Case:
    manifest: object
    request: ReplayRequest
    spec: object
    cells: list
    group_path: Path
    participant_path: Path
    delivered_path: Path
    provenance_path: Path


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
            target="unmapped_blank", cell_mutator=None) -> Case:
    case_dir = tmp_path / f"{participant}-{target}"
    case_dir.mkdir(parents=True)
    group_path = case_dir / "grouping.csv"
    participant_path = case_dir / "participants.csv"
    group_path.write_bytes(GROUP_SOURCE.read_bytes())
    participant_path.write_bytes(PARTICIPANT_SOURCE.read_bytes())
    document = yaml.safe_load(SPEC.read_text())
    document["sources"][0]["path"] = _relative(group_path)
    document["sources"][1]["path"] = _relative(participant_path)
    spec_path = case_dir / "spec.yaml"
    spec_path.write_text(yaml.safe_dump(document, sort_keys=False))

    schema = load_schema(SCHEMA)
    spec = load_spec(spec_path, target_schema=schema, require_executable=True)
    context = ReplayRunContext.create(
        ROOT, schema, (spec,), run_id=RUN_ID,
        application="synthetic-stage9-test")
    cells, diagnostics = execute(
        spec, ROOT, contract=None, run_id=RUN_ID)
    if cell_mutator is not None:
        cells = [cell_mutator(cell) for cell in cells]
    collector = ReplayEvidenceCollector(context)
    collector.collect(spec, diagnostics.sources, cells)
    by_key = {}
    for cell in cells:
        key = (cell.cimac_part_id, cell.Cimac_id, cell.Collection_Event)
        row = by_key.setdefault(key, {name: None for name in schema.column_order})
        row.update({"cimac_part_id": key[0], "Cimac.id": key[1],
                    "Collection_Event": key[2],
                    cell.harmonized_field: cell.value})
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
        (delivered["cimac_part_id"] == participant)].iloc[0]
    request = ReplayRequest(
        artifact_id=ARTIFACT_ID,
        row_key={name: selected[name] for name in schema.row_key},
        target_field=target)
    return Case(
        published.manifest, request, spec, cells, group_path,
        participant_path, published.delivered_path,
        published.provenance_path)


def _mapping(case: Case):
    return next(item for item in case.spec.mappings
                if item.target == case.request.target_field)


def _usage(case: Case):
    mapping = _mapping(case)
    return next(item for item in case.manifest.mapping_usages
                if item.spec_ref == case.spec.ref
                and item.mapping_id == mapping.mapping_id)


def _cell(case: Case):
    mapping = _mapping(case)
    return next(item for item in case.cells
                if item.mapping_id == mapping.mapping_id
                and item.cimac_part_id == case.request.row_key["cimac_part_id"]
                and item.Cimac_id == case.request.row_key["Cimac.id"])


def _anchor_binding(case: Case):
    return next(item for item in case.manifest.evidence
                if item.kind == "anchor")


def _mutate_provenance(case: Case, **updates) -> None:
    fields, rows = _read_rows(case.provenance_path)
    mapping = _mapping(case)
    row = next(item for item in rows
               if item["mapping_id"] == mapping.mapping_id
               and item["cimac_part_id"] == case.request.row_key["cimac_part_id"]
               and item["Cimac.id"] == case.request.row_key["Cimac.id"])
    row.update(updates)
    _write_rows(case.provenance_path, fields, rows)
    case.manifest = replace(
        case.manifest,
        provenance_artifact=replace(
            case.manifest.provenance_artifact,
            sha256=sha256_file(case.provenance_path)))


def _replace_usage(case: Case, usage) -> None:
    old = _usage(case)
    case.manifest = replace(
        case.manifest,
        mapping_usages=tuple(
            usage if item == old else item
            for item in case.manifest.mapping_usages))


@pytest.mark.parametrize("participant,route,raw", [
    ("DEVICE-A", "grouped", "bench-west"),
    ("DEVICE-C", "synthesized", "unmeasured"),
])
@pytest.mark.parametrize("target,state", [
    ("unmapped_blank", ""),
    ("unmapped_source_missing", "source_missing"),
])
def test_grouped_and_synthesized_routes_pass(
        tmp_path, participant, route, raw, target, state):
    case = _bundle(tmp_path, participant=participant, target=target)
    cell = _cell(case)
    contribution = json.loads(cell.contributions)[0]
    result = replay_cell(ROOT, case.manifest, case.request)
    assert (result.status, result.reason_code, result.recomputed_value) == (
        ReplayStatus.PASS, "REPLAY_MATCH", None), result.message
    assert (contribution["anchor_record_kind"], contribution["raw_value"],
            cell.value_state) == (route, raw, state)
    if route == "grouped":
        assert contribution["source_row_number"] == 0
        assert "anchor_record_origin" not in contribution
    else:
        assert contribution["source_row_number"] == ""
        assert contribution["anchor_record_origin"] == "missing_reading"


def test_runtime_anchor_binding_is_deterministic_and_usage_scoped(tmp_path):
    first = _bundle(tmp_path / "a")
    second = _bundle(tmp_path / "b")
    one = _anchor_binding(first)
    two = _anchor_binding(second)
    assert one.evidence_ref_id.startswith("anchor-")
    assert len(one.evidence_ref_id) == len("anchor-") + 64
    # Paths change SourceRefs, and therefore exact runtime construction identity.
    assert one.evidence_ref_id != two.evidence_ref_id
    cell = _cell(first)
    usage = _usage(first)
    assert cell.evidence_refs == one.evidence_ref_id
    assert usage.evidence_ref_ids == (one.evidence_ref_id,)
    direct = next(item for item in first.manifest.mapping_usages
                  if item.mapping_id == "anchor.direct.unsupported")
    assert direct.evidence_ref_ids == (one.evidence_ref_id,)


def test_first_row_selection_and_projection_are_reconstructed(tmp_path):
    case = _bundle(tmp_path)
    contribution = json.loads(_cell(case).contributions)[0]
    assert contribution["source_row_number"] == 0
    assert contribution["source_record_key"] == (
        "cimac_part_id=DEVICE-A;Cimac.id=READ-001;Collection_Event=bench-west")
    assert replay_cell(ROOT, case.manifest, case.request).status is ReplayStatus.PASS


@pytest.mark.parametrize("missing", ["cell", "usage", "binding"])
def test_missing_anchor_evidence_is_not_replayable(tmp_path, missing):
    case = _bundle(tmp_path)
    if missing == "cell":
        _mutate_provenance(case, evidence_refs="")
    elif missing == "usage":
        _replace_usage(case, replace(_usage(case), evidence_ref_ids=()))
    else:
        document = case.manifest.to_dict()
        document["evidence"] = [
            item for item in document["evidence"]
            if item["kind"] != "anchor"]
        manifest_path = tmp_path / "missing-anchor-manifest.json"
        manifest_path.write_text(json.dumps(document))
        case.manifest = manifest_path
    result = replay_cell(ROOT, case.manifest, case.request)
    assert (result.status, result.reason_code) == (
        ReplayStatus.NOT_REPLAYABLE, "ANCHOR_EVIDENCE_MISSING")


def test_conflicting_anchor_identity_fails(tmp_path):
    case = _bundle(tmp_path)
    binding = _anchor_binding(case)
    bad = replace(binding, reference="synthetic/wrong@1#record.anchor")
    case.manifest = replace(
        case.manifest,
        evidence=tuple(bad if item == binding else item
                       for item in case.manifest.evidence))
    result = replay_cell(ROOT, case.manifest, case.request)
    assert (result.status, result.reason_code) == (
        ReplayStatus.FAIL, "ANCHOR_IDENTITY_MISMATCH")


@pytest.mark.parametrize("participant", ["DEVICE-A", "DEVICE-C"])
def test_source_hash_drift_fails(tmp_path, participant):
    case = _bundle(tmp_path, participant=participant)
    path = case.group_path if participant == "DEVICE-A" else case.participant_path
    path.write_bytes(path.read_bytes() + b"\n")
    result = replay_cell(ROOT, case.manifest, case.request)
    assert (result.status, result.reason_code) == (
        ReplayStatus.FAIL, "SOURCE_HASH_MISMATCH")


@pytest.mark.parametrize("update,reason", [
    ({"rule_id": "wrong.rule"}, "RULE_IDENTITY_MISMATCH"),
    ({"confidence": "0.99"}, "CONFIDENCE_CONTEXT_MISMATCH"),
    ({"value_state": "not_applicable"}, "VALUE_STATE_CONTEXT_MISMATCH"),
    ({"source_file": "WRONG"}, "SOURCE_DECLARATION_MISMATCH"),
    ({"source_row_number": "0"}, "SOURCE_DECLARATION_MISMATCH"),
])
def test_context_contradictions_fail(tmp_path, update, reason):
    case = _bundle(tmp_path)
    _mutate_provenance(case, **update)
    result = replay_cell(ROOT, case.manifest, case.request)
    assert (result.status, result.reason_code) == (ReplayStatus.FAIL, reason)


def test_wrong_selected_member_and_scalar_fail(tmp_path):
    case = _bundle(tmp_path)
    contribution = json.loads(_cell(case).contributions)
    contribution[0]["source_row_number"] = 1
    _mutate_provenance(
        case, contributions=json.dumps(
            contribution, separators=(",", ":"), sort_keys=True))
    result = replay_cell(ROOT, case.manifest, case.request)
    assert (result.status, result.reason_code) == (
        ReplayStatus.FAIL, "ANCHOR_SELECTION_MISMATCH")

    case = _bundle(tmp_path / "raw")
    _mutate_provenance(case, raw_value_json='{"kind":"string","value":"east"}')
    result = replay_cell(ROOT, case.manifest, case.request)
    assert (result.status, result.reason_code) == (
        ReplayStatus.FAIL, "RAW_VALUE_MISMATCH")


def test_pure_recomputation_has_no_value_inputs():
    signature = inspect.signature(recompute_no_mapping_configured)
    assert tuple(signature.parameters) == ()
    outcome = recompute_no_mapping_configured()
    assert (outcome.value, outcome.branch) == (
        None, "no_mapping_configured")


def test_delivery_and_provenance_output_cannot_drive_recomputation(tmp_path):
    case = _bundle(tmp_path)
    _mutate_provenance(case, value="forged")
    result = replay_cell(ROOT, case.manifest, case.request)
    assert (result.status, result.reason_code) == (
        ReplayStatus.FAIL, "PROVENANCE_VALUE_MISMATCH")
    assert recompute_no_mapping_configured().value is None


def test_grouping_anchor_direct_is_replayable_after_stage10(tmp_path):
    case = _bundle(tmp_path, target="anchor_direct_unsupported")
    result = replay_cell(ROOT, case.manifest, case.request)
    assert (result.status, result.reason_code, result.recomputed_value) == (
        ReplayStatus.PASS, "REPLAY_MATCH", "bench-west")


def test_grouping_anchor_value_map_remains_unsupported(tmp_path):
    case = _bundle(tmp_path, target="anchor_map_unsupported")
    result = replay_cell(ROOT, case.manifest, case.request)
    assert (result.status, result.reason_code) == (
        ReplayStatus.NOT_REPLAYABLE, "SOURCE_NOT_SINGLE_FILE")


def test_publication_rejects_missing_anchor_binding_link(tmp_path):
    def mutate(cell):
        if cell.mapping_id == "no_mapping.blank":
            return replace(cell, evidence_refs="")
        return cell

    with pytest.raises(ReplayOrchestrationError, match="anchor"):
        _bundle(tmp_path, cell_mutator=mutate)


def test_publication_rejects_incomplete_anchor_scalar(tmp_path):
    def mutate(cell):
        if cell.mapping_id != "no_mapping.blank":
            return cell
        contribution = json.loads(cell.contributions)
        contribution[0].pop("raw_value_json")
        return replace(
            cell,
            contributions=json.dumps(
                contribution, separators=(",", ":"), sort_keys=True))

    with pytest.raises(ReplayOrchestrationError, match="scalar evidence"):
        _bundle(tmp_path, cell_mutator=mutate)


def test_anchor_contribution_route_vocabulary_is_closed():
    from lib.contributions import Contribution
    with pytest.raises(ValueError, match="anchor_record_kind"):
        Contribution(
            role="anchor_input", status="input", source_column="field",
            anchor_record_kind="rediscovered")


def test_pandas_environment_mismatch_blocks_replay(tmp_path):
    case = _bundle(tmp_path)
    dependencies = dict(case.manifest.producer.dependencies)
    dependencies["pandas"] = "0.0-stage9-wrong"
    case.manifest = replace(
        case.manifest,
        producer=replace(
            case.manifest.producer,
            dependencies=tuple(sorted(dependencies.items()))))
    result = replay_cell(ROOT, case.manifest, case.request)
    assert (result.status, result.reason_code) == (
        ReplayStatus.NOT_REPLAYABLE, "PRODUCER_ENVIRONMENT_MISMATCH")

def test_contract_versions_and_reason_codes():
    from lib.provenance import PROVENANCE_SCHEMA_VERSION
    from lib.replay_contract import REPLAY_API_VERSION, REPLAY_STATUS_VERSION
    from lib.replay_manifest import MANIFEST_VERSION
    assert (REPLAY_API_VERSION, REPLAY_STATUS_VERSION, MANIFEST_VERSION,
            PROVENANCE_SCHEMA_VERSION, REASON_CODES_VERSION) == (1, 1, 2, 1, 10)

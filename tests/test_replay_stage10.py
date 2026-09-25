"""Synthetic-only tests for grouping-anchor direct replay."""
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
from lib.replay import ReplayRequest, recompute_direct, replay_cell
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
SCHEMA = FIXTURE / "schema_stage10.yaml"
SPEC = FIXTURE / "spec_stage10.yaml"
GROUP_SOURCE = FIXTURE / "sources/stage10_grouping.csv"
PARTICIPANT_SOURCE = FIXTURE / "sources/stage10_participants.csv"
ATTRIBUTION_SOURCE = FIXTURE / "sources/stage10_attribution.csv"
RUN_ID = "synthetic-stage10-run-0001"
ARTIFACT_ID = "synthetic-stage10-delivered"


@dataclass
class Case:
    manifest: object
    request: ReplayRequest
    spec: object
    cells: list
    group_path: Path
    participant_path: Path
    attribution_path: Path
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


def _bundle(tmp_path: Path, *, participant="DEVICE-A", target="direct_none",
            cell_mutator=None, attribution_marker="FORGED",
            anchor_kind="grouping", conditional=False) -> Case:
    case_dir = tmp_path / f"{participant}-{target}"
    case_dir.mkdir(parents=True)
    group_path = case_dir / "grouping.csv"
    participant_path = case_dir / "participants.csv"
    attribution_path = case_dir / "attribution.csv"
    group_path.write_bytes(GROUP_SOURCE.read_bytes())
    participant_path.write_bytes(PARTICIPANT_SOURCE.read_bytes())
    attribution_path.write_text(
        ATTRIBUTION_SOURCE.read_text().replace("FORGED", attribution_marker))

    document = yaml.safe_load(SPEC.read_text())
    document["sources"][0]["path"] = _relative(group_path)
    document["sources"][1]["path"] = _relative(participant_path)
    document["sources"][2]["path"] = _relative(attribution_path)
    anchor_df = None
    if conditional:
        document["reads"] = [{
            "id": "attribution_marker",
            "source": "attribution",
            "column": "attribution_only_value",
        }]
        document["conditions"] = [{
            "id": "has_attribution",
            "read": "attribution_marker",
            "predicate": "not_null",
        }]
        conditional_mapping = next(
            item for item in document["mappings"]
            if item["mapping_id"] == "anchor.direct.none")
        conditional_mapping["when"] = "has_attribution"
    if anchor_kind == "external":
        document["record"] = {
            "anchor": "external",
            "participant_key": "cimac_part_id",
            "sample_key": "Cimac.id",
            "collection_event_key": "Collection_Event",
            "duplicates": {"policy": "error",
                           "by": ["cimac_part_id"]},
        }
        anchor_df = pd.DataFrame([{
            "cimac_part_id": participant,
            "Cimac.id": "EXTERNAL-001",
            "Collection_Event": "bench-east",
        }])
    spec_path = case_dir / "spec.yaml"
    spec_path.write_text(yaml.safe_dump(document, sort_keys=False))

    schema = load_schema(SCHEMA)
    spec = load_spec(spec_path, target_schema=schema, require_executable=True)
    context = ReplayRunContext.create(
        ROOT, schema, (spec,), run_id=RUN_ID,
        application="synthetic-stage10-test")
    cells, diagnostics = execute(
        spec, ROOT, anchor_df=anchor_df, contract=None, run_id=RUN_ID)
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
        delivered["cimac_part_id"] == participant].iloc[0]
    request = ReplayRequest(
        artifact_id=ARTIFACT_ID,
        row_key={name: selected[name] for name in schema.row_key},
        target_field=target)
    return Case(
        published.manifest, request, spec, cells, group_path,
        participant_path, attribution_path, published.delivered_path,
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


def _source(case: Case, source_id: str):
    return next(item for item in case.manifest.sources
                if item.spec_ref == case.spec.ref and item.source_id == source_id)


def _anchor_binding(case: Case):
    return next(item for item in case.manifest.evidence if item.kind == "anchor")


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


def _mutate_delivered(case: Case, **updates) -> None:
    fields, rows = _read_rows(case.delivered_path)
    row = next(item for item in rows
               if item["cimac_part_id"] == case.request.row_key["cimac_part_id"]
               and item["Cimac.id"] == case.request.row_key["Cimac.id"])
    row.update(updates)
    _write_rows(case.delivered_path, fields, rows)
    artifact = case.manifest.delivered_artifacts[0]
    case.manifest = replace(
        case.manifest,
        delivered_artifacts=(replace(
            artifact, sha256=sha256_file(case.delivered_path)),))


def _replace_usage(case: Case, usage) -> None:
    old = _usage(case)
    case.manifest = replace(
        case.manifest,
        mapping_usages=tuple(
            usage if item == old else item
            for item in case.manifest.mapping_usages))


@pytest.mark.parametrize("participant,route,target,expected,raw", [
    ("DEVICE-A", "grouped", "direct_none", "  bench   west  ",
     "  bench   west  "),
    ("DEVICE-A", "grouped", "direct_text", "bench west",
     "  bench   west  "),
    ("DEVICE-N", "grouped", "direct_numeric", 42.0, " 42 "),
    ("DEVICE-C", "synthesized", "direct_none", " 7.5 ", " 7.5 "),
    ("DEVICE-C", "synthesized", "direct_text", "7.5", " 7.5 "),
    ("DEVICE-C", "synthesized", "direct_numeric", 7.5, " 7.5 "),
])
def test_grouped_and_synthesized_direct_coercions_pass(
        tmp_path, participant, route, target, expected, raw):
    case = _bundle(tmp_path, participant=participant, target=target)
    cell = _cell(case)
    contribution = json.loads(cell.contributions)[0]
    result = replay_cell(ROOT, case.manifest, case.request)
    assert (result.status, result.reason_code, result.recomputed_value) == (
        ReplayStatus.PASS, "REPLAY_MATCH", expected), result.message
    assert contribution["anchor_record_kind"] == route
    assert contribution["raw_value"] == raw
    if route == "grouped":
        assert isinstance(contribution["source_row_number"], int)
        assert "anchor_record_origin" not in contribution
    else:
        assert contribution["source_row_number"] == ""
        assert contribution["anchor_record_origin"] == "missing_reading"
    assert cell.source_ref_id == _source(case, "attribution").source_ref_id
    assert cell.source_row_number in ("", None)


def test_unattributed_grouping_direct_passes_with_anchor_origin(tmp_path):
    case = _bundle(tmp_path, target="direct_unattributed")
    cell = _cell(case)
    result = replay_cell(ROOT, case.manifest, case.request)
    assert (result.status, result.recomputed_value) == (
        ReplayStatus.PASS, "  bench   west  ")
    assert (cell.source_file, cell.source_ref_id) == ("ANCHOR", "")


def test_producer_and_usage_compose_independent_evidence_routes(tmp_path):
    case = _bundle(tmp_path)
    cell = _cell(case)
    usage = _usage(case)
    contribution = json.loads(cell.contributions)
    anchor = _anchor_binding(case)
    assert cell.evidence_refs == anchor.evidence_ref_id
    assert len(contribution) == 1
    assert contribution[0]["role"] == "anchor_input"
    assert contribution[0]["source_ref_id"] == _source(case, "readings").source_ref_id
    assert set(usage.source_ref_ids) == {
        _source(case, "readings").source_ref_id,
        _source(case, "participants").source_ref_id,
        _source(case, "attribution").source_ref_id,
    }


@pytest.mark.parametrize("missing", ["anchor_ref", "contribution", "attribution"])
def test_publication_rejects_incomplete_direct_evidence(tmp_path, missing):
    def mutate(cell):
        if cell.mapping_id != "anchor.direct.none":
            return cell
        if missing == "anchor_ref":
            return replace(cell, evidence_refs="")
        if missing == "contribution":
            return replace(cell, contributions="")
        return replace(cell, source_ref_id="")

    with pytest.raises(ReplayOrchestrationError, match="anchor|attributed"):
        _bundle(tmp_path, cell_mutator=mutate)


def test_existing_direct_recomputation_is_the_only_operation_path(
        tmp_path, monkeypatch):
    case = _bundle(tmp_path, target="direct_text")
    calls = []
    original = replay_module.recompute_direct

    def spy(raw_source_value, *, coerce):
        calls.append((raw_source_value, coerce))
        return original(raw_source_value, coerce=coerce)

    monkeypatch.setattr(replay_module, "recompute_direct", spy)
    assert replay_cell(ROOT, case.manifest, case.request).status is ReplayStatus.PASS
    assert calls == [("  bench   west  ", "text")]
    assert tuple(inspect.signature(recompute_direct).parameters) == (
        "raw_source_value", "coerce")


@pytest.mark.parametrize("missing", ["cell", "contribution", "usage", "binding"])
def test_missing_anchor_evidence_is_not_replayable(tmp_path, missing):
    case = _bundle(tmp_path)
    if missing == "cell":
        _mutate_provenance(case, evidence_refs="")
    elif missing == "contribution":
        _mutate_provenance(case, contributions="")
    elif missing == "usage":
        _replace_usage(case, replace(_usage(case), evidence_ref_ids=()))
    else:
        document = case.manifest.to_dict()
        document["evidence"] = [
            item for item in document["evidence"] if item["kind"] != "anchor"]
        manifest_path = tmp_path / "missing-anchor-manifest.json"
        manifest_path.write_text(json.dumps(document))
        case.manifest = manifest_path
    result = replay_cell(ROOT, case.manifest, case.request)
    assert (result.status, result.reason_code) == (
        ReplayStatus.NOT_REPLAYABLE, "ANCHOR_EVIDENCE_MISSING")


def test_anchor_identity_selection_and_scalar_contradictions_fail(tmp_path):
    case = _bundle(tmp_path / "identity")
    anchor = _anchor_binding(case)
    case.manifest = replace(
        case.manifest,
        evidence=tuple(
            replace(item, reference="synthetic/wrong@1#record.anchor")
            if item == anchor else item for item in case.manifest.evidence))
    result = replay_cell(ROOT, case.manifest, case.request)
    assert (result.status, result.reason_code) == (
        ReplayStatus.FAIL, "ANCHOR_IDENTITY_MISMATCH")

    case = _bundle(tmp_path / "selection")
    contribution = json.loads(_cell(case).contributions)
    contribution[0]["source_row_number"] = 1
    _mutate_provenance(
        case, contributions=json.dumps(
            contribution, separators=(",", ":"), sort_keys=True))
    result = replay_cell(ROOT, case.manifest, case.request)
    assert (result.status, result.reason_code) == (
        ReplayStatus.FAIL, "ANCHOR_SELECTION_MISMATCH")

    case = _bundle(tmp_path / "scalar")
    contribution = json.loads(_cell(case).contributions)
    contribution[0]["raw_value"] = "FORGED"
    contribution[0]["raw_value_json"] = '{"kind":"string","value":"FORGED"}'
    _mutate_provenance(
        case, contributions=json.dumps(
            contribution, separators=(",", ":"), sort_keys=True))
    result = replay_cell(ROOT, case.manifest, case.request)
    assert (result.status, result.reason_code) == (
        ReplayStatus.FAIL, "RAW_VALUE_MISMATCH")


@pytest.mark.parametrize("missing", ["cell", "usage", "binding"])
def test_missing_attributed_source_link_is_not_replayable(tmp_path, missing):
    case = _bundle(tmp_path)
    attribution = _source(case, "attribution")
    if missing == "cell":
        _mutate_provenance(case, source_ref_id="")
    elif missing == "usage":
        usage = _usage(case)
        _replace_usage(case, replace(
            usage,
            source_ref_ids=tuple(
                item for item in usage.source_ref_ids
                if item != attribution.source_ref_id)))
    else:
        document = case.manifest.to_dict()
        document["sources"] = [
            item for item in document["sources"]
            if item["source_ref_id"] != attribution.source_ref_id]
        manifest_path = tmp_path / "missing-source-manifest.json"
        manifest_path.write_text(json.dumps(document))
        case.manifest = manifest_path
    result = replay_cell(ROOT, case.manifest, case.request)
    assert (result.status, result.reason_code) == (
        ReplayStatus.NOT_REPLAYABLE, "SOURCE_LINK_MISSING")


def test_attributed_source_contradictions_fail_independently(tmp_path):
    case = _bundle(tmp_path / "declaration")
    attribution = _source(case, "attribution")
    case.manifest = replace(
        case.manifest,
        sources=tuple(
            replace(item, logical_label="wrong.csv")
            if item == attribution else item for item in case.manifest.sources))
    result = replay_cell(ROOT, case.manifest, case.request)
    assert (result.status, result.reason_code) == (
        ReplayStatus.FAIL, "SOURCE_DECLARATION_MISMATCH")

    case = _bundle(tmp_path / "bytes")
    case.attribution_path.write_bytes(case.attribution_path.read_bytes() + b"\n")
    result = replay_cell(ROOT, case.manifest, case.request)
    assert (result.status, result.reason_code) == (
        ReplayStatus.FAIL, "SOURCE_HASH_MISMATCH")

    case = _bundle(tmp_path / "reader")
    attribution = _source(case, "attribution")
    bad_reader = replace(
        attribution.reader,
        effective_flags=tuple(
            (name, "latin-1" if name == "encoding" else value)
            for name, value in attribution.reader.effective_flags))
    case.manifest = replace(
        case.manifest,
        sources=tuple(
            replace(item, reader=bad_reader)
            if item == attribution else item for item in case.manifest.sources))
    result = replay_cell(ROOT, case.manifest, case.request)
    assert result.status is ReplayStatus.FAIL
    assert result.reason_code in {
        "READER_OBSERVATION_MISMATCH", "SOURCE_DECLARATION_MISMATCH"}


def test_attributed_file_never_supplies_semantic_scalar(tmp_path):
    first = _bundle(tmp_path / "a", attribution_marker="FIRST-FORGED")
    second = _bundle(tmp_path / "b", attribution_marker="SECOND-FORGED")
    one = replay_cell(ROOT, first.manifest, first.request)
    two = replay_cell(ROOT, second.manifest, second.request)
    assert (one.status, two.status) == (ReplayStatus.PASS, ReplayStatus.PASS)
    assert one.recomputed_value == two.recomputed_value == "  bench   west  "


def test_recorded_outputs_cannot_drive_recomputation(tmp_path):
    case = _bundle(tmp_path / "provenance")
    _mutate_provenance(case, value="FORGED")
    result = replay_cell(ROOT, case.manifest, case.request)
    assert (result.status, result.reason_code, result.recomputed_value) == (
        ReplayStatus.FAIL, "PROVENANCE_VALUE_MISMATCH", "  bench   west  ")

    case = _bundle(tmp_path / "delivery")
    _mutate_delivered(case, direct_none="FORGED")
    result = replay_cell(ROOT, case.manifest, case.request)
    assert (result.status, result.reason_code, result.recomputed_value) == (
        ReplayStatus.FAIL, "PROVENANCE_VALUE_MISMATCH", "  bench   west  ")
    assert recompute_direct("  bench   west  ", coerce="none") == "  bench   west  "


def test_pre_adapter_record_remains_not_replayable(tmp_path):
    case = _bundle(tmp_path)
    _mutate_provenance(case, evidence_refs="", contributions="")
    _replace_usage(case, replace(_usage(case), evidence_ref_ids=()))
    result = replay_cell(ROOT, case.manifest, case.request)
    assert (result.status, result.reason_code) == (
        ReplayStatus.NOT_REPLAYABLE, "ANCHOR_EVIDENCE_MISSING")


def test_condition_bearing_grouping_direct_remains_unsupported(tmp_path):
    case = _bundle(tmp_path, conditional=True)
    result = replay_cell(ROOT, case.manifest, case.request)
    assert (result.status, result.reason_code) == (
        ReplayStatus.NOT_REPLAYABLE, "NAMED_READ_EVIDENCE_MISSING")


@pytest.mark.parametrize("method,state,reason", [
    ("value_suppressed", "suppressed", "SUPPRESSION_UNSUPPORTED_STAGE1"),
    ("row_excluded", "row_excluded", "EXCLUSION_REQUIRES_ROW_REPLAY"),
    ("post:synthetic", "", "POST_RULE_UNSUPPORTED_STAGE1"),
])
def test_authority_overrides_remain_unsupported(
        tmp_path, method, state, reason):
    case = _bundle(tmp_path)
    _mutate_provenance(
        case, extraction_method=method, value_state=state)
    result = replay_cell(ROOT, case.manifest, case.request)
    assert (result.status, result.reason_code) == (
        ReplayStatus.NOT_REPLAYABLE, reason)


@pytest.mark.parametrize("target", ["direct_none", "anchor_map_unsupported"])
def test_unbound_external_anchor_cannot_publish_replay_evidence(tmp_path, target):
    with pytest.raises(ReplayOrchestrationError, match="runtime anchor binding"):
        _bundle(tmp_path, target=target, anchor_kind="external")


def test_grouping_value_map_remains_unsupported(tmp_path):
    case = _bundle(tmp_path, target="anchor_map_unsupported")
    result = replay_cell(ROOT, case.manifest, case.request)
    assert (result.status, result.reason_code) == (
        ReplayStatus.NOT_REPLAYABLE, "SOURCE_NOT_SINGLE_FILE")


def test_contract_versions_unchanged():
    from lib.provenance import PROVENANCE_SCHEMA_VERSION
    from lib.replay_contract import REPLAY_API_VERSION, REPLAY_STATUS_VERSION
    from lib.replay_manifest import MANIFEST_VERSION
    assert (REPLAY_API_VERSION, REPLAY_STATUS_VERSION, MANIFEST_VERSION,
            PROVENANCE_SCHEMA_VERSION, REASON_CODES_VERSION) == (1, 1, 2, 1, 10)

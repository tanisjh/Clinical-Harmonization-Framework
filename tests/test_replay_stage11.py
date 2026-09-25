"""Synthetic-only tests for trusted external-anchor strict replay."""
from __future__ import annotations

import csv
import inspect
import json
from dataclasses import dataclass, replace
from decimal import Decimal
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

import harmonize_spec
import lib.replay as replay_module
from lib.anchor_evidence import (
    EXTERNAL_ANCHOR_FRAME_VERSION,
    EXTERNAL_ANCHOR_READER_FLAGS,
    EXTERNAL_ANCHOR_READER_ID,
    RuntimeExternalAnchor,
    canonical_external_anchor_frame,
    load_external_anchor,
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
from lib.spec_model import load as load_spec


ROOT = Path(__file__).resolve().parents[1]
FIXTURE = ROOT / "tests/fixtures/synthetic_replay"
SCHEMA = FIXTURE / "schema_stage11.yaml"
SPEC = FIXTURE / "spec_stage11.yaml"
ANCHOR_SOURCE = FIXTURE / "sources/stage11_external_anchor.csv"
ATTRIBUTION_SOURCE = FIXTURE / "sources/stage11_attribution.csv"
RUN_ID = "synthetic-stage11-run-0001"
ARTIFACT_ID = "synthetic-stage11-delivered"


@dataclass
class Case:
    manifest: object
    request: ReplayRequest
    spec: object
    cells: list
    anchor_path: Path
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


def _bundle(tmp_path: Path, *, reading="READ-001", target="direct_none",
            cell_mutator=None) -> Case:
    case_dir = tmp_path / f"{reading}-{target}"
    case_dir.mkdir(parents=True)
    anchor_path = case_dir / "external_anchor.csv"
    attribution_path = case_dir / "attribution.csv"
    anchor_path.write_bytes(ANCHOR_SOURCE.read_bytes())
    attribution_path.write_bytes(ATTRIBUTION_SOURCE.read_bytes())

    document = yaml.safe_load(SPEC.read_text())
    document["sources"][0]["path"] = _relative(attribution_path)
    spec_path = case_dir / "spec.yaml"
    spec_path.write_text(yaml.safe_dump(document, sort_keys=False))
    schema = load_schema(SCHEMA)
    spec = load_spec(spec_path, target_schema=schema, require_executable=True)
    runtime_anchor = load_external_anchor(
        anchor_path, project_root=ROOT, spec_ref=spec.ref,
        filter_column="project", selector="alpha")
    context = ReplayRunContext.create(
        ROOT, schema, (spec,), run_id=RUN_ID,
        application="synthetic-stage11-test")
    cells, diagnostics = execute(
        spec, ROOT, anchor_df=runtime_anchor, contract=None, run_id=RUN_ID)
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
    selected = delivered.loc[delivered["Cimac.id"] == reading].iloc[0]
    request = ReplayRequest(
        artifact_id=ARTIFACT_ID,
        row_key={name: selected[name] for name in schema.row_key},
        target_field=target)
    return Case(
        published.manifest, request, spec, cells, anchor_path,
        attribution_path, published.delivered_path,
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
               if item["Cimac.id"] == case.request.row_key["Cimac.id"])
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


@pytest.mark.parametrize("reading,target,expected,original,filtered", [
    ("READ-001", "direct_none", "plain", 1, 0),
    ("READ-001", "direct_text", "spaced text", 1, 0),
    ("READ-001", "direct_numeric", 12.5, 1, 0),
    ("READ-001", "direct_dual", "plain", 1, 0),
    ("READ-001", "status_label", "ready", 1, 0),
    ("READ-002", "status_label", None, 2, 1),
    ("READ-003", "status_label", None, 3, 2),
])
def test_external_direct_and_inline_map_pass(
        tmp_path, reading, target, expected, original, filtered):
    case = _bundle(tmp_path, reading=reading, target=target)
    contribution = json.loads(_cell(case).contributions)[0]
    result = replay_cell(ROOT, case.manifest, case.request)
    assert (result.status, result.reason_code, result.recomputed_value) == (
        ReplayStatus.PASS, "REPLAY_MATCH", expected), result.message
    assert contribution["anchor_record_kind"] == "external"
    assert contribution["source_row_number"] == original
    assert contribution["anchor_filtered_position"] == filtered
    assert contribution["source_ref_id"] == _source(case, "$anchor").source_ref_id


def test_existing_operation_helpers_are_reused(tmp_path, monkeypatch):
    direct = _bundle(tmp_path / "direct", target="direct_text")
    direct_calls = []
    direct_impl = replay_module.recompute_direct
    monkeypatch.setattr(
        replay_module, "recompute_direct",
        lambda raw, *, coerce: (
            direct_calls.append((raw, coerce)),
            direct_impl(raw, coerce=coerce))[1])
    assert replay_cell(ROOT, direct.manifest, direct.request).status is ReplayStatus.PASS
    assert direct_calls == [("  spaced   text  ", "text")]

    mapped = _bundle(tmp_path / "map", target="status_label")
    map_calls = []
    map_impl = replay_module.recompute_value_map
    monkeypatch.setattr(
        replay_module, "recompute_value_map",
        lambda raw, *, mapping_table, on_unmapped: (
            map_calls.append((raw, dict(mapping_table), on_unmapped)),
            map_impl(raw, mapping_table=mapping_table,
                     on_unmapped=on_unmapped))[1])
    assert replay_cell(ROOT, mapped.manifest, mapped.request).status is ReplayStatus.PASS
    assert map_calls == [("ACTIVE", {"ACTIVE": "ready", "INACTIVE": "stopped"},
                          "null")]
    assert tuple(inspect.signature(direct_impl).parameters) == (
        "raw_source_value", "coerce")


def test_exact_reader_and_construction_identity_are_bound(tmp_path):
    case = _bundle(tmp_path)
    source = _source(case, "$anchor")
    anchor = _anchor_binding(case)
    details = dict(anchor.details)
    assert source.reader.reader_id == EXTERNAL_ANCHOR_READER_ID
    assert source.reader.flags == EXTERNAL_ANCHOR_READER_FLAGS
    assert source.n_rows == 4
    assert details["pre_filter_row_count"] == 4
    assert details["post_filter_row_count"] == 3
    assert tuple(details["original_source_rows"]) == (1, 2, 3)
    assert tuple(details["filtered_positions"]) == (0, 1, 2)
    assert details["canonicalization_version"] == EXTERNAL_ANCHOR_FRAME_VERSION


def test_canonical_frame_v1_is_typed_ordered_and_index_free():
    frame = pd.DataFrame([{
        "zero": 0, "negative_zero": -0.0, "one_float": 1.0,
        "small": 1e-9, "large": 1.25e20, "numpy_int": np.int64(7),
        "numpy_float": np.float64(2.5), "boolean": True,
        "null": None, "unicode": "café ",
    }], index=[91])
    first = canonical_external_anchor_frame(frame)
    second = canonical_external_anchor_frame(frame.reset_index(drop=True))
    assert first == second
    doc = json.loads(first)
    assert doc["canonicalization_version"] == 1
    assert doc["columns"] == list(frame.columns)
    typed = dict(zip(doc["columns"], doc["rows"][0]))
    assert typed["zero"]["kind"] == "integer"
    assert typed["one_float"]["kind"] == "number"
    assert typed["negative_zero"] == {"kind": "number", "value": "-0.0"}
    assert typed["boolean"] == {"kind": "boolean", "value": True}
    assert typed["null"] == {"kind": "null"}
    assert b"NaN" not in first
    assert first != canonical_external_anchor_frame(frame[list(reversed(frame.columns))])
    ordered = pd.DataFrame({"value": ["first", "second"]})
    assert (canonical_external_anchor_frame(ordered)
            != canonical_external_anchor_frame(ordered.iloc[::-1]))
    with pytest.raises(ValueError):
        canonical_external_anchor_frame(pd.DataFrame({"bad": [float("inf")]}))
    with pytest.raises(ValueError):
        canonical_external_anchor_frame(pd.DataFrame({"bad": [Decimal("NaN")]}))


def test_source_order_and_row_order_affect_runtime_identity(tmp_path):
    first = load_external_anchor(
        ANCHOR_SOURCE, project_root=ROOT, spec_ref="synthetic/example@1",
        filter_column="project", selector="alpha")
    reordered_path = tmp_path / "reordered.csv"
    rows = ANCHOR_SOURCE.read_text().splitlines()
    reordered_path.write_text("\n".join([rows[0], rows[1], rows[3], rows[2], rows[4]]) + "\n")
    second = load_external_anchor(
        reordered_path, project_root=ROOT, spec_ref="synthetic/example@1",
        filter_column="project", selector="alpha")
    assert first.evidence.evidence_ref_id != second.evidence.evidence_ref_id
    assert first.original_source_rows == second.original_source_rows == (1, 2, 3)


def test_factory_hashes_and_parses_one_captured_byte_string(monkeypatch):
    calls = []
    original = Path.read_bytes

    def spy(path):
        if path.resolve() == ANCHOR_SOURCE.resolve():
            calls.append(path)
        return original(path)

    monkeypatch.setattr(Path, "read_bytes", spy)
    runtime = load_external_anchor(
        ANCHOR_SOURCE, project_root=ROOT, spec_ref="synthetic/example@1",
        filter_column="project", selector="alpha")
    assert calls == [ANCHOR_SOURCE.resolve()]
    assert runtime.source_ref.sha256 == sha256_file(ANCHOR_SOURCE)


def test_runtime_wrapper_detects_frame_mutation():
    runtime = load_external_anchor(
        ANCHOR_SOURCE, project_root=ROOT, spec_ref="synthetic/example@1",
        filter_column="project", selector="alpha")
    runtime._frame.loc[0, "raw_none"] = "FORGED"
    with pytest.raises(ValueError, match="no longer matches"):
        runtime.frame_copy(spec_ref="synthetic/example@1")


def test_runtime_wrapper_cannot_be_forged_from_descriptive_metadata():
    runtime = load_external_anchor(
        ANCHOR_SOURCE, project_root=ROOT, spec_ref="synthetic/example@1",
        filter_column="project", selector="alpha")
    with pytest.raises(ValueError, match="must be created"):
        RuntimeExternalAnchor(
            source_ref=runtime.source_ref,
            evidence=runtime.evidence,
            original_source_rows=runtime.original_source_rows,
            filtered_positions=runtime.filtered_positions,
            _frame=runtime.frame_copy(spec_ref="synthetic/example@1"),
            _factory_token=object())


def test_cli_replay_mode_uses_trusted_wrapper(tmp_path):
    anchor_path = tmp_path / "cli_anchor.csv"
    anchor_path.write_text(
        "trial,cimac_part_id,Cimac.id,Collection_Event,raw_none\n"
        "synthetic_device_lab,DEVICE-A,READ-001,bench-west,plain\n")
    schema = load_schema(SCHEMA)
    spec = load_spec(SPEC, target_schema=schema, require_executable=True)
    trusted = harmonize_spec._anchor_for(
        spec, anchor_path, replay_enabled=True)
    legacy = harmonize_spec._anchor_for(
        spec, anchor_path, replay_enabled=False)
    assert isinstance(trusted, RuntimeExternalAnchor)
    assert isinstance(legacy, pd.DataFrame)
    assert trusted.source_ref.reader_id == EXTERNAL_ANCHOR_READER_ID


def test_bare_external_dataframe_is_refused_for_replay_publication(tmp_path):
    case_dir = tmp_path / "bare"
    case_dir.mkdir()
    document = yaml.safe_load(SPEC.read_text())
    attribution = case_dir / "attribution.csv"
    attribution.write_bytes(ATTRIBUTION_SOURCE.read_bytes())
    document["sources"][0]["path"] = _relative(attribution)
    spec_path = case_dir / "spec.yaml"
    spec_path.write_text(yaml.safe_dump(document, sort_keys=False))
    schema = load_schema(SCHEMA)
    spec = load_spec(spec_path, target_schema=schema, require_executable=True)
    bare = pd.read_csv(ANCHOR_SOURCE)
    bare = bare.loc[bare["project"] == "alpha"].reset_index(drop=True)
    context = ReplayRunContext.create(ROOT, schema, (spec,), run_id=RUN_ID)
    cells, diagnostics = execute(spec, ROOT, anchor_df=bare, run_id=RUN_ID)
    collector = ReplayEvidenceCollector(context)
    with pytest.raises(ReplayOrchestrationError, match="runtime anchor binding"):
        collector.collect(spec, diagnostics.sources, cells)


@pytest.mark.parametrize("missing", ["cell", "contribution", "usage", "binding"])
def test_missing_external_anchor_evidence_is_not_replayable(tmp_path, missing):
    case = _bundle(tmp_path)
    if missing == "cell":
        _mutate_provenance(case, evidence_refs="")
    elif missing == "contribution":
        _mutate_provenance(case, contributions="")
    elif missing == "usage":
        _replace_usage(case, replace(_usage(case), evidence_ref_ids=()))
    else:
        anchor_id = _anchor_binding(case).evidence_ref_id
        case.manifest = replace(
            case.manifest,
            evidence=tuple(item for item in case.manifest.evidence
                           if item.kind != "anchor"),
            mapping_usages=tuple(
                replace(item, evidence_ref_ids=tuple(
                    ref for ref in item.evidence_ref_ids if ref != anchor_id))
                for item in case.manifest.mapping_usages))
    result = replay_cell(ROOT, case.manifest, case.request)
    assert (result.status, result.reason_code) == (
        ReplayStatus.NOT_REPLAYABLE, "ANCHOR_EVIDENCE_MISSING")


@pytest.mark.parametrize("field,value", [
    ("source_row_number", 2),
    ("anchor_filtered_position", 1),
])
def test_both_row_locators_are_independently_reproduced(tmp_path, field, value):
    case = _bundle(tmp_path)
    contribution = json.loads(_cell(case).contributions)
    contribution[0][field] = value
    _mutate_provenance(
        case, contributions=json.dumps(
            contribution, separators=(",", ":"), sort_keys=True))
    result = replay_cell(ROOT, case.manifest, case.request)
    assert (result.status, result.reason_code) == (
        ReplayStatus.FAIL, "ANCHOR_SELECTION_MISMATCH")


def test_matching_duplicate_participant_key_cannot_replace_position(tmp_path):
    case = _bundle(tmp_path, reading="READ-001")
    contribution = json.loads(_cell(case).contributions)
    # READ-002 has the same participant-derived record key as READ-001.
    contribution[0]["source_row_number"] = 2
    contribution[0]["anchor_filtered_position"] = 1
    _mutate_provenance(
        case, contributions=json.dumps(
            contribution, separators=(",", ":"), sort_keys=True))
    result = replay_cell(ROOT, case.manifest, case.request)
    assert (result.status, result.reason_code) == (
        ReplayStatus.FAIL, "ANCHOR_SELECTION_MISMATCH")


def test_scalar_snapshots_are_verification_targets(tmp_path):
    case = _bundle(tmp_path / "contribution")
    contribution = json.loads(_cell(case).contributions)
    contribution[0]["raw_value"] = "FORGED"
    contribution[0]["raw_value_json"] = '{"kind":"string","value":"FORGED"}'
    _mutate_provenance(
        case, contributions=json.dumps(
            contribution, separators=(",", ":"), sort_keys=True))
    result = replay_cell(ROOT, case.manifest, case.request)
    assert (result.status, result.reason_code, result.recomputed_value) == (
        ReplayStatus.FAIL, "RAW_VALUE_MISMATCH", None)

    case = _bundle(tmp_path / "cell")
    _mutate_provenance(
        case, raw_value_json='{"kind":"string","value":"FORGED"}')
    result = replay_cell(ROOT, case.manifest, case.request)
    assert (result.status, result.reason_code) == (
        ReplayStatus.FAIL, "RAW_VALUE_MISMATCH")


def test_source_bytes_reader_and_frame_contradictions_fail(tmp_path):
    case = _bundle(tmp_path / "bytes")
    case.anchor_path.write_bytes(case.anchor_path.read_bytes() + b"\n")
    result = replay_cell(ROOT, case.manifest, case.request)
    assert (result.status, result.reason_code) == (
        ReplayStatus.FAIL, "SOURCE_HASH_MISMATCH")

    case = _bundle(tmp_path / "reader")
    source = _source(case, "$anchor")
    case.manifest = replace(
        case.manifest,
        sources=tuple(
            replace(item, reader=replace(item.reader, reader_id="cidc_csv@v1"))
            if item == source else item for item in case.manifest.sources))
    result = replay_cell(ROOT, case.manifest, case.request)
    assert (result.status, result.reason_code) == (
        ReplayStatus.FAIL, "READER_OBSERVATION_MISMATCH")

    case = _bundle(tmp_path / "frame")
    anchor = _anchor_binding(case)
    details = dict(anchor.details)
    details["filtered_frame_sha256"] = "0" * 64
    case.manifest = replace(
        case.manifest,
        evidence=tuple(
            replace(item, details=tuple(sorted(details.items())))
            if item == anchor else item for item in case.manifest.evidence))
    result = replay_cell(ROOT, case.manifest, case.request)
    assert (result.status, result.reason_code) == (
        ReplayStatus.FAIL, "ANCHOR_IDENTITY_MISMATCH")


def test_missing_external_source_and_reader_contract_fail_closed(tmp_path):
    case = _bundle(tmp_path / "missing")
    case.anchor_path.unlink()
    result = replay_cell(ROOT, case.manifest, case.request)
    assert (result.status, result.reason_code) == (
        ReplayStatus.FAIL, "SOURCE_FILE_MISSING")

    case = _bundle(tmp_path / "flags")
    source = _source(case, "$anchor")
    flags = dict(source.reader.effective_flags)
    flags["skip_blank_lines"] = False
    case.manifest = replace(
        case.manifest,
        sources=tuple(
            replace(item, reader=replace(
                item.reader, effective_flags=tuple(sorted(flags.items()))))
            if item == source else item for item in case.manifest.sources))
    result = replay_cell(ROOT, case.manifest, case.request)
    assert (result.status, result.reason_code) == (
        ReplayStatus.FAIL, "READER_OBSERVATION_MISMATCH")

    case = _bundle(tmp_path / "environment")
    source = _source(case, "$anchor")
    dependencies = dict(source.reader.dependencies)
    dependencies["numpy"] = "0.invalid"
    producer_dependencies = dict(case.manifest.producer.dependencies)
    producer_dependencies["numpy"] = "0.invalid"
    case.manifest = replace(
        case.manifest,
        producer=replace(
            case.manifest.producer,
            dependencies=tuple(sorted(producer_dependencies.items()))),
        sources=tuple(
            replace(item, reader=replace(
                item.reader, dependencies=tuple(sorted(dependencies.items()))))
            if item == source else item for item in case.manifest.sources))
    result = replay_cell(ROOT, case.manifest, case.request)
    assert (result.status, result.reason_code) == (
        ReplayStatus.FAIL, "READER_ENVIRONMENT_MISMATCH")


@pytest.mark.parametrize("mutation", [
    "filter_column", "selector", "pre_count", "post_count", "columns",
])
def test_filter_and_construction_contradictions_fail_identity(tmp_path, mutation):
    case = _bundle(tmp_path)
    anchor = _anchor_binding(case)
    details = dict(anchor.details)
    filter_doc = dict(details["filter"])
    if mutation == "filter_column":
        filter_doc["column"] = "raw_none"
    elif mutation == "selector":
        filter_doc["selector"] = {"kind": "string", "value": "other"}
    elif mutation == "pre_count":
        details["pre_filter_row_count"] += 1
    elif mutation == "post_count":
        details["post_filter_row_count"] += 1
    else:
        details["ordered_columns"] = tuple(reversed(details["ordered_columns"]))
    details["filter"] = filter_doc
    case.manifest = replace(
        case.manifest,
        evidence=tuple(
            replace(item, details=tuple(sorted(details.items())))
            if item == anchor else item for item in case.manifest.evidence))
    result = replay_cell(ROOT, case.manifest, case.request)
    assert (result.status, result.reason_code) == (
        ReplayStatus.FAIL, "ANCHOR_IDENTITY_MISMATCH")


def test_dual_attribution_is_independent_and_mandatory(tmp_path):
    case = _bundle(tmp_path / "pass", target="direct_dual")
    usage = _usage(case)
    assert set(usage.source_ref_ids) == {
        _source(case, "$anchor").source_ref_id,
        _source(case, "attribution").source_ref_id,
    }
    assert replay_cell(ROOT, case.manifest, case.request).status is ReplayStatus.PASS

    case = _bundle(tmp_path / "missing", target="direct_dual")
    attribution = _source(case, "attribution")
    _replace_usage(case, replace(
        _usage(case), source_ref_ids=tuple(
            item for item in _usage(case).source_ref_ids
            if item != attribution.source_ref_id)))
    result = replay_cell(ROOT, case.manifest, case.request)
    assert (result.status, result.reason_code) == (
        ReplayStatus.NOT_REPLAYABLE, "SOURCE_LINK_MISSING")

    case = _bundle(tmp_path / "changed", target="direct_dual")
    case.attribution_path.write_text(
        "cimac_part_id,attribution_only_value\nDEVICE-A,plain\n")
    result = replay_cell(ROOT, case.manifest, case.request)
    assert (result.status, result.reason_code) == (
        ReplayStatus.FAIL, "SOURCE_HASH_MISMATCH")


def test_recorded_outputs_cannot_drive_external_recomputation(tmp_path):
    case = _bundle(tmp_path / "provenance")
    _mutate_provenance(case, value="FORGED")
    result = replay_cell(ROOT, case.manifest, case.request)
    assert (result.status, result.reason_code, result.recomputed_value) == (
        ReplayStatus.FAIL, "PROVENANCE_VALUE_MISMATCH", "plain")

    case = _bundle(tmp_path / "delivery")
    _mutate_delivered(case, direct_none="FORGED")
    result = replay_cell(ROOT, case.manifest, case.request)
    assert result.status is ReplayStatus.FAIL
    assert result.recomputed_value == "plain"


def test_pre_stage11_external_history_remains_not_replayable(tmp_path):
    case = _bundle(tmp_path)
    anchor_source_id = _source(case, "$anchor").source_ref_id
    anchor_id = _anchor_binding(case).evidence_ref_id
    _mutate_provenance(case, evidence_refs="", contributions="")
    case.manifest = replace(
        case.manifest,
        evidence=tuple(item for item in case.manifest.evidence
                       if item.kind != "anchor"),
        sources=tuple(item for item in case.manifest.sources
                      if item.source_id != "$anchor"),
        mapping_usages=tuple(
            replace(
                item,
                evidence_ref_ids=tuple(
                    ref for ref in item.evidence_ref_ids if ref != anchor_id),
                source_ref_ids=tuple(
                    ref for ref in item.source_ref_ids
                    if ref != anchor_source_id))
            for item in case.manifest.mapping_usages))
    result = replay_cell(ROOT, case.manifest, case.request)
    assert (result.status, result.reason_code) == (
        ReplayStatus.NOT_REPLAYABLE, "ANCHOR_EVIDENCE_MISSING")


def test_contract_versions_unchanged():
    from lib.provenance import PROVENANCE_SCHEMA_VERSION
    from lib.replay_contract import REPLAY_API_VERSION, REPLAY_STATUS_VERSION
    from lib.replay_manifest import MANIFEST_VERSION
    assert (REPLAY_API_VERSION, REPLAY_STATUS_VERSION, MANIFEST_VERSION,
            PROVENANCE_SCHEMA_VERSION, REASON_CODES_VERSION) == (1, 1, 2, 1, 10)

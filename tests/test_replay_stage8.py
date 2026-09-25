"""Synthetic-only tests for source-attributed declared-absence replay."""
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
    recompute_declared_absence,
    replay_cell,
)
from lib.replay_contract import ReplayStatus, encode_typed_raw
from lib.replay_orchestration import (
    ReplayEvidenceCollector,
    ReplayPublication,
    ReplayRunContext,
)
from lib.schema_model import contract_for, load as load_schema
from lib.source_ref import sha256_file
from lib.spec_executor import execute
from lib.spec_model import load as load_spec


ROOT = Path(__file__).resolve().parents[1]
FIXTURE = ROOT / "tests/fixtures/synthetic_replay"
SCHEMA = FIXTURE / "schema_stage8.yaml"
GROUP_SCHEMA = FIXTURE / "schema_stage8_group.yaml"
SPEC = FIXTURE / "spec_stage8.yaml"
GROUP_SPEC = FIXTURE / "spec_stage8_group.yaml"
SOURCE = FIXTURE / "sources/stage8_attribution.csv"
RUN_ID = "synthetic-stage8-run-0001"
ARTIFACT_ID = "synthetic-stage8-delivered"


@dataclass
class Case:
    manifest: object
    request: ReplayRequest
    spec: object
    cells: list
    source_path: Path
    delivered_path: Path
    provenance_path: Path
    spec_path: Path


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


def _anchor(*, missing_record=False):
    return pd.DataFrame([{
        "device_id": "DEVICE-Z" if missing_record else "DEVICE-A",
        "reading_id": "READ-999" if missing_record else "READ-001",
        "location": "offsite" if missing_record else "bench-west",
        "cimac_part_id": "DEVICE-Z" if missing_record else "DEVICE-A",
        "Cimac.id": "READ-999" if missing_record else "READ-001",
        "Collection_Event": "offsite" if missing_record else "bench-west",
    }])


def _bundle(tmp_path: Path, *, target="attributed_not_applicable",
            grouped=False, missing_record=False, spec_mutator=None) -> Case:
    case_dir = tmp_path / target
    case_dir.mkdir(parents=True)
    source_path = case_dir / "attribution.csv"
    source_path.write_bytes(SOURCE.read_bytes())

    template = GROUP_SPEC if grouped else SPEC
    document = yaml.safe_load(template.read_text())
    document["sources"][0]["path"] = _relative(source_path)
    if spec_mutator:
        spec_mutator(document)
    spec_path = case_dir / "spec.yaml"
    spec_path.write_text(yaml.safe_dump(document, sort_keys=False))

    schema = load_schema(GROUP_SCHEMA if grouped else SCHEMA)
    spec = load_spec(spec_path, target_schema=schema, require_executable=True)
    context = ReplayRunContext.create(
        ROOT, schema, (spec,), run_id=RUN_ID,
        application="synthetic-stage8-test")
    cells, diagnostics = execute(
        spec, ROOT, None if grouped else _anchor(missing_record=missing_record),
        contract=contract_for(schema), targets=(target,), run_id=RUN_ID)
    collector = ReplayEvidenceCollector(context)
    collector.collect(spec, diagnostics.sources, cells)

    delivered_rows = []
    for cell in cells:
        row = {name: None for name in schema.column_order}
        row.update({
            spec.record.participant_key: cell.cimac_part_id,
            spec.record.sample_key: cell.Cimac_id,
            spec.record.collection_event_key: cell.Collection_Event,
            cell.harmonized_field: cell.value,
        })
        delivered_rows.append(row)
    delivered = pd.DataFrame(delivered_rows, columns=schema.column_order)
    provenance = pd.DataFrame([
        {name: getattr(cell, name, "")
         for name in harmonize_spec.PROVENANCE_COLUMNS}
        for cell in cells
    ]).rename(columns={"Cimac_id": "Cimac.id"})
    published = ReplayPublication(
        context, collector, case_dir / "published").publish(
            delivered, provenance, artifact_id=ARTIFACT_ID,
            delivered_name="delivered.csv", provenance_name="provenance.csv")
    selected = delivered_rows[0]
    request = ReplayRequest(
        artifact_id=ARTIFACT_ID,
        row_key={name: selected[name] for name in schema.row_key},
        target_field=target)
    return Case(
        published.manifest, request, spec, cells, source_path,
        published.delivered_path, published.provenance_path, spec_path)


def _mapping(case: Case):
    return next(mapping for mapping in case.spec.mappings
                if mapping.target == case.request.target_field)


def _cell(case: Case):
    mapping = _mapping(case)
    participant = case.spec.record.participant_key
    sample = case.spec.record.sample_key
    return next(
        cell for cell in case.cells
        if cell.mapping_id == mapping.mapping_id
        and cell.cimac_part_id == case.request.row_key[participant]
        and cell.Cimac_id == case.request.row_key[sample])


def _usage(case: Case):
    mapping = _mapping(case)
    return next(
        usage for usage in case.manifest.mapping_usages
        if usage.spec_ref == case.spec.ref
        and usage.mapping_id == mapping.mapping_id)


def _source_binding(case: Case):
    return next(source for source in case.manifest.sources
                if source.spec_ref == case.spec.ref
                and source.source_id == "readings")


def _mutate_provenance(case: Case, **updates) -> None:
    fields, rows = _read_rows(case.provenance_path)
    mapping_id = _mapping(case).mapping_id
    participant = case.spec.record.participant_key
    sample = case.spec.record.sample_key
    row = next(item for item in rows
               if item["mapping_id"] == mapping_id
               and item["cimac_part_id"] == case.request.row_key[participant]
               and item["Cimac.id"] == case.request.row_key[sample])
    row.update(updates)
    _write_rows(case.provenance_path, fields, rows)
    case.manifest = replace(
        case.manifest,
        provenance_artifact=replace(
            case.manifest.provenance_artifact,
            sha256=sha256_file(case.provenance_path)))


def _mutate_delivery(case: Case, value: str) -> None:
    fields, rows = _read_rows(case.delivered_path)
    row = next(item for item in rows
               if item["device_id"] == case.request.row_key["device_id"]
               and item["reading_id"] == case.request.row_key["reading_id"])
    row[case.request.target_field] = value
    _write_rows(case.delivered_path, fields, rows)
    binding = case.manifest.delivered_artifacts[0]
    case.manifest = replace(
        case.manifest,
        delivered_artifacts=(replace(
            binding, sha256=sha256_file(case.delivered_path)),))


def _replace_usage(case: Case, usage) -> None:
    old = _usage(case)
    case.manifest = replace(
        case.manifest,
        mapping_usages=tuple(
            usage if item == old else item
            for item in case.manifest.mapping_usages))


def _scope_stop(case: Case, mapping, reason: str) -> None:
    with pytest.raises(replay_module._Stop) as caught:
        replay_module._check_declared_absence_scope(
            mapping, case.spec, {}, case.manifest.delivered_artifacts[0])
    assert (caught.value.status, caught.value.reason_code) == (
        ReplayStatus.NOT_REPLAYABLE, reason)


@pytest.mark.parametrize("target,state,serialized", [
    ("attributed_not_applicable", "not_applicable", "not_applicable"),
    ("attributed_source_missing", "source_missing", ""),
])
def test_both_semantic_states_and_cell_policies_pass(
        tmp_path, target, state, serialized):
    case = _bundle(tmp_path, target=target)
    cell = _cell(case)
    result = replay_cell(ROOT, case.manifest, case.request)
    assert (_mapping(case).value_state, cell.value_state) == (state, serialized)
    assert (result.status, result.reason_code, result.recomputed_value,
            result.source_ref_id) == (
        ReplayStatus.PASS, "REPLAY_MATCH", None, cell.source_ref_id)


def test_exact_runtime_source_chain_is_usage_scoped(tmp_path):
    case = _bundle(tmp_path)
    cell = _cell(case)
    usage = _usage(case)
    binding = _source_binding(case)
    assert (cell.source_ref_id,) == usage.source_ref_ids
    assert binding.source_ref_id == cell.source_ref_id
    assert (cell.source_resolved_path, cell.source_sha256,
            cell.source_logical_label) == (
        binding.path, binding.sha256, binding.logical_label)
    assert usage.evidence_ref_ids == ()
    assert replay_cell(ROOT, case.manifest, case.request).status is ReplayStatus.PASS


def test_missing_source_record_and_absent_mapping_column_still_pass(tmp_path):
    case = _bundle(
        tmp_path, target="attributed_not_applicable", missing_record=True)
    source = pd.read_csv(case.source_path)
    assert case.request.row_key["device_id"] not in set(source["device_id"])
    assert _mapping(case).column not in source.columns
    cell = _cell(case)
    assert cell.source_row_number == ""
    assert replay_cell(ROOT, case.manifest, case.request).status is ReplayStatus.PASS


@pytest.mark.parametrize("target,expected", [
    ("attributed_not_applicable", "device_id=DEVICE-A"),
    ("attributed_literal_key", "config:stage8.literal"),
])
def test_participant_and_literal_record_keys_pass(tmp_path, target, expected):
    case = _bundle(tmp_path, target=target)
    assert _cell(case).source_record_key == expected
    assert replay_cell(ROOT, case.manifest, case.request).status is ReplayStatus.PASS


def test_grouped_record_key_passes_without_row_lookup(tmp_path):
    case = _bundle(
        tmp_path, target="attributed_group_key", grouped=True)
    assert _cell(case).source_record_key == \
        "device_id=DEVICE-A;reading_id=READ-001"
    assert replay_cell(ROOT, case.manifest, case.request).status is ReplayStatus.PASS


@pytest.mark.parametrize("target,has_raw", [
    ("attributed_not_applicable", True),
    ("attributed_source_missing", True),
    ("attributed_raw_none", False),
])
def test_typed_null_and_omitted_raw_policies_pass(tmp_path, target, has_raw):
    case = _bundle(tmp_path, target=target)
    payload = _cell(case).raw_value_json
    assert bool(payload) is has_raw
    if has_raw:
        assert payload == encode_typed_raw(None).to_json()
    assert replay_cell(ROOT, case.manifest, case.request).status is ReplayStatus.PASS


def test_default_source_column_is_target_and_no_scalar_is_read(tmp_path):
    case = _bundle(tmp_path, target="attributed_default_column")
    assert _mapping(case).column == ""
    assert _cell(case).source_column == "attributed_default_column"
    assert replay_cell(ROOT, case.manifest, case.request).status is ReplayStatus.PASS


def test_no_value_rule_uses_ordinary_mapping_confidence(tmp_path):
    case = _bundle(tmp_path, target="attributed_no_value_rule")
    cell = _cell(case)
    assert (cell.rule_id, cell.confidence) == (
        "absence.asserted_null.v1", 0.73)
    assert cell.confidence != 0.11
    assert replay_cell(ROOT, case.manifest, case.request).status is ReplayStatus.PASS


@pytest.mark.parametrize("field,value,reason", [
    ("value_state", "source_missing", "VALUE_STATE_CONTEXT_MISMATCH"),
    ("rule_id", "wrong.rule", "RULE_IDENTITY_MISMATCH"),
    ("confidence", "0.01", "CONFIDENCE_CONTEXT_MISMATCH"),
    ("raw_value_json", '{"kind":"string","value":"wrong"}',
     "RAW_VALUE_MISMATCH"),
])
def test_semantic_context_contradictions_fail(
        tmp_path, field, value, reason):
    case = _bundle(tmp_path)
    _mutate_provenance(case, **{field: value})
    result = replay_cell(ROOT, case.manifest, case.request)
    assert (result.status, result.reason_code) == (ReplayStatus.FAIL, reason)


def test_intentionally_blank_state_rejects_unexpected_serialization(tmp_path):
    case = _bundle(tmp_path, target="attributed_source_missing")
    _mutate_provenance(case, value_state="source_missing")
    result = replay_cell(ROOT, case.manifest, case.request)
    assert (result.status, result.reason_code) == (
        ReplayStatus.FAIL, "VALUE_STATE_CONTEXT_MISMATCH")


def test_missing_required_raw_is_historical_incompleteness(tmp_path):
    case = _bundle(tmp_path)
    _mutate_provenance(case, raw_value_json="")
    result = replay_cell(ROOT, case.manifest, case.request)
    assert (result.status, result.reason_code) == (
        ReplayStatus.NOT_REPLAYABLE, "RAW_VALUE_ENCODING_MISSING")


def test_raw_none_rejects_snapshot(tmp_path):
    case = _bundle(tmp_path, target="attributed_raw_none")
    _mutate_provenance(
        case, raw_value_json=encode_typed_raw(None).to_json())
    result = replay_cell(ROOT, case.manifest, case.request)
    assert (result.status, result.reason_code) == (
        ReplayStatus.FAIL, "RAW_VALUE_MISMATCH")


def test_source_hash_drift_and_missing_file_fail(tmp_path):
    drift = _bundle(tmp_path / "drift")
    drift.source_path.write_bytes(drift.source_path.read_bytes() + b"\n")
    result = replay_cell(ROOT, drift.manifest, drift.request)
    assert (result.status, result.reason_code) == (
        ReplayStatus.FAIL, "SOURCE_HASH_MISMATCH")

    missing = _bundle(tmp_path / "missing")
    missing.source_path.unlink()
    result = replay_cell(ROOT, missing.manifest, missing.request)
    assert (result.status, result.reason_code) == (
        ReplayStatus.FAIL, "SOURCE_FILE_MISSING")


def test_bound_path_and_reader_contradictions_fail(tmp_path):
    path_case = _bundle(tmp_path / "path")
    binding = _source_binding(path_case)
    other = path_case.source_path.with_name("other.csv")
    other.write_bytes(path_case.source_path.read_bytes())
    path_case.manifest = replace(
        path_case.manifest,
        sources=(replace(binding, path=_relative(other)),))
    result = replay_cell(ROOT, path_case.manifest, path_case.request)
    assert (result.status, result.reason_code) == (
        ReplayStatus.FAIL, "SOURCE_DECLARATION_MISMATCH")

    reader_case = _bundle(tmp_path / "reader")
    binding = _source_binding(reader_case)
    reader_case.manifest = replace(
        reader_case.manifest,
        sources=(replace(
            binding,
            reader=replace(binding.reader, reader_id="excel@v1")),))
    result = replay_cell(ROOT, reader_case.manifest, reader_case.request)
    assert (result.status, result.reason_code) == (
        ReplayStatus.FAIL, "SOURCE_DECLARATION_MISMATCH")


def test_reader_flags_environment_and_row_count_are_exact(tmp_path):
    flags_case = _bundle(tmp_path / "flags")
    binding = _source_binding(flags_case)
    flags = dict(binding.reader.flags)
    flags["skiprows"] = 1
    flags_case.manifest = replace(
        flags_case.manifest,
        sources=(replace(
            binding, reader=replace(
                binding.reader,
                effective_flags=tuple(sorted(flags.items())))),))
    result = replay_cell(ROOT, flags_case.manifest, flags_case.request)
    assert (result.status, result.reason_code) == (
        ReplayStatus.FAIL, "READER_OBSERVATION_MISMATCH")

    environment = _bundle(tmp_path / "environment")
    binding = _source_binding(environment)
    environment.manifest = replace(
        environment.manifest,
        sources=(replace(
            binding, reader=replace(
                binding.reader, dependencies=(("pandas", "0.0.0"),))),))
    result = replay_cell(ROOT, environment.manifest, environment.request)
    assert (result.status, result.reason_code) == (
        ReplayStatus.NOT_REPLAYABLE, "READER_ENVIRONMENT_MISMATCH")

    rows = _bundle(tmp_path / "rows")
    binding = _source_binding(rows)
    rows.manifest = replace(
        rows.manifest, sources=(replace(
            binding, n_rows=binding.n_rows + 1),))
    result = replay_cell(ROOT, rows.manifest, rows.request)
    assert (result.status, result.reason_code) == (
        ReplayStatus.FAIL, "SOURCE_ROW_COUNT_MISMATCH")


@pytest.mark.parametrize("identity", ["python", "yaml"])
def test_python_and_yaml_environments_are_exact(tmp_path, identity):
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


def test_rebuilt_source_ref_identity_must_match(tmp_path):
    case = _bundle(tmp_path)
    binding = _source_binding(case)
    fake = "src-" + "f" * 24
    _mutate_provenance(case, source_ref_id=fake)
    usage = replace(_usage(case), source_ref_ids=(fake,))
    case.manifest = replace(
        case.manifest,
        sources=(replace(
            binding, identity=f"source:{fake}", source_ref_id=fake),),
        mapping_usages=tuple(
            usage if item.mapping_id == usage.mapping_id else item
            for item in case.manifest.mapping_usages))
    result = replay_cell(ROOT, case.manifest, case.request)
    assert (result.status, result.reason_code) == (
        ReplayStatus.FAIL, "SOURCE_DECLARATION_MISMATCH")


@pytest.mark.parametrize("missing", ["cell", "usage", "binding"])
def test_missing_strict_source_links_are_not_replayable(tmp_path, missing):
    case = _bundle(tmp_path)
    if missing == "cell":
        _mutate_provenance(case, source_ref_id="")
        manifest_input = case.manifest
    elif missing == "usage":
        _replace_usage(case, replace(_usage(case), source_ref_ids=()))
        manifest_input = case.manifest
    else:
        document = case.manifest.to_dict()
        document["sources"] = []
        path = tmp_path / "missing-source-manifest.json"
        path.write_text(json.dumps(document))
        manifest_input = _relative(path)
    result = replay_cell(ROOT, manifest_input, case.request)
    assert (result.status, result.reason_code) == (
        ReplayStatus.NOT_REPLAYABLE, "SOURCE_LINK_MISSING")


def test_wrong_cell_and_usage_source_links_fail(tmp_path):
    cell_case = _bundle(tmp_path / "cell")
    _mutate_provenance(cell_case, source_ref_id="src-contradiction")
    result = replay_cell(ROOT, cell_case.manifest, cell_case.request)
    assert (result.status, result.reason_code) == (
        ReplayStatus.FAIL, "SOURCE_DECLARATION_MISMATCH")

    usage_case = _bundle(tmp_path / "usage")
    binding = _source_binding(usage_case)
    wrong_id = "src-" + "e" * 24
    wrong = replace(
        binding, identity=f"source:{wrong_id}", source_ref_id=wrong_id,
        source_id="other", config_key="other")
    usage = replace(_usage(usage_case), source_ref_ids=(wrong_id,))
    usage_case.manifest = replace(
        usage_case.manifest,
        sources=(binding, wrong),
        mapping_usages=tuple(
            usage if item.mapping_id == usage.mapping_id else item
            for item in usage_case.manifest.mapping_usages))
    result = replay_cell(ROOT, usage_case.manifest, usage_case.request)
    assert (result.status, result.reason_code) == (
        ReplayStatus.FAIL, "SOURCE_DECLARATION_MISMATCH")


def test_binding_declaration_identity_must_match(tmp_path):
    case = _bundle(tmp_path)
    binding = _source_binding(case)
    case.manifest = replace(
        case.manifest,
        sources=(replace(binding, config_key="different"),))
    result = replay_cell(ROOT, case.manifest, case.request)
    assert (result.status, result.reason_code) == (
        ReplayStatus.FAIL, "SOURCE_DECLARATION_MISMATCH")


@pytest.mark.parametrize("field,value", [
    ("source_file", "wrong.csv"),
    ("source_logical_label", "wrong.csv"),
    ("source_resolved_path", "wrong/path.csv"),
    ("source_sha256", "0" * 64),
    ("source_sheet", "Sheet1"),
    ("source_column", "invented_scalar"),
    ("source_record_key", "device_id=WRONG"),
    ("source_row_number", "0"),
    ("evidence_refs", "config-unexpected"),
    ("contributions", '[{"status":"input"}]'),
    ("derivation", '{"kind":"unexpected"}'),
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


def test_recomputation_signature_and_inputs_remain_source_free(
        tmp_path, monkeypatch):
    signature = inspect.signature(recompute_declared_absence)
    assert tuple(signature.parameters) == ("declared_state",)
    assert not set(signature.parameters) & {
        "source_ref", "source_path", "source_hash", "source_scalar",
        "delivered_value", "provenance_value", "value_state", "raw_value",
    }
    calls = []
    original = recompute_declared_absence

    def guarded(**kwargs):
        calls.append(kwargs)
        return original(**kwargs)

    monkeypatch.setattr(replay_module, "recompute_declared_absence", guarded)
    case = _bundle(tmp_path)
    result = replay_cell(ROOT, case.manifest, case.request)
    assert result.status is ReplayStatus.PASS
    assert calls == [{"declared_state": "not_applicable"}]


@pytest.mark.parametrize("mutation,reason", [
    ("delivery", "RECOMPUTED_VALUE_MISMATCH"),
    ("provenance", "PROVENANCE_VALUE_MISMATCH"),
    ("state", "VALUE_STATE_CONTEXT_MISMATCH"),
    ("raw", "RAW_VALUE_MISMATCH"),
])
def test_verification_inputs_cannot_change_recomputation(
        tmp_path, monkeypatch, mutation, reason):
    calls = []
    original = recompute_declared_absence

    def guarded(**kwargs):
        calls.append(kwargs)
        return original(**kwargs)

    monkeypatch.setattr(replay_module, "recompute_declared_absence", guarded)
    case = _bundle(tmp_path)
    if mutation == "delivery":
        _mutate_delivery(case, "tampered")
        _mutate_provenance(case, value="tampered")
    elif mutation == "provenance":
        _mutate_provenance(case, value="tampered")
    elif mutation == "state":
        _mutate_provenance(case, value_state="source_missing")
    else:
        _mutate_provenance(
            case, raw_value_json=encode_typed_raw("tampered").to_json())
    result = replay_cell(ROOT, case.manifest, case.request)
    assert (result.status, result.reason_code, result.recomputed_value) == (
        ReplayStatus.FAIL, reason, None)
    assert calls == [{"declared_state": "not_applicable"}]


def test_source_content_cannot_enter_pure_semantics(tmp_path):
    case = _bundle(tmp_path)
    before = recompute_declared_absence(declared_state="not_applicable")
    case.source_path.write_text(
        case.source_path.read_text().replace("alpha", "changed"))
    after = recompute_declared_absence(declared_state="not_applicable")
    assert before == after
    result = replay_cell(ROOT, case.manifest, case.request)
    assert (result.status, result.reason_code) == (
        ReplayStatus.FAIL, "SOURCE_HASH_MISMATCH")


def test_complete_v2_history_passes_and_current_file_cannot_repair_link(
        tmp_path):
    complete = _bundle(tmp_path / "complete")
    assert complete.manifest.manifest_version == 2
    assert replay_cell(ROOT, complete.manifest, complete.request).status is ReplayStatus.PASS

    incomplete = _bundle(tmp_path / "incomplete")
    assert incomplete.source_path.is_file()
    _replace_usage(
        incomplete, replace(_usage(incomplete), source_ref_ids=()))
    result = replay_cell(ROOT, incomplete.manifest, incomplete.request)
    assert (result.status, result.reason_code) == (
        ReplayStatus.NOT_REPLAYABLE, "SOURCE_LINK_MISSING")


@pytest.mark.parametrize("target,reason", [
    ("unsupported_no_mapping", "ANCHOR_SOURCE_UNSUPPORTED_STAGE1"),
    ("anchor_absence", "ANCHOR_SOURCE_UNSUPPORTED_STAGE1"),
])
def test_no_mapping_and_anchor_replay_remain_unsupported(
        tmp_path, target, reason):
    case = _bundle(tmp_path, target=target)
    result = replay_cell(ROOT, case.manifest, case.request)
    assert (result.status, result.reason_code) == (
        ReplayStatus.NOT_REPLAYABLE, reason)


@pytest.mark.parametrize("change,reason", [
    ({"when": "synthetic_condition"}, "MAPPING_CONDITION_UNSUPPORTED"),
    ({"group": "synthetic_group"}, "SOURCE_NOT_SINGLE_FILE"),
    ({"column_by_member": {"member": "column"}}, "SOURCE_NOT_SINGLE_FILE"),
])
def test_conditions_and_group_member_selection_remain_unsupported(
        tmp_path, change, reason):
    case = _bundle(tmp_path)
    _scope_stop(case, replace(_mapping(case), **change), reason)


@pytest.mark.parametrize("provenance_change,reason", [
    ({"row_from": "readings"}, "SOURCE_NOT_SINGLE_FILE"),
    ({"raw_from": "synthetic_read"}, "DERIVATION_UNSUPPORTED_STAGE2"),
    ({"contributions_from": "synthetic_group"}, "SOURCE_NOT_SINGLE_FILE"),
    ({"row_number": ""}, "SOURCE_NOT_SINGLE_FILE"),
    ({"source_ref": "group.readings"}, "SOURCE_NOT_SINGLE_FILE"),
])
def test_row_scalar_group_and_dynamic_source_shapes_remain_unsupported(
        tmp_path, provenance_change, reason):
    case = _bundle(tmp_path)
    mapping = _mapping(case)
    provenance = replace(mapping.provenance, **provenance_change)
    _scope_stop(case, replace(mapping, provenance=provenance), reason)


def test_condition_contributions_and_derivation_remain_unsupported(tmp_path):
    case = _bundle(tmp_path)
    mapping = _mapping(case)
    contributions = replace(
        mapping.contributions, conditions=("synthetic_condition",))
    _scope_stop(
        case, replace(mapping, contributions=contributions),
        "CONTRIBUTIONS_UNSUPPORTED_STAGE2")
    _scope_stop(
        case, replace(mapping, derivation=("synthetic",)),
        "DERIVATION_UNSUPPORTED_STAGE2")


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


def test_stage8_changes_no_contract_version(tmp_path):
    case = _bundle(tmp_path)
    result = replay_cell(ROOT, case.manifest, case.request)
    assert (result.api_version, result.status_version,
            result.reason_codes_version,
            case.manifest.manifest_version) == (1, 1, 10, 2)

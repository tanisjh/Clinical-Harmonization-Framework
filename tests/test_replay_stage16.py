"""Synthetic strict replay tests for Stage 16 fact composition."""
from __future__ import annotations

import json
from dataclasses import replace

import pytest

import lib.replay as replay_module
from lib.derivation import is_stage16_derivation
from lib.replay import replay_cell
from lib.replay_contract import ReplayStatus
from lib.replay_orchestration import ReplayOrchestrationError
from lib.spec_model import DerivationVariant, load as load_spec
import test_replay_stage12 as stage12
import test_replay_stage15 as stage15


def _group_derivation(doc, *, variants=False):
    doc["mappings"] = [
        item for item in doc["mappings"]
        if item["mapping_id"] != "group.recursive"
    ]
    doc["reads"] = [{
        "id": "selected_numeric", "source": "cohort.scalar",
        "column": "numeric_alpha",
        "column_by_member": {"alpha": "numeric_alpha", "beta": "numeric_beta"},
        "coerce": "numeric",
    }]
    doc["conditions"] = [{
        "id": "positive", "read": "selected_numeric",
        "predicate": "gt", "value": 0,
    }]
    mapping = next(item for item in doc["mappings"]
                   if item["mapping_id"] == "group.direct.numeric")
    record = {
        "kind": "synthetic_stage16",
        "value": {"$ref": "value"},
        "column": {"$ref": "source_column"},
        "read": {"$ref": "read.selected_numeric"},
        "condition": {"$ref": "condition.positive"},
        "member": {"$ref": "group_member.cohort"},
        "member_value": {"$ref": "group_value.cohort"},
        "member_label": {"$ref": "group_label.cohort"},
    }
    mapping["derivation"] = (
        [{"when": "positive", "record": record},
         {"record": {"kind": "fallback", "value": {"$ref": "value"}}}]
        if variants else [{"record": record}])


def _ordinary_cross_group(doc):
    _group_derivation(doc)
    mapping = next(item for item in doc["mappings"]
                   if item["mapping_id"] == "group.direct.numeric")
    mapping["source"] = "scalar_alpha"
    mapping.pop("column_by_member", None)
    mapping["column"] = "numeric_alpha"


def _raw_from_scalar(doc):
    _ordinary_cross_group(doc)
    mapping = next(item for item in doc["mappings"]
                   if item["mapping_id"] == "group.direct.numeric")
    mapping["provenance"] = {"raw_from": "selected_numeric"}


def _sourced_constant(doc):
    _group_derivation(doc)
    mapping = next(item for item in doc["mappings"]
                   if item["mapping_id"] == "group.direct.numeric")
    mapping.update({
        "op": "constant", "value": 42, "when": "positive",
        "column": "numeric_alpha",
        "column_by_member": {"alpha": "numeric_alpha", "beta": "numeric_beta"},
        "provenance": {
            "source_ref": "cohort.scalar", "raw_from": "selected_numeric",
            "contributions_from": "cohort",
            "record_key_from": "participant",
        },
        "contributions": {"conditions": ["selected_numeric"]},
        "derivation": [{"record": {
            "kind": "sourced_constant",
            "value": {"$ref": "value"},
            "read": {"$ref": "read.selected_numeric"},
            "member": {"$ref": "group_member.cohort"},
        }}],
    })
    mapping.pop("source", None)
    mapping.pop("coerce", None)


def _branch_derivation(doc, *, multiple=False):
    stage15._siblings(doc, multiple=multiple)
    for mapping in doc["mappings"]:
        if mapping["mapping_id"] not in {"branch.one", "branch.two"}:
            continue
        condition = mapping["when"]
        mapping["derivation"] = [{"record": {
            "kind": "branch_projection",
            "value": {"$ref": "value"},
            "column": {"$ref": "source_column"},
            "read": {"$ref": "read.gate_text"},
            "condition": {"$ref": f"condition.{condition}"},
        }}]


def test_all_approved_fact_namespaces_reuse_verified_group_and_read(tmp_path):
    case = stage12._bundle(
        tmp_path, participant="DEVICE-B", target="direct_numeric",
        spec_mutate=_group_derivation)
    result = replay_cell(stage12.ROOT, case.manifest, case.request)
    assert (result.status, result.reason_code, result.recomputed_value) == (
        ReplayStatus.PASS, "REPLAY_MATCH", 7.0), result.message


def test_operation_read_and_group_facts_share_one_group_decision(
        monkeypatch, tmp_path):
    original = replay_module.select_first_containing_record
    calls = []
    def spy(*args, **kwargs):
        calls.append(args)
        return original(*args, **kwargs)
    monkeypatch.setattr(replay_module, "select_first_containing_record", spy)
    case = stage12._bundle(
        tmp_path, participant="DEVICE-B", target="direct_numeric",
        spec_mutate=_group_derivation)
    assert replay_cell(
        stage12.ROOT, case.manifest, case.request).status is ReplayStatus.PASS
    assert len(calls) == 1


def test_ordered_variant_selection_uses_verified_condition(tmp_path):
    case = stage12._bundle(
        tmp_path, participant="DEVICE-A", target="direct_numeric",
        spec_mutate=lambda doc: _group_derivation(doc, variants=True))
    result = replay_cell(stage12.ROOT, case.manifest, case.request)
    assert (result.status, result.reason_code) == (
        ReplayStatus.PASS, "REPLAY_MATCH")
    event = stage12._cell(case)
    assert json.loads(event.derivation)["kind"] == "synthetic_stage16"


def test_cross_group_derivation_sources_are_usage_scoped(tmp_path):
    case = stage12._bundle(
        tmp_path, participant="DEVICE-A", target="direct_numeric",
        spec_mutate=_ordinary_cross_group)
    usage = stage12._usage(case)
    assert len(usage.source_ref_ids) == 3
    assert replay_cell(
        stage12.ROOT, case.manifest, case.request).status is ReplayStatus.PASS

    operation_ref = stage12._source(case, "scalar_alpha").source_ref_id
    stage12._replace_usage(case, type(usage)(
        spec_ref=usage.spec_ref, mapping_id=usage.mapping_id,
        source_ref_ids=(operation_ref,), rule_refs=usage.rule_refs,
        evidence_ref_ids=usage.evidence_ref_ids))
    result = replay_cell(stage12.ROOT, case.manifest, case.request)
    assert (result.status, result.reason_code) == (
        ReplayStatus.NOT_REPLAYABLE, "NAMED_READ_EVIDENCE_MISSING")


def test_publication_refuses_missing_derivation_only_usage_evidence(tmp_path):
    def strip_named_read(cell):
        if cell.mapping_id != "group.direct.numeric":
            return cell
        contributions = [
            item for item in json.loads(cell.contributions)
            if item.get("role") != "selected_numeric"
        ]
        return replace(
            cell, contributions=json.dumps(
                contributions, separators=(",", ":"), sort_keys=True))

    with pytest.raises(ReplayOrchestrationError, match="named-read evidence"):
        stage12._bundle(
            tmp_path, participant="DEVICE-A", target="direct_numeric",
            spec_mutate=_ordinary_cross_group, cell_mutator=strip_named_read)


def test_scalar_raw_from_uses_verified_named_read_not_recorded_snapshot(tmp_path):
    case = stage12._bundle(
        tmp_path, participant="DEVICE-A", target="direct_numeric",
        spec_mutate=_raw_from_scalar)
    result = replay_cell(stage12.ROOT, case.manifest, case.request)
    assert (result.status, result.reason_code, result.recomputed_value) == (
        ReplayStatus.PASS, "REPLAY_MATCH", 12.5)


def test_mapping_branch_composes_with_sourced_constant(tmp_path):
    case = stage12._bundle(
        tmp_path, participant="DEVICE-A", target="direct_numeric",
        spec_mutate=_sourced_constant)
    result = replay_cell(stage12.ROOT, case.manifest, case.request)
    assert (result.status, result.reason_code, result.recomputed_value) == (
        ReplayStatus.PASS, "REPLAY_MATCH", 42), result.message


def test_recorded_derivation_is_comparison_only(tmp_path):
    case = stage12._bundle(
        tmp_path, participant="DEVICE-A", target="direct_numeric",
        spec_mutate=_group_derivation)
    stage12._mutate_provenance(case, derivation='{"fabricated":true}')
    result = replay_cell(stage12.ROOT, case.manifest, case.request)
    assert (result.status, result.reason_code) == (
        ReplayStatus.FAIL, "DERIVATION_MISMATCH")


def test_missing_named_read_contribution_keeps_stage15_reason(tmp_path):
    case = stage12._bundle(
        tmp_path, participant="DEVICE-A", target="direct_numeric",
        spec_mutate=_group_derivation)
    stage12._mutate_contributions(
        case, lambda items: items.__setitem__(slice(None), [
            item for item in items if item.get("role") != "selected_numeric"
        ]))
    result = replay_cell(stage12.ROOT, case.manifest, case.request)
    assert (result.status, result.reason_code) == (
        ReplayStatus.NOT_REPLAYABLE, "NAMED_READ_EVIDENCE_MISSING")


def test_mapping_branch_authority_composes_before_derivation(tmp_path):
    case = stage15._bundle(
        tmp_path, target="preferred_raw", spec_mutate=_branch_derivation)
    result = replay_cell(stage15.ROOT, case.manifest, case.request)
    assert (result.status, result.reason_code, result.mapping_id) == (
        ReplayStatus.PASS, "REPLAY_MATCH", "branch.one")


def test_multiple_true_mapping_family_remains_not_replayable(tmp_path):
    case = stage15._bundle(
        tmp_path, target="preferred_raw",
        spec_mutate=lambda doc: _branch_derivation(doc, multiple=True))
    result = replay_cell(stage15.ROOT, case.manifest, case.request)
    assert (result.status, result.reason_code) == (
        ReplayStatus.NOT_REPLAYABLE, "MULTI_EVENT_BRANCH_UNSUPPORTED")


@pytest.mark.parametrize("ref", [
    "raw_value", "anchor_group_size", "anchor_selected_row",
])
def test_explicitly_excluded_fact_names_remain_unsupported(ref):
    assert not is_stage16_derivation((
        DerivationVariant(record={"fact": {"$ref": ref}}),))


def test_provisional_state_now_composes_with_stage16_facts():
    spec = load_spec(stage12.SPEC, vocabulary_dir=stage12.VOCABULARY.parent)
    mapping = next(item for item in spec.mappings
                   if item.mapping_id == "group.direct.numeric")
    mapping = replay_module.dc_replace(
        mapping, value_state="provisional",
        derivation=(DerivationVariant(
            record={"value": {"$ref": "value"}}),))
    assert replay_module._is_stage16_derivation_mapping(mapping, spec)

"""Synthetic tests for route-aware finalized raw replay (Stage 17)."""

from __future__ import annotations

import json
from dataclasses import replace

import pytest

import lib.replay as replay_module
from lib.raw_policy import (
    COMPLETION_FALLBACK_EXHAUSTED,
    FinalizedRaw,
    RawAuthorityUnavailable,
    VerifiedOperationRouteResult,
    VerifiedRawCandidate,
    finalize_raw_value,
    raw_from_is_causal,
)
from lib.replay import replay_cell
from lib.replay_contract import ReplayStatus, encode_typed_raw
from lib.spec_model import DerivationVariant, load as load_spec
import test_replay_stage4 as stage4
import test_replay_stage12 as stage12


def _route(value="delivered", raw="route", *, available=True,
           completion="value"):
    return VerifiedOperationRouteResult(
        value=value,
        raw_candidate=VerifiedRawCandidate(available, raw),
        completion=completion)


@pytest.mark.parametrize("policy", ["", "value"])
def test_default_value_uses_route_raw(policy):
    assert finalize_raw_value(_route(raw="001"), {}, raw_policy=policy) == \
        FinalizedRaw("001", True)


def test_default_value_uses_named_read_result_and_requires_it():
    route = _route(raw="source")
    assert finalize_raw_value(
        route, {"normalized": 7.0}, raw_from_id="normalized") == \
        FinalizedRaw(7.0, True)
    with pytest.raises(RawAuthorityUnavailable, match="raw_from"):
        finalize_raw_value(route, {}, raw_from_id="normalized")


@pytest.mark.parametrize("policy,expected", [
    ("coerced", FinalizedRaw(None, True)),
    ("none", FinalizedRaw(None, False)),
])
def test_masking_policies_do_not_require_raw_from(policy, expected):
    route = _route(value=None, raw="source", available=False)
    assert finalize_raw_value(
        route, {}, raw_from_id="missing", raw_policy=policy) == expected
    assert not raw_from_is_causal(
        raw_from_id="missing", raw_policy=policy)


def test_exhaustion_masks_raw_from_and_coerced_policy():
    route = _route(
        value=91, raw="winner", available=False,
        completion=COMPLETION_FALLBACK_EXHAUSTED)
    assert finalize_raw_value(
        route, {}, raw_from_id="missing", raw_policy="coerced") == \
        FinalizedRaw(None, False)
    assert not raw_from_is_causal(
        raw_from_id="missing", raw_policy="value",
        completion=COMPLETION_FALLBACK_EXHAUSTED)


@pytest.mark.parametrize("value", [None, "", False, 0, 3, 3.5, "003"])
def test_exact_scalar_type_and_present_null_are_preserved(value):
    first = finalize_raw_value(_route(raw=value), {})
    second = finalize_raw_value(_route(raw=value), {})
    assert first == second == FinalizedRaw(value, True)
    assert type(first.value) is type(value)


def test_unavailable_route_raw_is_not_present_null():
    with pytest.raises(RawAuthorityUnavailable, match="route raw"):
        finalize_raw_value(_route(raw=None, available=False), {})
    assert finalize_raw_value(_route(raw=None, available=True), {}) == \
        FinalizedRaw(None, True)


def _add_group_read(doc):
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


def _group_map_raw_derivation(doc, *, mapping_condition=False):
    _add_group_read(doc)
    doc["conditions"] = [{
        "id": "positive", "read": "selected_numeric",
        "predicate": "gt", "value": 0,
    }]
    mapping = next(item for item in doc["mappings"]
                   if item["mapping_id"] == "group.map.inline")
    if mapping_condition:
        mapping["when"] = "positive"
    mapping["derivation"] = [{"record": {
        "kind": "stage17_group_map",
        "raw": {"$ref": "raw_value"},
        "value": {"$ref": "value"},
        "column": {"$ref": "source_column"},
        "positive": {"$ref": "condition.positive"},
        "member": {"$ref": "group_member.cohort"},
    }}]


def test_group_value_map_raw_and_branch_evidence_reuse(tmp_path):
    case = stage12._bundle(
        tmp_path, participant="DEVICE-A", target="status_label",
        spec_mutate=lambda doc: _group_map_raw_derivation(
            doc, mapping_condition=True))
    event = stage12._cell(case)
    assert json.loads(event.derivation)["raw"] == "ON"
    result = replay_cell(stage12.ROOT, case.manifest, case.request)
    assert (result.status, result.reason_code, result.recomputed_value) == (
        ReplayStatus.PASS, "REPLAY_MATCH", "ready"), result.message


def _file_membership_raw_derivation(doc):
    doc["mappings"] = [
        item for item in doc["mappings"]
        if item["mapping_id"] != "group.recursive"
    ]
    mapping = next(item for item in doc["mappings"]
                   if item["mapping_id"] == "group.membership")
    mapping["derivation"] = [{"record": {
        "kind": "stage17_membership",
        "participant_raw": {"$ref": "raw_value"},
        "value": {"$ref": "value"},
        "member": {"$ref": "group_member.cohort"},
    }}]


def test_group_file_membership_keeps_participant_as_route_raw(tmp_path):
    case = stage12._bundle(
        tmp_path, participant="DEVICE-A", target="membership_label",
        spec_mutate=_file_membership_raw_derivation)
    event = stage12._cell(case)
    assert json.loads(event.derivation)["participant_raw"] == "DEVICE-A"
    result = replay_cell(stage12.ROOT, case.manifest, case.request)
    assert (result.status, result.reason_code, result.recomputed_value) == (
        ReplayStatus.PASS, "REPLAY_MATCH", "alpha-value"), result.message


def _group_conditional_raw_from(doc):
    _add_group_read(doc)
    doc["conditions"] = [{
        "id": "positive", "read": "selected_numeric",
        "predicate": "gt", "value": 0,
    }]
    mapping = next(item for item in doc["mappings"]
                   if item["mapping_id"] == "group.direct.text")
    mapping.pop("coerce", None)
    mapping["op"] = "conditional"
    mapping["clauses"] = [{
        "when": {"read": "selected_numeric", "predicate": "gt", "value": 0},
        "value": "chosen",
    }]
    mapping["otherwise"] = "not-chosen"
    mapping["provenance"] = {
        "raw_from": "selected_numeric", "contributions_from": "cohort"}
    mapping["contributions"] = {"conditions": ["selected_numeric"]}
    mapping["derivation"] = [{"record": {
        "kind": "stage17_group_conditional",
        "raw": {"$ref": "raw_value"},
        "value": {"$ref": "value"},
        "positive": {"$ref": "condition.positive"},
        "member_label": {"$ref": "group_label.cohort"},
    }}]


def test_group_parent_conditional_raw_from_reuses_one_decision(
        monkeypatch, tmp_path):
    original = replay_module.select_first_containing_record
    calls = []

    def spy(*args, **kwargs):
        calls.append(args)
        return original(*args, **kwargs)

    monkeypatch.setattr(replay_module, "select_first_containing_record", spy)
    case = stage12._bundle(
        tmp_path, participant="DEVICE-A", target="direct_text",
        spec_mutate=_group_conditional_raw_from)
    event = stage12._cell(case)
    assert json.loads(event.derivation)["raw"] == 12.5
    result = replay_cell(stage12.ROOT, case.manifest, case.request)
    assert (result.status, result.reason_code, result.recomputed_value) == (
        ReplayStatus.PASS, "REPLAY_MATCH", "chosen"), result.message
    assert len(calls) == 1


def _masked_group_raw_from(doc, policy):
    _add_group_read(doc)
    mapping = next(item for item in doc["mappings"]
                   if item["mapping_id"] == "group.direct.none")
    mapping["provenance"] = {
        "raw_from": "selected_numeric", "raw_value": policy}


@pytest.mark.parametrize("policy", ["coerced", "none"])
def test_masked_group_raw_from_is_not_produced_or_required(tmp_path, policy):
    case = stage12._bundle(
        tmp_path, participant="DEVICE-A", target="direct_none",
        spec_mutate=lambda doc: _masked_group_raw_from(doc, policy))
    event = stage12._cell(case)
    assert not any(item.get("role") == "selected_numeric"
                   for item in json.loads(event.contributions or "[]"))
    result = replay_cell(stage12.ROOT, case.manifest, case.request)
    assert (result.status, result.reason_code) == (
        ReplayStatus.PASS, "REPLAY_MATCH"), result.message


def _active_group_raw_from(doc):
    _masked_group_raw_from(doc, "value")
    mapping = next(item for item in doc["mappings"]
                   if item["mapping_id"] == "group.direct.none")
    mapping["derivation"] = [{"record": {
        "kind": "stage17_active_raw_from",
        "raw": {"$ref": "raw_value"},
    }}]


def test_active_raw_from_missing_named_read_evidence_is_not_replayable(
        tmp_path):
    case = stage12._bundle(
        tmp_path, participant="DEVICE-A", target="direct_none",
        spec_mutate=_active_group_raw_from)
    stage12._mutate_contributions(
        case,
        lambda items: items.__setitem__(
            slice(None),
            [item for item in items
             if item.get("role") != "selected_numeric"]))
    result = replay_cell(stage12.ROOT, case.manifest, case.request)
    assert (result.status, result.reason_code) == (
        ReplayStatus.NOT_REPLAYABLE, "NAMED_READ_EVIDENCE_MISSING")


def test_required_typed_raw_snapshot_absence_is_not_replayable(tmp_path):
    case = stage12._bundle(
        tmp_path, participant="DEVICE-A", target="status_label",
        spec_mutate=_group_map_raw_derivation)
    stage12._mutate_provenance(case, raw_value_json="")
    result = replay_cell(stage12.ROOT, case.manifest, case.request)
    assert (result.status, result.reason_code) == (
        ReplayStatus.NOT_REPLAYABLE, "RAW_VALUE_ENCODING_MISSING")


def test_suppressed_typed_raw_snapshot_presence_is_mismatch(tmp_path):
    case = stage12._bundle(
        tmp_path, participant="DEVICE-A", target="direct_none",
        spec_mutate=lambda doc: _masked_group_raw_from(doc, "none"))
    stage12._mutate_provenance(
        case, raw_value_json=encode_typed_raw(None).to_json())
    result = replay_cell(stage12.ROOT, case.manifest, case.request)
    assert (result.status, result.reason_code) == (
        ReplayStatus.FAIL, "RAW_VALUE_MISMATCH")


def _fallback_raw_from(doc, policy="value"):
    doc["reads"] = [{
        "id": "primary_read", "source": "secondary",
        "column": "candidate_value", "coerce": "none",
    }]
    mapping = next(item for item in doc["mappings"]
                   if item["mapping_id"] == "fallback.numeric")
    mapping["provenance"]["raw_from"] = "primary_read"
    if policy != "value":
        mapping["provenance"]["raw_value"] = policy


def test_fallback_exhaustion_masks_raw_from_without_evidence(tmp_path):
    case = stage4._bundle(
        tmp_path, device="DEVICE-D", spec_mutate=_fallback_raw_from)
    event = next(cell for cell in case.cells
                 if cell.mapping_id == "fallback.numeric")
    assert event.raw_value_json == ""
    assert not any(item.get("role") == "primary_read"
                   for item in json.loads(event.contributions))
    result = replay_cell(stage4.ROOT, case.manifest, case.request)
    assert (result.status, result.reason_code, result.recomputed_value) == (
        ReplayStatus.PASS, "REPLAY_MATCH", None), result.message


def test_nonexhausted_default_fallback_requires_and_uses_raw_from(tmp_path):
    case = stage4._bundle(
        tmp_path, device="DEVICE-A", spec_mutate=_fallback_raw_from)
    event = next(cell for cell in case.cells
                 if cell.mapping_id == "fallback.numeric")
    read_evidence = [item for item in json.loads(event.contributions)
                     if item.get("role") == "primary_read"]
    assert len(read_evidence) == 1
    assert json.loads(event.raw_value_json) == {
        "kind": "number", "value": "10.0"}
    result = replay_cell(stage4.ROOT, case.manifest, case.request)
    assert (result.status, result.reason_code, result.recomputed_value) == (
        ReplayStatus.PASS, "REPLAY_MATCH", 5.0), result.message


@pytest.mark.parametrize("policy", ["coerced", "none"])
def test_nonexhausted_fallback_masking_does_not_require_raw_from(
        tmp_path, policy):
    case = stage4._bundle(
        tmp_path, device="DEVICE-A",
        spec_mutate=lambda doc: _fallback_raw_from(doc, policy))
    event = next(cell for cell in case.cells
                 if cell.mapping_id == "fallback.numeric")
    assert not any(item.get("role") == "primary_read"
                   for item in json.loads(event.contributions))
    result = replay_cell(stage4.ROOT, case.manifest, case.request)
    assert (result.status, result.reason_code) == (
        ReplayStatus.PASS, "REPLAY_MATCH"), result.message


def test_raw_mismatch_precedes_derivation_mismatch(tmp_path):
    case = stage12._bundle(
        tmp_path, participant="DEVICE-A", target="status_label",
        spec_mutate=_group_map_raw_derivation)
    stage12._mutate_provenance(
        case, raw_value_json='{"kind":"string","value":"forged"}',
        derivation='{"forged":true}')
    result = replay_cell(stage12.ROOT, case.manifest, case.request)
    assert (result.status, result.reason_code) == (
        ReplayStatus.FAIL, "RAW_VALUE_MISMATCH")


def test_correct_raw_wrong_derivation_is_derivation_mismatch(tmp_path):
    case = stage12._bundle(
        tmp_path, participant="DEVICE-A", target="status_label",
        spec_mutate=_group_map_raw_derivation)
    stage12._mutate_provenance(case, derivation='{"forged":true}')
    result = replay_cell(stage12.ROOT, case.manifest, case.request)
    assert (result.status, result.reason_code) == (
        ReplayStatus.FAIL, "DERIVATION_MISMATCH")


def test_custom_raw_remains_ineligible_but_static_provisional_state_composes():
    spec = load_spec(stage12.SPEC, vocabulary_dir=stage12.VOCABULARY.parent)
    custom = next(item for item in spec.mappings
                  if item.mapping_id == "group.recursive")
    custom = replace(custom, derivation=(DerivationVariant(
        record={"raw": {"$ref": "raw_value"}}),))
    assert not replay_module._is_stage17_derivation_mapping(custom, spec)

    direct = next(item for item in spec.mappings
                  if item.mapping_id == "group.direct.none")
    provisional = replace(
        direct, value_state="provisional",
        derivation=(DerivationVariant(
            record={"raw": {"$ref": "raw_value"}}),))
    assert replay_module._is_stage17_derivation_mapping(provisional, spec)

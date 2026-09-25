"""Synthetic tests for static present-state replay policy (Stage 18)."""
from __future__ import annotations

import inspect
from dataclasses import replace

import pytest

import lib.replay as replay_module
import lib.spec_executor as executor_module
from lib import rule_registry
from lib.derivation import canonical_derivation_json
from lib.replay import replay_cell
from lib.replay_contract import ReplayStatus, encode_typed_raw
from lib.replay_orchestration import (
    ReplayEvidenceCollector,
    ReplayOrchestrationError,
    ReplayRunContext,
)
from lib.rule_registry import RegisteredRule
from lib.source_ref import sha256_file
from lib.value_states import (
    DELIVERED,
    PROVISIONAL,
    TRANSFORMED,
    FinalizedValueState,
    finalize_declared_value_state,
)
import test_replay_stage1 as stage1
import test_replay_stage3 as stage3
import test_replay_stage4 as stage4
import test_replay_stage12 as stage12
import test_replay_stage15 as stage15
import test_replay_stage17 as stage17
import test_replay_orchestration as orchestration


@pytest.fixture(autouse=True)
def _stage18_rules(monkeypatch):
    monkeypatch.setitem(
        rule_registry.REGISTRY, ("synthetic.adjusted.v1", "1"),
        RegisteredRule(
            "synthetic.adjusted.v1", "1", rule_registry.KIND_FIELD,
            stage3._adjusted, arity=2, params=("scale",)))
    monkeypatch.setitem(
        rule_registry.REGISTRY, ("synthetic.identity.v1", "1"),
        RegisteredRule(
            "synthetic.identity.v1", "1", rule_registry.KIND_FIELD,
            stage3._identity, arity=1))


def _derivation_record():
    return {"kind": "provisional_value", "value": {"$ref": "value"}}


def _set_static_state(doc, mapping_id, *, state=PROVISIONAL, mode=""):
    mapping = next(item for item in doc["mappings"]
                   if item["mapping_id"] == mapping_id)
    mapping["value_state"] = state
    mapping["derivation"] = [{"record": _derivation_record()}]
    if mode:
        mapping.setdefault("provenance", {})["value_state"] = mode


def _direct_case(tmp_path, *, state=PROVISIONAL, mode="",
                 recorded_state=..., raw_json=...):
    def mutate(doc):
        _set_static_state(
            doc, "direct.sequence_number", state=state, mode=mode)

    updates = {
        "value_state": ("" if mode == "none" else state),
        "derivation": canonical_derivation_json({
            "kind": "provisional_value", "value": 1.0}),
    }
    if recorded_state is not ...:
        updates["value_state"] = recorded_state
    if raw_json is not ...:
        updates["raw_value_json"] = raw_json
    return stage1._case(
        tmp_path, "direct.sequence_number", spec_mutator=mutate,
        provenance_updates=updates)


def _conditional(doc, *, mode=""):
    stage15._conditional(doc)
    _set_static_state(doc, "fallback.raw", mode=mode)


def _mark_stage3_inputs(doc, mapping_ids):
    for mapping_id in mapping_ids:
        _set_static_state(doc, mapping_id)


@pytest.mark.parametrize("declared,semantic,serialized", [
    ("", DELIVERED, ""),
    (DELIVERED, DELIVERED, DELIVERED),
    (TRANSFORMED, TRANSFORMED, TRANSFORMED),
    (PROVISIONAL, PROVISIONAL, PROVISIONAL),
])
def test_pure_static_present_state_policy(declared, semantic, serialized):
    result = finalize_declared_value_state(declared, "declared")
    assert result == FinalizedValueState(semantic, serialized)
    assert finalize_declared_value_state(declared, "") == result
    assert finalize_declared_value_state(declared, "declared") == result


@pytest.mark.parametrize("declared", ["", DELIVERED, TRANSFORMED, PROVISIONAL])
def test_none_serialization_preserves_semantic_state(declared):
    result = finalize_declared_value_state(declared, "none")
    assert result.semantic_state == (declared or DELIVERED)
    assert result.serialized_state == ""


@pytest.mark.parametrize("declared,mode", [
    ("source_missing", "declared"),
    ("suppressed", "declared"),
    (PROVISIONAL, "DEFAULT"),
    ("Provisional", "declared"),
])
def test_static_policy_refuses_other_authorities_and_normalization(
        declared, mode):
    with pytest.raises(ValueError):
        finalize_declared_value_state(declared, mode)


def test_policy_signature_has_no_recorded_or_runtime_authority_inputs():
    assert tuple(inspect.signature(
        finalize_declared_value_state).parameters) == (
            "declared_state", "provenance_state_mode")


def test_direct_numeric_provisional_value_derivation_passes(tmp_path):
    case = _direct_case(tmp_path)
    result = replay_cell(stage1.ROOT, case.manifest, case.request)
    assert (result.status, result.reason_code, result.recomputed_value) == (
        ReplayStatus.PASS, "REPLAY_MATCH", 1.0)


@pytest.mark.parametrize("state", [DELIVERED, TRANSFORMED, PROVISIONAL])
def test_explicit_static_present_state_round_trips(tmp_path, state):
    case = _direct_case(tmp_path, state=state)
    assert replay_cell(
        stage1.ROOT, case.manifest, case.request).status is ReplayStatus.PASS


def test_none_mode_serializes_blank_without_changing_semantic_state(tmp_path):
    case = _direct_case(tmp_path, mode="none")
    result = replay_cell(stage1.ROOT, case.manifest, case.request)
    assert (result.status, result.reason_code) == (
        ReplayStatus.PASS, "REPLAY_MATCH")


def test_explicit_delivered_is_distinct_from_blank_default(tmp_path):
    case = _direct_case(tmp_path, state=DELIVERED, recorded_state="")
    result = replay_cell(stage1.ROOT, case.manifest, case.request)
    assert (result.status, result.reason_code) == (
        ReplayStatus.FAIL, "VALUE_STATE_CONTEXT_MISMATCH")


def test_none_mode_rejects_unexpected_present_state_spelling(tmp_path):
    case = _direct_case(
        tmp_path, mode="none", recorded_state=PROVISIONAL)
    result = replay_cell(stage1.ROOT, case.manifest, case.request)
    assert (result.status, result.reason_code) == (
        ReplayStatus.FAIL, "VALUE_STATE_CONTEXT_MISMATCH")


def test_recorded_state_is_comparison_only(tmp_path):
    case = _direct_case(tmp_path, recorded_state="")
    result = replay_cell(stage1.ROOT, case.manifest, case.request)
    assert (result.status, result.reason_code) == (
        ReplayStatus.FAIL, "VALUE_STATE_CONTEXT_MISMATCH")


def test_raw_mismatch_precedes_state_mismatch(tmp_path):
    case = _direct_case(
        tmp_path, recorded_state="",
        raw_json=encode_typed_raw(999).to_json())
    result = replay_cell(stage1.ROOT, case.manifest, case.request)
    assert (result.status, result.reason_code) == (
        ReplayStatus.FAIL, "RAW_VALUE_MISMATCH")


def test_source_mismatch_precedes_state_mismatch(tmp_path):
    case = _direct_case(tmp_path, recorded_state="")
    case.source_path.write_bytes(case.source_path.read_bytes() + b"\n")
    result = replay_cell(stage1.ROOT, case.manifest, case.request)
    assert (result.status, result.reason_code) == (
        ReplayStatus.FAIL, "SOURCE_HASH_MISMATCH")


def test_state_mismatch_precedes_derivation_mismatch(tmp_path):
    case = _direct_case(tmp_path, recorded_state="")
    fields, rows = stage4._read_rows(case.provenance_path)
    rows[0]["derivation"] = '{"kind":"forged"}'
    stage4._write_rows(case.provenance_path, fields, rows)
    case.manifest = replace(case.manifest, provenance_artifact=replace(
        case.manifest.provenance_artifact,
        sha256=sha256_file(case.provenance_path)))
    result = replay_cell(stage1.ROOT, case.manifest, case.request)
    assert (result.status, result.reason_code) == (
        ReplayStatus.FAIL, "VALUE_STATE_CONTEXT_MISMATCH")


def test_correct_state_wrong_derivation_is_derivation_mismatch(tmp_path):
    case = _direct_case(tmp_path)
    fields, rows = stage4._read_rows(case.provenance_path)
    rows[0]["derivation"] = '{"kind":"forged"}'
    stage4._write_rows(case.provenance_path, fields, rows)
    case.manifest = replace(case.manifest, provenance_artifact=replace(
        case.manifest.provenance_artifact,
        sha256=sha256_file(case.provenance_path)))
    result = replay_cell(stage1.ROOT, case.manifest, case.request)
    assert (result.status, result.reason_code) == (
        ReplayStatus.FAIL, "DERIVATION_MISMATCH")


def test_correct_state_missing_derivation_is_historical_incompleteness(tmp_path):
    case = _direct_case(tmp_path)
    fields, rows = stage4._read_rows(case.provenance_path)
    rows[0]["derivation"] = ""
    stage4._write_rows(case.provenance_path, fields, rows)
    case.manifest = replace(case.manifest, provenance_artifact=replace(
        case.manifest.provenance_artifact,
        sha256=sha256_file(case.provenance_path)))
    result = replay_cell(stage1.ROOT, case.manifest, case.request)
    assert (result.status, result.reason_code) == (
        ReplayStatus.NOT_REPLAYABLE, "DERIVATION_EVIDENCE_MISSING")


def test_missing_value_state_column_is_schema_unsupported(tmp_path):
    case = _direct_case(tmp_path)
    fields, rows = stage4._read_rows(case.provenance_path)
    fields.remove("value_state")
    for row in rows:
        row.pop("value_state")
    stage4._write_rows(case.provenance_path, fields, rows)
    case.manifest = replace(case.manifest, provenance_artifact=replace(
        case.manifest.provenance_artifact,
        sha256=sha256_file(case.provenance_path)))
    result = replay_cell(stage1.ROOT, case.manifest, case.request)
    assert (result.status, result.reason_code) == (
        ReplayStatus.NOT_REPLAYABLE, "PROVENANCE_SCHEMA_UNSUPPORTED")


def test_conditional_provisional_reuses_verified_value_and_raw(tmp_path):
    case = stage4._bundle(
        tmp_path, target="preferred_raw", spec_mutate=_conditional)
    result = replay_cell(stage4.ROOT, case.manifest, case.request)
    assert (result.status, result.reason_code, result.recomputed_value) == (
        ReplayStatus.PASS, "REPLAY_MATCH", "present")


def test_conditional_lower_named_read_failure_precedes_state(tmp_path):
    def mutate(doc):
        stage15._conditional(doc, named=True)
        _set_static_state(doc, "fallback.raw")

    case = stage4._bundle(
        tmp_path, target="preferred_raw", spec_mutate=mutate)

    def forge(rows):
        event = stage4._event(rows, "fallback.raw")
        event["value_state"] = ""
        stage4._set_contribution(
            event, 0, raw_value_json=encode_typed_raw("forged").to_json())

    stage4._mutate_provenance(case, forge)
    result = replay_cell(stage4.ROOT, case.manifest, case.request)
    assert (result.status, result.reason_code) == (
        ReplayStatus.FAIL, "NAMED_READ_MISMATCH")


def test_conditional_raw_failure_precedes_state(tmp_path):
    case = stage4._bundle(
        tmp_path, target="preferred_raw", spec_mutate=_conditional)
    stage4._mutate_provenance(
        case, lambda rows: stage4._event(rows, "fallback.raw").update(
            value_state="",
            raw_value_json=encode_typed_raw("forged").to_json()))
    result = replay_cell(stage4.ROOT, case.manifest, case.request)
    assert (result.status, result.reason_code) == (
        ReplayStatus.FAIL, "RAW_VALUE_MISMATCH")


def test_conditional_none_mode_and_blank_recorded_state_pass(tmp_path):
    case = stage4._bundle(
        tmp_path, target="preferred_raw",
        spec_mutate=lambda doc: _conditional(doc, mode="none"))
    assert replay_cell(
        stage4.ROOT, case.manifest, case.request).status is ReplayStatus.PASS


def test_group_route_reuses_stage12_and_stage17_authority(tmp_path):
    def mutate(doc):
        stage17._group_map_raw_derivation(doc)
        mapping = next(item for item in doc["mappings"]
                       if item["mapping_id"] == "group.map.inline")
        mapping["value_state"] = PROVISIONAL

    case = stage12._bundle(
        tmp_path, participant="DEVICE-A", target="status_label",
        spec_mutate=mutate)
    result = replay_cell(stage12.ROOT, case.manifest, case.request)
    assert (result.status, result.reason_code, result.recomputed_value) == (
        ReplayStatus.PASS, "REPLAY_MATCH", "ready")


def test_fallback_route_accepts_static_provisional_metadata(tmp_path):
    def mutate(doc):
        mapping = next(item for item in doc["mappings"]
                       if item["mapping_id"] == "fallback.numeric")
        mapping["value_state"] = PROVISIONAL

    case = stage4._bundle(tmp_path, spec_mutate=mutate)
    result = replay_cell(stage4.ROOT, case.manifest, case.request)
    assert (result.status, result.reason_code, result.recomputed_value) == (
        ReplayStatus.PASS, "REPLAY_MATCH", 5.0)


def test_publication_rejects_cell_state_contradicting_pinned_policy():
    context, schema, spec, cells, diagnostics = orchestration._execution()
    mapping = next(item for item in spec.mappings
                   if item.mapping_id == "direct.reading_text")
    provisional = replace(mapping, value_state=PROVISIONAL)
    pinned = replace(spec, mappings=tuple(
        provisional if item == mapping else item for item in spec.mappings))
    context = ReplayRunContext.create(
        orchestration.ROOT, schema, (pinned,), run_id=context.run_id,
        application="synthetic-stage18-state-check")
    with pytest.raises(ReplayOrchestrationError, match="value_state"):
        ReplayEvidenceCollector(context).collect(
            pinned, diagnostics.sources, cells)


def test_executor_uses_shared_static_state_policy(tmp_path, monkeypatch):
    calls = []
    original = executor_module.finalize_declared_value_state

    def spy(declared_state, provenance_state_mode):
        calls.append((declared_state, provenance_state_mode))
        return original(declared_state, provenance_state_mode)

    monkeypatch.setattr(executor_module, "finalize_declared_value_state", spy)
    stage4._bundle(
        tmp_path, target="preferred_raw", spec_mutate=_conditional)
    assert (PROVISIONAL, "") in calls


def test_recursive_rule_uses_value_after_provisional_upstream_passes(tmp_path):
    case = stage3._bundle(
        tmp_path, spec_mutator=lambda doc: _mark_stage3_inputs(
            doc, ("direct.base",)))
    result = replay_cell(stage3.ROOT, case.manifest, case.request)
    assert (result.status, result.reason_code, result.recomputed_value) == (
        ReplayStatus.PASS, "REPLAY_MATCH", 23.0)


@pytest.mark.parametrize("source_value,expected", [("0", 0.0), ("", None)])
def test_direct_provisional_zero_and_null_are_values_not_state_authority(
        tmp_path, source_value, expected):
    case = stage3._bundle(
        tmp_path, target="base_measurement",
        source_updates={"base_measurement": source_value},
        spec_mutator=lambda doc: _mark_stage3_inputs(doc, ("direct.base",)))
    result = replay_cell(stage3.ROOT, case.manifest, case.request)
    assert (result.status, result.reason_code, result.recomputed_value) == (
        ReplayStatus.PASS, "REPLAY_MATCH", expected)


def test_upstream_state_mismatch_rejects_recursive_dependency(tmp_path):
    case = stage3._bundle(
        tmp_path, spec_mutator=lambda doc: _mark_stage3_inputs(
            doc, ("direct.base",)))
    stage3._mutate_provenance(
        case, lambda rows: stage3._event(
            rows, "direct.base").update(value_state=""))
    result = replay_cell(stage3.ROOT, case.manifest, case.request)
    assert (result.status, result.reason_code) == (
        ReplayStatus.FAIL, "DEPENDENCY_REPLAY_FAILED")
    assert "VALUE_STATE_CONTEXT_MISMATCH" in result.message


def test_two_provisional_dependencies_are_certified_without_state_inputs(
        tmp_path):
    case = stage3._bundle(
        tmp_path, spec_mutator=lambda doc: _mark_stage3_inputs(
            doc, ("direct.base", "direct.offset")))
    result = replay_cell(stage3.ROOT, case.manifest, case.request)
    assert (result.status, result.reason_code, result.recomputed_value) == (
        ReplayStatus.PASS, "REPLAY_MATCH", 23.0)


def test_stage18_does_not_broaden_raw_derivation_or_absence_authority():
    assert replay_module._DECLARED_ABSENCE_STATES == frozenset({
        "not_applicable", "source_missing"})
    with pytest.raises(ValueError):
        finalize_declared_value_state("row_excluded", "declared")

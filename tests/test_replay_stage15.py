"""Synthetic strict replay tests for Stage 15 branch authority."""
from __future__ import annotations

import json
from dataclasses import replace

import pytest

import lib.spec_executor as executor_module
from lib.spec_executor import ExecutionError
from lib.conditions import (
    PredicateDefinitionError,
    coerce_named_read,
    evaluate_predicate,
    validate_predicate_definition,
)
from lib.replay import replay_cell
import test_replay_stage12 as stage12_fixture
from lib.replay_contract import (
    REASON_CODES,
    REASON_CODES_VERSION,
    ReplayStatus,
    encode_typed_raw,
)
from test_replay_stage4 import (
    ROOT,
    _bundle,
    _event,
    _mutate_delivered,
    _mutate_provenance,
    _set_contribution,
)


def _gated(doc, *, wanted=10.0, position="first", after="rejected"):
    mapping = next(item for item in doc["mappings"]
                   if item["mapping_id"] == "fallback.numeric")
    mapping["candidates"][0]["when"] = {
        "source": "secondary", "column": "candidate_value",
        "predicate": "equals", "value": wanted,
    }
    mapping["contributions"] = {
        "record": "all", "condition_position": position,
        "after_decision": after,
    }


def _conditional(doc, *, named=False, yield_kind="literal"):
    mapping = next(item for item in doc["mappings"]
                   if item["mapping_id"] == "fallback.raw")
    common = {
        "mapping_id": mapping["mapping_id"],
        "target": mapping["target"],
        "op": "conditional",
        "source": "primary",
        "column": "candidate_value",
        "rule": mapping["rule"],
        "confidence": 0.9,
        "provenance": {"record_key_from": "participant"},
    }
    if named:
        doc["reads"] = [{
            "id": "gate_text", "source": "secondary",
            "column": "candidate_value", "coerce": "text",
        }]
        clause = {
            "when": {"read": "gate_text", "predicate": "equals",
                     "value": "10.0"},
        }
        if yield_kind == "read":
            clause["read"] = "gate_text"
        elif yield_kind == "column":
            clause.update({"source": "primary", "column": "candidate_value"})
        else:
            clause["value"] = "named-branch"
        common.update({
            "clauses": [clause], "otherwise": "other",
            "contributions": {"conditions": ["gate_text"]},
        })
    else:
        common["clauses"] = [
            {"when": {"source": "primary", "column": "candidate_value",
                      "predicate": "not_null"},
             "value": "present"},
            {"when": {"source": "primary", "column": "candidate_value",
                      "predicate": "is_null"},
             "value": "missing"},
        ]
    mapping.clear()
    mapping.update(common)


def _siblings(doc, *, multiple=False):
    doc["allow_duplicate_targets"] = ["preferred_raw"]
    doc["reads"] = [{
        "id": "gate_text", "source": "secondary",
        "column": "candidate_value", "coerce": "text",
    }]
    doc["conditions"] = [
        {"id": "branch_one", "read": "gate_text",
         "predicate": "not_null" if multiple else "equals",
         **({} if multiple else {"value": "10.0"})},
        {"id": "branch_two", "read": "gate_text",
         "predicate": "not_null" if multiple else "not_equals",
         **({} if multiple else {"value": "10.0"})},
    ]
    old = next(item for item in doc["mappings"]
               if item["mapping_id"] == "fallback.raw")
    common = {
        "target": "preferred_raw", "op": "direct", "source": "primary",
        "column": "candidate_value", "rule": old["rule"], "confidence": 0.9,
        "contributions": {"conditions": ["gate_text"]},
        "provenance": {"record_key_from": "participant"},
    }
    siblings = [
        dict(common, mapping_id="branch.one", when="branch_one"),
        dict(common, mapping_id="branch.two", when="branch_two"),
    ]
    doc["mappings"] = siblings + [
        item for item in doc["mappings"]
        if item["mapping_id"] != "fallback.raw"
    ]


@pytest.mark.parametrize("operator,operand,wanted,expected", [
    ("is_null", None, None, True),
    ("not_null", 0, None, True),
    ("equals", "  alpha\n beta ", "alpha beta", True),
    ("not_equals", "alpha", "beta", True),
    ("in", " x ", ["x", "y"], True),
    ("not_in", "z", ["x", "y"], True),
    ("gte", "2.5", 2.5, True),
    ("gt", "3", 2, True),
    ("lte", "2", 2, True),
    ("lt", "bad", 2, False),
])
def test_pure_scalar_predicate_language(operator, operand, wanted, expected):
    assert evaluate_predicate(operator, operand, wanted) is expected
    assert evaluate_predicate(operator, operand, wanted) is expected


@pytest.mark.parametrize("operand", [None, float("nan"), "", "  "])
def test_non_null_operators_are_false_for_missing(operand):
    assert evaluate_predicate("not_equals", operand, "x") is False
    assert evaluate_predicate("not_in", operand, ["x"]) is False
    assert evaluate_predicate("gte", operand, 0) is False


def test_zero_and_false_are_present_and_types_are_not_invented():
    assert evaluate_predicate("not_null", 0) is True
    assert evaluate_predicate("not_null", False) is True
    assert evaluate_predicate("equals", False, "False") is True


@pytest.mark.parametrize("operator,value", [
    ("AND", None), ("OR", None), ("NOT", None),
    ("is_null", "extra"), ("in", []), ("gte", True),
])
def test_predicate_validation_fails_closed(operator, value):
    with pytest.raises(PredicateDefinitionError):
        validate_predicate_definition(operator, value)


@pytest.mark.parametrize("raw,coercion,expected", [
    ("  alpha\n beta ", "text", "alpha beta"),
    (".", "numeric", None),
    ("bad", "numeric", None),
    ("0", "numeric", 0.0),
    (False, "none", False),
])
def test_named_read_coercion_is_pure(raw, coercion, expected):
    assert coerce_named_read(raw, coercion) == expected


@pytest.mark.parametrize("device,expected", [
    ("DEVICE-A", 5.0),
    ("DEVICE-B", 20.0),
])
def test_gated_fallback_reconstructs_gate_and_candidate_chain(
        tmp_path, device, expected):
    case = _bundle(tmp_path, device=device, spec_mutate=_gated)
    result = replay_cell(ROOT, case.manifest, case.request)
    assert (result.status, result.reason_code, result.recomputed_value) == (
        ReplayStatus.PASS, "REPLAY_MATCH", expected)


def test_missing_candidate_precedes_false_gate(tmp_path):
    case = _bundle(tmp_path, device="DEVICE-B", spec_mutate=_gated)
    event = next(cell for cell in case.cells
                 if cell.mapping_id == "fallback.numeric")
    contributions = json.loads(event.contributions)
    assert [item["status"] for item in contributions] == [
        "condition", "unavailable", "selected", "rejected"]
    assert replay_cell(ROOT, case.manifest, case.request).status is ReplayStatus.PASS


def test_gated_fallback_condition_contribution_is_verification_only(tmp_path):
    case = _bundle(tmp_path, spec_mutate=_gated)
    _mutate_provenance(case, lambda rows: _set_contribution(
        _event(rows), 0,
        raw_value_json=encode_typed_raw("fabricated").to_json()))
    result = replay_cell(ROOT, case.manifest, case.request)
    assert (result.status, result.reason_code) == (
        ReplayStatus.FAIL, "FALLBACK_CANDIDATE_CONTEXT_MISMATCH")


@pytest.mark.parametrize("named,yield_kind,expected", [
    (False, "literal", "present"),
    (True, "literal", "named-branch"),
    (True, "column", "5"),
    (True, "read", "10.0"),
])
def test_conditional_routes_only_from_verified_operands(
        tmp_path, named, yield_kind, expected):
    case = _bundle(
        tmp_path, target="preferred_raw",
        spec_mutate=lambda doc: _conditional(
            doc, named=named, yield_kind=yield_kind))
    result = replay_cell(ROOT, case.manifest, case.request)
    assert (result.status, result.reason_code, result.recomputed_value) == (
        ReplayStatus.PASS, "REPLAY_MATCH", expected)


def test_named_read_snapshot_cannot_drive_predicate_or_yield(tmp_path):
    case = _bundle(
        tmp_path, target="preferred_raw",
        spec_mutate=lambda doc: _conditional(doc, named=True, yield_kind="read"))
    _mutate_provenance(case, lambda rows: _set_contribution(
        _event(rows, "fallback.raw"), 0,
        raw_value_json=encode_typed_raw("other").to_json()))
    result = replay_cell(ROOT, case.manifest, case.request)
    assert (result.status, result.reason_code) == (
        ReplayStatus.FAIL, "NAMED_READ_MISMATCH")


def test_missing_usage_scoped_named_read_is_not_replayable(tmp_path):
    case = _bundle(
        tmp_path, target="preferred_raw",
        spec_mutate=lambda doc: _conditional(doc, named=True))
    usage = next(item for item in case.manifest.mapping_usages
                 if item.mapping_id == "fallback.raw")
    primary = next(source.source_ref_id for source in case.manifest.sources
                   if source.source_id == "primary")
    reduced = replace(usage, source_ref_ids=(primary,))
    case.manifest = replace(
        case.manifest,
        mapping_usages=tuple(reduced if item == usage else item
                             for item in case.manifest.mapping_usages))
    result = replay_cell(ROOT, case.manifest, case.request)
    assert (result.status, result.reason_code) == (
        ReplayStatus.NOT_REPLAYABLE, "NAMED_READ_EVIDENCE_MISSING")


def test_mapping_sibling_authority_passes_for_one_true_branch(tmp_path):
    case = _bundle(
        tmp_path, target="preferred_raw", spec_mutate=_siblings)
    result = replay_cell(ROOT, case.manifest, case.request)
    assert (result.status, result.reason_code, result.mapping_id) == (
        ReplayStatus.PASS, "REPLAY_MATCH", "branch.one")


def test_false_recorded_mapping_is_branch_mismatch(tmp_path):
    case = _bundle(
        tmp_path, target="preferred_raw", spec_mutate=_siblings)
    def mutate(rows):
        event = _event(rows, "branch.one")
        event["mapping_id"] = "branch.two"
    _mutate_provenance(case, mutate)
    old_usage = next(item for item in case.manifest.mapping_usages
                     if item.mapping_id == "branch.one")
    wrong_usage = replace(old_usage, mapping_id="branch.two")
    case.manifest = replace(
        case.manifest, mapping_usages=tuple(
            wrong_usage if item == old_usage else item
            for item in case.manifest.mapping_usages))
    result = replay_cell(ROOT, case.manifest, case.request)
    assert (result.status, result.reason_code) == (
        ReplayStatus.FAIL, "BRANCH_SELECTION_MISMATCH")


def test_multiple_true_siblings_are_not_a_contradiction(tmp_path):
    case = _bundle(
        tmp_path, target="preferred_raw",
        spec_mutate=lambda doc: _siblings(doc, multiple=True))
    result = replay_cell(ROOT, case.manifest, case.request)
    assert (result.status, result.reason_code) == (
        ReplayStatus.NOT_REPLAYABLE, "MULTI_EVENT_BRANCH_UNSUPPORTED")


def test_executor_uses_shared_predicate_helper(monkeypatch, tmp_path):
    original = executor_module.evaluate_predicate
    calls = []
    def spy(predicate, operand, wanted=None):
        calls.append((predicate, operand, wanted))
        return original(predicate, operand, wanted)
    monkeypatch.setattr(executor_module, "evaluate_predicate", spy)
    _bundle(
        tmp_path, target="preferred_raw",
        spec_mutate=lambda doc: _conditional(doc, named=True))
    assert calls


def test_absent_named_read_column_fails_but_present_null_is_valid(tmp_path):
    def missing(doc):
        _conditional(doc, named=True)
        doc["reads"][0]["column"] = "absent_column"
    with pytest.raises(ExecutionError, match="named read.*missing column"):
        _bundle(
            tmp_path / "missing", target="preferred_raw", spec_mutate=missing)

    case = _bundle(
        tmp_path / "null", device="DEVICE-D", target="preferred_raw",
        spec_mutate=lambda doc: _conditional(doc, named=True))
    result = replay_cell(ROOT, case.manifest, case.request)
    assert (result.status, result.reason_code, result.recomputed_value) == (
        ReplayStatus.PASS, "REPLAY_MATCH", "other")


def test_selected_group_member_named_read_reuses_stage12_authority(tmp_path):
    def mutate(doc):
        doc["reads"] = [{
            "id": "group_text", "source": "cohort.scalar",
            "column": "direct_text", "coerce": "text",
        }]
        mapping = next(item for item in doc["mappings"]
                       if item["mapping_id"] == "group.direct.none")
        replacement = {
            "mapping_id": mapping["mapping_id"],
            "target": mapping["target"],
            "op": "conditional",
            "clauses": [{
                "when": {"read": "group_text", "predicate": "not_null"},
                "read": "group_text",
            }],
            "otherwise": None,
            "confidence": {"value": 0.95, "no_value": 0.0},
            "rule": mapping["rule"],
            "contributions": {"conditions": ["group_text"]},
            "provenance": {
                "contributions_from": "cohort",
                "record_key_from": "participant",
            },
        }
        mapping.clear()
        mapping.update(replacement)
        doc["mappings"] = [
            item for item in doc["mappings"]
            if item["mapping_id"] != "group.recursive"
        ]

    case = stage12_fixture._bundle(
        tmp_path, participant="DEVICE-B", target="direct_none",
        spec_mutate=mutate)
    result = replay_cell(stage12_fixture.ROOT, case.manifest, case.request)
    assert (result.status, result.reason_code, result.recomputed_value) == (
        ReplayStatus.PASS, "REPLAY_MATCH", "spaced beta")


@pytest.mark.parametrize("updates", [
    {"role": "fabricated_read"},
    {"status": "selected"},
    {"rank": 99},
    {"rule_id": "fabricated.rule"},
    {"source_column": "fabricated_column"},
    {"source_row_number": 99},
    {"source_resolved_path": "fabricated.csv"},
    {"source_sha256": "0" * 64},
])
def test_named_read_contribution_context_is_verification_only(tmp_path, updates):
    case = _bundle(
        tmp_path, target="preferred_raw",
        spec_mutate=lambda doc: _conditional(doc, named=True))
    _mutate_provenance(case, lambda rows: _set_contribution(
        _event(rows, "fallback.raw"), 0, **updates))
    result = replay_cell(ROOT, case.manifest, case.request)
    assert (result.status, result.reason_code) == (
        ReplayStatus.FAIL, "NAMED_READ_MISMATCH")


def test_named_read_wrong_record_key_preserves_lower_reason(tmp_path):
    case = _bundle(
        tmp_path, target="preferred_raw",
        spec_mutate=lambda doc: _conditional(doc, named=True))
    _mutate_provenance(case, lambda rows: _set_contribution(
        _event(rows, "fallback.raw"), 0,
        source_record_key="device_id=FABRICATED"))
    result = replay_cell(ROOT, case.manifest, case.request)
    assert (result.status, result.reason_code) == (
        ReplayStatus.FAIL, "SOURCE_RECORD_KEY_MISMATCH")


def test_conditional_named_read_contribution_order_is_authoritative(tmp_path):
    def two_reads(doc):
        _conditional(doc, named=True)
        doc["reads"].append({
            "id": "gate_number", "source": "secondary",
            "column": "candidate_value", "coerce": "numeric",
        })
        mapping = next(item for item in doc["mappings"]
                       if item["mapping_id"] == "fallback.raw")
        mapping["contributions"]["conditions"] = [
            "gate_text", "gate_number"]

    case = _bundle(
        tmp_path, target="preferred_raw", spec_mutate=two_reads)
    def reorder(rows):
        event = _event(rows, "fallback.raw")
        items = json.loads(event["contributions"])
        items[0], items[1] = items[1], items[0]
        event["contributions"] = json.dumps(
            items, separators=(",", ":"), sort_keys=True)
    _mutate_provenance(case, reorder)
    result = replay_cell(ROOT, case.manifest, case.request)
    assert (result.status, result.reason_code) == (
        ReplayStatus.FAIL, "NAMED_READ_MISMATCH")


def test_missing_historical_condition_contribution_is_not_replayable(tmp_path):
    case = _bundle(
        tmp_path, target="preferred_raw",
        spec_mutate=lambda doc: _conditional(doc, named=True))
    _mutate_provenance(
        case, lambda rows: _event(rows, "fallback.raw").update(
            contributions=""))
    result = replay_cell(ROOT, case.manifest, case.request)
    assert (result.status, result.reason_code) == (
        ReplayStatus.NOT_REPLAYABLE, "NAMED_READ_EVIDENCE_MISSING")


def test_delivery_and_provenance_values_cannot_choose_conditional_branch(tmp_path):
    delivery = _bundle(
        tmp_path / "delivery", target="preferred_raw",
        spec_mutate=lambda doc: _conditional(doc, named=True))
    _mutate_delivered(
        delivery, lambda rows: rows[0].update(preferred_raw="fabricated"))
    _mutate_provenance(
        delivery, lambda rows: _event(
            rows, "fallback.raw").update(value="fabricated"))
    result = replay_cell(ROOT, delivery.manifest, delivery.request)
    assert (result.status, result.reason_code) == (
        ReplayStatus.FAIL, "RECOMPUTED_VALUE_MISMATCH")

    provenance = _bundle(
        tmp_path / "provenance", target="preferred_raw",
        spec_mutate=lambda doc: _conditional(doc, named=True))
    _mutate_provenance(
        provenance, lambda rows: _event(
            rows, "fallback.raw").update(value="fabricated"))
    result = replay_cell(ROOT, provenance.manifest, provenance.request)
    assert (result.status, result.reason_code) == (
        ReplayStatus.FAIL, "PROVENANCE_VALUE_MISMATCH")


def test_true_conditional_clause_yielding_null_wins(tmp_path):
    def null_winner(doc):
        _conditional(doc)
        mapping = next(item for item in doc["mappings"]
                       if item["mapping_id"] == "fallback.raw")
        mapping["clauses"] = [{
            "when": {"source": "primary", "column": "candidate_value",
                     "predicate": "is_null"},
            "value": None,
        }]
        mapping["otherwise"] = "must-not-win"
        mapping["confidence"] = {"value": 0.9, "no_value": 0.0}

    case = _bundle(
        tmp_path, device="DEVICE-D", target="preferred_raw",
        spec_mutate=null_winner)
    result = replay_cell(ROOT, case.manifest, case.request)
    assert (result.status, result.reason_code, result.recomputed_value) == (
        ReplayStatus.PASS, "REPLAY_MATCH", None)


def test_gated_fallback_recorded_winner_cannot_drive_selection(tmp_path):
    case = _bundle(tmp_path, spec_mutate=_gated)
    def forge_winner(rows):
        event = _event(rows)
        items = json.loads(event["contributions"])
        items[1]["status"] = "rejected"
        items[2]["status"] = "selected"
        event["contributions"] = json.dumps(
            items, separators=(",", ":"), sort_keys=True)
    _mutate_provenance(case, forge_winner)
    result = replay_cell(ROOT, case.manifest, case.request)
    assert (result.status, result.reason_code) == (
        ReplayStatus.FAIL, "FALLBACK_SELECTION_MISMATCH")


def test_stage15_reason_contract_only_bumps_reason_version():
    assert REASON_CODES_VERSION == 10
    assert {"NAMED_READ_MISMATCH", "BRANCH_SELECTION_MISMATCH"} <= \
        REASON_CODES[ReplayStatus.FAIL]
    assert {"NAMED_READ_EVIDENCE_MISSING",
            "MULTI_EVENT_BRANCH_UNSUPPORTED"} <= \
        REASON_CODES[ReplayStatus.NOT_REPLAYABLE]
